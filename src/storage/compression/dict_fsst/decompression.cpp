#include "duckdb/storage/compression/dict_fsst/decompression.hpp"

#include "duckdb/common/fsst.hpp"
#include "duckdb/common/vector/dictionary_vector.hpp"
#include "duckdb/common/vector/flat_vector.hpp"
#include "duckdb/common/vector/string_vector.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/filter/expression_filter.hpp"
#include "duckdb/storage/table/column_segment.hpp"
#include "fsst.h"

namespace duckdb {
namespace dict_fsst {

[[noreturn]] static void ThrowPackedValuesOutOfBounds(const char *context) {
	throw DataCorruptionException("Corrupted %s: packed values exceed the available space", context);
}

[[noreturn]] static void ThrowInvalidMode(DictFSSTMode mode) {
	throw DataCorruptionException("Corrupted DICT_FSST segment: unrecognized compression mode %d",
	                              static_cast<uint8_t>(mode));
}

[[noreturn]] static void ThrowInvalidDictionaryCount() {
	throw DataCorruptionException("Corrupted DICT_FSST segment: dictionary has no NULL entry");
}

[[noreturn]] static void ThrowZeroWidthDictionaryTooLarge() {
	throw DataCorruptionException(
	    "Corrupted DICT_FSST segment: zero-width lengths allow at most two dictionary entries");
}

[[noreturn]] static void ThrowDictionaryIndexOutOfBounds() {
	throw DataCorruptionException("Corrupted DICT_FSST segment: dictionary index is out of range");
}

[[noreturn]] static void ThrowInvalidNullLength() {
	throw DataCorruptionException("Corrupted DICT_FSST segment: NULL dictionary entry has a nonzero length");
}

[[noreturn]] static void ThrowStringLengthsOutOfBounds() {
	throw DataCorruptionException("Corrupted DICT_FSST segment: string lengths exceed the dictionary size");
}

[[noreturn]] static void ThrowInvalidBitpackingWidth() {
	throw DataCorruptionException("Failed to scan dictionary string - bitpacking width was invalid. Database file "
	                              "appears to be corrupted.");
}

[[noreturn]] static void ThrowInvalidSymbolTable() {
	throw DataCorruptionException("Failed to scan DICT_FSST string segment: invalid FSST symbol table. Database file "
	                              "appears to be corrupted.");
}

static unsafe_array_ptr<const uint8_t> ReadBitpackedValues(CompressionSegmentReader &reader, idx_t count,
                                                           bitpacking_width_t width, const char *context) {
	constexpr auto group_size = BitpackingPrimitives::BITPACKING_ALGORITHM_GROUP_SIZE;
	const auto group_count = count / group_size + (count % group_size != 0);
	const auto group_bytes = group_size * idx_t(width) / 8;
	// Check the number of groups before multiplying, so a stored count cannot overflow the byte size.
	if (group_bytes && group_count > reader.Remaining() / group_bytes) {
		ThrowPackedValuesOutOfBounds(context);
	}
	// The unpacker reads uint32_t words, even when the segment starts at a nonzero block offset.
	return reader.ReadBytesAligned<uint32_t>(group_count * group_bytes);
}

CompressedStringScanState::~CompressedStringScanState() {
	delete reinterpret_cast<duckdb_fsst_decoder_t *>(decoder);
}

string_t CompressedStringScanState::FetchStringFromDict(Vector &result, uint32_t dict_offset, idx_t dict_idx,
                                                        StringScanType string_scan_type) {
	D_ASSERT(dict_idx < layout.dict_count);
	if (dict_idx == 0) {
		return string_t(nullptr, 0);
	}
	uint32_t string_len = GetStringLength(dict_idx);

	// normal string: read string from this block
	auto string_data = layout.dictionary.GetBytes(dict_offset, string_len);

	auto str_ptr = const_char_ptr_cast(string_data.data());
	switch (layout.mode) {
	case DictFSSTMode::FSST_ONLY:
	case DictFSSTMode::DICT_FSST: {
		if (string_len == 0) {
			return string_t(nullptr, 0);
		}
		D_ASSERT(decoder);
		D_ASSERT(string_len <= string_t::MAX_STRING_SIZE / 8);
		if (all_values_inlined) {
			return FSSTPrimitives::DecompressInlinedValue(decoder, str_ptr, string_len);
		} else {
			return FSSTPrimitives::DecompressValue(decoder, StringVector::GetStringAllocator(result), str_ptr,
			                                       string_len);
		}
	}
	default:
		if (string_scan_type == StringScanType::COPY_STRINGS) {
			return StringVector::AddStringOrBlob(result, string_t(str_ptr, string_len));
		}
		// FIXME: the Vector doesn't seem to take ownership of the non-inlined string data???
		return string_t(str_ptr, string_len);
	}
}

CompressedStringScanState::SegmentLayout CompressedStringScanState::ReadLayout(const BufferHandle &handle,
                                                                               const ColumnSegment &segment) {
	auto reader = CompressionSegmentReader::FromSegment(handle, segment, "DICT_FSST segment");

	// Load header values
	const auto header = reader.Read<dict_fsst_compression_header_t>();
	if (header.mode >= DictFSSTMode::COUNT) {
		ThrowInvalidMode(header.mode);
	}
	if (header.dict_count == 0) {
		ThrowInvalidDictionaryCount();
	}
	// FSST_ONLY maps row i to dictionary entry i + 1, skipping the NULL entry.
	if (header.mode == DictFSSTMode::FSST_ONLY && segment.count.load() > header.dict_count - 1) {
		ThrowDictionaryIndexOutOfBounds();
	}
	if (header.dictionary_indices_width > sizeof(uint32_t) * 8 || header.string_lengths_width > sizeof(uint32_t) * 8) {
		ThrowInvalidBitpackingWidth();
	}
	if (header.string_lengths_width == 0) {
		// FSST segments retain nonempty training strings, so their lengths cannot all be zero.
		if (header.mode != DictFSSTMode::DICTIONARY) {
			ThrowInvalidBitpackingWidth();
		}
		// A zero-width dictionary has only NULL and possibly the empty string.
		if (header.dict_count > 2) {
			ThrowZeroWidthDictionaryTooLarge();
		}
	}

	reader.Align(sizeof(idx_t));
	auto dictionary = reader.ReadSubReader(header.dict_size, "DICT_FSST dictionary");
	reader.Align(sizeof(idx_t));
	auto symbol_table = reader.ReadSubReader(header.symbol_table_size, "DICT_FSST symbol table");
	reader.Align(sizeof(idx_t));
	auto lengths = ReadBitpackedValues(reader, header.dict_count, header.string_lengths_width, "DICT_FSST lengths");
	reader.Align(sizeof(idx_t));
	auto indices = ReadBitpackedValues(reader, segment.count.load(), header.dictionary_indices_width,
	                                   "DICT_FSST dictionary indices");
	return {header.mode,
	        header.dict_count,
	        header.dictionary_indices_width,
	        header.string_lengths_width,
	        dictionary,
	        symbol_table,
	        lengths,
	        indices};
}

void CompressedStringScanState::Initialize() {
	switch (layout.mode) {
	case DictFSSTMode::FSST_ONLY:
	case DictFSSTMode::DICT_FSST: {
		decoder = new duckdb_fsst_decoder_t;
		// in FSST_ONLY / DICT_FSST modes a symbol table is always present, so any failure (out-of-bounds or a
		// version mismatch) means the segment is corrupted
		auto symbol_table = layout.symbol_table.GetBytes(0, layout.symbol_table.Size());
		auto ret = duckdb_fsst_import(reinterpret_cast<duckdb_fsst_decoder_t *>(decoder), symbol_table.data(),
		                              symbol_table.size());
		if (ret == DUCKDB_FSST_IMPORT_VERSION_MISMATCH || ret == DUCKDB_FSST_IMPORT_OUT_OF_BOUNDS) {
			ThrowInvalidSymbolTable();
		}
		break;
	}
	default:
		break;
	}
}

void CompressedStringScanState::LoadStringLengths() {
	if (!string_lengths.empty()) {
		return;
	}
	constexpr auto group_size = BitpackingPrimitives::BITPACKING_ALGORITHM_GROUP_SIZE;
	string_lengths.resize(AlignValue<idx_t, group_size>(layout.dict_count));

	// Both the entry count and each length fit in uint32_t, so their sum fits in uint64_t.
	uint64_t total_bytes = 0;
	for (idx_t i = 0; i < layout.dict_count; i += group_size) {
		BitpackingPrimitives::UnPackBlock<uint32_t>(
		    data_ptr_cast(string_lengths.data() + i),
		    layout.string_lengths.data() + (i * layout.string_lengths_width) / 8, layout.string_lengths_width);

		// Validate only real entries, not the final group's padding.
		const auto end = MinValue<idx_t>(i + group_size, layout.dict_count);
		uint64_t group_bytes = 0;
		for (idx_t j = i; j < end; j++) {
			group_bytes += string_lengths[j];
		}

		// Each string offset sums earlier lengths, so bounding the total keeps all offsets in range.
		total_bytes += group_bytes;
		if (total_bytes > layout.dictionary.Size()) {
			ThrowStringLengthsOutOfBounds();
		}
	}

	// Index 0 represents NULL, so its length must be 0
	if (string_lengths[0] != 0) {
		ThrowInvalidNullLength();
	}
}

const uint32_t *CompressedStringScanState::GetStringLengthGroup(idx_t group_start) {
	D_ASSERT(group_start % BitpackingPrimitives::BITPACKING_ALGORITHM_GROUP_SIZE == 0 &&
	         group_start < layout.dict_count);
	if (string_length_group_start != group_start) {
		BitpackingPrimitives::UnPackBlock<uint32_t>(data_ptr_cast(string_length_group),
		                                            layout.string_lengths.data() +
		                                                (group_start * layout.string_lengths_width) / 8,
		                                            layout.string_lengths_width);
		string_length_group_start = group_start;
	}
	return string_length_group;
}

uint32_t CompressedStringScanState::GetStringLength(idx_t dict_idx) {
	if (!string_lengths.empty()) {
		return string_lengths[dict_idx];
	}
	const auto group_offset = dict_idx % BitpackingPrimitives::BITPACKING_ALGORITHM_GROUP_SIZE;
	return GetStringLengthGroup(dict_idx - group_offset)[group_offset];
}

idx_t CompressedStringScanState::SumStringLengths(idx_t start, idx_t end) {
	idx_t sum = 0;
	if (!string_lengths.empty()) {
		const auto lengths = string_lengths.data();
		for (auto i = start; i < end; i++) {
			sum += lengths[i];
		}
		return sum;
	}
	constexpr idx_t GROUP_SIZE = BitpackingPrimitives::BITPACKING_ALGORITHM_GROUP_SIZE;
	for (auto group_start = start - start % GROUP_SIZE; group_start < end; group_start += GROUP_SIZE) {
		const auto lengths = GetStringLengthGroup(group_start);
		const auto to = MinValue<idx_t>(end - group_start, GROUP_SIZE);
		for (auto i = MaxValue<idx_t>(start, group_start) - group_start; i < to; i++) {
			sum += lengths[i];
		}
	}
	return sum;
}

bool CompressedStringScanState::EnsureDictionary() {
	if (dictionary || layout.mode == DictFSSTMode::FSST_ONLY) {
		return dictionary != nullptr;
	}
	LoadStringLengths();
	dictionary = DictionaryVector::CreateReusableDictionary(segment.GetType(), layout.dict_count);
	auto &dict_data = dictionary->data;
	auto dict_child_data = FlatVector::GetDataMutable<string_t>(dict_data);
	auto &validity = FlatVector::ValidityMutable(dict_data);
	D_ASSERT(layout.dict_count >= 1);
	validity.SetInvalid(0);

	if (layout.mode == DictFSSTMode::DICT_FSST && !all_values_inlined) {
		DecompressDictionary(dict_data, dict_child_data);
		return true;
	}
	uint32_t offset = 0;
	for (uint32_t i = 0; i < layout.dict_count; i++) {
		//! We can uncompress during fetching, we need the length of the string inside the dictionary
		auto string_len = string_lengths[i];
		dict_child_data[i] = FetchStringFromDict(dict_data, offset, i);
		offset += string_len;
	}
	return true;
}

void CompressedStringScanState::DecompressDictionary(Vector &dict_data, string_t *dict_child_data) {
	idx_t compressed_total = 0;
	for (uint32_t i = 0; i < layout.dict_count; i++) {
		compressed_total += string_lengths[i];
	}
	auto &allocator = StringVector::GetStringAllocator(dict_data);
	const idx_t capacity = compressed_total * 8 + 64;
	const auto buffer = StringVector::AllocateShrinkableBuffer(allocator, capacity);
	const auto compressed = layout.dictionary.GetBytes(0, layout.dictionary.Size());
	auto fsst_decoder = static_cast<duckdb_fsst_decoder_t *>(decoder);
	idx_t used = 0;
	idx_t offset = string_lengths[0];
	dict_child_data[0] = string_t(nullptr, 0);
	for (uint32_t i = 1; i < layout.dict_count; i++) {
		const idx_t compressed_len = string_lengths[i];
		if (compressed_len == 0) {
			dict_child_data[i] = string_t(nullptr, 0);
			continue;
		}
		auto target = buffer + used;
		const auto decompressed_len = duckdb_fsst_decompress(
		    fsst_decoder, compressed_len, const_uchar_ptr_cast(compressed.data() + offset), capacity - used, target);
		if (decompressed_len == DUCKDB_FSST_DECOMPRESS_INVALID_INPUT || decompressed_len > capacity - used) {
			throw DataCorruptionException("Failed to scan DICT_FSST string segment: invalid compressed string. "
			                              "Database file appears to be corrupted.");
		}
		dict_child_data[i] = string_t(const_char_ptr_cast(target), UnsafeNumericCast<uint32_t>(decompressed_len));
		used += decompressed_len * (decompressed_len > string_t::INLINE_LENGTH);
		offset += compressed_len;
	}
	allocator.ShrinkHead(capacity - used);
}

void CompressedStringScanState::EnsureStringOffsets() {
	if (!string_offsets.empty()) {
		return;
	}
	LoadStringLengths();
	string_offsets.resize(layout.dict_count);
	uint32_t offset = 0;
	for (uint32_t i = 0; i < layout.dict_count; i++) {
		string_offsets[i] = offset;
		offset += string_lengths[i];
	}
}

uint32_t CompressedStringScanState::GetStringOffset(idx_t string_number, bool allow_restart) {
	D_ASSERT(string_number < layout.dict_count);
	if (!string_offsets.empty()) {
		return string_offsets[string_number];
	}
	if (string_number < decompress_position) {
		if (!allow_restart || summed_lengths >= layout.dict_count) {
			EnsureStringOffsets();
			return string_offsets[string_number];
		}
		decompress_position = 0;
		decompress_offset = 0;
	}
	// lengths read group by group are not validated up front, so bound each offset as LoadStringLengths does
	const auto offset = decompress_offset + SumStringLengths(decompress_position, string_number);
	if (offset > layout.dictionary.Size()) {
		ThrowStringLengthsOutOfBounds();
	}
	decompress_offset = UnsafeNumericCast<uint32_t>(offset);
	summed_lengths += string_number - decompress_position;
	decompress_position = string_number;
	return decompress_offset;
}

bool CompressedStringScanState::PreferSparseDecode(idx_t start, idx_t scan_count, idx_t sel_count) {
	if (!DictionaryPending()) {
		return false;
	}
	if (sel_count == 0) {
		return true;
	}
	const idx_t segment_rows = segment.count.load();
	const idx_t remaining_rows = segment_rows > start ? segment_rows - start : scan_count;
	const idx_t vectors_left = MaxValue<idx_t>(remaining_rows / STANDARD_VECTOR_SIZE, 1);
	return sel_count * vectors_left * 2 <= layout.dict_count;
}

void CompressedStringScanState::SparseDecodeByRow(Vector &result, idx_t start, idx_t scan_count,
                                                  const SelectionVector &sel, idx_t sel_count) {
	EnsureStringOffsets();
	auto &selvec = GetSelVec(start, scan_count);
	FlatVector::SetSize(result, scan_count);
	auto result_data = FlatVector::GetDataMutable<string_t>(result);
	auto &validity = FlatVector::ValidityMutable(result);
	memset(result_data, 0, scan_count * sizeof(string_t));
	if (sel_count == 0) {
		return;
	}
	const auto sel_data = sel.data();
	for (idx_t i = 0; i < sel_count; i++) {
		const idx_t row = sel_data ? sel_data[i] : i;
		const auto string_number = selvec.get_index(row);
		if (string_number == 0) {
			validity.SetInvalid(row);
			continue;
		}
		result_data[row] = FetchStringFromDict(result, string_offsets[string_number], string_number);
	}
}

static constexpr idx_t SUBSTRING_MAX_NEEDLE = 32;
static constexpr uint32_t SUBSTRING_ROW_STRIDE = 256;
static constexpr uint32_t SUBSTRING_ESCAPE_CODE = 255;

bool ParseSubstringFilter(const TableFilter &filter, string &needle_result, bool &negate_result) {
	if (filter.filter_type != TableFilterType::EXPRESSION_FILTER) {
		return false;
	}
	auto &expression_filter = filter.Cast<ExpressionFilter>();
	if (!expression_filter.column_indexes.empty()) {
		return false;
	}
	const Expression *expr = expression_filter.expr.get();
	bool negate = false;
	if (expr->GetExpressionClass() == ExpressionClass::BOUND_OPERATOR &&
	    expr->GetExpressionType() == ExpressionType::OPERATOR_NOT) {
		auto &not_expr = expr->Cast<BoundOperatorExpression>();
		if (not_expr.GetChildren().size() != 1) {
			return false;
		}
		negate = true;
		expr = not_expr.GetChildren()[0].get();
	}
	if (expr->GetExpressionClass() != ExpressionClass::BOUND_FUNCTION) {
		return false;
	}
	auto &function_expr = expr->Cast<BoundFunctionExpression>();
	if (function_expr.Function().GetName().GetIdentifierName() != "contains" ||
	    function_expr.GetChildren().size() != 2) {
		return false;
	}
	auto &column_expr = *function_expr.GetChildren()[0];
	auto &needle_expr = *function_expr.GetChildren()[1];
	if (!ExpressionFilter::IsSimpleFilterColumnRef(column_expr) ||
	    column_expr.GetReturnType().id() != LogicalTypeId::VARCHAR ||
	    needle_expr.GetExpressionClass() != ExpressionClass::BOUND_CONSTANT) {
		return false;
	}
	auto &needle_value = needle_expr.Cast<BoundConstantExpression>().GetValue();
	if (needle_value.IsNull() || needle_value.type().id() != LogicalTypeId::VARCHAR) {
		return false;
	}
	auto needle = StringValue::Get(needle_value);
	if (needle.empty() || needle.size() > SUBSTRING_MAX_NEEDLE) {
		return false;
	}
	needle_result = std::move(needle);
	negate_result = negate;
	return true;
}

bool CompressedStringScanState::PrepareSubstringFilter(const TableFilter &filter) {
	if (substring_filter_state >= 0) {
		return substring_filter_state == 1;
	}
	substring_filter_state = 0;
	if (layout.mode != DictFSSTMode::DICT_FSST) {
		return false;
	}
	if (ParseSubstringFilter(filter, substring_needle, substring_negate)) {
		substring_filter_state = 1;
	}
	return substring_filter_state == 1;
}

void CompressedStringScanState::BuildSubstringTable() {
	const idx_t needle_size = substring_needle.size();
	const idx_t state_count = needle_size + 1;
	vector<uint8_t> dfa(state_count * 256, 0);
	const auto needle = const_uchar_ptr_cast(substring_needle.data());
	dfa[needle[0]] = 1;
	uint32_t restart_state = 0;
	for (idx_t q = 1; q < needle_size; q++) {
		for (idx_t b = 0; b < 256; b++) {
			dfa[q * 256 + b] = dfa[restart_state * 256 + b];
		}
		dfa[q * 256 + needle[q]] = UnsafeNumericCast<uint8_t>(q + 1);
		restart_state = dfa[restart_state * 256 + needle[q]];
	}
	for (idx_t b = 0; b < 256; b++) {
		dfa[needle_size * 256 + b] = UnsafeNumericCast<uint8_t>(needle_size);
	}
	substring_table.assign(2 * state_count * SUBSTRING_ROW_STRIDE, 0);
	auto fsst_decoder = static_cast<duckdb_fsst_decoder_t *>(decoder);
	for (idx_t q = 0; q < state_count; q++) {
		auto row = substring_table.data() + q * SUBSTRING_ROW_STRIDE;
		auto escape_row = substring_table.data() + (state_count + q) * SUBSTRING_ROW_STRIDE;
		for (idx_t code = 0; code < SUBSTRING_ESCAPE_CODE; code++) {
			uint32_t state = UnsafeNumericCast<uint32_t>(q);
			const uint64_t symbol = fsst_decoder->symbol[code];
			const idx_t symbol_length = fsst_decoder->len[code];
			for (idx_t k = 0; k < symbol_length; k++) {
				state = dfa[state * 256 + ((symbol >> (8 * k)) & 0xFF)];
			}
			row[code] = UnsafeNumericCast<uint16_t>(state * SUBSTRING_ROW_STRIDE);
		}
		row[SUBSTRING_ESCAPE_CODE] = UnsafeNumericCast<uint16_t>((state_count + q) * SUBSTRING_ROW_STRIDE);
		for (idx_t b = 0; b < 256; b++) {
			escape_row[b] = UnsafeNumericCast<uint16_t>(dfa[q * 256 + b] * SUBSTRING_ROW_STRIDE);
		}
	}
	substring_accept = UnsafeNumericCast<uint32_t>(needle_size * SUBSTRING_ROW_STRIDE);
	substring_escape_start = UnsafeNumericCast<uint32_t>(state_count * SUBSTRING_ROW_STRIDE);
}

void CompressedStringScanState::MatchSubstringEntries(const uint32_t *entries, idx_t count) {
	constexpr idx_t LANES = 8;
	constexpr idx_t CHUNK_BYTES = 8;
	const unsigned char *positions[LANES];
	const unsigned char *ends[LANES];
	uint32_t states[LANES];
	uint32_t lane_entries[LANES];
	bool alive[LANES];
	const auto dict_bytes = layout.dictionary.GetBytes(0, layout.dictionary.Size()).data();
	const uint16_t *table = substring_table.data();
	const uint32_t accept = substring_accept;
	const uint32_t escape_start = substring_escape_start;
	auto entry_state = substring_entry_state.data();
	idx_t next_entry = 0;
	idx_t alive_lanes = 0;

	auto load = [&](idx_t lane) {
		while (next_entry < count) {
			const uint32_t entry = entries[next_entry++];
			const idx_t length = string_lengths[entry];
			if (length == 0) {
				entry_state[entry] = 1;
				continue;
			}
			positions[lane] = dict_bytes + string_offsets[entry];
			ends[lane] = positions[lane] + length;
			states[lane] = 0;
			lane_entries[lane] = entry;
			alive[lane] = true;
			return;
		}
		alive[lane] = false;
	};

	for (idx_t lane = 0; lane < LANES; lane++) {
		load(lane);
		alive_lanes += alive[lane];
	}
	while (alive_lanes > 0) {
#pragma GCC unroll 8
		for (idx_t lane = 0; lane < LANES; lane++) {
			if (!alive[lane]) {
				continue;
			}
			auto position = positions[lane];
			auto state = states[lane];
			const auto end = ends[lane];
			bool finished = false;
			if (static_cast<idx_t>(end - position) >= CHUNK_BYTES) {
				state = table[state + position[0]];
				state = table[state + position[1]];
				state = table[state + position[2]];
				state = table[state + position[3]];
				state = table[state + position[4]];
				state = table[state + position[5]];
				state = table[state + position[6]];
				state = table[state + position[7]];
				position += CHUNK_BYTES;
				finished = state == accept;
			} else {
				while (position < end && state != accept) {
					state = table[state + *position++];
				}
				if (state >= escape_start) {
					throw IOException("Failed to scan DICT_FSST string segment: invalid compressed string. "
					                  "Database file appears to be corrupted.");
				}
				finished = true;
			}
			if (finished) {
				entry_state[lane_entries[lane]] = state == accept ? 2 : 1;
				load(lane);
				alive_lanes -= !alive[lane];
			} else {
				positions[lane] = position;
				states[lane] = state;
			}
		}
	}
}

bool CompressedStringScanState::TryFilterSubstring(idx_t start, idx_t scan_count, Vector &result, SelectionVector &sel,
                                                   idx_t &sel_count, const TableFilter &filter) {
	if (!DictionaryPending() || !PrepareSubstringFilter(filter)) {
		return false;
	}
	const idx_t segment_rows = segment.count.load();
	const idx_t remaining_rows = segment_rows > start ? segment_rows - start : scan_count;
	const idx_t vectors_left = MaxValue<idx_t>(remaining_rows / STANDARD_VECTOR_SIZE, 1);
	if (substring_entry_state.empty() &&
	    MinValue<idx_t>(sel_count * vectors_left, layout.dict_count) < (substring_needle.size() + 1) * 8) {
		return false;
	}
	auto &selvec = GetSelVec(start, scan_count);
	if (substring_entry_state.empty()) {
		EnsureStringOffsets();
		BuildSubstringTable();
		substring_entry_state.assign(layout.dict_count, 0);
	}
	auto entry_state = substring_entry_state.data();
	const auto sel_data = sel.data();
	substring_todo.clear();
	for (idx_t i = 0; i < sel_count; i++) {
		const idx_t row = sel_data ? sel_data[i] : i;
		const auto entry = selvec.get_index(row);
		if (entry != 0 && entry_state[entry] == 0) {
			entry_state[entry] = 3;
			substring_todo.push_back(UnsafeNumericCast<uint32_t>(entry));
		}
	}
	MatchSubstringEntries(substring_todo.data(), substring_todo.size());

	const uint8_t wanted = substring_negate ? 1 : 2;
	SelectionVector new_sel(sel_count);
	const auto new_sel_data = new_sel.data();
	idx_t approved_tuple_count = 0;
	for (idx_t i = 0; i < sel_count; i++) {
		const idx_t row = sel_data ? sel_data[i] : i;
		const auto entry = selvec.get_index(row);
		new_sel_data[approved_tuple_count] = UnsafeNumericCast<sel_t>(row);
		approved_tuple_count += entry != 0 && entry_state[entry] == wanted;
	}
	if (approved_tuple_count < scan_count) {
		sel.Initialize(new_sel);
	}
	sel_count = approved_tuple_count;
	if (PreferSparseDecode(start, scan_count, sel_count)) {
		SparseDecodeByRow(result, start, scan_count, sel, sel_count);
	} else {
		EnsureDictionary();
		result.Dictionary(dictionary, GetSelVec(start, scan_count), scan_count);
	}
	return true;
}

const SelectionVector &CompressedStringScanState::GetSelVec(idx_t start, idx_t scan_count) {
	D_ASSERT(start <= segment.count.load());
	D_ASSERT(scan_count <= segment.count.load() - start);
	switch (layout.mode) {
	case DictFSSTMode::FSST_ONLY: {
		return *FlatVector::IncrementalSelectionVector();
	}
	default: {
		// Handling non-bitpacking-group-aligned start values;
		idx_t start_offset = start % BitpackingPrimitives::BITPACKING_ALGORITHM_GROUP_SIZE;

		// We will scan in blocks of BITPACKING_ALGORITHM_GROUP_SIZE, so we may scan some extra values.
		idx_t decompress_count = BitpackingPrimitives::RoundUpToAlgorithmGroupSize(scan_count + start_offset);

		// Start at the group containing start. Its leading values are removed after unpacking.
		const auto group_size = BitpackingPrimitives::BITPACKING_ALGORITHM_GROUP_SIZE;
		const auto group_bytes = group_size * idx_t(layout.dictionary_indices_width) / 8;
		const auto source_offset = (start / group_size) * group_bytes;
		const auto source_size =
		    BitpackingPrimitives::GetRequiredSize(decompress_count, layout.dictionary_indices_width);
		auto indices = layout.dictionary_indices.SubArray(source_offset, source_size);

		if (!sel_vec || sel_vec_size < decompress_count) {
			sel_vec_size = decompress_count;
			sel_vec = make_buffer<SelectionVector>(decompress_count);
		}

		sel_t *sel_vec_ptr = sel_vec->data();
		BitpackingPrimitives::UnPackBuffer<sel_t>(data_ptr_cast(sel_vec_ptr), indices.data(), decompress_count,
		                                          layout.dictionary_indices_width);

		sel_vec->ShiftLeft(start_offset, scan_count);

		// Validate the requested rows, not the leading values or final unpacking padding.
		bool has_invalid_index = false;
		for (idx_t i = 0; i < scan_count; i++) {
			has_invalid_index |= sel_vec->get_index_unsafe(i) >= layout.dict_count;
		}
		if (has_invalid_index) {
			ThrowDictionaryIndexOutOfBounds();
		}

		return *sel_vec;
	}
	}
}

void CompressedStringScanState::ScanToFlatVector(Vector &result, idx_t result_offset, idx_t start, idx_t scan_count,
                                                 StringScanType string_scan_type) {
	Select(result, result_offset, start, *FlatVector::IncrementalSelectionVector(), scan_count, string_scan_type);
	result.Verify();
}

void CompressedStringScanState::Select(Vector &result, idx_t result_offset, idx_t start, const SelectionVector &sel,
                                       idx_t sel_count, StringScanType string_scan_type) {
	// The last selected row bounds the unpacked range, so selections must be ordered.
#ifdef DEBUG
	for (idx_t i = 1; i < sel_count; i++) {
		D_ASSERT(sel.get_index(i - 1) <= sel.get_index(i));
	}
#endif
	const auto scan_count = sel_count == 0 ? 0 : sel.get_index(sel_count - 1) + 1;
	auto &selvec = GetSelVec(start, scan_count);

	//! (index 0 is reserved for NULL, which we don't have in this mode)
	const idx_t start_offset = layout.mode == DictFSSTMode::FSST_ONLY ? start + 1 : 0;

	auto result_data = FlatVector::Writer<string_t>(result, sel_count, result_offset);
	if (string_scan_type == StringScanType::COPY_STRINGS) {
		// a select of many rows unpacks every length up front and computes all offsets on its first step back
		const bool few_rows = sel_count < BitpackingPrimitives::BITPACKING_ALGORITHM_GROUP_SIZE;
		if (!few_rows) {
			LoadStringLengths();
		}
		for (idx_t i = 0; i < sel_count; i++) {
			auto string_number = selvec.get_index(start_offset + sel.get_index(i));
			if (string_number == 0) {
				result_data.WriteNull();
				continue;
			}
			result_data.WriteStringRef(FetchStringFromDict(result, GetStringOffset(string_number, few_rows),
			                                               string_number, StringScanType::COPY_STRINGS));
		}
	} else if (PreferSparseDecode(start, scan_count, sel_count)) {
		EnsureStringOffsets();
		for (idx_t i = 0; i < sel_count; i++) {
			auto string_number = selvec.get_index(start_offset + sel.get_index(i));
			if (string_number == 0) {
				result_data.WriteNull();
				continue;
			}
			result_data.WriteStringRef(FetchStringFromDict(result, string_offsets[string_number], string_number));
		}
	} else if (EnsureDictionary()) {
		// We have prepared the full dictionary, we can reference these strings directly
		auto dictionary_values = FlatVector::GetData<string_t>(dictionary->data);
		for (idx_t i = 0; i < sel_count; i++) {
			// Lookup dict offset in index buffer
			auto string_number = selvec.get_index(start_offset + sel.get_index(i));
			if (string_number == 0) {
				result_data.WriteNull();
				continue;
			}
			result_data.WriteStringRef(dictionary_values[string_number]);
		}
	} else {
		LoadStringLengths();
		for (idx_t i = 0; i < sel_count; i++) {
			// Lookup dict offset in index buffer
			auto string_number = selvec.get_index(start_offset + sel.get_index(i));
			if (string_number == 0) {
				result_data.WriteNull();
				continue;
			}
			if (decompress_position > string_number) {
				throw InternalException("DICT_FSST: not performing a sequential scan?");
			}
			for (; decompress_position < string_number; decompress_position++) {
				decompress_offset += string_lengths[decompress_position];
			}
			result_data.WriteStringRef(FetchStringFromDict(result, decompress_offset, string_number));
		}
	}
}

bool CompressedStringScanState::AllowDictionaryScan(idx_t scan_count) {
	if (layout.mode == DictFSSTMode::FSST_ONLY) {
		return false;
	}
	if (scan_count != STANDARD_VECTOR_SIZE) {
		return false;
	}
	return EnsureDictionary();
}

void CompressedStringScanState::ScanToDictionaryVector(ColumnSegment &segment, Vector &result, idx_t result_offset,
                                                       idx_t start, idx_t scan_count) {
	D_ASSERT(scan_count == STANDARD_VECTOR_SIZE);
	D_ASSERT(result_offset == 0);

	auto &selvec = GetSelVec(start, scan_count);
	result.Dictionary(dictionary, selvec, scan_count);
	result.Verify();
}

} // namespace dict_fsst
} // namespace duckdb
