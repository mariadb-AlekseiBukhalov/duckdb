#pragma once

#include "duckdb/common/bitpacking.hpp"
#include "duckdb/storage/buffer/buffer_handle.hpp"
#include "duckdb/storage/compression/compression_segment_reader.hpp"
#include "duckdb/storage/compression/dict_fsst/common.hpp"
#include "duckdb/storage/table/scan_state.hpp"

namespace duckdb {

class DictionaryEntry;

namespace dict_fsst {

bool ParseSubstringFilter(const TableFilter &filter, string &needle, bool &negate);

//===--------------------------------------------------------------------===//
// Scan
//===--------------------------------------------------------------------===//
struct CompressedStringScanState : public SegmentScanState {
private:
	struct SegmentLayout {
		//! Compression mode read from disk, validated as mode < DictFSSTMode::COUNT.
		DictFSSTMode mode;
		//! Dictionary entry count read from disk, validated as nonzero.
		//! In FSST_ONLY mode, dict_count - 1 is at least the segment row count.
		//! With zero-width lengths, at most the NULL and empty-string entries are present.
		uint32_t dict_count;
		//! Bit widths read from disk, both validated as at most 32.
		bitpacking_width_t dictionary_indices_width;
		//! String lengths have a nonzero width in FSST modes.
		bitpacking_width_t string_lengths_width;
		//! Dictionary string bytes, FSST-encoded unless mode is DICTIONARY.
		CompressionSegmentReader dictionary;
		//! Serialized FSST symbol table.
		CompressionSegmentReader symbol_table;
		//! Packed per-entry string lengths from disk, decoded lengths are unvalidated.
		unsafe_array_ptr<const uint8_t> string_lengths;
		//! Packed row-to-dictionary indices from disk, decoded indices are unvalidated.
		unsafe_array_ptr<const uint8_t> dictionary_indices;
	};

public:
	CompressedStringScanState(ColumnSegment &segment, BufferHandle &&handle_p)
	    : segment(segment), handle(std::move(handle_p)), layout(ReadLayout(handle, segment)) {
	}

	~CompressedStringScanState() override;

public:
	void Initialize();
	void ScanToFlatVector(Vector &result, idx_t result_offset, idx_t start, idx_t scan_count,
	                      StringScanType string_scan_type = StringScanType::STANDARD);
	void ScanToDictionaryVector(ColumnSegment &segment, Vector &result, idx_t result_offset, idx_t start,
	                            idx_t scan_count);
	const SelectionVector &GetSelVec(idx_t start, idx_t scan_count);
	void Select(Vector &result, idx_t result_offset, idx_t start, const SelectionVector &sel, idx_t sel_count,
	            StringScanType string_scan_type = StringScanType::STANDARD);

	bool AllowDictionaryScan(idx_t scan_count);

	bool DictionaryPending() const {
		return layout.mode == DictFSSTMode::DICT_FSST && !dictionary;
	}
	bool PreferSparseDecode(idx_t start, idx_t scan_count, idx_t sel_count);
	void SparseDecodeByRow(Vector &result, idx_t start, idx_t scan_count, const SelectionVector &sel, idx_t sel_count);
	bool TryFilterSubstring(idx_t start, idx_t scan_count, Vector &result, SelectionVector &sel, idx_t &sel_count,
	                        const TableFilter &filter);

private:
	static SegmentLayout ReadLayout(const BufferHandle &handle, const ColumnSegment &segment);
	//! With COPY_STRINGS the string is owned by the result instead of pointing into the block
	string_t FetchStringFromDict(Vector &result, uint32_t dict_offset, idx_t dict_idx,
	                             StringScanType string_scan_type = StringScanType::STANDARD);
	//! Builds the dictionary on first use, returns false if this scan has no dictionary
	bool EnsureDictionary();
	void DecompressDictionary(Vector &dict_data, string_t *dict_child_data);
	void EnsureStringOffsets();
	bool PrepareSubstringFilter(const TableFilter &filter);
	void BuildSubstringTable();
	void MatchSubstringEntries(const uint32_t *entries, idx_t count);
	//! A string's offset from a running sum, restarted on a step back if allowed and cheap, else from all offsets
	uint32_t GetStringOffset(idx_t string_number, bool allow_restart);
	//! Unpacks the length of every string, which scans need; a few fetched strings only unpack the groups they use
	void LoadStringLengths();
	const uint32_t *GetStringLengthGroup(idx_t group_start);
	uint32_t GetStringLength(idx_t dict_idx);
	idx_t SumStringLengths(idx_t start, idx_t end);

public:
	ColumnSegment &segment;
	BufferHandle handle;
	SegmentLayout layout;

	buffer_ptr<SelectionVector> sel_vec;
	idx_t sel_vec_size = 0;

	// decompress offset/position - used for scanning without a dictionary
	uint32_t decompress_offset = 0;
	idx_t decompress_position = 0;

	//! Decoded lengths for layout.dict_count entries, followed by unpacking padding.
	//! The NULL length is zero and the total length does not exceed the dictionary size.
	vector<uint32_t> string_lengths;
	//! The last group of string lengths unpacked without unpacking all of them
	uint32_t string_length_group[BitpackingPrimitives::BITPACKING_ALGORITHM_GROUP_SIZE];
	idx_t string_length_group_start = DConstants::INVALID_INDEX;
	vector<uint32_t> string_offsets;
	//! String lengths summed by GetStringOffset so far
	idx_t summed_lengths = 0;

	buffer_ptr<DictionaryEntry> dictionary;
	void *decoder = nullptr;
	bool all_values_inlined = false;

	int8_t substring_filter_state = -1;
	bool substring_negate = false;
	string substring_needle;
	vector<uint16_t> substring_table;
	uint32_t substring_accept = 0;
	uint32_t substring_escape_start = 0;
	vector<uint8_t> substring_entry_state;
	vector<uint32_t> substring_todo;

	unsafe_unique_array<bool> filter_result;
	bool null_filter_result_initialized = false;
};

} // namespace dict_fsst

} // namespace duckdb
