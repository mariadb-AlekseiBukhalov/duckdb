#include "duckdb/execution/radix_ht_fast_int.hpp"

#include "duckdb/common/types/hash.hpp"
#include "duckdb/common/vector/flat_vector.hpp"
#include "duckdb/common/vector/unified_vector_format.hpp"
#include "duckdb/execution/aggregate_state_spilling.hpp"
#include "duckdb/execution/operator/aggregate/aggregate_object.hpp"
#include "duckdb/execution/radix_ht_sink_state.hpp"
#include "duckdb/execution/radix_partitioned_hashtable.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/storage/buffer_manager.hpp"
#include "duckdb/storage/temporary_memory_manager.hpp"

#include <cstring>

namespace duckdb {

namespace {

constexpr idx_t PREFETCH_DISTANCE = 16;
constexpr hash_t HASH_MULTIPLIER = 0x9E3779B97F4A7C15ULL;
constexpr idx_t USABLE_RESERVATION_PERCENT = 80;
constexpr idx_t MINIMUM_SLOT_BITS = 11;
constexpr idx_t MAXIMUM_INITIAL_TABLE_ROWS = 1ULL << 15;
constexpr idx_t WORD_BYTES = sizeof(uint64_t);
constexpr idx_t BYPASS_ABSORBED_DIVISOR = 8;
constexpr idx_t RUNS_ONLY_ABSORBED_DIVISOR = 16;

inline void PrefetchAddress(const void *address) {
#if defined(__GNUC__) || defined(__clang__)
	__builtin_prefetch(address);
#else
	(void)address;
#endif
}

idx_t KeyWidth(const LogicalType &type) {
	switch (type.InternalType()) {
	case PhysicalType::INT8:
	case PhysicalType::UINT8:
		return 1;
	case PhysicalType::INT16:
	case PhysicalType::UINT16:
		return 2;
	case PhysicalType::INT32:
	case PhysicalType::UINT32:
		return 4;
	case PhysicalType::INT64:
	case PhysicalType::UINT64:
		return 8;
	default:
		return 0;
	}
}

bool IsPlainCountStar(const BoundAggregateExpression &aggr) {
	return aggr.GetAggregateType() == AggregateType::NON_DISTINCT && !aggr.GetFilter() && !aggr.GetOrderBys() &&
	       aggr.StateExportMode() == AggregateStateExportMode::NONE && aggr.GetChildren().empty() &&
	       !aggr.Function().HasStateDestructorCallback() &&
	       aggr.Function().GetName().GetIdentifierName() == "count_star";
}

inline uint64_t LoadWord(const_data_ptr_t pointer) {
	uint64_t result;
	memcpy(&result, pointer, sizeof(result));
	return result;
}

inline hash_t FastHash(uint64_t key) {
	return key * HASH_MULTIPLIER;
}

inline idx_t PartitionOf(hash_t hash) {
	return static_cast<idx_t>(hash >> (64 - FastIntSetup::RADIX_BITS));
}

bool ReserveMemory(ClientContext &context, RadixHTGlobalSinkState &gstate, idx_t additional_bytes) {
	const idx_t needed = gstate.fast->total_bytes.load(std::memory_order_relaxed) + additional_bytes;
	auto &temporary_memory = *gstate.temporary_memory_state;
	const auto usable = [&]() {
		return MinValue<idx_t>(temporary_memory.GetReservation() / 100 * USABLE_RESERVATION_PERCENT,
		                       gstate.memory_limit / 2);
	};
	if (needed <= usable()) {
		return true;
	}
	const annotated_lock_guard<annotated_mutex> guard {gstate.lock};
	if (needed <= usable()) {
		return true;
	}
	temporary_memory.SetRemainingSizeAndUpdateReservation(context, MinValue<idx_t>(2 * needed, gstate.memory_limit));
	return needed <= usable();
}

} // namespace

class FastIntTable {
public:
	FastIntTable(Allocator &allocator_p, idx_t entry_width_p, idx_t expected_rows)
	    : allocator(allocator_p), entry_width(entry_width_p), count(0) {
		slot_bits = MINIMUM_SLOT_BITS;
		while ((1ULL << slot_bits) < MinValue<idx_t>(expected_rows, MAXIMUM_INITIAL_TABLE_ROWS) * 2) {
			slot_bits++;
		}
		slot_capacity = 1ULL << slot_bits;
		slots_data = allocator.Allocate(slot_capacity * sizeof(uint32_t));
		slots = reinterpret_cast<uint32_t *>(slots_data.get());
		memset(slots, 0, slot_capacity * sizeof(uint32_t));
		entry_capacity = slot_capacity / 2;
		entries_data = allocator.Allocate(entry_capacity * entry_width);
		entries = entries_data.get();
	}

	idx_t Count() const {
		return count;
	}

	data_ptr_t Entry(idx_t id) const {
		return entries + id * entry_width;
	}

	void FindOrInsert(const uint64_t *keys, idx_t n, uint32_t *ids) {
		hash_t hashes[STANDARD_VECTOR_SIZE];
		for (idx_t i = 0; i < n; i++) {
			hashes[i] = FastHash(keys[i]);
		}
		const idx_t width = entry_width;
		uint32_t *table_slots = slots;
		data_ptr_t table_entries = entries;
		idx_t slot_shift = 64 - slot_bits;
		idx_t mask = slot_capacity - 1;
		const idx_t prefix = FastIntSetup::RADIX_BITS;
		const idx_t prefetch_slots = MinValue<idx_t>(PREFETCH_DISTANCE, n);
		for (idx_t i = 0; i < prefetch_slots; i++) {
			PrefetchAddress(&table_slots[(hashes[i] << prefix) >> slot_shift]);
		}
		for (idx_t i = 0; i < n; i++) {
			if (i + PREFETCH_DISTANCE < n) {
				PrefetchAddress(&table_slots[(hashes[i + PREFETCH_DISTANCE] << prefix) >> slot_shift]);
			}
			if (i + PREFETCH_DISTANCE / 2 < n) {
				const auto slot = table_slots[(hashes[i + PREFETCH_DISTANCE / 2] << prefix) >> slot_shift];
				if (slot) {
					PrefetchAddress(table_entries + static_cast<idx_t>(slot - 1) * width);
				}
			}
			const uint64_t key = keys[i];
			idx_t position = (hashes[i] << prefix) >> slot_shift;
			while (true) {
				const uint32_t slot = table_slots[position];
				if (slot == 0) {
					if (count == entry_capacity || (count + 1) * 2 > slot_capacity) {
						Grow();
						table_slots = slots;
						table_entries = entries;
						slot_shift = 64 - slot_bits;
						mask = slot_capacity - 1;
						position = (hashes[i] << prefix) >> slot_shift;
						while (table_slots[position]) {
							position = (position + 1) & mask;
						}
					}
					memcpy(table_entries + count * width, &key, sizeof(key));
					table_slots[position] = static_cast<uint32_t>(count + 1);
					ids[i] = static_cast<uint32_t>(count);
					count++;
					break;
				}
				if (LoadWord(table_entries + static_cast<idx_t>(slot - 1) * width) == key) {
					ids[i] = slot - 1;
					break;
				}
				position = (position + 1) & mask;
			}
		}
	}

private:
	void Grow() {
		if (count == entry_capacity) {
			GrowEntries();
		}
		if ((count + 1) * 2 > slot_capacity) {
			GrowSlots();
		}
	}

	void GrowEntries() {
		if (entry_capacity >= NumericLimits<uint32_t>::Maximum() / 2) {
			throw InternalException("Too many groups in one partition of the integer aggregate");
		}
		const auto new_capacity = entry_capacity * 2;
		auto new_data = allocator.Allocate(new_capacity * entry_width);
		memcpy(new_data.get(), entries, count * entry_width);
		entries_data = std::move(new_data);
		entries = entries_data.get();
		entry_capacity = new_capacity;
	}

	void GrowSlots() {
		const auto new_bits = slot_bits + 1;
		const auto new_capacity = 1ULL << new_bits;
		auto new_data = allocator.Allocate(new_capacity * sizeof(uint32_t));
		auto new_slots = reinterpret_cast<uint32_t *>(new_data.get());
		memset(new_slots, 0, new_capacity * sizeof(uint32_t));
		const auto new_mask = new_capacity - 1;
		for (idx_t id = 0; id < count; id++) {
			auto position =
			    (FastHash(LoadWord(entries + id * entry_width)) << FastIntSetup::RADIX_BITS) >> (64 - new_bits);
			while (new_slots[position]) {
				position = (position + 1) & new_mask;
			}
			new_slots[position] = static_cast<uint32_t>(id + 1);
		}
		slots_data = std::move(new_data);
		slots = new_slots;
		slot_bits = new_bits;
		slot_capacity = new_capacity;
		if (entry_capacity < slot_capacity / 2) {
			GrowEntries();
		}
	}

private:
	Allocator &allocator;
	idx_t entry_width;
	idx_t count;
	idx_t slot_bits;
	idx_t slot_capacity;
	idx_t entry_capacity;
	AllocatedData slots_data;
	uint32_t *slots;
	AllocatedData entries_data;
	data_ptr_t entries;
};

FastIntGlobalState::FastIntGlobalState(unique_ptr<FastIntSetup> setup_p)
    : setup(std::move(setup_p)), fallback(false), total_bytes(0), null_rows(0), total_rows(0), next_scan_partition(0),
      scanned_partitions(0) {
}

FastIntRowSink::FastIntRowSink(idx_t row_words_p, idx_t partition_count)
    : row_words(row_words_p), partitions(partition_count), write_pos(partition_count, nullptr),
      write_end(partition_count, nullptr) {
}

uint64_t *FastIntRowSink::NewChunk(Allocator &allocator, atomic<idx_t> &byte_counter, idx_t partition) {
	auto &chunks = partitions[partition];
	idx_t bytes = FastIntSetup::MINIMUM_CHUNK_BYTES;
	if (!chunks.empty()) {
		chunks.back().rows = chunks.back().capacity;
		bytes = MinValue<idx_t>(FastIntSetup::MAXIMUM_CHUNK_BYTES, chunks.back().capacity * row_words * WORD_BYTES * 2);
	}
	FastIntRawChunk chunk;
	chunk.capacity = bytes / (row_words * WORD_BYTES);
	const idx_t allocation_size = chunk.capacity * row_words * WORD_BYTES;
	chunk.data = allocator.Allocate(allocation_size);
	byte_counter.fetch_add(allocation_size, std::memory_order_relaxed);
	auto pointer = reinterpret_cast<uint64_t *>(chunk.data.get());
	write_pos[partition] = pointer;
	write_end[partition] = pointer + chunk.capacity * row_words;
	chunks.push_back(std::move(chunk));
	return pointer;
}

void FastIntRowSink::FinishChunks() {
	for (idx_t partition = 0; partition < partitions.size(); partition++) {
		auto &chunks = partitions[partition];
		if (chunks.empty()) {
			continue;
		}
		auto base = reinterpret_cast<uint64_t *>(chunks.back().data.get());
		chunks.back().rows = static_cast<idx_t>(write_pos[partition] - base) / row_words;
	}
}

FastIntLocalSink::FastIntLocalSink(Allocator &allocator, const FastIntSetup &setup)
    : rows(1, FastIntSetup::TotalPartitions()), input_rows(0), null_rows(0), mode(FastIntSetup::MODE_CACHED),
      mode_remaining(0), window_rows(0), window_runs(0), window_misses(0) {
	if (setup.count_star) {
		counted_rows = make_uniq<FastIntRowSink>(2, FastIntSetup::TotalPartitions());
	}
	const idx_t cache_bytes = (1ULL << FastIntSetup::CACHE_BITS) * 2 * sizeof(uint64_t);
	cache_data = allocator.Allocate(cache_bytes);
	cache = reinterpret_cast<uint64_t *>(cache_data.get());
	memset(cache, 0, cache_bytes);
}

FastIntLocalSource::FastIntLocalSource() {
}

FastIntLocalSource::~FastIntLocalSource() {
}

unique_ptr<FastIntGlobalState> FastIntAggregate::TryCreate(const RadixPartitionedHashTable &radix_ht,
                                                           idx_t estimated_rows, idx_t estimated_groups) {
	if (estimated_rows < FastIntSetup::MINIMUM_INPUT_ROWS || estimated_groups < FastIntSetup::MINIMUM_GROUPS) {
		return nullptr;
	}
	auto &op = radix_ht.op;
	if (radix_ht.grouping_set.size() != 1 || op.GroupCount() != 1 || !radix_ht.null_groups.empty() ||
	    !op.GetGroupingFunctions().empty() || radix_ht.group_types.size() != 1) {
		return nullptr;
	}
	auto setup = make_uniq<FastIntSetup>();
	setup->key_width = KeyWidth(radix_ht.group_types[0]);
	if (setup->key_width == 0) {
		return nullptr;
	}
	if (op.bindings.size() > 1) {
		return nullptr;
	}
	if (op.bindings.size() == 1) {
		if (!IsPlainCountStar(*op.bindings[0])) {
			return nullptr;
		}
		setup->count_star = true;
	}
	auto aggregate_objects = AggregateObject::CreateAggregateObjects(op.bindings);
	setup->state_layout.Initialize({LogicalType::BIGINT}, std::move(aggregate_objects),
	                               TupleDataValidityType::CANNOT_HAVE_NULL_VALUES);
	if (setup->state_layout.HasDestructor() || AggregateStateSpilling::TryCreateSpillPlan(setup->state_layout)) {
		return nullptr;
	}
	if (setup->count_star && setup->state_layout.GetAggregates()[0].payload_size != sizeof(int64_t)) {
		return nullptr;
	}
	setup->scan_types.push_back(radix_ht.group_types[0]);
	for (auto &type : op.aggregate_return_types) {
		setup->scan_types.push_back(type);
	}
	return make_uniq<FastIntGlobalState>(std::move(setup));
}

bool FastIntAggregate::Active(const RadixHTGlobalSinkState &gstate) {
	return gstate.fast && !gstate.fast->fallback.load();
}

void FastIntAggregate::ResetGlobal(RadixHTGlobalSinkState &gstate) {
	if (!gstate.fast) {
		return;
	}
	auto &state = *gstate.fast;
	state.fallback = false;
	state.total_bytes = 0;
	state.partitions.clear();
	state.counted_partitions.clear();
	state.null_rows = 0;
	state.total_rows = 0;
	state.scan_partitions.clear();
	state.next_scan_partition = 0;
	state.scanned_partitions = 0;
}

namespace {

struct FastIntReplayInput {
	vector<FastIntPartition> rows;
	vector<FastIntPartition> counted_rows;
	idx_t null_rows = 0;
};

class FastIntReplayer {
public:
	FastIntReplayer(const RadixPartitionedHashTable &radix_ht_p, ExecutionContext &context_p,
	                RadixHTGlobalSinkState &gstate_p, RadixHTLocalSinkState &lstate_p)
	    : radix_ht(radix_ht_p), context(context_p), gstate(gstate_p), lstate(lstate_p),
	      key_width(gstate_p.fast->Setup().key_width), pending(0) {
		key_chunk.Initialize(BufferAllocator::Get(context.client), radix_ht.group_types);
		for (idx_t i = 0; i < radix_ht.op.aggregates.size(); i++) {
			filter.push_back(i);
		}
	}

	void Replay(FastIntReplayInput &input) {
		for (auto &partition : input.rows) {
			for (auto &chunk : partition) {
				auto rows = reinterpret_cast<const uint64_t *>(chunk.data.get());
				for (idx_t i = 0; i < chunk.rows; i++) {
					Add(rows[i], 1);
				}
				Release(chunk, 1);
			}
			partition.clear();
		}
		for (auto &partition : input.counted_rows) {
			for (auto &chunk : partition) {
				auto rows = reinterpret_cast<const uint64_t *>(chunk.data.get());
				for (idx_t i = 0; i < chunk.rows; i++) {
					Add(rows[2 * i], rows[2 * i + 1]);
				}
				Release(chunk, 2);
			}
			partition.clear();
		}
		Flush(false);
		for (idx_t remaining = input.null_rows; remaining > 0;) {
			pending = MinValue<idx_t>(STANDARD_VECTOR_SIZE, remaining);
			remaining -= pending;
			Flush(true);
		}
	}

private:
	void Release(FastIntRawChunk &chunk, idx_t row_words) {
		chunk.data = AllocatedData();
		gstate.fast->total_bytes.fetch_sub(chunk.capacity * row_words * WORD_BYTES, std::memory_order_relaxed);
	}

	void Add(uint64_t key, uint64_t weight) {
		while (weight > 0) {
			const idx_t take = MinValue<uint64_t>(weight, STANDARD_VECTOR_SIZE - pending);
			for (idx_t i = 0; i < take; i++) {
				keys[pending + i] = key;
			}
			pending += take;
			weight -= take;
			if (pending == STANDARD_VECTOR_SIZE) {
				Flush(false);
			}
		}
	}

	void Flush(bool null_keys) {
		if (pending == 0) {
			return;
		}
		key_chunk.Reset();
		key_chunk.SetChildCardinality(pending);
		auto &key_vector = key_chunk.data[0];
		auto key_data = FlatVector::GetDataMutable(key_vector);
		for (idx_t i = 0; i < pending; i++) {
			memcpy(key_data + i * key_width, &keys[i], key_width);
		}
		if (null_keys) {
			FlatVector::ValidityMutable(key_vector).SetAllInvalid(pending);
		}
		lstate.group_chunk.data[0].Reference(key_vector);
		lstate.group_chunk.SetChildCardinality(pending);
		radix_ht.SinkInternal(context, gstate, lstate, empty_payload, filter);
		pending = 0;
	}

private:
	const RadixPartitionedHashTable &radix_ht;
	ExecutionContext &context;
	RadixHTGlobalSinkState &gstate;
	RadixHTLocalSinkState &lstate;
	const idx_t key_width;
	DataChunk key_chunk;
	DataChunk empty_payload;
	unsafe_vector<idx_t> filter;
	uint64_t keys[STANDARD_VECTOR_SIZE];
	idx_t pending;
};

inline void AppendSingle(FastIntRowSink &sink, Allocator &allocator, atomic<idx_t> &byte_counter, idx_t partition,
                         uint64_t key) {
	uint64_t *destination = sink.write_pos[partition];
	if (destination == sink.write_end[partition]) {
		destination = sink.NewChunk(allocator, byte_counter, partition);
	}
	*destination = key;
	sink.write_pos[partition] = destination + 1;
}

inline void AppendCounted(FastIntRowSink &sink, Allocator &allocator, atomic<idx_t> &byte_counter, idx_t partition,
                          uint64_t key, uint64_t count) {
	uint64_t *destination = sink.write_pos[partition];
	if (destination == sink.write_end[partition]) {
		destination = sink.NewChunk(allocator, byte_counter, partition);
	}
	destination[0] = key;
	destination[1] = count;
	sink.write_pos[partition] = destination + 2;
}

inline void EvictEntry(const FastIntSetup &setup, FastIntLocalSink &local, Allocator &allocator,
                       atomic<idx_t> &byte_counter, const uint64_t *entry) {
	const idx_t partition = PartitionOf(FastHash(entry[0]));
	if (!setup.count_star || entry[1] == 1) {
		AppendSingle(local.rows, allocator, byte_counter, partition, entry[0]);
	} else {
		AppendCounted(*local.counted_rows, allocator, byte_counter, partition, entry[0], entry[1]);
	}
}

void FlushCache(const FastIntSetup &setup, FastIntLocalSink &local, Allocator &allocator, atomic<idx_t> &byte_counter) {
	for (idx_t i = 0; i < (1ULL << FastIntSetup::CACHE_BITS); i++) {
		uint64_t *entry = local.cache + 2 * i;
		if (entry[1] != 0) {
			EvictEntry(setup, local, allocator, byte_counter, entry);
			entry[1] = 0;
		}
	}
}

void CollectLocal(const FastIntSetup &setup, FastIntLocalSink &local, Allocator &allocator, atomic<idx_t> &byte_counter,
                  FastIntReplayInput &output) {
	FlushCache(setup, local, allocator, byte_counter);
	local.rows.FinishChunks();
	output.rows = std::move(local.rows.partitions);
	if (local.counted_rows) {
		local.counted_rows->FinishChunks();
		output.counted_rows = std::move(local.counted_rows->partitions);
	}
	output.null_rows += local.null_rows;
}

void GiveUp(const RadixPartitionedHashTable &radix_ht, ExecutionContext &context, RadixHTGlobalSinkState &gstate,
            RadixHTLocalSinkState &lstate) {
	auto &fast = *gstate.fast;
	FastIntReplayInput stolen;
	{
		const annotated_lock_guard<annotated_mutex> guard {gstate.lock};
		if (!fast.fallback.exchange(true)) {
			stolen.rows = std::move(fast.partitions);
			stolen.counted_rows = std::move(fast.counted_partitions);
			stolen.null_rows = fast.null_rows;
			fast.partitions.clear();
			fast.counted_partitions.clear();
			fast.null_rows = 0;
		}
	}
	FastIntReplayer replayer(radix_ht, context, gstate, lstate);
	if (lstate.fast) {
		FastIntReplayInput own;
		CollectLocal(fast.Setup(), *lstate.fast, BufferAllocator::Get(context.client), fast.total_bytes, own);
		lstate.fast.reset();
		replayer.Replay(own);
	}
	replayer.Replay(stolen);
}

template <class U>
void PartitionKeys(const FastIntSetup &setup, FastIntLocalSink &local, Allocator &allocator,
                   atomic<idx_t> &byte_counter, const UnifiedVectorFormat &format, idx_t count) {
	const auto data = reinterpret_cast<const U *>(format.data);
	const auto &selection = *format.sel;
	const bool check_null = format.validity.CanHaveNull();
	const idx_t cache_shift = 64 - FastIntSetup::CACHE_BITS;
	auto cache = local.cache;
	auto &sink = local.rows;
	uint64_t run_keys[STANDARD_VECTOR_SIZE + 1];
	uint64_t run_lengths[STANDARD_VECTOR_SIZE + 1];
	idx_t done = 0;
	while (done < count) {
		if (local.mode == FastIntSetup::MODE_BYPASS) {
			const idx_t end = done + MinValue<idx_t>(count - done, local.mode_remaining);
			local.mode_remaining -= end - done;
			for (; done < end; done++) {
				const idx_t index = selection.get_index(done);
				if (check_null && !format.validity.RowIsValid(index)) {
					local.null_rows++;
					continue;
				}
				const uint64_t key = static_cast<U>(data[index]);
				AppendSingle(sink, allocator, byte_counter, PartitionOf(FastHash(key)), key);
			}
			if (local.mode_remaining == 0) {
				local.mode = FastIntSetup::MODE_CACHED;
			}
			continue;
		}
		const bool cached = local.mode == FastIntSetup::MODE_CACHED;
		const idx_t limit = cached ? FastIntSetup::WINDOW_ROWS - local.window_rows : local.mode_remaining;
		const idx_t end = done + MinValue<idx_t>(count - done, limit);
		const idx_t rows = end - done;
		idx_t run_count = 0;
		if (!check_null) {
			uint64_t current = static_cast<U>(data[selection.get_index(done)]);
			uint64_t length = 0;
			for (idx_t i = done; i < end; i++) {
				const uint64_t key = static_cast<U>(data[selection.get_index(i)]);
				const bool same = key == current;
				run_keys[run_count] = current;
				run_lengths[run_count] = length;
				run_count += !same;
				length = same ? length + 1 : 1;
				current = key;
			}
			run_keys[run_count] = current;
			run_lengths[run_count] = length;
			run_count++;
		} else {
			for (idx_t i = done; i < end; i++) {
				const idx_t index = selection.get_index(i);
				if (!format.validity.RowIsValid(index)) {
					local.null_rows++;
					continue;
				}
				run_keys[run_count] = static_cast<U>(data[index]);
				run_lengths[run_count] = 1;
				run_count++;
			}
		}
		done = end;
		if (cached) {
			idx_t misses = 0;
			for (idx_t r = 0; r < run_count; r++) {
				const uint64_t key = run_keys[r];
				uint64_t *entry = cache + 2 * ((FastHash(key) << FastIntSetup::RADIX_BITS) >> cache_shift);
				if (entry[1] != 0) {
					if (entry[0] == key) {
						entry[1] += run_lengths[r];
						continue;
					}
					EvictEntry(setup, local, allocator, byte_counter, entry);
				}
				entry[0] = key;
				entry[1] = run_lengths[r];
				misses++;
			}
			local.window_rows += rows;
			local.window_runs += run_count;
			local.window_misses += misses;
			if (local.window_rows >= FastIntSetup::WINDOW_ROWS) {
				const idx_t absorbed = local.window_rows - local.window_misses;
				const idx_t absorbed_by_cache = local.window_runs - local.window_misses;
				if (absorbed * BYPASS_ABSORBED_DIVISOR < local.window_rows) {
					FlushCache(setup, local, allocator, byte_counter);
					local.mode = FastIntSetup::MODE_BYPASS;
					local.mode_remaining = FastIntSetup::MODE_DURATION_ROWS;
				} else if (absorbed_by_cache * RUNS_ONLY_ABSORBED_DIVISOR < local.window_rows) {
					FlushCache(setup, local, allocator, byte_counter);
					local.mode = FastIntSetup::MODE_RUNS_ONLY;
					local.mode_remaining = FastIntSetup::MODE_DURATION_ROWS;
				}
				local.window_rows = 0;
				local.window_runs = 0;
				local.window_misses = 0;
			}
		} else {
			for (idx_t r = 0; r < run_count; r++) {
				const uint64_t key = run_keys[r];
				const idx_t partition = PartitionOf(FastHash(key));
				if (run_lengths[r] == 1 || !setup.count_star) {
					AppendSingle(sink, allocator, byte_counter, partition, key);
				} else {
					AppendCounted(*local.counted_rows, allocator, byte_counter, partition, key, run_lengths[r]);
				}
			}
			local.mode_remaining -= rows;
			if (local.mode_remaining == 0) {
				local.mode = FastIntSetup::MODE_CACHED;
			}
		}
	}
}

} // namespace

bool FastIntAggregate::Sink(const RadixPartitionedHashTable &radix_ht, ExecutionContext &context,
                            RadixHTGlobalSinkState &gstate, RadixHTLocalSinkState &lstate, DataChunk &group_chunk,
                            const unsafe_vector<idx_t> &filter) {
	auto &fast = *gstate.fast;
	if (fast.fallback.load()) {
		if (lstate.fast) {
			GiveUp(radix_ht, context, gstate, lstate);
		}
		return false;
	}
	auto &setup = fast.Setup();
	if (filter.size() != radix_ht.op.aggregates.size()) {
		GiveUp(radix_ht, context, gstate, lstate);
		return false;
	}
	const idx_t count = group_chunk.size();
	if (count == 0) {
		return true;
	}
	auto &allocator = BufferAllocator::Get(context.client);
	if (!lstate.fast) {
		lstate.fast = make_uniq<FastIntLocalSink>(allocator, setup);
	}
	auto &local = *lstate.fast;
	if (!ReserveMemory(context.client, gstate, count * sizeof(uint64_t))) {
		GiveUp(radix_ht, context, gstate, lstate);
		return false;
	}
	UnifiedVectorFormat key_format;
	group_chunk.data[0].ToUnifiedFormat(key_format);
	auto &counter = fast.total_bytes;
	switch (setup.key_width) {
	case 1:
		PartitionKeys<uint8_t>(setup, local, allocator, counter, key_format, count);
		break;
	case 2:
		PartitionKeys<uint16_t>(setup, local, allocator, counter, key_format, count);
		break;
	case 4:
		PartitionKeys<uint32_t>(setup, local, allocator, counter, key_format, count);
		break;
	default:
		PartitionKeys<uint64_t>(setup, local, allocator, counter, key_format, count);
		break;
	}
	local.input_rows += count;
	return true;
}

bool FastIntAggregate::Combine(const RadixPartitionedHashTable &radix_ht, ExecutionContext &context,
                               RadixHTGlobalSinkState &gstate, RadixHTLocalSinkState &lstate) {
	auto &fast = *gstate.fast;
	if (!lstate.fast) {
		return !fast.fallback.load();
	}
	FastIntReplayInput local_output;
	CollectLocal(fast.Setup(), *lstate.fast, BufferAllocator::Get(context.client), fast.total_bytes, local_output);
	const idx_t local_rows = lstate.fast->input_rows;
	{
		const annotated_lock_guard<annotated_mutex> guard {gstate.lock};
		if (!fast.fallback.load()) {
			if (fast.partitions.empty()) {
				fast.partitions.resize(FastIntSetup::TotalPartitions());
				fast.counted_partitions.resize(FastIntSetup::TotalPartitions());
			}
			for (idx_t partition = 0; partition < FastIntSetup::TotalPartitions(); partition++) {
				for (auto &chunk : local_output.rows[partition]) {
					fast.partitions[partition].push_back(std::move(chunk));
				}
				if (!local_output.counted_rows.empty()) {
					for (auto &chunk : local_output.counted_rows[partition]) {
						fast.counted_partitions[partition].push_back(std::move(chunk));
					}
				}
			}
			fast.null_rows += local_output.null_rows;
			fast.total_rows += local_rows;
			lstate.fast.reset();
			return true;
		}
	}
	lstate.fast.reset();
	FastIntReplayer replayer(radix_ht, context, gstate, lstate);
	replayer.Replay(local_output);
	return false;
}

void FastIntAggregate::Finalize(ClientContext &context, RadixHTGlobalSinkState &gstate) {
	auto &fast = *gstate.fast;
	fast.scan_partitions.clear();
	for (idx_t partition = 0; partition < fast.partitions.size(); partition++) {
		idx_t rows = 0;
		for (auto &chunk : fast.partitions[partition]) {
			rows += chunk.rows;
		}
		for (auto &chunk : fast.counted_partitions[partition]) {
			rows += chunk.rows;
		}
		if (partition == FastIntSetup::NullPartition()) {
			rows += fast.null_rows;
		}
		if (rows != 0) {
			fast.scan_partitions.push_back(partition);
		}
	}
	fast.next_scan_partition = 0;
	fast.scanned_partitions = 0;
	gstate.count_before_combining = fast.total_rows;
	gstate.temporary_memory_state->SetRemainingSizeAndUpdateReservation(
	    context, MinValue<idx_t>(fast.total_bytes.load(), gstate.memory_limit));
}

idx_t FastIntAggregate::MaxThreads(const RadixHTGlobalSinkState &gstate) {
	return MinValue<idx_t>(gstate.number_of_threads, gstate.fast->scan_partitions.size());
}

namespace {

void InitializeSource(ExecutionContext &context, const FastIntSetup &setup, FastIntLocalSource &source) {
	auto &allocator = BufferAllocator::Get(context.client);
	source.layout = setup.state_layout.Copy();
	source.allocator = make_uniq<ArenaAllocator>(allocator);
	source.row_state = make_uniq<RowOperationsState>(*source.allocator);
	source.scan_chunk.Initialize(allocator, setup.scan_types);
	source.addresses = make_uniq<Vector>(LogicalType::POINTER);
	source.initialized = true;
}

void BuildTable(const FastIntSetup &setup, ExecutionContext &context, FastIntLocalSource &source,
                FastIntPartition &rows_list, FastIntPartition &counted_list, idx_t null_rows) {
	idx_t total_rows = 0;
	for (auto &chunk : rows_list) {
		total_rows += chunk.rows;
	}
	for (auto &chunk : counted_list) {
		total_rows += chunk.rows;
	}
	auto &layout = source.layout;
	source.table = make_uniq<FastIntTable>(BufferAllocator::Get(context.client), layout.GetRowWidth(), total_rows);
	auto &table = *source.table;
	const idx_t aggregate_offset = layout.GetAggrOffset();
	uint64_t keys[STANDARD_VECTOR_SIZE];
	uint32_t ids[STANDARD_VECTOR_SIZE];

	auto process = [&](const uint64_t *batch, idx_t count, idx_t stride) {
		const uint64_t *key_pointer = batch;
		if (stride != 1) {
			for (idx_t i = 0; i < count; i++) {
				keys[i] = batch[i * stride];
			}
			key_pointer = keys;
		}
		const idx_t groups_before = table.Count();
		table.FindOrInsert(key_pointer, count, ids);
		if (!setup.count_star) {
			return;
		}
		for (idx_t id = groups_before; id < table.Count(); id++) {
			*reinterpret_cast<int64_t *>(table.Entry(id) + aggregate_offset) = 0;
		}
		for (idx_t i = 0; i < count; i++) {
			auto state = reinterpret_cast<int64_t *>(table.Entry(ids[i]) + aggregate_offset);
			*state += stride == 1 ? 1 : static_cast<int64_t>(batch[i * stride + 1]);
		}
	};

	for (auto &chunk : rows_list) {
		auto rows = reinterpret_cast<const uint64_t *>(chunk.data.get());
		for (idx_t offset = 0; offset < chunk.rows; offset += STANDARD_VECTOR_SIZE) {
			process(rows + offset, MinValue<idx_t>(STANDARD_VECTOR_SIZE, chunk.rows - offset), 1);
		}
		chunk.data = AllocatedData();
	}
	rows_list.clear();
	for (auto &chunk : counted_list) {
		auto rows = reinterpret_cast<const uint64_t *>(chunk.data.get());
		for (idx_t offset = 0; offset < chunk.rows; offset += STANDARD_VECTOR_SIZE) {
			process(rows + 2 * offset, MinValue<idx_t>(STANDARD_VECTOR_SIZE, chunk.rows - offset), 2);
		}
		chunk.data = AllocatedData();
	}
	counted_list.clear();
	if (null_rows != 0) {
		uint64_t null_batch[2] = {0, null_rows};
		process(null_batch, 1, 2);
	}
}

} // namespace

bool FastIntAggregate::Scan(const RadixPartitionedHashTable &radix_ht, ExecutionContext &context,
                            RadixHTGlobalSinkState &gstate, FastIntLocalSource &source, DataChunk &chunk) {
	auto &fast = *gstate.fast;
	auto &setup = fast.Setup();
	if (!source.initialized) {
		InitializeSource(context, setup, source);
	}
	while (!source.table || source.position >= source.table->Count()) {
		if (source.table) {
			source.table.reset();
			if (++fast.scanned_partitions == fast.scan_partitions.size()) {
				gstate.temporary_memory_state->SetZero();
			}
		}
		source.position = 0;
		const idx_t task = fast.next_scan_partition++;
		if (task >= fast.scan_partitions.size()) {
			return false;
		}
		const idx_t partition = fast.scan_partitions[task];
		source.null_partition = partition == FastIntSetup::NullPartition();
		FastIntPartition no_rows;
		BuildTable(setup, context, source, fast.partitions[partition],
		           fast.counted_partitions.empty() ? no_rows : fast.counted_partitions[partition],
		           source.null_partition ? fast.null_rows : 0);
	}

	auto &table = *source.table;
	const idx_t count = MinValue<idx_t>(STANDARD_VECTOR_SIZE, table.Count() - source.position);
	auto &scan_chunk = source.scan_chunk;
	scan_chunk.Reset();
	scan_chunk.SetChildCardinality(count);
	auto key_data = FlatVector::GetDataMutable(scan_chunk.data[0]);
	auto pointers = FlatVector::GetDataMutable<data_ptr_t>(*source.addresses);
	FlatVector::SetSize(*source.addresses, count);
	for (idx_t i = 0; i < count; i++) {
		auto entry = table.Entry(source.position + i);
		memcpy(key_data + i * setup.key_width, entry, setup.key_width);
		pointers[i] = entry;
	}
	if (source.null_partition) {
		FlatVector::ValidityMutable(scan_chunk.data[0]).SetAllInvalid(count);
	}
	source.position += count;
	RowOperations::FinalizeStates(*source.row_state, source.layout, *source.addresses, scan_chunk, 1);

	idx_t chunk_index = 0;
	for (auto &entry : radix_ht.grouping_set) {
		chunk.data[entry].Reference(scan_chunk.data[chunk_index++]);
	}
	for (idx_t col_idx = 0; col_idx < radix_ht.op.aggregates.size(); col_idx++) {
		chunk.data[radix_ht.op.GroupCount() + col_idx].Reference(scan_chunk.data[1 + col_idx]);
	}
	return true;
}

} // namespace duckdb
