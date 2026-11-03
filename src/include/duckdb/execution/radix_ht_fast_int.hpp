
#pragma once

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/atomic.hpp"
#include "duckdb/common/common.hpp"
#include "duckdb/common/row_operations/row_operations.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/row/tuple_data_layout.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/storage/arena_allocator.hpp"

namespace duckdb {

class ClientContext;
class ExecutionContext;
class RadixPartitionedHashTable;
class RadixHTGlobalSinkState;
class RadixHTLocalSinkState;

struct FastIntRawChunk {
	AllocatedData data;
	idx_t rows = 0;
	idx_t capacity = 0;
};

using FastIntPartition = vector<FastIntRawChunk>;

struct FastIntSetup {
public:
	static constexpr idx_t RADIX_BITS = 10;
	static constexpr idx_t CACHE_BITS = 10;
	static constexpr idx_t MINIMUM_INPUT_ROWS = 4ULL << 20;
	static constexpr idx_t MINIMUM_GROUPS = 1ULL << 17;
	static constexpr idx_t MINIMUM_CHUNK_BYTES = 4096;
	static constexpr idx_t MAXIMUM_CHUNK_BYTES = 65536;
	static constexpr idx_t WINDOW_ROWS = 1ULL << 15;
	static constexpr idx_t MODE_DURATION_ROWS = 1ULL << 20;
	static constexpr idx_t MODE_CACHED = 0;
	static constexpr idx_t MODE_RUNS_ONLY = 1;
	static constexpr idx_t MODE_BYPASS = 2;

	idx_t key_width = 0;
	bool count_star = false;
	TupleDataLayout state_layout;
	vector<LogicalType> scan_types;

	static constexpr idx_t NullPartition() {
		return 1ULL << RADIX_BITS;
	}
	static constexpr idx_t TotalPartitions() {
		return (1ULL << RADIX_BITS) + 1;
	}
};

class FastIntGlobalState {
public:
	explicit FastIntGlobalState(unique_ptr<FastIntSetup> setup_p);

	const FastIntSetup &Setup() const {
		return *setup;
	}

public:
	unique_ptr<FastIntSetup> setup;
	atomic<bool> fallback;
	atomic<idx_t> total_bytes;
	vector<FastIntPartition> partitions;
	vector<FastIntPartition> counted_partitions;
	idx_t null_rows;
	idx_t total_rows;
	vector<idx_t> scan_partitions;
	atomic<idx_t> next_scan_partition;
	atomic<idx_t> scanned_partitions;
};

class FastIntRowSink {
public:
	FastIntRowSink(idx_t row_words, idx_t partition_count);

	uint64_t *NewChunk(Allocator &allocator, atomic<idx_t> &byte_counter, idx_t partition);
	void FinishChunks();

public:
	const idx_t row_words;
	vector<FastIntPartition> partitions;
	vector<uint64_t *> write_pos;
	vector<uint64_t *> write_end;
};

class FastIntLocalSink {
public:
	FastIntLocalSink(Allocator &allocator, const FastIntSetup &setup);

public:
	FastIntRowSink rows;
	unique_ptr<FastIntRowSink> counted_rows;
	idx_t input_rows;
	idx_t null_rows;
	AllocatedData cache_data;
	uint64_t *cache;
	idx_t mode;
	idx_t mode_remaining;
	idx_t window_rows;
	idx_t window_runs;
	idx_t window_misses;
};

class FastIntTable;

class FastIntLocalSource {
public:
	FastIntLocalSource();
	~FastIntLocalSource();

public:
	bool initialized = false;
	bool null_partition = false;
	idx_t position = 0;
	unique_ptr<FastIntTable> table;
	TupleDataLayout layout;
	unique_ptr<ArenaAllocator> allocator;
	unique_ptr<RowOperationsState> row_state;
	DataChunk scan_chunk;
	unique_ptr<Vector> addresses;
};

class FastIntAggregate {
public:
	static unique_ptr<FastIntGlobalState> TryCreate(const RadixPartitionedHashTable &radix_ht, idx_t estimated_rows,
	                                                idx_t estimated_groups);

	static bool Sink(const RadixPartitionedHashTable &radix_ht, ExecutionContext &context,
	                 RadixHTGlobalSinkState &gstate, RadixHTLocalSinkState &lstate, DataChunk &group_chunk,
	                 const unsafe_vector<idx_t> &filter);
	static bool Combine(const RadixPartitionedHashTable &radix_ht, ExecutionContext &context,
	                    RadixHTGlobalSinkState &gstate, RadixHTLocalSinkState &lstate);
	static void Finalize(ClientContext &context, RadixHTGlobalSinkState &gstate);

	static bool Active(const RadixHTGlobalSinkState &gstate);
	static idx_t MaxThreads(const RadixHTGlobalSinkState &gstate);
	static void ResetGlobal(RadixHTGlobalSinkState &gstate);
	static bool Scan(const RadixPartitionedHashTable &radix_ht, ExecutionContext &context,
	                 RadixHTGlobalSinkState &gstate, FastIntLocalSource &lsource, DataChunk &chunk);
};

} // namespace duckdb
