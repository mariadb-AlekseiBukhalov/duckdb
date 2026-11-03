#include "duckdb/execution/operator/aggregate/physical_limited_distinct.hpp"

#include "duckdb/execution/aggregate_hashtable.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/execution/operator/aggregate/aggregate_object.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/filter/table_filter_functions.hpp"

#include <queue>

namespace duckdb {

static vector<LogicalType> GetPayloadTypes(const vector<unique_ptr<Expression>> &aggregates) {
	vector<LogicalType> payload_types;
	for (auto &aggr_expr : aggregates) {
		auto &aggr = aggr_expr->Cast<BoundAggregateExpression>();
		for (auto &child : aggr.GetChildren()) {
			payload_types.push_back(child->GetReturnType());
		}
	}
	return payload_types;
}

static vector<AggregateObject> GetAggregateObjects(const vector<unique_ptr<Expression>> &aggregates) {
	vector<BoundAggregateExpression *> aggregate_ptrs;
	for (auto &aggr : aggregates) {
		aggregate_ptrs.push_back(&aggr->Cast<BoundAggregateExpression>());
	}
	return AggregateObject::CreateAggregateObjects(aggregate_ptrs);
}

PhysicalLimitedDistinct::PhysicalLimitedDistinct(PhysicalPlan &physical_plan, vector<LogicalType> types,
                                                 vector<unique_ptr<Expression>> groups_p,
                                                 vector<unique_ptr<Expression>> aggregates_p, idx_t limit_p,
                                                 idx_t estimated_cardinality)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::LIMITED_DISTINCT, std::move(types), estimated_cardinality),
      groups(std::move(groups_p)), aggregates(std::move(aggregates_p)), limit(limit_p) {
	for (auto &group : groups) {
		group_types.push_back(group->GetReturnType());
	}
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
struct LimitedDistinctGlobalSinkState : public GlobalSinkState {
	LimitedDistinctGlobalSinkState(const PhysicalLimitedDistinct &op, ClientContext &context) : limit_reached(false) {
		auto aggregate_objects = GetAggregateObjects(op.aggregates);
		auto payload_types = GetPayloadTypes(op.aggregates);
		ht = make_uniq<GroupedAggregateHashTable>(context, BufferAllocator::Get(context), op.group_types, payload_types,
		                                          aggregate_objects);
		if (op.threshold_filter) {
			op.threshold_filter->Reset();
		} else if (!op.aggregates.empty()) {
			keys = make_uniq<GroupedAggregateHashTable>(context, BufferAllocator::Get(context), op.group_types);
			keys_frozen = op.limit == 0;
		}
	}

	unique_ptr<GroupedAggregateHashTable> ht;
	atomic<bool> limit_reached;
	mutex combine_lock;

	unique_ptr<GroupedAggregateHashTable> keys;
	mutex keys_lock;
	atomic<bool> keys_frozen {false};

	atomic<bool> has_threshold {false};
	atomic<int64_t> threshold {NumericLimits<int64_t>::Maximum()};
	mutex threshold_lock;
};

struct LimitedDistinctLocalSinkState : public LocalSinkState {
	LimitedDistinctLocalSinkState(const PhysicalLimitedDistinct &op, ExecutionContext &context)
	    : group_executor(context.client, op.groups), payload_executor(context.client) {
		auto &allocator = BufferAllocator::Get(context.client);

		auto aggregate_objects = GetAggregateObjects(op.aggregates);
		auto payload_types = GetPayloadTypes(op.aggregates);

		ht = make_uniq<GroupedAggregateHashTable>(context.client, allocator, op.group_types, payload_types,
		                                          aggregate_objects);

		group_chunk.Initialize(allocator, op.group_types);

		// Build payload executor for aggregate children
		for (auto &aggr_expr : op.aggregates) {
			auto &aggr = aggr_expr->Cast<BoundAggregateExpression>();
			for (auto &child : aggr.GetChildren()) {
				payload_executor.AddExpression(*child);
			}
		}

		if (!payload_types.empty()) {
			payload_chunk.Initialize(allocator, payload_types);
		}
	}

	unique_ptr<GroupedAggregateHashTable> ht;
	ExpressionExecutor group_executor;
	ExpressionExecutor payload_executor;
	DataChunk group_chunk;
	DataChunk payload_chunk;
	AggregateHTLookupState lookup_state;
	SelectionVector found_sel {STANDARD_VECTOR_SIZE};
	SelectionVector new_sel {STANDARD_VECTOR_SIZE};
	Vector new_addresses {LogicalType::POINTER};
	Vector key_addresses {LogicalType::POINTER};
	std::priority_queue<int64_t> smallest_groups;
};

unique_ptr<GlobalSinkState> PhysicalLimitedDistinct::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<LimitedDistinctGlobalSinkState>(*this, context);
}

unique_ptr<LocalSinkState> PhysicalLimitedDistinct::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<LimitedDistinctLocalSinkState>(*this, context);
}

template <class T>
static idx_t SelectAtMost(Vector &vector, idx_t count, int64_t bound, SelectionVector &sel) {
	UnifiedVectorFormat format;
	vector.ToUnifiedFormat(format);
	const auto data = UnifiedVectorFormat::GetData<T>(format);
	idx_t selected = 0;
	for (idx_t i = 0; i < count; i++) {
		const auto index = format.sel->get_index(i);
		if (format.validity.RowIsValid(index) && static_cast<int64_t>(data[index]) <= bound) {
			sel.set_index(selected++, i);
		}
	}
	return selected;
}

static idx_t SelectAtMost(Vector &vector, idx_t count, int64_t bound, SelectionVector &sel) {
	switch (vector.GetType().InternalType()) {
	case PhysicalType::INT8:
		return SelectAtMost<int8_t>(vector, count, bound, sel);
	case PhysicalType::INT16:
		return SelectAtMost<int16_t>(vector, count, bound, sel);
	case PhysicalType::INT32:
		return SelectAtMost<int32_t>(vector, count, bound, sel);
	case PhysicalType::INT64:
		return SelectAtMost<int64_t>(vector, count, bound, sel);
	case PhysicalType::UINT8:
		return SelectAtMost<uint8_t>(vector, count, bound, sel);
	case PhysicalType::UINT16:
		return SelectAtMost<uint16_t>(vector, count, bound, sel);
	case PhysicalType::UINT32:
		return SelectAtMost<uint32_t>(vector, count, bound, sel);
	default:
		throw InternalException("Unsupported type for the limited aggregate threshold");
	}
}

template <class T>
static void OfferGroupValues(Vector &vector, idx_t count, const SelectionVector &rows, idx_t row_count, idx_t limit,
                             std::priority_queue<int64_t> &heap) {
	UnifiedVectorFormat format;
	vector.ToUnifiedFormat(format);
	const auto data = UnifiedVectorFormat::GetData<T>(format);
	for (idx_t i = 0; i < row_count; i++) {
		const auto index = format.sel->get_index(rows.get_index(i));
		if (!format.validity.RowIsValid(index)) {
			continue;
		}
		const auto value = static_cast<int64_t>(data[index]);
		if (heap.size() < limit) {
			heap.push(value);
		} else if (value < heap.top()) {
			heap.pop();
			heap.push(value);
		}
	}
}

static void OfferGroupValues(Vector &vector, idx_t count, const SelectionVector &rows, idx_t row_count, idx_t limit,
                             std::priority_queue<int64_t> &heap) {
	switch (vector.GetType().InternalType()) {
	case PhysicalType::INT8:
		return OfferGroupValues<int8_t>(vector, count, rows, row_count, limit, heap);
	case PhysicalType::INT16:
		return OfferGroupValues<int16_t>(vector, count, rows, row_count, limit, heap);
	case PhysicalType::INT32:
		return OfferGroupValues<int32_t>(vector, count, rows, row_count, limit, heap);
	case PhysicalType::INT64:
		return OfferGroupValues<int64_t>(vector, count, rows, row_count, limit, heap);
	case PhysicalType::UINT8:
		return OfferGroupValues<uint8_t>(vector, count, rows, row_count, limit, heap);
	case PhysicalType::UINT16:
		return OfferGroupValues<uint16_t>(vector, count, rows, row_count, limit, heap);
	case PhysicalType::UINT32:
		return OfferGroupValues<uint32_t>(vector, count, rows, row_count, limit, heap);
	default:
		throw InternalException("Unsupported type for the limited aggregate threshold");
	}
}

SinkResultType PhysicalLimitedDistinct::SinkThresholdAggregates(DataChunk &chunk,
                                                                LimitedDistinctGlobalSinkState &gstate,
                                                                LimitedDistinctLocalSinkState &lstate) const {
	const auto count = chunk.size();
	lstate.group_chunk.Reset();
	lstate.group_executor.Execute(chunk, lstate.group_chunk);
	const bool has_payload = lstate.payload_chunk.ColumnCount() > 0;
	if (has_payload) {
		lstate.payload_chunk.Reset();
		lstate.payload_executor.Execute(chunk, lstate.payload_chunk);
	} else {
		lstate.payload_chunk.SetChildCardinality(count);
	}

	if (gstate.has_threshold.load(std::memory_order_acquire)) {
		const auto bound = gstate.threshold.load(std::memory_order_relaxed);
		const auto selected = SelectAtMost(lstate.group_chunk.data[0], count, bound, lstate.found_sel);
		if (selected == 0) {
			return SinkResultType::NEED_MORE_INPUT;
		}
		if (selected != count) {
			lstate.group_chunk.Slice(lstate.found_sel, selected);
			lstate.payload_chunk.Slice(lstate.found_sel, selected);
		}
	}

	const auto new_groups = lstate.ht->AddChunkAndGetNewGroups(
	    lstate.group_chunk, lstate.payload_chunk, AggregateType::NON_DISTINCT, lstate.new_addresses, lstate.new_sel);
	if (new_groups == 0) {
		return SinkResultType::NEED_MORE_INPUT;
	}
	OfferGroupValues(lstate.group_chunk.data[0], lstate.group_chunk.size(), lstate.new_sel, new_groups, limit,
	                 lstate.smallest_groups);
	if (lstate.smallest_groups.size() < limit) {
		return SinkResultType::NEED_MORE_INPUT;
	}
	const auto local_bound = lstate.smallest_groups.top();
	if (gstate.has_threshold.load(std::memory_order_acquire) &&
	    local_bound >= gstate.threshold.load(std::memory_order_relaxed)) {
		return SinkResultType::NEED_MORE_INPUT;
	}
	lock_guard<mutex> guard(gstate.threshold_lock);
	if (gstate.has_threshold.load(std::memory_order_relaxed) &&
	    local_bound >= gstate.threshold.load(std::memory_order_relaxed)) {
		return SinkResultType::NEED_MORE_INPUT;
	}
	gstate.threshold.store(local_bound, std::memory_order_relaxed);
	gstate.has_threshold.store(true, std::memory_order_release);
	threshold_filter->SetValue(Value::Numeric(group_types[0], local_bound));
	return SinkResultType::NEED_MORE_INPUT;
}

SinkResultType PhysicalLimitedDistinct::SinkExactAggregates(DataChunk &chunk, LimitedDistinctGlobalSinkState &gstate,
                                                            LimitedDistinctLocalSinkState &lstate) const {
	const auto count = chunk.size();
	lstate.group_chunk.Reset();
	lstate.group_executor.Execute(chunk, lstate.group_chunk);
	const bool has_payload = lstate.payload_chunk.ColumnCount() > 0;
	if (has_payload) {
		lstate.payload_chunk.Reset();
		lstate.payload_executor.Execute(chunk, lstate.payload_chunk);
	} else {
		lstate.payload_chunk.SetChildCardinality(count);
	}

	if (gstate.keys_frozen.load(std::memory_order_acquire)) {
		const auto found = gstate.keys->LookupGroups(lstate.group_chunk, lstate.lookup_state, lstate.found_sel);
		if (found == 0) {
			return SinkResultType::NEED_MORE_INPUT;
		}
		if (found != count) {
			lstate.group_chunk.Slice(lstate.found_sel, found);
			lstate.payload_chunk.Slice(lstate.found_sel, found);
		}
		lstate.ht->AddChunk(lstate.group_chunk, lstate.payload_chunk, AggregateType::NON_DISTINCT);
		return SinkResultType::NEED_MORE_INPUT;
	}

	const auto new_groups = lstate.ht->AddChunkAndGetNewGroups(
	    lstate.group_chunk, lstate.payload_chunk, AggregateType::NON_DISTINCT, lstate.new_addresses, lstate.new_sel);
	if (new_groups == 0) {
		return SinkResultType::NEED_MORE_INPUT;
	}
	lock_guard<mutex> guard(gstate.keys_lock);
	if (gstate.keys_frozen.load(std::memory_order_relaxed)) {
		return SinkResultType::NEED_MORE_INPUT;
	}
	lstate.group_chunk.Slice(lstate.new_sel, new_groups);
	gstate.keys->FindOrCreateGroups(lstate.group_chunk, lstate.key_addresses);
	if (gstate.keys->Count() >= limit) {
		gstate.keys_frozen.store(true, std::memory_order_release);
	}
	return SinkResultType::NEED_MORE_INPUT;
}

SinkResultType PhysicalLimitedDistinct::Sink(ExecutionContext &context, DataChunk &chunk,
                                             OperatorSinkInput &input) const {
	auto &gstate = input.global_state.Cast<LimitedDistinctGlobalSinkState>();
	auto &lstate = input.local_state.Cast<LimitedDistinctLocalSinkState>();

	if (limit == 0) {
		return SinkResultType::FINISHED;
	}

	if (threshold_filter) {
		return SinkThresholdAggregates(chunk, gstate, lstate);
	}
	if (!aggregates.empty()) {
		return SinkExactAggregates(chunk, gstate, lstate);
	}

	if (gstate.limit_reached) {
		return SinkResultType::FINISHED;
	}

	// Execute group expressions
	lstate.group_chunk.Reset();
	lstate.group_executor.Execute(chunk, lstate.group_chunk);

	// Execute payload expressions (aggregate children)
	if (lstate.payload_chunk.ColumnCount() > 0) {
		lstate.payload_chunk.Reset();
		lstate.payload_executor.Execute(chunk, lstate.payload_chunk);
		unsafe_vector<idx_t> filter;
		lstate.ht->AddChunk(lstate.group_chunk, lstate.payload_chunk, filter);
	} else {
		// No aggregates - just track groups
		DataChunk empty_payload;
		unsafe_vector<idx_t> filter;
		lstate.ht->AddChunk(lstate.group_chunk, empty_payload, filter);
	}

	// Check if we have enough groups for early termination
	if (limit > 0 && lstate.ht->Count() >= limit) {
		gstate.limit_reached = true;
		return SinkResultType::FINISHED;
	}

	return SinkResultType::NEED_MORE_INPUT;
}

SinkCombineResultType PhysicalLimitedDistinct::Combine(ExecutionContext &context,
                                                       OperatorSinkCombineInput &input) const {
	auto &gstate = input.global_state.Cast<LimitedDistinctGlobalSinkState>();
	auto &lstate = input.local_state.Cast<LimitedDistinctLocalSinkState>();

	if (lstate.ht->Count() > 0) {
		lock_guard<mutex> guard(gstate.combine_lock);
		gstate.ht->Combine(*lstate.ht);
	}
	return SinkCombineResultType::FINISHED;
}

SinkFinalizeType PhysicalLimitedDistinct::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                   OperatorSinkFinalizeInput &input) const {
	return SinkFinalizeType::READY;
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
struct LimitedDistinctGlobalSourceState : public GlobalSourceState {
	LimitedDistinctGlobalSourceState(const PhysicalLimitedDistinct &op, ClientContext &context)
	    : initialized(false), total_rows(op.sink_state->Cast<LimitedDistinctGlobalSinkState>().ht->Count()),
	      rows_scanned(0) {
		group_chunk.Initialize(Allocator::Get(context), op.group_types);

		vector<LogicalType> payload_types;
		for (auto &aggr_expr : op.aggregates) {
			auto &aggr = aggr_expr->Cast<BoundAggregateExpression>();
			payload_types.push_back(aggr.GetReturnType());
		}
		if (!payload_types.empty()) {
			payload_chunk.Initialize(Allocator::Get(context), payload_types);
		}
	}

	AggregateHTScanState scan_state;
	AggregateHTLookupState lookup_state;
	SelectionVector found_sel {STANDARD_VECTOR_SIZE};
	bool initialized;
	const idx_t total_rows;
	atomic<idx_t> rows_scanned;
	DataChunk group_chunk;
	DataChunk payload_chunk;
};

unique_ptr<GlobalSourceState> PhysicalLimitedDistinct::GetGlobalSourceState(ClientContext &context) const {
	return make_uniq<LimitedDistinctGlobalSourceState>(*this, context);
}

ProgressData PhysicalLimitedDistinct::GetProgress(ClientContext &context, GlobalSourceState &gstate) const {
	auto &state = gstate.Cast<LimitedDistinctGlobalSourceState>();
	ProgressData progress;
	progress.total = double(MaxValue<idx_t>(state.total_rows, 1));
	progress.done = state.total_rows == 0 ? 1.0 : double(state.rows_scanned.load(std::memory_order_relaxed));
	return progress;
}

void PhysicalLimitedDistinct::SourceFinished(ClientContext &context, GlobalSourceState &gstate) const {
	auto &state = gstate.Cast<LimitedDistinctGlobalSourceState>();
	state.rows_scanned.store(state.total_rows, std::memory_order_relaxed);
}

unique_ptr<LocalSourceState> PhysicalLimitedDistinct::GetLocalSourceState(ExecutionContext &context,
                                                                          GlobalSourceState &gstate) const {
	return make_uniq<LocalSourceState>();
}

SourceResultType PhysicalLimitedDistinct::GetDataInternal(ExecutionContext &context, DataChunk &chunk,
                                                          OperatorSourceInput &input) const {
	auto &gstate_sink = sink_state->Cast<LimitedDistinctGlobalSinkState>();
	auto &gstate_source = input.global_state.Cast<LimitedDistinctGlobalSourceState>();

	if (!gstate_source.initialized) {
		gstate_sink.ht->InitializeScan(gstate_source.scan_state);
		gstate_source.initialized = true;
	}

	while (true) {
		gstate_source.group_chunk.Reset();
		gstate_source.payload_chunk.Reset();

		if (!gstate_sink.ht->Scan(gstate_source.scan_state, gstate_source.group_chunk, gstate_source.payload_chunk)) {
			return SourceResultType::FINISHED;
		}
		// Aggregate hash table scan returns "has more" when advancing between partitions, even if the current
		// output chunk is empty. Keep scanning until we either see rows or exhaust the partitions.
		if (gstate_source.group_chunk.size() > 0 && threshold_filter &&
		    gstate_sink.has_threshold.load(std::memory_order_acquire)) {
			const auto before = gstate_source.group_chunk.size();
			const auto selected =
			    SelectAtMost(gstate_source.group_chunk.data[0], before,
			                 gstate_sink.threshold.load(std::memory_order_relaxed), gstate_source.found_sel);
			if (selected != before) {
				gstate_source.group_chunk.Slice(gstate_source.found_sel, selected);
				gstate_source.payload_chunk.Slice(gstate_source.found_sel, selected);
			}
		}
		if (gstate_source.group_chunk.size() > 0 && gstate_sink.keys) {
			const auto before = gstate_source.group_chunk.size();
			const auto found = gstate_sink.keys->LookupGroups(gstate_source.group_chunk, gstate_source.lookup_state,
			                                                  gstate_source.found_sel);
			if (found != before) {
				gstate_source.group_chunk.Slice(gstate_source.found_sel, found);
				gstate_source.payload_chunk.Slice(gstate_source.found_sel, found);
			}
		}
		if (gstate_source.group_chunk.size() > 0) {
			break;
		}
	}

	// Copy results into output chunk (groups + aggregate results)
	idx_t col = 0;
	for (idx_t i = 0; i < gstate_source.group_chunk.ColumnCount(); i++) {
		chunk.data[col].Reference(gstate_source.group_chunk.data[i]);
		col++;
	}
	for (idx_t i = 0; i < gstate_source.payload_chunk.ColumnCount(); i++) {
		chunk.data[col].Reference(gstate_source.payload_chunk.data[i]);
		col++;
	}

	gstate_source.rows_scanned.fetch_add(chunk.size(), std::memory_order_relaxed);
	return chunk.size() == 0 ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;
}

InsertionOrderPreservingMap<string> PhysicalLimitedDistinct::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	string groups_str;
	for (idx_t i = 0; i < groups.size(); i++) {
		if (i > 0) {
			groups_str += "\n";
		}
		groups_str += groups[i]->GetName();
	}
	result["Groups"] = groups_str;
	result["Limit"] = to_string(limit);
	return result;
}

} // namespace duckdb
