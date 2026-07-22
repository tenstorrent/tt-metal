// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <any>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/reflection.hpp>
#include <tuple>
#include <variant>
#include "ttnn/graph/capture_program_config.hpp"
#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/experimental/mock_device/mock_allocator.hpp>
#include <ttnn/distributed/tensor_topology.hpp>

namespace ttnn::graph {

// Pairs a tt::tt_metal::TensorSpec with a tt::tt_metal::TensorTopology, allowing callers to specify
// distribution (shard/replicate) when creating tensors in query_op_constraints.
struct DistributedTensorSpec {
    tt::tt_metal::TensorSpec tensor_spec;
    tt::tt_metal::TensorTopology tensor_topology;
};

namespace detail {
// Helper to temporarily change logger level
class LogLevelGuard {
public:
    explicit LogLevelGuard(spdlog::level::level_enum new_level) :
        saved_level_(tt::LoggerRegistry::instance().get(tt::LogOp)->level()) {
        tt::LoggerRegistry::instance().set_level(new_level);
    }
    ~LogLevelGuard() { tt::LoggerRegistry::instance().set_level(saved_level_); }
    LogLevelGuard(const LogLevelGuard&) = delete;
    LogLevelGuard& operator=(const LogLevelGuard&) = delete;

private:
    spdlog::level::level_enum saved_level_;
};

// Helper to invoke an operation with arguments, allowing implicit conversions.
// This differs from std::apply by not using perfect forwarding, which enables
// implicit conversions (e.g., T -> std::optional<T>).
template <typename Op, typename Tuple, std::size_t... Is>
auto invoke_op_impl(Op&& op, Tuple& args, std::index_sequence<Is...>) {
    return std::invoke(std::forward<Op>(op), std::get<Is>(args)...);
}

template <typename Op, typename Tuple>
auto invoke_op(Op&& op, Tuple& args) {
    return invoke_op_impl(
        std::forward<Op>(op), args, std::make_index_sequence<std::tuple_size_v<std::decay_t<Tuple>>>{});
}

// These overloaded extract_output_tensor functions abstract the return type of an arbitrary op from the rest of the
// constraints query function. An overload resolution failure means the return type for the op in that query is not yet
// supported and a new overload should be added

// Generic function to extract all tensors from most return types using reflection. The return type could be:
// - A single Tensor (Most ops)
// - std::vector<Tensor> or std::tuple<Tensor, Tensor, ...> (multi-output ops, e.g., sort,
// split_query_key_value_and_split_heads)
template <typename T>
inline std::vector<Tensor> extract_output_tensors(const T& result) {
    std::vector<Tensor> tensors;
    ttsl::reflection::visit_object_of_type<Tensor>([&tensors](auto&& tensor) { tensors.push_back(tensor); }, result);
    return tensors;
}

// Specialized overload for conv2d output (std::variant with different tuple combinations)
template <typename... Ts>
inline std::vector<Tensor> extract_output_tensors(const std::variant<Ts...>& result) {
    std::vector<Tensor> tensors;
    std::visit(
        [&tensors](auto&& value) {
            ttsl::reflection::visit_object_of_type<Tensor>(
                [&tensors](auto&& tensor) { tensors.push_back(tensor); }, value);
        },
        result);
    return tensors;
}

// Transform a query argument into the value passed to the op: tt::tt_metal::TensorSpec/DistributedTensorSpec
// (and their optional/vector forms) become device tensors via create_device_tensor; a MeshDevice
// is wrapped in a reference_wrapper; everything else is forwarded unchanged.
template <typename Arg>
auto materialize_arg(tt::tt_metal::distributed::MeshDevice* device, Arg&& arg) {
    if constexpr (std::is_same_v<std::decay_t<Arg>, DistributedTensorSpec>) {
        return ttnn::create_device_tensor(arg.tensor_spec, device, arg.tensor_topology);
    } else if constexpr (std::is_same_v<std::decay_t<Arg>, std::optional<DistributedTensorSpec>>) {
        return arg ? std::optional<Tensor>(ttnn::create_device_tensor(arg->tensor_spec, device, arg->tensor_topology))
                   : std::nullopt;
    } else if constexpr (std::is_same_v<std::decay_t<Arg>, std::vector<DistributedTensorSpec>>) {
        std::vector<Tensor> result(arg.size());
        std::transform(arg.begin(), arg.end(), result.begin(), [device](auto&& item) {
            return ttnn::create_device_tensor(item.tensor_spec, device, item.tensor_topology);
        });
        return result;
    } else if constexpr (std::is_same_v<std::decay_t<Arg>, tt::tt_metal::TensorSpec>) {
        return ttnn::create_device_tensor(arg, device);
    } else if constexpr (std::is_same_v<std::decay_t<Arg>, std::optional<tt::tt_metal::TensorSpec>>) {
        return arg ? std::optional<Tensor>(ttnn::create_device_tensor(*arg, device)) : std::nullopt;
    } else if constexpr (std::is_same_v<std::decay_t<Arg>, std::vector<tt::tt_metal::TensorSpec>>) {
        std::vector<Tensor> result(arg.size());
        std::transform(arg.begin(), arg.end(), result.begin(), [device](auto&& item) {
            return ttnn::create_device_tensor(item, device);
        });
        return result;
    } else if constexpr (std::is_same_v<std::decay_t<Arg>, tt::tt_metal::distributed::MeshDevice>) {
        // MeshDevice is non-copyable; wrap in reference_wrapper so make_tuple can store it.
        // reference_wrapper<MeshDevice> implicitly converts to MeshDevice& at the call site.
        return std::ref(arg);
    } else {
        return std::forward<Arg>(arg);
    }
}

}  // namespace detail

struct ResourceUsage {
    size_t cb_peak_size_per_core = 0;
    size_t l1_buffers_peak_per_core = 0;
    size_t peak_memory_usage_per_core = 0;
    size_t l1_output_buffer_per_core = 0;
};

struct ConstraintQueryResponse {
    ExecutionStatus status = ExecutionStatus::Error;
    ResourceUsage resource_usage;
    std::optional<std::vector<tt::tt_metal::TensorSpec>> output_tensor_specs;
    std::optional<std::string> error_message;
};

// Result of the pure, state-threading query. `new_state` is the allocator state
// after the op's outputs are allocated on top of the caller-supplied initial state.
// `output_allocations` records each output buffer's placement (one per output). The caller keeps
// the records of still-live tensors and rebuilds the state via MockAllocatorState::with_allocations;
// "evicting" a tensor is just dropping its record. Empty on the stateless path (outputs not allocated).
struct QueryOutput {
    ConstraintQueryResponse response;
    tt::tt_metal::experimental::MockAllocatorState new_state;
    std::vector<tt::tt_metal::experimental::AllocationRecord> output_allocations;
    // The program config ttnn auto-selected for the queried op, owned here and holding the op's
    // concrete config type (e.g. MatmulProgramConfig) for the caller to any_cast. nullopt when no
    // registered extractor matched. See capture_program_config.hpp.
    std::optional<std::any> captured_config;
};

namespace detail {

// Build a success response from the captured op trace and its output tensors.
inline ConstraintQueryResponse build_success_response(
    const nlohmann::json& op_trace, const std::vector<Tensor>& outputs) {
    const auto& [cb_peak_size_per_core, l1_buffers_peak_per_core, peak_memory_usage_per_core] =
        extract_resource_usage_per_core(op_trace);

    size_t l1_output_buffer_per_core = 0;
    for (const auto& output : outputs) {
        if (!output.buffer()->is_dram()) {
            l1_output_buffer_per_core += extract_l1_output_buffer_allocation_size_per_core(output);
        }
    }

    std::vector<tt::tt_metal::TensorSpec> output_specs;
    output_specs.reserve(outputs.size());
    std::transform(outputs.begin(), outputs.end(), std::back_inserter(output_specs), [](const Tensor& t) {
        return t.tensor_spec();
    });

    return ConstraintQueryResponse{
        ExecutionStatus::Success,
        {cb_peak_size_per_core, l1_buffers_peak_per_core, peak_memory_usage_per_core, l1_output_buffer_per_core},
        std::make_optional(std::move(output_specs))};
}

inline ConstraintQueryResponse error_response(const std::string& message) {
    return ConstraintQueryResponse{
        ExecutionStatus::Error,
        {.cb_peak_size_per_core = 0,
         .l1_buffers_peak_per_core = 0,
         .peak_memory_usage_per_core = 0,
         .l1_output_buffer_per_core = 0},
        /* output_tensor_specs= */ std::nullopt,
        message};
}

}  // namespace detail

/**
 * @brief Captures the graph operations and extracts resource usage constraints.
 *
 * This function runs graph capture by invoking the provided operation with the given arguments,
 * then extracts and returns the resource usage constraints from the captured trace.
 *
 * @tparam Op The type of the operation that will be invoked to capture the graph operations.
 * @tparam Args The types of the arguments that will be passed to the operation.
 * @param op The operation that will be invoked to capture the graph operations.
 * @param device A pointer to the Device object, which provides information about the compute grid size.
 * @param args The arguments that will be passed to the operation.
 * @return ConstraintQueryResponse containing the execution status and resource usage constraints.
 *         - On success: ExecutionStatus::Success and the resource usage details.
 *         - On failure: ExecutionStatus::Error, zeroed resource usage, and an error message.
 */
template <typename Op, typename... Args>
auto query_op_constraints(Op op, tt::tt_metal::distributed::MeshDevice* device, Args&&... args) {
    detail::LogLevelGuard log_guard(spdlog::level::level_enum::off);
    nlohmann::json op_trace;
    std::vector<Tensor> outputs;
    // outer graph capture is to avoid dispatching/allocating dummy input tensors
    {
        auto capture_outer = ScopedGraphCapture(GraphProcessor::RunMode::NO_DISPATCH);
        auto transformed_args = std::make_tuple(detail::materialize_arg(device, std::forward<Args>(args))...);

        // inner graph capture is to capture the actual op graph trace
        try {
            auto capture_inner = ScopedGraphCapture(GraphProcessor::RunMode::NO_DISPATCH);
            outputs = detail::extract_output_tensors(detail::invoke_op(op, transformed_args));
        }  // end of inner graph capture
        catch (const std::exception& e) {
            log_debug(tt::LogOp, "Error during graph capture: {}", e.what());
            return detail::error_response(e.what());
        }
        op_trace = capture_outer.end_graph_capture();
    }  // end of outer graph capture

    return detail::build_success_response(op_trace, outputs);
}

/**
 * @brief Pure, state-threading variant of query_op_constraints.
 *
 * The caller supplies an opaque allocator state (`initial_state`); `QueryOutput::new_state` is the
 * allocator state after the op runs on top of it. Nothing is retained between calls — advancing is
 * `state = result.new_state`.
 *
 * This is the always-stateful core. Callers that may or may not have a state to thread (e.g. a
 * single op-model entry serving both stateless and stateful queries) should call
 * `query_op_constraints_with_optional_state`, which dispatches on the optional.
 *
 * Mechanism (two *sequential*, non-nested graph captures):
 *   - Phase 1 (NO_DISPATCH): materialize the input tensors as weightless handles (address=0,
 *     allocator untouched). Their real footprint is already encoded in `initial_state`, so they must
 *     not be allocated again here — doing so would double-count and risk a false OOM.
 *   - Phase 2 (NORMAL, outermost capture): run the op for real. A fresh capture installs the global
 *     allocation hook with block=false, so the op's outputs (and transiently its intermediates) are
 *     allocated through the MockAllocator on top of `initial_state`, reproducing real placement and
 *     fragmentation. `new_state` is snapshotted while the outputs are still alive.
 *
 * The captures must be sequential, not nested: the allocation block flag is set once by the
 * outermost capture, so a NORMAL capture nested inside a NO_DISPATCH one would stay blocked.
 *
 * If the op cannot fit (allocation throws), the result is `ExecutionStatus::Error`.
 *
 * Requires `device` to be a mock/planning device whose allocator is a MockAllocator. `move` and
 * `reallocate` are unsupported as direct query targets (they branch on input buffer addresses,
 * which are 0 for the weightless inputs).
 *
 * @tparam Op The type of the operation that will be invoked to capture the graph operations.
 * @tparam Args The types of the arguments that will be passed to the operation.
 * @param op The operation that will be invoked to capture the graph operations.
 * @param device A pointer to a mock MeshDevice used for planning.
 * @param initial_state The caller-owned allocator state to evaluate the op against.
 * @param args The arguments that will be passed to the operation.
 * @return QueryOutput { response, new_state, output_allocations, captured_config }.
 */
template <typename Op, typename... Args>
QueryOutput query_op_constraints_with_initial_state(
    Op op,
    tt::tt_metal::distributed::MeshDevice* device,
    const tt::tt_metal::experimental::MockAllocatorState& initial_state,
    Args&&... args) {
    detail::LogLevelGuard log_guard(spdlog::level::level_enum::off);

    // A stateful query requires a mock device whose allocator supports checkpoint/restore.
    // override_mock_allocator_state TT_FATALs on a non-mock device; since this is a public API,
    // surface the misconfiguration as an Error rather than aborting the process.
    if (tt::tt_metal::experimental::get_mock_allocator(*device) == nullptr) {
        return QueryOutput{
            detail::error_response("query_op_constraints_with_initial_state requires a mock device"),
            tt::tt_metal::experimental::MockAllocatorState{}};
    }

    // Precondition: the caller must run with the device's program cache disabled. Phase 2 enqueues a
    // real MeshWorkload; a cached workload outlives the sub-devices it references and crashes at
    // device teardown (tenstorrent/tt-metal#45646).
    tt::tt_metal::experimental::override_mock_allocator_state(*device, initial_state);

    // Install the passive config-capture listener below the phase captures. Popped via RAII; the
    // local shared_ptr keeps it alive until this function returns, so take_result() below is valid
    // regardless of pop order.
    auto capture_listener = std::make_shared<ProgramConfigCaptureProcessor>(program_config_extractors());
    tt::tt_metal::GraphTracker::instance().push_processor(capture_listener);
    struct PopGuard {
        ~PopGuard() { tt::tt_metal::GraphTracker::instance().pop_processor(); }
    } pop_guard;

    // Phase 1: materialize inputs as weightless handles (address=0, allocator untouched).
    auto transformed_args = [&] {
        auto phase1 = ScopedGraphCapture(GraphProcessor::RunMode::NO_DISPATCH);
        return std::make_tuple(detail::materialize_arg(device, std::forward<Args>(args))...);
    }();  // phase1 capture ends here -> global hook removed

    // Phase 2: run the op under the outermost NORMAL capture so outputs are allocated for real.
    auto phase2 = ScopedGraphCapture(GraphProcessor::RunMode::NORMAL);
    try {
        auto outputs = detail::extract_output_tensors(detail::invoke_op(op, transformed_args));
        auto op_trace = phase2.end_graph_capture();
        // Snapshot + record output placements while the output buffers are alive (after they are
        // allocated, before they drop). Each record locates an output in new_state for later eviction.
        auto new_state = tt::tt_metal::experimental::extract_mock_allocator_state(*device);
        std::vector<tt::tt_metal::experimental::AllocationRecord> output_allocations;
        output_allocations.reserve(outputs.size());
        for (const auto& output : outputs) {
            const auto& buffer = output.buffer();
            output_allocations.push_back({buffer->buffer_type(), buffer->address(), buffer->aligned_size_per_bank()});
        }
        return QueryOutput{
            detail::build_success_response(op_trace, outputs),
            std::move(new_state),
            std::move(output_allocations),
            capture_listener->take_result()};
    } catch (const std::exception& e) {
        log_debug(tt::LogOp, "Error during stateful graph capture: {}", e.what());
        return QueryOutput{
            detail::error_response(e.what()),
            tt::tt_metal::experimental::extract_mock_allocator_state(*device),
            {},
            capture_listener->take_result()};
    }
}

/**
 * @brief Optional-state dispatcher giving callers a single call shape: std::nullopt runs the
 * stateless `query_op_constraints` (empty `new_state`); a value delegates to
 * `query_op_constraints_with_initial_state`.
 */
template <typename Op, typename... Args>
QueryOutput query_op_constraints_with_optional_state(
    Op op,
    tt::tt_metal::distributed::MeshDevice* device,
    const std::optional<tt::tt_metal::experimental::MockAllocatorState>& initial_state,
    Args&&... args) {
    if (initial_state.has_value()) {
        return query_op_constraints_with_initial_state(op, device, *initial_state, std::forward<Args>(args)...);
    }
    // query_op_constraints manages its own log level, so no guard is needed here.
    return QueryOutput{
        query_op_constraints(op, device, std::forward<Args>(args)...),
        tt::tt_metal::experimental::MockAllocatorState{}};
}

}  // namespace ttnn::graph
