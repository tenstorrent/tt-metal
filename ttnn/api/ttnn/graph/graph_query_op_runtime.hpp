// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>

#include "ttnn/graph/graph_trace_utils.hpp"
#include "ttnn/operations/trace.hpp"

namespace ttnn::graph {

struct RuntimeQueryResponse {
    ExecutionStatus status = ExecutionStatus::Error;
    uint64_t runtime = 0;
    std::optional<std::string> error_message;
};

static constexpr int NUM_TRACE_EXECUTIONS = 10;

/**
 * @brief Extracts a trace of the operation(s) and returns the trace ID.
 *
 * This function guarantees that the capture will be stopped and released if running the op(s)
 * throws an exception.
 *
 * @tparam Op The type of the operation or a callable op chain that will be invoked to capture the trace operations.
 * @tparam Args The types of the arguments that will be passed to the operation or op chain.
 * @param op The operation or op chain that will be traced.
 * @param device A pointer to the Device object. Must be opened with trace region size set to a sufficiently high
 * amount.
 * @param args The arguments that will be passed to the operation or callable op chain.
 * @return ID for captured trace.
 */
template <typename Op, typename... Args>
auto capture_op_trace(Op op, MeshDevice* device, Args&&... args) {
    // helper lambda to transform TensorSpec to DeviceTensor
    auto transform_arg = [device](auto&& arg) {
        if constexpr (std::is_same_v<std::decay_t<decltype(arg)>, TensorSpec>) {
            return create_device_tensor(arg, device);
        } else if constexpr (std::is_same_v<std::decay_t<decltype(arg)>, std::optional<TensorSpec>>) {
            return arg ? std::optional<Tensor>(create_device_tensor(*arg, device)) : std::nullopt;
        } else if constexpr (std::is_same_v<std::decay_t<decltype(arg)>, std::vector<TensorSpec>>) {
            std::vector<Tensor> result(arg.size());
            std::transform(arg.begin(), arg.end(), result.begin(), [device](auto&& arg) {
                return create_device_tensor(arg, device);
            });
            return result;
        } else {
            return std::forward<decltype(arg)>(arg);
        }
    };
    auto transformed_args = std::make_tuple(transform_arg(std::forward<Args>(args))...);

    device->enable_program_cache();
    {  // warm up the program cache - required for trace capture
        std::apply(op, transformed_args);
    }

    auto trace_id = ttnn::operations::trace::begin_trace_capture(device, ttnn::DefaultQueueId);
    try {
        std::apply(op, transformed_args);
    } catch (const std::exception& e) {
        // Ensure trace capture is stopped and released before returning to avoid a memory leak
        ttnn::operations::trace::end_trace_capture(device, trace_id, ttnn::DefaultQueueId);
        ttnn::operations::trace::release_trace(device, trace_id);
        throw e;
    }
    ttnn::operations::trace::end_trace_capture(device, trace_id, ttnn::DefaultQueueId);

    return trace_id;
}

/**
 * @brief Executes a trace, releases the trace, and returns the runtime in nanoseconds.
 *
 * This function guarantees release_trace will be called even if executing the trace throws an exception.
 *
 * @tparam TraceID The type of the trace id returned by trace capture APIs.
 * @param trace_id ID of the captured trace.
 * @param device A pointer to the Device object
 * @return Trace runtime in nanoseconds.
 */
template <typename TraceID>
uint64_t execute_time_and_release_trace(TraceID trace_id, MeshDevice* device) {
    try {
        uint64_t duration = 0;
        for (int i = 0; i < NUM_TRACE_EXECUTIONS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            ttnn::operations::trace::execute_trace(device, trace_id, ttnn::DefaultQueueId, /* blocking = */ true);
            auto end = std::chrono::high_resolution_clock::now();
            duration += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        }

        ttnn::operations::trace::release_trace(device, trace_id);
        return duration / NUM_TRACE_EXECUTIONS;

    } catch (const std::exception& e) {
        // Ensure captured trace is released before returning to avoid a memory leak
        ttnn::operations::trace::release_trace(device, trace_id);
        throw e;
    }
}

/**
 * @brief Extracts a trace of the graph operations and returns the trace execution runtime.
 *
 * This function runs trace capture by invoking the provided operation with the given arguments,
 * then excutes the trace and returns the runtime of the trace in nanoseconds.
 *
 * @tparam Op The type of the operation or a callable op chain that will be invoked to capture the trace operations.
 * @tparam Args The types of the arguments that will be passed to the operation or op chain.
 * @param op The operation or op chain that will be traced and have its runtime measured.
 * @param device A pointer to the Device object. Must be opened with trace region size set to a sufficiently high
 * amount.
 * @param args The arguments that will be passed to the operation or callable op chain.
 * @return RuntimeQueryResponse containing the execution status and the runtime, in nanoseconds.
 *         - On success: ExecutionStatus::Success and runtime in nanoseconds.
 *         - On failure: ExecutionStatus::Error, zeroed runtime, and an error message.
 */
template <typename Op, typename... Args>
auto query_op_runtime(Op op, MeshDevice* device, Args&&... args) {
    try {
        auto trace_id = capture_op_trace(op, device, std::forward<Args>(args)...);
        auto runtime = execute_time_and_release_trace(trace_id, device);
        return RuntimeQueryResponse{ExecutionStatus::Success, runtime};

    } catch (const std::exception& e) {
        log_debug(tt::LogOp, "op_runtime - error: {}", e.what());
        return RuntimeQueryResponse{ExecutionStatus::Error, 0, e.what()};
    }
}

}  // namespace ttnn::graph
