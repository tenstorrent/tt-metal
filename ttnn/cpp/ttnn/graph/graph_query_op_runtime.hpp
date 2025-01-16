// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>

#include "ttnn/graph/graph_trace_utils.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::graph {

struct RuntimeQueryResponse {
    ExecutionStatus status = ExecutionStatus::Error;
    uint64_t runtime;
    std::optional<std::string> error_message;
};

static constexpr int NUM_TRACE_EXECUTIONS = 10;

/**
 * @brief Extracts a trace of the graph operations and returns the trace execution runtime.
 *
 * This function runs trace capture by invoking the provided operation with the given arguments,
 * then excutes the trace and returns the runtime of the trace in nanoseconds.
 *
 * @tparam Op The type of the operation or a callable op chain that will be invoked to capture the trace operations.
 * @tparam Args The types of the arguments that will be passed to the operation or op chain.
 * @param op The operation or op chain that will be traced and have its runtime measured.
 * @param device A pointer to the Device object, which provides information about the compute grid size.
 * @param args The arguments that will be passed to the operation or callable op chain.
 * @return QueryResponse containing the execution status and the runtime, in nanoseconds.
 *         - On success: ExecutionStatus::Success and the resource usage details.
 *         - On failure: ExecutionStatus::Error, zeroed resource usage, and an error message.
 */
template <typename Op, typename... Args>
auto query_op_runtime(Op op, IDevice* device, Args&&... args) {
    try {
        // helper lambda to transform TensorSpec to DeviceTensor
        auto transform_arg = [device](auto&& arg) {
            if constexpr (std::is_same_v<std::decay_t<decltype(arg)>, TensorSpec>) {
                return create_device_tensor(arg, device);
            } else {
                return std::forward<decltype(arg)>(arg);
            }
        };
        auto transformed_args = std::make_tuple(transform_arg(std::forward<Args>(args))...);

        device->enable_async(false);
        device->enable_program_cache();
        {  // warm up the program cache - required for trace capture
            std::apply(op, transformed_args);
        }

        // capture the trace
        auto trace_id = ttnn::operations::core::begin_trace_capture(device, ttnn::DefaultQueueId);
        std::apply(op, transformed_args);
        ttnn::operations::core::end_trace_capture(device, trace_id, ttnn::DefaultQueueId);

        device->synchronize();
        uint64_t duration = 0;
        for (int i = 0; i < NUM_TRACE_EXECUTIONS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            ttnn::operations::core::execute_trace(device, trace_id, ttnn::DefaultQueueId, true);
            auto end = std::chrono::high_resolution_clock::now();
            duration += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        }
        ttnn::operations::core::release_trace(device, trace_id);

        return RuntimeQueryResponse{ExecutionStatus::Success, duration / NUM_TRACE_EXECUTIONS};

    } catch (const std::exception& e) {
        tt::log_debug(tt::LogOp, "op_constraints - error: {}", e.what());
        return RuntimeQueryResponse{ExecutionStatus::Error, 0, e.what()};
    }
}

}  // namespace ttnn::graph
