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

#ifdef BUILD_TTNN_OP_RUNTIME_PREDICTOR
#include "interface.hpp"
#include "tt_stl/tt_stl/span.hpp"
#include "ttnn/decorators.hpp"

#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/operations/conv/conv2d/prepare_conv2d_weights.hpp"
#include "ttnn/operations/conv/conv_transpose2d/conv_transpose2d.hpp"
#include "ttnn/operations/conv/conv_transpose2d/prepare_conv_transpose2d_weights.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/repeat/repeat.hpp"
#include "ttnn/operations/data_movement/repeat_interleave/repeat_interleave.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/sort/sort.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/eltwise/quantization/quantization.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "ttnn/operations/embedding/embedding.hpp"
#include "ttnn/operations/embedding_backward/embedding_backward.hpp"
#include "ttnn/operations/kv_cache/kv_cache.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/moreh/moreh_cumsum/moreh_cumsum.hpp"
#include "ttnn/operations/normalization/batch_norm/batch_norm.hpp"
#include "ttnn/operations/normalization/rmsnorm/rmsnorm.hpp"
#include "ttnn/operations/normalization/softmax/softmax.hpp"
#include "ttnn/operations/pool/generic/generic_pools.hpp"
#include "ttnn/operations/pool/upsample/upsample.hpp"
#include "ttnn/operations/rand/rand.hpp"
#include "ttnn/operations/reduction/argmax/argmax.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/reduction/prod/prod.hpp"
#include "ttnn/operations/transformer/concatenate_heads/concatenate_heads.hpp"

#endif

namespace ttnn::graph {

struct RuntimeQueryResponse {
    ExecutionStatus status = ExecutionStatus::Error;
    uint64_t runtime = 0;
    std::optional<std::string> error_message;
};

static constexpr size_t NUM_TRACE_EXECUTIONS = 20;
static constexpr size_t WARMUP_TRACE_EXECUTIONS = 5;

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
    device->enable_program_cache();
    {  // warm up the program cache - required for trace capture
        std::apply(op, std::make_tuple(std::forward<Args>(args)...));
    }

    auto trace_id = ttnn::operations::trace::begin_trace_capture(device, ttnn::DefaultQueueId);
    try {
        std::apply(op, std::make_tuple(std::forward<Args>(args)...));
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
        for (size_t i = 0; i < WARMUP_TRACE_EXECUTIONS; ++i) {
            ttnn::operations::trace::execute_trace(device, trace_id, ttnn::DefaultQueueId, /* blocking = */ true);
        }

        uint64_t duration = 0;
        for (size_t i = 0; i < NUM_TRACE_EXECUTIONS; ++i) {
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

#ifdef BUILD_TTNN_OP_RUNTIME_PREDICTOR

// helper function checking for base_name()
// if it does, this implies Op op in query_op_runtime() is a registered operation
template <typename T>
concept HasBaseName = requires(const T& t) {
    { t.base_name() } -> std::convertible_to<std::string>;
};

template <typename T>
auto get_op_name(const T& op) -> decltype(op.base_name()) {
    return op.base_name();
}

inline std::string get_op_name(const std::string& op) { return op; }

// helper function for ttnn-op-runtime-predictor
template <typename Op, typename... Args>
std::optional<RuntimeQueryResponse> query_ttnn_op_runtime_predictor(
    const Op& op, const std::tuple<Args...>& transformed_args) {
    if constexpr (HasBaseName<Op>) {
        // helper lambda to make nlohmann::json objects from args
        auto transform_to_json = [](auto&& arg) {
            using ArgType = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<ArgType, nlohmann::json>) {
                return arg;
            } else {
                auto json_arg = ttsl::json::to_json(arg);
                return json_arg;
            }
        };

        auto json_args_tuple = std::apply(
            [&](auto&&... unpacked_args) { return std::make_tuple(transform_to_json(unpacked_args)...); },
            transformed_args);

        const auto& op_name = get_op_name(op);

        uint64_t runtime = std::apply(
            [&](auto&&... json_args) { return op_perf::get_runtime_from_model(op_name, json_args...); },
            json_args_tuple);
        if (runtime != 0) {
            return RuntimeQueryResponse{ExecutionStatus::Success, runtime};
        }
    }
    return std::nullopt;
}

#endif

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

#ifdef BUILD_TTNN_OP_RUNTIME_PREDICTOR
    // query_response is an std::optional<RuntimeQueryResponse>
    // if it has a value, return it
    if (auto query_response = query_ttnn_op_runtime_predictor(op, transformed_args)) {
        return *query_response;
    }
#endif

    try {
        auto trace_id = std::apply(
            [&](auto&&... unpacked_args) {
                return capture_op_trace(op, device, std::forward<decltype(unpacked_args)>(unpacked_args)...);
            },
            transformed_args);
        auto runtime = execute_time_and_release_trace(trace_id, device);
        return RuntimeQueryResponse{ExecutionStatus::Success, runtime};

    } catch (const std::exception& e) {
        log_debug(tt::LogOp, "op_runtime - error: {}", e.what());
        return RuntimeQueryResponse{ExecutionStatus::Error, 0, e.what()};
    }
}

}  // namespace ttnn::graph
