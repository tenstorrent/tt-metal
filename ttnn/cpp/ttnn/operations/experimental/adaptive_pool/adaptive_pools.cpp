// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "adaptive_pools.hpp"
#include "adaptive_pool_utils.hpp"

#include "tt-metalium/constants.hpp"
#include <tt-metalium/buffer_types.hpp>
#include "ttnn/operations/conv/conv2d/conv2d_utils.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/pool/pool_utils.hpp"
#include "ttnn/operations/pool/generic/generic_pools.hpp"
#include "ttnn/operations/sliding_window/halo/halo.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/operations/data_movement/move/move.hpp"
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/math.hpp>
#include <tt-logger/tt-logger.hpp>

namespace ttnn {
namespace operations::experimental::adaptive_pool {

using namespace ttnn::operations::pool;

// We'll reuse the generic pool2d functionality from the regular pool operations

Tensor AdaptiveAvgPool2DOp::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    uint32_t batch_size,
    uint32_t input_h,
    uint32_t input_w,
    uint32_t channels,
    std::array<uint32_t, 2> output_size,
    const std::optional<const MemoryConfig>& memory_config,
    const std::optional<const TensorMemoryLayout> applied_shard_scheme,
    bool in_place_halo) {
    log_debug(
        tt::LogOp,
        "[Experimental AdaptiveAvgPool2D] ENTRY: input_tensor.dtype={}, input_tensor.layout={}",
        static_cast<int>(input_tensor.dtype()),
        static_cast<int>(input_tensor.layout()));
    log_debug(
        tt::LogOp,
        "[Experimental AdaptiveAvgPool2D] PARAMS: batch={}, input_h={}, input_w={}, channels={}, output_size=[{}, {}]",
        batch_size,
        input_h,
        input_w,
        channels,
        output_size[0],
        output_size[1]);

    uint32_t output_h = output_size[0];
    uint32_t output_w = output_size[1];

    // PATTERN-BASED HYBRID: Analyze kernel patterns and apply padding/dilation as needed
    log_info(
        tt::LogOp, "[Experimental AdaptiveAvgPool2D] Using PATTERN-BASED HYBRID approach - prioritizing correctness");

    auto hybrid_config = calculate_pattern_based_hybrid_config(input_h, input_w, output_h, output_w);
    auto params = convert_hybrid_to_legacy(hybrid_config, input_h, input_w);

    // PATTERN-BASED HYBRID RESULTS
    log_info(tt::LogOp, "[Experimental AdaptiveAvgPool2D] === PATTERN-BASED HYBRID RESULTS ===");
    log_info(
        tt::LogOp,
        "[Experimental AdaptiveAvgPool2D] Strategy: {} | Coverage improvement: {:.1f}%",
        static_cast<int>(hybrid_config.strategy),
        hybrid_config.coverage_improvement_percent);
    log_info(
        tt::LogOp,
        "[Experimental AdaptiveAvgPool2D] Variance: {}×{} -> {}×{}",
        hybrid_config.h_variance,
        hybrid_config.w_variance,
        hybrid_config.h_variance_after,
        hybrid_config.w_variance_after);
    log_info(
        tt::LogOp,
        "[Experimental AdaptiveAvgPool2D] Kernel: {}×{} | Stride: {}×{}",
        params.kernel_size[0],
        params.kernel_size[1],
        params.stride[0],
        params.stride[1]);
    log_info(
        tt::LogOp,
        "[Experimental AdaptiveAvgPool2D] Padding: [{},{},{},{}] | Dilation: {}×{}",
        params.padding[0],
        params.padding[1],
        params.padding[2],
        params.padding[3],
        hybrid_config.dilation[0],
        hybrid_config.dilation[1]);
    if (params.memory_overhead_percent > 0.0) {
        log_info(
            tt::LogOp,
            "[Experimental AdaptiveAvgPool2D] Memory overhead: {:.1f}% | Beneficial: {}",
            params.memory_overhead_percent,
            hybrid_config.is_beneficial() ? "YES" : "NO");
    }

    log_debug(tt::LogOp, "[Experimental AdaptiveAvgPool2D] CALLING pool2d_invoke with PATTERN-BASED HYBRID parameters");

    return ttnn::operations::pool::AvgPool2DOp::invoke(
        queue_id,
        input_tensor,
        batch_size,
        input_h,
        input_w,
        channels,
        params.kernel_size,
        params.stride,
        params.padding,
        false,         // ceil_mode = false
        false,         // count_include_pad = FALSE (ignores padding values)
        std::nullopt,  // divisor_override
        memory_config,
        applied_shard_scheme,
        in_place_halo,
        false,  // deallocate_input = false
        true);  // reallocate_halo_output = true
}

Tensor AdaptiveMaxPool2DOp::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    uint32_t batch_size,
    uint32_t input_h,
    uint32_t input_w,
    uint32_t channels,
    std::array<uint32_t, 2> output_size,
    const std::optional<const MemoryConfig>& memory_config,
    const std::optional<const TensorMemoryLayout> applied_shard_scheme,
    bool in_place_halo) {
    log_debug(
        tt::LogOp,
        "[Experimental AdaptiveMaxPool2D] ENTRY: input_tensor.dtype={}, input_tensor.layout={}",
        static_cast<int>(input_tensor.dtype()),
        static_cast<int>(input_tensor.layout()));
    log_debug(
        tt::LogOp,
        "[Experimental AdaptiveMaxPool2D] PARAMS: batch={}, input_h={}, input_w={}, channels={}, output_size=[{}, {}]",
        batch_size,
        input_h,
        input_w,
        channels,
        output_size[0],
        output_size[1]);

    uint32_t output_h = output_size[0];
    uint32_t output_w = output_size[1];

    // PATTERN-BASED HYBRID: Same approach as AvgPool
    log_info(
        tt::LogOp, "[Experimental AdaptiveMaxPool2D] Using PATTERN-BASED HYBRID approach - prioritizing correctness");

    auto hybrid_config = calculate_pattern_based_hybrid_config(input_h, input_w, output_h, output_w);
    auto params = convert_hybrid_to_legacy(hybrid_config, input_h, input_w);

    // PATTERN-BASED HYBRID RESULTS
    log_info(tt::LogOp, "[Experimental AdaptiveMaxPool2D] === PATTERN-BASED HYBRID RESULTS ===");
    log_info(
        tt::LogOp,
        "[Experimental AdaptiveMaxPool2D] Strategy: {} | Coverage improvement: {:.1f}%",
        static_cast<int>(hybrid_config.strategy),
        hybrid_config.coverage_improvement_percent);
    log_info(
        tt::LogOp,
        "[Experimental AdaptiveMaxPool2D] Variance: {}×{} -> {}×{}",
        hybrid_config.h_variance,
        hybrid_config.w_variance,
        hybrid_config.h_variance_after,
        hybrid_config.w_variance_after);
    log_info(
        tt::LogOp,
        "[Experimental AdaptiveMaxPool2D] Kernel: {}×{} | Stride: {}×{}",
        params.kernel_size[0],
        params.kernel_size[1],
        params.stride[0],
        params.stride[1]);
    log_info(
        tt::LogOp,
        "[Experimental AdaptiveMaxPool2D] Padding: [{},{},{},{}] | Dilation: {}×{}",
        params.padding[0],
        params.padding[1],
        params.padding[2],
        params.padding[3],
        hybrid_config.dilation[0],
        hybrid_config.dilation[1]);
    if (params.memory_overhead_percent > 0.0) {
        log_info(
            tt::LogOp,
            "[Experimental AdaptiveMaxPool2D] Memory overhead: {:.1f}% | Beneficial: {}",
            params.memory_overhead_percent,
            hybrid_config.is_beneficial() ? "YES" : "NO");
    }

    log_debug(tt::LogOp, "[Experimental AdaptiveMaxPool2D] CALLING pool2d_invoke with PATTERN-BASED HYBRID parameters");

    return ttnn::operations::pool::MaxPool2DOp::invoke(
        queue_id,
        input_tensor,
        batch_size,
        input_h,
        input_w,
        channels,
        params.kernel_size,
        params.stride,
        params.padding,
        hybrid_config.dilation,  // PATTERN-BASED: Calculated dilation based on kernel analysis for MaxPool too
        false,                   // ceil_mode = false
        memory_config,
        applied_shard_scheme,
        in_place_halo,
        false,  // deallocate_input = false
        true);  // reallocate_halo_output = true
}

}  // namespace operations::experimental::adaptive_pool
}  // namespace ttnn
