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
namespace ttnn {
namespace operations::experimental::adaptive_pool {

using namespace ttnn::operations::pool;

// Reusing the generic pool2d functionality from the regular pool operations
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
    bool in_place_halo,
    bool deallocate_input,
    bool reallocate_output) {
    uint32_t output_h = output_size[0];
    uint32_t output_w = output_size[1];

    // Validate that this adaptive pooling configuration is feasible
    validate_adaptive_pool_feasibility(input_h, input_w, output_h, output_w);

    auto params = calculate_adaptive_pool_params(input_h, input_w, output_h, output_w);

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
        false,         // ceil_mode
        false,         // count_include_pad always false because we want to ignore padding values
        std::nullopt,  // divisor_override
        memory_config,
        applied_shard_scheme,
        in_place_halo,
        deallocate_input,
        reallocate_output);
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
    bool in_place_halo,
    bool deallocate_input,
    bool reallocate_output) {
    uint32_t output_h = output_size[0];
    uint32_t output_w = output_size[1];

    // Validate that this adaptive pooling configuration is feasible
    validate_adaptive_pool_feasibility(input_h, input_w, output_h, output_w);

    auto params = calculate_adaptive_pool_params(input_h, input_w, output_h, output_w);

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
        {1, 1},  // dilation
        false,   // ceil_mode
        memory_config,
        applied_shard_scheme,
        in_place_halo,
        deallocate_input,
        reallocate_output);
}

}  // namespace operations::experimental::adaptive_pool
}  // namespace ttnn
