// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <variant>
#include "adaptive_pools.hpp"
#include "adaptive_pool_utils.hpp"
#include "ttnn/operations/pool/generic/generic_pools.hpp"
namespace ttnn {
namespace operations::experimental::adaptive_pool {

// Reusing the generic pool2d functionality from the regular pool operations
Tensor AdaptiveAvgPool2DOp::invoke(
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

    auto result = ttnn::operations::pool::MaxPool2DOp::invoke(
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
        reallocate_output,
        false /*return_indices*/);

    // Since return_indices=false, the result variant should always contain a Tensor
    TT_FATAL(std::holds_alternative<Tensor>(result), "Expected Tensor result when return_indices is false");
    return std::get<Tensor>(result);
}

}  // namespace operations::experimental::adaptive_pool
}  // namespace ttnn
