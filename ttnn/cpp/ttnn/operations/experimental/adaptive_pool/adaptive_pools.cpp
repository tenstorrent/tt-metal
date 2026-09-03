// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "adaptive_pools.hpp"
#include "adaptive_pool_utils.hpp"

namespace ttnn::operations::experimental::adaptive_pool {

// Reusing the generic pool2d functionality from the regular pool operations
Tensor adaptive_avg_pool2d(
    const Tensor& input_tensor,
    uint32_t batch_size,
    uint32_t input_h,
    uint32_t input_w,
    uint32_t channels,
    std::array<uint32_t, 2> output_size,
    const std::optional<const MemoryConfig>& memory_config,
    const std::optional<op_slicing::Op2DSliceConfig>& dram_slice_config,
    const std::optional<const TensorMemoryLayout> applied_shard_scheme,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    bool deallocate_input,
    bool reallocate_output) {
    uint32_t output_h = output_size[0];
    uint32_t output_w = output_size[1];

    // Validate that this adaptive pooling configuration is feasible
    validate_adaptive_pool_feasibility(input_h, input_w, output_h, output_w);

    auto params = calculate_adaptive_pool_params(input_h, input_w, output_h, output_w);

    (void)batch_size;
    (void)channels;
    (void)memory_config;
    (void)dram_slice_config;
    (void)applied_shard_scheme;
    (void)compute_kernel_config;
    (void)deallocate_input;
    (void)reallocate_output;
    (void)params;
    return /* TODO(nuked-op): restore ttnn::avg_pool2d */ (input_tensor);
}

Tensor adaptive_max_pool2d(
    const Tensor& input_tensor,
    uint32_t batch_size,
    uint32_t input_h,
    uint32_t input_w,
    uint32_t channels,
    std::array<uint32_t, 2> output_size,
    const std::optional<const MemoryConfig>& memory_config,
    const std::optional<op_slicing::Op2DSliceConfig>& dram_slice_config,
    const std::optional<const TensorMemoryLayout> applied_shard_scheme,
    bool deallocate_input,
    bool reallocate_output) {
    uint32_t output_h = output_size[0];
    uint32_t output_w = output_size[1];

    // Validate that this adaptive pooling configuration is feasible
    validate_adaptive_pool_feasibility(input_h, input_w, output_h, output_w);

    auto params = calculate_adaptive_pool_params(input_h, input_w, output_h, output_w);

    (void)batch_size;
    (void)channels;
    (void)memory_config;
    (void)dram_slice_config;
    (void)applied_shard_scheme;
    (void)deallocate_input;
    (void)reallocate_output;
    (void)params;
    auto result = /* TODO(nuked-op): restore ttnn::max_pool2d */ std::vector<Tensor>{input_tensor};

    // Since return_indices=false, the result variant should always contain a Tensor
    TT_FATAL(result.size() == 1, "Expected Tensor result when return_indices is false");
    return result.at(0);
}

}  // namespace ttnn::operations::experimental::adaptive_pool
