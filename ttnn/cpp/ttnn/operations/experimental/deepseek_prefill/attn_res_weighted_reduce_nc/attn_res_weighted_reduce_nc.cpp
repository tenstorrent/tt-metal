// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/deepseek_prefill/attn_res_weighted_reduce_nc/attn_res_weighted_reduce_nc.hpp"

#include "ttnn/operations/experimental/deepseek_prefill/attn_res_weighted_reduce_nc/device/attn_res_weighted_reduce_nc_device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::attn_res_weighted_reduce_nc {

ttnn::Tensor attn_res_weighted_reduce_nc(
    const ttnn::Tensor& input,
    const ttnn::Tensor& weight,
    int32_t dim,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    TT_FATAL(
        input.storage_type() == StorageType::DEVICE,
        "Input tensor storage type must be DEVICE but got {}",
        input.storage_type());

    const auto rank = static_cast<int32_t>(input.padded_shape().rank());
    const int32_t normalized_dim = (dim < 0) ? dim + rank : dim;

    // HiFi4 with fp32 dest accumulation, matching deepseek_moe_fast_reduce_nc_fused
    // — the other op that MACs into dst. Two reasons, and only one is accuracy:
    // the accumulator is read back and written every candidate, so a bf16 dest
    // rounds the running sum C times; and matching fast_reduce_nc's HiFi4 default
    // keeps an A/B against it a measurement of the fusion, not of the fidelity.
    auto kernel_config_val = init_device_compute_kernel_config(
        input.device()->arch(),
        compute_kernel_config,
        tt::tt_metal::MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true);

    return ttnn::prim::attn_res_weighted_reduce_nc(
        input, weight, normalized_dim, memory_config.value_or(input.memory_config()), kernel_config_val);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::attn_res_weighted_reduce_nc
