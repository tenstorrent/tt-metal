// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/reduction/attn_res_accum_stats/attn_res_accum_stats.hpp"

#include "ttnn/operations/experimental/reduction/attn_res_accum_stats/device/attn_res_accum_stats_device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::reduction {

std::array<ttnn::Tensor, 2> attn_res_accum_stats(
    const ttnn::Tensor& a,
    const ttnn::Tensor& b,
    const ttnn::Tensor& q,
    const std::optional<ttnn::DataType>& stats_dtype,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    TT_FATAL(
        a.storage_type() == StorageType::DEVICE,
        "Input tensor storage type must be DEVICE but got {}",
        a.storage_type());

    // Both statistics are sums over the full `d` shard, so the accumulation is where the
    // precision goes; HiFi4 and fp32 dest keep it out of the input dtype until the pack.
    auto kernel_config_val = init_device_compute_kernel_config(
        a.device()->arch(),
        compute_kernel_config,
        tt::tt_metal::MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true);

    // The sum takes the addends' memory config rather than the statistics': it is the
    // residual stream the caller already holds, and only the statistics are the new shape.
    return ttnn::prim::attn_res_accum_stats(
        a,
        b,
        q,
        stats_dtype.value_or(a.dtype()),
        a.memory_config(),
        memory_config.value_or(a.memory_config()),
        kernel_config_val);
}

}  // namespace ttnn::experimental::reduction
