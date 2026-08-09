// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/reduction/attn_res_stats/attn_res_stats.hpp"

#include "ttnn/operations/experimental/reduction/attn_res_stats/device/attn_res_stats_device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::reduction {

ttnn::Tensor attn_res_stats(
    const ttnn::Tensor& v,
    const ttnn::Tensor& q,
    const std::optional<ttnn::DataType>& dtype,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    TT_FATAL(
        v.storage_type() == StorageType::DEVICE,
        "Input tensor storage type must be DEVICE but got {}",
        v.storage_type());

    // Both statistics are sums over the full `d` shard, so the accumulation is
    // where the precision goes; HiFi4 and fp32 dest keep it out of the input
    // dtype until the pack.
    auto kernel_config_val = init_device_compute_kernel_config(
        v.device()->arch(),
        compute_kernel_config,
        tt::tt_metal::MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true);

    return ttnn::prim::attn_res_stats(
        v, q, dtype.value_or(v.dtype()), memory_config.value_or(v.memory_config()), kernel_config_val);
}

}  // namespace ttnn::experimental::reduction
