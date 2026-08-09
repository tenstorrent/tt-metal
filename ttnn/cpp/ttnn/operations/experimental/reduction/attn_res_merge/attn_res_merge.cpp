// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/reduction/attn_res_merge/attn_res_merge.hpp"

#include "ttnn/operations/experimental/reduction/attn_res_merge/device/attn_res_merge_device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::reduction {

ttnn::Tensor attn_res_merge(
    const ttnn::Tensor& partial,
    const ttnn::Tensor& prefix_sum,
    const ttnn::Tensor& shift,
    const ttnn::Tensor& mass,
    const ttnn::Tensor& live_scores,
    uint32_t site,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    TT_FATAL(
        partial.storage_type() == StorageType::DEVICE,
        "Input tensor storage type must be DEVICE but got {}",
        partial.storage_type());

    // HiFi4 with fp32 dest accumulation, matching fast_weighted_reduce_nc — the
    // other op whose full-width path MACs into dst. The scalar chain runs in dst
    // too, and `exp` then `recip` on a bf16 dest would round the denominator
    // twice before it ever divides a value.
    auto kernel_config_val = init_device_compute_kernel_config(
        partial.device()->arch(),
        compute_kernel_config,
        tt::tt_metal::MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true);

    return ttnn::prim::attn_res_merge(
        partial,
        prefix_sum,
        shift,
        mass,
        live_scores,
        site,
        memory_config.value_or(partial.memory_config()),
        kernel_config_val);
}

}  // namespace ttnn::experimental::reduction
