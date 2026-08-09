// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/reduction/attn_res_scores/attn_res_scores.hpp"

#include "ttnn/operations/experimental/reduction/attn_res_scores/device/attn_res_scores_device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::reduction {

ttnn::Tensor attn_res_scores(
    const ttnn::Tensor& stats,
    float inv_hidden_size,
    float eps,
    const std::optional<ttnn::DataType>& dtype,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    TT_FATAL(
        stats.storage_type() == StorageType::DEVICE,
        "Input tensor storage type must be DEVICE but got {}",
        stats.storage_type());

    // fp32 dest accumulation because the whole point of the fusion is that the
    // scale, the epsilon and the reciprocal square root never round to the
    // output dtype between steps — a caller handing over fp32 statistics is
    // asking for exactly that.
    auto kernel_config_val = init_device_compute_kernel_config(
        stats.device()->arch(),
        compute_kernel_config,
        tt::tt_metal::MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true);

    return ttnn::prim::attn_res_scores(
        stats,
        inv_hidden_size,
        eps,
        dtype.value_or(stats.dtype()),
        memory_config.value_or(stats.memory_config()),
        kernel_config_val);
}

}  // namespace ttnn::experimental::reduction
