// SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/transformer/rotary_embedding_fused_qk/rotary_embedding_fused_qk.hpp"

#include "ttnn/operations/experimental/transformer/rotary_embedding_fused_qk/device/rotary_embedding_fused_qk_device_operation.hpp"

namespace ttnn::experimental {

std::tuple<ttnn::Tensor, ttnn::Tensor> rotary_embedding_fused_qk(
    const Tensor& q,
    const Tensor& k,
    const Tensor& cos_cache,
    const Tensor& sin_cache,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    using namespace tt::constants;

    TT_FATAL(q.storage_type() == StorageType::DEVICE, "q must be on device");
    uint32_t seq_len = q.padded_shape()[-2];

    auto arch = q.device()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(
        arch, compute_kernel_config, tt::tt_metal::MathFidelity::HiFi4, true, false, false);

    tt::tt_metal::MemoryConfig default_memory_config = q.memory_config();

    return ttnn::prim::rotary_embedding_fused_qk(
        q, k, cos_cache, sin_cache, seq_len, memory_config.value_or(default_memory_config), kernel_config_val);
}

}  // namespace ttnn::experimental
