// SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads_rope/nlp_create_qkv_heads_rope.hpp"

#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads_rope/device/nlp_create_qkv_heads_rope_device_operation.hpp"

namespace ttnn::experimental {

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> nlp_create_qkv_heads_rope(
    const Tensor& qkv,
    const Tensor& cos_cache,
    const Tensor& sin_cache,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    using namespace tt::constants;

    TT_FATAL(qkv.storage_type() == StorageType::DEVICE, "qkv must be on device");
    uint32_t seq_len = qkv.padded_shape()[-2];
    uint32_t head_dim = qkv.padded_shape()[-1] / (num_q_heads + 2 * num_kv_heads);

    auto arch = qkv.device()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(
        arch, compute_kernel_config, tt::tt_metal::MathFidelity::HiFi4, true, false, false);

    return ttnn::prim::nlp_create_qkv_heads_rope(
        qkv,
        cos_cache,
        sin_cache,
        num_q_heads,
        num_kv_heads,
        head_dim,
        seq_len,
        memory_config.value_or(qkv.memory_config()),
        kernel_config_val);
}

}  // namespace ttnn::experimental
