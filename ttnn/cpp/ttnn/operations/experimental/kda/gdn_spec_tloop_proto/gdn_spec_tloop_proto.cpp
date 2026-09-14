// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "gdn_spec_tloop_proto.hpp"

#include <cmath>

#include "device/gdn_spec_tloop_proto_device_operation.hpp"

namespace ttnn::experimental::kda {

ttnn::Tensor gdn_spec_tloop_proto(
    const ttnn::Tensor& qkv,
    const ttnn::Tensor& dt_bias,
    const ttnn::Tensor& neg_exp_A,
    const ttnn::Tensor& ring,
    const ttnn::Tensor& weight,
    uint32_t num_value_heads,
    uint32_t num_key_heads,
    uint32_t key_dim,
    uint32_t value_dim,
    uint32_t T,
    uint32_t B,
    uint32_t qkvz_dim,
    uint32_t s0_slot,
    std::optional<float> scale,
    float l2_epsilon,
    float norm_epsilon,
    bool row_batched,
    bool write_ring,
    uint32_t opt_flags,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    DataType output_dtype) {
    TT_FATAL(
        qkv.storage_type() == StorageType::DEVICE && qkv.buffer() != nullptr,
        "gdn_spec_tloop_proto: qkv must be an allocated device tensor");
    const auto output_memory_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    const auto kernel_config = init_device_compute_kernel_config(
        qkv.device()->arch(),
        compute_kernel_config,
        MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false);
    const float scale_value = scale.value_or(1.0f / std::sqrt(static_cast<float>(key_dim)));
    return ttnn::experimental::prim::gdn_spec_tloop_proto(
        qkv,
        dt_bias,
        neg_exp_A,
        ring,
        weight,
        num_value_heads,
        num_key_heads,
        key_dim,
        value_dim,
        T,
        B,
        qkvz_dim,
        s0_slot,
        scale_value,
        l2_epsilon,
        norm_epsilon,
        row_batched,
        write_ring,
        opt_flags,
        output_memory_config,
        kernel_config,
        output_dtype);
}

}  // namespace ttnn::experimental::kda
