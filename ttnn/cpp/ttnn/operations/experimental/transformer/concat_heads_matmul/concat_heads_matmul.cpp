// SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/transformer/concat_heads_matmul/concat_heads_matmul.hpp"

#include "ttnn/operations/experimental/transformer/concat_heads_matmul/device/concat_heads_matmul_device_operation.hpp"

namespace ttnn::experimental {

ttnn::Tensor concat_heads_matmul(
    const Tensor& attn,
    const Tensor& weight,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<tt::tt_metal::DataType> output_dtype,
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    std::optional<ttnn::operations::matmul::MatmulProgramConfig> program_config) {
    using namespace tt::constants;

    TT_FATAL(attn.storage_type() == StorageType::DEVICE, "attn must be on device");
    uint32_t seq_len = attn.padded_shape()[2];

    auto arch = attn.device()->arch();
    // Replicate ttnn.matmul's EXACT default config (matmul_device_operation.cpp L1513-1522/1561-67)
    // so this op is numerically identical to the O-proj it replaces: with a program_config supplied,
    // increase_fidelity is false -> LoFi, packer_l1_acc=true (bf16 inputs).
    using tt::tt_metal::DataType;
    bool low_prec = (attn.dtype() == DataType::BFLOAT8_B || attn.dtype() == DataType::BFLOAT4_B) &&
                    (weight.dtype() == DataType::BFLOAT8_B || weight.dtype() == DataType::BFLOAT4_B);
    bool inputs_32f = attn.dtype() == DataType::FLOAT32 && weight.dtype() == DataType::FLOAT32;
    bool has_pc = program_config.has_value();
    auto math_fidelity = (!has_pc && !low_prec) ? tt::tt_metal::MathFidelity::HiFi2 : tt::tt_metal::MathFidelity::LoFi;
    math_fidelity = inputs_32f ? (arch == tt::ARCH::WORMHOLE_B0 ? tt::tt_metal::MathFidelity::HiFi3
                                                                : tt::tt_metal::MathFidelity::HiFi4)
                               : math_fidelity;
    auto kernel_config_val = init_device_compute_kernel_config(
        arch,
        compute_kernel_config,
        math_fidelity,
        /*approx=*/false,
        /*fp32_acc=*/inputs_32f,
        /*l1_acc=*/!inputs_32f);

    return ttnn::prim::concat_heads_matmul(
        attn,
        weight,
        seq_len,
        memory_config.value_or(attn.memory_config()),
        output_dtype.value_or(tt::tt_metal::DataType::BFLOAT16),
        kernel_config_val,
        std::move(program_config));
}

}  // namespace ttnn::experimental
