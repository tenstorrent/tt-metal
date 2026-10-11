// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "qkv_causal_conv1d_silu.hpp"
#include "device/qkv_causal_conv1d_silu_device_operation.hpp"
#include "device/convolution_history_rows_device_operation.hpp"

namespace ttnn::experimental::kda {

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> qkv_causal_conv1d_silu(
    const ttnn::Tensor& input,
    const ttnn::Tensor& history,
    const ttnn::Tensor& tap0,
    const ttnn::Tensor& tap1,
    const ttnn::Tensor& tap2,
    const ttnn::Tensor& tap3,
    uint32_t q_width,
    uint32_t k_width,
    uint32_t v_width,
    const QkvCausalConv1dSiluProgramConfig& program_config,
    const ttnn::Tensor& actual_start,
    const ttnn::Tensor& predecessor_carry,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    uint32_t sequence_parallel_axis,
    uint32_t input_column_offset) {
    TT_FATAL(
        input.storage_type() == StorageType::DEVICE && input.buffer() != nullptr,
        "qkv_causal_conv1d_silu: input must be an allocated device tensor");
    const auto output_memory_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    const auto kernel_config = init_device_compute_kernel_config(
        input.device()->arch(),
        compute_kernel_config,
        tt::tt_metal::MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/false,
        /*default_l1_acc=*/false);
    auto outputs = ttnn::experimental::prim::qkv_causal_conv1d_silu(
        input,
        history,
        tap0,
        tap1,
        tap2,
        tap3,
        q_width,
        k_width,
        v_width,
        program_config.channel_chunk_size,
        output_memory_config,
        kernel_config,
        actual_start,
        sequence_parallel_axis,
        predecessor_carry,
        std::nullopt,
        std::nullopt,
        input_column_offset);
    return {outputs[0], outputs[1], outputs[2]};
}

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> qkv_causal_conv1d_silu_with_carry(
    const ttnn::Tensor& input,
    const ttnn::Tensor& history,
    const ttnn::Tensor& tap0,
    const ttnn::Tensor& tap1,
    const ttnn::Tensor& tap2,
    const ttnn::Tensor& tap3,
    uint32_t q_width,
    uint32_t k_width,
    uint32_t v_width,
    const QkvCausalConv1dSiluProgramConfig& program_config,
    const ttnn::Tensor& actual_start,
    const ttnn::Tensor& history_table,
    const std::optional<ttnn::Tensor>& actual_end,
    uint32_t input_column_offset,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    uint32_t sequence_parallel_axis) {
    TT_FATAL(
        input.storage_type() == StorageType::DEVICE && input.buffer() != nullptr,
        "qkv_causal_conv1d_silu_with_carry: input must be an allocated device tensor");
    const auto kernel_config = init_device_compute_kernel_config(
        input.device()->arch(),
        compute_kernel_config,
        tt::tt_metal::MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/false,
        /*default_l1_acc=*/false);
    auto outputs = ttnn::experimental::prim::qkv_causal_conv1d_silu(
        input,
        history,
        tap0,
        tap1,
        tap2,
        tap3,
        q_width,
        k_width,
        v_width,
        program_config.channel_chunk_size,
        memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG),
        kernel_config,
        actual_start,
        sequence_parallel_axis,
        std::nullopt,
        history_table,
        actual_end,
        input_column_offset);
    return {outputs[0], outputs[1], outputs[2], outputs[3]};
}

ttnn::Tensor convolution_history_rows(
    const ttnn::Tensor& input,
    const ttnn::Tensor& actual_start,
    uint32_t width,
    uint32_t column_offset,
    const std::optional<ttnn::Tensor>& actual_end,
    uint32_t sequence_parallel_axis) {
    return ttnn::experimental::prim::convolution_history_rows(
        input, actual_start, width, column_offset, actual_end, sequence_parallel_axis);
}

}  // namespace ttnn::experimental::kda
