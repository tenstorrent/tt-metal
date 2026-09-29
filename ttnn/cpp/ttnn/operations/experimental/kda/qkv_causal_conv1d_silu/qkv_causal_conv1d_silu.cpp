// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "qkv_causal_conv1d_silu.hpp"
#include "device/qkv_causal_conv1d_silu_device_operation.hpp"

namespace ttnn::experimental::kda {

namespace {

std::vector<ttnn::Tensor> run_qkv_causal_conv1d_silu(
    const ttnn::Tensor& input,
    const std::optional<ttnn::Tensor>& history,
    const ttnn::Tensor& tap0,
    const ttnn::Tensor& tap1,
    const ttnn::Tensor& tap2,
    const ttnn::Tensor& tap3,
    uint32_t q_width,
    uint32_t k_width,
    uint32_t v_width,
    const std::optional<QkvCausalConv1dSiluProgramConfig>& program_config,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    bool return_conv_state,
    const std::optional<ttnn::Tensor>& conv_state_output = std::nullopt) {
    TT_FATAL(
        input.storage_type() == StorageType::DEVICE && input.buffer() != nullptr,
        "qkv_causal_conv1d_silu: input must be an allocated device tensor");
    uint32_t channel_chunk_size = 0;
    if (program_config.has_value()) {
        channel_chunk_size = program_config->channel_chunk_size;
    } else {
        TT_FATAL(
            input.layout() == tt::tt_metal::Layout::TILE,
            "qkv_causal_conv1d_silu: program_config is required for ROW_MAJOR input (only TILE input has a "
            "default channel_chunk_size)");
        channel_chunk_size = ttnn::experimental::prim::qkv_causal_conv1d_silu_tiled::default_channel_chunk_size(
            q_width, k_width, v_width);
    }
    const auto output_memory_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    const auto kernel_config = init_device_compute_kernel_config(
        input.device()->arch(),
        compute_kernel_config,
        MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/false,
        /*default_l1_acc=*/false);
    return ttnn::experimental::prim::qkv_causal_conv1d_silu(
        input,
        history,
        tap0,
        tap1,
        tap2,
        tap3,
        q_width,
        k_width,
        v_width,
        channel_chunk_size,
        return_conv_state,
        output_memory_config,
        kernel_config,
        conv_state_output);
}

}  // namespace

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> qkv_causal_conv1d_silu(
    const ttnn::Tensor& input,
    const std::optional<ttnn::Tensor>& history,
    const ttnn::Tensor& tap0,
    const ttnn::Tensor& tap1,
    const ttnn::Tensor& tap2,
    const ttnn::Tensor& tap3,
    uint32_t q_width,
    uint32_t k_width,
    uint32_t v_width,
    const std::optional<QkvCausalConv1dSiluProgramConfig>& program_config,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    auto outputs = run_qkv_causal_conv1d_silu(
        input,
        history,
        tap0,
        tap1,
        tap2,
        tap3,
        q_width,
        k_width,
        v_width,
        program_config,
        memory_config,
        compute_kernel_config,
        /*return_conv_state=*/false);
    return {outputs[0], outputs[1], outputs[2]};
}

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> qkv_causal_conv1d_silu_with_conv_state(
    const ttnn::Tensor& input,
    const std::optional<ttnn::Tensor>& history,
    const ttnn::Tensor& tap0,
    const ttnn::Tensor& tap1,
    const ttnn::Tensor& tap2,
    const ttnn::Tensor& tap3,
    uint32_t q_width,
    uint32_t k_width,
    uint32_t v_width,
    const std::optional<QkvCausalConv1dSiluProgramConfig>& program_config,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<ttnn::Tensor>& conv_state_output) {
    auto outputs = run_qkv_causal_conv1d_silu(
        input,
        history,
        tap0,
        tap1,
        tap2,
        tap3,
        q_width,
        k_width,
        v_width,
        program_config,
        memory_config,
        compute_kernel_config,
        /*return_conv_state=*/true,
        conv_state_output);
    return {outputs[0], outputs[1], outputs[2], outputs[3]};
}

}  // namespace ttnn::experimental::kda
