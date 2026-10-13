// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "qkv_causal_conv1d_silu.hpp"
#include "device/qkv_causal_conv1d_silu_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/to_layout/to_layout_op.hpp"

namespace ttnn::experimental::kda {

namespace {

namespace prim = ttnn::experimental::prim;

prim::QkvCausalConv1dSiluParams make_params(
    const ttnn::Tensor& input,
    uint32_t q_width,
    uint32_t k_width,
    uint32_t v_width,
    const QkvCausalConv1dSiluProgramConfig& program_config,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    uint32_t sequence_parallel_axis,
    uint32_t input_column_offset) {
    TT_FATAL(
        input.storage_type() == StorageType::DEVICE && input.buffer() != nullptr && input.logical_shape().rank() == 3,
        "qkv_causal_conv1d_silu: input must be an allocated device tensor of rank 3");
    return {
        .sequence = static_cast<uint32_t>(input.logical_shape()[1]),
        .q_width = q_width,
        .k_width = k_width,
        .v_width = v_width,
        .channel_chunk_size = program_config.channel_chunk_size,
        .sequence_parallel_axis = sequence_parallel_axis,
        .input_column_offset = input_column_offset,
        .output_mem_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG),
        .compute_kernel_config = init_device_compute_kernel_config(
            input.device()->arch(),
            compute_kernel_config,
            tt::tt_metal::MathFidelity::HiFi4,
            /*default_approx_mode=*/false,
            /*default_fp32_acc=*/false,
            /*default_l1_acc=*/false)};
}

}  // namespace

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
    auto params = make_params(
        input,
        q_width,
        k_width,
        v_width,
        program_config,
        memory_config,
        compute_kernel_config,
        sequence_parallel_axis,
        input_column_offset);
    // The op reads TILE input in place; a ROW_MAJOR [1, T, Q+K+V] input is tilized first.
    const auto tiled_input = input.layout() == Layout::TILE ? input : ttnn::to_layout(input, Layout::TILE);
    auto outputs = ttnn::device_operation::launch<prim::QkvCausalConv1dSiluOperation>(
        params,
        prim::QkvCausalConv1dSiluInputs{
            .input = tiled_input,
            .history = history,
            .tap0 = tap0,
            .tap1 = tap1,
            .tap2 = tap2,
            .tap3 = tap3,
            .actual_start = actual_start,
            .predecessor_carry = predecessor_carry});
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
    const ttnn::Tensor& exchange_buffer,
    const std::vector<tt::tt_metal::GlobalSemaphore>& exchange_semaphores,
    const std::optional<ttnn::Tensor>& actual_end,
    uint32_t input_column_offset,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    uint32_t sequence_parallel_axis) {
    auto params = make_params(
        input,
        q_width,
        k_width,
        v_width,
        program_config,
        memory_config,
        compute_kernel_config,
        sequence_parallel_axis,
        input_column_offset);
    TT_FATAL(
        exchange_semaphores.size() == params.exchange_semaphore_addresses.size(),
        "qkv_causal_conv1d_silu_with_carry: pass {} exchange_semaphores",
        params.exchange_semaphore_addresses.size());
    // The exchange runs on one core outside the convolution grid; its semaphores must live there.
    const CoreCoord exchange_core =
        prim::qkv_causal_conv1d_silu_exchange_core(input.device()->compute_with_storage_grid_size());
    for (uint32_t i = 0; i < exchange_semaphores.size(); ++i) {
        TT_FATAL(
            std::get<0>(exchange_semaphores[i].attribute_values()).contains(exchange_core),
            "qkv_causal_conv1d_silu_with_carry: exchange semaphores must cover core {}",
            exchange_core);
        params.exchange_semaphore_addresses[i] = static_cast<uint32_t>(exchange_semaphores[i].address());
    }
    auto outputs = ttnn::device_operation::launch<prim::QkvCausalConv1dSiluOperation>(
        params,
        prim::QkvCausalConv1dSiluInputs{
            .input = input,
            .history = history,
            .tap0 = tap0,
            .tap1 = tap1,
            .tap2 = tap2,
            .tap3 = tap3,
            .actual_start = actual_start,
            .exchange_buffer = exchange_buffer,
            .actual_end = actual_end});
    return {outputs[0], outputs[1], outputs[2], outputs[3]};
}

}  // namespace ttnn::experimental::kda
