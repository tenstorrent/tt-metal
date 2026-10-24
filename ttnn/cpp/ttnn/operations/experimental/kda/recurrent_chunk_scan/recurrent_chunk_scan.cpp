// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "recurrent_chunk_scan.hpp"

#include <array>
#include <utility>

#include "device/recurrent_chunk_scan_device_operation.hpp"
#include "device/tile_column_activation_device_operation.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

namespace ttnn::experimental::kda {
namespace {

void validate_protocol_inputs(
    const ttnn::Tensor& v_beta,
    const ttnn::Tensor& kd,
    const ttnn::Tensor& q_decay,
    const ttnn::Tensor& intra,
    const ttnn::Tensor& k_dec_t,
    const ttnn::Tensor& final_decay,
    const ttnn::Tensor& t_inv,
    std::string_view operation_name) {
    using namespace ttnn::experimental::prim::kda_factory_detail;
    const std::array<std::pair<const ttnn::Tensor*, const char*>, 7> inputs = {
        {{&v_beta, "v_beta"},
         {&kd, "kd"},
         {&q_decay, "q_decay"},
         {&intra, "intra"},
         {&k_dec_t, "k_dec_t"},
         {&final_decay, "final_decay"},
         {&t_inv, "t_inv"}}};
    for (const auto& [tensor, name] : inputs) {
        check_allocated_device_tensor(*tensor, operation_name, name);
        TT_FATAL(tensor->logical_shape().rank() == 4, "{}: {} must be rank 4", operation_name, name);
    }
}

std::pair<ttnn::MemoryConfig, ttnn::DeviceComputeKernelConfig> resolve_configs(
    const ttnn::Tensor& anchor,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    const auto output_memory_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    auto kernel_config = init_device_compute_kernel_config(
        anchor.device()->arch(),
        compute_kernel_config,
        tt::tt_metal::MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false);
    return {output_memory_config, kernel_config};
}

}  // namespace

std::vector<ttnn::Tensor> recurrent_chunk_scan(
    const ttnn::Tensor& v_beta,
    const ttnn::Tensor& kd,
    const ttnn::Tensor& q_decay,
    const ttnn::Tensor& intra,
    const ttnn::Tensor& k_dec_t,
    const ttnn::Tensor& final_decay,
    const ttnn::Tensor& t_inv,
    const ttnn::Tensor& group_entry_states,
    const ttnn::Tensor& actual_start,
    const ttnn::Tensor& tail_entry_states,
    uint32_t groups_per_head,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    uint32_t sequence_parallel_axis,
    const std::optional<Tensor>& actual_end) {
    using namespace ttnn::experimental::prim::kda_factory_detail;
    constexpr std::string_view operation_name = "recurrent_chunk_scan";
    validate_protocol_inputs(v_beta, kd, q_decay, intra, k_dec_t, final_decay, t_inv, operation_name);
    check_allocated_device_tensor(group_entry_states, operation_name, "group_entry_states");
    TT_FATAL(group_entry_states.logical_shape().rank() == 3, "{}: group_entry_states must be rank 3", operation_name);
    auto [output_memory_config, kernel_config] = resolve_configs(v_beta, memory_config, compute_kernel_config);
    check_output_interleaved(output_memory_config, operation_name);
    return ttnn::experimental::prim::recurrent_chunk_scan(
        v_beta,
        kd,
        q_decay,
        intra,
        k_dec_t,
        final_decay,
        t_inv,
        group_entry_states,
        tail_entry_states,
        ttnn::experimental::prim::RecurrentChunkScanMode::RECURRENT,
        groups_per_head,

        output_memory_config,
        kernel_config,
        actual_start,
        sequence_parallel_axis,
        actual_end);
}

std::vector<ttnn::Tensor> summarize_chunk_recurrence(
    const ttnn::Tensor& v_beta,
    const ttnn::Tensor& kd,
    const ttnn::Tensor& q_decay,
    const ttnn::Tensor& intra,
    const ttnn::Tensor& k_dec_t,
    const ttnn::Tensor& final_decay,
    const ttnn::Tensor& t_inv,
    const ttnn::Tensor& actual_start,
    uint32_t groups_per_head,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    uint32_t sequence_parallel_axis,
    const std::optional<Tensor>& actual_end,
    bool packed_head,
    bool emit_outputs,
    const std::optional<ttnn::MemoryConfig>& emit_memory_config,
    const std::optional<ttnn::Tensor>& column_activation_input,
    uint32_t column_activation_input_column,
    const std::optional<ttnn::Tensor>& column_activation_output,
    uint32_t column_activation_output_column,
    uint32_t column_activation_columns,
    const std::vector<ttnn::operations::unary::EltwiseUnaryWithParam>& column_activation) {
    constexpr std::string_view operation_name = "summarize_chunk_recurrence";
    validate_protocol_inputs(v_beta, kd, q_decay, intra, k_dec_t, final_decay, t_inv, operation_name);
    auto [output_memory_config, kernel_config] = resolve_configs(v_beta, memory_config, compute_kernel_config);
    std::optional<ttnn::experimental::prim::kda_factory_detail::TileColumnActivationParams> activation;
    if (column_activation_input) {
        activation = ttnn::experimental::prim::kda_factory_detail::TileColumnActivationParams{
            .op_chain = column_activation,
            .input_column = column_activation_input_column,
            .output_column = column_activation_output_column,
            .columns = column_activation_columns,
            // FP32 accumulation and precise SFPU: an FP32 output carries the chain at full precision.
            .compute_kernel_config = init_device_compute_kernel_config(
                v_beta.device()->arch(),
                std::nullopt,
                tt::tt_metal::MathFidelity::HiFi4,
                /*default_approx_mode=*/false,
                /*default_fp32_acc=*/true,
                /*default_l1_acc=*/false)};
    }
    // The activation runs beside the summary on the worker cores its value blocks leave idle; when they cover the
    // grid it runs right after, as a program of its own on the whole grid.
    const bool beside = activation.has_value() && !ttnn::experimental::prim::summary_idle_cores(v_beta).empty();
    auto outputs = ttnn::experimental::prim::recurrent_chunk_scan(
        v_beta,
        kd,
        q_decay,
        intra,
        k_dec_t,
        final_decay,
        t_inv,
        std::nullopt,
        std::nullopt,
        ttnn::experimental::prim::RecurrentChunkScanMode::SUMMARY,
        groups_per_head,

        output_memory_config,
        kernel_config,
        actual_start,
        sequence_parallel_axis,
        actual_end,
        packed_head,
        emit_outputs,
        emit_memory_config,
        beside ? activation : std::nullopt,
        beside ? column_activation_input : std::nullopt,
        beside ? column_activation_output : std::nullopt);
    if (activation.has_value() && !beside) {
        TT_FATAL(column_activation_output.has_value(), "{}: a column activation needs its output", operation_name);
        ttnn::experimental::prim::activate_tile_columns(
            *activation, *column_activation_input, *column_activation_output);
    }
    return outputs;
}

}  // namespace ttnn::experimental::kda
