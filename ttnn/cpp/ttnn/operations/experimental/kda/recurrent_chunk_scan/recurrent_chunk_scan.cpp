// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "recurrent_chunk_scan.hpp"

#include <array>
#include <utility>

#include "device/recurrent_chunk_scan_device_operation.hpp"

namespace ttnn::experimental::kda {
namespace {

void validate_allocated_device_tensor(const ttnn::Tensor& tensor, const char* name) {
    TT_FATAL(
        tensor.storage_type() == StorageType::DEVICE && tensor.buffer() != nullptr,
        "recurrent_chunk_scan: {} must be an allocated device tensor",
        name);
}

void validate_protocol_inputs(
    const ttnn::Tensor& v_beta,
    const ttnn::Tensor& kd,
    const ttnn::Tensor& q_decay,
    const ttnn::Tensor& intra,
    const ttnn::Tensor& k_dec_t,
    const ttnn::Tensor& final_decay,
    const ttnn::Tensor& t_inv) {
    const std::array<std::pair<const ttnn::Tensor*, const char*>, 7> inputs = {
        {{&v_beta, "v_beta"},
         {&kd, "kd"},
         {&q_decay, "q_decay"},
         {&intra, "intra"},
         {&k_dec_t, "k_dec_t"},
         {&final_decay, "final_decay"},
         {&t_inv, "t_inv"}}};
    for (const auto& [tensor, name] : inputs) {
        validate_allocated_device_tensor(*tensor, name);
        TT_FATAL(tensor->logical_shape().rank() == 4, "recurrent_chunk_scan: {} must be rank 4", name);
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
        MathFidelity::HiFi4,
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
    const ttnn::Tensor& initial_state,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    validate_protocol_inputs(v_beta, kd, q_decay, intra, k_dec_t, final_decay, t_inv);
    validate_allocated_device_tensor(initial_state, "initial_state");
    TT_FATAL(initial_state.logical_shape().rank() == 3, "recurrent_chunk_scan: initial_state must be rank 3");
    auto [output_memory_config, kernel_config] = resolve_configs(v_beta, memory_config, compute_kernel_config);
    TT_FATAL(!output_memory_config.is_sharded(), "recurrent_chunk_scan: output memory must be interleaved");
    return ttnn::experimental::prim::recurrent_chunk_scan(
        v_beta,
        kd,
        q_decay,
        intra,
        k_dec_t,
        final_decay,
        t_inv,
        initial_state,
        ttnn::experimental::prim::RecurrentChunkScanMode::RECURRENT,
        output_memory_config,
        kernel_config);
}

std::vector<ttnn::Tensor> summarize_chunk_recurrence(
    const ttnn::Tensor& v_beta,
    const ttnn::Tensor& kd,
    const ttnn::Tensor& q_decay,
    const ttnn::Tensor& intra,
    const ttnn::Tensor& k_dec_t,
    const ttnn::Tensor& final_decay,
    const ttnn::Tensor& t_inv,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    validate_protocol_inputs(v_beta, kd, q_decay, intra, k_dec_t, final_decay, t_inv);
    auto [output_memory_config, kernel_config] = resolve_configs(v_beta, memory_config, compute_kernel_config);
    return ttnn::experimental::prim::recurrent_chunk_scan(
        v_beta,
        kd,
        q_decay,
        intra,
        k_dec_t,
        final_decay,
        t_inv,
        std::nullopt,
        ttnn::experimental::prim::RecurrentChunkScanMode::SUMMARY,
        output_memory_config,
        kernel_config);
}

}  // namespace ttnn::experimental::kda
