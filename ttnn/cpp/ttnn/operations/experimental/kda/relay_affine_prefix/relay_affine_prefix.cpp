// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "relay_affine_prefix.hpp"

#include "device/relay_affine_prefix_device_operation.hpp"

namespace ttnn::experimental::kda {

std::tuple<ttnn::Tensor, ttnn::Tensor, std::optional<ttnn::Tensor>> relay_affine_prefix(
    const ttnn::Tensor& packed,
    const ttnn::Tensor& initial_state,
    const ttnn::Tensor& actual_start,
    const ttnn::Tensor& entry_landing,
    const ttnn::Tensor& state_staging,
    const ttnn::Tensor& final_landing,
    const std::vector<tt::tt_metal::GlobalSemaphore>& semaphores,
    uint32_t local_rows,
    uint32_t num_links,
    const std::vector<tt::tt_metal::CoreCoord>& cores,
    uint32_t sequence_parallel_axis,
    const std::optional<ttnn::MemoryConfig>& entry_memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<ttnn::Tensor>& output_gate_input,
    const std::optional<ttnn::Tensor>& output_gate_weight,
    const std::optional<ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>&
        output_gate_program_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& output_gate_compute_kernel_config,
    const std::optional<ttnn::MemoryConfig>& output_gate_memory_config,
    const std::optional<ttnn::Tensor>& output_gate_copy_source,
    uint32_t output_gate_copy_source_column,
    uint32_t output_gate_copied_columns,
    const std::vector<tt::tt_metal::CoreCoord>& output_gate_copy_cores) {
    TT_FATAL(
        packed.storage_type() == StorageType::DEVICE && packed.buffer() != nullptr,
        "relay_affine_prefix: packed must be an allocated device tensor");
    TT_FATAL(semaphores.size() == 2, "relay_affine_prefix: needs two global semaphores, got {}", semaphores.size());
    const auto arch = packed.device()->arch();
    std::optional<ttnn::experimental::prim::AffineRelayOutputGateParams> output_gate;
    if (output_gate_program_config) {
        output_gate = ttnn::experimental::prim::AffineRelayOutputGateParams{
            .program_config = *output_gate_program_config,
            // HiFi2 unless given, approximation off, FP32 accumulation.
            .compute_kernel_config = ttnn::init_device_compute_kernel_config(
                arch,
                output_gate_compute_kernel_config,
                tt::tt_metal::MathFidelity::HiFi2,
                /*default_approx_mode=*/false,
                /*default_fp32_acc=*/true),
            .output_mem_config = output_gate_memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG),
            .output_column = output_gate_copied_columns,
            .copy_source_column = output_gate_copy_source_column,
            .copy_cores = output_gate_copy_cores};
    }
    // The hop's matmul runs at the affine prefix's fidelity, accumulating in an FP32 DST.
    const auto kernel_config = ttnn::init_device_compute_kernel_config(
        arch,
        compute_kernel_config,
        tt::tt_metal::MathFidelity::HiFi2,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false);
    return ttnn::experimental::prim::relay_affine_prefix(
        packed,
        initial_state,
        actual_start,
        entry_landing,
        state_staging,
        final_landing,
        entry_memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG),
        kernel_config,
        {static_cast<uint32_t>(semaphores[0].address()), static_cast<uint32_t>(semaphores[1].address())},
        sequence_parallel_axis,
        local_rows,
        num_links,
        cores,
        output_gate,
        output_gate_input,
        output_gate_weight,
        output_gate_copy_source);
}

}  // namespace ttnn::experimental::kda
