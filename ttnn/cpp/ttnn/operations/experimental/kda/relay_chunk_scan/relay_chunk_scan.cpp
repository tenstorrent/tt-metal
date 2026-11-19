// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "relay_chunk_scan.hpp"

#include "device/relay_chunk_scan_device_operation.hpp"

namespace ttnn::experimental::kda {

std::tuple<ttnn::Tensor, ttnn::Tensor> relay_chunk_scan(
    const ttnn::Tensor& packed,
    const ttnn::Tensor& terms,
    const ttnn::Tensor& initial_state,
    const ttnn::Tensor& final_e,
    const ttnn::Tensor& final_b,
    const ttnn::Tensor& gate_input,
    const ttnn::Tensor& gate_weight,
    const ttnn::Tensor& norm_weight,
    const ttnn::Tensor& actual_start,
    const ttnn::Tensor& landing,
    const std::vector<tt::tt_metal::GlobalSemaphore>& semaphores,
    uint32_t local_rows,
    uint32_t num_links,
    const std::vector<tt::tt_metal::CoreCoord>& cores,
    uint32_t num_heads,
    const std::optional<ttnn::Tensor>& actual_end,
    uint32_t sequence_parallel_axis,
    const std::optional<ttnn::Tensor>& gate_local,
    tt::tt_metal::MathFidelity gate_math_fidelity,
    float norm_epsilon,
    tt::tt_metal::DataType output_dtype,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<ttnn::Tensor>& slab_index,
    uint32_t slab_index_scale,
    uint32_t slab_index_offset) {
    TT_FATAL(
        packed.storage_type() == StorageType::DEVICE && packed.is_allocated(),
        "relay_chunk_scan: packed must be an allocated device tensor");
    TT_FATAL(
        semaphores.size() == prim::relay_semaphores,
        "relay_chunk_scan: needs {} global semaphores, got {}",
        prim::relay_semaphores,
        semaphores.size());
    std::array<uint32_t, prim::relay_semaphores> semaphore_addresses{};
    for (uint32_t i = 0; i < prim::relay_semaphores; ++i) {
        semaphore_addresses[i] = static_cast<uint32_t>(semaphores[i].address());
    }
    // The hop's, the correction's and the norm's products accumulate in an FP32 DST.
    const auto kernel_config = ttnn::init_device_compute_kernel_config(
        packed.device()->arch(),
        compute_kernel_config,
        tt::tt_metal::MathFidelity::HiFi3,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false);
    return ttnn::experimental::prim::relay_chunk_scan(
        packed,
        terms,
        initial_state,
        final_e,
        final_b,
        gate_input,
        gate_local,
        gate_weight,
        norm_weight,
        actual_start,
        actual_end,
        landing,
        num_heads,
        gate_math_fidelity,
        norm_epsilon,
        output_dtype,
        memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG),
        kernel_config,
        semaphore_addresses,
        sequence_parallel_axis,
        local_rows,
        num_links,
        cores,
        slab_index,
        slab_index_scale,
        slab_index_offset);
}

}  // namespace ttnn::experimental::kda
