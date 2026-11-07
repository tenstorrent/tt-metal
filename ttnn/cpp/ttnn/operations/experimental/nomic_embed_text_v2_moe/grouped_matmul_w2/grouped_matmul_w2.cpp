// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "grouped_matmul_w2.hpp"

#include "device/grouped_matmul_w2_device_operation.hpp"
#include "ttnn/operations/experimental/nomic_embed_text_v2_moe/grouped_matmul_w1/grouped_matmul_w1.hpp"

namespace ttnn::experimental::nomic_embed_text_v2_moe {

Tensor grouped_matmul_w2(
    const Tensor& input_tensor,
    const Tensor& weight,
    const Tensor& expert_counts,
    uint32_t weight_chunk_tiles,
    uint32_t weight_n,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    uint32_t first_slots,
    uint32_t second_slots) {
    // Without a config: HiFi4 and fp32 accumulation, the most precise setting.
    const auto kernel_config = init_device_compute_kernel_config(
        input_tensor.device()->arch(),
        compute_kernel_config,
        tt::tt_metal::MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true);
    return ttnn::prim::grouped_matmul_w2(
        input_tensor,
        weight,
        expert_counts,
        memory_config.value_or(
            tt::tt_metal::MemoryConfig(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1)),
        kernel_config,
        weight_chunk_tiles,
        weight_n,
        first_slots,
        second_slots);
}

Tensor prepare_grouped_matmul_w2_weight(
    const Tensor& weight, uint32_t num_experts, uint32_t chunk_tiles, ttnn::MeshDevice* device) {
    // w2's slices are w1's transposed read: one layout, one implementation.
    return prepare_grouped_matmul_w1_weight(
        weight, num_experts, /*transpose_weight=*/true, std::nullopt, chunk_tiles, device);
}

tt::tt_metal::MemoryConfig grouped_matmul_w2_input_memory_config(uint32_t rows, uint32_t k, ttnn::MeshDevice* device) {
    TT_FATAL(device != nullptr, "grouped_matmul_w2_input_memory_config: device must be given");
    return grouped_matmul_w2_input_memory_config(rows, k, device->compute_with_storage_grid_size());
}

uint32_t grouped_matmul_w2_circular_buffer_bytes(
    uint32_t k,
    uint32_t n,
    uint32_t first_slots,
    uint32_t second_slots,
    uint32_t num_experts,
    ttnn::MeshDevice* device) {
    TT_FATAL(device != nullptr, "grouped_matmul_w2_circular_buffer_bytes: device must be given");
    return grouped_matmul_w2_circular_buffer_bytes(
        k, n, first_slots, second_slots, num_experts, device->compute_with_storage_grid_size());
}

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe
