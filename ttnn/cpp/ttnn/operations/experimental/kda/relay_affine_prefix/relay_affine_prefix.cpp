// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "relay_affine_prefix.hpp"

#include "device/relay_affine_prefix_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::kda {

std::tuple<ttnn::Tensor, ttnn::Tensor> relay_affine_prefix(
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
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    TT_FATAL(semaphores.size() == 2, "relay_affine_prefix: needs two global semaphores, got {}", semaphores.size());
    auto outputs = ttnn::device_operation::launch<ttnn::experimental::prim::RelayAffinePrefixOperation>(
        ttnn::experimental::prim::RelayAffinePrefixParams{
            .sequence_parallel_axis = sequence_parallel_axis,
            .local_rows = local_rows,
            .num_links = num_links,
            .cores = cores,
            .entry_mem_config = entry_memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG),
            .compute_kernel_config = ttnn::init_device_compute_kernel_config(
                packed.device()->arch(), compute_kernel_config, tt::tt_metal::MathFidelity::HiFi2, false, true),
            .semaphore_addresses =
                {static_cast<uint32_t>(semaphores[0].address()), static_cast<uint32_t>(semaphores[1].address())}},
        ttnn::experimental::prim::RelayAffinePrefixInputs{
            .packed = packed,
            .initial_state = initial_state,
            .actual_start = actual_start,
            .entry_landing = entry_landing,
            .state_staging = state_staging,
            .final_landing = final_landing});
    return {outputs[0], outputs[1]};
}

}  // namespace ttnn::experimental::kda
