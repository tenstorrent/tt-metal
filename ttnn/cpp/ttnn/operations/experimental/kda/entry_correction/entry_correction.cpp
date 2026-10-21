// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "entry_correction.hpp"

#include "device/entry_correction_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::kda {

std::tuple<ttnn::Tensor, ttnn::Tensor> entry_correction(
    const ttnn::Tensor& zero_output,
    const ttnn::Tensor& identity_output,
    const ttnn::Tensor& final_a,
    const ttnn::Tensor& final_b,
    const ttnn::Tensor& entry,
    const ttnn::Tensor& tail_entry,
    const ttnn::Tensor& actual_start,
    uint32_t local_rows,
    const std::vector<tt::tt_metal::CoreCoord>& cores,
    const std::optional<ttnn::Tensor>& actual_end,
    uint32_t sequence_parallel_axis,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    auto outputs = ttnn::device_operation::launch<ttnn::experimental::prim::EntryCorrectionOperation>(
        ttnn::experimental::prim::EntryCorrectionParams{
            .sequence_parallel_axis = sequence_parallel_axis,
            .local_rows = local_rows,
            .cores = cores,
            .output_mem_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG),
            .compute_kernel_config = ttnn::init_device_compute_kernel_config(
                final_b.device()->arch(), compute_kernel_config, tt::tt_metal::MathFidelity::HiFi2, false, true)},
        ttnn::experimental::prim::EntryCorrectionInputs{
            .zero_output = zero_output,
            .identity_output = identity_output,
            .final_a = final_a,
            .final_b = final_b,
            .entry = entry,
            .tail_entry = tail_entry,
            .actual_start = actual_start,
            .actual_end = actual_end});
    return {outputs[0], outputs[1]};
}

}  // namespace ttnn::experimental::kda
