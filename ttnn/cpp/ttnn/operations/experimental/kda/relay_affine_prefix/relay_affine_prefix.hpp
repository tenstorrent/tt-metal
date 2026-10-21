// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/global_semaphore.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

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
    uint32_t sequence_parallel_axis = 0,
    const std::optional<ttnn::MemoryConfig>& entry_memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    const std::optional<ttnn::Tensor>& output_gate_input = std::nullopt,
    const std::optional<ttnn::Tensor>& output_gate_weight = std::nullopt,
    const std::optional<ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>&
        output_gate_program_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& output_gate_compute_kernel_config = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& output_gate_memory_config = std::nullopt,
    const std::optional<ttnn::Tensor>& output_gate_copy_source = std::nullopt,
    uint32_t output_gate_copy_source_column = 0,
    uint32_t output_gate_copied_columns = 0,
    const std::vector<tt::tt_metal::CoreCoord>& output_gate_copy_cores = {});

}  // namespace ttnn::experimental::kda
