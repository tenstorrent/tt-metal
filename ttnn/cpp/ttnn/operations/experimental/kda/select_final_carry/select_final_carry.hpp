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

ttnn::Tensor select_final_carry(
    const ttnn::Tensor& rank_final,
    const ttnn::Tensor& prefix_final,
    const ttnn::Tensor& actual_start,
    uint32_t local_rows,
    const std::optional<ttnn::Tensor>& actual_end = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    uint32_t sequence_parallel_axis = 0,
    std::optional<uint32_t> num_links = std::nullopt,
    const std::optional<std::vector<tt::tt_metal::CoreCoord>>& cores = std::nullopt,
    bool prefix_on_last_rank = false,
    const std::optional<std::vector<tt::tt_metal::GlobalSemaphore>>& semaphores = std::nullopt);

// The final carry with projection_input @ projection_weight in its dispatch, on worker cores the carry's `cores` leave
// (ttnn.linear's 2D multicast matmul with projection_program_config, on its cores): returns the final carry and the
// projection's output.
std::tuple<ttnn::Tensor, ttnn::Tensor> select_final_carry_with_projection(
    const ttnn::Tensor& rank_final,
    const ttnn::Tensor& prefix_final,
    const ttnn::Tensor& projection_input,
    const ttnn::Tensor& projection_weight,
    const ttnn::Tensor& actual_start,
    uint32_t local_rows,
    const std::vector<tt::tt_metal::CoreCoord>& cores,
    const ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig& projection_program_config,
    const std::optional<ttnn::Tensor>& actual_end = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    uint32_t sequence_parallel_axis = 0,
    std::optional<uint32_t> num_links = std::nullopt,
    bool prefix_on_last_rank = false,
    const std::optional<ttnn::MemoryConfig>& projection_memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& projection_compute_kernel_config = std::nullopt,
    const std::optional<std::vector<tt::tt_metal::GlobalSemaphore>>& semaphores = std::nullopt);

}  // namespace ttnn::experimental::kda
