// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The all-gather MoE block's device programs (models/demos/mimo_v2_d_p/tt/moe_ag.py): the route plan, the local
// weighted reduce (with the two fused send-back phases), the row adds and the untilizes.

#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_ag {

std::vector<ttnn::Tensor> moe_ag_route_plan(
    const ttnn::Tensor& topk_indices,
    const ttnn::Tensor& local_slot_map,
    uint32_t experts_per_chip,
    uint32_t num_rows,
    const std::optional<std::vector<ttnn::Tensor>>& outputs = std::nullopt);

std::vector<ttnn::Tensor> moe_ag_local_reduce(
    const ttnn::Tensor& y,
    const ttnn::Tensor& y_slot,
    const ttnn::Tensor& weights,
    const ttnn::Tensor& chip_info,
    uint32_t chunk_size_per_chip,
    uint32_t phase = 0,
    bool split = false,
    bool tiled = false,
    const std::optional<ttnn::Tensor>& peer = std::nullopt,
    const std::optional<std::vector<ttnn::Tensor>>& outputs = std::nullopt);

ttnn::Tensor moe_ag_sum_rows_tiled(
    const ttnn::Tensor& src,
    uint32_t num_rows,
    uint32_t num_blocks,
    uint32_t block_stride,
    const std::optional<ttnn::Tensor>& output = std::nullopt);

ttnn::Tensor moe_ag_add_rows(
    const ttnn::Tensor& a,
    const ttnn::Tensor& b,
    const ttnn::Tensor& chip_info,
    uint32_t num_rows,
    uint32_t a_offset = 0,
    uint32_t b_offset = 0,
    bool info_offset = false,
    const std::optional<ttnn::Tensor>& output = std::nullopt);

ttnn::Tensor moe_ag_untilize_active(
    const ttnn::Tensor& y,
    const ttnn::Tensor& counts,
    const ttnn::Tensor& regions,
    const ttnn::Tensor& local_slot_map,
    uint32_t experts_per_chip,
    uint32_t tiles_per_block = 32,
    const std::optional<ttnn::Tensor>& output = std::nullopt);

ttnn::Tensor moe_ag_untilize_x(const ttnn::Tensor& x, const std::optional<ttnn::Tensor>& output = std::nullopt);

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_ag
