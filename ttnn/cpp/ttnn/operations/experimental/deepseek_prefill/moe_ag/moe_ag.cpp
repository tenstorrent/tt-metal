// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_ag.hpp"

#include "device/moe_ag_local_reduce_device_operation.hpp"
#include "device/moe_ag_route_plan_device_operation.hpp"
#include "device/moe_ag_row_ops_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_ag {

std::vector<ttnn::Tensor> moe_ag_route_plan(
    const ttnn::Tensor& topk_indices,
    const ttnn::Tensor& local_slot_map,
    uint32_t experts_per_chip,
    uint32_t num_rows,
    const std::optional<std::vector<ttnn::Tensor>>& outputs) {
    return ttnn::prim::moe_ag_route_plan(
        topk_indices, local_slot_map, experts_per_chip, num_rows, outputs.value_or(std::vector<ttnn::Tensor>{}));
}

std::vector<ttnn::Tensor> moe_ag_local_reduce(
    const ttnn::Tensor& y,
    const ttnn::Tensor& y_slot,
    const ttnn::Tensor& weights,
    const ttnn::Tensor& chip_info,
    uint32_t chunk_size_per_chip,
    uint32_t phase,
    bool split,
    bool tiled,
    const std::optional<ttnn::Tensor>& peer,
    const std::optional<std::vector<ttnn::Tensor>>& outputs) {
    return ttnn::prim::moe_ag_local_reduce(
        y,
        y_slot,
        weights,
        chip_info,
        chunk_size_per_chip,
        phase,
        split,
        tiled,
        peer,
        outputs.value_or(std::vector<ttnn::Tensor>{}));
}

ttnn::Tensor moe_ag_sum_rows_tiled(
    const ttnn::Tensor& src,
    uint32_t num_rows,
    uint32_t num_blocks,
    uint32_t block_stride,
    const std::optional<ttnn::Tensor>& output) {
    return ttnn::prim::moe_ag_sum_rows_tiled(src, num_rows, num_blocks, block_stride, output);
}

ttnn::Tensor moe_ag_add_rows(
    const ttnn::Tensor& a,
    const ttnn::Tensor& b,
    const ttnn::Tensor& chip_info,
    uint32_t num_rows,
    uint32_t a_offset,
    uint32_t b_offset,
    bool info_offset,
    const std::optional<ttnn::Tensor>& output) {
    return ttnn::prim::moe_ag_add_rows(a, b, chip_info, num_rows, a_offset, b_offset, info_offset, output);
}

ttnn::Tensor moe_ag_untilize_active(
    const ttnn::Tensor& y,
    const ttnn::Tensor& counts,
    const ttnn::Tensor& regions,
    const ttnn::Tensor& local_slot_map,
    uint32_t experts_per_chip,
    uint32_t tiles_per_block,
    const std::optional<ttnn::Tensor>& output) {
    return ttnn::prim::moe_ag_untilize_active(
        y, counts, regions, local_slot_map, experts_per_chip, tiles_per_block, output);
}

ttnn::Tensor moe_ag_untilize_x(const ttnn::Tensor& x, const std::optional<ttnn::Tensor>& output) {
    return ttnn::prim::moe_ag_untilize_x(x, output);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_ag
