// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ttnn.experimental.moe_block.routed_experts_combine: the flat routed experts and the sparse combine in one program per
// chip (optionally the dispatch as the combine ports' first phase). See README.md.

#pragma once

#include <cstdint>
#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "device/routed_experts_combine_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::moe_block {
class MoEBlockResources;
}

namespace ttnn::operations::experimental::moe_block::routed_experts_combine {

// plan / table: the fused head's planner outputs (the device plan, the hand-off table). cells: the logical cells the
// flat plan leaves free; writers: the flat y writers (logical), writer_words_offset: their words, arena-relative.
// Returns y_own [1, 1, S, Hr] bf16 row-major (persistent: the resources' "routed_experts_combine/out").
ttnn::Tensor routed_experts_combine(
    const ttnn::Tensor& plan,
    const ttnn::Tensor& table,
    MoEBlockResources& resources,
    const std::vector<tt::tt_metal::CoreCoord>& cells,
    const std::vector<tt::tt_metal::CoreCoord>& writers,
    uint32_t writer_words_offset,
    const CombineConfig& combine = {});

}  // namespace ttnn::operations::experimental::moe_block::routed_experts_combine
