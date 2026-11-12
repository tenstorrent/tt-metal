// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ttnn.bringup.moe_block_main.expert_slot_copy: table-driven copies of routed-expert weight slots between chips (2D
// fabric unicast, any chip to any chip) or within a chip. See README.md.

#pragma once

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

#include "../common/moe_block_resources.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::bringup::moe_block_main::expert_slot_copy {

// jobs: (src chip, dst chip, run table) with chips as r C + c. src != dst: a fabric copy - chip src reads the table's
// runs from src_gu / src_d and writes them into chip dst's dst_gu / dst_d (a destination at most once a call; the
// destination's program waits for every byte, and the source waits for the destination's program to start). src ==
// dst: a local copy on that chip (src_* -> dst_*), split over its port cores. links: sender cores a fabric copy (the
// first hop's links; default all the resources' links). Returns dst_gu.
Tensor expert_slot_copy(
    const Tensor& src_gu,
    const Tensor& src_d,
    const Tensor& dst_gu,
    const Tensor& dst_d,
    const Tensor& runs,
    MoEBlockResources& resources,
    const std::vector<std::tuple<uint32_t, uint32_t, uint32_t>>& jobs,
    std::optional<uint32_t> links = std::nullopt,
    uint32_t depth = 8,
    uint32_t batch = 4);

// The chunk (fabric packet payload) a run is cut into, bytes.
uint32_t expert_slot_copy_chunk_bytes(const MoEBlockResources& resources);

}  // namespace ttnn::operations::bringup::moe_block_main::expert_slot_copy
