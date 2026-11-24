// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ttnn.experimental.moe_block.sparse_dispatch_2d: the sparse 2D dispatch of the routed rows (multi-hop unicast trees
// over the port cores). See README.md.

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/operations/experimental/moe_block/common/moe_block_resources.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::moe_block::sparse_dispatch_2d {

// Every chip's own S rows sit at slots (r * C + c) * S + t of xbuf ([1, 1, N * S, W] ROW_MAJOR, DRAM interleaved);
// route ([1, 1, 1, RW * S] uint32) holds their route words. Each row lands in the same slot of every chip its route
// word names; xbuf is returned (filled in place). Rows a reader -> sender hand-off: dispatch_group(resources); rows a
// fabric packet: 2 when two rows fit the payload; manifest entries a stream batch: 16.
// The row group the dispatch takes by default: the block's L1 plan's (BlockL1Plan.dispatch_group), else the largest
// supported (kGroups).
uint32_t dispatch_group(const MoEBlockResources& resources);

// arena: the block's L1 arena of the call; the CBs go in its first CbSizes::arena_bytes() on every port core
// (device/sparse_dispatch_2d_plan.hpp) instead of being static CBs. Absent: static CBs.
Tensor sparse_dispatch_2d(
    const Tensor& xbuf,
    const Tensor& route,
    MoEBlockResources& resources,
    const std::optional<Tensor>& arena = std::nullopt);

}  // namespace ttnn::operations::experimental::moe_block::sparse_dispatch_2d
