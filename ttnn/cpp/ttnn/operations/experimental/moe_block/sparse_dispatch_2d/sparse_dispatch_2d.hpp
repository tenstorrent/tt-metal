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
// word names; xbuf is returned (filled in place).
//  group: rows a reader -> sender hand-off (default: dispatch_group(resources)); manifest_batch: manifest entries a
//  stream batch; pack: rows a fabric packet (default: 2 when two rows fit the payload).
// The row group the dispatch takes by default: the block's L1 plan's (BlockL1Plan.dispatch_group), else the largest
// supported (kGroups).
uint32_t dispatch_group(const MoEBlockResources& resources);

//  arena: the block's L1 arena of the call (the CBs go in its first sparse_dispatch_2d_arena_bytes a port core: no
//  static CB under the arena); absent: static CBs.
Tensor sparse_dispatch_2d(
    const Tensor& xbuf,
    const Tensor& route,
    MoEBlockResources& resources,
    std::optional<uint32_t> group = std::nullopt,
    uint32_t manifest_batch = 16,
    std::optional<uint32_t> pack = std::nullopt,
    const std::optional<Tensor>& arena = std::nullopt);

// Bytes a port core of the block's arena the dispatch's CBs take with arena= (default group, manifest_batch 16), for
// rows of `width` elements of `dtype_bytes` bytes (64 B aligned rows, as the DRAM xbuf's pages).
uint32_t sparse_dispatch_2d_arena_bytes(MoEBlockResources& resources, uint32_t width, uint32_t dtype_bytes = 2);

}  // namespace ttnn::operations::experimental::moe_block::sparse_dispatch_2d
