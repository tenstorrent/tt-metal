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
//  group: rows a reader -> sender hand-off (default: 8 if 2 W <= 7168, else 2); manifest_batch: manifest entries a
//  stream batch; pack: rows a fabric packet (default: 2 when two rows fit the payload); debug / spin_limit: a
//  [8 N, 16] uint32 DRAM tensor receiving each reader's state when it idles past spin_limit polls.
Tensor sparse_dispatch_2d(
    const Tensor& xbuf,
    const Tensor& route,
    MoEBlockResources& resources,
    std::optional<uint32_t> group = std::nullopt,
    uint32_t manifest_batch = 16,
    std::optional<uint32_t> pack = std::nullopt,
    const std::optional<Tensor>& debug = std::nullopt,
    uint32_t spin_limit = 0);

// Creates the op's persistent resources (the port cores' semaphore words) now: call before any per-call allocation
// (and before freeze()).
void ensure_sparse_dispatch_2d_resources(MoEBlockResources& resources);

}  // namespace ttnn::operations::experimental::moe_block::sparse_dispatch_2d
