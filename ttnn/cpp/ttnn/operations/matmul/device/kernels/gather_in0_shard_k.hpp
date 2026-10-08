// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// How much of K each in0 shard of the Metal 2.0 gather_in0 matmul's ring holds, shared by the ring's
// dataflow and compute kernels. Shards are `shard_width_in_tiles` wide and cover K in ring order, so when
// K does not fill the ring the last shards are short or empty.

#pragma once

#include <stdint.h>

#include "internal/risc_attribs.h"

// The K tiles the in0 shard of ring position `ring_pos` holds: from ring_pos * shard_width_in_tiles up
// to K, at most one shard wide.
FORCE_INLINE uint32_t in0_shard_k_tiles(uint32_t ring_pos, uint32_t shard_width_in_tiles, uint32_t k_tiles) {
    const uint32_t shard_k_start = ring_pos * shard_width_in_tiles;
    const uint32_t k_tiles_left = shard_k_start < k_tiles ? k_tiles - shard_k_start : 0;
    return k_tiles_left < shard_width_in_tiles ? k_tiles_left : shard_width_in_tiles;
}
