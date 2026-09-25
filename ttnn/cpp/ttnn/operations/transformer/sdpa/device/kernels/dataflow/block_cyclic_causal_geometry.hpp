// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Per-device causal geometry for a query chunk written block-cyclic ("slab") across SP. Shared by the host
// factories and the kernels of indexer_score (tile granularity) and sparse_sdpa_msa (token granularity), so a
// device resolving its start from an on-device chunk_start and the host resolving it from an int agree by
// construction. Pure integer math in ELEMENTS (tokens); callers divide by their own granularity. Keep this
// header free of host-only and kernel-only dependencies.
#pragma once

#include <stdint.h>

namespace tt::block_cyclic {

struct CausalGeometry {
    uint32_t chunk_start;    // global (logical) position of this device's query row 0
    uint32_t straddle_q;     // query row at/after which positions jump by straddle_jump (0 unless straddling)
    uint32_t straddle_jump;  // position jump for rows >= straddle_q (0 unless straddling)
};

// Global position of this device's query row `row`, given its geometry.
inline uint32_t query_position(const CausalGeometry& g, uint32_t row) {
    return g.chunk_start + row + (g.straddle_jump != 0 && row >= g.straddle_q ? g.straddle_jump : 0u);
}

// The global chunk [chunk_start_idx, chunk_start_idx + sp*chunk_local) is written round-robin across the sp
// chips, so chip c's Sq queries are a CONTIGUOUS logical block whose start follows the writer's rotation --
// NOT the linear chunk_start_idx + c*Sq. Two effects of a mid-slab chunk_start_idx:
//   (a) block rotation: the starting block (chunk_start_idx / chunk_local) can land on a chip != 0
//       (boundary_chip), rotating which chip owns which block; and
//   (b) straddle: the boundary chip's Sq queries cross a slab boundary, so its positions JUMP by
//       (chunk_global - chunk_local) at query row (chunk_local - offset).
// Chunk-aligned (offset == 0, boundary_chip == 0) reduces to linear. No block-cyclic layout -> linear.
//
// `rotation_exact` selects the rotation-exact SP mapping (device_index is the SP rank, tp_index the row range
// within its slab) over the flat both-axes approximation (device_index linear over every device, straddle
// only). The caller computes the predicate once on the host and hands the same value to the kernel.
inline CausalGeometry causal_geometry(
    uint32_t chunk_start_idx,
    bool has_block_cyclic,
    bool rotation_exact,
    uint32_t sp,
    uint32_t chunk_local,
    uint32_t device_index,
    uint32_t tp_index,
    uint32_t Sq) {
    if (!has_block_cyclic) {
        // Contiguous K -> linear at chunk_start + (seq-shard rank)*Sq. The rank is device_index for an SP-only
        // seq shard; a 2D SP x TP sub-shard whose SP axis is size-1 is stored as no-block-cyclic, and there the
        // query is seq-sharded over the TP axis, so the rank is tp_index. The two are mutually exclusive
        // nonzero here, so their sum is the rank.
        return {chunk_start_idx + (device_index + tp_index) * Sq, 0u, 0u};
    }

    const uint32_t chunk_global = sp * chunk_local;

    if (rotation_exact) {
        const uint32_t boundary_slab = chunk_start_idx / chunk_global;
        const uint32_t boundary_chip = (chunk_start_idx / chunk_local) % sp;
        const uint32_t offset = chunk_start_idx % chunk_local;
        const uint32_t update_idx = device_index < boundary_chip    ? (boundary_slab + 1) * chunk_local
                                    : device_index == boundary_chip ? boundary_slab * chunk_local + offset
                                                                    : boundary_slab * chunk_local;
        const uint32_t lr0 = update_idx + tp_index * Sq;  // this device's first slab-local row
        const uint32_t loff = lr0 % chunk_local;          // its offset within the current slab
        const uint32_t logical_start = (lr0 / chunk_local) * chunk_global + device_index * chunk_local + loff;
        if (loff != 0 && loff + Sq > chunk_local) {  // this device's Sq rows cross a slab boundary
            return {logical_start, chunk_local - loff, chunk_global - chunk_local};
        }
        return {logical_start, 0u, 0u};
    }

    // Both-axes (SP axis unset): linear + within-block straddle.
    const uint32_t chunk_start = chunk_start_idx + device_index * Sq;
    const uint32_t offset = chunk_start % chunk_local;
    if (offset != 0 && offset + Sq > chunk_local) {
        return {chunk_start, chunk_local - offset, chunk_global - chunk_local};
    }
    return {chunk_start, 0u, 0u};
}

}  // namespace tt::block_cyclic
