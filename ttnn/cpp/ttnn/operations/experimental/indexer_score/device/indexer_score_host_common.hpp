// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Host-side helpers shared by BOTH indexer_score program factories (the classic Program-model factory and the
// ring-fused ProgramDescriptor-model factory): per-device causal geometry, the SP-ring index lookup, and the
// persistent-cache runtime args. Kept in one place so the two factories can never drift (they produce
// byte-identical values by construction). Pure host math -- no kernel/program-model dependency.

#include <algorithm>
#include <cstdint>
#include <vector>

#include <tt_stl/assert.hpp>           // TT_FATAL
#include <tt-metalium/constants.hpp>   // tt::constants::TILE_WIDTH
#include <tt-metalium/core_coord.hpp>  // CoreCoord

#include "ttnn/distributed/types.hpp"                // ttnn::MeshCoordinate
#include "ttnn/operations/ccl/ccl_common.hpp"        // get_linearized_index_from_physical_coord
#include "indexer_score_device_operation_types.hpp"  // operation_attributes_t, Tensor

namespace ttnn::operations::experimental::indexer_score::program {

// This device's SP-ring size: the mesh extent along cluster_axis, or the whole mesh when cluster_axis is
// unset (both-axes block-cyclic). Shared by the fused factory (schedule/AG) and the op's validate so they
// cannot disagree on the ring extent.
inline uint32_t ring_size_for(const operation_attributes_t& args, const Tensor& q) {
    return args.cluster_axis.has_value() ? static_cast<uint32_t>(q.device()->get_view().shape()[*args.cluster_axis])
                                         : static_cast<uint32_t>(q.device()->get_view().shape().mesh_size());
}

// relu(q.kT) gate-mul batching that sizes cb_qk (and the compute kernel's DEST batch): batch a tile-row of
// heads (QC>1, capped at TILE_HEIGHT) or the whole plane's heads (QC==1), and — only on the single-chunk
// full-strip path (the whole plane batched, KC>=2, no head streaming) — batch the whole k chunk's columns per
// matmul<->mul switch. `subblock_basis` is the per-plane head count. Identical derivation in both factories
// (classic + fused), so it lives here; each factory keeps its own cb_qk make-call (the classic overrides it to
// 1 tile on its fuse_single path). The divisibility invariant is asserted here (only reachable at QC>1 with a
// >32-head plane; deployed cases never hit the 32-cap).
struct DsaQkBatching {
    uint32_t qk_batch_heads;  // heads batched per matmul<->mul switch (multiple of qk_subblock_h)
    uint32_t qk_col_batch;    // k-chunk columns batched (KC on the single-chunk full-strip path, else 1)
};
inline DsaQkBatching dsa_qk_batching(uint32_t subblock_basis, uint32_t QC, uint32_t KC, bool stream_heads) {
    const uint32_t qk_batch_cap = (QC == 1) ? subblock_basis : tt::constants::TILE_HEIGHT;
    const uint32_t qk_batch_heads = std::min<uint32_t>(subblock_basis, qk_batch_cap);
    TT_FATAL(
        subblock_basis % qk_batch_heads == 0,
        "indexer_score: per-plane head count {} not divisible by qk_batch_heads {} (QC>1 with >32 heads); reduce "
        "head_group_size, q_chunk_size, or raise num_groups",
        subblock_basis,
        qk_batch_heads);
    const bool single_chunk = (qk_batch_heads == subblock_basis) && !stream_heads;
    const uint32_t qk_col_batch = (KC >= 2 && single_chunk) ? KC : 1u;
    return {qk_batch_heads, qk_col_batch};
}

// Q/W row multicast bounding box for a schedule row: the horizontal rect spanning the used columns (py is
// constant along the row) and the diagonal sender column. Identical geometry in both factories (the banded
// schedule is shared), so it lives here to keep them from drifting.
struct QMcastBBox {
    uint32_t xs, xe, py, diag_col;
    CoreCoord sender;
};
inline QMcastBBox q_mcast_bbox(const std::vector<std::vector<CoreCoord>>& phys, uint32_t row, uint32_t cols_used) {
    uint32_t xs = static_cast<uint32_t>(phys[row][0].x), xe = xs;
    for (uint32_t col = 0; col < cols_used; ++col) {
        xs = std::min<uint32_t>(xs, static_cast<uint32_t>(phys[row][col].x));
        xe = std::max<uint32_t>(xe, static_cast<uint32_t>(phys[row][col].x));
    }
    const uint32_t diag_col = std::min<uint32_t>(row, cols_used - 1);  // diagonal sender column
    return {xs, xe, static_cast<uint32_t>(phys[row][0].y), diag_col, phys[row][diag_col]};
}

// K column multicast bounding box for a (block, col): the vertical rect spanning only the block's group_rows
// rows (px is constant down the column) and the block-top sender. Identical geometry in both factories.
struct KMcastBBox {
    uint32_t ys, ye, px;
    CoreCoord sender;
};
inline KMcastBBox k_mcast_bbox(
    const std::vector<std::vector<CoreCoord>>& phys, uint32_t block_base, uint32_t col, uint32_t group_rows) {
    uint32_t ys = static_cast<uint32_t>(phys[block_base][col].y), ye = ys;
    for (uint32_t row = block_base; row < block_base + group_rows; ++row) {
        ys = std::min<uint32_t>(ys, static_cast<uint32_t>(phys[row][col].y));
        ye = std::max<uint32_t>(ye, static_cast<uint32_t>(phys[row][col].y));
    }
    return {ys, ye, static_cast<uint32_t>(phys[block_base][col].x), phys[block_base][col]};
}

// Per-device causal geometry for the block-cyclic slab layout, all in tiles. The global chunk
// [chunk_start_idx, chunk_start_idx + chunk_global) is written round-robin across the sp chips by
// update_padded_kv_cache, so chip c's Sq queries are a CONTIGUOUS logical block whose start follows the
// writer's rotation -- NOT the linear chunk_start_idx + c*Sq. Two effects of a mid-slab chunk_start_idx:
//   (a) block rotation: the starting block index (chunk_start_idx / chunk_local) can land on a chip != 0
//       (boundary_chip), rotating which chip owns which block -- so chip c's logical start is the writer's
//       update_idxt, mirroring rotated_chip_positions[c][0], not chunk_start_idx + c*chunk_local; and
//   (b) straddle: the boundary chip's Sq queries cross a slab boundary, so its causal diagonal JUMPS by
//       (chunk_global - chunk_local) tiles at q-row (chunk_local - offset).
// The linear form only misses (a) when boundary_chip != 0 -- exactly the mid-slab, non-chip-0-start case
// (e.g. the multi-turn rotated prefill). Chunk-aligned (offset == 0, boundary_chip == 0) reduces to linear.
// No block_cyclic -> plain linear. The both-axes case (cluster_axis unset, block_cyclic_chunk_local == tp*Sq)
// keeps the prior linear+straddle form. Shared by create_at (device_index from the coordinate) and override
// (stored device_index).
struct DeviceCausalGeometry {
    uint32_t chunk_start_tiles;    // global position of this device's q-row 0 (tiles)
    uint32_t straddle_q_tile;      // q-tile-row at/after which the diagonal jumps (only when this device straddles)
    uint32_t straddle_jump_tiles;  // diagonal jump in tiles (0 unless this device straddles)
};
inline DeviceCausalGeometry device_causal_geometry(
    const operation_attributes_t& args, uint32_t device_index, uint32_t tp_index, uint32_t Sq) {
    const uint32_t TW = tt::constants::TILE_WIDTH;
    if (!args.block_cyclic.has_value()) {
        // Contiguous K -> linear diagonal at chunk_start + (seq-shard rank)*Sq. The rank is device_index for an
        // SP-only seq shard; but a 2D SP×TP sub-shard whose SP axis is size-1 (e.g. QuietBox sp=1) is stored
        // as no-block-cyclic (identity permutation), and there the query is seq-sharded over the TP axis, so the
        // rank is tp_index. The two are mutually exclusive nonzero here (tp_index!=0 requires block_cyclic_sp_axis
        // set with sp==1, which forces device_index==0; no sub-shard -> tp_index==0), so their sum is the rank.
        return {(args.chunk_start_idx + (device_index + tp_index) * Sq) / TW, 0u, 0u};
    }
    const uint32_t sp = args.block_cyclic->sp;
    const uint32_t chunk_local = args.block_cyclic->chunk_local;  // cache per-shard slab width (elements)
    const uint32_t chunk_global = sp * chunk_local;

    if (args.cluster_axis.has_value()) {
        TT_FATAL(
            device_index < sp,
            "indexer_score: device_index {} out of range for block-cyclic sp={} (check cluster_axis vs "
            "block_cyclic_sp_axis)",
            device_index,
            sp);
        // Block-cyclic, named SP axis. device_index is the SP-ring index; its slab starts at the writer's
        // update_idxt (== rotated_chip_positions[device_index][0]), handling the boundary_chip rotation the
        // linear form misses. tp_index (SP×TP 2D sub-shard) selects this device's Sq-row sub-range within that
        // slab: it owns local rows [tp_index*Sq, (tp_index+1)*Sq). lr0 is its first slab-local row; the mapping
        // and straddle below are EXACT for both the SP-only case (tp_index==0, Sq==chunk_local) and the 2D case.
        const uint32_t boundary_slab = args.chunk_start_idx / chunk_global;
        const uint32_t boundary_chip = (args.chunk_start_idx / chunk_local) % sp;
        const uint32_t offset = args.chunk_start_idx % chunk_local;
        const uint32_t update_idxt = device_index < boundary_chip    ? (boundary_slab + 1) * chunk_local
                                     : device_index == boundary_chip ? boundary_slab * chunk_local + offset
                                                                     : boundary_slab * chunk_local;
        const uint32_t lr0 = update_idxt + tp_index * Sq;  // this device's first slab-local row (TP sub-offset)
        const uint32_t loff = lr0 % chunk_local;           // its offset within the current slab
        const uint32_t logical_start = (lr0 / chunk_local) * chunk_global + device_index * chunk_local + loff;
        uint32_t straddle_q_tile = 0, straddle_jump_tiles = 0;
        if (loff != 0 && loff + Sq > chunk_local) {  // this device's Sq rows cross a slab boundary
            straddle_q_tile = (chunk_local - loff) / TW;
            straddle_jump_tiles = (chunk_global - chunk_local) / TW;
        }
        return {logical_start / TW, straddle_q_tile, straddle_jump_tiles};
    }

    // Both-axes (cluster_axis unset): prior linear + within-block straddle geometry.
    const uint32_t chunk_start = args.chunk_start_idx + device_index * Sq;
    const uint32_t offset = chunk_start % chunk_local;
    uint32_t straddle_q_tile = 0, straddle_jump_tiles = 0;
    if (offset != 0 && offset + Sq > chunk_local) {
        straddle_q_tile = (chunk_local - offset) / TW;
        straddle_jump_tiles = (chunk_global - chunk_local) / TW;
    }
    return {chunk_start / TW, straddle_q_tile, straddle_jump_tiles};
}

// This device's linearized SP-ring index; 0 on a single device (no coordinate lookup needed).
inline uint32_t device_index_for(
    const operation_attributes_t& args, const ttnn::MeshCoordinate& coord, const Tensor& q) {
    if (q.device_storage().get_coords().size() <= 1) {
        return 0;
    }
    return ttnn::ccl::get_linearized_index_from_physical_coord(q, coord, args.cluster_axis);
}

// The two non-hashed runtime args derived from k's shape + the optionals. Single source for both create()
// (bakes at miss) and override_runtime_arguments() (re-patches on a hit).
struct PersistentCacheArgs {
    uint32_t k_batch_page_offset;  // cache_batch_idx * Tt * Dt; 0 when not indexed
    uint32_t kv_len_tiles;         // valid key prefix in tiles; full Tt when kv_len unset
};
inline PersistentCacheArgs persistent_cache_args(const operation_attributes_t& attrs, const Tensor& k) {
    const auto& shape = k.logical_shape();
    const uint32_t Tt = shape[2] / tt::constants::TILE_WIDTH;
    const uint32_t Dt = shape[3] / tt::constants::TILE_WIDTH;
    return {
        .k_batch_page_offset = attrs.cache_batch_idx.value_or(0) * Tt * Dt,
        .kv_len_tiles = attrs.kv_len.value_or(shape[2]) / tt::constants::TILE_WIDTH};
}

}  // namespace ttnn::operations::experimental::indexer_score::program
