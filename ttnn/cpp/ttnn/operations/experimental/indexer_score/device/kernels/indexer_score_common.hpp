// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared by reader/compute/writer: common compile-time dims, CB indices, derived tile counts, and
// WorkUnitSpan (this core's cursor over its (group-phase x k-band) rectangle; one cell = QC x up-to-KC).

#pragma once

#include <cstdint>

#include "indexer_score_cb.hpp"          // shared host/device CB-index argument layout (CbArg)
#include "indexer_score_work_split.hpp"  // shared host/device causal work-split formula

namespace iscore = ttnn::operations::experimental::indexer_score;

// Common dim args 0..8 (same in every kernel). chunk_start is NOT here -- it's a per-device RUNTIME arg,
// so distinct values reuse one program.
constexpr uint32_t num_heads = get_compile_time_arg_val(0);         // indexer heads
constexpr uint32_t q_len_tiles = get_compile_time_arg_val(1);       // q chunk rows, in tiles
constexpr uint32_t k_len_tiles = get_compile_time_arg_val(2);       // total k positions, in tiles
constexpr uint32_t head_dim_tiles = get_compile_time_arg_val(3);    // head dim, in tiles
constexpr uint32_t q_tiles_per_unit = get_compile_time_arg_val(4);  // q-tile-rows per work unit (q_chunk knob)
constexpr uint32_t k_tiles_per_unit = get_compile_time_arg_val(5);  // k tiles per work unit (k_chunk knob)
constexpr uint32_t heads_per_group = get_compile_time_arg_val(6);   // heads resident at once (head_group knob)
constexpr uint32_t num_out_groups = get_compile_time_arg_val(7);    // output groups; score [B, num_out_groups, Sq, T]
constexpr uint32_t block_tiles = get_compile_time_arg_val(8);       // block-max-pool width in k-tiles; 0 = no pooling
constexpr uint32_t num_dim_args = 9;

// Heads summed per output plane. num_out_groups==1 sums all heads into one plane; >1 partitions into
// num_out_groups groups of reduce_heads each, summed within a group (per-group planes).
constexpr uint32_t reduce_heads = num_heads / num_out_groups;

// Block-max-pool: each block_tiles consecutive k-tiles max-reduce to ONE block score, so a unit emits
// blocks_per_unit tiles (col-0 = per-query block max) instead of the full KC strip.
constexpr bool block_pool = block_tiles != 0;
constexpr uint32_t blocks_per_unit = block_pool ? (k_tiles_per_unit / block_tiles) : k_tiles_per_unit;

// CB indices, forwarded from the factory in CbArg order after the dim args.
constexpr uint32_t cb_q = get_compile_time_arg_val(num_dim_args + iscore::cb_q_arg);
constexpr uint32_t cb_k = get_compile_time_arg_val(num_dim_args + iscore::cb_k_arg);
constexpr uint32_t cb_w = get_compile_time_arg_val(num_dim_args + iscore::cb_w_arg);
constexpr uint32_t cb_mask = get_compile_time_arg_val(num_dim_args + iscore::cb_mask_arg);
constexpr uint32_t cb_qk = get_compile_time_arg_val(num_dim_args + iscore::cb_qk_arg);
constexpr uint32_t cb_acc_strip = get_compile_time_arg_val(num_dim_args + iscore::cb_acc_strip_arg);
constexpr uint32_t cb_out_strip = get_compile_time_arg_val(num_dim_args + iscore::cb_out_strip_arg);
constexpr uint32_t cb_scaler = get_compile_time_arg_val(num_dim_args + iscore::cb_scaler_arg);
constexpr uint32_t cb_pool_scratch = get_compile_time_arg_val(num_dim_args + iscore::cb_pool_scratch_arg);

// Dim args + CB indices are common to all kernels; per-kernel compile-time args start here.
constexpr uint32_t num_common_ct_args = num_dim_args + iscore::num_cb_args;

// Mask tile count, as a bare name for the kernels (defined in indexer_score_cb.hpp).
constexpr uint32_t num_mask_tiles = iscore::num_mask_tiles;

// True when heads don't all fit L1 resident, so they stream in groups.
constexpr bool stream_heads = heads_per_group < num_heads;

// Fused single-head k-column batch: score columns matmul'd per DEST acquire on the compute side, and the
// k-tile sub-chunk the reader streams+pushes so compute consumes a batch while the reader fetches the next
// (read/compute overlap). Sized to the usable DEST tile capacity in half-sync bf16 (the 16-tile DEST is
// split into two 8-tile banks so math and pack ping-pong) -> one score column per DEST reg, 8 per acquire.
// Shared so the producer (reader) and consumer (compute) granularities can't drift apart.
constexpr uint32_t mm_col_batch = 8;

// Derived tile counts for the per-unit circular-buffer blocks.
constexpr uint32_t q_group_tiles = heads_per_group * q_tiles_per_unit * head_dim_tiles;  // [QC][HB][Dt]
constexpr uint32_t w_group_tiles = num_heads * q_tiles_per_unit;                         // [QC][Hi]
constexpr uint32_t k_chunk_tiles = k_tiles_per_unit * head_dim_tiles;                    // [KC][Dt]

// Thin wrapper binding the shared work-split formula to this kernel's CT dims: unmasked prefix
// k-tiles of absolute q-tile-row q_row_abs in a unit. chunk_start_tiles is the per-device runtime
// chunk-start offset (in tiles); straddle_* carry the mid-slab boundary-chip diagonal jump (0 otherwise).
// All three are compute-kernel runtime args.
inline uint32_t row_valid_prefix(
    uint32_t q_row_abs,
    uint32_t k_tile_start,
    uint32_t k_tiles_in_unit,
    uint32_t chunk_start_tiles,
    uint32_t straddle_q_tile,
    uint32_t straddle_jump_tiles) {
    return iscore::valid_prefix_tiles(
        q_row_abs, k_tile_start, k_tiles_in_unit, chunk_start_tiles, straddle_q_tile, straddle_jump_tiles);
}

/** (group, band) cell cursor. group = absolute q-row-group index; band = absolute k-band index; the
 *  per-cell accessors feed every per-unit body (matmul / mask / untilize) identically.
 *
 *  The grid stays keyed on the COMPILE-TIME k_len_tiles (fixed). valid_k_len_tiles (runtime, <= k_len_tiles)
 *  only narrows the valid-column count per cell: bands entirely past it score nothing (k_tiles() == 0). */
struct WorkUnitSpan {
    uint32_t group = 0;
    uint32_t band = 0;
    uint32_t valid_k_len_tiles = k_len_tiles;  // populated key prefix this dispatch; default = full buffer

    void set(uint32_t g, uint32_t b) {
        group = g;
        band = b;
    }
    /** Set the runtime valid KV length (in tiles). Pass kv_len/32, or k_len_tiles for the full buffer. */
    void set_valid_k_len_tiles(uint32_t tiles) { valid_k_len_tiles = tiles; }

    uint32_t q_tile_start() const { return group * q_tiles_per_unit; }  // first q-tile-row of this cell
    uint32_t k_tile_start() const { return band * k_tiles_per_unit; }   // first k-tile of this cell
    uint32_t k_tiles() const {  // valid k-tiles in this cell: < full on the edge, 0 entirely past kv_len
        const uint32_t start = k_tile_start();
        const uint32_t left = valid_k_len_tiles > start ? valid_k_len_tiles - start : 0;
        return left < k_tiles_per_unit ? left : k_tiles_per_unit;
    }
};
