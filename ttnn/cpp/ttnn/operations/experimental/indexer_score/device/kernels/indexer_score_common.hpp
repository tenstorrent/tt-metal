// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared by indexer_score reader / compute / writer: common compile-time dims (args
// 0..7), CB indices, derived group/chunk tile counts, and WorkUnitSpan (this core's
// cursor over its (group-phase x k-band) rectangle). One cell = QC q-tile-rows x up-to-KC k-tiles.

#pragma once

#include <cstdint>

#include "indexer_score_cb.hpp"          // shared host/device CB-index argument layout (CbArg)
#include "indexer_score_work_split.hpp"  // shared host/device causal work-split formula

namespace iscore = ttnn::operations::experimental::indexer_score;

// Common dim args 0..7 (same in every kernel).
constexpr uint32_t num_heads = get_compile_time_arg_val(0);          // indexer heads
constexpr uint32_t q_len_tiles = get_compile_time_arg_val(1);        // q chunk rows, in tiles
constexpr uint32_t k_len_tiles = get_compile_time_arg_val(2);        // total k positions, in tiles
constexpr uint32_t head_dim_tiles = get_compile_time_arg_val(3);     // head dim, in tiles
constexpr uint32_t chunk_start_tiles = get_compile_time_arg_val(4);  // q chunk start offset, in tiles
constexpr uint32_t q_tiles_per_unit = get_compile_time_arg_val(5);   // q-tile-rows per work unit (q_chunk knob)
constexpr uint32_t k_tiles_per_unit = get_compile_time_arg_val(6);   // k tiles per work unit (k_chunk knob)
constexpr uint32_t heads_per_group = get_compile_time_arg_val(7);    // heads resident at once (head_group knob)
constexpr uint32_t num_dim_args = 8;

// CB indices, forwarded from the factory in CbArg order right after the dim args. Bare names so
// kernels (and their template-arg uses) read like the host-side constants did.
constexpr uint32_t cb_q = get_compile_time_arg_val(num_dim_args + iscore::cb_q_arg);
constexpr uint32_t cb_k = get_compile_time_arg_val(num_dim_args + iscore::cb_k_arg);
constexpr uint32_t cb_w = get_compile_time_arg_val(num_dim_args + iscore::cb_w_arg);
constexpr uint32_t cb_mask = get_compile_time_arg_val(num_dim_args + iscore::cb_mask_arg);
constexpr uint32_t cb_qk = get_compile_time_arg_val(num_dim_args + iscore::cb_qk_arg);
constexpr uint32_t cb_acc_strip = get_compile_time_arg_val(num_dim_args + iscore::cb_acc_strip_arg);
constexpr uint32_t cb_out_strip = get_compile_time_arg_val(num_dim_args + iscore::cb_out_strip_arg);

// Dim args + CB indices are common to all kernels; per-kernel compile-time args start here.
constexpr uint32_t num_common_ct_args = num_dim_args + iscore::num_cb_args;

// Mask tile count, as a bare name for the kernels (defined in indexer_score_cb.hpp).
constexpr uint32_t num_mask_tiles = iscore::num_mask_tiles;

// True when heads don't all fit L1 resident, so they stream in groups.
constexpr bool stream_heads = heads_per_group < num_heads;

// Derived tile counts for the per-unit circular-buffer blocks.
constexpr uint32_t q_group_tiles = heads_per_group * q_tiles_per_unit * head_dim_tiles;  // [QC][HB][Dt]
constexpr uint32_t w_group_tiles = num_heads * q_tiles_per_unit;                         // [QC][Hi]
constexpr uint32_t k_chunk_tiles = k_tiles_per_unit * head_dim_tiles;                    // [KC][Dt]

// Thin wrapper binding the shared work-split formula to this kernel's CT dims: unmasked prefix
// k-tiles of absolute q-tile-row q_row_abs in a unit (bound to this kernel's chunk_start_tiles).
inline uint32_t row_valid_prefix(uint32_t q_row_abs, uint32_t k_tile_start, uint32_t k_tiles_in_unit) {
    return iscore::valid_prefix_tiles(q_row_abs, k_tile_start, k_tiles_in_unit, chunk_start_tiles);
}

/** (group, band) cell cursor. The generalized scheduler maps groups -> grid rows and k-bands ->
 *  grid columns; each core walks its own (group-phase x band) rectangle and sets the cursor per cell.
 *  group = absolute q-row-group index; band = absolute k-band index. The per-cell accessors below feed
 *  every per-unit body (matmul / mask / untilize) identically, independent of how the grid was tiled. */
struct WorkUnitSpan {
    uint32_t group = 0;
    uint32_t band = 0;

    void set(uint32_t g, uint32_t b) {
        group = g;
        band = b;
    }

    uint32_t q_tile_start() const { return group * q_tiles_per_unit; }  // first q-tile-row of this cell
    uint32_t k_tile_start() const { return band * k_tiles_per_unit; }   // first k-tile of this cell
    uint32_t k_tiles() const {                                          // valid k-tiles (< full on the edge band)
        uint32_t left = k_len_tiles - k_tile_start();
        return left < k_tiles_per_unit ? left : k_tiles_per_unit;
    }
};
