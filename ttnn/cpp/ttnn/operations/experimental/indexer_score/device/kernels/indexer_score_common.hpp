// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared by indexer_score reader / compute / writer: common compile-time dims (args
// 0..7), CB indices, derived group/chunk tile counts, and WorkUnitSpan (this core's
// walk over its flat span of units). One unit = QC q-tile-rows x up-to-KC k-tiles.

#pragma once

#include <cstdint>

#include "indexer_score_cb.hpp"          // shared host/device circular-buffer indices
#include "indexer_score_work_split.hpp"  // shared host/device causal work-split formula

// Re-export shared CB indices to global scope so kernels use bare names.
using ttnn::operations::experimental::indexer_score::cb_acc_strip;
using ttnn::operations::experimental::indexer_score::cb_k;
using ttnn::operations::experimental::indexer_score::cb_mask;
using ttnn::operations::experimental::indexer_score::cb_out_strip;
using ttnn::operations::experimental::indexer_score::cb_q;
using ttnn::operations::experimental::indexer_score::cb_qk;
using ttnn::operations::experimental::indexer_score::cb_scratch;
using ttnn::operations::experimental::indexer_score::cb_w;
using ttnn::operations::experimental::indexer_score::num_mask_tiles;

constexpr uint32_t num_heads = get_compile_time_arg_val(0);          // indexer heads
constexpr uint32_t q_len_tiles = get_compile_time_arg_val(1);        // q chunk rows, in tiles
constexpr uint32_t k_len_tiles = get_compile_time_arg_val(2);        // total k positions, in tiles
constexpr uint32_t head_dim_tiles = get_compile_time_arg_val(3);     // head dim, in tiles
constexpr uint32_t chunk_start_tiles = get_compile_time_arg_val(4);  // q chunk start offset, in tiles
constexpr uint32_t q_tiles_per_unit = get_compile_time_arg_val(5);   // q-tile-rows per work unit (q_chunk knob)
constexpr uint32_t k_tiles_per_unit = get_compile_time_arg_val(6);   // k tiles per work unit (k_chunk knob)
constexpr uint32_t heads_per_group = get_compile_time_arg_val(7);    // heads resident at once (head_group knob)

// Args 0..7 are common to all kernels; per-kernel compile-time args start here.
constexpr uint32_t num_common_ct_args = 8;

// True when heads don't all fit L1 resident, so they stream in groups.
constexpr bool stream_heads = heads_per_group < num_heads;

// Derived tile counts for the per-unit circular-buffer blocks.
constexpr uint32_t q_group_tiles = heads_per_group * q_tiles_per_unit * head_dim_tiles;  // [QC][HB][Dt]
constexpr uint32_t w_group_tiles = num_heads * q_tiles_per_unit;                         // [QC][Hi]
constexpr uint32_t k_chunk_tiles = k_tiles_per_unit * head_dim_tiles;                    // [KC][Dt]

// Thin wrappers binding the shared work-split formula to this kernel's CT dims.
namespace ws = ttnn::operations::experimental::indexer_score;

/** Valid k-tiles of q-row-group `group` = causal-valid k-tiles of its (widest) last row. */
inline uint32_t group_valid_k_tiles(uint32_t group) {
    return ws::valid_k_tiles_in_group(group, q_tiles_per_unit, chunk_start_tiles, k_len_tiles);
}

/** Unmasked prefix k-tiles of absolute q-tile-row q_row_abs in a unit (bound to this kernel's
 *  chunk_start_tiles). */
inline uint32_t row_valid_prefix(uint32_t q_row_abs, uint32_t k_tile_start, uint32_t k_tiles_in_unit) {
    return ws::valid_prefix_tiles(q_row_abs, k_tile_start, k_tiles_in_unit, chunk_start_tiles);
}

/** Number of k-chunk units in q-row-group `group`. */
inline uint32_t units_in_group(uint32_t group) {
    return ws::units_in_group(group, q_tiles_per_unit, k_tiles_per_unit, chunk_start_tiles, k_len_tiles);
}

/** (group, unit) cursor over causal-valid work units, starting at a flat index. */
struct WorkUnitSpan {
    uint32_t group = 0;
    uint32_t unit = 0;

    void start(uint32_t flat) {
        uint32_t rowsum = 0;
        while (flat >= rowsum + units_in_group(group)) {
            rowsum += units_in_group(group);
            ++group;
        }
        unit = flat - rowsum;
    }

    /** Advance one unit; true when a new q-row-group begins. */
    bool advance() {
        if (++unit == units_in_group(group)) {
            ++group;
            unit = 0;
            return true;
        }
        return false;
    }

    uint32_t q_tile_start() const { return group * q_tiles_per_unit; }  // first q-tile-row of this unit
    uint32_t k_tile_start() const { return unit * k_tiles_per_unit; }   // first k-tile of this unit
    uint32_t k_tiles() const {                                          // valid k-tiles in this unit (< full on edge)
        uint32_t left = group_valid_k_tiles(group) - k_tile_start();
        return left < k_tiles_per_unit ? left : k_tiles_per_unit;
    }
    uint32_t valid_k_tiles() const {
        return group_valid_k_tiles(group);
    }  // compute fills [0, this) for every group row
    bool last_in_group() const { return unit == units_in_group(group) - 1; }
};
