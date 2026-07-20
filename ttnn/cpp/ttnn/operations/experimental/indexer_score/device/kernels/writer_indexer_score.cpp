// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer for indexer_score: drains compute's output and scatters it row-major (page = row; future keys
// pre-stamped -inf). Loops num_out_groups planes (page offset g*Sq). block_size==0: scatter each untilized
// KC strip's 32 rows. block_size>0: extract per-query block maxes from the pooled tiles' col 0, force each
// query's own block to +inf, scatter (forced-local block / sparse_local_block).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include <tt-metalium/constants.hpp>

#include "indexer_score_common.hpp"  // shared CB indices, compile-time dims, work-unit walk

constexpr uint32_t page_bytes = get_compile_time_arg_val(num_common_ct_args);  // row-major page = T*2 bytes

constexpr uint32_t frag_bytes = tt::constants::TILE_WIDTH * sizeof(uint16_t);  // one bf16 tile row

/** Scatter one KC-wide strip into 32 output rows from page `page_row_start`, col tile k_tile_start (1
 *  async_write/row). `valid_w` = in-bounds tiles (< KC on a partial last unit); the CB always pops full KC. */
template <typename OutAcc>
inline void write_strip(
    Noc noc, const OutAcc& out_acc, uint32_t page_row_start, uint32_t k_tile_start, uint32_t valid_w) {
    CircularBuffer cb(cb_out_strip);
    cb.wait_front(k_tiles_per_unit);  // always pop the full KC strip
    if (valid_w != 0) {               // 0 when the cell is entirely past valid kv_len
        uint32_t src = cb.get_read_ptr();
        const uint32_t row_pitch = k_tiles_per_unit * frag_bytes;  // strip row stride (full KC)
        const uint32_t write_bytes = valid_w * frag_bytes;         // only in-bounds columns
        for (uint32_t rr = 0; rr < tt::constants::TILE_HEIGHT; ++rr) {
            noc.async_write(
                CoreLocalMem<uint32_t>(src),
                out_acc,
                write_bytes,
                {},
                {.page_id = page_row_start + rr, .offset_bytes = k_tile_start * frag_bytes});
            src += row_pitch;
        }
        noc.async_write_barrier();
    }
    cb.pop_front(k_tiles_per_unit);
}

// ---- block-max-pool output (block_size>0) -----------------------------------------------------------
// Compute pushes blocks_per_unit tiles per q-tile-row, each tile's COLUMN 0 holding that block's per-query
// max. A 32x32 bf16 tile is four 16x16 faces; col 0 is in the left faces, so row R's col-0 datum is at
// (R/16)*POOL_FACE_ROW_STRIDE + (R%16)*FACE_WIDTH (uint16 elements).
constexpr uint32_t POOL_FACE_ROWS = tt::constants::TILE_HEIGHT / tt::constants::FACE_HEIGHT;
constexpr uint32_t POOL_FACE_ROW_STRIDE =
    (tt::constants::TILE_WIDTH / tt::constants::FACE_WIDTH) * tt::constants::FACE_HEIGHT * tt::constants::FACE_WIDTH;
constexpr uint32_t POOL_TILE_HW = tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH;
constexpr uint16_t POOL_POS_INF_BF16 = 0x7F80;                                                  // +inf in bf16
constexpr uint32_t POOL_BLOCK_KEYS = block_pool ? block_tiles * tt::constants::TILE_WIDTH : 1;  // 1: avoid /0 codegen

/** Gather each pooled tile's col-0 into a query-major [TILE_HEIGHT][valid_blocks] scratch, force each
 *  query's own block to +inf (forced-local / sparse_local_block), then write each query row's run once.
 *  `q_seq_row0` = sequence-local index of this tile-row's query 0 (within Sq). */
template <typename OutAcc>
inline void write_pooled_strip(
    Noc noc,
    const OutAcc& out_acc,
    uint32_t page_row_start,
    uint32_t q_seq_row0,
    uint32_t col_off_blocks,
    uint32_t valid_blocks,
    uint32_t chunk_start_keys,
    uint32_t straddle_q_keys,
    uint32_t straddle_jump_keys) {
    CircularBuffer cb(cb_out_strip);
    cb.wait_front(blocks_per_unit);
    volatile tt_l1_ptr uint16_t* src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb.get_read_ptr());

    CircularBuffer scratch_cb(cb_pool_scratch);
    const uint32_t scratch_addr = scratch_cb.get_write_ptr();
    uint16_t* scratch = reinterpret_cast<uint16_t*>(scratch_addr);  // query-major

    for (uint32_t b = 0; b < valid_blocks; ++b) {
        volatile tt_l1_ptr uint16_t* tile = src + b * POOL_TILE_HW;
        uint32_t qrow = 0;
        for (uint32_t fr = 0; fr < POOL_FACE_ROWS; ++fr) {
            const uint32_t face_base = fr * POOL_FACE_ROW_STRIDE;
            for (uint32_t rr = 0; rr < tt::constants::FACE_HEIGHT; ++rr) {
                scratch[qrow * valid_blocks + b] = tile[face_base + rr * tt::constants::FACE_WIDTH];  // col 0, row qrow
                ++qrow;
            }
        }
    }

    // Forced-local block (sparse_local_block=1): force each query's own block to +inf so top-k always
    // keeps it. Query (q_seq_row0 + rr)'s block = q_pos / POOL_BLOCK_KEYS, where q_pos is the diagonal key
    // position -- causal_diag_tile evaluated in KEYS (all args key units; the straddle jump is applied for
    // the mid-slab boundary chip, 0 otherwise so the common case is unchanged). Stamp only when it lands in
    // this unit's [col_off_blocks, +valid).
    for (uint32_t rr = 0; rr < tt::constants::TILE_HEIGHT; ++rr) {
        const uint32_t q_seq = q_seq_row0 + rr;
        const uint32_t q_pos = iscore::causal_diag_tile(q_seq, chunk_start_keys, straddle_q_keys, straddle_jump_keys);
        const uint32_t local_block = q_pos / POOL_BLOCK_KEYS;
        if (local_block >= col_off_blocks && local_block < col_off_blocks + valid_blocks) {
            scratch[rr * valid_blocks + (local_block - col_off_blocks)] = POOL_POS_INF_BF16;
        }
    }

    const uint32_t row_bytes = valid_blocks * sizeof(uint16_t);
    const uint32_t col_off_bytes = col_off_blocks * sizeof(uint16_t);
    for (uint32_t rr = 0; rr < tt::constants::TILE_HEIGHT; ++rr) {
        noc.async_write(
            CoreLocalMem<uint32_t>(scratch_addr + rr * row_bytes),
            out_acc,
            row_bytes,
            {},
            {.page_id = page_row_start + rr, .offset_bytes = col_off_bytes});
    }
    noc.async_write_barrier();  // drain before scratch / the CB page is reused
    cb.pop_front(blocks_per_unit);
}

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    // Banded schedule (matches reader/compute): group-phase x band rectangle.
    const uint32_t row_group0 = get_arg_val<uint32_t>(1);
    const uint32_t group_stride = get_arg_val<uint32_t>(2);
    const uint32_t num_groups = get_arg_val<uint32_t>(3);
    const uint32_t band0 = get_arg_val<uint32_t>(4);
    const uint32_t num_bands = get_arg_val<uint32_t>(5);
    // [6] max_bands (unused). [7] kv_len_tiles caps columns written per cell (full when unset).
    const uint32_t kv_len_tiles = get_arg_val<uint32_t>(7);
    // [8] per-device chunk-start (tiles); runtime so distinct values reuse one program. Only the block-pool
    // forced-local stamp uses it; always set.
    const uint32_t chunk_start_keys = get_arg_val<uint32_t>(8) * tt::constants::TILE_WIDTH;
    // [9],[10] mid-slab boundary-chip forced-local block jump (keys); both 0 off the boundary chip.
    const uint32_t straddle_q_keys = get_arg_val<uint32_t>(9) * tt::constants::TILE_WIDTH;
    const uint32_t straddle_jump_keys = get_arg_val<uint32_t>(10) * tt::constants::TILE_WIDTH;

    constexpr auto out_args = TensorAccessorArgs<num_common_ct_args + 1>();
    const auto out_acc = TensorAccessor(out_args, out_addr, page_bytes);

    Noc noc;

    WorkUnitSpan span;
    span.set_valid_k_len_tiles(kv_len_tiles);

    // Output [B, num_out_groups, Sq, T]: plane g occupies rows [g*Sq, (g+1)*Sq). Compute pushes
    // num_out_groups * QC strips per cell in g-major order; drain them the same way.
    constexpr uint32_t sq_rows = q_len_tiles * tt::constants::TILE_HEIGHT;  // rows per plane (Sq)

    for (uint32_t phase = 0; phase < num_groups; ++phase) {
        const uint32_t group = row_group0 + phase * group_stride;
        for (uint32_t band_i = 0; band_i < num_bands; ++band_i) {
#ifdef FUSED_RING
            // Reordered band-visit order, IDENTICAL to reader/compute (perm offsets at rt slots 11.., see factory).
            const uint32_t band = get_arg_val<uint32_t>(11 + band_i);
#else
            const uint32_t band = band_i;
#endif
            span.set(group, band0 + band);
            const uint32_t k_tile0 = span.k_tile_start();
            const uint32_t valid_w = span.k_tiles();  // == KC for interior bands, < KC for a partial last band
            // block-pool: this band's slice starts at block-column k_tile0/block_tiles, width valid_blocks.
            const uint32_t col_off_blocks = block_pool ? (k_tile0 / block_tiles) : 0;
            const uint32_t valid_blocks = block_pool ? (valid_w / block_tiles) : 0;  // == blocks_per_unit (no partial)
            for (uint32_t g = 0; g < num_out_groups; ++g) {
                const uint32_t plane_row0 = g * sq_rows;
                for (uint32_t q_row = 0; q_row < q_tiles_per_unit; ++q_row) {
                    const uint32_t q_seq_row0 = (span.q_tile_start() + q_row) * tt::constants::TILE_HEIGHT;  // within Sq
                    const uint32_t page_row_start = plane_row0 + q_seq_row0;
                    if constexpr (block_pool) {
                        write_pooled_strip(
                            noc,
                            out_acc,
                            page_row_start,
                            q_seq_row0,
                            col_off_blocks,
                            valid_blocks,
                            chunk_start_keys,
                            straddle_q_keys,
                            straddle_jump_keys);
                    } else {
                        write_strip(noc, out_acc, page_row_start, k_tile0, valid_w);
                    }
                }
            }
        }
    }
}
