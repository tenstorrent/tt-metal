// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer for indexer_score. Drains compute's output and scatters it into the row-major output (page = row,
// one contiguous run per row); future keys are already stamped to -inf, so there is no row-tail fill.
// Outer loop over num_out_groups output planes (page offset g*Sq) for the per-GQA-group M3 path.
//   block_size==0: pop untilized bf16 strips and scatter each strip's 32 rows (DeepSeek/GLM, M3-token).
//   block_size>0 : pop the block-max-pooled tiles, extract per-query block maxes from tile column 0 into a
//                  single-tile scratch, force each query's own (local) block to +inf, and scatter each
//                  query row's blocks_per_unit-wide slice (M3 blocks + sparse_local_block=1).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include <tt-metalium/constants.hpp>

#include "indexer_score_common.hpp"  // shared CB indices, compile-time dims, work-unit walk

constexpr uint32_t page_bytes = get_compile_time_arg_val(num_common_ct_args);  // row-major page = T*2 bytes

constexpr uint32_t frag_bytes = tt::constants::TILE_WIDTH * sizeof(uint16_t);  // one bf16 tile row

/** Scatter one strip into 32 consecutive output rows starting at page `page_row_start`, column tile
 *  k_tile_start. Strip is always KC tiles wide; each row written as ONE contiguous run (1 async_write/row,
 *  not KC fragments). `valid_w` = KC tiles inside T (< KC on a partial last unit; KC need not divide Tt).
 *  CB always pops full KC; only the in-bounds prefix is written. The output is [B, num_out_groups, Sq, T]
 *  flattened to rows, so the caller folds the group plane into page_row_start. */
template <typename OutAcc>
inline void write_strip(
    Noc noc, const OutAcc& out_acc, uint32_t page_row_start, uint32_t k_tile_start, uint32_t valid_w) {
    CircularBuffer cb(cb_out_strip);
    cb.wait_front(k_tiles_per_unit);
    uint32_t src = cb.get_read_ptr();
    const uint32_t row_pitch = k_tiles_per_unit * frag_bytes;  // strip row stride (full KC)
    const uint32_t write_bytes = valid_w * frag_bytes;         // only the in-bounds columns
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
    cb.pop_front(k_tiles_per_unit);
}

// ---- block-max-pool output (block_size>0) -----------------------------------------------------------
// Compute pushes, per q-tile-row, blocks_per_unit tilized tiles whose COLUMN 0 holds that block's
// per-query max (rows = the 32 queries). The output is row-major [B, G, Sq, T/block_size], so this core's
// unit owns the contiguous block-column slice [unit*blocks_per_unit, +blocks_per_unit) of each query row.
// bf16 tile face layout: a 32x32 tile is four 16x16 faces in [TL,TR,BL,BR] order; tile col 0 lives in the
// left faces, so logical row R's col-0 datum is at face_row*FACE_ROW_STRIDE + (R%16)*FACE_W.
constexpr uint32_t POOL_FACE_H = tt::constants::FACE_HEIGHT;                                 // 16
constexpr uint32_t POOL_FACE_W = tt::constants::FACE_WIDTH;                                  // 16
constexpr uint32_t POOL_FACE_ROWS = tt::constants::TILE_HEIGHT / POOL_FACE_H;                // 2
constexpr uint32_t POOL_FACES_PER_ROW = tt::constants::TILE_WIDTH / POOL_FACE_W;             // 2
constexpr uint32_t POOL_FACE_ROW_STRIDE = POOL_FACES_PER_ROW * (POOL_FACE_H * POOL_FACE_W);  // 512 (uint16)
constexpr uint32_t POOL_TILE_HW = tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH;    // 1024 (uint16/tile)

// MiniMax M3 forced-local block (sparse_local_block=1): a query's own block is always selected, so its
// block score is forced to +inf (downstream top-k then always keeps it). +inf > every real/-inf score, so
// stamping it after the pool is correct regardless of the block's visible max. block_size in keys and the
// global key position of q-row 0 (chunk_start) are needed to map a query to its own block column.
constexpr uint16_t POOL_POS_INF_BF16 = 0x7F80;                                                  // +inf in bf16
constexpr uint32_t POOL_BLOCK_KEYS = block_pool ? block_tiles * tt::constants::TILE_WIDTH : 1;  // 1: avoid /0 codegen

/** Scatter one q-tile-row's pooled blocks into the row-major output. Extract column 0 of each of the
 *  blocks_per_unit tiles (one bf16 value per query row) into a query-major [TILE_HEIGHT][valid_blocks]
 *  scratch, force each query's own (local) block to +inf, then write each query row's valid_blocks-wide
 *  run once (16 B-aligned: validate guarantees blocks_per_unit % 8 == 0 and no partial unit, so
 *  valid_blocks == blocks_per_unit). `q_seq_row0` is the sequence-local index of this tile-row's query 0
 *  (within Sq, plane offset excluded), used to find each query's own block. */
template <typename OutAcc>
inline void write_pooled_strip(
    Noc noc,
    const OutAcc& out_acc,
    uint32_t page_row_start,
    uint32_t q_seq_row0,
    uint32_t col_off_blocks,
    uint32_t valid_blocks,
    uint32_t chunk_start_keys) {
    CircularBuffer cb(cb_out_strip);
    cb.wait_front(blocks_per_unit);
    volatile tt_l1_ptr uint16_t* src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb.get_read_ptr());

    CircularBuffer scratch_cb(cb_pool_scratch);
    const uint32_t scratch_addr = scratch_cb.get_write_ptr();
    uint16_t* scratch = reinterpret_cast<uint16_t*>(scratch_addr);  // query-major [TILE_HEIGHT][valid_blocks]

    for (uint32_t b = 0; b < valid_blocks; ++b) {
        volatile tt_l1_ptr uint16_t* tile = src + b * POOL_TILE_HW;
        uint32_t qrow = 0;
        for (uint32_t fr = 0; fr < POOL_FACE_ROWS; ++fr) {
            const uint32_t face_base = fr * POOL_FACE_ROW_STRIDE;
            for (uint32_t rr = 0; rr < POOL_FACE_H; ++rr) {
                scratch[qrow * valid_blocks + b] = tile[face_base + rr * POOL_FACE_W];  // tile col 0, logical row qrow
                ++qrow;
            }
        }
    }

    // Forced-local block: query (q_seq_row0 + rr) sits at global key position chunk_start_keys +
    // q_seq_row0 + rr, so its own block is that / POOL_BLOCK_KEYS. chunk_start_keys is the runtime
    // per-device causal chunk start in KEYS (cb_offset's tiles * TILE_WIDTH), or the compile-time
    // constant when no chunk_offset is bound. The 32 queries of a tile-row can straddle
    // a block boundary, so this is per-query, not per-tile-row. Only this unit owns block columns
    // [col_off_blocks, +valid_blocks); the query's local block lands in exactly one unit, so stamp only when
    // it falls in this slice (other units skip it).
    for (uint32_t rr = 0; rr < tt::constants::TILE_HEIGHT; ++rr) {
        const uint32_t local_block = (chunk_start_keys + q_seq_row0 + rr) / POOL_BLOCK_KEYS;
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
    noc.async_write_barrier();  // drain before scratch is reused by the next q-tile-row / CB page is recycled
    cb.pop_front(blocks_per_unit);
}

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t flat_start = get_arg_val<uint32_t>(1);
    const uint32_t flat_count = get_arg_val<uint32_t>(2);

    constexpr auto out_args = TensorAccessorArgs<num_common_ct_args + 1>();
    const auto out_acc = TensorAccessor(out_args, out_addr, page_bytes);

    Noc noc;

    // Per-device causal chunk start (in tiles), filled once by the reader into cb_offset (DRAM-read from
    // the optional chunk_offset tensor, else the compile-time chunk_start_tiles constant). Only the
    // block-pool forced-local-block stamp needs it; read once, never popped (compute also cb_wait_front's
    // the same depth-1 push -- non-consuming for both). chunk_start_keys = tiles * TILE_WIDTH.
    uint32_t chunk_start_keys = 0;
    if constexpr (block_pool) {
        CircularBuffer(cb_offset).wait_front(1);
        const uint32_t chunk_start_tiles_rt =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(CircularBuffer(cb_offset).get_read_ptr())[0];
        chunk_start_keys = chunk_start_tiles_rt * tt::constants::TILE_WIDTH;
    }

    WorkUnitSpan span;
    span.start(flat_start);

    // Output is [B, num_out_groups, Sq, T]: plane g occupies rows [g*Sq, (g+1)*Sq). Compute pushes
    // num_out_groups * QC strips per unit in g-major order, so drain them the same way.
    constexpr uint32_t sq_rows = q_len_tiles * tt::constants::TILE_HEIGHT;  // rows per output plane (Sq)

    for (uint32_t i = 0; i < flat_count; ++i) {
        const uint32_t k_tile0 = span.k_tile_start();
        const uint32_t valid_w = span.k_tiles();  // == KC for interior units, < KC for a partial last unit
        // block_size==0: one KC-wide untilized strip per row (write only valid_w columns). block-pool: each
        // unit contributes a blocks_per_unit-wide slice starting at block-column (k_tile0/block_tiles)
        // (write_pooled_strip converts this block-column offset to bytes).
        const uint32_t col_off_blocks = block_pool ? (k_tile0 / block_tiles) : 0;
        const uint32_t valid_blocks = block_pool ? (valid_w / block_tiles) : 0;  // == blocks_per_unit (no partial unit)
        for (uint32_t g = 0; g < num_out_groups; ++g) {
            const uint32_t plane_row0 = g * sq_rows;
            for (uint32_t r = 0; r < q_tiles_per_unit; ++r) {
                const uint32_t q_seq_row0 = (span.q_tile_start() + r) * tt::constants::TILE_HEIGHT;  // within Sq
                const uint32_t page_row_start = plane_row0 + q_seq_row0;
                if constexpr (block_pool) {
                    write_pooled_strip(
                        noc, out_acc, page_row_start, q_seq_row0, col_off_blocks, valid_blocks, chunk_start_keys);
                } else {
                    write_strip(noc, out_acc, page_row_start, k_tile0, valid_w);
                }
            }
        }
        span.advance();
    }
}
