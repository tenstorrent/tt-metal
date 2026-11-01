// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm reader (NCRISC / NoC0).
//
// The gamma slice is loaded by the writer (rms_norm_writer.cpp), off this kernel's x stream.
// Per block: load_x_block
//   TILE: one cb_reserve_back per block; per tile-row: core_w_tiles reads, one barrier, W-tail zero fill
//         (last W member only), one cb_push_back of x_row_pitch pages (the gap pages, if any, are never read).
//   RM  : per (tile-row, chunk) one reserve/barrier/push of rm_chunk_w_tiles stick-piece pages in cb_x_rm;
//         full chunks use read_sticks_for_tilize, the W-tail chunk is read raw + zero-filled to its tile edge.
// RMS_SHARDED (WIDTH / BLOCK): the input shard is resident in this core's L1 and is never re-read through an
// accessor.
//   TILE: cb_x IS the shard (zero-copy CB) — the reader only publishes its pages to compute. The W tail is not
//         zeroed in place (the caller's shard is never mutated): the tail member emits a 0/1 row mask
//         (prepare_reduce_mask) that compute applies to the tail tile's x^2 inside the collapse reduce.
//   RM  : per (tile-row, chunk) the valid stick pieces are copied locally (own-L1 NoC reads) from the shard into
//         the cb_x_rm tilize staging, zero-filled to the tile edge (covers the W tail and shard padding): one read
//         per tile-row when the shard pitch equals the staging pitch, else stateful per-stick reads; single-chunk
//         shards zero each staging slot's tail only on the slot's first use.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "rms_norm_common.hpp"

using namespace rms_norm_cb;

constexpr uint32_t tensor_w_tiles = get_compile_time_arg_val(0);
constexpr uint32_t W = get_compile_time_arg_val(1);
constexpr uint32_t H = get_compile_time_arg_val(2);
constexpr uint32_t image_row_tiles = get_compile_time_arg_val(3);
constexpr uint32_t x_row_pitch = get_compile_time_arg_val(4);
constexpr uint32_t rm_chunk_w_tiles = get_compile_time_arg_val(5);
constexpr uint32_t x_page_bytes = get_compile_time_arg_val(6);
constexpr uint32_t elem_bytes = get_compile_time_arg_val(7);
constexpr uint32_t x_block_depth = get_compile_time_arg_val(8);
constexpr auto x_args = TensorAccessorArgs<9>();

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t FACE_DIM = 16;
constexpr uint32_t w_valid_in_last_tile = W - TILE_DIM * (tensor_w_tiles - 1);

// Zero columns [v, 32) of a tile in L1 (v = valid datums in the tile's row, 1..31).
// Whole faces go through the NoC zero engine; partial face rows (< 16 datums each) are RISC-V stores.
FORCE_INLINE void zero_w_tail_of_tile(const Noc& noc, const CircularBuffer& cb, uint32_t tile_offset_bytes) {
    constexpr uint32_t face_bytes = FACE_DIM * FACE_DIM * elem_bytes;
    constexpr uint32_t face_row_bytes = FACE_DIM * elem_bytes;
    constexpr uint32_t v = w_valid_in_last_tile;
    const uint32_t tile_addr = cb.get_write_ptr() + tile_offset_bytes;
    if constexpr (v <= FACE_DIM) {
        noc.async_write_zeros(cb, face_bytes, {.offset_bytes = tile_offset_bytes + face_bytes});      // face 1
        noc.async_write_zeros(cb, face_bytes, {.offset_bytes = tile_offset_bytes + 3 * face_bytes});  // face 3
        if constexpr (v < FACE_DIM) {
            for (uint32_t f = 0; f < 4; f += 2) {  // faces 0 and 2: cols v..15 of every row
                for (uint32_t row = 0; row < FACE_DIM; ++row) {
                    dataflow_kernel_lib::fill_l1_range<elem_bytes>(
                        tile_addr + f * face_bytes + row * face_row_bytes + v * elem_bytes,
                        (FACE_DIM - v) * elem_bytes,
                        0);
                }
            }
        }
        noc.write_zeros_l1_barrier();
    } else {
        for (uint32_t f = 1; f < 4; f += 2) {  // faces 1 and 3: cols (v-16)..15 of every row
            for (uint32_t row = 0; row < FACE_DIM; ++row) {
                dataflow_kernel_lib::fill_l1_range<elem_bytes>(
                    tile_addr + f * face_bytes + row * face_row_bytes + (v - FACE_DIM) * elem_bytes,
                    (TILE_DIM - v) * elem_bytes,
                    0);
            }
        }
    }
}

// Perf ablation hooks (perf probes only; never set by the op): RMS_ABLATE_EMPTY returns at once (fixed-cost
// floor of the same program on the same Tensix cores); RMS_ABLATE_DM drops the interleaved TILE x / out NoC
// payload but keeps every CB handshake and barrier.
#ifdef RMS_ABLATE_DM
constexpr bool ablate_dm = true;
#else
constexpr bool ablate_dm = false;
#endif

void kernel_main() {
#ifdef RMS_ABLATE_EMPTY
    return;
#endif
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start = get_arg_val<uint32_t>(1);  // first global tile-row of this core's group
    const uint32_t num_blocks = get_arg_val<uint32_t>(2);
    const uint32_t block_row_tiles = get_arg_val<uint32_t>(3);
    const uint32_t last_block_row_tiles = get_arg_val<uint32_t>(4);
    const uint32_t w_tile_start = get_arg_val<uint32_t>(5);
    const uint32_t core_w_tiles = get_arg_val<uint32_t>(6);
    [[maybe_unused]] const uint32_t chunks_this = get_arg_val<uint32_t>(7);
    [[maybe_unused]] const uint32_t valid_cols = get_arg_val<uint32_t>(8);    // valid columns of the slice
    [[maybe_unused]] const uint32_t valid_sticks = get_arg_val<uint32_t>(9);  // RM sharded: valid sticks in shard
    [[maybe_unused]] const uint32_t owns_tail = get_arg_val<uint32_t>(10);    // sharded TILE: owns the W tail tile

    // 1.0 scaler of compute's FPU row collapse (reduce_tile ROW). Built after block 0 of x is published —
    // compute needs it only after pass 1 of block 0 — so it never delays x.
    auto prep_ones = [](uint32_t blk) {
        if (blk == 0) {
            dataflow_kernel_lib::
                calculate_and_prepare_reduce_scaler<cb_ones, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
        }
    };
#ifdef RMS_MASK_TAIL
    // The tail tile's x^2 is masked (columns >= w_valid_in_last_tile) inside compute's collapse reduce.
    if (owns_tail) {
        dataflow_kernel_lib::prepare_reduce_mask<cb_tail_mask, ckernel::ReduceDim::REDUCE_ROW>(w_valid_in_last_tile);
    }
#endif

    // The member owning the global last W tile zero-fills its columns >= W (padding never enters sum x^2).
    [[maybe_unused]] const bool owns_w_tail =
        (w_valid_in_last_tile < TILE_DIM) && (w_tile_start + core_w_tiles == tensor_w_tiles);

#if defined(RMS_SHARDED) && defined(RMS_RM)
    // ---- RM shard: local stick copy (own L1) into the tilize staging, zero-filled to the tile edge ----
    // Helper bypass (documented, Perf 1 changelog "Helper bypasses"): dataflow_kernel_lib's local copy
    // (local_copy_helpers_dataflow.hpp set_read_state / read_with_state) fixes the SOURCE and varies only the
    // destination; this staging steps both a shard-stick source pitch and a staging-row destination pitch, so the raw
    // one_packet set_state / with_state pair is used (wid_rm_2048x256: per-stick NoC 163.6 us -> stateful 157.1 us).
    // Narrow pieces are transaction-count-bound (~20 ns per local NoC read on BH; a RISC-V word copy is slower
    // still at every legal piece width >= 16 B). So: shard row pitch == staging pitch and a full piece -> ONE
    // read per (tile-row, chunk); otherwise one read per stick through the one-packet set_state / with_state pair
    // (NoC coordinates + length programmed once per tile-row).
    // Tail zero fill: single-chunk shards write pieces of one width, and nothing else writes cb_x_rm, so a slot's
    // tail columns stay zero once filled -> zeroed on the slot's first use only (the per-stick fill was ~60 % of
    // the staging time on 16 B pieces). A ragged tile-row (nsticks < 32) is the shard's last, so its slot is never
    // reused after a short fill.
    constexpr uint32_t chunk_row_bytes = rm_chunk_w_tiles * TILE_DIM * elem_bytes;
    constexpr uint32_t tile_row_bytes = TILE_DIM * elem_bytes;
    static_assert(chunk_row_bytes <= NOC_MAX_BURST_SIZE, "a stick piece must be one NoC packet");
    const uint32_t stage_slots = get_local_cb_interface(cb_x_rm).fifo_size / (TILE_DIM * chunk_row_bytes);
    uint32_t reservations = 0;
    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        const uint32_t rows = (blk + 1 == num_blocks) ? last_block_row_tiles : block_row_tiles;
        MaybeDeviceZoneScope("rd_block");  // occupancy: local stick copies + back-pressure on cb_x_rm
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t s0 = (blk * block_row_tiles + r) * TILE_DIM;  // first stick of the tile-row in the shard
            const uint32_t nsticks = (valid_sticks - s0) < TILE_DIM ? (valid_sticks - s0) : TILE_DIM;
            for (uint32_t c = 0; c < chunks_this; ++c) {
                const uint32_t c0 = c * rm_chunk_w_tiles * TILE_DIM;  // first column of the chunk in the shard
                const uint32_t cols =
                    (valid_cols - c0) < rm_chunk_w_tiles * TILE_DIM ? (valid_cols - c0) : rm_chunk_w_tiles * TILE_DIM;
                const uint32_t bytes = cols * elem_bytes;
                cb_reserve_back(cb_x_rm, rm_chunk_w_tiles);
                const uint32_t l1 = get_write_ptr(cb_x_rm);
                const uint32_t src = x_addr + s0 * x_page_bytes + c0 * elem_bytes;
                if (x_page_bytes == chunk_row_bytes && bytes == chunk_row_bytes) {
                    noc_async_read(get_noc_addr(src), l1, nsticks * chunk_row_bytes);
                } else {
                    noc_async_read_one_packet_set_state(get_noc_addr(src), bytes);
                    for (uint32_t s = 0; s < nsticks; ++s) {
                        noc_async_read_one_packet_with_state(src + s * x_page_bytes, l1 + s * chunk_row_bytes);
                    }
                }
                const uint32_t tail_end = ((bytes + tile_row_bytes - 1) / tile_row_bytes) * tile_row_bytes;
                if (tail_end > bytes && (chunks_this > 1 || reservations < stage_slots)) {
                    for (uint32_t s = 0; s < nsticks; ++s) {
                        dataflow_kernel_lib::fill_l1_range<elem_bytes>(
                            l1 + s * chunk_row_bytes + bytes, tail_end - bytes, 0);
                    }
                }
                ++reservations;
                noc_async_read_barrier();
                cb_push_back(cb_x_rm, rm_chunk_w_tiles);
            }
        }
        prep_ones(blk);
    }
#elif defined(RMS_SHARDED)
    // ---- TILE shard: cb_x is backed on the resident shard (zero-copy) — publish its tile-rows to compute ----
    {
        const uint32_t shard_rows = (num_blocks - 1) * block_row_tiles + last_block_row_tiles;
        cb_reserve_back(cb_x, shard_rows * x_row_pitch);
        cb_push_back(cb_x, shard_rows * x_row_pitch);
        prep_ones(0);
    }
#elif defined(RMS_RM)
    const auto x_acc = TensorAccessor(x_args, x_addr, x_page_bytes);
    constexpr uint32_t chunk_row_bytes = rm_chunk_w_tiles * TILE_DIM * elem_bytes;  // L1 stride of one stick
    constexpr uint32_t tile_row_bytes = TILE_DIM * elem_bytes;
    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        const uint32_t rows = (blk + 1 == num_blocks) ? last_block_row_tiles : block_row_tiles;
        MaybeDeviceZoneScope("rd_block");  // occupancy: stick reads + back-pressure on cb_x_rm
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t t = row_start + blk * block_row_tiles + r;
            const uint32_t image = t / image_row_tiles;
            const uint32_t tr = t - image * image_row_tiles;
            const uint32_t first_stick = image * H + tr * TILE_DIM;
            const uint32_t valid_rows = (H - tr * TILE_DIM) < TILE_DIM ? (H - tr * TILE_DIM) : TILE_DIM;
            for (uint32_t c = 0; c < chunks_this; ++c) {
                const uint32_t chunk_tiles = (core_w_tiles - c * rm_chunk_w_tiles) < rm_chunk_w_tiles
                                                 ? (core_w_tiles - c * rm_chunk_w_tiles)
                                                 : rm_chunk_w_tiles;
                const uint32_t col0 = (w_tile_start + c * rm_chunk_w_tiles) * TILE_DIM;
                const uint32_t col_end_tiles = (w_tile_start + c * rm_chunk_w_tiles + chunk_tiles) * TILE_DIM;
                const uint32_t col_end = col_end_tiles < W ? col_end_tiles : W;
                const uint32_t bytes = (col_end - col0) * elem_bytes;
                if (bytes == chunk_row_bytes) {
                    dataflow_kernel_lib::read_sticks_for_tilize<cb_x_rm>(
                        x_acc, valid_rows, bytes, first_stick, col0 * elem_bytes);
                } else {
                    // Ragged chunk (pad tiles and/or W tail): raw read; zero to the next tile edge.
                    cb_reserve_back(cb_x_rm, rm_chunk_w_tiles);
                    const uint32_t l1 = get_write_ptr(cb_x_rm);
                    for (uint32_t s = 0; s < valid_rows; ++s) {
                        noc_async_read(
                            x_acc.get_noc_addr(first_stick + s, col0 * elem_bytes), l1 + s * chunk_row_bytes, bytes);
                    }
                    const uint32_t tail_end = ((bytes + tile_row_bytes - 1) / tile_row_bytes) * tile_row_bytes;
                    if (tail_end > bytes) {
                        for (uint32_t s = 0; s < valid_rows; ++s) {
                            dataflow_kernel_lib::fill_l1_range<elem_bytes>(
                                l1 + s * chunk_row_bytes + bytes, tail_end - bytes, 0);
                        }
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_x_rm, rm_chunk_w_tiles);
                }
            }
        }
        prep_ones(blk);
    }
#else
    const auto x_acc = TensorAccessor(x_args, x_addr, x_page_bytes);
    constexpr uint32_t tile_bytes = x_page_bytes;
    Noc noc;
    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        const uint32_t rows = (blk + 1 == num_blocks) ? last_block_row_tiles : block_row_tiles;
        const uint32_t cb_id = (x_block_depth == 2 && (blk & 1)) ? cb_x_alt : cb_x;
        CircularBuffer cb(cb_id);
        {
            MaybeDeviceZoneScope("rd_reserve");  // back-pressure: compute still holds this x buffer
            cb_reserve_back(cb_id, rows * x_row_pitch);
        }
        MaybeDeviceZoneScope("rd_read");  // issue + one barrier per tile-row (+ W-tail zero fill)
        const uint32_t base = get_write_ptr(cb_id);
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t t = row_start + blk * block_row_tiles + r;
            const uint32_t first_tile = t * tensor_w_tiles + w_tile_start;
            const uint32_t row_off = r * x_row_pitch * tile_bytes;
            for (uint32_t j = 0; j < core_w_tiles; ++j) {
                if constexpr (!ablate_dm) {
                    noc_async_read(x_acc.get_noc_addr(first_tile + j), base + row_off + j * tile_bytes, tile_bytes);
                }
            }
            noc_async_read_barrier();
            if (owns_w_tail) {
                // Offset relative to the CB's CURRENT write pointer, which has advanced by the r rows pushed so far.
                const uint32_t tail_tile_addr = base + row_off + (core_w_tiles - 1) * tile_bytes;
                zero_w_tail_of_tile(noc, cb, tail_tile_addr - cb.get_write_ptr());
            }
            cb_push_back(cb_id, x_row_pitch);
        }
        prep_ones(blk);
    }
#endif
}
