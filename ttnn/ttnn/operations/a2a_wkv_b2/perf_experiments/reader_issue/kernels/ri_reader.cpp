// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// a2a_wkv_b2 worker reader — reader_issue experiment variant (raw NoC issue; bypasses
// dataflow_kernel_lib::read_sticks_for_tilize).
//
// Why the helper is bypassed: read_sticks_for_tilize issues every stick on the kernel's one NoC and ends each
// 32-stick tile row with a full noc_async_read_barrier(). This variant needs two things it cannot express:
//   (1) per-bank NoC choice (RI_DUALNOC): each stick goes out on the NoC whose response path from the stick's
//       DRAM bank to this core does not wrap past a DRAM column (host mask, RT arg 4: bit b = bank b on NoC 1).
//       NCRISC issues on both NoCs through each NoC's read command buffer (buf 1; the worker writer on the other
//       NoC never reads, so that NoC's read buffer is free). Plain barriers count per-RISC issued reads only on
//       the kernel's own NoC, so completion is tracked with transaction ids (hardware per-trid counters).
//   (2) a pipelined CB fill (RI_PIPE): tile row r+1 is issued before tile row r's completion is waited on, each
//       tile row under its own trid, so a row's barrier never waits on younger requests and the request stream
//       has no per-row drain bubble. W gets its own trid.
// Options: RI_STAGGER rotates each tile row's stick issue order by head index (bank phase per core).
//
// Per head h owned by this core:
//   W[h] (Kt x Nt bf8_b tiles) -> cb_weight, resident across mt (compute pops it after the head)
//   attn sticks, one tile row (tile_h sticks) at a time -> cb_attn_rm (Kt tile-pages per tile row)

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#ifdef RI_PRINT
#include "api/debug/dprint.h"
#endif

#ifndef RI_DUALNOC
#define RI_DUALNOC 0
#endif
#ifndef RI_PIPE
#define RI_PIPE 0
#endif
#ifndef RI_STAGGER
#define RI_STAGGER 0
#endif

namespace {
constexpr uint32_t W_TRID = 15;

FORCE_INLINE void set_trid_all(uint32_t trid) {
    noc_async_read_set_trid(trid, noc_index);
#if RI_DUALNOC
    noc_async_read_set_trid(trid, 1 - noc_index);
#endif
}

FORCE_INLINE void barrier_trid_all(uint32_t trid) {
    noc_async_read_barrier_with_trid(trid, noc_index);
#if RI_DUALNOC
    noc_async_read_barrier_with_trid(trid, 1 - noc_index);
#endif
}

template <typename Acc>
FORCE_INLINE void issue_page(
    const Acc& acc, uint32_t page, uint32_t l1, uint32_t bytes, [[maybe_unused]] uint32_t noc1_mask) {
#if RI_DUALNOC
    // unit = bank + NUM_DRAM_BANKS * (pass over the banks % 2): two NoC choices per bank (16-bit mask)
    const uint32_t unit = (page % NUM_DRAM_BANKS) + NUM_DRAM_BANKS * ((page / NUM_DRAM_BANKS) & 1);
    const uint8_t noc = (noc1_mask >> unit) & 1;
    noc_async_read(acc.get_noc_addr(page, 0, noc), l1, bytes, noc);
#else
    noc_async_read(acc.get_noc_addr(page), l1, bytes);
#endif
}
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_weight = get_compile_time_arg_val(0);
    constexpr uint32_t cb_attn_rm = get_compile_time_arg_val(1);
    constexpr uint32_t Kt = get_compile_time_arg_val(2);
    constexpr uint32_t Nt = get_compile_time_arg_val(3);
    constexpr uint32_t Mt = get_compile_time_arg_val(4);
    constexpr uint32_t mt_block = get_compile_time_arg_val(5);
    constexpr uint32_t tokens = get_compile_time_arg_val(6);  // T: sticks per head
    constexpr uint32_t stick_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t attn_page_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t w_tile_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t tile_h = get_compile_time_arg_val(10);
    constexpr auto attn_args = TensorAccessorArgs<11>();
    constexpr auto w_args = TensorAccessorArgs<attn_args.next_compile_time_args_offset()>();
    constexpr uint32_t attn_rm_rows = get_compile_time_arg_val(w_args.next_compile_time_args_offset());  // CB depth

    constexpr uint32_t w_block_tiles = Kt * Nt;
    constexpr uint32_t row_tile_bytes = Kt * get_tile_size(cb_attn_rm);  // one tile row in cb_attn_rm
    constexpr uint32_t padded_stick_bytes = row_tile_bytes / tile_h;
    static_assert(RI_PIPE == 0 || attn_rm_rows >= 2, "pipelined fill needs >= 2 tile rows of cb_attn_rm");

    const uint32_t attn_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t head_start = get_arg_val<uint32_t>(2);
    const uint32_t num_heads = get_arg_val<uint32_t>(3);
    [[maybe_unused]] const uint32_t bank_mask = get_arg_val<uint32_t>(4);  // bit b: bank b on NoC 1 (RI_DUALNOC)

#ifdef RI_PRINT
    DPRINT(
        "me noc0 {} {} noc1 {} {} idx {}\n",
        (uint32_t)my_x[0],
        (uint32_t)my_y[0],
        (uint32_t)my_x[1],
        (uint32_t)my_y[1],
        (uint32_t)noc_index);
    for (uint32_t b = 0; b < NUM_DRAM_BANKS; ++b) {
        DPRINT("bank {} n0 {} n1 {}\n", b, (uint32_t)dram_bank_to_noc_xy[0][b], (uint32_t)dram_bank_to_noc_xy[1][b]);
    }
#endif
    const auto attn = TensorAccessor(attn_args, attn_addr, attn_page_bytes);
    const auto w = TensorAccessor(w_args, w_addr, w_tile_bytes);

    const uint32_t attn_base = get_write_ptr(cb_attn_rm);  // slot 0 (fifo starts here, rows wrap at depth)
    uint32_t row_seq = 0;                                  // tile rows pushed so far (slot = row_seq % depth)

    for (uint32_t head_idx = 0; head_idx < num_heads; ++head_idx) {
        const uint32_t h = head_start + head_idx;
        const uint32_t page0 = h * tokens;
#if RI_STAGGER
        const uint32_t rot = h % NUM_DRAM_BANKS;
#else
        constexpr uint32_t rot = 0;
#endif

        auto issue_row = [&](uint32_t r, uint32_t trid) {
            // tile row r of this head (rows <= tile_h sticks) into its CB slot, tagged trid
            const uint32_t tok0 = r * tile_h;
            const uint32_t rows = (tokens - tok0) < tile_h ? (tokens - tok0) : tile_h;
            const uint32_t slot = attn_base + ((row_seq + r) % attn_rm_rows) * row_tile_bytes;
            set_trid_all(trid);
#ifdef RI_BIGREQ
            // PERF PROBE ONLY (wrong row order): one bank-contiguous read per bank covering its rows/8 sticks
            for (uint32_t b = 0; b < NUM_DRAM_BANKS; ++b) {
                noc_async_read(
                    attn.get_noc_addr(page0 + tok0 + b),
                    slot + b * (rows / NUM_DRAM_BANKS) * padded_stick_bytes,
                    (rows / NUM_DRAM_BANKS) * attn_page_bytes);
            }
            if (true) {
                return;
            }
#endif
            for (uint32_t i = 0; i < rows; ++i) {
                uint32_t j = i + rot;
                if (j >= rows) {
                    j -= rows;
                }
#ifndef A2A_ABL_NO_READ
                issue_page(attn, page0 + tok0 + j, slot + j * padded_stick_bytes, stick_bytes, bank_mask);
#endif
            }
        };

        constexpr uint32_t num_rows = Mt;  // tile rows per head (mt_block only groups compute)

        // W[h]: reserve (compute releases the previous head's W) + issue under its own trid.
        {
            MaybeDeviceZoneScope("reader_w_issue");
            cb_reserve_back(cb_weight, w_block_tiles);
            uint32_t l1 = get_write_ptr(cb_weight);
            const uint32_t w_tile0 = h * w_block_tiles;
#if RI_PIPE
            // first attn tile row goes out ahead of W (compute tilizes it before it needs W)
            cb_reserve_back(cb_attn_rm, Kt);
            issue_row(0, 1);
#endif
            set_trid_all(W_TRID);
#ifdef RI_BIGREQ
            for (uint32_t b = 0; b < NUM_DRAM_BANKS; ++b) {
                noc_async_read(
                    w.get_noc_addr(w_tile0 + b),
                    l1 + b * (w_block_tiles / NUM_DRAM_BANKS) * w_tile_bytes,
                    (w_block_tiles / NUM_DRAM_BANKS) * w_tile_bytes);
            }
            if (false)
#endif
                for (uint32_t t = 0; t < w_block_tiles; ++t) {
#ifndef A2A_ABL_NO_READ
                    issue_page(w, w_tile0 + t, l1, w_tile_bytes, bank_mask);
#endif
                    l1 += w_tile_bytes;
                }
        }

#if RI_PIPE
        // tile row r is in flight under trid 1 + (r % depth); issue r+1 before waiting on r.
        for (uint32_t r = 0; r < num_rows; ++r) {
            MaybeDeviceZoneScope("reader_attn_block");
            if (r + 1 < num_rows) {
                cb_reserve_back(cb_attn_rm, 2 * Kt);  // this row's slot (held) + the next one
                issue_row(r + 1, 1 + ((r + 1) % attn_rm_rows));
            }
            barrier_trid_all(1 + (r % attn_rm_rows));
            cb_push_back(cb_attn_rm, Kt);
            if (r == 0) {
                barrier_trid_all(W_TRID);
                cb_push_back(cb_weight, w_block_tiles);
            }
        }
#else
        for (uint32_t r = 0; r < num_rows; ++r) {
            MaybeDeviceZoneScope("reader_attn_block");
            cb_reserve_back(cb_attn_rm, Kt);
            issue_row(r, 1 + (r % attn_rm_rows));
            barrier_trid_all(1 + (r % attn_rm_rows));
            if (r == 0) {
                barrier_trid_all(W_TRID);
            }
            cb_push_back(cb_attn_rm, Kt);
            if (r == 0) {
                cb_push_back(cb_weight, w_block_tiles);
            }
        }
#endif
        row_seq += num_rows;
    }
    set_trid_all(0);
}
