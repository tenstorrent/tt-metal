// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// a2a_wkv_b2 worker reader (NCRISC) — per-bank NoC choice (Perf 1, idea reader_issue).
//
// Per head h owned by this core:
//   load_weight_block : W[h] = Kt x Nt bf8_b tiles -> cb_weight (resident across mt); in flight behind the
//                       first attn tile row, pushed right after it (compute tilizes row 0 before it needs W)
//   load_attn_block   : one 32-stick tile row at a time -> cb_attn_rm (Kt tile-pages per tile row)
//
// Every page (attn stick or W tile) goes out on the NoC picked by a host mask (16 bits, in the core's common
// arg table word, see below): bit
// bank + 8 * ((page / num_banks) & 1) set = NoC 1, else NoC 0. The host spreads each core's DRAM responses
// over both NoCs so that no NoC link (the NoC 1 north links of the two DRAM columns carry the R2 "halves"
// split) saturates; see a2a_wkv_b2_program_descriptor.py NOC1_BANK_MASKS. The reader-alone time is unchanged
// (~52.4 us, a DRAM ceiling); the win (82.4 -> 78.6 us per call measured in isolation) is the NoC 1 DRAM-
// column links handed back to the fabric_tx pulls / sends.
//
// Raw NoC calls instead of dataflow_kernel_lib::read_sticks_for_tilize (capability, not ergonomics): the helper
// issues every stick on the kernel's one NoC and ends each tile row with noc_async_read_barrier(), which only
// counts this RISC's reads on its own NoC. Here NCRISC also issues on the other NoC through that NoC's read
// command buffer (buf 1; the worker writer on that NoC never reads, so the buffer is free), and completion is
// tracked with transaction ids (hardware per-trid counters, valid on both NoCs): trid ROW_TRID per tile row,
// W_TRID for W. The trid is reset to 0 before exit. Measured: helper vs this kernel on ONE NoC 52.4 vs 52.3 us
// reader-alone (a tie); the bypass is justified only by the dual-NoC issue (78.6 vs 82.4 us per call).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace {
constexpr uint32_t ROW_TRID = 1;
constexpr uint32_t W_TRID = 2;

FORCE_INLINE void set_trid_both(uint32_t trid) {
    noc_async_read_set_trid(trid, 0);
    noc_async_read_set_trid(trid, 1);
}

FORCE_INLINE void barrier_trid_both(uint32_t trid) {
    noc_async_read_barrier_with_trid(trid, 0);
    noc_async_read_barrier_with_trid(trid, 1);
}

template <typename Acc>
FORCE_INLINE void read_page(const Acc& acc, uint32_t page, uint32_t l1, uint32_t bytes, uint32_t noc1_mask) {
    const uint32_t unit = (page % NUM_DRAM_BANKS) + NUM_DRAM_BANKS * ((page / NUM_DRAM_BANKS) & 1);
    const uint8_t noc = (noc1_mask >> unit) & 1;
    noc_async_read(acc.get_noc_addr(page, 0, noc), l1, bytes, noc);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_weight = get_compile_time_arg_val(0);
    constexpr uint32_t cb_attn_rm = get_compile_time_arg_val(1);
    constexpr uint32_t Kt = get_compile_time_arg_val(2);
    constexpr uint32_t Nt = get_compile_time_arg_val(3);
    constexpr uint32_t Mt = get_compile_time_arg_val(4);
    constexpr uint32_t tokens = get_compile_time_arg_val(6);  // T: sticks per head
    constexpr uint32_t stick_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t attn_page_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t w_tile_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t tile_h = get_compile_time_arg_val(10);  // token rows per tile row (host TILE)
    constexpr auto attn_args = TensorAccessorArgs<11>();
    constexpr auto w_args = TensorAccessorArgs<attn_args.next_compile_time_args_offset()>();

    constexpr uint32_t w_block_tiles = Kt * Nt;
    constexpr uint32_t padded_stick_bytes = Kt * get_tile_size(cb_attn_rm) / tile_h;

    // Common runtime args only (no per-core args: launch gap, Perf 1):
    //   [attn_addr, w_addr, grid_x, per logical core index y * grid_x + x:
    //    head_start | num_heads << 8 | noc1_mask << 16]
    const uint32_t attn_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t core_word = get_common_arg_val<uint32_t>(
        3 + get_relative_logical_y() * get_common_arg_val<uint32_t>(2) + get_relative_logical_x());
    const uint32_t head_start = core_word & 0xFF;
    const uint32_t num_heads = (core_word >> 8) & 0xFF;
    const uint32_t noc1_mask = core_word >> 16;

    const auto attn = TensorAccessor(attn_args, attn_addr, attn_page_bytes);
    const auto w = TensorAccessor(w_args, w_addr, w_tile_bytes);

    for (uint32_t head_idx = 0; head_idx < num_heads; ++head_idx) {
        const uint32_t h = head_start + head_idx;

        {
            MaybeDeviceZoneScope("reader_w_issue");  // reserve (compute releases the previous W) + issue
            cb_reserve_back(cb_weight, w_block_tiles);
            uint32_t l1 = get_write_ptr(cb_weight);
            const uint32_t w_tile0 = h * w_block_tiles;
            set_trid_both(W_TRID);
            for (uint32_t t = 0; t < w_block_tiles; ++t) {
#ifndef A2A_ABL_NO_READ
                read_page(w, w_tile0 + t, l1, w_tile_bytes, noc1_mask);
#endif
                l1 += w_tile_bytes;
            }
        }

        for (uint32_t r = 0; r < Mt; ++r) {  // one tile row per iteration (mt_block only groups compute)
            MaybeDeviceZoneScope("reader_attn_block");
            const uint32_t tok0 = r * tile_h;
            const uint32_t rows = (tokens - tok0) < tile_h ? (tokens - tok0) : tile_h;
            cb_reserve_back(cb_attn_rm, Kt);
            uint32_t l1 = get_write_ptr(cb_attn_rm);
            set_trid_both(ROW_TRID);
            for (uint32_t i = 0; i < rows; ++i) {
#ifndef A2A_ABL_NO_READ
                read_page(attn, h * tokens + tok0 + i, l1, stick_bytes, noc1_mask);
#endif
                l1 += padded_stick_bytes;
            }
            barrier_trid_both(ROW_TRID);
            cb_push_back(cb_attn_rm, Kt);
            if (r == 0) {
                barrier_trid_both(W_TRID);  // W landed behind the first tile row
                cb_push_back(cb_weight, w_block_tiles);
            }
        }
    }
    set_trid_both(0);
}
