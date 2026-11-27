// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// a2a_wkv_b2 worker reader — reader_issue experiment: windowed (flow-controlled) DRAM read stream.
//
// Raw NoC issue; bypasses dataflow_kernel_lib::read_sticks_for_tilize. Mechanism: the worker read stream is
// DRAM-bandwidth bound in its middle phase (~510 GB/s of 512), and with ~100 KB in flight per core (W + a tile
// row, ~10 MB over 96 cores) the requests back up into the NoC, where router arbitration (parking-lot) hands
// bandwidth by core position: some cores finish at ~15 us, others at ~52 us, and the tail runs at half rate.
// Here every core keeps at most RI_WIN_K groups of RI_WIN_G requests in flight (sliding window over one stream:
// attn tile row 0, W, attn tile rows 1..), each group under its own transaction id; a new group is issued only
// after the oldest one has landed. The helper cannot express this: it issues a whole tile row, then a full
// barrier (one window shape, no W in the stream, no per-group trids).
// Optional RI_DUALNOC: per-(bank, pass) NoC choice from a 16-bit host mask (RT arg 4), as in ri_reader.cpp.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

#ifndef RI_DUALNOC
#define RI_DUALNOC 0
#endif
#ifndef RI_WIN_G
#define RI_WIN_G 4
#endif
#ifndef RI_WIN_K
#define RI_WIN_K 4
#endif

namespace {
constexpr uint32_t G = RI_WIN_G;       // requests per group (one trid)
constexpr uint32_t K = RI_WIN_K;       // max groups in flight (per-core window: RT arg 5, <= K)
constexpr uint32_t NUM_TRIDS = K + 1;  // group i uses trid 1 + i % NUM_TRIDS (reused only after it was waited)
static_assert(NUM_TRIDS <= 15, "trids 1..15");
constexpr uint32_t RING = 16;  // per-group completion actions (>= K)
static_assert(K <= RING, "");

enum : uint8_t { ACT_NONE = 0, ACT_ROW = 1, ACT_W = 2 };

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
    const Acc& acc, uint32_t page, uint32_t l1, uint32_t bytes, [[maybe_unused]] uint32_t mask) {
#ifndef A2A_ABL_NO_READ
#if RI_DUALNOC
    const uint32_t unit = (page % NUM_DRAM_BANKS) + NUM_DRAM_BANKS * ((page / NUM_DRAM_BANKS) & 1);
    const uint8_t noc = (mask >> unit) & 1;
    noc_async_read(acc.get_noc_addr(page, 0, noc), l1, bytes, noc);
#else
    noc_async_read(acc.get_noc_addr(page), l1, bytes);
#endif
#endif
}
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_weight = get_compile_time_arg_val(0);
    constexpr uint32_t cb_attn_rm = get_compile_time_arg_val(1);
    constexpr uint32_t Kt = get_compile_time_arg_val(2);
    constexpr uint32_t Nt = get_compile_time_arg_val(3);
    constexpr uint32_t Mt = get_compile_time_arg_val(4);
    constexpr uint32_t tokens = get_compile_time_arg_val(6);
    constexpr uint32_t stick_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t attn_page_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t w_tile_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t tile_h = get_compile_time_arg_val(10);
    constexpr auto attn_args = TensorAccessorArgs<11>();
    constexpr auto w_args = TensorAccessorArgs<attn_args.next_compile_time_args_offset()>();
    constexpr uint32_t depth = get_compile_time_arg_val(w_args.next_compile_time_args_offset());  // tile rows

    constexpr uint32_t w_block_tiles = Kt * Nt;
    constexpr uint32_t row_tile_bytes = Kt * get_tile_size(cb_attn_rm);
    constexpr uint32_t padded_stick_bytes = row_tile_bytes / tile_h;

    const uint32_t attn_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t head_start = get_arg_val<uint32_t>(2);
    const uint32_t num_heads = get_arg_val<uint32_t>(3);
    [[maybe_unused]] const uint32_t mask = get_arg_val<uint32_t>(4);
    const uint32_t win_rt = get_arg_val<uint32_t>(5);
    const uint32_t win = (win_rt == 0 || win_rt > K) ? K : win_rt;  // this core's window (groups)

    const auto attn = TensorAccessor(attn_args, attn_addr, attn_page_bytes);
    const auto w = TensorAccessor(w_args, w_addr, w_tile_bytes);
    const uint32_t attn_base = get_write_ptr(cb_attn_rm);

    uint8_t act[RING];
    uint32_t issued = 0, waited = 0;  // groups
    uint32_t in_group = 0;            // requests in the open group
    uint32_t rows_open = 0;           // tile rows reserved, not yet pushed
    uint32_t row_seq = 0;             // tile rows started (slot = row_seq % depth)

    auto wait_oldest = [&]() {
        barrier_trid_all(1 + (waited % NUM_TRIDS));
        const uint8_t a = act[waited % RING];
        if (a == ACT_ROW) {
            cb_push_back(cb_attn_rm, Kt);
            --rows_open;
        } else if (a == ACT_W) {
            cb_push_back(cb_weight, w_block_tiles);
        }
        ++waited;
    };
    auto open_group_if_needed = [&]() {
        if (in_group == 0) {
            while (issued - waited >= win) {
                wait_oldest();
            }
            set_trid_all(1 + (issued % NUM_TRIDS));
        }
    };
    auto close_group = [&](uint8_t a) {
        act[issued % RING] = a;
        ++issued;
        in_group = 0;
    };
    auto request = [&](auto& acc, uint32_t page, uint32_t l1, uint32_t bytes, bool last_of_item, uint8_t a) {
        open_group_if_needed();
        issue_page(acc, page, l1, bytes, mask);
        ++in_group;
        if (last_of_item) {
            close_group(a);
        } else if (in_group == G) {
            close_group(ACT_NONE);
        }
    };
    auto read_row = [&](uint32_t h, uint32_t r) {
        MaybeDeviceZoneScope("reader_attn_block");
        while (rows_open >= depth) {
            wait_oldest();
        }
        cb_reserve_back(cb_attn_rm, (rows_open + 1) * Kt);
        const uint32_t slot = attn_base + (row_seq % depth) * row_tile_bytes;
        ++row_seq;
        ++rows_open;
        const uint32_t tok0 = r * tile_h;
        const uint32_t rows = (tokens - tok0) < tile_h ? (tokens - tok0) : tile_h;
        const uint32_t page0 = h * tokens + tok0;
        for (uint32_t i = 0; i < rows; ++i) {
            request(attn, page0 + i, slot + i * padded_stick_bytes, stick_bytes, i + 1 == rows, ACT_ROW);
        }
    };

    for (uint32_t head_idx = 0; head_idx < num_heads; ++head_idx) {
        const uint32_t h = head_start + head_idx;
        if (head_idx > 0) {
            while (waited < issued) {  // previous head fully pushed before blocking on W space
                wait_oldest();
            }
        }
        read_row(h, 0);
        {
            MaybeDeviceZoneScope("reader_w_issue");
            cb_reserve_back(cb_weight, w_block_tiles);
            uint32_t l1 = get_write_ptr(cb_weight);
            const uint32_t w_tile0 = h * w_block_tiles;
            for (uint32_t t = 0; t < w_block_tiles; ++t) {
                request(w, w_tile0 + t, l1, w_tile_bytes, t + 1 == w_block_tiles, ACT_W);
                l1 += w_tile_bytes;
            }
        }
        for (uint32_t r = 1; r < Mt; ++r) {
            read_row(h, r);
        }
    }
    while (waited < issued) {
        wait_oldest();
    }
    set_trid_all(0);
}
