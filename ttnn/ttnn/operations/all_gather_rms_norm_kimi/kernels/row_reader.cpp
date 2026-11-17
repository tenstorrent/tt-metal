// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Row core reader (one core per (tile-row block, group slice p)).
//  * x: own slice (p == my position) -> read from DRAM into the landing half of this call's parity (RowLayout slots);
//       a peer's slice -> wait for the tag of every packet of my block (they land in any order).
//    Every packet this chip must pass on (its own slice, or a peer's that has more hops to go) is pushed, as soon as it
//    is here, with its tag, into the staging slot of the port core that sends it (one NoC write per destination).
//    Then the whole landing CB (both parities) is published; compute indexes the parity half.
//  * gamma: the core's Wt sticks -> row 0 of Wt tiles (CB_GAMMA)
//  * reduce scaler (CB_SCALER)

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "agrn_common.hpp"

using namespace agrn;

void kernel_main() {
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);    // tiles per tile-row of one slice
    constexpr uint32_t rows = get_compile_time_arg_val(2);  // tile-rows per core
    constexpr uint32_t half_a = get_compile_time_arg_val(3);
    constexpr uint32_t P = get_compile_time_arg_val(4);
    constexpr uint32_t stage_slot_bytes = get_compile_time_arg_val(5);
    constexpr auto in_args = TensorAccessorArgs<6>();
    constexpr auto g_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    using RL = RowLayout<Wt, half_a, P>;
    constexpr uint32_t blk = RL::slots * rows;  // tile slots per parity half
    constexpr uint32_t npk = RL::npk * rows;
    static_assert(npk <= 32, "packets per row core");

    size_t a = 0;
    const uint32_t in_addr = get_arg_val<uint32_t>(a++);
    const uint32_t g_addr = get_arg_val<uint32_t>(a++);
    const uint32_t ctr_addr = get_arg_val<uint32_t>(a++);
    const uint32_t is_local = get_arg_val<uint32_t>(a++);
    const uint32_t in_page0 = get_arg_val<uint32_t>(a++);      // first input page of my block (row r0, col 0)
    const uint32_t in_row_pitch = get_arg_val<uint32_t>(a++);  // input pages per tile-row (Wt)
    const uint32_t g_stick0 = get_arg_val<uint32_t>(a++);      // first gamma stick (= first output tile column)
    const uint32_t salt = get_arg_val<uint32_t>(a++);
    const uint32_t token_sem = get_semaphore(get_arg_val<uint32_t>(a++));
    const uint32_t token_wait = get_arg_val<uint32_t>(a++);  // local: wait for a push token before pushing
    const uint32_t token_next = get_arg_val<uint32_t>(a++);  // local: (x << 16) | y of the core to pass it to, or 0
    const uint32_t fwd_idx = a;  // per packet (row-major, RowLayout order): count, then count x ((x << 16) | y, slot)

    const uint32_t ctr = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctr_addr);
    const uint32_t parity = ctr & 1;
    const uint32_t tag = ctr + 1;
    const uint32_t stage = get_write_ptr(CB_STAGE);  // same address on every core of the program

    cb_reserve_back(CB_LAND, 2 * blk);
    const uint32_t land = get_write_ptr(CB_LAND) + parity * blk * tile_bytes;
    const auto in = TensorAccessor(in_args, in_addr, tile_bytes);
    if (is_local) {
        AGRN_ZONE("rr_local_issue");
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t j = 0; j < Wt; ++j) {
                noc_async_read_tile(
                    in_page0 + r * in_row_pitch + j, in, land + (r * RL::slots + RL::slot_of(j)) * tile_bytes);
            }
        }
    }

    // gamma sticks (64 B, 32 bf16) -> row 0 of faces 0 and 1. Every row core reads them (many tiny reads of the same
    // DRAM pages): a local row core issues them only after it has pushed its packets, which the ports wait for.
    const auto g = TensorAccessor(g_args, g_addr, 64);
    cb_reserve_back(CB_GAMMA, Wt);
    const uint32_t gbase = get_write_ptr(CB_GAMMA);
    auto read_gamma = [&]() {
        for (uint32_t j = 0; j < Wt; ++j) {
            noc_async_read(g.get_noc_addr(g_stick0 + j), gbase + j * tile_bytes, 64);
        }
    };
    if (!is_local) {
        read_gamma();
    }

    // packet k of my block: its landing address and forward list
    uint32_t pk_addr[npk], pk_bytes[npk], pk_fwd[npk];
    {
        uint32_t d = fwd_idx, i = 0;
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t k = 0; k < RL::npk; ++k, ++i) {
                pk_addr[i] = land + (r * RL::slots + RL::slot_of(RL::first(k))) * tile_bytes;
                pk_bytes[i] = RL::count(k) * tile_bytes;
                pk_fwd[i] = d;
                d += 1 + 2 * get_arg_val<uint32_t>(d);
            }
        }
    }
    auto forward = [&](uint32_t i) {
        const uint32_t d = pk_fwd[i];
        const uint32_t cnt = get_arg_val<uint32_t>(d);
        for (uint32_t e = 0; e < cnt; ++e) {
            const uint32_t xy = get_arg_val<uint32_t>(d + 1 + 2 * e);
            const uint32_t slot = get_arg_val<uint32_t>(d + 2 + 2 * e);
            noc_async_write(
                pk_addr[i],
                get_noc_addr(xy >> 16, xy & 0xFFFF, stage + slot * stage_slot_bytes),
                pk_bytes[i] + TAG_BYTES);
        }
    };

    if (is_local) {
        {
            AGRN_ZONE("rr_x_wait");
            noc_async_read_barrier();
        }
        AGRN_ZONE("rr_local_push");
        // the ports take this chip's packets in tile-row order and their NoC ingress is shared by every local row core:
        // push in order (a token chain over the local row cores of a link, a few rows ahead), so the packet a port
        // needs next is not queued behind ones it needs later
        if (token_wait) {
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(token_sem), 1);
        }
        for (uint32_t i = 0; i < npk; ++i) {
            write_tag(pk_addr[i] + pk_bytes[i], tag, salt);
            forward(i);
        }
        if (token_next) {
            noc_async_writes_flushed();
            noc_semaphore_inc(get_noc_addr(token_next >> 16, token_next & 0xFFFF, token_sem), 1);
        }
        read_gamma();
    }
    // reduce scaler: 1.0 in row 0 of every face, zeros elsewhere (bf16)
    {
        cb_reserve_back(CB_SCALER, 1);
        volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_SCALER));
        for (uint32_t i = 0; i < tile_bytes / 4; ++i) {
            s[i] = 0;
        }
        for (uint32_t f = 0; f < 4; ++f) {
            for (uint32_t i = 0; i < 8; ++i) {
                s[f * 128 + i] = 0x3F803F80u;
            }
        }
        cb_push_back(CB_SCALER, 1);
    }
    noc_async_read_barrier();  // gamma (and my own slice)
    for (uint32_t j = 0; j < Wt; ++j) {
        volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gbase + j * tile_bytes);
        for (uint32_t i = 0; i < 8; ++i) {
            t[128 + i] = t[8 + i];  // elements 16..31 -> face 1 row 0
        }
    }
    cb_push_back(CB_GAMMA, Wt);
    if (!is_local) {
        AGRN_ZONE("rr_arrive_wait");
        // forward every packet the moment it lands (in whatever order they come)
        uint32_t pending = (npk == 32) ? 0xFFFFFFFFu : ((1u << npk) - 1);
        while (pending) {
            invalidate_l1_cache();
            for (uint32_t i = 0; i < npk; ++i) {
                if ((pending >> i) & 1) {
                    if (tag_ok(pk_addr[i] + pk_bytes[i], tag, salt)) {
                        pending &= ~(1u << i);
                        forward(i);
                    }
                }
            }
        }
    }
    cb_push_back(CB_LAND, rows);

    noc_async_write_barrier();
}
