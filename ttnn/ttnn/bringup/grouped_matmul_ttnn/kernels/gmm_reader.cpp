// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ttnn.bringup.grouped_matmul reader (the reader config's RISC / NoC). Core (cx, row) of group g, per K block kb:
//   in1: its own weight tiles W[g, kb BW + k, n0 + j] (k < BW, j < PN) -> CB_IN1 slot [k][j] (row stride PNX: the
//        widest column count of the grid, so every core's blocks are the same size; a narrower core leaves its last
//        columns unread)
//   in0: the row's sender (column 0) reads x[g, m, kb BW + k] (m < MT, k < BW) -> CB_IN0 slot [m][k] and multicasts
//        the block to the row's other cores (same CB address everywhere: every core runs the same CB sequence);
//        receivers credit the sender's READY semaphore once their slot is reserved and wait for VALID.
// Every semaphore wait is bounded (WAIT_LIMIT polls): a lost credit gives a wrong result, never a hang.
// (A version reading DEPTH - 1 blocks ahead under NoC transaction ids measured slower on the 8x4: 0.143 vs 0.133 ms.)
// compile-time args: MT, KT, NT, BW, PN, GX, CB_IN0, CB_IN1, IN0_TB, IN1_TB, PNX, x accessor, w accessor
// common args: x address, w address; per-core args: g, n0, sender, mc_x0, mc_x1, mc_y, sender_x, sender_y

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t WAIT_LIMIT = 1u << 26;

FORCE_INLINE void wait_ge(volatile tt_l1_ptr uint32_t* p, uint32_t v) {
    for (uint32_t i = 0; i < WAIT_LIMIT; ++i) {
        invalidate_l1_cache();
        if (*p >= v) {
            return;
        }
    }
}

void kernel_main() {
    constexpr uint32_t MT = get_compile_time_arg_val(0);
    constexpr uint32_t KT = get_compile_time_arg_val(1);
    constexpr uint32_t NT = get_compile_time_arg_val(2);
    constexpr uint32_t BW = get_compile_time_arg_val(3);
    constexpr uint32_t PN = get_compile_time_arg_val(4);
    constexpr uint32_t GX = get_compile_time_arg_val(5);
    constexpr uint32_t CB_IN0 = get_compile_time_arg_val(6);
    constexpr uint32_t CB_IN1 = get_compile_time_arg_val(7);
    constexpr uint32_t IN0_TB = get_compile_time_arg_val(8);
    constexpr uint32_t IN1_TB = get_compile_time_arg_val(9);
    constexpr uint32_t PNX = get_compile_time_arg_val(10);
    constexpr auto x_args = TensorAccessorArgs<11>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr uint32_t NBLK = KT / BW;
    constexpr uint32_t IN0_PAGES = MT * BW, IN1_PAGES = BW * PNX;
    constexpr uint32_t IN0_BLOCK = IN0_PAGES * IN0_TB;

    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t g = get_arg_val<uint32_t>(0);
    const uint32_t n0 = get_arg_val<uint32_t>(1);
    const bool sender = get_arg_val<uint32_t>(2) != 0;
    const uint32_t mc_x0 = get_arg_val<uint32_t>(3);
    const uint32_t mc_x1 = get_arg_val<uint32_t>(4);
    const uint32_t mc_y = get_arg_val<uint32_t>(5);
    const uint32_t sender_x = get_arg_val<uint32_t>(6);
    const uint32_t sender_y = get_arg_val<uint32_t>(7);

    const auto xa = TensorAccessor(x_args, x_addr, IN0_TB);
    const auto wa = TensorAccessor(w_args, w_addr, IN1_TB);
    const uint32_t ready_addr = get_semaphore(0);
    const uint32_t valid_addr = get_semaphore(1);
    volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
    volatile tt_l1_ptr uint32_t* valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(valid_addr);
    if (sender) {
        *valid = 1;  // (the source word of the VALID multicast)
    }
    const uint32_t x_row0 = g * MT * KT;
    const uint32_t w_row0 = g * KT * NT + n0;

    for (uint32_t kb = 0; kb < NBLK; ++kb) {
        cb_reserve_back(CB_IN0, IN0_PAGES);
        const uint32_t a = get_write_ptr(CB_IN0);
        if (sender) {
            uint32_t l1 = a;
            for (uint32_t m = 0; m < MT; ++m) {
                const uint32_t p0 = x_row0 + m * KT + kb * BW;
                for (uint32_t k = 0; k < BW; ++k) {
                    noc_async_read(xa.get_noc_addr(p0 + k), l1, IN0_TB);
                    l1 += IN0_TB;
                }
            }
        }
        cb_reserve_back(CB_IN1, IN1_PAGES);
        {
            const uint32_t l1 = get_write_ptr(CB_IN1);
            for (uint32_t k = 0; k < BW; ++k) {
                const uint32_t p0 = w_row0 + (kb * BW + k) * NT;
                for (uint32_t j = 0; j < PN; ++j) {
                    noc_async_read(wa.get_noc_addr(p0 + j), l1 + (k * PNX + j) * IN1_TB, IN1_TB);
                }
            }
        }
        if constexpr (GX > 1) {
            if (sender) {
                noc_async_read_barrier();
                wait_ge(ready, GX - 1);
                noc_semaphore_set(ready, 0);
                noc_async_write_multicast(a, get_noc_multicast_addr(mc_x0, mc_y, mc_x1, mc_y, a), IN0_BLOCK, GX - 1);
                // (the VALID flag after the data, same NoC and path: it lands after the block)
                noc_semaphore_set_multicast(
                    valid_addr, get_noc_multicast_addr(mc_x0, mc_y, mc_x1, mc_y, valid_addr), GX - 1);
            } else {
                noc_semaphore_set(valid, 0);
                noc_semaphore_inc(get_noc_addr(sender_x, sender_y, ready_addr), 1);
                noc_async_read_barrier();
                wait_ge(valid, 1);
            }
        } else {
            noc_async_read_barrier();
        }
        cb_push_back(CB_IN0, IN0_PAGES);
        cb_push_back(CB_IN1, IN1_PAGES);
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
