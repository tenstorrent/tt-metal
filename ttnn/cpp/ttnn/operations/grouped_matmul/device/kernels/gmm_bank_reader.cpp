// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ttnn.bringup.grouped_matmul reader, bank path (option w_layout; the reader config's RISC, NoC 0). Core of group g,
// region tile0 in DRAM bank `bank` (bank.py), per K block kb:
//   in1: ONE contiguous read of its BW x PN weight tiles from its own bank (bank-local address tile0 + kb BW PN) into
//        CB_IN1 slot (kb % DEPTH) ([k][j], row stride PNX; a PN < PNX core reads BW rows of PN tiles), W_INFLIGHT
//        blocks in flight under transaction ids TRID_W0 + kb % W_INFLIGHT
//   in0: the cluster's sender reads x[g, m, kb BW + k] (interleaved, m < MT, k < BW) into CB_IN0 slot (kb % DEPTH)
//        and multicasts the block to the cluster's rectangle (sender inside, not written: non-loopback; the sender is
//        the rectangle's first core on the 8x4, so every multicast leaves it eastward / southward on NoC 0); receivers
//        credit READY once their slot is reserved and wait for VALID (today's handshake). X_AHEAD (0..3): the sender
//        also issues the reads of blocks kb + 1 .. kb + X_AHEAD whose slots are already free (never waits for one),
//        under transaction ids TRID_X0 + block % 4
// The compute and writer kernels are today's (same CB protocol): same tiles, same K order, same output.
// Every semaphore wait is bounded (WAIT_LIMIT polls): a lost credit gives a wrong result, never a hang. NoC reads
// complete by themselves (the trid barriers).
// Diagnostic defines (WA.2, never set by a model): GMM_DIAG_NO_W skips the weight reads, GMM_DIAG_NO_X the x reads
// and the multicast handshake (every core then runs on its own).
// compile-time args: MT, KT, BW, PN, PNX, CB_IN0, CB_IN1, IN0_TB, IN1_TB, DEPTH, W_INFLIGHT, X_AHEAD, x accessor
// common args: x address, w address (the bank-local base); per-core args: g, tile0, bank, sender, mc_x0, mc_y0, mc_x1,
// mc_y1, ndest (cores of the rectangle, sender included), sender_x, sender_y (and PN under GMM_PN_RT: one kernel group
// for every core width, WA.3: fewer core ranges, less host time a launch). Slots: grouped_matmul_args.hpp.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/grouped_matmul/device/kernels/grouped_matmul_args.hpp"

using namespace grouped_matmul;

constexpr uint32_t WAIT_LIMIT = 1u << 26;
constexpr uint32_t TRID_W0 = 1;   // weight blocks: 1 .. W_INFLIGHT (<= 8)
constexpr uint32_t TRID_X0 = 12;  // x blocks: 12 .. 15

FORCE_INLINE void wait_ge(volatile tt_l1_ptr uint32_t* p, uint32_t v) {
    for (uint32_t i = 0; i < WAIT_LIMIT; ++i) {
        invalidate_l1_cache();
        if (*p >= v) {
            return;
        }
    }
}

void kernel_main() {
    constexpr uint32_t MT = get_compile_time_arg_val(bank_reader_ct::MT);
    constexpr uint32_t KT = get_compile_time_arg_val(bank_reader_ct::KT);
    constexpr uint32_t BW = get_compile_time_arg_val(bank_reader_ct::BW);
#ifdef GMM_PN_RT
    const uint32_t PN = get_arg_val<uint32_t>(bank_reader_rt::PN);  // (one kernel group for every core width, WA.3)
#else
    constexpr uint32_t PN = get_compile_time_arg_val(bank_reader_ct::PN);
#endif
    constexpr uint32_t PNX = get_compile_time_arg_val(bank_reader_ct::PNX);
    constexpr uint32_t CB_IN0 = get_compile_time_arg_val(bank_reader_ct::CB_IN0);
    constexpr uint32_t CB_IN1 = get_compile_time_arg_val(bank_reader_ct::CB_IN1);
    constexpr uint32_t IN0_TB = get_compile_time_arg_val(bank_reader_ct::IN0_TB);
    constexpr uint32_t IN1_TB = get_compile_time_arg_val(bank_reader_ct::IN1_TB);
    constexpr uint32_t DEPTH = get_compile_time_arg_val(bank_reader_ct::DEPTH);
    constexpr uint32_t W_INFLIGHT = get_compile_time_arg_val(bank_reader_ct::W_INFLIGHT);
    constexpr uint32_t X_AHEAD = get_compile_time_arg_val(bank_reader_ct::X_AHEAD);
    constexpr auto x_args = TensorAccessorArgs<bank_reader_ct::COUNT>();
    static_assert(W_INFLIGHT >= 1 && W_INFLIGHT <= DEPTH && W_INFLIGHT <= 8, "w_inflight in [1, min(depth, 8)]");
    static_assert(X_AHEAD <= 3 && X_AHEAD < DEPTH, "x_ahead in [0, min(3, depth - 1)]");
    constexpr uint32_t NBLK = KT / BW;
    constexpr uint32_t IN0_PAGES = MT * BW, IN1_PAGES = BW * PNX;
    constexpr uint32_t IN0_BLOCK = IN0_PAGES * IN0_TB, IN1_BLOCK = IN1_PAGES * IN1_TB;
    constexpr uint32_t NW = W_INFLIGHT < NBLK ? W_INFLIGHT : NBLK;

    const uint32_t x_addr = get_common_arg_val<uint32_t>(reader_common::X_ADDR);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(reader_common::W_ADDR);
    const uint32_t g = get_arg_val<uint32_t>(bank_reader_rt::G);
    const uint32_t tile0 = get_arg_val<uint32_t>(bank_reader_rt::TILE0);
    const uint32_t bank = get_arg_val<uint32_t>(bank_reader_rt::BANK);
    const bool sender = get_arg_val<uint32_t>(bank_reader_rt::SENDER) != 0;
    const uint32_t mc_x0 = get_arg_val<uint32_t>(bank_reader_rt::MC_X0);
    const uint32_t mc_y0 = get_arg_val<uint32_t>(bank_reader_rt::MC_Y0);
    const uint32_t mc_x1 = get_arg_val<uint32_t>(bank_reader_rt::MC_X1);
    const uint32_t mc_y1 = get_arg_val<uint32_t>(bank_reader_rt::MC_Y1);
    const uint32_t ndest = get_arg_val<uint32_t>(bank_reader_rt::NDEST);
    const uint32_t sender_x = get_arg_val<uint32_t>(bank_reader_rt::SENDER_X);
    const uint32_t sender_y = get_arg_val<uint32_t>(bank_reader_rt::SENDER_Y);

    const auto xa = TensorAccessor(x_args, x_addr, IN0_TB);
    const uint32_t ready_addr = get_semaphore(sem::READY);
    const uint32_t valid_addr = get_semaphore(sem::VALID);
    volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
    volatile tt_l1_ptr uint32_t* valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(valid_addr);
    if (sender) {
        *valid = 1;  // (the source word of the VALID multicast)
    }
    // (both CBs are fresh at launch: their write pointers are the bases, slot s at base + s * block)
    const uint32_t in0_base = get_write_ptr(CB_IN0);
    const uint32_t in1_base = get_write_ptr(CB_IN1);
    const uint32_t x_row0 = g * MT * KT;
    const uint32_t w_src0 = w_addr + tile0 * IN1_TB;

    auto issue_w = [&](uint32_t kb) {
#ifdef GMM_DIAG_NO_W
        return;
#endif
        noc_async_read_set_trid(TRID_W0 + kb % W_INFLIGHT);
        const uint32_t src = w_src0 + kb * BW * PN * IN1_TB;
        const uint32_t dst = in1_base + (kb % DEPTH) * IN1_BLOCK;
        if (PN == PNX) {
            noc_async_read(get_noc_addr_from_bank_id<true>(bank, src), dst, BW * PN * IN1_TB);
        } else {
            for (uint32_t k = 0; k < BW; ++k) {
                noc_async_read(
                    get_noc_addr_from_bank_id<true>(bank, src + k * PN * IN1_TB), dst + k * PNX * IN1_TB, PN * IN1_TB);
            }
        }
    };
    auto issue_x = [&](uint32_t kb) {
#ifdef GMM_DIAG_NO_X
        return;
#endif
        if (kb >= DEPTH) {
            noc_async_writes_flushed();  // (the multicast that read this slot DEPTH blocks ago has left L1)
        }
        noc_async_read_set_trid(TRID_X0 + (kb & 3));
        uint32_t l1 = in0_base + (kb % DEPTH) * IN0_BLOCK;
        for (uint32_t m = 0; m < MT; ++m) {
            const uint32_t p0 = x_row0 + m * KT + kb * BW;
            for (uint32_t k = 0; k < BW; ++k) {
                noc_async_read(xa.get_noc_addr(p0 + k), l1, IN0_TB);
                l1 += IN0_TB;
            }
        }
    };

    cb_reserve_back(CB_IN1, NW * IN1_PAGES);
    for (uint32_t i = 0; i < NW; ++i) {
        issue_w(i);
    }
    uint32_t xi = 0;  // (sender: the next x block to read)
    // the x blocks after kb (<= kb + X_AHEAD) whose slots are free now: slot kb is reserved, kb + j needs j + 1 blocks
    auto x_ahead = [&](uint32_t kb) {
        if constexpr (X_AHEAD > 0) {
            while (xi < NBLK && xi <= kb + X_AHEAD && cb_pages_reservable_at_back(CB_IN0, (xi - kb + 1) * IN0_PAGES)) {
                issue_x(xi);
                ++xi;
            }
        }
    };
    for (uint32_t kb = 0; kb < NBLK; ++kb) {
        cb_reserve_back(CB_IN0, IN0_PAGES);
        const uint32_t a = in0_base + (kb % DEPTH) * IN0_BLOCK;
        if (sender) {
            if (xi == kb) {
                issue_x(kb);
                ++xi;
            }
            x_ahead(kb);
            noc_async_read_barrier_with_trid(TRID_X0 + (kb & 3));
#ifndef GMM_DIAG_NO_X
            if (ndest > 1) {
                wait_ge(ready, ndest - 1);
                noc_semaphore_set(ready, 0);
                const uint64_t mc = get_noc_multicast_addr(mc_x0, mc_y0, mc_x1, mc_y1, a);
                noc_async_write_multicast(a, mc, IN0_BLOCK, ndest - 1);
                // (the VALID flag after the data, same NoC and path: it lands after the block)
                noc_semaphore_set_multicast(
                    valid_addr, get_noc_multicast_addr(mc_x0, mc_y0, mc_x1, mc_y1, valid_addr), ndest - 1);
            }
#endif
            x_ahead(kb);
        } else {
#ifndef GMM_DIAG_NO_X
            noc_semaphore_set(valid, 0);
            noc_semaphore_inc(get_noc_addr(sender_x, sender_y, ready_addr), 1);
            wait_ge(valid, 1);
#endif
        }
        noc_async_read_barrier_with_trid(TRID_W0 + kb % W_INFLIGHT);
        cb_push_back(CB_IN0, IN0_PAGES);
        cb_push_back(CB_IN1, IN1_PAGES);
        if (kb + NW < NBLK) {
            cb_reserve_back(CB_IN1, NW * IN1_PAGES);  // (slots kb + 1 .. kb + NW: kb + NW is free)
            issue_w(kb + NW);
        }
    }
    noc_async_read_set_trid(0);
    noc_async_read_barrier();
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
