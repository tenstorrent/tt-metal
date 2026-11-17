// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Hub (RISCV_0, NoC0): the stats collection point of one chip. Communication only, no compute.      [OVERLAP]
//   broadcast_gamma  (GAMMA_HUB_MCAST, R6) read the Wt gamma sticks once, multicast them into the compute cores'
//                    stats-scratch tail, then a multicast set of their gamma_go
//   forward_stats    own_partials == nb (every own core exported its rows): slot p -> both link-0 ports'
//                    stats scratch, stats_tx++ on each; as the relayed slots arrive (stats_fwd >= i / stats_bwd >= i)
//                    hand them to the forward / backward port the same way
//   broadcast_stats  all G slots in: multicast the whole stats scratch (G * Mt * 128 B) to the compute-core
//                    rectangles, then a multicast set of their stats_ready
//   drain_stats      stats_fwd -= kf, stats_bwd -= kb (re-arm by what was consumed)
// Raw multicast: the compute cores form one or two rectangles (full rows + a partial row) of unequal size, which the
// McastArgs helper's equal-group pipe does not describe.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t G = get_compile_time_arg_val(0);
constexpr uint32_t slot_bytes = get_compile_time_arg_val(1);
constexpr uint32_t sem_own_partials = get_compile_time_arg_val(2);
constexpr uint32_t sem_stats_tx = get_compile_time_arg_val(3);
constexpr uint32_t sem_stats_ready = get_compile_time_arg_val(4);
constexpr uint32_t sem_gamma_go = get_compile_time_arg_val(5);
constexpr bool gamma_hub_mcast = get_compile_time_arg_val(6) != 0;
constexpr uint32_t gamma_page = get_compile_time_arg_val(7);
constexpr uint32_t Wt = get_compile_time_arg_val(8);
constexpr auto gamma_args = TensorAccessorArgs<9>();

void kernel_main() {
    size_t a = 0;
    const uint32_t stats_addr = get_arg_val<uint32_t>(a++);
    const uint32_t own_count = get_arg_val<uint32_t>(a++);
    const uint32_t p = get_arg_val<uint32_t>(a++);
    const uint32_t fwd_xy = get_arg_val<uint32_t>(a++);
    const uint32_t bwd_xy = get_arg_val<uint32_t>(a++);
    const uint32_t kf = get_arg_val<uint32_t>(a++);  // slots the forward port sends (own + relays) = slots arriving fwd
    const uint32_t kb = get_arg_val<uint32_t>(a++);
    const uint32_t sfwd_addr = get_arg_val<uint32_t>(a++);
    const uint32_t sbwd_addr = get_arg_val<uint32_t>(a++);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(a++);
    const uint32_t n_rects = get_arg_val<uint32_t>(a++);  // then x0, y0, x1, y1, n_cores (virtual, NoC0 order)
    const size_t rects_idx = a;

    volatile tt_l1_ptr uint32_t* own_partials =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_own_partials));
    volatile tt_l1_ptr uint32_t* sfwd = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sfwd_addr);
    volatile tt_l1_ptr uint32_t* sbwd = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sbwd_addr);
    const uint32_t stats_tx_addr = get_semaphore(sem_stats_tx);

    auto hand_over = [&](uint32_t q, uint32_t xy) {
        const uint32_t off = stats_addr + q * slot_bytes;
        noc_async_write(off, get_noc_addr(xy >> 8, xy & 0xFF, off), slot_bytes);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(xy >> 8, xy & 0xFF, stats_tx_addr), 1);
    };

    // multicast destination over compute-core rectangle r (NoC0 virtual coords from the host)
    auto rect = [&](uint32_t r, uint32_t addr) {
        const size_t ri = rects_idx + 5 * r;
        return get_noc_multicast_addr(
            get_arg_val<uint32_t>(ri),
            get_arg_val<uint32_t>(ri + 1),
            get_arg_val<uint32_t>(ri + 2),
            get_arg_val<uint32_t>(ri + 3),
            addr);
    };
    auto rect_cores = [&](uint32_t r) { return get_arg_val<uint32_t>(rects_idx + 5 * r + 4); };

    // ---- broadcast_gamma (R6): the Wt sticks, read once per chip, into every compute core's scratch tail ----
    // Runs before the own partials can arrive (the hub has nothing else to do yet). Replaces 20 cores per slice
    // reading the same pages; each compute reader waits gamma_go, then copies its block_w sticks from local L1.
    if constexpr (gamma_hub_mcast) {
        const uint32_t gbase = (stats_addr + G * slot_bytes + 63) & ~63u;  // DRAM reads land 64 B aligned
        const auto gamma = TensorAccessor(gamma_args, gamma_addr, gamma_page);
        for (uint32_t n = 0; n < Wt; ++n) {
            noc_async_read(gamma.get_noc_addr(n), gbase + n * gamma_page, gamma_page);
        }
        noc_async_read_barrier();
        for (uint32_t r = 0; r < n_rects; ++r) {
            noc_async_write_multicast(gbase, rect(r, gbase), Wt * gamma_page, rect_cores(r));
        }
        noc_async_write_barrier();
        const uint32_t go_addr = get_semaphore(sem_gamma_go);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_addr) = 1;
        for (uint32_t r = 0; r < n_rects; ++r) {
            noc_semaphore_set_multicast(go_addr, rect(r, go_addr), rect_cores(r));
        }
    }

    // ---- forward_stats ----
    noc_semaphore_wait_min(own_partials, own_count);
    if (kf > 0) {
        hand_over(p, fwd_xy);
    }
    if (kb > 0) {
        hand_over(p, bwd_xy);
    }
    for (uint32_t i = 1; i < kf; ++i) {  // the forward port relays P_{p-1}, ..., P_{p-kf+1}
        noc_semaphore_wait_min(sfwd, i);
        hand_over((p + G - i) % G, fwd_xy);
    }
    for (uint32_t i = 1; i < kb; ++i) {
        noc_semaphore_wait_min(sbwd, i);
        hand_over((p + i) % G, bwd_xy);
    }
    noc_semaphore_wait_min(sfwd, kf);
    noc_semaphore_wait_min(sbwd, kb);

    // ---- broadcast_stats ----
    const uint32_t ready_addr = get_semaphore(sem_stats_ready);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr) = 1;
    for (uint32_t r = 0; r < n_rects; ++r) {
        const uint32_t x0 = get_arg_val<uint32_t>(rects_idx + 5 * r + 0);
        const uint32_t y0 = get_arg_val<uint32_t>(rects_idx + 5 * r + 1);
        const uint32_t x1 = get_arg_val<uint32_t>(rects_idx + 5 * r + 2);
        const uint32_t y1 = get_arg_val<uint32_t>(rects_idx + 5 * r + 3);
        const uint32_t n = get_arg_val<uint32_t>(rects_idx + 5 * r + 4);
        noc_async_write_multicast(stats_addr, get_noc_multicast_addr(x0, y0, x1, y1, stats_addr), G * slot_bytes, n);
    }
    noc_async_write_barrier();
    for (uint32_t r = 0; r < n_rects; ++r) {
        const uint32_t x0 = get_arg_val<uint32_t>(rects_idx + 5 * r + 0);
        const uint32_t y0 = get_arg_val<uint32_t>(rects_idx + 5 * r + 1);
        const uint32_t x1 = get_arg_val<uint32_t>(rects_idx + 5 * r + 2);
        const uint32_t y1 = get_arg_val<uint32_t>(rects_idx + 5 * r + 3);
        const uint32_t n = get_arg_val<uint32_t>(rects_idx + 5 * r + 4);
        noc_semaphore_set_multicast(ready_addr, get_noc_multicast_addr(x0, y0, x1, y1, ready_addr), n);
    }
    noc_async_write_barrier();

    // ---- drain_stats ----
    noc_semaphore_inc(get_noc_addr(sfwd_addr), 0u - kf);
    noc_semaphore_inc(get_noc_addr(sbwd_addr), 0u - kb);
    noc_async_atomic_barrier();
}
