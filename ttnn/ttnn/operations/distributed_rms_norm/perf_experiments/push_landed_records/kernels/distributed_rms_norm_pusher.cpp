// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// distributed_rms_norm PUSHER (perf experiment push_landed_records): the second RISC of a port core (RISCV_1, NoC0).
// It never touches the fabric. Per wave, in a fixed order, it multicasts every slot its port holds the moment it is
// complete — the own slot p (only the link's own-pusher port, once every root of the link published it) then the
// entries in landing order (arrival counter >= w + 1) — from the port's landing into the push region of every compute
// core (the same address everywhere: the landing tensor spans the whole grid), each as one LINKED chain per
// destination rectangle: the data, then a multicast set of the cumulative push count on this port's push semaphore.
// Every compute core scatters its root's runs locally and finalizes itself (reader.cpp / compute.cpp, PUSH).
// It runs on the port's idle RISC so the fabric sender (port.cpp, RISCV_0 / NoC1) is never held behind a
// multicast: measured, the same multicasts issued from the sender stalled its relays by up to 6 us.
// The sender re-arms the arrival counters only after this kernel published its last count (it polls the count word).
// Raw NoC: no kernel_lib helper multicasts an L1 run linked with a semaphore set over several rectangles.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t G = get_compile_time_arg_val(0);
    constexpr uint32_t num_waves = get_compile_time_arg_val(1);
    constexpr uint32_t sem_own_stats_id = get_compile_time_arg_val(2);

    size_t a = 0;
    const uint32_t landing = get_arg_val<uint32_t>(a++);
    const uint32_t seg_stride = get_arg_val<uint32_t>(a++);
    const uint32_t seg_bytes = get_arg_val<uint32_t>(a++);
    const uint32_t push_per_wave = get_arg_val<uint32_t>(a++);
    const uint32_t push_own = get_arg_val<uint32_t>(a++);
    const uint32_t own_q = get_arg_val<uint32_t>(a++);
    const uint32_t n_roots = get_arg_val<uint32_t>(a++);
    const uint32_t push_dst = get_arg_val<uint32_t>(a++);
    const uint32_t push_sem_addr = get_semaphore(get_arg_val<uint32_t>(a++));
    const uint32_t n_rects = get_arg_val<uint32_t>(a++);  // > 0: "grid" mode, the whole slot to every rectangle
    const uint32_t rect_idx = a;
    a += 3 * 4;
    // "groups" / "roots" modes (n_rects == 0): per root of the link its record run (rec_bytes at rec_off) to its
    // destinations as row-runs: [n_dst_roots, then per root: rec_off, n_runs, (lo, hi, count) x n_runs]
    const uint32_t rec_bytes = get_arg_val<uint32_t>(a++);
    const bool unicast = get_arg_val<uint32_t>(a++) != 0;  // "roots_uc": unicast writes, one barrier, then incs
    const uint32_t n_dst_roots = get_arg_val<uint32_t>(a++);
    const uint32_t dst_idx = a;
    for (uint32_t r = 0; r < n_dst_roots; ++r) {
        a += 2 + 3 * get_arg_val<uint32_t>(a + 1);
    }
    const uint32_t n_entries = push_per_wave - push_own;
    const uint32_t slot_idx = a;  // entry i -> slot q
    a += n_entries;
    const uint32_t arrival_idx = a;  // entry i -> arrival counter address (global semaphore on this core)
    a += n_entries;

    volatile tt_l1_ptr uint32_t* own_stats =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_own_stats_id));
    volatile tt_l1_ptr uint32_t* push_val = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(push_sem_addr);
    auto mc_at = [&](uint32_t lo, uint32_t hi, uint32_t l1) -> uint64_t {
        if constexpr (noc_index == 1) {
            return get_noc_multicast_addr(hi >> 16, hi & 0xFFFF, lo >> 16, lo & 0xFFFF, l1);
        } else {
            return get_noc_multicast_addr(lo >> 16, lo & 0xFFFF, hi >> 16, hi & 0xFFFF, l1);
        }
    };
    auto mc = [&](uint32_t r, uint32_t l1) -> uint64_t {
        const uint32_t lo = get_arg_val<uint32_t>(rect_idx + 3 * r);
        const uint32_t hi = get_arg_val<uint32_t>(rect_idx + 3 * r + 1);
        if constexpr (noc_index == 1) {
            return get_noc_multicast_addr(hi >> 16, hi & 0xFFFF, lo >> 16, lo & 0xFFFF, l1);
        } else {
            return get_noc_multicast_addr(lo >> 16, lo & 0xFFFF, hi >> 16, hi & 0xFFFF, l1);
        }
    };

    const uint32_t push_total = push_per_wave * num_waves;
    for (uint32_t pushed = 0; pushed < push_total;) {
        const uint32_t w = pushed / push_per_wave;
        const uint32_t k = pushed % push_per_wave;
        uint32_t q;
        invalidate_l1_cache();
        if (push_own && k == 0) {
            if (*own_stats < (w + 1) * n_roots) {
                continue;
            }
            q = own_q;
        } else {
            const uint32_t i = k - push_own;
            if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(arrival_idx + i)) < w + 1) {
                continue;
            }
            q = get_arg_val<uint32_t>(slot_idx + i);
        }
        DeviceZoneScopedN("psh_push");
        const uint32_t off = (w * G + q) * seg_stride;
        // the previous count's source word has left L1 before a chain opens (nothing waits inside a linked chain)
        noc_async_writes_flushed();
        *push_val = ++pushed;
        for (uint32_t r = 0; r < n_rects; ++r) {
            const uint32_t cnt = get_arg_val<uint32_t>(rect_idx + 3 * r + 2);
            if (seg_bytes > 0) {
                noc_async_write_multicast(landing + off, mc(r, push_dst + off), seg_bytes, cnt, true);
            }
            noc_semaphore_set_multicast(push_sem_addr, mc(r, push_sem_addr), cnt);
        }
        if (unicast) {
            // every destination is one core (the root): its run, ONE write barrier for all, then +1 on its counter
            for (uint32_t r = 0, i = dst_idx; r < n_dst_roots; ++r, i += 5) {
                const uint32_t rec_off = get_arg_val<uint32_t>(i);
                const uint32_t lo = get_arg_val<uint32_t>(i + 2);
                if (rec_bytes > 0) {
                    noc_async_write(
                        landing + off + rec_off,
                        get_noc_addr(lo >> 16, lo & 0xFFFF, push_dst + off + rec_off),
                        rec_bytes);
                }
            }
            noc_async_write_barrier();
            for (uint32_t r = 0, i = dst_idx; r < n_dst_roots; ++r, i += 5) {
                const uint32_t lo = get_arg_val<uint32_t>(i + 2);
                noc_semaphore_inc(get_noc_addr(lo >> 16, lo & 0xFFFF, push_sem_addr), 1);
            }
            continue;
        }
        for (uint32_t r = 0, i = dst_idx; r < n_dst_roots; ++r) {
            const uint32_t rec_off = get_arg_val<uint32_t>(i);
            const uint32_t n_runs = get_arg_val<uint32_t>(i + 1);
            i += 2;
            for (uint32_t k = 0; k < n_runs; ++k, i += 3) {
                const uint32_t lo = get_arg_val<uint32_t>(i);
                const uint32_t hi = get_arg_val<uint32_t>(i + 1);
                const uint32_t cnt = get_arg_val<uint32_t>(i + 2);
                if (rec_bytes > 0) {
                    noc_async_write_multicast(
                        landing + off + rec_off, mc_at(lo, hi, push_dst + off + rec_off), rec_bytes, cnt, true);
                }
                noc_semaphore_set_multicast(push_sem_addr, mc_at(lo, hi, push_sem_addr), cnt);
            }
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
