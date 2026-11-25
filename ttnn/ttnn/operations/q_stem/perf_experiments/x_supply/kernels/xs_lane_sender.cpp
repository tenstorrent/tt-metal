// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// x_supply bench, CANDIDATE multicast lane (raw NoC multicast; bypasses the kernel_lib Mcast helper).
//
// A lane sends tiles [t_lo, t_lo + n_t) of the rounds r = r0, r0 + rstep, ... into receiver slot r % depth, then
// sets that slot's data-ready flag (its own flag word per (lane, slot)). Per-slot flags remove the helper's single
// Flag-word constraint (a second round may not be sent before the receiver has seen and cleared the first), so the
// receivers can credit up to `depth` rounds ahead and the injector multicasts back to back.
// Why raw: the Mcast helper has one data-ready word per family (Flag or Counter); it cannot address a per-slot flag,
// and its Counter mode pays a write barrier + atomic barrier per round. It also hard-codes non-posted, path-reserving
// multicast; this kernel exposes both as knobs (PATH_RESERVE, POSTED).
//
// MODE 0: the source is cb_x_round, filled by xs_reader.cpp on the other RISC (the op's injector structure).
// MODE 1: the lane reads its own tiles (2-slot local ring in cb_inj, on this RISC's NoC), overlapping the read of the
//         next round with the multicast of this one; used when both RISCs of the injector are multicast lanes.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_inj = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t round_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t n_rounds = get_compile_time_arg_val(4);
    constexpr uint32_t r0 = get_compile_time_arg_val(5);
    constexpr uint32_t rstep = get_compile_time_arg_val(6);
    constexpr uint32_t t_lo = get_compile_time_arg_val(7);
    constexpr uint32_t n_t = get_compile_time_arg_val(8);
    constexpr uint32_t depth = get_compile_time_arg_val(9);
    constexpr uint32_t receivers = get_compile_time_arg_val(10);
    constexpr uint32_t sem_credit = get_compile_time_arg_val(11);
    constexpr uint32_t sem_flag0 = get_compile_time_arg_val(12);
    constexpr bool path_reserve = get_compile_time_arg_val(13) != 0;
    constexpr bool posted = get_compile_time_arg_val(14) != 0;
    constexpr uint32_t mx0 = get_compile_time_arg_val(15);
    constexpr uint32_t my0 = get_compile_time_arg_val(16);
    constexpr uint32_t mx1 = get_compile_time_arg_val(17);
    constexpr uint32_t my1 = get_compile_time_arg_val(18);
    constexpr uint32_t num_dests = get_compile_time_arg_val(19);
    constexpr uint32_t mode = get_compile_time_arg_val(20);
    constexpr uint32_t chunk = get_compile_time_arg_val(21);  // bytes per multicast command (<= 16 KiB)
    constexpr auto x_args = TensorAccessorArgs<22>();
    constexpr uint32_t round_bytes = round_tiles * tile_bytes;
    constexpr uint32_t lane_off = t_lo * tile_bytes;
    constexpr uint32_t lane_bytes = n_t * tile_bytes;

    // receivers' slot 0 = this core's cb_x_round base (same CB layout on rect + injectors; CB still fresh here)
    const uint32_t dst_base = get_read_ptr(cb_x);
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const auto x = TensorAccessor(x_args, x_addr, tile_bytes);

    volatile tt_l1_ptr uint32_t* credit = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_credit));
    // flag source words: our own copies of the flag semaphores (never received on here), always VALID (1)
    for (uint32_t s = 0; s < depth; ++s) {
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_flag0 + s)) = 1;
    }

    auto mcast = [&](uint32_t src, uint32_t dst, uint32_t bytes) {
        while (bytes > 0) {
            const uint32_t b = bytes < chunk ? bytes : chunk;
            while (!noc_cmd_buf_ready(noc_index, write_cmd_buf));
            ncrisc_noc_fast_write<noc_mode>(
                noc_index,
                write_cmd_buf,
                src,
                get_noc_multicast_addr(mx0, my0, mx1, my1, dst, noc_index),
                b,
                NOC_MULTICAST_WRITE_VC,
                true /* mcast */,
                true /* linked: the flag below terminates the sequence */,
                num_dests,
                path_reserve,
                posted);
            src += b;
            dst += b;
            bytes -= b;
        }
    };
#ifdef XS_SETCOUNT
    // SETCOUNT signal: one monotone word per lane = rounds sent so far, published with a plain (unlinked) multicast
    // write behind the linked payload (same path, so it lands after the data). Receivers wait_min, never clear, so
    // any number of rounds may be outstanding. The source word is rewritten only after the previous publish left
    // (flushed() precedes every rewrite).
    auto flag = [&](uint32_t slot, uint32_t k) {
        const uint32_t f = get_semaphore(sem_flag0);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(f) = k + 1;
        noc_semaphore_set_multicast(f, get_noc_multicast_addr(mx0, my0, mx1, my1, f, noc_index), num_dests, false);
    };
#else
    auto flag = [&](uint32_t slot, uint32_t) {
        const uint32_t f = get_semaphore(sem_flag0 + slot);
        noc_semaphore_set_multicast(f, get_noc_multicast_addr(mx0, my0, mx1, my1, f, noc_index), num_dests, false);
    };
#endif
    auto flushed = [&]() {
        if constexpr (posted) {
            noc_async_posted_writes_flushed();
        } else {
            noc_async_writes_flushed();
        }
    };

    if constexpr (mode == 0) {
        uint32_t k = 0;
        for (uint32_t r = r0; r < n_rounds; r += rstep, ++k) {
            cb_wait_front(cb_x, round_tiles);
            const uint32_t src = get_read_ptr(cb_x) + lane_off;
            noc_semaphore_wait_min(credit, receivers * (k + 1));
            mcast(src, dst_base + (r % depth) * round_bytes + lane_off, lane_bytes);
            flag(r % depth, k);
            flushed();
            cb_pop_front(cb_x, round_tiles);
        }
    } else {
        cb_reserve_back(cb_inj, 2 * round_tiles);  // private ring, never pushed
        const uint32_t ring = get_write_ptr(cb_inj) + lane_off;
        auto issue_read = [&](uint32_t r, uint32_t slot) {
            const uint32_t off = ring + slot * round_bytes;
            for (uint32_t t = 0; t < n_t; ++t) {
                noc_async_read(x.get_noc_addr(r * round_tiles + t_lo + t), off + t * tile_bytes, tile_bytes);
            }
        };
        uint32_t k = 0;
        if (r0 < n_rounds) {
            issue_read(r0, 0);
        }
        for (uint32_t r = r0; r < n_rounds; r += rstep, ++k) {
            noc_async_read_barrier();  // round k is in ring slot k & 1
            if (r + rstep < n_rounds) {
                flushed();  // ring slot (k+1) & 1 was round k-1's source: it must have departed
                issue_read(r + rstep, (k + 1) & 1);  // overlaps this round's multicast
            }
            noc_semaphore_wait_min(credit, receivers * (k + 1));
            mcast(ring + (k & 1) * round_bytes, dst_base + (r % depth) * round_bytes + lane_off, lane_bytes);
#ifdef XS_SETCOUNT
            flushed();  // the counter word's previous publish must have left before it is rewritten
#endif
            flag(r % depth, k);
        }
    }
    noc_async_write_barrier();
    noc_async_posted_writes_flushed();
}
