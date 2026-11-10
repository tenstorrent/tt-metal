// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TailEngine gather port reader (RISCV_1, NoC0): my units' shares from this chip's owners (once landed / tilized
// there), into the staging CB - for each token loop in turn (the gate a loop: tail_engine_common.hpp).

#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/mb_args.hpp"
#include "tail_engine_port_common.hpp"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    mb_args::load();  // (my args: this chip's block of the DRAM table, common/args_table.hpp)
    const uint32_t land = get_write_ptr(CB_LAND), poll = get_write_ptr(CB_POLL);
    size_t a = 0;
    const uint32_t sl = get_write_ptr(CB_WD);  // (the owners' landed words: counted per loop)
    const uint32_t n = mb_args::arg(a++);
    const uint32_t units = a, ak = units + n, xs = ak + 2 * GX, ys = xs + GX;  // (unit: part | position here << 8)
    const uint32_t gw = mb_args::arg(ys + PAIRS);  // (the engine's gate lead: x | y << 16)
    volatile tt_l1_ptr uint32_t* pw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(poll);
    // a group: PG packets' slots reserved, read, one barrier, pushed whole (padded) with its packet count in CBM - a
    // share not landed yet ends the group early (never hold read packets while waiting)
    // (two groups in flight: a group's pages pushed once its reads are issued, its packet count (CBM - what the sender
    // waits for first) once they landed - its transaction id's barrier, after the next group's reads went out: a
    // barrier a group was the reader's - and the gather's - pace, ~16 GB/s a link of 23.7)
    uint32_t b = 0, wp = 0, gt = 1, pend_b = 0, pend_t = 0;
    auto land_pending = [&]() {
        if (pend_t) {
            noc_async_read_barrier_with_trid(pend_t);
            cb_reserve_back(CBM, 1);
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CBM)) = pend_b;
            cb_push_back(CBM, 1);
            pend_t = 0;
        }
    };
    auto flush = [&]() {
        cb_push_back(CBP, PG * RP);
        land_pending();
        pend_b = b, pend_t = gt, gt = 3 - gt, b = 0;
    };
    // (the shares' landed words polled a unit at a time: every not-yet-seen share's word in one batch of NoC reads,
    // one barrier - one round trip a share in turn was the gather's pace; a share still not in is then waited alone)
    bool seen[RT * GX];
    const uint32_t pbuf = poll + 16;  // (16 B a share, past the poll word)
    for (uint32_t l = 0; l < LOOPS; ++l) {
        const uint32_t ep = l + 1;  // (loop l's landed words: l + 1 once a share landed)
        wait_zero_gate(PZR);        // (loop 0: the owners' words zeroed; loop l: everyone done with loop l - 1)
        for (uint32_t u = 0; u < n; ++u) {
            DeviceZoneScopedN("TP-RUNIT");
            const uint32_t w = mb_args::arg(units + u), part = w & 0xFF, pos = w >> 8;
            for (uint32_t i = 0; i < RT * GX; ++i) {
                seen[i] = false;
            }
            auto poll_all = [&]() {
                for (uint32_t r = 0; r < RT; ++r) {
                    const uint32_t t = pos * RT + r;
                    for (uint32_t c = 0; c < GX; ++c) {
                        if (seen[r * GX + c] || (part == 1 && c >= GX / 2) || (part == 2 && c < GX / 2)) {
                            continue;
                        }
                        noc_async_read(
                            get_noc_addr(
                                mb_args::arg(xs + c),
                                mb_args::arg(ys + t % PAIRS),
                                sl + 16 * (t / PAIRS)),
                            pbuf + 16 * (r * GX + c),
                            4);
                    }
                }
                noc_async_read_barrier();
                invalidate_l1_cache();
                for (uint32_t i = 0; i < RT * GX; ++i) {
                    if (!seen[i] && *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pbuf + 16 * i) >= ep) {
                        seen[i] = true;
                    }
                }
            };
            walk(part, pos, ak, [&](uint32_t pr, uint32_t s, uint32_t cx, uint32_t k0, uint32_t t0, uint32_t m, bool) {
                const uint32_t ox = mb_args::arg(xs + cx), oy = mb_args::arg(ys + pr);
                const uint32_t si = (s * PAIRS + pr - pos * RT) * GX + cx;
                if (t0 == k0 &&
                    !seen[si]) {  // (the share landed / tilized at its owner here: its landed word, over the NoC)
                    if (b == 0) {
                        poll_all();  // (no reads in flight of mine: the batch's barrier waits for its words only)
                    }
                    auto in = [&]() {
                        noc_async_read(get_noc_addr(ox, oy, sl + 16 * s), poll, 4);
                        noc_async_read_barrier();
                        invalidate_l1_cache();  // (the copy just landed: not the cached word)
                        return *pw >= ep;
                    };
                    if (!seen[si] && !in()) {
                        if (b > 0) {
                            flush();
                        }
                        land_pending();                 // (the sender has everything read so far while I wait)
                        DeviceZoneScopedN("TP-RWAIT");  // (a share not landed / tilized yet)
                        while (!in()) {
                        }
                    }
                }
                if (b == 0) {
                    cb_reserve_back(CBP, PG * RP);
                    wp = get_write_ptr(CBP);
                }
                noc_async_read_set_trid(gt);
                noc_async_read(get_noc_addr(ox, oy, land + (s * SHMAX + t0 - k0) * 2048), wp + b * RP * 2048, m * 2048);
                if (++b == PG) {
                    flush();
                }
            });
        }
        if (b > 0) {
            flush();
        }
        land_pending();  // (every landing slot read of loop l has landed here)
        if (l + 1 < LOOPS) {  // (done with loop l's landing slots: count in on the engine's gate)
            noc_semaphore_inc(get_noc_addr(gw & 0xFFFF, gw >> 16, get_semaphore(5 /* the engine's SZ */)), 1);
            noc_async_atomic_barrier();
        }
    }
}

