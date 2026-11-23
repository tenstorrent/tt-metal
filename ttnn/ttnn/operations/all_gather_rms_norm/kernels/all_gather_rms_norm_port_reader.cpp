// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// all_gather_rms_norm port reader (NCRISC / NoC0), one per (direction, link). gather_normalized_slices, read side
// (TRANSPORT; fabric_all_gather's port reader with two changes):
//   own entry (k = 0)   waits until every active worker announced its normalized block resident (own_ready), then
//                       PULLS the pages from the workers' cb_out_block (same L1 address on every core of the chip)
//                       straight into the port CB, packed in destination-run order: no DRAM re-read of the own slice.
//   relay entry k >= 1  waits the arrival counter for the chunk (upstream's increment every inc_every chunks and on
//                       each entry's last; threshold A + (B + c) / inc_every, A / B host-computed), then reads the run
//                       from the local destination (output pages / R_U staging), contiguous in one bank.
//   end                 waits every expected arrival; R_U: announces "landed" to every active worker (after own_ready,
//                       so the own staging slot is complete too); re-arms the arrival counter (trace-safe).
// Batches as in the example: a batch ends at a multiple of `group` chunks in the 2-group CB or at an entry end.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "agrn_port_common.hpp"

using namespace agrn_port;

void kernel_main() {
    constexpr auto dst_args = TensorAccessorArgs<PORT_CT_ARGS>();

    size_t a = 0;
    const uint32_t dst_addr = get_arg_val<uint32_t>(a++);
    const uint32_t first_bank = get_arg_val<uint32_t>(a++);
    const uint32_t bank_stride = get_arg_val<uint32_t>(a++);
    const uint32_t arrival_addr = get_arg_val<uint32_t>(a++);
    const uint32_t expect_in = get_arg_val<uint32_t>(a++);
    const uint32_t n_active = get_arg_val<uint32_t>(a++);
    const uint32_t notify_landed = get_arg_val<uint32_t>(a++);
    const uint32_t n_entries = get_arg_val<uint32_t>(a++);
    const uint32_t entries_idx = a;
    a += 3 * n_entries;
    const uint32_t workers_idx = a;  // n_active worker NoC coords (x | y << 16), ordered worker index

    const auto dst = TensorAccessor(dst_args, dst_addr, page_bytes);
    volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_addr);
    volatile tt_l1_ptr uint32_t* own_ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(agrn::sem_own_ready));
    // cb_out_block is declared on every core of the program with one descriptor: same base address everywhere.
    const uint32_t out_block_base = get_write_ptr(agrn::cb_out_block);

    constexpr uint32_t rows_lo_div = rows_hi > 1 ? rows_hi - 1 : 1;
    uint32_t batch = 0, need = 0, pos = 0, wptr = 0;
    for (uint32_t k = 0; k < n_entries; ++k) {
        const uint32_t e = get_arg_val<uint32_t>(entries_idx + 3 * k);
        const uint32_t A = get_arg_val<uint32_t>(entries_idx + 3 * k + 1);
        const uint32_t B = get_arg_val<uint32_t>(entries_idx + 3 * k + 2);
        const uint32_t q = e & 0xFFFF, part = e >> 16;
        if (k == 0) {
            AGRN_ZONE("pr_wait_own");
            noc_semaphore_wait_min(own_ready, n_active);
        }
        AGRN_ZONE("pr_entry");
        uint32_t c = 0;
        for_each_packet(q, part, first_bank, bank_stride, [&](const Packet& pk) {
            if (batch == 0) {
                need = group - pos % group;
                cb_reserve_back(cb_chunks, run_pages * need);
                wptr = get_write_ptr(cb_chunks);
            }
            uint32_t l1 = wptr + batch * chunk_bytes;
            if (k > 0) {
                noc_semaphore_wait_min(arrived, A + (B + c) / inc_every);
            }
            for (uint32_t ri = 0; ri < pk.nr; ++ri) {
                const uint32_t n = pk.n[ri];
                if (k == 0) {
                    for (uint32_t t = 0; t < n; ++t) {
                        const uint32_t f = pk.jj[ri] + t * nb;
                        const uint32_t r = flat_slot ? f / Wlt : pk.rr[ri];
                        const uint32_t j = flat_slot ? f - r * Wlt : f;
                        uint32_t g, rs;
                        if (r < big * rows_hi) {
                            g = r / rows_hi;
                            rs = g * rows_hi;
                        } else {
                            g = big + (r - big * rows_hi) / rows_lo_div;
                            rs = big * rows_hi + (g - big) * (rows_hi - 1);
                        }
                        const uint32_t m = j / cw_nom;
                        const uint32_t ws = m * cw_nom;
                        const uint32_t wm = (Wlt - ws) < cw_nom ? (Wlt - ws) : cw_nom;
                        const uint32_t xy = get_arg_val<uint32_t>(workers_idx + g * K + m);
                        const uint32_t src = out_block_base + ((r - rs) * wm + (j - ws)) * page_bytes;
                        noc_async_read(get_noc_addr(xy & 0xFFFF, xy >> 16, src), l1 + t * page_bytes, page_bytes);
                    }
                } else {
                    noc_async_read(dst.get_noc_addr(pk.first[ri]), l1, n * page_bytes);
                }
                l1 += n * page_bytes;
            }
            ++c;
            if (++batch == need) {
                noc_async_read_barrier();
                cb_push_back(cb_chunks, run_pages * batch);
                pos += batch;
                batch = 0;
            }
        });
        if (batch > 0) {
            noc_async_read_barrier();
            cb_push_back(cb_chunks, run_pages * batch);
            pos += batch;
            batch = 0;
        }
    }

    if (expect_in > 0) {
        AGRN_ZONE("pr_wait_all");
        noc_semaphore_wait_min(arrived, expect_in);
    }
    if (notify_landed) {
        noc_semaphore_wait_min(own_ready, n_active);
        const uint32_t sem = get_semaphore(agrn::sem_landed);
        for (uint32_t i = 0; i < n_active; ++i) {
            const uint32_t xy = get_arg_val<uint32_t>(workers_idx + i);
            noc_semaphore_inc(get_noc_addr(xy & 0xFFFF, xy >> 16, sem), 1);
        }
    }
    if (expect_in > 0) {
        noc_semaphore_inc(get_noc_addr(arrival_addr), 0u - expect_in);
    }
    noc_async_atomic_barrier();
}
