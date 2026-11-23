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
// Reads are pipelined per packet (read_depth packets in flight, one NoC transaction id each) and each packet is pushed
// as soon as its reads landed; the sender pops on group boundaries of the 2-group CB.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "agrn_port_common.hpp"
#if defined(AGRN_ABL_TIMING)
#include "api/debug/device_print.h"
#define TT_(...) __VA_ARGS__
#else
#define TT_(...)
#endif

using namespace agrn_port;

#if defined(AGRN_OWN_FROM_DST)
constexpr bool own_from_dst = true;  // perf probe override
#else
constexpr bool own_from_dst = own_from_dst_ct != 0;
#endif

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
    a += n_active;
    const uint32_t hub_sem = get_arg_val<uint32_t>(a++);   // 0: not a hub; else 1 + the stats-ready semaphore id
    const uint32_t mc_start = get_arg_val<uint32_t>(a++);  // program grid rectangle (NoC0 coords x | y << 16)
    const uint32_t mc_end = get_arg_val<uint32_t>(a++);
    const uint32_t mc_dests = get_arg_val<uint32_t>(a++);  // cores in the rectangle, excluding this one

    const auto dst = TensorAccessor(dst_args, dst_addr, page_bytes);
    volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_addr);
    volatile tt_l1_ptr uint32_t* own_ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(agrn::sem_own_ready));
    // cb_out_block is declared on every core of the program with one descriptor: same base address everywhere.
    const uint32_t out_block_base = get_write_ptr(agrn::cb_out_block);

    constexpr uint32_t rows_lo_div = rows_hi > 1 ? rows_hi - 1 : 1;
    // announce_stats_ready (OVERLAP): once the sender saw the table complete, ONE multicast semaphore set over the
    // whole program grid (this core included in the rectangle, not written: non-loopback) replaces a unicast per worker
    if (hub_sem > 0) {
        AGRN_ZONE("pr_stats_mcast");
        volatile tt_l1_ptr uint32_t* done =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(agrn::sem_stats_done));
        noc_semaphore_wait_min(done, 1);
        const uint32_t sem = get_semaphore(hub_sem - 1);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem) = 1;
        noc_semaphore_set_multicast(
            sem,
            get_noc_multicast_addr(mc_start & 0xFFFF, mc_start >> 16, mc_end & 0xFFFF, mc_end >> 16, sem),
            mc_dests);
        noc_async_write_barrier();
    }
    // Pipelined reads: packet i is read into chunk slot (pushed + inflight) of the 2-group CB under NoC transaction id
    // 1 + i % read_depth; once read_depth packets are in flight the oldest is waited (its trid only) and pushed, so
    // the sender starts on a packet as soon as its own reads landed and reads of later packets keep the NoC busy.
    constexpr uint32_t n_slots = 2 * group;
    const uint32_t chunks_base = get_write_ptr(cb_chunks);  // nothing pushed yet: the CB's first slot
    const uint32_t desc_base = get_write_ptr(cb_desc);
    uint32_t pushed = 0, inflight = 0, issued = 0;
    TT_(uint32_t t_bar = 0, t_res = 0, t_iss = 0, t_tot = 0, t_arr = 0; uint32_t t_start = 0;)
    auto retire_oldest = [&]() {
        TT_(uint32_t tb = get_timestamp_32b();)
        noc_async_read_barrier_with_trid(1 + (issued - inflight) % read_depth);
        TT_(t_bar += get_timestamp_32b() - tb;)
        cb_push_back(cb_chunks, run_pages);
        ++pushed;
        --inflight;
    };
    for (uint32_t k = 0; k < n_entries; ++k) {
        const uint32_t e = get_arg_val<uint32_t>(entries_idx + 3 * k);
        const uint32_t A = get_arg_val<uint32_t>(entries_idx + 3 * k + 1);
        const uint32_t B = get_arg_val<uint32_t>(entries_idx + 3 * k + 2);
        const uint32_t q = e & 0xFFFF, part = e >> 16;
        if (k == 0) {
            AGRN_ZONE("pr_wait_own");
            noc_semaphore_wait_min(own_ready, n_active);
        }
        if (k == 0) {
            AGRN_ZONE("pr_own_first");  // marker: own entry starts
        }
        TT_(
            if (k == 0) { t_start = get_timestamp_32b(); } if (k == 1) {
                t_tot = get_timestamp_32b() - t_start;
                DEVICE_PRINT("AGRN_R own tot={} bar={} res={} iss={} pk={}\n", t_tot, t_bar, t_res, t_iss, issued);
            })
#if defined(AGRN_ABL_MINSEND)
        if (k == 0) {
            continue;  // perf ablation only: the sender sends the own entry without a source
        }
#endif
        uint32_t c = 0;
        for_each_packet(q, part, first_bank, bank_stride, [&](const Packet& pk) {
            if (inflight == read_depth) {
                retire_oldest();
            }
            TT_(uint32_t tr = get_timestamp_32b();)
            cb_reserve_back(cb_chunks, run_pages * (inflight + 1));
            TT_(t_res += get_timestamp_32b() - tr;)
            uint32_t l1 = chunks_base + ((pushed + inflight) % n_slots) * chunk_bytes;
            if (k > 0) {
                const uint32_t need_arrivals = A + (B + c) / inc_every;
                if (*arrived < need_arrivals) {
                    // hand every packet already read to the sender before blocking: the arrivals this waits for
                    // may depend on them (the peers' relays of my earlier packets)
                    while (inflight > 0) {
                        retire_oldest();
                    }
                    noc_semaphore_wait_min(arrived, need_arrivals);
                }
            }
            TT_(uint32_t ti = get_timestamp_32b();) {
                volatile tt_l1_ptr PacketDesc* dsc =
                    reinterpret_cast<volatile tt_l1_ptr PacketDesc*>(desc_base + (issued % n_slots) * DESC_BYTES);
                dsc->nr = pk.nr;
                dsc->pages = pk.pages;
                for (uint32_t ri = 0; ri < pk.nr; ++ri) {
                    dsc->first[ri] = pk.first[ri];
                    dsc->n[ri] = pk.n[ri];
                }
            }
            noc_async_read_set_trid(1 + issued % read_depth);
            for (uint32_t ri = 0; ri < pk.nr; ++ri) {
                const uint32_t n = pk.n[ri];
#if defined(AGRN_ABL_NOREAD)
                if (k == 0) {  // perf ablation only: the own entry is sent without reading it (wrong output)
                } else
#endif
                    if (k == 0 && !own_from_dst) {
                    for (uint32_t t = 0; t < n; ++t) {
                        const uint32_t f = pk.jj[ri] + t * run_stride;
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
            TT_(t_iss += get_timestamp_32b() - ti;)
            ++issued;
            ++inflight;
            ++c;
        });
    }
    while (inflight > 0) {
        retire_oldest();
    }
    noc_async_read_set_trid(0);

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
