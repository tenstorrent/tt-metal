// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// all_gather_rms_norm port reader (NCRISC / NoC0), one per (direction, link). gather_normalized_slices, read side
// (TRANSPORT; fabric_all_gather's port reader, adapted):
//   own entry (k = 0)   waits until every active worker announced its normalized block written (own_ready), then
//                       reads each bank run back from this chip's own destination pages (own_from_dst, default: one
//                       read per run) or pulls the pages one by one from the workers' cb_out_block (L1).
//   relay entry k >= 1  waits the arrival counter for the chunk (upstream's increment every inc_every chunks and on
//                       each entry's last; threshold A + (B + c) / inc_every, A / B host-computed), then reads the run
//                       from the local destination (output pages / R_U staging), contiguous in one bank.
//   end                 waits every expected arrival; R_U: announces "landed" to every active worker (after own_ready,
//                       so the own staging slot is complete too); re-arms the arrival counter (trace-safe).
// The bulk is bound by per-packet work on the port core, so this reader is the only one that walks the packets: each
// packet's runs go to the sender in cb_desc (one page per chunk slot). Reads are pipelined per packet (read_depth
// packets in flight, one NoC transaction id each) and each packet is pushed as soon as its own reads landed.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "agrn_port_common.hpp"
#include "ppb_common.hpp"

using namespace agrn_port;

constexpr bool own_from_dst = own_from_dst_ct != 0;

void kernel_main() {
    constexpr auto dst_args = TensorAccessorArgs<PORT_CT_ARGS>();

    size_t a = 0;
    const uint32_t dst_addr = get_arg_val<uint32_t>(a++);
    const uint32_t first_bank = get_arg_val<uint32_t>(a++);
    const uint32_t bank_stride = get_arg_val<uint32_t>(a++);
    const uint32_t arrival_addr = get_arg_val<uint32_t>(a++);
#if defined(AGRN_ABL_NO_BULK)
    const uint32_t expect_in = 0 * get_arg_val<uint32_t>(a++);  // ablation: bulk stubbed
#else
    const uint32_t expect_in = get_arg_val<uint32_t>(a++);
#endif
    const uint32_t n_active = get_arg_val<uint32_t>(a++);
    const uint32_t notify_landed = get_arg_val<uint32_t>(a++);
    const uint32_t n_entries_args = get_arg_val<uint32_t>(a++);
#if defined(AGRN_ABL_NO_BULK)
    const uint32_t n_entries = 0;  // ablation: bulk stubbed (output garbage; perf ablation only)
#else
    const uint32_t n_entries = n_entries_args;
#endif
    const uint32_t entries_idx = a;
    a += 3 * n_entries_args;
    const uint32_t workers_idx = a;  // n_active worker NoC coords (x | y << 16), ordered worker index
    a += n_active;
    const uint32_t hub_sem = get_arg_val<uint32_t>(a++);   // 0: not a hub; else 1 + the stats-ready semaphore id
    const uint32_t mc_start = get_arg_val<uint32_t>(a++);  // program grid rectangle (NoC0 coords x | y << 16)
    const uint32_t mc_end = get_arg_val<uint32_t>(a++);
    const uint32_t mc_dests = get_arg_val<uint32_t>(a++);  // cores in the rectangle, excluding this one
    // hubs: stats-table multicast (OVERLAP, Refinement 3) — every slot this hub holds is pushed into the same table
    // address of every core of the program grid as soon as it is complete, so the workers gather from local L1
    const uint32_t stats_arrived_addr = get_arg_val<uint32_t>(a++);
    const uint32_t slot_bytes = get_arg_val<uint32_t>(a++);
    const uint32_t mc_own_slot = get_arg_val<uint32_t>(a++);  // 0xFFFF: not this hub's; else the own slot index
    const uint32_t n_rec_writers = get_arg_val<uint32_t>(a++);
    const uint32_t n_recv = get_arg_val<uint32_t>(a++);  // slots landing in my table, in arrival order
    const uint32_t recv_idx = a;
    a += n_recv;
    // own_bands > 1: per band the announcements it needs (k per row-group), its end tile-row, its counter address
    const uint32_t band_need_idx = a;
    const uint32_t band_end_idx = a + own_bands;
    const uint32_t band_addr_idx = a + 2 * own_bands;
    a += 3 * own_bands;
    // own-entry staging (Perf 1, OVERLAP): my lane's own walk lies in run_pages chunks in cb_stage of the active
    // workers (chunk g = lane * nch + c, home: agrn_worker_args.hpp stage_chunk_home); every worker increments my
    // readiness counter (st_ctr, this core's L1) once its pages landed there. Every port re-arms it, staged or not.
    const bool st_on = get_arg_val<uint32_t>(a++) != 0;
    uint32_t st_pages = 0, st_ctr = 0, st_nact = 1, st_nch = 0, st_lane = 0, st_L = 1, st_cp = 1;
    if (st_on) {
        st_pages = get_arg_val<uint32_t>(a++);  // staged pages of my lane walk (a prefix)
        st_ctr = get_arg_val<uint32_t>(a++);
        st_nact = get_arg_val<uint32_t>(a++);
        st_nch = get_arg_val<uint32_t>(a++);
        st_lane = get_arg_val<uint32_t>(a++);
        st_L = get_arg_val<uint32_t>(a++);
        st_cp = get_arg_val<uint32_t>(a++);
    }
    const uint32_t ppb_H = get_arg_val<uint32_t>(a++);  // parallel_packet_builders: my helpers
    const uint32_t ppb_helpers_idx = a;
    a += ppb_H;
    const uint32_t st_tbl = workers_idx;  // the staging cores are the active workers (same NoC coord list)

    const auto dst = TensorAccessor(dst_args, dst_addr, page_bytes);
    // NoC address of every destination bank's first page and the bank page pitch: a run's read address is then
    // base + offset (no accessor lookup per run; interleaved DRAM destination)
    constexpr bool fast_banks = shard_w == 0 && nb <= 16;
    uint64_t bank_base[fast_banks ? nb : 1];
    uint32_t bank_pitch = 0;
    if constexpr (fast_banks) {
        for (uint32_t b = 0; b < nb; ++b) {
            bank_base[b] = dst.get_noc_addr(b);
        }
        bank_pitch = static_cast<uint32_t>(dst.get_noc_addr(nb) - dst.get_noc_addr(0));
    }
    volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_addr);
    volatile tt_l1_ptr uint32_t* own_ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(agrn::sem_own_ready));
    // cb_out_block is declared on every core of the program with one descriptor: same base address everywhere.
    const uint32_t out_block_base = get_write_ptr(agrn::cb_out_block);

    constexpr uint32_t rows_lo_div = rows_hi > 1 ? rows_hi - 1 : 1;
    // announce_stats_ready (OVERLAP): each slot of my table is multicast over the whole program grid as soon as it is
    // complete (the own slot once every record writer wrote it — fwd hub only — and each upstream slot on its
    // arrival), so every worker's own cb_stats_table holds the slots and the gather reads local L1 instead of 80+
    // workers converging on the two hubs. Once the sender saw the table complete: ONE multicast semaphore set (this
    // core included in the rectangle, not written: non-loopback).
    if (hub_sem > 0) {
        AGRN_ZONE("pr_stats_mcast");
        const uint32_t table = get_write_ptr(agrn::cb_stats_table);  // same address on every core
        auto mcast_slot = [&](uint32_t q, bool linked) {
            const uint32_t src = table + q * slot_bytes;
            noc_async_write_multicast(
                src,
                get_noc_multicast_addr(mc_start & 0xFFFF, mc_start >> 16, mc_end & 0xFFFF, mc_end >> 16, src),
                slot_bytes,
                mc_dests,
                linked);
        };
        // every slot but the last is multicast as soon as it is complete; the last one goes out linked to the
        // stats-ready flag (kernel_lib mcast pipe's Flag protocol: a linked payload terminated by an unlinked
        // semaphore set lands before it, no ack round trip in between)
        const bool has_own = mc_own_slot != 0xFFFF;
        const uint32_t n_mc = (has_own ? 1u : 0u) + n_recv;
        auto slot_of = [&](uint32_t i) {
            if (has_own) {
                return i == 0 ? mc_own_slot : get_arg_val<uint32_t>(recv_idx + i - 1);
            }
            return get_arg_val<uint32_t>(recv_idx + i);
        };
        volatile tt_l1_ptr uint32_t* st_arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stats_arrived_addr);
        auto wait_complete = [&](uint32_t i) {
            if (has_own && i == 0) {
                noc_semaphore_wait_min(
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(agrn::sem_records_local)),
                    n_rec_writers);
            } else {
                noc_semaphore_wait_min(st_arrived, i + 1 - (has_own ? 1u : 0u));  // upstream's slot landed
            }
        };
        for (uint32_t i = 0; i + 1 < n_mc; ++i) {
            wait_complete(i);
            mcast_slot(slot_of(i), false);
        }
        if (n_mc > 0) {
            wait_complete(n_mc - 1);
        }
        volatile tt_l1_ptr uint32_t* done =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(agrn::sem_stats_done));
        noc_semaphore_wait_min(done, 1);  // the sender is past its last wait on stats_arrived
        noc_async_write_barrier();        // the earlier slots (long since landed)
        if (n_mc > 0) {
            mcast_slot(slot_of(n_mc - 1), true);
        }
        const uint32_t sem = get_semaphore(hub_sem - 1);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem) = 1;
        noc_semaphore_set_multicast(
            sem,
            get_noc_multicast_addr(mc_start & 0xFFFF, mc_start >> 16, mc_end & 0xFFFF, mc_end >> 16, sem),
            mc_dests);
        noc_async_write_barrier();
        if (n_recv > 0) {
            noc_semaphore_inc(get_noc_addr(stats_arrived_addr), 0u - n_recv);  // re-arm (trace-safe)
            noc_async_atomic_barrier();
        }
    }
    // parallel_packet_builders: the helpers read and build every bulk packet; this RISC only forwards what they wait
    // for — own-slice readiness (SEM_OWN: 1 = staged prefix ready, 2 = own pages written) and the arrival counter
    // (SEM_MIRROR) — with inline writes into every helper's L1.
    (void)st_tbl;
    (void)out_block_base;
    if (n_entries > 0 && ppb_H > 0) {
        AGRN_ZONE("pr_forward");
        auto to_helpers = [&](uint32_t sem, uint32_t v) {
            const uint32_t addr = get_semaphore(sem);
            for (uint32_t hh = 0; hh < ppb_H; ++hh) {
                const uint32_t xy = get_arg_val<uint32_t>(ppb_helpers_idx + hh);
                noc_inline_dw_write(get_noc_addr(xy & 0xFFFF, xy >> 16, addr), v);
            }
        };
        uint32_t own_lvl = 0, last = 0;
        while (own_lvl < 2 || last < expect_in) {
            invalidate_l1_cache();
            if (own_lvl == 0 && (!st_on || *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(st_ctr) >= n_active)) {
                own_lvl = 1;
                to_helpers(ppb::SEM_OWN, 1);
            }
            if (own_lvl == 1 && *own_ready >= n_active) {
                own_lvl = 2;
                to_helpers(ppb::SEM_OWN, 2);
            }
            const uint32_t v = *arrived;
            if (v > last) {
                last = v;
                to_helpers(ppb::SEM_MIRROR, v);
            }
        }
        noc_async_write_barrier();
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
    if (st_on) {
        // re-arm the readiness counter (persistent, trace-safe): every worker's increment of this call has landed
        volatile tt_l1_ptr uint32_t* ctr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(st_ctr);
        noc_semaphore_wait_min(ctr, n_active);
        *ctr = 0;
    }
    if constexpr (own_bands > 1) {
        // re-arm every band counter (persistent, trace-safe) once all its workers announced — also on a port whose
        // walk never touched the band, so a late announcement cannot leak into the next call
        for (uint32_t b = 0; b < own_bands; ++b) {
            const uint32_t addr = get_arg_val<uint32_t>(band_addr_idx + b);
            const uint32_t need = get_arg_val<uint32_t>(band_need_idx + b);
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr), need);
            noc_semaphore_inc(get_noc_addr(addr), 0u - need);
        }
    }
    noc_async_atomic_barrier();
}
