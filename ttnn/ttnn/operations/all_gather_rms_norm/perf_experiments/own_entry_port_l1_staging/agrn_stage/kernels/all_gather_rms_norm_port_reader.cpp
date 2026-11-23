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
    // EXPERIMENT own_entry_port_l1_staging: staged pages of my lane (the walk's leading pages), counters
    const uint32_t stage_pages = get_arg_val<uint32_t>(a++);
    const uint32_t st_nctr = get_arg_val<uint32_t>(a++);
    const uint32_t st_need_idx = a;
    const uint32_t st_addr_idx = a + st_nctr;
    a += 2 * st_nctr;
    // EXPERIMENT STAGE_MODE 2: the lane's walk staged in run_pages chunks over the worker cores (chunk g = lane * nch +
    // c on core g % n_act, slot g / n_act of cb_stage), one counter per chunk at m2_ctr + 4 c (this core's L1)
    const bool m2_on = get_arg_val<uint32_t>(a++) != 0;  // every port re-arms its counter, staged lane or not
    const uint32_t m2_pages = m2_on ? get_arg_val<uint32_t>(a++) : 0u;
    bool first_now = false;
    uint32_t m2_ctr = 0, m2_nact = 1, m2_nch = 0, m2_lane = 0, m2_L = 1, m2_tbl = 0, m2_single = 0, m2_cp = 1,
             m2_head = 0, m2_head_off = 0;
    if (m2_on) {
        m2_ctr = get_arg_val<uint32_t>(a++);
        m2_nact = get_arg_val<uint32_t>(a++);
        m2_nch = get_arg_val<uint32_t>(a++);
        m2_lane = get_arg_val<uint32_t>(a++);
        m2_L = get_arg_val<uint32_t>(a++);
        m2_single = get_arg_val<uint32_t>(a++);    // > 0: one counter (word 0), this many worker increments
        m2_cp = get_arg_val<uint32_t>(a++);        // pages per staging chunk
        m2_head = get_arg_val<uint32_t>(a++);      // walk pages staged in my own cb_stage (no read)
        m2_head_off = get_arg_val<uint32_t>(a++);  // their page offset in cb_stage
        first_now = get_arg_val<uint32_t>(a++) != 0;
        m2_tbl = workers_idx;  // the staging cores are the active workers (same NoC coord list)
    }

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
    // Pipelined reads: packet i is read into chunk slot (pushed + inflight) of the 2-group CB under NoC transaction id
    // 1 + i % read_depth; once read_depth packets are in flight the oldest is waited (its trid only) and pushed, so
    // the sender starts on a packet as soon as its own reads landed and reads of later packets keep the NoC busy.
    constexpr uint32_t n_slots = 2 * group;
    const uint32_t chunks_base = get_write_ptr(cb_chunks);  // nothing pushed yet: the CB's first slot
    const uint32_t desc_base = get_write_ptr(cb_desc);
    uint32_t pushed = 0, inflight = 0, issued = 0;
    // own-ready bands: a band, once seen complete, is never polled again; after the first few packets all are
    constexpr uint32_t all_bands = own_bands >= 32 ? 0xFFFFFFFFu : (1u << own_bands) - 1u;
    uint32_t bands_ready = 0;
    const uint32_t stage_base = stage_on ? get_write_ptr(cb_stage) : 0u;
    const uint32_t st_all = st_nctr >= 32 ? 0xFFFFFFFFu : (1u << st_nctr) - 1u;
    uint32_t st_ready = 0;
    bool own_waited = false;
    uint32_t m2_ready = 0;  // chunks [0, m2_ready) seen complete
    auto wait_bands = [&](const Packet& pk) {
        for (uint32_t ri = 0; ri < pk.nr; ++ri) {
            const uint32_t r = flat_slot ? 0u : pk.rr[ri];
            uint32_t b = 0;
            while (b + 1 < own_bands && get_arg_val<uint32_t>(band_end_idx + b) <= r) {
                ++b;
            }
            if (!((bands_ready >> b) & 1u)) {
                noc_semaphore_wait_min(
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(band_addr_idx + b)),
                    get_arg_val<uint32_t>(band_need_idx + b));
                bands_ready |= 1u << b;
            }
        }
    };
    auto retire_oldest = [&]() {
#if defined(AGRN_PROFILE_ZONES)
        if (pushed == 0) {
            AGRN_ZONE("pr_push0");
            noc_async_read_barrier_with_trid(1 + (issued - inflight) % read_depth);
        }
#endif
        noc_async_read_barrier_with_trid(1 + (issued - inflight) % read_depth);
        cb_push_back(cb_chunks, run_pages);
        ++pushed;
        --inflight;
    };
    for (uint32_t k = 0; k < n_entries; ++k) {
        const uint32_t e = get_arg_val<uint32_t>(entries_idx + 3 * k);
        const uint32_t A = get_arg_val<uint32_t>(entries_idx + 3 * k + 1);
        const uint32_t B = get_arg_val<uint32_t>(entries_idx + 3 * k + 2);
        const uint32_t q = e & 0xFFFF, part = e >> 16;
        if (k == 0 && own_bands == 1 && !(stage_on && (stage_pages > 0 || m2_pages > 0))) {
            AGRN_ZONE("pr_wait_own");
#if !defined(AGRN_ABL_NO_OWN_WAIT)  // ablation: the bulk starts without the normalized slice (output garbage)
            noc_semaphore_wait_min(own_ready, n_active);
#endif
        }
        if (k == 0) {
            AGRN_ZONE("pr_own_first");  // marker: own entry starts
        }
        uint32_t c = 0;
        uint32_t s_pages = 0;  // own entry: pages walked so far (staging position)
        for_each_packet(q, part, first_bank, bank_stride, [&](const Packet& pk) {
            uint32_t st_src = 0;
            bool m2 = false;
            uint32_t m2_s = 0;
            if constexpr (stage_on) {
                if (k == 0) {
                    if (s_pages + pk.pages <= m2_pages) {
                        // chunks [s / rp, (s + pages - 1) / rp] must have all their pages staged
                        m2 = true;
                        m2_s = s_pages;
                        const uint32_t c1 = (s_pages + pk.pages - 1) / m2_cp;
                        if (m2_single > 0 && m2_ready == 0) {
                            {
                                AGRN_ZONE("pr_m2_wait");
                                noc_semaphore_wait_min(
                                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(m2_ctr), m2_single);
                            }
                            m2_ready = 0xFFFFFFFFu;
                        }
                        while (m2_ready <= c1) {
                            const uint32_t left = m2_pages - m2_ready * m2_cp;
                            noc_semaphore_wait_min(
                                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(m2_ctr + 4 * m2_ready),
                                left < m2_cp ? left : m2_cp);
                            ++m2_ready;
                        }
                        if (s_pages + pk.pages <= m2_head) {
                            m2 = false;
                            st_src = stage_base + (m2_head_off + s_pages) * page_bytes;
                        }
                    } else if (s_pages + pk.pages <= stage_pages) {
                        // staged: wait the counters of the packet's lane units, send from cb_stage
                        if (st_ready != st_all) {
                            for (uint32_t ri = 0; ri < pk.nr; ++ri) {
                                const uint32_t u =
                                    shard_w ? pk.rr[ri] / bank_stride : (pk.first[ri] % nb) / bank_stride;
                                const uint32_t ci = u < st_nctr ? u : st_nctr - 1;
                                if (!((st_ready >> ci) & 1u)) {
                                    noc_semaphore_wait_min(
                                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                                            get_arg_val<uint32_t>(st_addr_idx + ci)),
                                        get_arg_val<uint32_t>(st_need_idx + ci));
                                    st_ready |= 1u << ci;
                                }
                            }
                        }
                        st_src = stage_base + s_pages * page_bytes;
                    } else if (!own_waited) {
                        noc_semaphore_wait_min(own_ready, n_active);
                        own_waited = true;
                    }
                    s_pages += pk.pages;
                }
            }
            if constexpr (own_bands > 1) {
                if (k == 0 && bands_ready != all_bands) {
                    wait_bands(pk);
                }
            }
            if (inflight == read_depth) {
                retire_oldest();
            }
            cb_reserve_back(cb_chunks, run_pages * (inflight + 1));
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
            {
                volatile tt_l1_ptr PacketDesc* dsc =
                    reinterpret_cast<volatile tt_l1_ptr PacketDesc*>(desc_base + (issued % n_slots) * DESC_BYTES);
                dsc->nr = pk.nr;
                dsc->pages = pk.pages;
                dsc->src = st_src;
                for (uint32_t ri = 0; ri < pk.nr; ++ri) {
                    dsc->first[ri] = pk.first[ri];
                    dsc->n[ri] = pk.n[ri];
                }
            }
            noc_async_read_set_trid(1 + issued % read_depth);
            if (m2) {
                // one L1 read per chunk piece (the packet's pages are consecutive in the lane walk)
                uint32_t s0 = m2_s;
                const uint32_t s1 = m2_s + pk.pages;
                if (s0 < m2_head) {
                    // head pages: in my own cb_stage
                    const uint32_t e = m2_head < s1 ? m2_head : s1;
                    noc_async_read(
                        get_noc_addr(my_x[noc_index], my_y[noc_index], stage_base + (m2_head_off + s0) * page_bytes),
                        l1,
                        (e - s0) * page_bytes);
                    l1 += (e - s0) * page_bytes;
                    s0 = e;
                }
                while (s0 < s1) {
                    const uint32_t c = s0 / m2_cp;
                    const uint32_t ce = (c + 1) * m2_cp;
                    const uint32_t e = ce < s1 ? ce : s1;
                    const uint32_t g = m2_lane * m2_nch + c;
                    const uint32_t n_ch = m2_L * m2_nch;  // chunk homes: agrn_worker_args.hpp stage_chunk_home
                    const uint32_t core = n_ch <= m2_nact ? g * m2_nact / n_ch : g % m2_nact;
                    const uint32_t slot = n_ch <= m2_nact ? 0u : g / m2_nact;
                    const uint32_t xy = get_arg_val<uint32_t>(m2_tbl + core);
                    const uint32_t src = stage_base + (slot * m2_cp + (s0 - c * m2_cp)) * page_bytes;
                    noc_async_read(get_noc_addr(xy & 0xFFFF, xy >> 16, src), l1, (e - s0) * page_bytes);
                    l1 += (e - s0) * page_bytes;
                    s0 = e;
                }
            }
            for (uint32_t ri = 0; ri < ((st_src || m2) ? 0u : pk.nr); ++ri) {
                const uint32_t n = pk.n[ri];
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
                } else if constexpr (fast_banks) {
                    noc_async_read(bank_base[pk.first[ri] % nb] + (pk.first[ri] / nb) * bank_pitch, l1, n * page_bytes);
                } else {
                    noc_async_read(dst.get_noc_addr(pk.first[ri]), l1, n * page_bytes);
                }
                l1 += n * page_bytes;
            }
            ++issued;
            ++inflight;
            ++c;
            if (first_now && k == 0 && c == 1) {
                // EXPERIMENT (STAGE_FIRST): the entry's first packet is handed over as soon as its reads land, before
                // the next packet is walked (the sender is idle until then)
                retire_oldest();
            }
            if constexpr (eager_push) {
                // hand over every packet whose reads already landed (non-blocking check), so the sender starts on
                // the first packet of an entry without waiting for read_depth packets to be issued
                while (inflight > 0 &&
                       ncrisc_noc_read_with_transaction_id_flushed(noc_index, 1 + (issued - inflight) % read_depth)) {
                    retire_oldest();
                }
            }
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
    if (m2_on) {
        // re-arm the chunk counters (persistent, trace-safe): every worker's increments for this call have landed
        const uint32_t nch = m2_single > 0 ? 0u : (m2_pages + m2_cp - 1) / m2_cp;
        if (m2_single > 0) {
            volatile tt_l1_ptr uint32_t* ctr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(m2_ctr);
            noc_semaphore_wait_min(ctr, m2_single);
            *ctr = 0;
        }
        for (uint32_t c = 0; c < nch; ++c) {
            const uint32_t left = m2_pages - c * m2_cp;
            volatile tt_l1_ptr uint32_t* ctr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(m2_ctr + 4 * c);
            noc_semaphore_wait_min(ctr, left < m2_cp ? left : m2_cp);
            *ctr = 0;
        }
    }
    if constexpr (stage_on) {
        // re-arm the stage counters (persistent, trace-safe) once every worker's staged pages landed — also on a port
        // with no own entry (its lane's workers wrote into it regardless)
        for (uint32_t ci = 0; ci < st_nctr; ++ci) {
            const uint32_t addr = get_arg_val<uint32_t>(st_addr_idx + ci);
            const uint32_t need = get_arg_val<uint32_t>(st_need_idx + ci);
            if (need > 0) {
                noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr), need);
                noc_semaphore_inc(get_noc_addr(addr), 0u - need);
            }
        }
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
