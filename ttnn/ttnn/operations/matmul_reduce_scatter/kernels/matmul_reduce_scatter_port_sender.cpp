// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// matmul_reduce_scatter — port-core BRISC: send_block over Fabric (raw worker API; the kernel-lib CCL sender
// programs 1-D hop routes only, and this op runs under the FABRIC_2D family, where routes are set by destination
// fabric node).
//
// Every segment of cb_xport_sum (relay sums, or a line end's own partials) is one packet into the downstream chip's
// relay_scratch page (slot * segs_per_block + seg). Entries are waves of the blocks (R4: one per compute unit, waves
// in order; the entry gives its first segment, count and window, see matmul_reduce_scatter_segments.hpp).
// Increment streams (fused write + atomic inc on every inc_every-th segment of an entry and on the entry's last --
// never deferred across an entry boundary: in a ring the downstream port's next block depends on this chip's later
// blocks only through the ring, so a deferred increment would close a wait cycle; with waves the downstream reader
// releases each wave as its own hand-off slot): relay entries count on the downstream port's arrival counter; the
// last block's entries (the downstream chip's own) on the arrival counter (A forward / B backward) of the downstream
// final core that owns the segment (alternating halves h = (seg / L) & 1).
// Ready fence (as fabric_reduce_scatter): first tell the peer port that writes into this chip that it may, then
// wait for my own peer's ready before the first data packet. At the end: wait for everything upstream sends me,
// then re-arm my arrival counter.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "matmul_reduce_scatter_segments.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

// Stage zones (permanent; opt-in via KERNEL_PERF_ZONES): snd_fence = ready fence with the downstream peer, snd_entry =
// one entry's packets (includes waiting on cb_xport_sum, i.e. on the reader / add upstream, and on router slots),
// snd_tail = final flush, snd_drain_in = waiting for everything upstream sends into this chip. MMRS_ABLATE_LINK (perf
// ablation only, wrong results) sends header-only atomic increments in place of the data packets.

using namespace tt::tt_fabric;

inline void route_one_hop(volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr, uint16_t chip_id, uint16_t mesh_id) {
#if (defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0)) || defined(EMULE_FABRIC_2D)
    (void)fabric_set_unicast_route(hdr, chip_id, mesh_id);
#else
    (void)chip_id;
    (void)mesh_id;
    (void)fabric_set_unicast_route<false>(
        reinterpret_cast<volatile tt_l1_ptr LowLatencyPacketHeader*>(hdr), static_cast<uint16_t>(1));
#endif
}

#ifndef MMRS_FINALS_PER_LINK
#define MMRS_FINALS_PER_LINK 2
#endif

void kernel_main() {
    constexpr uint32_t cb_xport_sum = get_compile_time_arg_val(0);
    constexpr uint32_t seg_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t group = get_compile_time_arg_val(2);  // packets in flight before a flush (header ring)
    constexpr uint32_t inc_every = get_compile_time_arg_val(3);
    constexpr uint32_t cap_segs = get_compile_time_arg_val(4);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(5);
    constexpr auto scr_args = TensorAccessorArgs<6>();
    constexpr uint32_t seg_bytes = seg_tiles * tile_bytes;
    constexpr uint32_t num_headers = group;

    size_t arg = 0;
    const uint32_t scr_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t segs_per_block = get_arg_val<uint32_t>(arg++);
    const uint32_t segs_per_row = get_arg_val<uint32_t>(arg++);
    const uint32_t blk_n_tiles = get_arg_val<uint32_t>(arg++);
    const uint32_t seg_stride = get_arg_val<uint32_t>(arg++);      // L
    const uint32_t arrival_addr = get_arg_val<uint32_t>(arg++);    // A: my counter, and the downstream port's
    const uint32_t final_sem_addr = get_arg_val<uint32_t>(arg++);  // A or B on the downstream final cores
    const uint32_t expect_in = get_arg_val<uint32_t>(arg++);
    const uint32_t ready_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t send_ready = get_arg_val<uint32_t>(arg++);
    const uint32_t ready_x = get_arg_val<uint32_t>(arg++);
    const uint32_t ready_y = get_arg_val<uint32_t>(arg++);
    const uint32_t peer_x = get_arg_val<uint32_t>(arg++);
    const uint32_t peer_y = get_arg_val<uint32_t>(arg++);
    constexpr uint32_t num_fin = MMRS_FINALS_PER_LINK;  // downstream final cores of this link: (x, y) each
    const uint32_t finals_idx = arg;
    arg += 2 * num_fin;
    const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(arg++));
    const uint16_t dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(arg++));
    const uint32_t num_blocks = get_arg_val<uint32_t>(arg++);  // entries
    // num_blocks x [landing slot at the receiver, first seg, seg count, downstream own block (to the finals),
    //               its segments owned by each of the num_fin finals in this wave, wave segment columns s0, s1,
    //               this chip's arrival slot of the block, upstream (relay) entry]
    constexpr uint32_t entry_stride = 8 + num_fin;
    const uint32_t entries_idx = arg;
    arg += entry_stride * num_blocks;
    const auto scr = TensorAccessor(scr_args, scr_addr, seg_bytes);

#ifdef MMRS_PORT_ARR_CB
    // Arrival pump (relay ports): this BRISC, not the port's reader, reads the upstream chip's relayed segments from
    // this chip's relay scratch into the arrival CB, on its own NoC, under the same arrival-counter gating the reader
    // used (segment idx of an entry needs inc_base + idx / inc_every + 1 increments). It never blocks: it is run
    // whenever the send loop would otherwise wait (fence, sums), so a reader-bound relay gets a second read port.
    constexpr uint32_t cb_arr = MMRS_PORT_ARR_CB;
    volatile tt_l1_ptr uint32_t* arr_ctr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_addr);
    uint32_t pk = 0, pidx = 0, pseg = 0, pinc_base = 0, pbatch = 0, papos = 0, pwptr = 0;
    bool pseg_set = false;
    auto pflush = [&]() {
        if (pbatch > 0) {
            noc_async_read_barrier();
            cb_push_back(cb_arr, seg_tiles * pbatch);
            papos += pbatch;
            if (papos == cap_segs) {
                papos = 0;
            }
            pbatch = 0;
        }
    };
    auto pump = [&]() {
        while (pk < num_blocks) {
            const uint32_t e = entries_idx + entry_stride * pk;
            const bool up = get_arg_val<uint32_t>(e + 7 + num_fin) != 0;
            const uint32_t count = get_arg_val<uint32_t>(e + 2);
            if (!up || pidx == count) {
                if (up) {
                    pflush();
                    pinc_base += (count + inc_every - 1) / inc_every;
                }
                ++pk;
                pidx = 0;
                pseg_set = false;
                continue;
            }
            if (!pseg_set) {
                pseg = get_arg_val<uint32_t>(e + 1);
                pseg_set = true;
            }
            if (*arr_ctr < pinc_base + pidx / inc_every + 1) {
                pflush();
                return;
            }
            if (pbatch == group || papos + pbatch == cap_segs) {
                pflush();
            }
            if (!cb_pages_reservable_at_back(cb_arr, seg_tiles * (pbatch + 1))) {
                pflush();
                return;
            }
            cb_reserve_back(cb_arr, seg_tiles * (pbatch + 1));
            if (pbatch == 0) {
                pwptr = get_write_ptr(cb_arr);
            }
            const uint32_t row = pseg / segs_per_row;
            const uint32_t c0 = (pseg - row * segs_per_row) * seg_tiles;
            const uint32_t valid = blk_n_tiles - c0 < seg_tiles ? blk_n_tiles - c0 : seg_tiles;
            const uint32_t base = get_arg_val<uint32_t>(e + 6 + num_fin) * segs_per_block;
#ifndef MMRS_ABLATE_ARRREADS
            noc_async_read(scr.get_noc_addr(base + pseg), pwptr + pbatch * seg_bytes, valid * tile_bytes);
#endif
            ++pbatch;
            ++pidx;
            if (pidx < count) {
                pseg = mmrs::next_wave_seg(
                    pseg,
                    seg_stride,
                    segs_per_row,
                    get_arg_val<uint32_t>(e + 4 + num_fin),
                    get_arg_val<uint32_t>(e + 5 + num_fin));
            }
        }
        pflush();
    };
#define MMRS_PUMP() pump()
#else
#define MMRS_PUMP() ((void)0)
#endif

    if (num_blocks > 0 || send_ready) {
        auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg);
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[num_headers];
        for (uint32_t h = 0; h < num_headers; ++h) {
            hdrs[h] = PacketHeaderPool::allocate_header();
            route_one_hop(hdrs[h], dst_chip_id, dst_mesh_id);
        }
        conn.open();
        if (send_ready) {
            hdrs[0]->to_noc_unicast_atomic_inc(
                NocUnicastAtomicIncCommandHeader{get_noc_addr(ready_x, ready_y, ready_addr), 1, true});
            conn.wait_for_empty_write_slot();
            conn.send_payload_flush_blocking_from_address(
                reinterpret_cast<uint32_t>(hdrs[0]), sizeof(PACKET_HEADER_TYPE));
        }
        if (num_blocks > 0) {
            MaybeDeviceZoneScope("snd_fence");
            volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
            while (*ready < 1) {
                MMRS_PUMP();
            }
            noc_semaphore_inc(get_noc_addr(ready_addr), 0u - 1u);
            noc_async_atomic_barrier();
        }
        const uint64_t relay_noc = get_noc_addr(peer_x, peer_y, arrival_addr);
        uint64_t final_noc[num_fin];
        for (uint32_t f = 0; f < num_fin; ++f) {
            final_noc[f] = get_noc_addr(
                get_arg_val<uint32_t>(finals_idx + 2 * f),
                get_arg_val<uint32_t>(finals_idx + 2 * f + 1),
                final_sem_addr);
        }
        uint32_t h = 0, unflushed = 0, rpos = 0;
        for (uint32_t k = 0; k < num_blocks; ++k) {
            MaybeDeviceZoneScope("snd_entry");
            const uint32_t e = entries_idx + entry_stride * k;
            const uint32_t base = get_arg_val<uint32_t>(e) * segs_per_block;
            const uint32_t count = get_arg_val<uint32_t>(e + 2);
            const bool fin = get_arg_val<uint32_t>(e + 3) != 0;
            uint32_t sent_f[num_fin] = {};
            const uint32_t wave_s0 = get_arg_val<uint32_t>(e + 4 + num_fin);
            const uint32_t wave_s1 = get_arg_val<uint32_t>(e + 5 + num_fin);
            uint32_t seg = get_arg_val<uint32_t>(e + 1);
            for (uint32_t i = 0; i < count; ++i) {
                if (i > 0) {
                    seg = mmrs::next_wave_seg(seg, seg_stride, segs_per_row, wave_s0, wave_s1);
                }
                const uint32_t row = seg / segs_per_row;
                const uint32_t c0 = (seg - row * segs_per_row) * seg_tiles;
                const uint32_t valid = blk_n_tiles - c0 < seg_tiles ? blk_n_tiles - c0 : seg_tiles;
                const uint32_t bytes = valid * tile_bytes;
#ifdef MMRS_PORT_ARR_CB
                while (!cb_pages_available_at_front(cb_xport_sum, seg_tiles * (unflushed + 1))) {
                    MMRS_PUMP();
                }
#endif
                cb_wait_front(cb_xport_sum, seg_tiles * (unflushed + 1));
                const uint32_t src = get_read_ptr(cb_xport_sum) + unflushed * seg_bytes;
                const uint64_t dst = scr.get_noc_addr(base + seg, 0, 0);
                volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[h];
                h = (h + 1 == num_headers) ? 0 : h + 1;
                bool inc;
                uint64_t ctr;
                if (fin) {
                    const uint32_t hf = (seg / seg_stride) % num_fin;  // seg = l + t L: owner final (l, t mod F)
                    const uint32_t sent = ++sent_f[hf];
                    inc = (sent % inc_every) == 0 || sent == get_arg_val<uint32_t>(e + 4 + hf);
                    ctr = final_noc[hf];
                } else {
                    inc = ((i + 1) % inc_every) == 0 || i + 1 == count;
                    ctr = relay_noc;
                }
#ifdef MMRS_ABLATE_LINK
                (void)src;
                (void)dst;
                if (inc) {
                    hdr->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{ctr, 1, true});
                    conn.wait_for_empty_write_slot();
                    conn.send_payload_flush_blocking_from_address(
                        reinterpret_cast<uint32_t>(hdr), sizeof(PACKET_HEADER_TYPE));
                }
#else
                if (inc) {
                    hdr->to_noc_fused_unicast_write_atomic_inc(
                        NocUnicastAtomicIncFusedCommandHeader{dst, ctr, 1, true}, bytes);
                } else {
                    hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, bytes);
                }
                conn.wait_for_empty_write_slot();
                conn.send_current_slot_non_blocking(src, bytes, reinterpret_cast<uint32_t>(hdr));
#endif
                ++unflushed;
                if (unflushed == group || rpos + unflushed == cap_segs) {
                    noc_async_writes_flushed();  // sources + headers of the in-flight packets have left L1
                    cb_pop_front(cb_xport_sum, seg_tiles * unflushed);
                    rpos += unflushed;
                    if (rpos == cap_segs) {
                        rpos = 0;
                    }
                    unflushed = 0;
                }
            }
        }
#ifdef MMRS_PORT_ARR_CB
        while (pk < num_blocks) {  // arrivals of entries with no segment left to send here (never the case for a
            MMRS_PUMP();           // relay, whose sums need them -- kept for safety)
        }
#endif
        MaybeDeviceZoneScope("snd_tail");
        if (unflushed > 0) {  // the tail (an entry may hold no segment of this port, so flush after the walk)
            noc_async_writes_flushed();
            cb_pop_front(cb_xport_sum, seg_tiles * unflushed);
        }
        conn.close();
    }
    if (expect_in > 0) {
        MaybeDeviceZoneScope("snd_drain_in");
        volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_addr);
        noc_semaphore_wait_min(arrived, expect_in);
        noc_semaphore_inc(get_noc_addr(arrival_addr), 0u - expect_in);
        noc_async_atomic_barrier();
    }
}
