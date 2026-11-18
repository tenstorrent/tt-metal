// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Port sender (RISCV_0, NoC1), one per (link, direction). Adapted from examples/fabric_all_gather's sender:
//   fence      one header-only atomic inc of the ready counter of the peer port that writes into this chip, then wait
//              for my own ready >= 1 and consume it (the fence between back-to-back calls)       [TRANSPORT]
//   bulk       every chunk of my stream into the SAME `landing` offset of the SAME logical compute core on the peer;
//              every INC_EVERY-th chunk and each entry's last chunk carry a fused increment of the peer receiver's
//              arrival counter (the receiver turns it into per-core landing notifications)       [TRANSPORT]
//   stats      (link 0) whenever the hub has handed me a slot (stats_tx), send it into the peer hub's stats scratch,
//              the last packet fused with an increment of the peer hub's stats_fwd / stats_bwd. Polled while waiting
//              for bulk chunks and at every batch flush, so the stats never wait behind the bulk. While a slot is still
//              owed, at most STATS_BULK_WINDOW bulk packets stay queued in the router channel.  [OVERLAP]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "all_gather_rms_norm_kimi_perf.hpp"

using namespace tt::tt_fabric;

constexpr uint32_t cb = get_compile_time_arg_val(0);
constexpr uint32_t chunk_tiles = get_compile_time_arg_val(1);
constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
constexpr uint32_t batch = get_compile_time_arg_val(3);
constexpr uint32_t inc_every = get_compile_time_arg_val(4);
constexpr uint32_t sem_stats_tx = get_compile_time_arg_val(7);
constexpr uint32_t slot_bytes = get_compile_time_arg_val(8);
constexpr uint32_t max_payload = get_compile_time_arg_val(9);
constexpr uint32_t stats_window = get_compile_time_arg_val(10);  // bulk packets queued while stats are owed; 0 = off
constexpr uint32_t chunk_bytes = chunk_tiles * tile_bytes;
constexpr uint32_t cb_chunks = get_compile_time_arg_val(13);  // port_supply_pipeline: CB capacity in chunks
// port_supply_pipeline sender knobs (defines):
//   PSP_POP_EVERY  flush + pop the consumed chunks every this many sends (default: batch), and early whenever the
//                  next chunk has not arrived yet (frees the reader's space before waiting on it)
//   PSP_STATEFUL   program the payload/header/credit cmd bufs once (setup_stateful_send_cmd_bufs) and send each chunk
//                  with send_current_slot_stateful_non_blocking; re-programmed after every stats send (which uses the
//                  non-stateful path)
#ifdef PSP_POP_EVERY
constexpr uint32_t pop_every = PSP_POP_EVERY;
#else
constexpr uint32_t pop_every = batch;
#endif
static_assert(pop_every >= 1 && pop_every <= cb_chunks, "pop_every");
// a header is rewritten only after the flush that follows its send: one header per send between flushes
constexpr uint32_t n_hdrs = pop_every > batch ? pop_every : batch;
static_assert(n_hdrs <= 7, "packet-header pool: n_hdrs bulk headers + 1 control header");

// One hop: 2D fabrics route by destination fabric node, 1D fabrics by hop count (ROUTING_MODE from the build).
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

void kernel_main() {
    size_t a = 0;
    const uint32_t landing_addr = get_arg_val<uint32_t>(a++);
    const uint32_t stats_addr = get_arg_val<uint32_t>(a++);
    const uint32_t active = get_arg_val<uint32_t>(a++);
    uint32_t chunks_left = get_arg_val<uint32_t>(a++);
    const uint32_t n_blocks = get_arg_val<uint32_t>(a++);
    const uint32_t n_stats = get_arg_val<uint32_t>(a++);
    const uint32_t ready_addr = get_arg_val<uint32_t>(a++);
    const uint32_t ready_xy = get_arg_val<uint32_t>(a++);  // the peer's port that writes into this chip
    const uint32_t arrival_addr = get_arg_val<uint32_t>(a++);
    const uint32_t recv_xy = get_arg_val<uint32_t>(a++);         // the peer's receiver of my stream
    const uint32_t stats_sem_addr = get_arg_val<uint32_t>(a++);  // the peer hub's stats_fwd / stats_bwd
    const uint32_t hub_xy = get_arg_val<uint32_t>(a++);
    const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint16_t dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint32_t stats_idx = a;  // n_stats slots
    a += n_stats;
    const uint32_t blocks_idx = a;  // per block: core xy, n_chunks | entry_end << 17
    a += 2 * n_blocks;
    if (!active) {
        return;
    }

    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);  // appended last
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[n_hdrs];
    for (uint32_t h = 0; h < n_hdrs; ++h) {
        hdrs[h] = PacketHeaderPool::allocate_header();
        route_one_hop(hdrs[h], dst_chip_id, dst_mesh_id);
    }
    volatile tt_l1_ptr PACKET_HEADER_TYPE* ctl_hdr = PacketHeaderPool::allocate_header();
    route_one_hop(ctl_hdr, dst_chip_id, dst_mesh_id);
    {
        MaybeDeviceZoneScope("sender_open");
        conn.open();
    }

    // ---- fence ----
    {
        MaybeDeviceZoneScope("sender_fence");
        ctl_hdr->to_noc_unicast_atomic_inc(
            NocUnicastAtomicIncCommandHeader{get_noc_addr(ready_xy >> 8, ready_xy & 0xFF, ready_addr), 1, true});
        conn.wait_for_empty_write_slot();
        conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(ctl_hdr), sizeof(PACKET_HEADER_TYPE));
        {
            volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
            noc_semaphore_wait_min(ready, 1);
            noc_semaphore_inc(get_noc_addr(ready_addr), 0u - 1u);
            noc_async_atomic_barrier();
        }
    }  // sender_fence

    // ---- stats: send every slot the hub has handed over so far ----
    volatile tt_l1_ptr uint32_t* stats_tx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_stats_tx));
    const uint64_t stats_sem_noc = get_noc_addr(hub_xy >> 8, hub_xy & 0xFF, stats_sem_addr);
    uint32_t stats_sent = 0;
    auto try_stats = [&]() -> bool {
        bool any = false;
        while (stats_sent < n_stats && *stats_tx > stats_sent) {
            const uint32_t q = get_arg_val<uint32_t>(stats_idx + stats_sent);
            const uint32_t off = q * slot_bytes;
            for (uint32_t s = 0; s < slot_bytes; s += max_payload) {
                const uint32_t bytes = (slot_bytes - s) < max_payload ? (slot_bytes - s) : max_payload;
                const uint64_t dst = get_noc_addr(hub_xy >> 8, hub_xy & 0xFF, stats_addr + off + s);
                if (s + bytes == slot_bytes) {
                    ctl_hdr->to_noc_fused_unicast_write_atomic_inc(
                        NocUnicastAtomicIncFusedCommandHeader{dst, stats_sem_noc, 1, true}, bytes);
                } else {
                    ctl_hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, bytes);
                }
                conn.wait_for_empty_write_slot();
                conn.send_current_slot_non_blocking(stats_addr + off + s, bytes, reinterpret_cast<uint32_t>(ctl_hdr));
                noc_async_writes_flushed();  // ctl_hdr is reused right away
            }
            ++stats_sent;
            any = true;
        }
        return any;
    };
#ifdef PSP_STATEFUL
    auto stats_then_restate = [&]() {
        if (try_stats()) {
            conn.setup_stateful_send_cmd_bufs();
        }
    };
#else
    auto stats_then_restate = [&]() { try_stats(); };
#endif

    // ---- bulk ----
    const uint64_t arrival_noc = get_noc_addr(recv_xy >> 8, recv_xy & 0xFF, arrival_addr);
    const uint32_t cb_base = get_read_ptr(cb);  // fifo start (the CB interface is re-initialized every launch)
    uint32_t h = 0, unpopped = 0, rd = 0, sent = 0;
    uint32_t d_cb = 0, d_win = 0, d_slot = 0, d_send = 0, d_flush = 0, d_ts = 0, n_cbwait = 0, n_slotwait = 0,
             d_hdr = 0;
    (void)d_cb, (void)d_win, (void)d_slot, (void)d_send, (void)d_flush, (void)d_ts, (void)n_cbwait, (void)n_slotwait;
    (void)d_hdr;
    auto flush_pop = [&]() {
        {
            PSP_T0(t);
            noc_async_writes_flushed();  // sources + headers of the in-flight chunks have left L1
            PSP_ACC(d_flush, t);
        }
        cb_pop_front(cb, chunk_tiles * unpopped);
        rd += unpopped;
        rd = rd >= cb_chunks ? rd - cb_chunks : rd;
        unpopped = 0;
    };
#ifdef PSP_STATEFUL
    conn.setup_stateful_send_cmd_bufs();
#endif
    PSP_T0(t_bulk);
    {
        MaybeDeviceZoneScope("sender_bulk");
        for (uint32_t i = 0; i < n_blocks; ++i) {
            const uint32_t xy = get_arg_val<uint32_t>(blocks_idx + 2 * i);
            const uint32_t info = get_arg_val<uint32_t>(blocks_idx + 2 * i + 1);
            const uint32_t n = info & 0xFFFF;
            const bool entry_end = (info >> 17) & 1;
            for (uint32_t k = 0; k < n; ++k) {
                {
                    PSP_T0(t);
                    if (!cb_pages_available_at_front(cb, chunk_tiles * (unpopped + 1))) {
#ifdef PSP_DIAG
                        ++n_cbwait;
#endif
                        if (unpopped > 0) {
                            flush_pop();
                        }
                        while (!cb_pages_available_at_front(cb, chunk_tiles)) {
                            stats_then_restate();
                        }
                    }
                    PSP_ACC(d_cb, t);
                }
                if constexpr (stats_window > 0) {
                    // a stats slot still owed (the relay) would queue behind every bulk packet in the channel: keep at
                    // most stats_window of them queued until it has gone out
                    PSP_T0(t);
                    while (stats_sent < n_stats &&
                           conn.num_buffers_per_channel - conn.get_num_free_write_slots() >= stats_window) {
                        stats_then_restate();
                    }
                    PSP_ACC(d_win, t);
                }
                PSP_T0(t_hdr);
                uint32_t slot = rd + unpopped;
                slot = slot >= cb_chunks ? slot - cb_chunks : slot;
                const uint32_t src = cb_base + slot * chunk_bytes;
                const uint64_t dst = get_noc_addr(xy >> 8, xy & 0xFF, landing_addr + k * chunk_bytes);
                volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[h];
                h = (h + 1 == n_hdrs) ? 0 : h + 1;
                ++sent;
                if ((sent % inc_every) == 0 || (entry_end && k + 1 == n)) {
                    hdr->to_noc_fused_unicast_write_atomic_inc(
                        NocUnicastAtomicIncFusedCommandHeader{dst, arrival_noc, 1, true}, chunk_bytes);
                } else {
                    hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, chunk_bytes);
                }
                PSP_ACC(d_hdr, t_hdr);
                {
                    PSP_T0(t);
#ifdef PSP_DIAG
                    if (!conn.edm_has_space_for_packet()) {
                        ++n_slotwait;
                    }
#endif
                    conn.wait_for_empty_write_slot();
                    PSP_ACC(d_slot, t);
                }
                {
                    PSP_T0(t);
#ifdef PSP_STATEFUL
                    conn.send_current_slot_stateful_non_blocking(src, chunk_bytes, reinterpret_cast<uint32_t>(hdr));
#else
                    conn.send_current_slot_non_blocking(src, chunk_bytes, reinterpret_cast<uint32_t>(hdr));
#endif
                    PSP_ACC(d_send, t);
                }
                --chunks_left;
                if (++unpopped == pop_every || chunks_left == 0) {
                    flush_pop();
                    PSP_T0(t2);
                    stats_then_restate();
                    PSP_ACC(d_ts, t2);
                }
            }
        }
    }  // sender_bulk
    {
        uint32_t d_tot = 0;
        (void)d_tot;
        PSP_ACC(d_tot, t_bulk);
        PSP_EMIT("psp_s_total", d_tot);
        PSP_EMIT("psp_s_cb", d_cb);
        PSP_EMIT("psp_s_win", d_win);
        PSP_EMIT("psp_s_slot", d_slot);
        PSP_EMIT("psp_s_send", d_send);
        PSP_EMIT("psp_s_flush", d_flush);
        PSP_EMIT("psp_s_ts", d_ts);
        PSP_EMIT("psp_s_hdr", d_hdr);
        PSP_EMIT("psp_s_ncbwait", n_cbwait);
        PSP_EMIT("psp_s_nslotwait", n_slotwait);
    }
    // ---- the stats that became ready only after the bulk ----
    {
        MaybeDeviceZoneScope("sender_tail_stats");
        while (stats_sent < n_stats) {
            try_stats();
        }
    }
    conn.close();
}
