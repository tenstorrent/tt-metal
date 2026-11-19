// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// all_gather_rms_norm port sender (BRISC / NoC1), one per (direction, link) per chip. TRANSPORT.
// Adapted from the fabric_all_gather example's sender: pre-routed header ring, ready fence between calls, chunks sent
// with send_current_slot_non_blocking, flush per batch. Destinations are the downstream chip's MEMBER L1 landing slots
// (the same (core, offset) as the source slot on this chip: worker set and cb_x_landing base are identical on every
// chip). The last chunk of every item is a fused write + atomic increment of the downstream port's arrival counter
// (one increment per item; items are >= item_min_chunks chunks where the shape allows).
//
// Raw fabric API (documented helper bypass, op_design.md -> Helpers considered and rejected): kernel_lib's
// ccl::FabricStream arms a fixed payload size and its signal() closes the connection; chunks here vary in size and
// the ready goes out on the same open connection that then carries the data. Perf 1 (measured, keep): the ready is
// sent after routing header 0 only; the chunk count and the header ring (byte copies of header 0) are built in the
// shadow of the peer's ready. FabricStream routes its headers before it can signal: focus case 23.81 -> 22.83 us
// together with the port reader's input-direct first items (changelog Perf 1).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "all_gather_rms_norm_common.hpp"
#include "all_gather_rms_norm_port_walk.hpp"

using namespace tt::tt_fabric;
using namespace agrms_port;

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
#if defined(AGRMS_ABLATE_EMPTY) || defined(AGRMS_ABLATE_LOCAL)
    return;
#endif
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t run_pages = get_compile_time_arg_val(1);
    constexpr uint32_t group = get_compile_time_arg_val(2);
    constexpr uint32_t R_t = get_compile_time_arg_val(3);
    constexpr uint32_t n_groups = get_compile_time_arg_val(4);
    constexpr uint32_t k = get_compile_time_arg_val(5);
    constexpr uint32_t w_pitch = get_compile_time_arg_val(6);
    constexpr uint32_t xfer_rows = get_compile_time_arg_val(7);
    constexpr uint32_t Wt_lp = get_compile_time_arg_val(8);
    constexpr uint32_t relay_lag = get_compile_time_arg_val(9);
    constexpr uint32_t sem_arrival_id = get_compile_time_arg_val(10);
    constexpr uint32_t rows_per_group = get_compile_time_arg_val(11);  // 0: round-robin rows (see Geom::place)
    constexpr uint32_t chunk_bytes = run_pages * page_bytes;
    constexpr uint32_t num_headers = group;
    constexpr uint32_t cb = agrms_cb::cb_port_chunks;

    size_t a = 0;
    const uint32_t E_out = get_arg_val<uint32_t>(a++);
    const uint32_t out_idx = a;
    a += E_out;
    const uint32_t segidx_idx = a;  // per send entry: (first segment << 16) | segment count
    a += E_out;
    const uint32_t n_segs = get_arg_val<uint32_t>(a++);
    const uint32_t segs_idx = a;  // member segments of the send entries (see Entry)
    a += n_segs;
    const uint32_t ready_addr = get_arg_val<uint32_t>(a++);  // global semaphore (same address on every port core)
    const uint32_t send_ready = get_arg_val<uint32_t>(a++);  // 1: the peer's port that sends to me waits for my ready
    const uint32_t ready_x = get_arg_val<uint32_t>(a++);
    const uint32_t ready_y = get_arg_val<uint32_t>(a++);
    const uint32_t peer_x = get_arg_val<uint32_t>(a++);  // downstream port (same direction and link), NoC coords
    const uint32_t peer_y = get_arg_val<uint32_t>(a++);
    const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint16_t dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint32_t open_conn = get_arg_val<uint32_t>(a++);
    if (open_conn == 0) {
        return;  // no peer on this side (line end) and nothing to fence
    }

    const Geom g{
        page_bytes,
        run_pages,
        R_t,
        n_groups,
        k,
        w_pitch,
        xfer_rows,
        Wt_lp,
        get_write_ptr(agrms_cb::cb_x_landing),
        rows_per_group,
        0};
    const uint32_t n_xb = g.n_xb();
    auto out_entry = [&](uint32_t e) {
        return unpack_entry(get_arg_val<uint32_t>(out_idx + e), get_arg_val<uint32_t>(segidx_idx + e), segs_idx);
    };

    // Pre-send critical path (Perf 1): everything in the call is shifted by when the first packet leaves, so the
    // ready goes out as early as possible. Header 0 alone is routed before the fence (a 2D fabric_set_unicast_route
    // costs ~0.36 us); the chunk count and the rest of the header ring are built while the peer's ready is in flight.
    // Every header of the ring goes to the same one-hop destination, so headers 1.. are byte copies of the routed
    // header 0 (its command fields are rewritten for every packet). Routing each header measured 1.46 us of setup
    // in front of the fence.
    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);  // appended last
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[num_headers];
    {
        AGRMS_ZONE("AG_SETUP_HDR0");
        hdrs[0] = PacketHeaderPool::allocate_header();
        route_one_hop(hdrs[0], dst_chip_id, dst_mesh_id);
    }
    uint32_t chunks_left = 0;
    {
        AGRMS_ZONE("AG_FENCE");  // open + ready out + (count, header ring) + own ready in
        {
            AGRMS_ZONE("AG_OPEN");
            conn.open();
        }
        // Fence between calls: this program running here means every earlier command on this chip is done, so the
        // peer may now write into my members' L1: say so. Then send nothing until the peer has said the same.
        if (send_ready) {
            AGRMS_ZONE("AG_READY_OUT");
            hdrs[0]->to_noc_unicast_atomic_inc(
                NocUnicastAtomicIncCommandHeader{get_noc_addr(ready_x, ready_y, ready_addr), 1, true});
            conn.wait_for_empty_write_slot();
            conn.send_payload_flush_blocking_from_address(
                reinterpret_cast<uint32_t>(hdrs[0]), sizeof(PACKET_HEADER_TYPE));
        }
        {
            AGRMS_ZONE("AG_SETUP_REST");  // in the shadow of the peer's ready: chunk count + header ring copies
            for (uint32_t e = 0; e < E_out; ++e) {
                const Entry en = out_entry(e);
                chunks_left += en.rows_in(0, R_t) * g.chunks_per_row(en);
            }
            constexpr uint32_t n_words = sizeof(PACKET_HEADER_TYPE) / sizeof(uint32_t);
            const tt_l1_ptr uint32_t* s0 =
                reinterpret_cast<const tt_l1_ptr uint32_t*>(const_cast<PACKET_HEADER_TYPE*>(hdrs[0]));
            uint32_t words[n_words];
#pragma GCC unroll 64
            for (uint32_t i = 0; i < n_words; ++i) {
                words[i] = s0[i];
            }
            for (uint32_t h = 1; h < num_headers; ++h) {
                hdrs[h] = PacketHeaderPool::allocate_header();
                tt_l1_ptr uint32_t* d = reinterpret_cast<tt_l1_ptr uint32_t*>(const_cast<PACKET_HEADER_TYPE*>(hdrs[h]));
#pragma GCC unroll 64
                for (uint32_t i = 0; i < n_words; ++i) {
                    d[i] = words[i];
                }
            }
        }
        if (chunks_left > 0) {
            AGRMS_ZONE("AG_READY_WAIT");
            volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
            noc_semaphore_wait_min(ready, 1);
            noc_semaphore_inc(get_noc_addr(ready_addr), 0u - 1u);
            noc_async_atomic_barrier();
        }
    }
    if (chunks_left > 0) {
        const uint64_t arrival_noc = get_noc_addr(peer_x, peer_y, get_semaphore(sem_arrival_id), 0);
        uint32_t h = 0, unflushed = 0;
#ifdef AGRMS_PSR_CACHED_SLOTS
        uint32_t free_slots = 0;
#endif
#ifdef AGRMS_PSR_STATEFUL
        conn.setup_stateful_send_cmd_bufs();
#endif
        auto send_item = [&](const Entry& e, uint32_t xb) {
            for_each_chunk_of_item(g, e, xb, [&](uint32_t w, uint32_t addr, uint32_t bytes, bool last) {
#ifdef AGRMS_PSR_DUMMY_ZONE
                {
                    AGRMS_PZONE("AG_P_DUMMY");
                }
#endif
                {
                    AGRMS_PZONE("AG_P_CBW");
                    cb_wait_front(cb, run_pages * (unflushed + 1));
                }
#ifdef AGRMS_PSR_HDR_ZONE
                DeviceZoneScopedN("AG_P_HDR");  // closes at the end of the chunk body (includes ISSUE/FLUSH)
#endif
                const uint32_t src = get_read_ptr(cb) + unflushed * chunk_bytes;
                const uint32_t xy = get_common_arg_val<uint32_t>(w);
                const uint64_t dst = get_noc_addr(xy & 0xFFFF, xy >> 16, addr, 0);
                volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[h];
                h = (h + 1 == num_headers) ? 0 : h + 1;
                if (last) {
                    hdr->to_noc_fused_unicast_write_atomic_inc(
                        NocUnicastAtomicIncFusedCommandHeader{dst, arrival_noc, 1, true}, bytes);
                } else {
                    hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, bytes);
                }
                {
                    AGRMS_PZONE("AG_P_ISSUE");
#ifdef AGRMS_PSR_CACHED_SLOTS
                    // poll the router's free-slot count only when the cached count runs out
                    while (free_slots == 0) {
                        free_slots = conn.get_num_free_write_slots();
                    }
                    --free_slots;
#else
                    conn.wait_for_empty_write_slot();
#endif
#ifdef AGRMS_PSR_STATEFUL
                    // payload + header through the pre-programmed write_reg cmd buf, credit through write_at's
                    conn.send_current_slot_stateful_non_blocking(src, bytes, reinterpret_cast<uint32_t>(hdr));
#else
                    conn.send_current_slot_non_blocking(src, bytes, reinterpret_cast<uint32_t>(hdr));
#endif
                }
                --chunks_left;
                if (++unflushed == group || chunks_left == 0) {
                    AGRMS_PZONE("AG_P_FLUSH");
                    noc_async_writes_flushed();  // sources + headers of the in-flight chunks have left L1
                    cb_pop_front(cb, run_pages * unflushed);
                    unflushed = 0;
                }
            });
        };
        AGRMS_ZONE("AG_SEND");  // every packet of the call (own + relayed items)
#ifdef AGRMS_PSR_DESC
        // port_stream_rate: the port reader hands over each chunk's destination with the chunk (descriptor ring in
        // this core's cb_x_landing, slot = CB chunk slot); no walk, no item boundaries here.
        volatile tt_l1_ptr uint32_t* desc =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(agrms_cb::cb_x_landing));
        uint32_t slot = 0;
#ifdef AGRMS_PSR_DESC_HYBRID
        {
            // the first own items come from the input tensor (port reader, no descriptors): walk them here. Steps
            // t < dram_items carry only own items (dram_items <= relay_lag, host-checked)
            const Entry own = out_entry(0);
            const uint32_t before = chunks_left;
            for (uint32_t t = 0; t < AGRMS_PSR_DESC_HYBRID && t < n_xb && !own.empty(); ++t) {
                send_item(own, t);
            }
            slot = (before - chunks_left) % (2 * group);
        }
#endif
        while (chunks_left > 0) {
            {
                AGRMS_PZONE("AG_P_CBW");
                cb_wait_front(cb, run_pages * (unflushed + 1));
            }
            invalidate_l1_cache();
            volatile tt_l1_ptr uint32_t* d = desc + 3 * slot;
            const uint32_t addr = d[0], xy = d[1], bw = d[2];
            slot = (slot + 1 == 2 * group) ? 0 : slot + 1;
            const uint32_t bytes = bw & 0x7FFFFFFFu;
            const uint32_t src = get_read_ptr(cb) + unflushed * chunk_bytes;
            const uint64_t dst = get_noc_addr(xy & 0xFFFF, xy >> 16, addr, 0);
            volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[h];
            h = (h + 1 == num_headers) ? 0 : h + 1;
            if (bw >> 31) {
                hdr->to_noc_fused_unicast_write_atomic_inc(
                    NocUnicastAtomicIncFusedCommandHeader{dst, arrival_noc, 1, true}, bytes);
            } else {
                hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, bytes);
            }
            {
                AGRMS_PZONE("AG_P_ISSUE");
#ifdef AGRMS_PSR_CACHED_SLOTS
                while (free_slots == 0) {
                    free_slots = conn.get_num_free_write_slots();
                }
                --free_slots;
#else
                conn.wait_for_empty_write_slot();
#endif
#ifdef AGRMS_PSR_STATEFUL
                conn.send_current_slot_stateful_non_blocking(src, bytes, reinterpret_cast<uint32_t>(hdr));
#else
                conn.send_current_slot_non_blocking(src, bytes, reinterpret_cast<uint32_t>(hdr));
#endif
            }
            --chunks_left;
            if (++unflushed == group || chunks_left == 0) {
                AGRMS_PZONE("AG_P_FLUSH");
                noc_async_writes_flushed();
                cb_pop_front(cb, run_pages * unflushed);
                unflushed = 0;
            }
        }
        (void)send_item;
#else
        const uint32_t steps = n_xb + (E_out - 1) * relay_lag;
        const Entry own = out_entry(0);
        for (uint32_t t = 0; t < steps; ++t) {
            if (t < n_xb && !own.empty()) {
                send_item(own, t);
            }
            for (uint32_t e = 1; e < E_out; ++e) {
                if (t < e * relay_lag) {
                    break;
                }
                const uint32_t xb = t - e * relay_lag;
                if (xb < n_xb) {
                    const Entry en = out_entry(e);
                    if (!en.empty()) {
                        send_item(en, xb);
                    }
                }
            }
        }
#endif
    }
    AGRMS_ZONE("AG_CLOSE");
    conn.close();
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
