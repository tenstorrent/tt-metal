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
// the ready goes out on the same open connection that then carries the data.

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
    [[maybe_unused]] const uint32_t host_chunks = get_arg_val<uint32_t>(a++);  // presend: host chunk count
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

    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);  // appended last
#ifdef AGRMS_PRESEND_EARLY_OPEN
    // presend lever (c): the connection handshake's NoC reads (router cursor + free slots) are in flight while
    // header 0 is routed; open_finish() barriers on them just before the ready goes out
    {
        AGRMS_ZONE("AG_OPEN_START");
        conn.open_start();
    }
#endif
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[num_headers];
    uint32_t chunks_left = 0;
    auto count_chunks = [&]() {
        AGRMS_ZONE("AG_SETUP_COUNT");  // chunk count of the send entries
#ifdef AGRMS_PRESEND_HOSTCOUNT
        chunks_left = host_chunks;
        return;
#endif
        for (uint32_t e = 0; e < E_out; ++e) {
            const Entry en = out_entry(e);
            chunks_left += en.rows_in(0, R_t) * g.chunks_per_row(en);
        }
    };
    // presend_critical_path lever (a): fabric_set_unicast_route costs ~0.36 us per header (2D route-table decode);
    // every header of the ring goes to the same one-hop destination, so route header 0 once and copy its bytes
    // (routing fields, route buffer, destination node) to the others. The command fields are rewritten per packet.
    [[maybe_unused]] auto copy_hdr = [&](volatile tt_l1_ptr PACKET_HEADER_TYPE* dst) {
        constexpr uint32_t n_words = sizeof(PACKET_HEADER_TYPE) / sizeof(uint32_t);
        tt_l1_ptr uint32_t* d = reinterpret_cast<tt_l1_ptr uint32_t*>(const_cast<PACKET_HEADER_TYPE*>(dst));
        const tt_l1_ptr uint32_t* s0 =
            reinterpret_cast<const tt_l1_ptr uint32_t*>(const_cast<PACKET_HEADER_TYPE*>(hdrs[0]));
#pragma GCC unroll 64
        for (uint32_t i = 0; i < n_words; ++i) {
            d[i] = s0[i];
        }
    };
    auto route_rest = [&]() {
        AGRMS_ZONE("AG_SETUP_HDRS_REST");
        for (uint32_t h = 1; h < num_headers; ++h) {
            hdrs[h] = PacketHeaderPool::allocate_header();
#if defined(AGRMS_PRESEND_LAZY)
            // copied from header 0 at its first use (send loop)
#elif defined(AGRMS_PRESEND_DMA_COPY)
            // the copy as a local NoC write (DMA): the RISC only issues it; completed before the send loop
            noc_async_write(
                reinterpret_cast<uint32_t>(hdrs[0]),
                get_noc_addr(reinterpret_cast<uint32_t>(hdrs[h])),
                sizeof(PACKET_HEADER_TYPE));
#elif defined(AGRMS_PRESEND_A)
            // plain (non-volatile) word copy, unrolled: the loads issue back to back instead of one L1 round trip
            // per word (a volatile word loop measured ~0.2 us per 96 B header)
            constexpr uint32_t n_words = sizeof(PACKET_HEADER_TYPE) / sizeof(uint32_t);
            tt_l1_ptr uint32_t* d = reinterpret_cast<tt_l1_ptr uint32_t*>(const_cast<PACKET_HEADER_TYPE*>(hdrs[h]));
            const tt_l1_ptr uint32_t* s0 =
                reinterpret_cast<const tt_l1_ptr uint32_t*>(const_cast<PACKET_HEADER_TYPE*>(hdrs[0]));
            uint32_t tmp[n_words];
#pragma GCC unroll 64
            for (uint32_t i = 0; i < n_words; ++i) {
                tmp[i] = s0[i];
            }
#pragma GCC unroll 64
            for (uint32_t i = 0; i < n_words; ++i) {
                d[i] = tmp[i];
            }
#else
            route_one_hop(hdrs[h], dst_chip_id, dst_mesh_id);
#endif
        }
    };
#ifndef AGRMS_PRESEND_SPLIT_FENCE
    count_chunks();
#endif
    {
        AGRMS_ZONE("AG_SETUP_HDR0");
        hdrs[0] = PacketHeaderPool::allocate_header();
        route_one_hop(hdrs[0], dst_chip_id, dst_mesh_id);
    }
#ifndef AGRMS_PRESEND_SPLIT_FENCE
    route_rest();
#endif
    {
        AGRMS_ZONE("AG_FENCE");  // open + ready out + own ready in
        {
            AGRMS_ZONE("AG_OPEN");
#ifdef AGRMS_PRESEND_EARLY_OPEN
            conn.open_finish();
#else
            conn.open();
#endif
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
#ifdef AGRMS_PRESEND_SPLIT_FENCE
        // lever (c): the ready leaves first; the chunk count and the other headers overlap the peer's ready
        count_chunks();
        route_rest();
#endif
        if (chunks_left > 0) {
            AGRMS_ZONE("AG_READY_WAIT");
            volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
            noc_semaphore_wait_min(ready, 1);
            noc_semaphore_inc(get_noc_addr(ready_addr), 0u - 1u);
            noc_async_atomic_barrier();
        }
    }
#ifdef AGRMS_PRESEND_DMA_COPY
    noc_async_write_barrier();  // header copies landed
#endif
    if (chunks_left > 0) {
        const uint64_t arrival_noc = get_noc_addr(peer_x, peer_y, get_semaphore(sem_arrival_id), 0);
        uint32_t h = 0, unflushed = 0;
        [[maybe_unused]] uint32_t n_init = 1;
        [[maybe_unused]] bool first = true;
        auto send_item = [&](const Entry& e, uint32_t xb) {
            for_each_chunk_of_item(g, e, xb, [&](uint32_t w, uint32_t addr, uint32_t bytes, bool last) {
#ifdef AGRMS_PROFILE_ZONES
                if (first) {
                    AGRMS_ZONE("AG_FIRST_CHUNK_WAIT");  // the first chunk's staging (the reader's first push)
                    cb_wait_front(cb, run_pages);
                    first = false;
                }
#endif
                cb_wait_front(cb, run_pages * (unflushed + 1));
                const uint32_t src = get_read_ptr(cb) + unflushed * chunk_bytes;
                const uint32_t xy = get_common_arg_val<uint32_t>(w);
                const uint64_t dst = get_noc_addr(xy & 0xFFFF, xy >> 16, addr, 0);
                volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[h];
#ifdef AGRMS_PRESEND_LAZY
                if (h == n_init) {  // first use of header h (h cycles 0, 1, ...): copy the routed header 0
                    copy_hdr(hdr);
                    ++n_init;
                }
#endif
                h = (h + 1 == num_headers) ? 0 : h + 1;
                if (last) {
                    hdr->to_noc_fused_unicast_write_atomic_inc(
                        NocUnicastAtomicIncFusedCommandHeader{dst, arrival_noc, 1, true}, bytes);
                } else {
                    hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, bytes);
                }
                conn.wait_for_empty_write_slot();
                conn.send_current_slot_non_blocking(src, bytes, reinterpret_cast<uint32_t>(hdr));
                --chunks_left;
                if (++unflushed == group || chunks_left == 0) {
                    noc_async_writes_flushed();  // sources + headers of the in-flight chunks have left L1
                    cb_pop_front(cb, run_pages * unflushed);
                    unflushed = 0;
                }
            });
        };
        AGRMS_ZONE("AG_SEND");  // every packet of the call (own + relayed items)
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
    }
    AGRMS_ZONE("AG_CLOSE");
    conn.close();
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
