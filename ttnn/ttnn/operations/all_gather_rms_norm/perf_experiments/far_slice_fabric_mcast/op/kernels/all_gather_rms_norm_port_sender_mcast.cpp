// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// all_gather_rms_norm port sender, FABRIC MULTICAST variant (perf experiment far_slice_fabric_mcast).
// One per (direction, link) per chip. Sends ONLY this chip's own slice (this link's columns), split into entries by
// row filter, each entry with its own hop count h: h == 1 is a one-hop unicast (as the base op), h >= 2 is a 2D line
// multicast (write-and-forward) landing on the next h chips of this direction at the same (core, offset) (the
// landing layout is identical on every chip). No relays: the far chip gets the far rows straight from their owner.
// Every item's last chunk is a fused write + atomic increment of arrival counter SEM_ARR_BASE + p (p = this chip's
// ring position) on port (d, l) - the port cores are placed identically on every chip, so that one NoC address is
// the receiving port of the stream on EVERY destination chip.
// Fence: the ready of port (opposite direction, l) of every destination chip arrives as one multicast atomic inc
// (each chip mcasts its ready back over as many hops as its farthest source); wait for H of them (H = farthest hop).
//
// Raw fabric API (documented helper bypass): kernel_lib's ccl::FabricStream arms a fixed payload size and one route;
// here chunks vary in size and each entry carries its own (unicast / multicast) route.

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

// h hops in this port's direction: 1 = unicast to the neighbour, >= 2 = line multicast over the next h chips.
inline void route_hops(volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr, uint16_t chip_id, uint16_t mesh_id, uint32_t h) {
#if (defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0)) || defined(EMULE_FABRIC_2D)
    if (h <= 1) {
        (void)fabric_set_unicast_route(hdr, chip_id, mesh_id);
        return;
    }
    uint16_t hops[4] = {0, 0, 0, 0};  // E, W, N, S
    const uint32_t dir = static_cast<uint32_t>(get_next_hop_router_direction(mesh_id, chip_id));
    hops[dir < 4 ? dir : 0] = static_cast<uint16_t>(h);
    fabric_set_mcast_route(hdr, chip_id, mesh_id, hops[0], hops[1], hops[2], hops[3]);
#else
    (void)chip_id;
    (void)mesh_id;
    auto* ll = reinterpret_cast<volatile tt_l1_ptr LowLatencyPacketHeader*>(hdr);
    if (h <= 1) {
        (void)fabric_set_unicast_route<false>(ll, static_cast<uint16_t>(1));
    } else {
        ll->to_chip_multicast(MulticastRoutingCommandHeader{1, static_cast<uint8_t>(h)});
    }
#endif
}

constexpr uint32_t MAX_ROUTES = 2;

inline void route_to_chip(volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr, uint16_t chip_id, uint16_t mesh_id, uint32_t j) {
#if (defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0)) || defined(EMULE_FABRIC_2D)
    (void)j;
    (void)fabric_set_unicast_route(hdr, chip_id, mesh_id);
#else
    (void)chip_id;
    (void)mesh_id;
    (void)fabric_set_unicast_route<false>(
        reinterpret_cast<volatile tt_l1_ptr LowLatencyPacketHeader*>(hdr), static_cast<uint16_t>(j));
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
    constexpr uint32_t rows_per_group = get_compile_time_arg_val(9);
    constexpr uint32_t chunk_bytes = run_pages * page_bytes;
    constexpr uint32_t cb = agrms_cb::cb_port_chunks;

    size_t a = 0;
    const uint32_t E_out = get_arg_val<uint32_t>(a++);
    const uint32_t out_idx = a;
    a += E_out;
    const uint32_t hops_idx = a;  // hop count per send entry
    a += E_out;
    const uint32_t segidx_idx = a;
    a += E_out;
    const uint32_t n_segs = get_arg_val<uint32_t>(a++);
    const uint32_t segs_idx = a;
    a += n_segs;
    const uint32_t ready_addr = get_arg_val<uint32_t>(a++);  // global semaphore (same address on every port core)
    const uint32_t ready_hops = get_arg_val<uint32_t>(a++);  // my ready reaches this many chips (0: none)
    const uint32_t ready_x = get_arg_val<uint32_t>(a++);     // port (opposite direction, l): same core on every chip
    const uint32_t ready_y = get_arg_val<uint32_t>(a++);
    const uint32_t ready_wait = get_arg_val<uint32_t>(a++);  // readies to wait for (= destination chips)
    const uint32_t peer_x = get_arg_val<uint32_t>(a++);      // port (d, l): same core on every chip
    const uint32_t peer_y = get_arg_val<uint32_t>(a++);
    const uint32_t arrival_sem_id = get_arg_val<uint32_t>(a++);  // SEM_ARR_BASE + my ring position
    const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint16_t dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint32_t open_conn = get_arg_val<uint32_t>(a++);
    // Variants (host env): ready_unicast = the ready goes as ready_hops unicasts (chips at distance 1..R) instead of
    // one R-hop multicast atomic inc; data_direct = an h-hop entry goes as h unicasts (distance 1..h) instead of one
    // h-hop multicast. chips: fabric chip id at distance 1..n_chips in this direction.
    const uint32_t ready_unicast = get_arg_val<uint32_t>(a++);
    const uint32_t data_direct = get_arg_val<uint32_t>(a++);
    const uint32_t n_chips = get_arg_val<uint32_t>(a++);
    const uint32_t chips_idx = a;
    a += n_chips;
    if (open_conn == 0) {
        return;
    }
    auto chip_at = [&](uint32_t j) { return static_cast<uint16_t>(get_arg_val<uint32_t>(chips_idx + j - 1)); };

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

    uint32_t chunks_left = 0;
    for (uint32_t e = 0; e < E_out; ++e) {
        const Entry en = out_entry(e);
        chunks_left += en.rows_in(0, R_t) * g.chunks_per_row(en);
    }

    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);  // appended last
    // pre-routed header rings, one per route: multicast = route r is send entry r (unicast or h-hop multicast);
    // direct = route r is the unicast to the chip at distance r + 1. Host: routes * group + 1 <= headers per RISC.
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[MAX_ROUTES][group];
    volatile tt_l1_ptr PACKET_HEADER_TYPE* ready_hdr = nullptr;
    uint32_t n_routes = 0;
    {
        AGRMS_ZONE("AG_SETUP_HDRS");
        uint32_t hmax = 0;
        for (uint32_t e = 0; e < E_out; ++e) {
            const uint32_t h = get_arg_val<uint32_t>(hops_idx + e);
            hmax = h > hmax ? h : hmax;
        }
        n_routes = data_direct ? hmax : E_out;
        for (uint32_t r = 0; r < n_routes; ++r) {
            for (uint32_t i = 0; i < group; ++i) {
                hdrs[r][i] = PacketHeaderPool::allocate_header();
                if (data_direct) {
                    route_to_chip(hdrs[r][i], chip_at(r + 1), dst_mesh_id, r + 1);
                } else {
                    route_hops(hdrs[r][i], dst_chip_id, dst_mesh_id, get_arg_val<uint32_t>(hops_idx + r));
                }
            }
        }
        if (ready_hops > 0) {
            ready_hdr = PacketHeaderPool::allocate_header();
            if (!ready_unicast) {
                route_hops(ready_hdr, dst_chip_id, dst_mesh_id, ready_hops);
            }
        }
    }
    {
        AGRMS_ZONE("AG_FENCE");
        conn.open();
        if (ready_hops > 0) {
            const uint32_t n_ready = ready_unicast ? ready_hops : 1;
            for (uint32_t r = 0; r < n_ready; ++r) {
                if (ready_unicast) {
                    route_to_chip(ready_hdr, chip_at(r + 1), dst_mesh_id, r + 1);
                }
                ready_hdr->to_noc_unicast_atomic_inc(
                    NocUnicastAtomicIncCommandHeader{get_noc_addr(ready_x, ready_y, ready_addr), 1, true});
                conn.wait_for_empty_write_slot();
                conn.send_payload_flush_blocking_from_address(
                    reinterpret_cast<uint32_t>(ready_hdr), sizeof(PACKET_HEADER_TYPE));
            }
        }
        if (chunks_left > 0) {
            volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
            noc_semaphore_wait_min(ready, ready_wait);
            noc_semaphore_inc(get_noc_addr(ready_addr), 0u - ready_wait);
            noc_async_atomic_barrier();
        }
    }
    if (chunks_left > 0) {
        const uint64_t arrival_noc = get_noc_addr(peer_x, peer_y, get_semaphore(arrival_sem_id), 0);
        uint32_t hi[MAX_ROUTES] = {0, 0};
        uint32_t unflushed = 0;
        auto send_item = [&](uint32_t ei, const Entry& e, uint32_t xb) {
            const uint32_t r0 = data_direct ? 0 : ei;
            const uint32_t r1 = data_direct ? get_arg_val<uint32_t>(hops_idx + ei) : ei + 1;
            for_each_chunk_of_item(g, e, xb, [&](uint32_t w, uint32_t addr, uint32_t bytes, bool last) {
                cb_wait_front(cb, run_pages * (unflushed + 1));
                const uint32_t src = get_read_ptr(cb) + unflushed * chunk_bytes;
                const uint32_t xy = get_common_arg_val<uint32_t>(w);
                const uint64_t dst = get_noc_addr(xy & 0xFFFF, xy >> 16, addr, 0);
                for (uint32_t r = r0; r < r1; ++r) {
                    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[r][hi[r]];
                    hi[r] = (hi[r] + 1 == group) ? 0 : hi[r] + 1;
                    if (last) {
                        hdr->to_noc_fused_unicast_write_atomic_inc(
                            NocUnicastAtomicIncFusedCommandHeader{dst, arrival_noc, 1, true}, bytes);
                    } else {
                        hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, bytes);
                    }
                    conn.wait_for_empty_write_slot();
                    conn.send_current_slot_non_blocking(src, bytes, reinterpret_cast<uint32_t>(hdr));
                }
                --chunks_left;
                if (++unflushed == group || chunks_left == 0) {
                    noc_async_writes_flushed();
                    cb_pop_front(cb, run_pages * unflushed);
                    unflushed = 0;
                }
            });
        };
        AGRMS_ZONE("AG_SEND");
        for (uint32_t t = 0; t < n_xb; ++t) {
            for (uint32_t e = 0; e < E_out; ++e) {
                const Entry en = out_entry(e);
                if (!en.empty()) {
                    send_item(e, en, t);
                }
            }
        }
    }
    AGRMS_ZONE("AG_CLOSE");
    conn.close();
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
