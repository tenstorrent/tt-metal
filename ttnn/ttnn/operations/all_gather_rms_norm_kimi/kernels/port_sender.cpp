// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Port core sender: one (ring direction, link) of this chip. Sends its host-fixed list of packets, one hop each: item k
// is in staging slot k (CB_STAGE), put there - payload + tag - by the local row core that holds it (this chip's own
// slice, or a peer's slice with more hops to go, as soon as it has landed). Every packet lands in the landing buffer
// of the downstream chip's row core for that (block, slice); the receiver polls the tag (no semaphore increment: a
// fused write + increment stalls the receiving router until the write has landed).
//
// No fence between calls: everything of call N lands in parity-(N % 2) landing halves, last used by call N - 2. Every
// chip of the group receives from every other chip in each call, so this chip having finished call N - 1 means every
// receiver has started call N - 1, i.e. finished call N - 2.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "agrn_common.hpp"

using namespace tt::tt_fabric;
using namespace agrn;

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
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t stage_slot_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t group = get_compile_time_arg_val(2);  // packets between header flushes
    constexpr uint32_t blk = get_compile_time_arg_val(3);    // landing tile slots per parity half

    size_t a = 0;
    const uint32_t ctr_addr = get_arg_val<uint32_t>(a++);
    const uint32_t land_addr = get_arg_val<uint32_t>(a++);
    const uint16_t dst_chip = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint16_t dst_mesh = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint32_t salt = get_arg_val<uint32_t>(a++);
    const uint32_t n_items = get_arg_val<uint32_t>(a++);
    const uint32_t items = a;  // per item: (x << 16) | y of the landing core, (ntiles << 20) | landing tile slot
    a += 2 * n_items;

    volatile tt_l1_ptr uint32_t* ctr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctr_addr);
    const uint32_t parity = *ctr & 1;
    const uint32_t tag = *ctr + 1;
    const uint32_t land_base = land_addr + parity * blk * tile_bytes;
    const uint32_t stage = get_write_ptr(CB_STAGE);

    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[group];
    {
        AGRN_ZONE("ps_hdrs");
        // the route lookup is slow (~0.35 us): do it once, copy the routed header to the others
        for (uint32_t h = 0; h < group; ++h) {
            hdrs[h] = PacketHeaderPool::allocate_header();
        }
        {
            AGRN_ZONE("ps_route");
            route_one_hop(hdrs[0], dst_chip, dst_mesh);
        }
        for (uint32_t h = 1; h < group; ++h) {
            volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hdrs[h]);
            volatile tt_l1_ptr uint32_t* s0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hdrs[0]);
            for (uint32_t w = 0; w < sizeof(PACKET_HEADER_TYPE) / 4; ++w) {
                d[w] = s0[w];
            }
        }
    }
    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
    {
        AGRN_ZONE("ps_open");
        conn.open();
    }

    AGRN_ZONE("ps_send_all");
    uint32_t unflushed = 0;
    for (uint32_t k = 0; k < n_items; ++k) {
        const uint32_t xy = get_arg_val<uint32_t>(items + 2 * k);
        const uint32_t info = get_arg_val<uint32_t>(items + 2 * k + 1);
        const uint32_t n = info >> 20, off = info & 0xFFFFF;
        const uint32_t src = stage + k * stage_slot_bytes;
        {
            AGRN_ZONE("ps_wait");
            wait_tag(src + n * tile_bytes, tag, salt);
        }
        const uint32_t bytes = n * tile_bytes + TAG_BYTES;
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[unflushed];
        hdr->to_noc_unicast_write(
            NocUnicastCommandHeader{get_noc_addr(xy >> 16, xy & 0xFFFF, land_base + off * tile_bytes)}, bytes);
        conn.wait_for_empty_write_slot();
        conn.send_current_slot_non_blocking(src, bytes, reinterpret_cast<uint32_t>(hdr));
        if (++unflushed == group || k + 1 == n_items) {
            noc_async_writes_flushed();  // the in-flight headers have left L1 (they are reused next)
            unflushed = 0;
        }
    }
    conn.close();
    noc_async_write_barrier();
    *ctr = *ctr + 1;
}
