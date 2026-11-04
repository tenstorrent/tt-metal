// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TailEngine gather port sender (RISCV_0, NoC1): the fence (my ready to the port that writes into this chip; the next
// chip's ready before I write there), then every packet of my units into the next chip's owners.

#include "tail_engine_port_common.hpp"
#include "api/debug/kernel_profiler.hpp"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
using namespace tt::tt_fabric;

void kernel_main() {
    const uint32_t land = get_write_ptr(CB_LAND);
    size_t a = 0;
    const uint32_t sl = get_arg_val<uint32_t>(a++), ready = get_arg_val<uint32_t>(a++);
    const uint32_t send_ready = get_arg_val<uint32_t>(a++), rx = get_arg_val<uint32_t>(a++),
                   ry = get_arg_val<uint32_t>(a++);
    const uint16_t mesh_id = get_arg_val<uint32_t>(a++), chip_id = get_arg_val<uint32_t>(a++);
    const uint32_t n = get_arg_val<uint32_t>(a++), units = a, ak = units + n, xs = ak + 2 * GX, ys = xs + GX;
    a = ys + PAIRS;
    if (n == 0 && !send_ready) {
        return;
    }
    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = PacketHeaderPool::allocate_header();
    (void)fabric_set_unicast_route(hdr, chip_id, mesh_id);
    conn.open();
    if (send_ready) {  // (this program runs: everything before it on this chip is done - my landing regions are free)
        hdr->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{get_noc_addr(rx, ry, ready), 1, true});
        conn.wait_for_empty_write_slot();
        conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hdr), sizeof(PACKET_HEADER_TYPE));
    }
    if (n > 0) {
        volatile tt_l1_ptr uint32_t* rd = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready);
        noc_semaphore_wait_min(rd, 1);
        noc_semaphore_inc(get_noc_addr(ready), 0u - 1u);  // (re-arm: one ready a call, one waiter)
        noc_async_atomic_barrier();
    }
    // the reader's groups: PG slots each, its packet count in CBM; a header a slot, one flush a group
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[PG];
    hdrs[0] = hdr;
    for (uint32_t h = 1; h < PG; ++h) {
        hdrs[h] = PacketHeaderPool::allocate_header();
        (void)fabric_set_unicast_route(hdrs[h], chip_id, mesh_id);
    }
    uint32_t gi = 0, gn = 0;
    auto next_group = [&]() {
        if (gn > 0) {
            noc_async_writes_flushed();  // (the group's packets and headers have left L1)
            cb_pop_front(CBP, PG * RP);
        }
        {
            DeviceZoneScopedN("TP-SGWAIT");  // (the reader's next group not read yet)
            cb_wait_front(CBM, 1);
        }
        gn = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(CBM));
        cb_pop_front(CBM, 1);
        cb_wait_front(CBP, PG * RP);
        gi = 0;
    };
    for (uint32_t u = 0; u < n; ++u) {
        DeviceZoneScopedN("TP-SUNIT");
        const uint32_t w = get_arg_val<uint32_t>(units + u), part = w & 0xFF, pos = w >> 8;  // (pos: there)
        walk(part, pos, ak, [&](uint32_t pr, uint32_t s, uint32_t cx, uint32_t k0, uint32_t t0, uint32_t m, bool last) {
            if (gi == gn) {
                next_group();
            }
            const uint32_t ox = get_arg_val<uint32_t>(xs + cx), oy = get_arg_val<uint32_t>(ys + pr);
            const uint64_t dst = get_noc_addr(ox, oy, land + (s * SHMAX + t0 - k0) * 2048, 0);
            volatile tt_l1_ptr PACKET_HEADER_TYPE* hd = hdrs[gi];
            const uint32_t c0 = part == 2 ? GX / 2 : 0, c1 = part == 1 ? GX / 2 : GX;
            if (last && ((cx - c0) % WG == WG - 1 || cx == c1 - 1)) {  // (a watcher's share: its group's last packet)
                hd->to_noc_fused_unicast_write_atomic_inc(
                    NocUnicastAtomicIncFusedCommandHeader{dst, get_noc_addr(ox, oy, sl + 16 * s, 0), 1, true},
                    m * 2048);
            } else {
                hd->to_noc_unicast_write(NocUnicastCommandHeader{dst}, m * 2048);
            }
            conn.wait_for_empty_write_slot();
            conn.send_current_slot_non_blocking(
                get_read_ptr(CBP) + gi * RP * 2048, m * 2048, reinterpret_cast<uint32_t>(hd));
            ++gi;
        });
    }
    if (gn > 0) {
        noc_async_writes_flushed();
        cb_pop_front(CBP, PG * RP);
    }
    conn.close();
}
