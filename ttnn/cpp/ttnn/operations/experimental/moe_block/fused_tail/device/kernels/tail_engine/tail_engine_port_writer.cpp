// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TailEngine gather port sender (RISCV_0, NoC1): the fence (my ready to the port that writes into this chip; the next
// chip's ready before I write there), then every packet of my units into the next chip's owners - for each token loop
// in turn (the gate a loop: tail_engine_common.hpp); the fabric connection and the headers set up once.

#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/mb_args.hpp"
#include "tail_engine_port_common.hpp"
#include "api/debug/kernel_profiler.hpp"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
using namespace tt::tt_fabric;

void kernel_main() {
    mb_args::load();  // (my args: this chip's block of the DRAM table, common/args_table.hpp)
    const uint32_t land = get_write_ptr(CB_LAND);
    size_t a = 0;
    const uint32_t sl = get_write_ptr(CB_WD), ready = mb_args::arg(a++);  // (the next chip's landed words)
    const uint32_t send_ready = mb_args::arg(a++), rx = mb_args::arg(a++), ry = mb_args::arg(a++);
    const uint16_t mesh_id = mb_args::arg(a++), chip_id = mb_args::arg(a++);
    const uint32_t n = mb_args::arg(a++), units = a, ak = units + n, xs = ak + 2 * GX, ys = xs + GX;
    a = ys + PAIRS;
    if (n == 0 && !send_ready) {  // (no link: still in every gate - its go is incremented a loop, re-armed here)
        for (uint32_t l = 0; l < LOOPS; ++l) {
            wait_zero_gate(PZW);
        }
        return;
    }
    // the fabric connection and the packet headers: once a call, for every loop
    // (the fabric connection's eth channel, teardown and buffer-index semaphores: the block's last three)
    auto conn = WorkerToFabricEdmSender::build_from_eth_channel<ProgrammableCoreType::TENSIX>(
        mb_args::arg(a), mb_args::arg(a + 1), mb_args::arg(a + 2));
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[PG];
    for (uint32_t h = 0; h < PG; ++h) {
        hdrs[h] = PacketHeaderPool::allocate_header();
        (void)fabric_set_unicast_route(hdrs[h], chip_id, mesh_id);
    }
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[0];
    conn.open();
    for (uint32_t l = 0; l < LOOPS; ++l) {
        // (loop 0: this chip's engine words zeroed; loop l: everyone here done with loop l - 1 - my ready goes out only
        // then: the previous chip's loop l increments and shares land in them)
        wait_zero_gate(PZW);
        if (send_ready) {  // (my landing regions are free for this loop)
            hdr->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{get_noc_addr(rx, ry, ready), 1, true});
            conn.wait_for_empty_write_slot();
            conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hdr), sizeof(PACKET_HEADER_TYPE));
        }
        if (n > 0) {
            volatile tt_l1_ptr uint32_t* rd = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready);
            noc_semaphore_wait_min(rd, 1);
            noc_semaphore_inc(get_noc_addr(ready), 0u - 1u);  // (re-arm: one ready a loop, one waiter)
            noc_async_atomic_barrier();
        }
        // the reader's groups: PG slots each, its packet count in CBM; a header a slot, one flush a group
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
            const uint32_t w = mb_args::arg(units + u), part = w & 0xFF, pos = w >> 8;  // (pos: there)
            auto send = [&](uint32_t pr, uint32_t s, uint32_t cx, uint32_t k0, uint32_t t0, uint32_t m, bool last) {
                if (gi == gn) {
                    next_group();
                }
                const uint32_t ox = mb_args::arg(xs + cx), oy = mb_args::arg(ys + pr);
                const uint64_t dst = get_noc_addr(ox, oy, land + (s * SHMAX + t0 - k0) * 2048, 0);
                volatile tt_l1_ptr PACKET_HEADER_TYPE* hd = hdrs[gi];
                const uint32_t c0 = part == 2 ? GX / 2 : 0, c1 = part == 1 ? GX / 2 : GX;
                // (a watcher's share: its group's last packet)
                if (last && ((cx - c0) % WG == WG - 1 || cx == c1 - 1)) {
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
            };
            walk(part, pos, ak, send);
        }
        if (gn > 0) {
            noc_async_writes_flushed();
            cb_pop_front(CBP, PG * RP);
        }
    }
    conn.close();
}

