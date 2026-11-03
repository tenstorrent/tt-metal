// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reference: models/demos/moe_block/tt/routing_stream.py:_Y_SEND (+ RA_SEND) (mstaletovic/moe-trim).
// Generated from the reference by tests/.../moe_block/fused_head/gen_head_kernels.py; edit both together.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
using namespace tt::tt_fabric;
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/fused_head/device/kernels/ra_send.hpp"

void kernel_main() {
    RASend rs;
    rs.init(0);
    size_t a = RASend::len(0);
    const uint32_t remote = get_arg_val<uint32_t>(a++);
    if (remote) {
        const uint32_t hello = get_arg_val<uint32_t>(a++), pc = get_arg_val<uint32_t>(a++),
                       pm = get_arg_val<uint32_t>(a++);
        auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
        // the hellos that land here a call (the rows whose first Y direction points at this row): one on a ring; on a
        // line none at the near end, two at the row before the far end
        const uint32_t hellos = get_arg_val<uint32_t>(a++);
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = PacketHeaderPool::allocate_header();
        conn.open();
        // measurement only: "started" to the Y neighbour's watcher; the neighbour's landing here is stamped when seen
        // (between the sends' polls, never waited on before they are done) - the rows' launch skew
        (void)fabric_set_unicast_route(hdr, static_cast<uint16_t>(pc), static_cast<uint16_t>(pm));
        hdr->to_noc_unicast_atomic_inc(
            NocUnicastAtomicIncCommandHeader{get_noc_addr(my_x[0], my_y[0], hello), 1, true});
        conn.wait_for_empty_write_slot();
        conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hdr), sizeof(PACKET_HEADER_TYPE));
        volatile tt_l1_ptr uint32_t* hw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hello);
        bool seen = hellos == 0;
        while (!rs.done) {
            invalidate_l1_cache();
            if (!seen && *hw) {
                DeviceTimestampedData("SE-YHELLO", 1);
                seen = true;
            }
            rs.poll(conn, hdr, false, RA_N_, RA_M_, RA_EPC_, RA_ME_, RA_HBP_, RA_BB_, RA_BBP_, RA_MAXP_);
        }
        if (hellos) {
            noc_semaphore_wait_min(hw, hellos);
            if (!seen) {
                DeviceTimestampedData("SE-YHELLO", 0);  // (late: after the sends)
            }
            noc_semaphore_inc(get_noc_addr(hello), 0u - hellos);  // (re-arm)
            noc_async_atomic_barrier();
        }
        rs.poll(conn, hdr, true, RA_N_, RA_M_, RA_EPC_, RA_ME_, RA_HBP_, RA_BB_, RA_BBP_, RA_MAXP_);
        conn.close();
    } else {
        struct NoConn {
            void wait_for_empty_write_slot() {}
            void send_current_slot_non_blocking(uint32_t, uint32_t, uint32_t) {}
            void send_payload_flush_blocking_from_address(uint32_t, uint32_t) {}
        } nc;
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = nullptr;
        rs.poll(nc, hdr, true, RA_N_, RA_M_, RA_EPC_, RA_ME_, RA_HBP_, RA_BB_, RA_BBP_, RA_MAXP_);
    }
    noc_async_atomic_barrier();
}
