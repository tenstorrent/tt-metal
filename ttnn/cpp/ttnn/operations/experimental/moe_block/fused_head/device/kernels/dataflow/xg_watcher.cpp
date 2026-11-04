// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The x all-gather's watcher core (under W_RA0 also the routing exchange's send, ra_send.hpp).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#ifdef W_RA0
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
using namespace tt::tt_fabric;
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/fused_head/device/kernels/ra_send.hpp"

#endif
// chunk m's shard of rank k (k = 1 .. G - 1 in the engine's order) is in once each stream carrying it counted it
// (a stream counts every shard it sends: m entries + its index + 1): then XREADY + 1 on every engine core
// (XREADY = m (G - 1) + k). args: chunks, the XREADY id, ranks G - 1, engine cores nc, counters ncnt; (counter address,
// final) x ncnt; cores (x, y) x nc; per rank: nreq, (counter address, the entry's index, the stream's entries a chunk)
void kernel_main() {
    size_t a = 0;
    const uint32_t nch = get_arg_val<uint32_t>(a++), sem = get_semaphore(get_arg_val<uint32_t>(a++));
    const uint32_t nk = get_arg_val<uint32_t>(a++), nc = get_arg_val<uint32_t>(a++), ncnt = get_arg_val<uint32_t>(a++);
    const uint32_t cnt0 = a, c0 = cnt0 + 2 * ncnt, r0 = c0 + 2 * nc;
    for (uint32_t m = 0; m < nch; ++m) {
        uint32_t b = r0;
        for (uint32_t k = 0; k < nk; ++k) {
            const uint32_t nreq = get_arg_val<uint32_t>(b++);
            for (uint32_t i = 0; i < nreq; ++i, b += 3) {
                noc_semaphore_wait_min(
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(b)),
                    m * get_arg_val<uint32_t>(b + 2) + get_arg_val<uint32_t>(b + 1) + 1);
            }
            for (uint32_t c = 0; c < nc; ++c) {
                noc_semaphore_inc(
                    get_noc_addr(get_arg_val<uint32_t>(c0 + 2 * c), get_arg_val<uint32_t>(c0 + 2 * c + 1), sem), 1);
            }
        }
    }
    for (uint32_t i = 0; i < ncnt; ++i) {  // (re-arm)
        noc_semaphore_inc(
            get_noc_addr(get_arg_val<uint32_t>(cnt0 + 2 * i)), 0u - get_arg_val<uint32_t>(cnt0 + 2 * i + 1));
    }
    noc_async_atomic_barrier();
#ifdef W_RA0
    {  // (the routing exchange's second Y direction: after the watch)
        RASend rs;
        rs.init(W_RA0);
        size_t a2 = W_RA0 + RASend::len(W_RA0);
        if (get_arg_val<uint32_t>(a2++)) {
            a2 += 3;  // (the Y hello's args: the BRISC's direction only)
            auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a2);
            volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = PacketHeaderPool::allocate_header();
            conn.open();
            rs.poll(conn, hdr, true, RA_N_, RA_M_, RA_EPC_, RA_ME_, RA_HBP_, RA_BB_, RA_BBP_, RA_MAXP_);
            conn.close();
        }
        noc_async_atomic_barrier();
    }
#endif
}
