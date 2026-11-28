// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Moved verbatim from models/demos/deepseek_v41_flash_d_p/tt/snapshot_send.py _PORT_KERNEL (SnapshotSend static)
// (NC.3).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
using namespace tt::tt_fabric;

constexpr uint32_t PAGE = get_compile_time_arg_val(0);  // bytes a row
constexpr uint32_t K = get_compile_time_arg_val(1);     // rows a packet when the slab rows are consecutive
constexpr uint32_t NB = 4;
constexpr uint32_t BR = 32;
constexpr auto g_args = TensorAccessorArgs<2>();
constexpr auto s_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();

void kernel_main() {
    // [ring, slab, page0, rr0..rr3 | role0..3, sem0..3, hx, hy, mesh_id, chip_id, connection...]
    size_t a = 0;
    const uint32_t ring = get_arg_val<uint32_t>(a++);
    const uint32_t slab = get_arg_val<uint32_t>(a++);
    const uint32_t page0 = get_arg_val<uint32_t>(a++);
    uint32_t rr[NB];
    for (uint32_t b = 0; b < NB; ++b) {
        rr[b] = get_arg_val<uint32_t>(a++);
    }
    uint32_t role[NB], sem[NB];
    for (uint32_t b = 0; b < NB; ++b) {
        role[b] = get_arg_val<uint32_t>(a++);
    }
    for (uint32_t b = 0; b < NB; ++b) {
        sem[b] = get_arg_val<uint32_t>(a++);
    }
    const uint32_t hx = get_arg_val<uint32_t>(a++);
    const uint32_t hy = get_arg_val<uint32_t>(a++);
    const uint16_t mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint16_t chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const auto G = TensorAccessor(g_args, ring, PAGE);
    const auto S = TensorAccessor(s_args, slab, PAGE);
    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
    const uint32_t buf = (get_write_ptr(0) + 63) & ~63u;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = PacketHeaderPool::allocate_header();
#if (defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0))
    (void)fabric_set_unicast_route(hdr, chip_id, mesh_id);
#else
#error "the snapshot send needs a 2D fabric (agent rule 4)"
#endif
    conn.open();
    for (uint32_t b = 0; b < NB; ++b) {
        if (role[b] == 0) {
            continue;
        }
        if (role[b] == 2) {  // relay: the block has landed in my slab (the home core's signal)
            volatile tt_l1_ptr uint32_t* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(b));
            noc_semaphore_wait_min(flag, 1);
        }
        const uint32_t p0 = page0 + b * BR;
        for (uint32_t i = 0; i < BR; ++i) {
            noc_async_read(role[b] == 1 ? G.get_noc_addr(rr[b] + i) : S.get_noc_addr(p0 + i), buf + i * PAGE, PAGE);
        }
        noc_async_read_barrier();
        const uint64_t arrival = get_noc_addr(hx, hy, sem[b]);
        for (uint32_t i = 0; i < BR; i += K) {
            const uint64_t dst = S.get_noc_addr(p0 + i);
            const bool run = (K > 1) && (S.get_noc_addr(p0 + i + K - 1) == dst + (K - 1) * PAGE);
            const uint32_t n = run ? K : 1;
            for (uint32_t j = 0; j < K; j += n) {
                const uint64_t d = run ? dst : S.get_noc_addr(p0 + i + j);
                const uint32_t bytes = n * PAGE;
                if (i + j + n == BR) {
                    hdr->to_noc_fused_unicast_write_atomic_inc(
                        NocUnicastAtomicIncFusedCommandHeader{d, arrival, 1, true}, bytes);
                } else {
                    hdr->to_noc_unicast_write(NocUnicastCommandHeader{d}, bytes);
                }
                conn.wait_for_empty_write_slot();
                conn.send_current_slot_non_blocking(buf + (i + j) * PAGE, bytes, reinterpret_cast<uint32_t>(hdr));
                noc_async_writes_flushed();  // the header left L1 before the next packet rewrites it
            }
        }
    }
    conn.close();
    for (uint32_t b = 0; b < NB; ++b) {
        if (role[b] == 2) {
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(b)), 0);
        }
    }
}
