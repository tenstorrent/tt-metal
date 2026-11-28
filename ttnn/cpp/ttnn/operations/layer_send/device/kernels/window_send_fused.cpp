// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Moved verbatim from models/demos/deepseek_v41_flash_d_p/tt/window_send.py _KERNEL_FUSED (WindowSend.fused, the send)
// (NC.3).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
using namespace tt::tt_fabric;

constexpr uint32_t PAGE = get_compile_time_arg_val(0);  // bytes a page (a 32-row block)
constexpr uint32_t SEG = get_compile_time_arg_val(1);   // bytes a packet (whole 64 B lines)
constexpr uint32_t NSEG = get_compile_time_arg_val(2);  // packets a page
constexpr uint32_t NP = get_compile_time_arg_val(3);    // pages sent
constexpr uint32_t P0 = get_compile_time_arg_val(4);    // the region's first page sent (its last NP pages)
constexpr uint32_t B = get_compile_time_arg_val(5);     // pages read a NoC barrier (BATCH; 1 when NSEG > 1)
constexpr uint32_t SLOT = (SEG + 63) & ~63u;            // L1 bytes a page's packet takes in the buffer
constexpr auto r_args = TensorAccessorArgs<6>();
constexpr auto t_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
constexpr auto o_args = TensorAccessorArgs<t_args.next_compile_time_args_offset()>();
constexpr uint32_t Q0 = get_compile_time_arg_val(o_args.next_compile_time_args_offset());

inline void route_to(volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr, uint16_t chip_id, uint16_t mesh_id) {
#if (defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0))
    (void)fabric_set_unicast_route(hdr, chip_id, mesh_id);
#else
#error "the window send needs a 2D fabric (agent rule 4)"
#endif
}

void kernel_main() {
    size_t a = 0;
    const uint32_t region = get_arg_val<uint32_t>(a++);
    const uint32_t tail = get_arg_val<uint32_t>(a++);
    const uint32_t rows = get_arg_val<uint32_t>(a++);
    const uint32_t last = get_arg_val<uint32_t>(a++);  // 1: send the kept tail, then keep my own last pages
    const uint32_t sem = get_arg_val<uint32_t>(a++);   // the arrival counter (same address on every chip)
    const uint32_t px = get_arg_val<uint32_t>(a++);    // its core (NoC coordinates, the same on every chip)
    const uint32_t py = get_arg_val<uint32_t>(a++);
    const uint16_t mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint16_t chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const auto R = TensorAccessor(r_args, region, PAGE);
    const auto T = TensorAccessor(t_args, tail, PAGE);
    const auto O = TensorAccessor(o_args, rows, PAGE);
    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
    cb_reserve_back(0, 1);
    const uint32_t buf = (get_write_ptr(0) + 63) & ~63u;  // 64 B aligned, as the DRAM pages
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = PacketHeaderPool::allocate_header();
    route_to(hdr, chip_id, mesh_id);
    conn.open();
    const uint64_t arrival = get_noc_addr(px, py, sem);
    constexpr uint32_t total = NP * NSEG;
    uint32_t k = 0;
    for (uint32_t p0 = 0; p0 < NP; p0 += B) {
        const uint32_t nb = (NP - p0) < B ? (NP - p0) : B;
        for (uint32_t s = 0; s < NSEG; ++s) {
            const uint32_t off = s * SEG;
            const uint32_t bytes = (PAGE - off) < SEG ? (PAGE - off) : SEG;
            for (uint32_t i = 0; i < nb; ++i) {
                const uint32_t p = p0 + i;
                const uint64_t src = last ? T.get_noc_addr(p, off) : O.get_noc_addr(Q0 + p, off);
                noc_async_read(src, buf + i * SLOT, bytes);
            }
            noc_async_read_barrier();
            for (uint32_t i = 0; i < nb; ++i) {
                const uint64_t dst = R.get_noc_addr(p0 + i, off);  // the receiver's region (same buffer address)
                if (++k == total) {
                    hdr->to_noc_fused_unicast_write_atomic_inc(
                        NocUnicastAtomicIncFusedCommandHeader{dst, arrival, 1, true}, bytes);
                } else {
                    hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, bytes);
                }
                conn.wait_for_empty_write_slot();
                conn.send_current_slot_non_blocking(buf + i * SLOT, bytes, reinterpret_cast<uint32_t>(hdr));
                noc_async_writes_flushed();  // the header left L1 before the next packet rewrites it
            }
        }
    }
    conn.close();
    if (last) {  // my last NP pages are chip 0's carry of the next chunk
        for (uint32_t p0 = 0; p0 < NP; p0 += B) {
            const uint32_t nb = (NP - p0) < B ? (NP - p0) : B;
            for (uint32_t s = 0; s < NSEG; ++s) {
                const uint32_t off = s * SEG;
                const uint32_t bytes = (PAGE - off) < SEG ? (PAGE - off) : SEG;
                for (uint32_t i = 0; i < nb; ++i) {
                    noc_async_read(O.get_noc_addr(Q0 + p0 + i, off), buf + i * SLOT, bytes);
                }
                noc_async_read_barrier();
                for (uint32_t i = 0; i < nb; ++i) {
                    noc_async_write(buf + i * SLOT, T.get_noc_addr(p0 + i, off), bytes);
                }
                noc_async_write_barrier();
            }
        }
    }
    volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem);
    noc_semaphore_wait_min(arrived, 1);
    noc_semaphore_inc(get_noc_addr(sem), 0u - 1u);
    noc_async_atomic_barrier();
}
