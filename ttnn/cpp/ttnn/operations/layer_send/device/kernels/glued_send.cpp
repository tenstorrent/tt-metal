// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Moved verbatim from models/demos/deepseek_v41_flash_d_p/tt/packed_glued.py _GSEND_KERNEL (GluedSend; G_SEND / G_RECV
// are defines) (NC.3).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#if G_SEND
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
using namespace tt::tt_fabric;
#endif

constexpr uint32_t PAGE = get_compile_time_arg_val(0);  // bytes a page (a row)
constexpr uint32_t SEG = get_compile_time_arg_val(1);   // bytes a packet (whole 64 B lines)
constexpr uint32_t NSEG = get_compile_time_arg_val(2);  // packets a page
constexpr uint32_t NP = get_compile_time_arg_val(3);    // rows a carry (128)
constexpr uint32_t B = get_compile_time_arg_val(4);     // pages read a NoC barrier
constexpr uint32_t Q0 = get_compile_time_arg_val(5);    // the first of the chip's own rows sent (its last NP)
constexpr uint32_t C0 = get_compile_time_arg_val(6);    // the region row of carry block 0 (128 + R)
constexpr uint32_t NS = get_compile_time_arg_val(7);    // segments (carry blocks, tail tensors)
constexpr uint32_t SLOT = (SEG + 63) & ~63u;
constexpr auto r_args = TensorAccessorArgs<8>();
constexpr auto o_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
constexpr auto t0_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
constexpr auto t1_args = TensorAccessorArgs<t0_args.next_compile_time_args_offset()>();
constexpr auto t2_args = TensorAccessorArgs<t1_args.next_compile_time_args_offset()>();
constexpr auto t3_args = TensorAccessorArgs<t2_args.next_compile_time_args_offset()>();
constexpr auto t4_args = TensorAccessorArgs<t3_args.next_compile_time_args_offset()>();
constexpr auto t5_args = TensorAccessorArgs<t4_args.next_compile_time_args_offset()>();

template <typename Acc, typename Reg>
inline void carry_copy(const Acc& T, const Reg& Rg, uint32_t buf, uint32_t base, uint32_t rot, uint32_t row0) {
    for (uint32_t p0 = 0; p0 < NP; p0 += B) {
        const uint32_t nb = (NP - p0) < B ? (NP - p0) : B;
        for (uint32_t s = 0; s < NSEG; ++s) {
            const uint32_t off = s * SEG;
            const uint32_t bytes = (PAGE - off) < SEG ? (PAGE - off) : SEG;
            for (uint32_t i = 0; i < nb; ++i) {
                noc_async_read(T.get_noc_addr(base + (p0 + i + rot) % NP, off), buf + i * SLOT, bytes);
            }
            noc_async_read_barrier();
            for (uint32_t i = 0; i < nb; ++i) {
                noc_async_write(buf + i * SLOT, Rg.get_noc_addr(row0 + p0 + i, off), bytes);
            }
            noc_async_write_barrier();
        }
    }
}

void kernel_main() {
    size_t a = 0;
    const uint32_t region = get_arg_val<uint32_t>(a++);
    const uint32_t rows = get_arg_val<uint32_t>(a++);
    const uint32_t sem = get_arg_val<uint32_t>(a++);  // the arrival counter (same address on every chip)
    const uint32_t px = get_arg_val<uint32_t>(a++);
    const uint32_t py = get_arg_val<uint32_t>(a++);
    const uint16_t mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint16_t chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    uint32_t tail[6];
    for (uint32_t t = 0; t < 6; ++t) {
        tail[t] = get_arg_val<uint32_t>(a++);
    }
    const size_t v = a;  // per segment: active, tail tensor, first page, rotation (per-run values)
    a += 4 * NS;
    const auto R = TensorAccessor(r_args, region, PAGE);
    const auto O = TensorAccessor(o_args, rows, PAGE);
    cb_reserve_back(0, 1);
    const uint32_t buf = (get_write_ptr(0) + 63) & ~63u;
#if G_SEND
    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = PacketHeaderPool::allocate_header();
#if (defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0))
    (void)fabric_set_unicast_route(hdr, chip_id, mesh_id);
#else
#error "the glued window send needs a 2D fabric (agent rule 4)"
#endif
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
                noc_async_read(O.get_noc_addr(Q0 + p0 + i, off), buf + i * SLOT, bytes);
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
                noc_async_writes_flushed();
            }
        }
    }
    conn.close();
#else
    (void)px;
    (void)py;
    (void)mesh_id;
    (void)chip_id;
#endif
    // every continuing segment's slot carry into its block (the same rows on every chip)
    for (uint32_t s = 0; s < NS; ++s) {
        if (get_arg_val<uint32_t>(v + 4 * s) == 0) {
            continue;
        }
        const uint32_t t = get_arg_val<uint32_t>(v + 4 * s + 1);
        const uint32_t base = get_arg_val<uint32_t>(v + 4 * s + 2);
        const uint32_t rot = get_arg_val<uint32_t>(v + 4 * s + 3);
        const uint32_t row0 = C0 + s * NP;
        if (t == 0) {
            carry_copy(TensorAccessor(t0_args, tail[0], PAGE), R, buf, base, rot, row0);
        } else if (t == 1) {
            carry_copy(TensorAccessor(t1_args, tail[1], PAGE), R, buf, base, rot, row0);
        } else if (t == 2) {
            carry_copy(TensorAccessor(t2_args, tail[2], PAGE), R, buf, base, rot, row0);
        } else if (t == 3) {
            carry_copy(TensorAccessor(t3_args, tail[3], PAGE), R, buf, base, rot, row0);
        } else if (t == 4) {
            carry_copy(TensorAccessor(t4_args, tail[4], PAGE), R, buf, base, rot, row0);
        } else {
            carry_copy(TensorAccessor(t5_args, tail[5], PAGE), R, buf, base, rot, row0);
        }
    }
#if G_RECV
    volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem);
    noc_semaphore_wait_min(arrived, 1);
    noc_semaphore_inc(get_noc_addr(sem), 0u - 1u);
    noc_async_atomic_barrier();
#endif
}
