// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TailA2A port sender (RISCV_0, NoC1): readies to the chips that send into my output through me; every destination's
// ready awaited; my runs into the destinations' output pages; at the end every arrival I expect - for each token loop
// in turn (LOOPS: loop l's output rows are the output's l x SO further on every chip). The fabric connection and the
// packet headers are set up once, for all loops. The staging CB is popped in whole groups (as the reader pushes them).

#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/mb_args.hpp"
#include "tail_a2a_port_common.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/fused_tail/device/kernels/tail_a2a/tail_rows.hpp"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
using namespace tt::tt_fabric;

// 2D fabrics route by destination node (multi-hop); 1D by hop count
inline void route_to(volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr, uint16_t chip_id, uint16_t mesh_id, uint32_t hops) {
#if (defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0)) || defined(EMULE_FABRIC_2D)
    (void)hops;
    (void)fabric_set_unicast_route(hdr, chip_id, mesh_id);
#else
    (void)chip_id;
    (void)mesh_id;
    (void)fabric_set_unicast_route<false>(
        reinterpret_cast<volatile tt_l1_ptr LowLatencyPacketHeader*>(hdr), static_cast<uint16_t>(hops));
#endif
}

constexpr uint32_t MAXD = 8;

void kernel_main() {
    mb_args::load();  // (my args: this chip's block of the DRAM table, common/args_table.hpp)
    constexpr auto out_args = TensorAccessorArgs<6>();
    constexpr uint32_t LOOPS = get_compile_time_arg_val(out_args.next_compile_time_args_offset());
    constexpr uint32_t SO = get_compile_time_arg_val(out_args.next_compile_time_args_offset() + 1);  // (bytes a loop)
    size_t a = 0;
    const uint32_t out_addr = mb_args::arg(a++), me = mb_args::arg(a++), RT = mb_args::arg(a++);
    const uint32_t NTB = mb_args::arg(a++), l = mb_args::arg(a++), L = mb_args::arg(a++);
    const uint32_t arr_addr = mb_args::arg(a++), expect_in = mb_args::arg(a++);
    const uint32_t ready_addr = mb_args::arg(a++), nready = mb_args::arg(a++);
#ifdef TL_SKIP
    const uint32_t la = a;  // (the live args, at a fixed place)
    a += 7;
#endif
    const uint32_t nrs = mb_args::arg(a++);  // (readies I send: x | y << 16, chip, mesh, hops)
    const uint32_t rs0 = a;
    a += 4 * nrs;
    const uint32_t nd = mb_args::arg(a++);  // (destinations: q | part << 8, chip, mesh, hops, runs, x | y << 16)
    const uint32_t d0 = a;
    a += 6 * nd;
    const bool sends = nd > 0 || nrs > 0;
    WorkerToFabricEdmSender conn;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[GROUP + 1];  // (the last: a padding row tile's bare increments)
    if (sends) {
        // (the fabric connection's eth channel, teardown and buffer-index semaphores: the block's last three)
        conn = WorkerToFabricEdmSender::build_from_eth_channel<ProgrammableCoreType::TENSIX>(
            mb_args::arg(a), mb_args::arg(a + 1), mb_args::arg(a + 2));
        for (uint32_t h = 0; h < GROUP + 1; ++h) {
            hdrs[h] = PacketHeaderPool::allocate_header();
        }
        conn.open();
    }
    for (uint32_t lp = 0; lp < LOOPS; ++lp) {
        const auto out = TensorAccessor(out_args, out_addr + lp * SO, PB);
#ifdef TL_SKIP
        const uint64_t live = tl_live(la, get_write_ptr(CB_LIVE) + LIVE_BRISC, RT, lp);
#else
        constexpr uint64_t live = 0;
#endif
        if (sends) {
            for (uint32_t i = 0; i < nrs; ++i) {  // (my output may be written: say so to each of its senders here)
                const uint32_t xy = mb_args::arg(rs0 + 4 * i);
                route_to(
                    hdrs[0],
                    mb_args::arg(rs0 + 4 * i + 1),
                    mb_args::arg(rs0 + 4 * i + 2),
                    mb_args::arg(rs0 + 4 * i + 3));
                hdrs[0]->to_noc_unicast_atomic_inc(
                    NocUnicastAtomicIncCommandHeader{get_noc_addr(xy & 0xFFFF, xy >> 16, ready_addr), 1, true});
                conn.wait_for_empty_write_slot();
                conn.send_payload_flush_blocking_from_address(
                    reinterpret_cast<uint32_t>(hdrs[0]), sizeof(PACKET_HEADER_TYPE));
            }
            if (nd > 0) {
                volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
                noc_semaphore_wait_min(ready, nready);
                noc_semaphore_inc(get_noc_addr(ready_addr), 0u - nready);
                noc_async_atomic_barrier();
            }
            uint32_t sent[MAXD] = {};
            uint32_t h = 0, unflushed = 0;
            for (uint32_t r = 0; r < RT; ++r) {
                for (uint32_t k = 0; k < nd; ++k) {
                    const uint32_t e = mb_args::arg(d0 + 6 * k), part = (e >> 8) & 3;
                    const uint16_t chip = mb_args::arg(d0 + 6 * k + 1), mesh = mb_args::arg(d0 + 6 * k + 2);
                    const uint32_t hops = mb_args::arg(d0 + 6 * k + 3), runs = mb_args::arg(d0 + 6 * k + 4);
                    const uint32_t pxy = mb_args::arg(d0 + 6 * k + 5);
                    const uint64_t arr = get_noc_addr(pxy & 0xFFFF, pxy >> 16, arr_addr);
                    const uint32_t base = (me * RT + r) * NTB;  // (my rows' row tile r in the destination's output)
                    // (a padding row tile: nothing sent, its runs counted - an increment due goes alone, so the
                    // destination's arrival count stays the plan's)
                    if (!tl_is_live(live, r)) {
                        for_each_run(NTB, l, L, part, [&](uint32_t, uint32_t) {
                            if (++sent[k] % INC == 0 || sent[k] == runs) {
                                route_to(hdrs[GROUP], chip, mesh, hops);
                                hdrs[GROUP]->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{arr, 1, true});
                                conn.wait_for_empty_write_slot();
                                conn.send_payload_flush_blocking_from_address(
                                    reinterpret_cast<uint32_t>(hdrs[GROUP]), sizeof(PACKET_HEADER_TYPE));
                            }
                        });
                        continue;
                    }
                    for_each_run(NTB, l, L, part, [&](uint32_t m, uint32_t n) {
                        cb_wait_front(CB, RP * (unflushed + 1));
                        const uint32_t src = get_read_ptr(CB) + unflushed * RP * PB;
                        // fabric router writes on NOC1: resolve the DRAM endpoint for its NoC (SYS-1419)
                        // (pages m, m + NB, ..: consecutive)
                        const uint64_t dst = tt::tt_fabric::linear::addrgen_detail::get_noc_address(out, base + m);
                        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[h];
                        h = h + 1 == GROUP ? 0 : h + 1;
                        route_to(hdr, chip, mesh, hops);
                        if (++sent[k] % INC == 0 || sent[k] == runs) {
                            hdr->to_noc_fused_unicast_write_atomic_inc(
                                NocUnicastAtomicIncFusedCommandHeader{dst, arr, 1, true}, n * PB);
                        } else {
                            hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, n * PB);
                        }
                        conn.wait_for_empty_write_slot();
                        conn.send_current_slot_non_blocking(src, n * PB, reinterpret_cast<uint32_t>(hdr));
                        if (++unflushed == GROUP) {
                            noc_async_writes_flushed();
                            cb_pop_front(CB, RP * GROUP);
                            unflushed = 0;
                        }
                    });
                }
            }
            if (unflushed) {  // (a whole group: the reader pushed a whole group)
                noc_async_writes_flushed();
                cb_wait_front(CB, RP * GROUP);
                cb_pop_front(CB, RP * GROUP);
            }
        }
        if (expect_in > 0) {  // (everything sent into my output through my side landed; then re-armed)
            volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arr_addr);
            noc_semaphore_wait_min(arrived, expect_in);
            noc_semaphore_inc(get_noc_addr(arr_addr), 0u - expect_in);
            noc_async_atomic_barrier();
        }
    }  // (token loop lp)
    if (sends) {
        conn.close();
    }
}

