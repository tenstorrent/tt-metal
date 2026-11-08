// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// mhc_post feed port sender (deepseek_v41_flash_d_p P.36; BRISC, NoC 1). Port (side d, link l): its targets are my
// group's positions t != me with side(me, t) == d (the feed table, common args 3..; the fabric channel of my chip's
// slot from the table's channel bytes, written into runtime arg 2 before the connection reads it). Steps:
//   1. ready fence: one atomic increment into the ready semaphore (global, common arg 1) of each target's port toward
//      this chip on link l (slot side(t, me) L + l: "my feed_output is free for you"), then wait for one from each;
//   2. per target: its stream (mhc_post_feed_common.hpp) as fabric writes into the target's feed_output pages
//      (me RT + r) cpt + k - t cpt, scatter writes of up to P tiles; the last tile alone, fused with an increment of
//      the target's arrival semaphore (global, common arg 2) on its port 0 (an empty stream: the increment alone);
//   3. port 0 only: wait for the (G - 1) L arrivals into this chip's feed_output.
// The waits re-arm their semaphores (decrement by what they saw) before anything is sent that the peers' next call
// could answer.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "mhc_post_feed_common.hpp"

namespace {

inline void wait_and_rearm(uint32_t addr, uint32_t n) {
    if (n == 0) {
        return;
    }
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    noc_semaphore_wait_min(s, n);
    noc_semaphore_inc(get_noc_addr(addr), 0u - n);
    noc_async_atomic_barrier();
}

}  // namespace

void kernel_main() {
    using namespace tt::tt_fabric;
    constexpr uint32_t cb_feed = get_compile_time_arg_val(0);
    constexpr uint32_t page = get_compile_time_arg_val(1);
    constexpr uint32_t P = get_compile_time_arg_val(2);
    constexpr uint32_t rt = get_compile_time_arg_val(3);
    constexpr uint32_t bw = get_compile_time_arg_val(4);
    constexpr uint32_t ct = get_compile_time_arg_val(5);
    constexpr uint32_t cpt = get_compile_time_arg_val(6);
    constexpr auto f_args = TensorAccessorArgs<7>();
    constexpr uint32_t slots = 8;  // cb_feed's packet slots (FEED_SLOTS)
    constexpr uint32_t num_headers = 4;
    constexpr uint32_t FT0 = 3;

    const auto f_acc = TensorAccessor(f_args, get_common_arg_val<uint32_t>(0), page);
    const uint32_t ready_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t arrival_addr = get_common_arg_val<uint32_t>(2);
    const mhc_feed::Table<FT0> ft;
    const uint32_t me = ft.me, links = ft.L, G = ft.G;

    const uint32_t d = get_arg_val<uint32_t>(0);
    const uint32_t l = get_arg_val<uint32_t>(1);
    const uint32_t recv_expect = (d == 0 && l == 0) ? (G - 1) * links : 0;  // port 0: every other chip's L streams
    const uint32_t recv_xy = ft.port_xy(0);
    uint32_t n_targets = 0;
    for (uint32_t t = 0; t < G; ++t) {
        n_targets += (t != me && ft.side(me, t) == d) ? 1 : 0;
    }

    if (n_targets > 0) {
        // my chip's fabric channel for this slot -> runtime arg 2 (then: arg 3, 4 = the connection's semaphore ids)
        const uint32_t k = ft.pos * mhc_feed_layout::FT_MAX_PORTS + d * links + l;
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_addr(2)) =
            mhc_feed::Table<FT0>::byte(mhc_feed_layout::FT_WORDS, k);
        size_t a = 2;
        auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[num_headers];
        for (uint32_t h = 0; h < num_headers; ++h) {
            hdrs[h] = PacketHeaderPool::allocate_header();
        }
        conn.open();
        // 1. the ready fence
        for (uint32_t t = 0; t < G; ++t) {
            if (t == me || ft.side(me, t) != d) {
                continue;
            }
            const uint32_t rxy = ft.port_xy(ft.side(t, me) * links + l);
            volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[0];
            (void)fabric_set_unicast_route(hdr, static_cast<uint16_t>(ft.chip_of(t)), static_cast<uint16_t>(ft.mesh));
            hdr->to_noc_unicast_atomic_inc(
                NocUnicastAtomicIncCommandHeader{get_noc_addr(rxy >> 16, rxy & 0xFFFFu, ready_addr), 1, true});
            conn.wait_for_empty_write_slot();
            conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hdr), sizeof(PACKET_HEADER_TYPE));
        }
        wait_and_rearm(ready_addr, n_targets);

        // 2. the streams
        const uint64_t arrival_noc_base = get_noc_addr(recv_xy >> 16, recv_xy & 0xFFFFu, arrival_addr);
        const uint32_t base = get_read_ptr(cb_feed);
        uint32_t pos = 0, unflushed = 0, h = 0;
        auto flush = [&]() {
            if (unflushed) {
                noc_async_writes_flushed();
                cb_pop_front(cb_feed, unflushed);
                unflushed = 0;
            }
        };
        for (uint32_t t = 0; t < G; ++t) {
            if (t == me || ft.side(me, t) != d) {
                continue;
            }
            const uint32_t u_lo = ft.u_lo(t), u_hi = ft.u_hi(t);
            flush();  // (the headers get this target's route)
            for (uint32_t q = 0; q < num_headers; ++q) {
                (void)fabric_set_unicast_route(
                    hdrs[q], static_cast<uint16_t>(ft.chip_of(t)), static_cast<uint16_t>(ft.mesh));
            }
            uint32_t left = mhc_feed::count_tiles(t, u_lo, u_hi, l, links, rt, bw, ct, cpt);
            if (left == 0) {
                volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[0];
                hdr->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{arrival_noc_base, 1, true});
                conn.wait_for_empty_write_slot();
                conn.send_payload_flush_blocking_from_address(
                    reinterpret_cast<uint32_t>(hdr), sizeof(PACKET_HEADER_TYPE));
                continue;
            }
            uint64_t addrs[P > 1 ? P : 2];
            uint16_t sizes[P > 1 ? P : 2];
            uint32_t in_pkt = 0, size = 0;
            mhc_feed::for_each_tile(t, u_lo, u_hi, l, links, rt, bw, ct, cpt, [&](uint32_t r, uint32_t kk) {
                if (in_pkt == 0) {
                    size = mhc_feed::packet_size(left, P);
                }
                addrs[in_pkt] = f_acc.get_noc_addr((me * rt + r) * cpt + (kk - t * cpt), 0, 0);
                sizes[in_pkt] = static_cast<uint16_t>(page);
                --left;
                if (++in_pkt < size) {
                    return;
                }
                cb_wait_front(cb_feed, unflushed + 1);
                const uint32_t src = base + (pos % slots) * (P * page);
                ++pos;
                volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[h];
                h = (h + 1 == num_headers) ? 0 : h + 1;
                if (left == 0) {  // the stream's last tile (alone): + the arrival increment
                    hdr->to_noc_fused_unicast_write_atomic_inc(
                        NocUnicastAtomicIncFusedCommandHeader{addrs[0], arrival_noc_base, 1, true}, page);
                } else if (size == 1) {
                    hdr->to_noc_unicast_write(NocUnicastCommandHeader{addrs[0]}, page);
                } else {
                    hdr->to_noc_unicast_scatter_write(
                        NocUnicastScatterCommandHeader(addrs, sizes, static_cast<uint8_t>(size)), size * page);
                }
                conn.wait_for_empty_write_slot();
                conn.send_current_slot_non_blocking(src, size * page, reinterpret_cast<uint32_t>(hdr));
                in_pkt = 0;
                if (++unflushed == num_headers) {
                    flush();
                }
            });
        }
        flush();
        conn.close();
    }
    // 3. port 0: everything the group's other chips send me
    wait_and_rearm(arrival_addr, recv_expect);
}
