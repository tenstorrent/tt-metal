// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The x all-gather's sender on the port cores (xg_ct.hpp, xg_walk.hpp, xg_la_sender.hpp, xg_rs_sender.hpp,
// router_lstream.hpp, ra_send.hpp).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "api/debug/kernel_profiler.hpp"
using namespace tt::tt_fabric;
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/fused_head/device/kernels/xg_ct.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/fused_head/device/kernels/xg_walk.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/fused_head/device/kernels/xg_la_sender.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/fused_head/device/kernels/xg_rs_sender.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/fused_head/device/kernels/router_lstream.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/fused_head/device/kernels/ra_send.hpp"

void kernel_main() {
    size_t a = 0;
    const uint32_t xown = get_arg_val<uint32_t>(a++), v = get_arg_val<uint32_t>(a++);
    const uint32_t arrival = get_arg_val<uint32_t>(a++), expect_in = get_arg_val<uint32_t>(a++);  // (my counter)
    const uint32_t ready = get_arg_val<uint32_t>(a++), wait_ready = get_arg_val<uint32_t>(a++);
    const uint32_t send_ready = get_arg_val<uint32_t>(a++), rx = get_arg_val<uint32_t>(a++),
                   ry = get_arg_val<uint32_t>(a++);
    const uint32_t px = get_arg_val<uint32_t>(a++), py = get_arg_val<uint32_t>(a++);      // downstream port (NoC)
    const uint16_t mesh = get_arg_val<uint32_t>(a++), chip = get_arg_val<uint32_t>(a++);  // the neighbour
    const uint32_t wx = get_arg_val<uint32_t>(a++), wy = get_arg_val<uint32_t>(a++), watch = get_arg_val<uint32_t>(a++);
    const uint32_t n = get_arg_val<uint32_t>(a++);
    const uint32_t ent = a, T = ent + 3 * n;  // entries (q, part, packets a chunk); tables
    a = T + G + S0 + S1 + 3 * GX + GY;        // (past the tables: the RS units, then the fabric connection)
    const uint32_t rs0 = a, rnu = get_arg_val<uint32_t>(rs0), rnc = get_arg_val<uint32_t>(rs0 + 1 + 2 * rnu);
    const uint32_t ls0 = rs0 + 2 + 2 * rnu + 2 * rnc + G + RSX;
    // (chunks: the zone credits - my zone-free word (my chip's cores, a chunk's arrivals from my peer read), the
    // signals a chunk; forwarded to the peer's port facing me (its zone-credit word): NCH - 1 a call)
    const uint32_t rse = rs0 + 2 + 2 * rnu + 2 * rnc + G;
    volatile tt_l1_ptr uint32_t* zfree = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(rse + 7));
    const uint32_t zneed = get_arg_val<uint32_t>(rse + 8), zexp = get_arg_val<uint32_t>(rse + 9);
    const uint32_t zcred = get_arg_val<uint32_t>(rse + 6);
    const uint32_t NZF = zneed && NCH > 1 ? NCH - 1 : 0;
    // (the router's buffers over this chip's arrival zones / LATZ: every matmul core counts its router phase over on
    // my "router done" word (rneed of them) - only then the peer's chunk 0 credit and the latent's "started" go)
    volatile tt_l1_ptr uint32_t* rdone =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(rse + 10));
    const uint32_t rw_ = get_arg_val<uint32_t>(rse + 11), rneed = rw_ & 0xFFFF;
    const bool rzone = (rw_ >> 16) & 1, rlat = (rw_ >> 17) & 1;  // (what waits for it: chunk 0's credit, "started")
    bool rtold = rneed == 0;
    uint32_t zfw = 0;
    LStream ls;
    ls.init(ls0);
    const uint32_t la0 = ls0 + LStream::len(ls0), lnu = get_arg_val<uint32_t>(la0);
    // (the latent's end-to-end fence: my fence word, the "started" count my sends need (one a destination chip), then
    // the chips I tell "started" - their latent ports toward me: (chip | mesh << 16, x | y << 8))
    const uint32_t lf0 = la0 + 2 + 2 * lnu + 2 * get_arg_val<uint32_t>(la0 + 1 + 2 * lnu) + 5;
    const uint32_t lfence = get_arg_val<uint32_t>(lf0), lneed = get_arg_val<uint32_t>(lf0 + 1),
                   lns = get_arg_val<uint32_t>(lf0 + 2);
    // (the logits' end-to-end fence: my fence word, the "started" count my logits need (one an owner chip I send to),
    // then the chips whose logits come to me through this port's direction and link: route, their port)
    const uint32_t lg0 = lf0 + 3 + 2 * lns;
    const uint32_t lgfence = get_arg_val<uint32_t>(lg0), lgneed = get_arg_val<uint32_t>(lg0 + 1),
                   lgns = get_arg_val<uint32_t>(lg0 + 2);
    volatile tt_l1_ptr uint32_t* lgf = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lgfence);
    const uint32_t ra0 = lg0 + 3 + 2 * lgns;
    RASend ra;
    ra.init(ra0);
    const uint32_t rf0 = ra0 + RASend::len(ra0);
    RAFwd rf;
    rf.init(rf0);
    a = rf0 + RAFwd::len(rf0);
    if (n > 0 || send_ready || rnu > 0 || ls.nu > 0 || lnu > 0 || lns > 0 || lgns > 0 || !ra.done || rf.left || NZF ||
        (rzone && zneed)) {
        auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[GROUP + 1 + 3];
        for (uint32_t h = 0; h <= GROUP; ++h) {
            hdrs[h] = PacketHeaderPool::allocate_header();
            (void)fabric_set_unicast_route(hdrs[h], chip, mesh);
        }
        volatile tt_l1_ptr PACKET_HEADER_TYPE* rahdr =
            PacketHeaderPool::allocate_header();  // (routing: routed a packet)
        volatile tt_l1_ptr PACKET_HEADER_TYPE* lshdr = PacketHeaderPool::allocate_header();  // (logits: routed a unit)
        for (uint32_t un = 0; un < lnu && un < 3; ++un) {  // (the latent's destinations: routed a unit)
            const uint32_t w0 = get_arg_val<uint32_t>(la0 + 1 + 2 * un);
            hdrs[GROUP + 1 + un] = PacketHeaderPool::allocate_header();
            (void)fabric_set_unicast_route(
                hdrs[GROUP + 1 + un],
                static_cast<uint16_t>(w0 >> 16),
                static_cast<uint16_t>(get_arg_val<uint32_t>(la0 + 2 + 2 * un)));
        }
        conn.open();
        if (send_ready) {  // fence: I started this call - the neighbour's port facing me may write into my L1
            hdrs[GROUP]->to_noc_unicast_atomic_inc(
                NocUnicastAtomicIncCommandHeader{get_noc_addr(rx, ry, ready), 1, true});
            conn.wait_for_empty_write_slot();
            conn.send_payload_flush_blocking_from_address(
                reinterpret_cast<uint32_t>(hdrs[GROUP]), sizeof(PACKET_HEADER_TYPE));
        }
        volatile tt_l1_ptr PACKET_HEADER_TYPE* fh = lns > 0 ? PacketHeaderPool::allocate_header() : nullptr;
        auto lstarted = [&]() {  // (every chip sending me latent may write my LATZ - its latent port's fence word)
            for (uint32_t i = 0; i < lns; ++i) {
                const uint32_t w = get_arg_val<uint32_t>(lf0 + 3 + 2 * i), xy = get_arg_val<uint32_t>(lf0 + 4 + 2 * i);
                (void)fabric_set_unicast_route(fh, static_cast<uint16_t>(w & 0xFFFF), static_cast<uint16_t>(w >> 16));
                fh->to_noc_unicast_atomic_inc(
                    NocUnicastAtomicIncCommandHeader{get_noc_addr(xy & 0xFF, (xy >> 8) & 0xFF, lfence), 1, true});
                conn.wait_for_empty_write_slot();
                conn.send_payload_flush_blocking_from_address(
                    reinterpret_cast<uint32_t>(fh), sizeof(PACKET_HEADER_TYPE));
            }
        };
        if (!rlat) {
            lstarted();  // (I started this call: LATZ is free - no router buffers over it)
        }
        if (lgns > 0) {  // (I started this call: every chip whose logits come to me through this port may write my
            // router cores' arrival zone - its logits port's fence word; my neighbour's ready only fences the first
            // hop)
            volatile tt_l1_ptr PACKET_HEADER_TYPE* gh = lshdr;  // (free until the first logits packet: re-routed each)
            for (uint32_t i = 0; i < lgns; ++i) {
                const uint32_t w = get_arg_val<uint32_t>(lg0 + 3 + 2 * i), xy = get_arg_val<uint32_t>(lg0 + 4 + 2 * i);
                (void)fabric_set_unicast_route(gh, static_cast<uint16_t>(w & 0xFFFF), static_cast<uint16_t>(w >> 16));
                gh->to_noc_unicast_atomic_inc(
                    NocUnicastAtomicIncCommandHeader{get_noc_addr(xy & 0xFF, (xy >> 8) & 0xFF, lgfence), 1, true});
                conn.wait_for_empty_write_slot();
                conn.send_payload_flush_blocking_from_address(
                    reinterpret_cast<uint32_t>(gh), sizeof(PACKET_HEADER_TYPE));
            }
        }
        bool lgok = lgneed == 0;  // (every owner I send logits to started this call)
        if (wait_ready) {
            DeviceZoneScopedN("SE-RDYWAIT");  // (its end: the neighbour's port started - launch skew + a hop)
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready), 1);
            noc_semaphore_inc(get_noc_addr(ready), 0u - 1u);
            noc_async_atomic_barrier();
        }
        const uint64_t arr_noc = get_noc_addr(px, py, arrival);
        uint32_t h = 0, unflushed = 0;
        auto rpoll = [&]() {  // (every matmul core's router phase is over: the zones, LATZ free - also between the x
            // gather's packets: the upstream's chunk 0 reduce-scatter waits for it)
            if (!rtold && *rdone >= rneed) {
                if (rzone && zneed) {  // (chunk 0's credit)
                    hdrs[GROUP]->to_noc_unicast_atomic_inc(
                        NocUnicastAtomicIncCommandHeader{get_noc_addr(rx, ry, zcred), 1, true});
                    conn.wait_for_empty_write_slot();
                    conn.send_payload_flush_blocking_from_address(
                        reinterpret_cast<uint32_t>(hdrs[GROUP]), sizeof(PACKET_HEADER_TYPE));
                }
                if (rlat) {
                    lstarted();
                }
                rtold = true;
            }
        };
        auto zpoll = [&]() {  // (my chip's cores read a chunk's zones: credit the peer's sends, a blocking header -
            // the gather's fence / watcher header, never in flight across this)
            rpoll();
            if (rtold && zfw < NZF && *zfree >= (zfw + 1) * zneed) {
                hdrs[GROUP]->to_noc_unicast_atomic_inc(
                    NocUnicastAtomicIncCommandHeader{get_noc_addr(rx, ry, zcred), 1, true});
                conn.wait_for_empty_write_slot();
                conn.send_payload_flush_blocking_from_address(
                    reinterpret_cast<uint32_t>(hdrs[GROUP]), sizeof(PACKET_HEADER_TYPE));
                ++zfw;
            }
        };
        auto lpoll = [&]() {
            zpoll();
            if (!lgok) {
                invalidate_l1_cache();
                lgok = *lgf >= lgneed;
            }
            if (lgok) {
                ls.poll(conn, lshdr, false);
            }
            ra.poll(conn, rahdr, false, RA_N_, RA_M_, RA_EPC_, RA_ME_, RA_HBP_, RA_BB_, RA_BBP_, RA_MAXP_);
            rf.poll(conn, rahdr, false, RA_N_, RA_M_, RA_HBP_, RA_MAXP_);
        };
        for (uint32_t m = 0; m < NCH && n > 0; ++m) {
            for (uint32_t k = 0; k < n; ++k) {
                const uint32_t q = get_arg_val<uint32_t>(ent + 3 * k), part = get_arg_val<uint32_t>(ent + 3 * k + 1);
                const uint32_t npk = get_arg_val<uint32_t>(ent + 3 * k + 2);
                uint32_t sent = 0;
                walk(m, q, part, v, [&](uint32_t R, uint32_t j, uint32_t, uint32_t kk0, uint32_t np, uint32_t) {
                    if (NCH > 1 && !rtold) {  // (chunks: the upstream's chunk 0 reduce-scatter comes before
                        // my gather of the last chunk's x ends; one chunk: after it - no blocking header between
                        // the x packets, it cost K3 640 ~3 us of gather)
                        invalidate_l1_cache();
                        rpoll();
                    }
#if LS_FIRST
                    lpoll();  // (GLM-type: the logits ahead of the x gather)
                    while (!cb_pages_available_at_front(0, RP * (unflushed + 1))) lpoll();
#endif
                    cb_wait_front(0, RP * (unflushed + 1));
                    const uint32_t src = get_read_ptr(0) + unflushed * RP * 2048;
                    const Dst d = dst_of(T, xown, m, R, j, kk0);
                    const uint64_t dst = get_noc_addr(d.x, d.y, d.addr);
                    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[h];
                    h = h + 1 == GROUP ? 0 : h + 1;
                    if ((++sent % INC) == 0 || sent == npk) {
                        hdr->to_noc_fused_unicast_write_atomic_inc(
                            NocUnicastAtomicIncFusedCommandHeader{dst, arr_noc, 1, true}, np * 2048);
                    } else {
                        hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, np * 2048);
                    }
                    conn.wait_for_empty_write_slot();
                    conn.send_current_slot_non_blocking(src, np * 2048, reinterpret_cast<uint32_t>(hdr));
                    if (++unflushed == GROUP || sent == npk) {
                        noc_async_writes_flushed();
                        cb_pop_front(0, RP * GROUP);
                        unflushed = 0;
                    }
                });
                // this chunk's shard k all sent: the neighbour's watcher (behind its packets on this link)
                hdrs[GROUP]->to_noc_unicast_atomic_inc(
                    NocUnicastAtomicIncCommandHeader{get_noc_addr(wx, wy, watch), 1, true});
                conn.wait_for_empty_write_slot();
                conn.send_payload_flush_blocking_from_address(
                    reinterpret_cast<uint32_t>(hdrs[GROUP]), sizeof(PACKET_HEADER_TYPE));
            }
        }
        lpoll();
        uint32_t rpos = 0;  // (as the reader's pos)
        for (uint32_t m = 0; m < NCH; ++m) {
            la_send(la0, conn, hdrs, lpoll, rpos, m);
            rs_send(rs0, conn, hdrs, lpoll, rpos, m);
        }
        while (zfw < NZF || !rtold) {  // (the chunks' credits still owed; the router-done credit / "started")
            invalidate_l1_cache();
            lpoll();
        }
        if (lgneed) {
            noc_semaphore_wait_min(lgf, lgneed);
        }
        ls.poll(conn, lshdr, true);  // (whatever is left of the logits)
        ra.poll(conn, rahdr, true, RA_N_, RA_M_, RA_EPC_, RA_ME_, RA_HBP_, RA_BB_, RA_BBP_, RA_MAXP_);  // (the routing)
        rf.poll(conn, rahdr, true, RA_N_, RA_M_, RA_HBP_, RA_MAXP_);  // (and what passes through)
        noc_async_writes_flushed();
        if (NZF) {
            noc_semaphore_inc(get_noc_addr(reinterpret_cast<uint32_t>(zfree)), 0u - NZF * zneed);  // (re-arm)
        }
        if (zexp) {  // (every credit of the call came: they were waited for, the last before the last chunk)
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zcred), zexp);
            noc_semaphore_inc(get_noc_addr(zcred), 0u - zexp);
        }
        ls.rearm();
        if (lneed) {
            noc_semaphore_inc(get_noc_addr(lfence), 0u - lneed);  // (re-arm: a next call's early "started" stays)
        }
        if (lgneed) {
            noc_semaphore_inc(get_noc_addr(lgfence), 0u - lgneed);  // (re-arm, as the latent's)
        }
        noc_async_atomic_barrier();
        conn.close();
    }
    if (rneed) {  // (re-arm: every matmul core counted its router phase here - also on a port with nothing to send)
        noc_semaphore_wait_min(rdone, rneed);
        noc_semaphore_inc(get_noc_addr(reinterpret_cast<uint32_t>(rdone)), 0u - rneed);
        noc_async_atomic_barrier();
    }
    if (expect_in > 0) {  // everything from upstream landed; re-arm for the next call
        volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival);
        noc_semaphore_wait_min(arrived, expect_in);
        noc_semaphore_inc(get_noc_addr(arrival), 0u - expect_in);
        noc_async_atomic_barrier();
    }
}
