// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The shared expert's matmul cores' BRISC kernel.
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/fused_head/device/kernels/se_defs.hpp"

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "api/debug/kernel_profiler.hpp"
using namespace dataflow_kernel_lib;
constexpr uint32_t GX = get_compile_time_arg_val(0), MT = get_compile_time_arg_val(1), KB = get_compile_time_arg_val(2);
constexpr uint32_t NSG = get_compile_time_arg_val(3), NPG = get_compile_time_arg_val(4),
                   NCH = get_compile_time_arg_val(5);
constexpr uint32_t POMAX = get_compile_time_arg_val(6), S0 = get_compile_time_arg_val(7),
                   S1 = get_compile_time_arg_val(8);
constexpr uint32_t XREADY = get_compile_time_arg_val(9);
constexpr auto x_args = TensorAccessorArgs<10>();
constexpr auto brx_args = TensorAccessorArgs<SE_RTR_BCT0>();  // (the router's x: the chip's x shard [rows, H / TP])
constexpr uint32_t CB_XIN = 0, CB_HOWN = 2, CB_HALL = 3;
constexpr uint32_t NF = NSG / 2, NOWN0 = NF / S0, NOWN1 = NF / S1;  // (steps a family a pass; per owner)
// x position j (the j-th K step this chip processes: its own shard's first, then each shard as it lands): family j % 2,
// owner = the family's subset[(j / 2) % S] (S divides a shard's steps / 2, so the rotation over (shard, pass, step)
// rounds keeps every position on one owner), its local index (j / 2) / S
FORCE_INLINE uint32_t own_slot(uint32_t j) { return (j & 1) ? (j / 2) / S1 : (j / 2) / S0; }
// (the args equal on every core of the chip go once, as common args [S, NC, those NC values]: arg i is the core's
// own arg i below S, common below S + NC, the core's arg i - NC past them - every index stays the full list's)
FORCE_INLINE uint32_t se_ra(uint32_t i) {
    const uint32_t s = get_common_arg_val<uint32_t>(0), nc = get_common_arg_val<uint32_t>(1);
    return i < s        ? get_arg_val<uint32_t>(i)
           : i < s + nc ? get_common_arg_val<uint32_t>(2 + i - s)
                        : get_arg_val<uint32_t>(i - nc);
}
void kernel_main() {
    size_t a = 0;
    // (mine first - then from x_addr to y's address the chip's: se_ra's common span)
    const uint32_t cx = se_ra(a++), cy = se_ra(a++);
    const uint32_t po = se_ra(a++), h0 = se_ra(a++);
    const uint32_t i0 = se_ra(a++), i1 = se_ra(a++);  // my index in family 0 / 1 (or GX)
    const uint32_t ry = se_ra(a++);
    const uint32_t x_addr = se_ra(a++), x_ct = se_ra(a++), x_k0 = se_ra(a++);
    const uint32_t xown = se_ra(a++), xin = se_ra(a++), hown = se_ra(a++);
    const uint32_t hall = se_ra(a++), lx = se_ra(a++), rx = se_ra(a++);
    const uint32_t rtr_x = se_ra(a++), rtr_xa = se_ra(a++);  // (the router's x; RX)
    const uint32_t ft0 = a;  // (the x families: family 0 / 1 columns [S0] [S1], per column (nown, family-1 base slot))
    a += S0 + S1 + 2 * GX;
    const uint32_t vx0 = a, sps = se_ra(vx0 + GX), g = se_ra(vx0 + GX + 1);
    const uint32_t ord0 = vx0 + GX + 2;  // (the row's NoC x; steps a shard, shards; the shard order: own, then arrival)
    auto step_of = [&](uint32_t j) { return se_ra(ord0 + j / sps) * sps + j % sps; };  // (position j)
    const uint32_t nown = (i0 < GX ? NOWN0 : 0) + (i1 < GX ? NOWN1 : 0);               // (my steps a chunk)
    auto mine = [&](uint32_t j) { return (j & 1) ? ((j / 2) % S1 == i1) : ((j / 2) % S0 == i0); };
    // my steps' x-own slot: [chunk][family 0 slots, then family 1][MT][KB]
#ifndef SE_XRING
#define SE_XRING NCH  // (x own: every chunk resident)
#endif
    auto own_addr = [&](uint32_t m, uint32_t j) {  // (chunk m in slot m % SE_XRING: the DRAM fallback's ring)
        const uint32_t slot = (j & 1) ? (i0 < GX ? NOWN0 : 0) + own_slot(j) : own_slot(j);
        return xown + (((m % SE_XRING) * nown + slot) * MT) * KB * 2048;
    };
    auto rtrx = [&]() {  // (called after my own-shard x load: the gather's ports wait on every matmul core's - a row
        // lead's x load behind its ~20 us of router x held the gather back that long: GLM-5.3 ring 640 235.8 -> 219.2
        // us)
        if (cx ==
            0) {  // the router's x: the row's first core reads the row's x (MT row tiles, the chip's K slice) a K chunk
            // a round into RX and multicasts it along the row on NOC0 (the NCRISC's W_r^T goes down the columns on
            // NOC1), then sets the round on the row's RX-ready word (from my own source word RXS: the NCRISC re-arms
            // the ready word)
            DeviceZoneScopedN("SE-RTRX");
            const auto rxg = TensorAccessor(brx_args, rtr_x, 2048);
            constexpr uint32_t XS =
                SE_KC * SE_MTR * 2048;  // (SE_MTR: my MT rows of every chunk - row rr of chunk rr / MT)
            volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SE_EW + RXS_ID);
            constexpr uint32_t NRD = SE_KTO / SE_KC;
            volatile tt_l1_ptr uint32_t* xack = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SE_EW + XACK_ID);
#ifdef SE_RTRX_L1
            {  // (the row's x is the own shard the row's cores just loaded into x own: read from their L1, not the 2 MB
               // again from DRAM beside the own-x loads and W_r^T - once every row core counted its load on my XRL
               // word)
                volatile tt_l1_ptr uint32_t* xrl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SE_EW + XRL_ID);
                while (*xrl < GX) invalidate_l1_cache();
                *xrl = 0;  // (re-armed: one count a row core a call, all in)
            }
#endif
            for (uint32_t rd = 0; rd < NRD; ++rd) {
                const uint32_t xa = rtr_xa + (rd & 1) * XS;  // (two buffers; past two rounds the row's acks first)
                if (NRD > 2 && rd >= 2)
                    while (*xack < (rd - 1) * GX) invalidate_l1_cache();
                for (uint32_t r = 0; r < SE_MTR; ++r)
                    for (uint32_t k = 0; k < SE_KC; ++k) {
#ifdef SE_RTRX_L1
                        // (K tile kt of the chip's slice: own-shard position kt / KB - its owner column and x-own slot
                        // as x_gather3's dst_of; row r: chunk r / MT, local row r % MT)
                        const uint32_t kt = rd * SE_KC + k, j = kt / KB, hh = j / 2;
                        const uint32_t col = (j & 1) ? se_ra(ft0 + S0 + hh % S1) : se_ra(ft0 + hh % S0);
                        const uint32_t nw = se_ra(ft0 + S0 + S1 + 2 * col), b1 = se_ra(ft0 + S0 + S1 + 2 * col + 1);
                        const uint32_t slot = (j & 1) ? b1 + hh / S1 : hh / S0;
                        const uint32_t xa_ = xown + ((((r / MT) * nw + slot) * MT + r % MT) * KB + kt % KB) * 2048;
                        noc_async_read(get_noc_addr(se_ra(vx0 + col), ry, xa_), xa + (k * SE_MTR + r) * 2048, 2048);
#else
                        noc_async_read(
                            rxg.get_noc_addr(GROW(r / MT, cy * MT + r % MT) * SE_KTO + rd * SE_KC + k),
                            xa + (k * SE_MTR + r) * 2048,
                            2048);
#endif
                    }
                noc_async_read_barrier();
                noc_async_write_multicast_loopback_src(xa, get_noc_multicast_addr(lx, ry, rx, ry, xa), XS, GX);
                noc_async_writes_flushed();
                *src = rd + 1;
                noc_semaphore_set_multicast_loopback_src(
                    SE_EW + RXS_ID, get_noc_multicast_addr(lx, ry, rx, ry, SE_EW + RXR_ID), GX);
                noc_async_writes_flushed();
            }
            if (NRD > 2) {  // (re-arm: every ack landed - the last were waited for)
                noc_semaphore_inc(get_noc_addr(SE_EW + XACK_ID), 0u - (NRD - 2) * GX);
                noc_async_atomic_barrier();
            }
        }
    };
    const auto xg = TensorAccessor(x_args, x_addr, 2048);
    const uint32_t xports = se_ra(ord0 + g);         // (the gather's ports read my own shard from here)
    auto xload_step = [&](uint32_t m, uint32_t j) {  // (one owned step of chunk m from DRAM x [rows, x_ct])
        const uint32_t s = step_of(j);
        if (!(mine(j) && s * KB >= x_k0 && s * KB < x_k0 + x_ct)) {
            return;
        }
        for (uint32_t r = 0; r < MT; ++r) {
            for (uint32_t kk = 0; kk < KB; ++kk) {
                noc_async_read(
                    xg.get_noc_addr(GROW(m, cy * MT + r) * x_ct + s * KB + kk - x_k0),
                    own_addr(m, j) + (r * KB + kk) * 2048,
                    2048);
            }
        }
    };
    if (x_ct) {  // (alone) my steps of every chunk from DRAM x [rows, x_ct]: columns x_k0 ..
        DeviceZoneScopedN("SE-XLOAD");
        for (uint32_t m = 0; m < SE_XRING; ++m) {  // (the ring's first chunks; the rest as their slots free)
            for (uint32_t j = 0; j < NSG; ++j) {
                xload_step(m, j);
            }
        }
        noc_async_read_barrier();
    }
    {  // (my own-shard x is loaded: count it on the gather's ports - they send it from here - and let my W load start)
        const uint32_t np0 = ord0 + g, nports = se_ra(np0), xl_addr = se_ra(np0 + 1);
        for (uint32_t i = 0; i < nports; ++i) {
            noc_semaphore_inc(get_noc_addr(se_ra(np0 + 2 + 2 * i), se_ra(np0 + 3 + 2 * i), xl_addr), 1);
        }
#ifdef SE_RTRX_L1
        noc_semaphore_inc(get_noc_addr(se_ra(vx0), ry, SE_EW + XRL_ID), 1);  // (and on my row's lead: the router's x)
#endif
        noc_async_atomic_barrier();
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(se_ra(np0 + 2 + 2 * nports))) = 1;
    }
    rtrx();  // (the router's x after my own x)
    constexpr auto mx0 =
        McastArgs<get_named_compile_time_arg_val("x0_ct_offset"), get_named_compile_time_arg_val("x0_rt_offset")>();
    constexpr auto mx1 =
        McastArgs<get_named_compile_time_arg_val("x1_ct_offset"), get_named_compile_time_arg_val("x1_rt_offset")>();
    constexpr auto mh =
        McastArgs<get_named_compile_time_arg_val("h_ct_offset"), get_named_compile_time_arg_val("h_rt_offset")>();
    Noc noc;
    auto sh = mh.optional_sender(noc);
    auto rh = mh.optional_receiver(noc);
    auto s0 = mx0.optional_sender(noc);
    auto r0 = mx0.optional_receiver(noc);
    auto s1 = mx1.optional_sender(noc);
    auto r1 = mx1.optional_receiver(noc);
    volatile tt_l1_ptr uint32_t* xready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(XREADY));
    for (uint32_t m = 0; m < NCH; ++m) {
        if (SE_XRING < NCH && m >= SE_XRING && x_ct) {  // (the ring: chunk m's x own, issued at chunk m - 1's end)
            DeviceZoneScopedN("SE-XRELOAD");
            noc_async_read_barrier();
        }
        {
            DeviceZoneScopedN("SE-XROUNDS");
            // shard-major: each shard (own first, then as it lands) through every pass before the next shard
            for (uint32_t k = 0; k < g; ++k) {
                for (uint32_t pg = 0; pg < NPG; ++pg) {
                    for (uint32_t st = 0; st < sps; ++st) {
                        const uint32_t j =
                            k * sps + st;  // (position; its family j % 2 = st % 2: a block's rounds are even)
                        const uint32_t round = ((m * g + k) * NPG + pg) * (sps / 2) + st / 2;  // (the family's)
                        cb_reserve_back(CB_XIN, MT * KB);
                        const uint32_t dst = xin + (j & 1) * MT * KB * 2048;
                        if (mine(j)) {  // (position j: the shard of rank j / sps must be in - the own shard always is)
                            if (j >= sps) {
                                noc_semaphore_wait_min(xready, m * (g - 1) + j / sps);
                            }
                            constexpr uint32_t XBYTES = MT * KB * 2048;
                            if (j & 1) {
                                s1->send(own_addr(m, j), dst, XBYTES);
                            } else {
                                s0->send(own_addr(m, j), dst, XBYTES);
                            }
                        } else {
                            if (j & 1) {
                                r1->receive(round);
                            } else {
                                r0->receive(round);
                            }
                        }
                        cb_push_back(CB_XIN, MT * KB);
                    }
                }
            }
        }
#ifdef SE_LAT
        {  // (the latent's all-to-all: once it is packed, +1 on each destination's port word of mine)
            DeviceZoneScopedN("SE-LHAND");
            const uint32_t np0 = ord0 + g, nports = se_ra(np0), r0 = np0 + 5 + 2 * nports;
            const uint32_t l0 = r0 + 2 + 2 * se_ra(r0), nl = se_ra(l0);
            const uint32_t lw = se_ra(l0 + 1);
            cb_wait_front(SE_LAT, (m + 1) * SE_LATP);  // (chunk m's block; the words count up over the chunks)
            for (uint32_t j = 0; j < nl; ++j) {
                noc_semaphore_inc(get_noc_addr(se_ra(l0 + 2 + 2 * j), se_ra(l0 + 3 + 2 * j), lw), 1);
            }
        }
#endif
        {  // (its own zone: the hand-offs below are not h exchange)
            DeviceZoneScopedN("SE-HX");
            // h: every core's h block [MT][POMAX] into every row core's h-all slot (m % 2) as block cx
            // ([GX][MT][POMAX]): one kernel_lib rotating round a column; the receivers' readiness ack is the slot's
            // credit
            constexpr uint32_t HA = GX * MT * POMAX;
            cb_wait_front(CB_HOWN, MT * POMAX);
            cb_reserve_back(CB_HALL, HA);
            const uint32_t hs = hall + (SE_HSL > 1 ? (m & 1) : 0) * HA * 2048;  // (h-all slots: SE_HSL)
            for (uint32_t c = 0; c < GX; ++c) {
                if (c == cx) {
                    sh->send(hown, hs + c * MT * POMAX * 2048, MT * POMAX * 2048);
                } else {
                    rh->receive(m * GX + c);
                }
                cb_push_back(CB_HALL, MT * POMAX);  // (column c's block: the down's first pass starts on its K tiles)
            }
            cb_pop_front(CB_HOWN, MT * POMAX);
        }
        {  // (down + reduce-scatter) each finished sub-block's hand-off: +1 on its port's mailbox word of mine (the
           // words count up over the chunks: the port's targets do)
            DeviceZoneScopedN("SE-HAND");
            const uint32_t np0 = ord0 + g, nports = se_ra(np0), r0 = np0 + 5 + 2 * nports;
            const uint32_t nh = se_ra(r0), mbox = se_ra(r0 + 1);
            for (uint32_t j = 0; j < nh; ++j) {
                cb_wait_front(SE_FINC, j + 1);
                noc_semaphore_inc(get_noc_addr(se_ra(r0 + 2 + 2 * j), se_ra(r0 + 3 + 2 * j), mbox), 1);
            }
            if constexpr (NCH > 1) {
                cb_pop_front(SE_FINC, nh);  // (the next chunk's hand-offs from the front again)
            }
        }
        if (SE_XRING < NCH && m + SE_XRING < NCH && x_ct) {
            // (the ring: chunk m + SE_XRING into chunk m's slot - its rounds sent, its prep done (my hand-offs above
            // waited for the chunk's fins, after its prep op): the load lands under the chunks between)
#ifdef SE_XFREE
            cb_wait_front(SE_XFREE, 1);
            cb_pop_front(SE_XFREE, 1);
#endif
            noc_async_writes_flushed();
            for (uint32_t j = 0; j < NSG; ++j) {
                xload_step(m + SE_XRING, j);
            }
        }
    }
#ifdef SE_XFREE
    {  // (the last chunks' credits, unused: popped - a call ends with the CB empty, the next one's credits its own)
        constexpr uint32_t left = SE_XRING < NCH ? SE_XRING : NCH;
        cb_wait_front(SE_XFREE, left);
        cb_pop_front(SE_XFREE, left);
    }
#endif
    noc_async_atomic_barrier();
}
