// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The router cores' NCRISC kernel (router_rows.hpp; under RA_PLAN also the routing planner, ra_plan.hpp).
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/fused_head/device/kernels/router_addr.hpp"
#ifdef RA_PLAN
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/fused_head/device/kernels/ra_plan.hpp"
#endif

#include <cstdint>
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/fused_head/device/kernels/router_rows.hpp"
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
using namespace dataflow_kernel_lib;
constexpr uint32_t KT = get_compile_time_arg_val(0), ET = get_compile_time_arg_val(1), G = get_compile_time_arg_val(2);
constexpr uint32_t KBR = get_compile_time_arg_val(3);
constexpr auto x_args = TensorAccessorArgs<5>();
constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
constexpr auto b_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
constexpr auto i_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
constexpr uint32_t XR = 0, WR = 1, ARRF = 3, ARRB = 4, BIAS = 5, IDX = 11, K = get_compile_time_arg_val(4), RDY = 12;
// args: x, W_r^T, bias^T, expert index tiles; my rows [G]; the arrivals (na: every other chip's partial of my owner
// rows, slot = its source index) and their words (a word a slot, 16 B apart); the slots in their fixed sum order [G -
// 1]
void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0), w_addr = get_arg_val<uint32_t>(1),
                   b_addr = get_arg_val<uint32_t>(2);
    const uint32_t i_addr = get_arg_val<uint32_t>(3);
    const uint32_t r0 = 4, na = get_arg_val<uint32_t>(r0 + G);
    volatile tt_l1_ptr uint32_t* ctr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(r0 + G + 1));
    const uint32_t ab0 = r0 + G, nab = G - 1;  // (the order: ab0 + 2 ..; the tail: ab0 + nab + 2 ..)
    {  // (the top-K's done words: zero before anyone can write them - they are written after the scores; with the
       // split word 128 too, the partner's best K landed - the arena is call scratch: stale, it let the merge take a
       // stale page, the picks a call behind)
        const uint32_t w0 = get_arg_val<uint32_t>(ab0 + nab + 2);
        for (uint32_t i = 0; i < 9; ++i) {
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w0 + 16 * i) = 0;
        }
    }
    const auto xg = TensorAccessor(x_args, x_addr, 2048);
    const auto wg = TensorAccessor(w_args, w_addr, 2048);
    const auto bg = TensorAccessor(b_args, b_addr, 4096);
    const auto ig = TensorAccessor(i_args, i_addr, 2048);
    cb_reserve_back(BIAS, RT_ETF);
    cb_reserve_back(IDX, ET);
    for (uint32_t t = 0; t < RT_ETF; ++t) {
        noc_async_read(bg.get_noc_addr(t), get_write_ptr(BIAS) + t * 4096, 4096);
    }
    for (uint32_t t = 0; t < ET; ++t) {
        noc_async_read(ig.get_noc_addr(RT_HB + t), get_write_ptr(IDX) + t * 2048, 2048);
    }
    noc_async_read_barrier();
    cb_push_back(BIAS, RT_ETF);
    cb_push_back(IDX, ET);
#ifdef RT_ENG
    {  // the engine computed my partial logits (every engine core its expert tiles x its rows, written into my HOLD
       // with an increment of this counter a tile): all G x ET in -> the compute (RDY)
        DeviceZoneScopedN("SE-RENGWAIT");
        volatile tt_l1_ptr uint32_t* ec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(RT_ENG);
        noc_semaphore_wait_min(ec, ET);  // (my own block's tiles: the others' go straight to the ports)
        noc_semaphore_inc(get_noc_addr(RT_ENG), 0u - ET);  // (re-arm)
        noc_async_atomic_barrier();
        cb_reserve_back(RDY, 1);
        cb_push_back(RDY, 1);
        // (and word 112 for the compute's PACK thread: the RDY wait holds only UNPACK, and PACK's hand-off pushes
        // (FINC) would hand HOLD to the ports before the engine's partials landed - stale HOLD: the last call's
        // planner overlay in it, K3's picks 99.10%)
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(ab0 + nab + 2) + 112) = 1;
    }
#else
    // W_r^T is read from DRAM once: K block kb by the column's rotating sender kb % RB, multicast down the router
    // column into every core's WR slot (the kernel_lib family's handshake: a core is ready once its slot is reserved)
#ifndef RT_RW_SOLO  // (one router core: it reads every K block itself - no family)
    constexpr auto mw =
        McastArgs<get_named_compile_time_arg_val("rw_ct_offset"), get_named_compile_time_arg_val("rw_rt_offset")>();
    Noc noc;
    auto sw = mw.optional_sender(noc);
    auto rw = mw.optional_receiver(noc);
#endif
    for (uint32_t kb = 0; kb < KT / KBR;
         ++kb) {  // K block kb: x [KBR][G] (k-major: the matmul's in1) and W_r^T [ET][KBR]
        cb_reserve_back(XR, G * KBR);
        cb_reserve_back(WR, KBR * ET);
        const uint32_t xl = get_write_ptr(XR), l1 = get_write_ptr(WR);
        for (uint32_t e = 0; e < G; ++e) {
            const uint32_t R = get_arg_val<uint32_t>(r0 + e);
            for (uint32_t k = 0; k < KBR; ++k) {
                noc_async_read(xg.get_noc_addr(R * KT + kb * KBR + k), xl + (k * G + e) * 2048, 2048);
            }
        }
#ifdef RT_RW_SOLO
        for (uint32_t e = 0; e < ET; ++e) {
            for (uint32_t k = 0; k < KBR; ++k) {
                noc_async_read(wg.get_noc_addr(e * KT + kb * KBR + k), l1 + (e * KBR + k) * 2048, 2048);
            }
        }
#else
        if (mw.should_send(kb)) {
            for (uint32_t e = 0; e < ET; ++e) {
                for (uint32_t k = 0; k < KBR; ++k) {
                    noc_async_read(wg.get_noc_addr(e * KT + kb * KBR + k), l1 + (e * KBR + k) * 2048, 2048);
                }
            }
            noc_async_read_barrier();
            sw->send(l1, l1, KBR * ET * 2048);
        } else {
            rw->receive(kb);
        }
#endif
        noc_async_read_barrier();
        cb_push_back(XR, G * KBR);
        cb_push_back(WR, KBR * ET);
    }
#endif
    if (na) {  // the sources' partials as they land (a fused increment of the source slot's own word, 16 B apart): each
               // slot to the compute in a fixed order (RT_ARR_ORDERED: the order args - the predicted landing order, so
               // the same sum every run at ~no wait), else in landing order (ARRB) - it sums the early ones while the
               // last is on its way
        DeviceZoneScopedN("SE-RLWAIT");
        for (uint32_t a = 0, seen = 0; a < na;) {
            invalidate_l1_cache();
            for (uint32_t s = 0; s < na; ++s) {
#ifdef RT_ARR_ORDERED
                if (s != get_arg_val<uint32_t>(ab0 + 2 + a)) {  // (the fixed order)
                    continue;
                }
#endif
                if ((seen >> s & 1) || ctr[4 * s] == 0) {
                    continue;
                }
                ctr[4 * s] =
                    0;  // (re-arm: one increment a call - the source's next one comes after this chip's exchange)
                seen |= 1u << s;
                ++a;
                cb_reserve_back(ARRB, 1);
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(ARRB)) = s;
                cb_push_back(ARRB, 1);
            }
        }
    }
#ifdef RT_SPLIT
#if RT_HALF == 0
    {  // (my partner's best K - the row tile's other expert half - written into my PART: to the compute's merge)
        volatile tt_l1_ptr uint32_t* pw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(RT_PWORD);
        while (*pw == 0) {
            invalidate_l1_cache();
        }
        *pw = 0;  // (re-arm: one a call)
        cb_reserve_back(RT_PART_CB, 1);
        cb_push_back(RT_PART_CB, 1);
    }
#endif
#endif
    {  // (my share of the joint rows - tokens 8 .. 15 - once the top-K is done; split half 1: of its picks' scores)
        const uint32_t t2 = ab0 + nab + 2, words = get_arg_val<uint32_t>(t2);
        volatile tt_l1_ptr uint32_t* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words + 64);
        while (*done == 0) {
            invalidate_l1_cache();
        }
        *done = 0;
        DeviceZoneScopedN("SE-RROWS");
        set_l1_data_cache<true>();
        invalidate_l1_cache();
#if RT_HALF == 0
        rows_::make<K, RT_ETF, ET>(
            get_arg_val<uint32_t>(t2 + 1),
            get_arg_val<uint32_t>(t2 + 4),
            get_arg_val<uint32_t>(t2 + 2),
            get_arg_val<uint32_t>(t2 + 3),
            8,
            16);
#else
        rows_::picked<K, RT_ETF, ET>(get_arg_val<uint32_t>(t2 + 1), get_arg_val<uint32_t>(t2 + 4), 8, 16);
#endif
        set_l1_data_cache<false>();
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words + 16) = 1;
    }
#ifdef RA_PLAN
    {  // the second planner instance on this core (the routing exchange + planner: routing_stream)
        const uint32_t rr = ab0 + nab + 2 + 5, rows2 = get_arg_val<uint32_t>(rr), B = get_arg_val<uint32_t>(rr + 1);
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows2), RA_NP_);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows2) = 0;
        if (B) {
            ra::plan(rr + 2, B);  // (no instance here when the chip has fewer experts than planner RISCs)
        }
    }
#endif
}
