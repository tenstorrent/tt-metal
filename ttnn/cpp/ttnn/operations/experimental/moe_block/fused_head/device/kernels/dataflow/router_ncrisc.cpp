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
constexpr uint32_t ET = get_compile_time_arg_val(1), G = get_compile_time_arg_val(2);
constexpr auto b_args = TensorAccessorArgs<4>();
constexpr auto i_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
constexpr uint32_t ARRF = 3, ARRB = 4, BIAS = 5, IDX = 11, K = get_compile_time_arg_val(3), RDY = 12;
// args: bias^T, expert index tiles; my rows [G]; the arrivals (na: every other chip's partial of my owner
// rows, slot = its source index) and their words (a word a slot, 16 B apart); the slots in their fixed sum order [G -
// 1]
void kernel_main() {
    const uint32_t b_addr = get_arg_val<uint32_t>(0), i_addr = get_arg_val<uint32_t>(1);
    const uint32_t r0 = 2, na = get_arg_val<uint32_t>(r0 + G);
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
    const auto bg = TensorAccessor(b_args, b_addr, 4096);
    const auto ig = TensorAccessor(i_args, i_addr, 2048);
    cb_reserve_back(BIAS, ET);
    cb_reserve_back(IDX, ET);
    for (uint32_t t = 0; t < ET; ++t) {
        noc_async_read(bg.get_noc_addr(t), get_write_ptr(BIAS) + t * 4096, 4096);
    }
    for (uint32_t t = 0; t < ET; ++t) {
        noc_async_read(ig.get_noc_addr(t), get_write_ptr(IDX) + t * 2048, 2048);
    }
    noc_async_read_barrier();
    cb_push_back(BIAS, ET);
    cb_push_back(IDX, ET);
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
    {  // (my share of the joint rows - tokens 8 .. 15 - once the top-K is done)
        const uint32_t t2 = ab0 + nab + 2, words = get_arg_val<uint32_t>(t2);
        volatile tt_l1_ptr uint32_t* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words + 64);
        while (*done == 0) {
            invalidate_l1_cache();
        }
        *done = 0;
        DeviceZoneScopedN("SE-RROWS");
        set_l1_data_cache<true>();
        invalidate_l1_cache();
        rows_::make<K, ET>(
            get_arg_val<uint32_t>(t2 + 1),
            get_arg_val<uint32_t>(t2 + 4),
            get_arg_val<uint32_t>(t2 + 2),
            get_arg_val<uint32_t>(t2 + 3),
            8,
            16);
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
