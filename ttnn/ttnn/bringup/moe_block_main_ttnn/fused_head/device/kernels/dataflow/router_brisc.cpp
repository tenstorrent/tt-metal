// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The router cores' BRISC kernel (router_rows.hpp; under RA_PLAN also the routing planner, ra_plan.hpp).
#include "../router_addr.hpp"
#ifdef RA_PLAN
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "../ra_plan.hpp"
#endif

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
constexpr uint32_t ET = get_compile_time_arg_val(1), K = get_compile_time_arg_val(3);
constexpr auto o_args = TensorAccessorArgs<4>();
constexpr uint32_t HOLD = 2, BIAS = 5, BEST = 9;
#include "../router_rows.hpp"

// args: the out address, my owner page, route scale bits, the joint rows' L1 scratch, the done word, the BEST page
void kernel_main() {
#if defined(RT_ROUTE) && defined(RT_VALID)
    // (a padded chunk's actual_start / actual_end, for the route words: into the route words' L1 (256 B) - the compute
    // writes them only after word 80, by when they are read - early, read once the rows are written, long after)
    const uint32_t vl1 = rt_valid_slot(get_arg_val<uint32_t>(6));
    rt_valid_issue(vl1);
    uint32_t sv = 0;  // (my real tokens, read below)
#endif
    constexpr uint32_t t0 = 0;
    const uint32_t o_addr = get_arg_val<uint32_t>(t0), page = get_arg_val<uint32_t>(t0 + 1);
    const uint32_t o_l1 = get_arg_val<uint32_t>(t0 + 3);
    const uint32_t words = get_arg_val<uint32_t>(t0 + 4);
    // the top K of every token column, sorted descending in rows 0 .. K - 1 of the best page;
    // tokens 0 .. 7 here, 8 .. 15 the NCRISC, 16 .. 23 the compute's MATH thread, 24 .. 31 its PACK (done words 16 / 32
    // / 48)
    const uint32_t bv = get_arg_val<uint32_t>(t0 + 5);
    const auto og = TensorAccessor(o_args, o_addr, 32 * 4 * K);
#ifdef RT_NIT
    // router iterations (RT_ITA args at t0 + 10): am I router core 0, my chip's "rit" word, router core 0's NoC, the
    // logits ports' count, the ports, the ewords, the matmul cores' NoC x's and y's
    constexpr uint32_t NIT = RT_NIT, ti = t0 + 10;
    const bool c0me = get_arg_val<uint32_t>(ti) != 0;
    const uint32_t rit = get_arg_val<uint32_t>(ti + 1), c0 = get_arg_val<uint32_t>(ti + 2);
    const uint32_t nact = get_arg_val<uint32_t>(ti + 3), npts = get_arg_val<uint32_t>(ti + 4), pt0 = ti + 5;
    const uint32_t ew = get_arg_val<uint32_t>(pt0 + npts), gx = get_arg_val<uint32_t>(pt0 + npts + 1);
    const uint32_t xs0 = pt0 + npts + 2, gy = get_arg_val<uint32_t>(xs0 + gx), ys0 = xs0 + gx + 1;
#else
    constexpr uint32_t NIT = 1;
#endif
    auto route_out = [&](uint32_t it) {  // dispatch prep: my owned tokens' route words (the compute's, word 96)
#ifdef RT_ROUTE
        volatile tt_l1_ptr uint32_t* rd = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words + 96);
        while (*rd == 0) {
            invalidate_l1_cache();
        }
        *rd = 0;  // (re-arm)
        const uint32_t rl1 = get_arg_val<uint32_t>(t0 + 6);
#ifdef RT_NIT
        const uint32_t tok0 = get_arg_val<uint32_t>(t0 + 8) + it * 32 * RT_RBI;  // (my row tile of iteration it)
#else
        const uint32_t tok0 = get_arg_val<uint32_t>(t0 + 8);
#endif
#ifdef RT_VALID
        {  // (padding tokens - t >= sv - get no route: not dispatched)
            invalidate_l1_cache();
            volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rl1);
            for (uint32_t c = sv > tok0 ? sv - tok0 : 0; c < 32; ++c) {
                for (uint32_t i = 0; i < RT_RW; ++i) {
                    o[RT_RW * c + i] = 0;
                }
            }
        }
#endif
        const auto rg = TensorAccessor(TensorAccessorArgs<RT_CT0>(), get_arg_val<uint32_t>(t0 + 7), RT_PAGE);
        noc_async_write(rl1, rg.get_noc_addr(0) + tok0 * 4 * RT_RW, 32 * 4 * RT_RW);
        noc_async_write_barrier();
#endif
    };
    cb_wait_front(BIAS, ET);
    for (uint32_t it = 0; it < NIT; ++it) {
        volatile tt_l1_ptr uint32_t* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words);
        while (*done == 0) {
            invalidate_l1_cache();
        }
        *done = 0;  // (re-arm)
        {
            {
                DeviceZoneScopedN("SE-RROWS");
                set_l1_data_cache<true>();  // (write-through; the packed tiles read after an invalidate)
                invalidate_l1_cache();
#ifdef RT_NIT
                const uint32_t mt0 = get_arg_val<uint32_t>(t0 + 8) + it * 32 * RT_RBI;  // (my row tile of iteration it)
#else
                const uint32_t mt0 = get_arg_val<uint32_t>(t0 + 8);
#endif
                rows_::make<K, ET>(bv, get_arg_val<uint32_t>(t0 + 9), o_l1, get_arg_val<uint32_t>(t0 + 2), 0, 8, mt0);
                set_l1_data_cache<false>();
            }
            for (uint32_t i = 1; i < 4; ++i) {
                volatile tt_l1_ptr uint32_t* wd = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words + 16 * i);
                while (*wd == 0) {
                    invalidate_l1_cache();
                }
                *wd = 0;  // (re-arm)
            }
#ifdef RT_ROUTE
#ifdef RT_VALID
            if (it == 0) {
                noc_async_read_barrier();
                invalidate_l1_cache();
                sv = rt_valid_local(vl1);  // (my real tokens: the route words of the rest are zeroed below)
            }
#endif
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words + 80) = 1;  // (every share of the rows written)
#endif
        }
#ifdef RT_NIT
        noc_async_write(o_l1, og.get_noc_addr(page + it * RT_RBI), 32 * 4 * K);  // (my row tile of iteration it)
#else
        noc_async_write(o_l1, og.get_noc_addr(page), 32 * 4 * K);
#endif
        noc_async_write_barrier();
#ifdef RA_PLAN
        {  // (my rows of this iteration are written: every router core's two planner RISCs told)
            const uint32_t r0 = t0 + 10 + RT_ITA, rows = get_arg_val<uint32_t>(r0),
                           rows2 = get_arg_val<uint32_t>(r0 + 1);
            for (uint32_t i = 0; i < RA_NC_; ++i) {
                const uint32_t x = get_arg_val<uint32_t>(r0 + 3 + 2 * i), y = get_arg_val<uint32_t>(r0 + 4 + 2 * i);
                noc_semaphore_inc(get_noc_addr(x, y, rows), 1);
                noc_semaphore_inc(get_noc_addr(x, y, rows2), 1);
            }
        }
#endif
#ifdef RT_NIT
        if (it + 1 <
            NIT) {  // (not the last iteration: its route words now - the L1 is reused - then this core is free)
            route_out(it);
            noc_async_atomic_barrier();
            for (uint32_t i = 0; i < npts;
                 ++i) {  // (every port: the chips sending me logits may send iteration it + 1)
                const uint32_t p = get_arg_val<uint32_t>(pt0 + i);
                noc_semaphore_inc(get_noc_addr(p & 0xFFFF, p >> 16, rit + 16), 1);
            }
            noc_semaphore_inc(get_noc_addr(c0 & 0xFFFF, c0 >> 16, rit), 1);
            if (c0me) {  // (router core 0: every router core and logits port done with it -> every matmul core's go)
                volatile tt_l1_ptr uint32_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rit);
                while (*cnt < (RT_RBI + nact) * (it + 1)) {
                    invalidate_l1_cache();
                }
                for (uint32_t x = 0; x < gx; ++x) {
                    for (uint32_t y = 0; y < gy; ++y) {
                        noc_semaphore_inc(
                            get_noc_addr(get_arg_val<uint32_t>(xs0 + x), get_arg_val<uint32_t>(ys0 + y), ew + RT_RGO),
                            1);
                    }
                }
                if (it + 2 == NIT) {  // (re-arm: the last count came)
                    noc_semaphore_inc(get_noc_addr(rit), 0u - (RT_RBI + nact) * (NIT - 1));
                }
            }
            noc_async_atomic_barrier();
        }
#endif
    }
#ifdef RA_PLAN
    {  // the routing exchange + planner (routing_stream): every router core's rows written, then routing_a2a's planner
        DeviceZoneScopedN("SE-RPLAN");
        const uint32_t r0 = t0 + 10 + RT_ITA, rows = get_arg_val<uint32_t>(r0);
        const uint32_t B = get_arg_val<uint32_t>(r0 + 2);
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows), RA_NP_);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows) = 0;  // (re-arm: all increments landed)
        if (B) {
            ra::plan(r0 + 3 + 2 * RA_NC_, B);  // (no instance here: fewer planner instances than router cores)
        }
    }
#endif
    route_out(NIT - 1);  // (the last iteration's after the planner: off the routing chain)
}
