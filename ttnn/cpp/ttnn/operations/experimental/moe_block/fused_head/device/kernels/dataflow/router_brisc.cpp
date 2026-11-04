// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The router cores' BRISC kernel (router_rows.hpp; under RA_PLAN also the routing planner, ra_plan.hpp).
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/fused_head/device/kernels/router_addr.hpp"
#ifdef RA_PLAN
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/fused_head/device/kernels/ra_plan.hpp"
#endif

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
constexpr uint32_t ET = get_compile_time_arg_val(1), K = get_compile_time_arg_val(4);
constexpr auto o_args = TensorAccessorArgs<5>();
constexpr uint32_t HOLD = 2, BIAS = 5, FINC = 7, BEST = 9;
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/fused_head/device/kernels/router_rows.hpp"

// args: nh, per hand-off (entry, port x, port y, port L1 offset of the sub-block, port word address); the out address,
// my owner page, route scale bits, the joint rows' L1 scratch, the done word, the BEST page
void kernel_main() {
    const uint32_t nh = get_arg_val<uint32_t>(0), h0 = 1;
    const uint32_t hold = get_read_ptr(HOLD);
    for (uint32_t j = 0; j < nh; ++j) {
        DeviceZoneScopedN("SE-RHAND");
        cb_wait_front(FINC, j + 1);
        const uint32_t e = get_arg_val<uint32_t>(h0 + 6 * j), px = get_arg_val<uint32_t>(h0 + 6 * j + 1);
        const uint32_t py = get_arg_val<uint32_t>(h0 + 6 * j + 2), sw = get_arg_val<uint32_t>(h0 + 6 * j + 5);
        if (sw) {  // (the port's sub-block slot is shared with an earlier core of my link: once the port sent its)
            volatile tt_l1_ptr uint32_t* f = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sw);
            while (*f == 0) {
                invalidate_l1_cache();
            }
            *f = 0;  // (re-arm: one a call - the next comes after this call's hand-offs)
        }
        noc_async_write(hold + e * ET * 4096, get_noc_addr(px, py, get_arg_val<uint32_t>(h0 + 6 * j + 3)), ET * 4096);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(px, py, get_arg_val<uint32_t>(h0 + 6 * j + 4)), 1);
    }
    const uint32_t t0 = h0 + 6 * nh;
    const uint32_t o_addr = get_arg_val<uint32_t>(t0), page = get_arg_val<uint32_t>(t0 + 1);
    const uint32_t o_l1 = get_arg_val<uint32_t>(t0 + 3);
    const uint32_t words = get_arg_val<uint32_t>(t0 + 4);
    // the top K of every token column, sorted descending in rows 0 .. K - 1 of the best page;
    // tokens 0 .. 7 here, 8 .. 15 the NCRISC, 16 .. 23 the compute's MATH thread, 24 .. 31 its PACK (done words 16 / 32
    // / 48)
    const uint32_t bv = get_arg_val<uint32_t>(t0 + 5);
    cb_wait_front(BIAS, ET);
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
            rows_::make<K, ET>(bv, get_arg_val<uint32_t>(t0 + 9), o_l1, get_arg_val<uint32_t>(t0 + 2), 0, 8);
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
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words + 80) = 1;  // (every share of the rows written)
#endif
    }
    const auto og = TensorAccessor(o_args, o_addr, 32 * 4 * K);
    noc_async_write(o_l1, og.get_noc_addr(page), 32 * 4 * K);
    noc_async_write_barrier();
#ifdef RA_PLAN
    {  // the routing exchange + planner (routing_stream): every router core's rows written, then routing_a2a's planner
        DeviceZoneScopedN("SE-RPLAN");
        const uint32_t r0 = t0 + 10, rows = get_arg_val<uint32_t>(r0), rows2 = get_arg_val<uint32_t>(r0 + 1);
        const uint32_t B = get_arg_val<uint32_t>(r0 + 2);
        for (uint32_t i = 0; i < RA_NC_; ++i) {  // (my rows are written: every router core's two planner RISCs told)
            const uint32_t x = get_arg_val<uint32_t>(r0 + 3 + 2 * i), y = get_arg_val<uint32_t>(r0 + 4 + 2 * i);
            noc_semaphore_inc(get_noc_addr(x, y, rows), 1);
            noc_semaphore_inc(get_noc_addr(x, y, rows2), 1);
        }
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows), RA_NP_);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows) = 0;  // (re-arm: all increments landed)
        if (B) {
            ra::plan(r0 + 3 + 2 * RA_NC_, B);  // (no instance here: fewer planner instances than router cores)
        }
    }
#endif
#ifdef RT_ROUTE
    {  // dispatch prep: my owned tokens' route words (the compute's, word 96) to the route tensor
        volatile tt_l1_ptr uint32_t* rd = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words + 96);
        while (*rd == 0) {
            invalidate_l1_cache();
        }
        *rd = 0;  // (re-arm)
        const uint32_t rl1 = get_arg_val<uint32_t>(t0 + 6), tok0 = get_arg_val<uint32_t>(t0 + 8);
        const auto rg = TensorAccessor(TensorAccessorArgs<RT_CT0>(), get_arg_val<uint32_t>(t0 + 7), RT_PAGE);
        noc_async_write(rl1, rg.get_noc_addr(0) + tok0 * 4 * RT_RW, 32 * 4 * RT_RW);
        noc_async_write_barrier();
    }
#endif
}
