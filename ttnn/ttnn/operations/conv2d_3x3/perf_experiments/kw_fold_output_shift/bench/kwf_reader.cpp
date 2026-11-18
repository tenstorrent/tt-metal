// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// kw-fold bench reader: seeds every valid ACC region (32 rows of one output tile-row) with the fp32 bias rows compute
// packed once into cb_bias_rm, one self-aimed NoC copy per region, and hands each seeded region to compute (cb_tok).
// Compute's first pack into a region accumulates onto it, so no pack ever needs to overwrite (pack order is free).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

void kernel_main() {
    constexpr uint32_t cb_acc = 4;
    constexpr uint32_t cb_bias_rm = 5;
    constexpr uint32_t cb_tok = 6;
    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t bm = get_compile_time_arg_val(2);
    constexpr uint32_t d = get_compile_time_arg_val(3);
    constexpr uint32_t tpr = get_compile_time_arg_val(4);
    constexpr uint32_t PITCH = N * 32 * 4;
    constexpr uint32_t REGION = 32 * PITCH;
    const uint32_t t0 = get_arg_val<uint32_t>(0);

    MaybeDeviceZoneScope("kwf_seed");
    cb_wait_front(cb_bias_rm, 1);
    const uint64_t src = get_noc_addr(get_read_ptr(cb_bias_rm));
    const uint32_t acc = get_write_ptr(cb_acc);
    const uint32_t img0 = (t0 - 1 + tpr) / tpr;
    for (uint32_t i = 1; i <= bm; ++i) {
        const uint32_t img = (t0 - 1 + i + tpr) / tpr;
        const uint32_t row = d + 32 * i + d * (img - img0);
        noc_async_read(src, acc + row * PITCH, REGION);
        noc_async_read_barrier();
        cb_reserve_back(cb_tok, 1);
        cb_push_back(cb_tok, 1);
    }
    cb_pop_front(cb_bias_rm, 1);
}
