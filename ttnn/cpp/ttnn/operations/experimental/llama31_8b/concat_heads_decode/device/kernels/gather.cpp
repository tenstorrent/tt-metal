// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t n = get_arg(args::n);
    const auto plan = get_arg(args::plan);
    constexpr uint32_t ROW = get_arg(args::row_bytes);
    constexpr uint32_t PAIR = 2 * ROW;
    const auto in = TensorAccessor(tensor::input);
    const auto out = TensorAccessor(tensor::output);
    for (uint32_t e = 0; e < n; ++e) {
        const uint32_t a = plan[2 * e];
        const uint32_t b = plan[2 * e + 1];
        const uint32_t dst = static_cast<uint32_t>(out.get_noc_addr(a >> 8) & 0xFFFFFFFFull) + (b >> 16);
        noc_async_read(in.get_noc_addr(a & 0xFF) + ((b & 0xFFFF) & ~(PAIR - 1)), dst + PAIR, PAIR);
    }
    noc_async_read_barrier();
    for (uint32_t e = 0; e < n; ++e) {
        const uint32_t a = plan[2 * e];
        const uint32_t b = plan[2 * e + 1];
        const uint32_t dst = static_cast<uint32_t>(out.get_noc_addr(a >> 8) & 0xFFFFFFFFull) + (b >> 16);
        volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
        volatile tt_l1_ptr uint32_t* s =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst + PAIR + ((b & 0xFFFF) & (PAIR - 1)));
        for (uint32_t w = 0; w < ROW / 4; ++w) {
            d[w] = s[w];
        }
    }
}
