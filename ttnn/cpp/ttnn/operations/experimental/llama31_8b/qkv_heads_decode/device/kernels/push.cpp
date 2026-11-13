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
    Noc noc;
    const uint32_t n = get_arg(args::n);
    const auto plan = get_arg(args::plan);
    constexpr uint32_t ROW = get_arg(args::row_bytes);
    const auto qkv = TensorAccessor(tensor::qkv_in);
    const auto q_out = TensorAccessor(tensor::q_out);
    const auto k_out = TensorAccessor(tensor::k_out);
    const auto v_out = TensorAccessor(tensor::v_out);
    for (uint32_t e = 0; e < n; ++e) {
        const uint32_t a = plan[2 * e];
        const uint32_t b = plan[2 * e + 1];
        const uint32_t src = static_cast<uint32_t>(qkv.get_noc_addr(a & 0xFF) & 0xFFFFFFFFull) + (b & 0xFFFF);
        const uint32_t t = (a >> 8) & 0x3;
        const uint32_t page = (a >> 10) & 0x7;
        const uint32_t off = b >> 16;
        if (t == 0) {
            noc.async_write(CoreLocalMem<uint32_t>(src), q_out, ROW, {}, {.page_id = page, .offset_bytes = off});
        } else if (t == 1) {
            noc.async_write(CoreLocalMem<uint32_t>(src), k_out, ROW, {}, {.page_id = page, .offset_bytes = off});
        } else {
            noc.async_write(CoreLocalMem<uint32_t>(src), v_out, ROW, {}, {.page_id = page, .offset_bytes = off});
        }
    }
    noc.async_write_barrier();
}
