// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// rotary_embedding_indexed chain mode writer. Per block (b, h, t): OUT_RM: the 32 untilized rows (W 2 bytes each)
// as row pages b H S + h S + 32 t + r of [B, H, S, W] ROW_MAJOR; else the WT output tiles at tile column
// (h % G) WT + j of tile row t of group h / G of [B, H / G, S, G W] TILE (G = 1: [B, H, S, W]).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "chain_share.hpp"

void kernel_main() {
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t OUT_RM = get_compile_time_arg_val(1);
    constexpr uint32_t H = get_compile_time_arg_val(2);
    constexpr uint32_t Ht = get_compile_time_arg_val(3);
    constexpr uint32_t WT = get_compile_time_arg_val(4);
    constexpr uint32_t G = get_compile_time_arg_val(5);
    constexpr uint32_t out_page = get_compile_time_arg_val(6);
    constexpr uint32_t GX = get_compile_time_arg_val(7);
    constexpr auto out_args = TensorAccessorArgs<8>();

    uint32_t first, count;
    chain_share(get_common_arg_val<uint32_t>(1), get_common_arg_val<uint32_t>(2), GX, first, count);
    const auto out_acc = TensorAccessor(out_args, out_addr, out_page);
    for (uint32_t k = first; k < first + count; ++k) {
        const uint32_t t = k % Ht;
        const uint32_t bh = k / Ht;
        cb_wait_front(cb_out, WT);
        uint32_t l1 = get_read_ptr(cb_out);
        if constexpr (OUT_RM) {
            const uint32_t row0 = bh * Ht * 32 + t * 32;
            for (uint32_t r = 0; r < 32; ++r, l1 += out_page) {
                noc_async_write(l1, out_acc.get_noc_addr(row0 + r), out_page);
            }
        } else {
            const uint32_t h = bh % H;
            const uint32_t b = bh / H;
            const uint32_t page = ((b * (H / G) + h / G) * Ht + t) * (G * WT) + (h % G) * WT;
            for (uint32_t j = 0; j < WT; ++j, l1 += out_page) {
                noc_async_write(l1, out_acc.get_noc_addr(page + j), out_page);
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, WT);
    }
}
