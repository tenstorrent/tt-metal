// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ttnn.bringup.pair_pool reader (NoC 0). Per unit (output tile-row i, column tile j; pair_pool_common.hpp): the four
// fp32 input tiles of input tile-rows 2i, 2i + 1 at columns j (kv) and HT + j (gate) -> cb_stage, then the rows are
// split by parity into cb_in: [kv even | gate even | kv odd | gate odd], row r of each = input row 64 i + 2 r (+ 1).
// Word copies in L1 (bit-exact), 16 words (one face row) at a time.
// compile-time args: cb_stage, cb_in, HT, PAIR_POOL_GX (define), src accessor; common args: src addr, per, extra

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "pair_pool_common.hpp"

void kernel_main() {
    constexpr uint32_t cb_stage = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in = get_compile_time_arg_val(1);
    constexpr uint32_t HT = get_compile_time_arg_val(2);
    constexpr auto src_args = TensorAccessorArgs<3>();
    const uint32_t src_addr = get_common_arg_val<uint32_t>(0);
    uint32_t first, count;
    pair_pool_share(get_common_arg_val<uint32_t>(1), get_common_arg_val<uint32_t>(2), PAIR_POOL_GX, first, count);

    constexpr uint32_t page = 32 * 32 * 4;
    const auto src = TensorAccessor(src_args, src_addr, page);
    using ptr_t = volatile tt_l1_ptr uint32_t*;

    cb_reserve_back(cb_stage, 4);  // private scratch (never pushed)
    const uint32_t st = get_write_ptr(cb_stage);
    for (uint32_t u = first; u < first + count; ++u) {
        const uint32_t i = u / HT, j = u % HT;
        // stage pages: 0 kv rows 64i..64i+31, 1 gate rows 64i..64i+31, 2 kv rows 64i+32.., 3 gate rows 64i+32..
        for (uint32_t b = 0; b < 2; ++b) {
            const uint32_t row_page = (2 * i + b) * (2 * HT);
            noc_async_read(src.get_noc_addr(row_page + j), st + (2 * b) * page, page);
            noc_async_read(src.get_noc_addr(row_page + HT + j), st + (2 * b + 1) * page, page);
        }
        noc_async_read_barrier();
        cb_reserve_back(cb_in, 4);
        const uint32_t in = get_write_ptr(cb_in);
        // dst tile 2 p + q (parity p, q 0 kv / 1 gate): row r <- input row ir = 2 r + p of stage page 2 (ir / 32) + q
        for (uint32_t p = 0; p < 2; ++p) {
            for (uint32_t q = 0; q < 2; ++q) {
                ptr_t dst = reinterpret_cast<ptr_t>(in + (2 * p + q) * page);
                for (uint32_t r = 0; r < 32; ++r) {
                    const uint32_t ir = 2 * r + p, sr = ir & 31u;
                    const ptr_t s = reinterpret_cast<ptr_t>(st + (2 * (ir >> 5) + q) * page);
                    for (uint32_t h = 0; h < 2; ++h) {
                        const ptr_t sw = s + ((sr >> 4) * 2 + h) * 256 + (sr & 15u) * 16;
                        const ptr_t dw = dst + ((r >> 4) * 2 + h) * 256 + (r & 15u) * 16;
#pragma GCC unroll 16
                        for (uint32_t w = 0; w < 16; ++w) {
                            dw[w] = sw[w];
                        }
                    }
                }
            }
        }
        cb_push_back(cb_in, 4);
    }
}
