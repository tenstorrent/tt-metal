// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// N-block row sum with a tiled output (out[r] = sum_i src[i * S + r], rows of H bf16 -> TILE), reader (NCRISC): per
// block (tile row tr, 1024-column chunk c) and input i, the 32 row segments (2 KB each, one page per row) into c_0.
// Blocks j = me, me + P, ...
// CT: 0 ROW_BYTES, 1 NCH, 2 P, 3 N   Common RT: 0 src addr, 1 S (rows per input block), 2 blocks, 3 grid x
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "core_range.hpp"

void kernel_main() {
    constexpr uint32_t ROW_BYTES = get_compile_time_arg_val(0), NCH = get_compile_time_arg_val(1);
    constexpr uint32_t P = get_compile_time_arg_val(2), N = get_compile_time_arg_val(3);
    const InterleavedAddrGen<true> a = {.bank_base_address = get_common_arg_val<uint32_t>(0), .page_size = ROW_BYTES};
    const uint32_t S = get_common_arg_val<uint32_t>(1), blocks = get_common_arg_val<uint32_t>(2),
                   me = core_index(get_common_arg_val<uint32_t>(3));
    for (uint32_t j = me; j < blocks; j += P) {
        const uint32_t tr = j / NCH, c = j % NCH;
        for (uint32_t i = 0; i < N; ++i) {
            cb_reserve_back(tt::CBIndex::c_0, 32);
            const uint32_t la = get_write_ptr(tt::CBIndex::c_0);
            for (uint32_t r = 0; r < 32; ++r) {
                noc_async_read(get_noc_addr(i * S + tr * 32 + r, a, c * 2048), la + r * 2048, 2048);
            }
            noc_async_read_barrier();
            cb_push_back(tt::CBIndex::c_0, 32);
        }
    }
}
