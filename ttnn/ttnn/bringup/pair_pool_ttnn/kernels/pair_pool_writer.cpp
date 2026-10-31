// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ttnn.bringup.pair_pool writer (NoC 1): per unit (i, j), the pooled tile -> output page i HT + j.
// compile-time args: cb_out, HT, PAIR_POOL_GX (define), out accessor; common args: out addr, per, extra

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "pair_pool_common.hpp"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t HT = get_compile_time_arg_val(1);
    constexpr auto out_args = TensorAccessorArgs<2>();
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    uint32_t first, count;
    pair_pool_share(get_common_arg_val<uint32_t>(1), get_common_arg_val<uint32_t>(2), PAIR_POOL_GX, first, count);
    const auto out = TensorAccessor(out_args, out_addr, 32 * 32 * 4);
    for (uint32_t u = first; u < first + count; ++u) {
        cb_wait_front(cb_out, 1);
        noc_async_write_page(u, out, get_read_ptr(cb_out));  // (page i HT + j = u)
        noc_async_writes_flushed();
        cb_pop_front(cb_out, 1);
    }
    noc_async_write_barrier();
}
