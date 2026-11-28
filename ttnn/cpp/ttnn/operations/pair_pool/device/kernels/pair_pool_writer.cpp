// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ttnn.bringup.pair_pool writer (NoC 1): per unit (i, j), the pooled tile -> output page i HT + j.
// compile-time args: cb_out, HT, PAIR_POOL_GX (define), out accessor; common args: out addr, per, extra

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "pair_pool_args.hpp"
#include "pair_pool_common.hpp"

void kernel_main() {
    using namespace pair_pool;
    constexpr uint32_t cb_out = get_compile_time_arg_val(writer_ct::CB_OUT);
    [[maybe_unused]] constexpr uint32_t HT = get_compile_time_arg_val(writer_ct::HT);
    constexpr auto out_args = TensorAccessorArgs<writer_ct::COUNT>();
    const uint32_t out_addr = get_common_arg_val<uint32_t>(writer_common::OUT_ADDR);
    uint32_t first, count;
    pair_pool_share(
        get_common_arg_val<uint32_t>(writer_common::PER),
        get_common_arg_val<uint32_t>(writer_common::EXTRA),
        PAIR_POOL_GX,
        first,
        count);
    const auto out = TensorAccessor(out_args, out_addr, 32 * 32 * 4);
    for (uint32_t u = first; u < first + count; ++u) {
        cb_wait_front(cb_out, 1);
        noc_async_write_page(u, out, get_read_ptr(cb_out));  // (page i HT + j = u)
        noc_async_writes_flushed();
        cb_pop_front(cb_out, 1);
    }
    noc_async_write_barrier();
}
