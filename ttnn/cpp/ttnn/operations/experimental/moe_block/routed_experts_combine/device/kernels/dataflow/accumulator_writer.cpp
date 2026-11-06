// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Accumulator writer: accumulator `me`'s finished token rows (tokens me, me + A, ...)
// to the output.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_out = get_compile_time_arg_val(0);
constexpr uint32_t row_bytes = get_compile_time_arg_val(1);
constexpr uint32_t NT = get_compile_time_arg_val(2);
constexpr uint32_t A = get_compile_time_arg_val(3);
constexpr auto o_args = TensorAccessorArgs<4>();

void kernel_main() {
    const uint32_t o_addr = get_arg_val<uint32_t>(0), n_tok = get_arg_val<uint32_t>(1), me = get_arg_val<uint32_t>(2);
    const auto o = TensorAccessor(o_args, o_addr, row_bytes);
    for (uint32_t i = 0; i < n_tok; ++i) {
        cb_wait_front(cb_out, NT);
        noc_async_write(get_read_ptr(cb_out), o.get_noc_addr(i * A + me), row_bytes);
        noc_async_writes_flushed();
        cb_pop_front(cb_out, NT);
    }
    noc_async_write_barrier();
}
