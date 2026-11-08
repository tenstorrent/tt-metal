// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Accumulator writer: accumulator `me`'s finished token rows (tokens me, me + A, ...)
// to the output.

#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/mb_loop.hpp"
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

#ifndef CMB_CT0  // (the first compile-time arg of this role: 0 alone, past the other role's in a pool cell's binary)
#define CMB_CT0 0
#endif
constexpr uint32_t cb_out = get_compile_time_arg_val(CMB_CT0 + 0);
constexpr uint32_t row_bytes = get_compile_time_arg_val(CMB_CT0 + 1);
constexpr uint32_t NT = get_compile_time_arg_val(CMB_CT0 + 2);
constexpr uint32_t A = get_compile_time_arg_val(CMB_CT0 + 3);
constexpr auto o_args = TensorAccessorArgs<CMB_CT0 + 4>();

#ifdef CMB_ROLE_FN  // (a role of a pool cell's binary - combine_cell_reader / _writer.cpp)
void CMB_ROLE_FN() {
#else
MB_LOOP_BODY {
#endif
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

#ifndef CMB_ROLE_FN
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/mb_loop_main.hpp"
#endif
