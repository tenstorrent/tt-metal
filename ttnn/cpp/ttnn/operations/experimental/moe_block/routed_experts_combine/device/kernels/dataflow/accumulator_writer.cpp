// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// accumulator_writer: writes an accumulator's finished token rows (tokens me, me + A, ...) to the combine output
// Runs on: combine accumulators that are not port cores (ports apart: inside combine_cell_writer on the pool cells),
//          RISCV_0 (BRISC), NoC 1 (combine dm_kernel: RISCV_0 is pinned to NoC 1)
// Responsibilities:
//  - reads each finished row from cb_out (L1), produced by accumulator_compute
//  - writes it to the output tensor (TensorAccessor, row i * A + me) over NoC 1
//  - synchronises only through cb_out with the compute
// Note: on acc_on_ports the port cores' combine_port_writer (REC_ACC_ON_PORTS) does these writes instead

// Accumulator writer: accumulator `me`'s finished token rows (tokens me, me + A, ...)
// to the output.

#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/moe_block_args.hpp"
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
void kernel_main() {
#endif
#ifndef CMB_ROLE_FN
    moe_block_args::load();  // (a cell loads its block once, then picks its role: combine_cell_*.cpp)
#endif
    const uint32_t o_addr = moe_block_args::arg(0), n_tok = moe_block_args::arg(1), me = moe_block_args::arg(2);
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
#endif
