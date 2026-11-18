// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// accumulator_writer: writes an accumulator's finished token rows (tokens accumulator_index, accumulator_index + A,
// ...) to the combine output
// Runs on: combine accumulators that are not port cores (alone, or as the accumulator role of
//          combine_port_or_accumulator_writer.cpp),
//          RISCV_0 (BRISC), NoC 1 (combine dm_kernel: RISCV_0 is pinned to NoC 1)
// Responsibilities:
//  - reads each finished row from cb_out (L1), produced by accumulator_compute
//  - writes it to the output tensor (TensorAccessor, row i * A + accumulator_index) over NoC 1
//  - synchronises only through cb_out with the compute
// Note: with accumulators_on_ports the port cores' combine_port_writer does these writes instead

// Accumulator writer: accumulator `accumulator_index`'s finished token rows (tokens accumulator_index,
// accumulator_index + A, ...) to the output.

#include "../../../../common/kernels/moe_block_args.hpp"
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "../combine_names.hpp"

// Compile-time args: all named (combine_names.hpp), the same whether this role runs alone or in a two-role kernel;
// the output's accessor args are the only positional ones
constexpr uint32_t cb_out = get_named_compile_time_arg_val(combine_names::kOutputCb);
constexpr uint32_t row_bytes = get_named_compile_time_arg_val(combine_names::kRowBytes);
constexpr uint32_t row_tiles = get_named_compile_time_arg_val(combine_names::kRowTiles);
constexpr uint32_t A = get_named_compile_time_arg_val(combine_names::kAccumulatorCount);
constexpr auto o_args =
    TensorAccessorArgs<get_named_compile_time_arg_val(combine_names::kAccumulatorWriterOutputAccessorFirstArg)>();

#ifdef COMBINE_ROLE_FUNCTION_NAME  // (one role of a two-role kernel: combine_port_or_accumulator_reader / _writer.cpp)
void COMBINE_ROLE_FUNCTION_NAME() {
#else
void kernel_main() {
    moe_block_args::load();  // (a two-role kernel loads the core's args once, then picks this chip's role)
#endif
    const uint32_t o_addr = moe_block_args::arg(0), n_tok = moe_block_args::arg(1),
                   accumulator_index = moe_block_args::arg(2);
    const auto o = TensorAccessor(o_args, o_addr, row_bytes);
    for (uint32_t i = 0; i < n_tok; ++i) {
        cb_wait_front(cb_out, row_tiles);
        noc_async_write(get_read_ptr(cb_out), o.get_noc_addr(i * A + accumulator_index), row_bytes);
        noc_async_writes_flushed();
        cb_pop_front(cb_out, row_tiles);
    }
    noc_async_write_barrier();
}
