// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Writer: reads output chunks and final state from CBs, writes to DRAM.
//
// Runtime args:
//   0  head_idx
//   1  num_chunks
//   2  out_addr        (output [BH, NC, C, Dv])
//   3  final_state_addr (final_state [BH, Dk, Dv])
//
// Compile-time args: Ct, Kt, Vt

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Ct = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Vt = get_compile_time_arg_val(2);

    const uint32_t head_idx = get_arg_val<uint32_t>(0);
    const uint32_t NC = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);
    const uint32_t st_addr = get_arg_val<uint32_t>(3);

    constexpr uint32_t out_tiles = Ct * Vt;
    constexpr uint32_t state_tiles = Kt * Vt;

    constexpr uint32_t cb_out = tt::CBIndex::c_24;
    constexpr uint32_t cb_final_state = tt::CBIndex::c_27;

    constexpr uint32_t f32_tile = get_tile_size(cb_out);

    // TensorAccessors for the interleaved fp32 DRAM outputs. The per-tensor
    // TensorAccessorArgs compile-time blocks are appended (in this order) by the
    // program factory right after the {Ct, Kt, Vt} compile-time args, so the first
    // block starts at compile-time-arg offset 3 and the second chains off it.
    constexpr auto out_args = TensorAccessorArgs<3>();
    const auto out_gen = TensorAccessor(out_args, out_addr, f32_tile);
    constexpr auto st_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    const auto st_gen = TensorAccessor(st_args, st_addr, f32_tile);

    const uint32_t h_off_out = head_idx * NC * out_tiles;
    const uint32_t h_off_st = head_idx * state_tiles;

    // Write output chunks as they become available.
    for (uint32_t c = 0; c < NC; c++) {
        cb_wait_front(cb_out, out_tiles);
        uint32_t tile_off = h_off_out + c * out_tiles;
        for (uint32_t t = 0; t < out_tiles; t++) {
            uint64_t na = out_gen.get_noc_addr(tile_off + t);
            noc_async_write(get_read_ptr(cb_out) + t * f32_tile, na, f32_tile);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, out_tiles);
    }

    // Write final state (pushed by compute kernel after all chunks complete).
    cb_wait_front(cb_final_state, state_tiles);
    for (uint32_t t = 0; t < state_tiles; t++) {
        uint64_t na = st_gen.get_noc_addr(h_off_st + t);
        noc_async_write(get_read_ptr(cb_final_state) + t * f32_tile, na, f32_tile);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_final_state, state_tiles);
}
