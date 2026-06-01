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
    // Multi-core-per-head value split: write this core's value slice [v_off, v_off+Vt)
    // into the global value dim (Vt_global tiles). split_v=1 => v_off=0, Vt_global=Vt (unchanged).
    const uint32_t v_off = get_arg_val<uint32_t>(4);
    const uint32_t Vt_global = get_arg_val<uint32_t>(5);

    constexpr uint32_t out_tiles = Ct * Vt;  // LOCAL value-tile count
    constexpr uint32_t state_tiles = Kt * Vt;

    constexpr uint32_t cb_out = tt::CBIndex::c_24;
    constexpr uint32_t cb_final_state = tt::CBIndex::c_27;

    constexpr uint32_t f32_tile = get_tile_size(cb_out);

    const InterleavedAddrGenFast<true> out_gen = {.bank_base_address = out_addr, .page_size = f32_tile};
    const InterleavedAddrGenFast<true> st_gen = {.bank_base_address = st_addr, .page_size = f32_tile};

    // Write output chunks (global out [BH,NC,C,Dv]); place this core's value slice.
    for (uint32_t c = 0; c < NC; c++) {
        cb_wait_front(cb_out, out_tiles);
        uint32_t out_chunk_base = head_idx * NC * Ct * Vt_global + c * Ct * Vt_global;
        for (uint32_t t = 0; t < out_tiles; t++) {
            uint32_t ct = t / Vt;  // Vt = local value-tile count
            uint32_t vl = t % Vt;
            uint32_t gtile = out_chunk_base + ct * Vt_global + v_off + vl;
            uint64_t na = get_noc_addr(gtile, out_gen);
            noc_async_write(get_read_ptr(cb_out) + t * f32_tile, na, f32_tile);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, out_tiles);
    }

    // Write final state (global [BH,Dk,Dv]); place this core's value slice.
    cb_wait_front(cb_final_state, state_tiles);
    for (uint32_t t = 0; t < state_tiles; t++) {
        uint32_t kt = t / Vt;
        uint32_t vl = t % Vt;
        uint32_t gtile = head_idx * Kt * Vt_global + kt * Vt_global + v_off + vl;
        uint64_t na = get_noc_addr(gtile, st_gen);
        noc_async_write(get_read_ptr(cb_final_state) + t * f32_tile, na, f32_tile);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_final_state, state_tiles);
}
