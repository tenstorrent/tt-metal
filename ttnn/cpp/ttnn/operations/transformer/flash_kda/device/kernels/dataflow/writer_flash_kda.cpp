// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Writer: writes the two per-core Flash KDA outputs (S_new, out) from CBs to DRAM.
//
// Runtime args:
//   0  item_idx
//   1  S_new_addr
//   2  out_addr
//
// Compile-time args: Kt, Vt

#include <stdint.h>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr std::uint32_t Kt = get_compile_time_arg_val(0);
    constexpr std::uint32_t Vt = get_compile_time_arg_val(1);

    const std::uint32_t item_idx = get_arg_val<std::uint32_t>(0);
    const std::uint32_t s_new_addr = get_arg_val<std::uint32_t>(1);
    const std::uint32_t out_addr = get_arg_val<std::uint32_t>(2);

    constexpr std::uint32_t state_tiles = Kt * Vt;

    constexpr std::uint32_t cb_S_new = tt::CBIndex::c_12;
    constexpr std::uint32_t cb_out = tt::CBIndex::c_13;

    constexpr std::uint32_t f32_tile = get_tile_size(cb_S_new);

    // TensorAccessors for the interleaved fp32 DRAM outputs. The per-tensor
    // TensorAccessorArgs compile-time blocks are appended (in this order) by the
    // program factory right after the {Kt, Vt} compile-time args.
    constexpr auto sn_args = TensorAccessorArgs<2>();
    const auto sn_gen = TensorAccessor(sn_args, s_new_addr, f32_tile);
    constexpr auto out_args = TensorAccessorArgs<sn_args.next_compile_time_args_offset()>();
    const auto out_gen = TensorAccessor(out_args, out_addr, f32_tile);

    const std::uint32_t s_new_off = item_idx * state_tiles;
    const std::uint32_t out_off = item_idx * Vt;

    Noc noc;
    CircularBuffer cb_S_new_o(cb_S_new);
    CircularBuffer cb_out_o(cb_out);

    cb_S_new_o.wait_front(state_tiles);
    for (std::uint32_t t = 0; t < state_tiles; t++) {
        noc.async_write(cb_S_new_o, sn_gen, f32_tile, {.offset_bytes = t * f32_tile}, {.page_id = s_new_off + t});
    }
    noc.async_write_barrier();
    cb_S_new_o.pop_front(state_tiles);

    cb_out_o.wait_front(Vt);
    for (std::uint32_t t = 0; t < Vt; t++) {
        noc.async_write(cb_out_o, out_gen, f32_tile, {.offset_bytes = t * f32_tile}, {.page_id = out_off + t});
    }
    noc.async_write_barrier();
    cb_out_o.pop_front(Vt);
}
