// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Writer: reads output chunks and final state from DFBs, writes to DRAM.
//
// Runtime args: head_idx, num_chunks
//
// Tensor bindings: output (output [BH, NC, C, Dv]), final_state (final_state [BH, Dk, Dv])
//
// Compile-time args: Ct, Kt, Vt

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t Ct = get_arg(args::Ct);
    constexpr uint32_t Kt = get_arg(args::Kt);
    constexpr uint32_t Vt = get_arg(args::Vt);

    const uint32_t head_idx = get_arg(args::head_idx);
    const uint32_t NC = get_arg(args::num_chunks);

    constexpr uint32_t out_tiles = Ct * Vt;
    constexpr uint32_t state_tiles = Kt * Vt;

    constexpr auto dfb_out = dfb::out;
    constexpr auto dfb_final_state = dfb::final_state;

    constexpr uint32_t f32_tile = get_tile_size(dfb_out);

    // TensorAccessors for the interleaved fp32 DRAM outputs, one per tensor binding.
    const auto out_gen = TensorAccessor(tensor::output);
    const auto st_gen = TensorAccessor(tensor::final_state);

    const uint32_t h_off_out = head_idx * NC * out_tiles;
    const uint32_t h_off_st = head_idx * state_tiles;

    Noc noc;
    DataflowBuffer dfb_out_o(dfb_out);
    DataflowBuffer dfb_final_state_o(dfb_final_state);

    // Write output chunks as they become available.
    for (uint32_t c = 0; c < NC; c++) {
        dfb_out_o.wait_front(out_tiles);
        uint32_t tile_off = h_off_out + c * out_tiles;
        for (uint32_t t = 0; t < out_tiles; t++) {
            noc.async_write(dfb_out_o, out_gen, f32_tile, {.offset_bytes = t * f32_tile}, {.page_id = tile_off + t});
        }
        noc.async_write_barrier();
        dfb_out_o.pop_front(out_tiles);
    }

    // Write final state (pushed by compute kernel after all chunks complete).
    dfb_final_state_o.wait_front(state_tiles);
    for (uint32_t t = 0; t < state_tiles; t++) {
        noc.async_write(dfb_final_state_o, st_gen, f32_tile, {.offset_bytes = t * f32_tile}, {.page_id = h_off_st + t});
    }
    noc.async_write_barrier();
    dfb_final_state_o.pop_front(state_tiles);
}
