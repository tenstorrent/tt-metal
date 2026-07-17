// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Writer: reads output chunks and final state from CBs, writes to DRAM.
//
// Runtime args:
//   0  head_idx
//   1  num_chunks
//   2  out_addr        (output: head-major [BH, NC, C, Dv] or token-major [B, T, H*Dv])
//   3  final_state_addr (final_state [BH, Dk, Dv])
//
// Compile-time args: Ct, Kt, Vt, TOKEN_MAJOR, H, T_tiles, then out + final_state accessor args.
//   TOKEN_MAJOR=0 → head-major output; the {H, T_tiles} args are ignored.
//   TOKEN_MAJOR=1 → scatter each head's chunk output into its token-major column range.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t Ct = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Vt = get_compile_time_arg_val(2);
    constexpr uint32_t TOKEN_MAJOR = get_compile_time_arg_val(3);
    constexpr uint32_t H = get_compile_time_arg_val(4);        // v-heads per batch (token-major only)
    constexpr uint32_t T_tiles = get_compile_time_arg_val(5);  // dest tile-rows = ceil(T/32) (token-major only)

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
    // program factory right after the {Ct, Kt, Vt, TOKEN_MAJOR, H, T_tiles} compile-time
    // args, so the first block starts at compile-time-arg offset 6 and the second chains off it.
    constexpr auto out_args = TensorAccessorArgs<6>();
    const auto out_gen = TensorAccessor(out_args, out_addr, f32_tile);
    constexpr auto st_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    const auto st_gen = TensorAccessor(st_args, st_addr, f32_tile);

    const uint32_t h_off_st = head_idx * state_tiles;

    Noc noc;
    CircularBuffer cb_out_o(cb_out);
    CircularBuffer cb_final_state_o(cb_final_state);

    // Write output chunks as they become available.
    if constexpr (TOKEN_MAJOR) {
        // Scatter into token-major [B, T, H*Dv]. cb_out holds one chunk = Ct*Vt tiles ordered
        // t = ct*Vt + vt. Destination page-id (interleaved, row-major):
        //   page(c,ct,vt) = (b*T_tiles + c*Ct + ct)*(H*Vt) + h*Vt + vt.
        // The Vt V-tiles of a row are contiguous (a Vt-tile burst); rows/chunks are strided.
        constexpr uint32_t HVt = H * Vt;
        const uint32_t b = head_idx / H;
        const uint32_t h = head_idx % H;
        const uint32_t head_page_base = b * T_tiles * HVt + h * Vt;  // page of (t_tile=0, vt=0) for this head
        for (uint32_t c = 0; c < NC; c++) {
            cb_out_o.wait_front(out_tiles);
            for (uint32_t ct = 0; ct < Ct; ct++) {
                const uint32_t t_tile = c * Ct + ct;
                if (t_tile >= T_tiles) {
                    continue;  // chunk-padding tail row: drop (CB is popped per chunk below)
                }
                const uint32_t row_page = head_page_base + t_tile * HVt;
                for (uint32_t vt = 0; vt < Vt; vt++) {
                    noc.async_write(
                        cb_out_o,
                        out_gen,
                        f32_tile,
                        {.offset_bytes = (ct * Vt + vt) * f32_tile},
                        {.page_id = row_page + vt});
                }
            }
            noc.async_write_barrier();
            cb_out_o.pop_front(out_tiles);
        }
    } else {
        // Legacy head-major [BH, NC, C, Dv]: chunk is a contiguous out_tiles run.
        const uint32_t h_off_out = head_idx * NC * out_tiles;
        for (uint32_t c = 0; c < NC; c++) {
            cb_out_o.wait_front(out_tiles);
            uint32_t tile_off = h_off_out + c * out_tiles;
            for (uint32_t t = 0; t < out_tiles; t++) {
                noc.async_write(cb_out_o, out_gen, f32_tile, {.offset_bytes = t * f32_tile}, {.page_id = tile_off + t});
            }
            noc.async_write_barrier();
            cb_out_o.pop_front(out_tiles);
        }
    }

    // Write final state (pushed by compute kernel after all chunks complete).
    cb_final_state_o.wait_front(state_tiles);
    for (uint32_t t = 0; t < state_tiles; t++) {
        noc.async_write(cb_final_state_o, st_gen, f32_tile, {.offset_bytes = t * f32_tile}, {.page_id = h_off_st + t});
    }
    noc.async_write_barrier();
    cb_final_state_o.pop_front(state_tiles);
}
