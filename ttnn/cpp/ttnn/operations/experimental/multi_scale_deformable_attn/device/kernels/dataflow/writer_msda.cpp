// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer kernel for fused multi-scale deformable attention (FPU path,
// 32-query batched). Each output tile from compute carries up to 32
// query outputs stacked vertically. For each tile, copy v_rows rows out
// of the tile faces and NoC-write each as one output stick (D=32 bf16).
//
// Tile face layout (bf16, 32x32 = 4 faces of 16x16, 2048 B):
//   row r ∈ [0, 16): TL[r*32..r*32+31] + TR[512+r*32..512+r*32+31]
//   row r ∈ [16, 32): BL[1024+(r-16)*32..] + BR[1536+(r-16)*32..]
// One row = 32 bf16 = D values laid out as the two 32-byte halves above.
//
// Per-tile runtime args (2 per tile): (start_stick_id, v_rows).
// start_stick_id_t = n_t * Q + q_start_t; rows 0..v_rows-1 are written
// to consecutive sticks starting at that id.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/operations/experimental/multi_scale_deformable_attn/device/kernels/msda_tile_layout.hpp"

constexpr uint32_t output_tile_cb_index = get_compile_time_arg_val(0);
constexpr uint32_t output_scratch_cb_index = get_compile_time_arg_val(1);
constexpr uint32_t output_stick_nbytes = get_compile_time_arg_val(2);

constexpr auto output_args = TensorAccessorArgs<3>();

constexpr uint32_t HALF_STICK_NBYTES = 32;
constexpr uint32_t HALF_WORDS = HALF_STICK_NBYTES / sizeof(uint32_t);

void kernel_main() {
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_output_tiles = get_arg_val<uint32_t>(1);

    const auto output_acc = TensorAccessor(output_args, output_addr, output_stick_nbytes);

    Noc noc;
    CircularBuffer output_tile_cb(output_tile_cb_index);
    CircularBuffer output_scratch_cb(output_scratch_cb_index);

    output_scratch_cb.reserve_back(1);
    const uint32_t scratch_l1 = output_scratch_cb.get_write_ptr();

    uint32_t arg_idx = 2;
    for (uint32_t t = 0; t < num_output_tiles; ++t) {
        const uint32_t start_id = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t v_rows = get_arg_val<uint32_t>(arg_idx++);

        output_tile_cb.wait_front(1);
        const uint32_t tile_l1 = output_tile_cb.get_read_ptr();

        for (uint32_t r = 0; r < v_rows; ++r) {
            const auto off = msda_tile_layout::tile_row_offsets(r);
            const uint32_t src_lo = tile_l1 + off.lo;
            const uint32_t src_hi = tile_l1 + off.hi;

            CoreLocalMem<volatile uint32_t> dst(scratch_l1);
            CoreLocalMem<volatile uint32_t> sl(src_lo);
            CoreLocalMem<volatile uint32_t> sh(src_hi);
            for (uint32_t i = 0; i < HALF_WORDS; ++i) {
                dst[i] = sl[i];
            }
            for (uint32_t i = 0; i < HALF_WORDS; ++i) {
                dst[HALF_WORDS + i] = sh[i];
            }

            CoreLocalMem<uint32_t> src(scratch_l1);
            noc.async_write(src, output_acc, output_stick_nbytes, {.offset_bytes = 0}, {.page_id = start_id + r});
            noc.async_writes_flushed();
        }
        noc.async_write_barrier();
        output_tile_cb.pop_front(1);
    }
}
