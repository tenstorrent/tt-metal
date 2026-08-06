// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer kernel for fused multi-scale deformable attention (FPU path,
// 32-query batched). Each output tile from compute carries up to 32
// query outputs stacked vertically for one D-width chunk (≤ 32 channels).
// For each query-group, copy v_rows rows out of each d_tile's faces and
// NoC-write them into the matching byte offset of each output stick.
//
// Tile face layout (bf16, 32x32 = 4 faces of 16x16, 2048 B):
//   row r ∈ [0, 16): TL[r*32..r*32+31] + TR[512+r*32..512+r*32+31]
//   row r ∈ [16, 32): BL[1024+(r-16)*32..] + BR[1536+(r-16)*32..]
// One full-width row = 32 bf16 laid out as two 32-byte face-halves.
// D=16 uses only the lo half; D=64 emits two tiles (two face pairs).
//
// Per-group runtime args (2 per group): (start_stick_id, v_rows).
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
constexpr uint32_t D = get_compile_time_arg_val(3);

constexpr auto output_args = TensorAccessorArgs<4>();

constexpr uint32_t ELEMENT_SIZE = 2;  // bf16
constexpr uint32_t TILE_WIDTH = 32;
constexpr uint32_t NUM_D_TILES = (D + TILE_WIDTH - 1) / TILE_WIDTH;

// Gather `nbytes` from tile row `r` into a linear scratch stick chunk,
// walking face-halves (16 bf16 = 32 B each).
inline void copy_tile_row_to_stick_chunk(
    uint32_t scratch_l1, uint32_t tile_l1, uint32_t r, uint32_t nbytes) {
    constexpr uint32_t FACE_ROW_NBYTES = msda_tile_layout::WITHIN_FACE_ROW_STRIDE;
    constexpr uint32_t FACE_ROW_WORDS = FACE_ROW_NBYTES / sizeof(uint32_t);

    const uint32_t num_halves = nbytes / FACE_ROW_NBYTES;
    const auto off = msda_tile_layout::tile_row_offsets(r);
    CoreLocalMem<volatile uint32_t> dst(scratch_l1);

    for (uint32_t h = 0; h < num_halves; ++h) {
        const uint32_t src_off = (h == 0) ? off.lo : off.hi;
        CoreLocalMem<volatile uint32_t> src(tile_l1 + src_off);
        const uint32_t dst_word = h * FACE_ROW_WORDS;
        for (uint32_t i = 0; i < FACE_ROW_WORDS; ++i) {
            dst[dst_word + i] = src[i];
        }
    }
}

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

        for (uint32_t d_tile = 0; d_tile < NUM_D_TILES; ++d_tile) {
            const uint32_t d_start = d_tile * TILE_WIDTH;
            const uint32_t d_chunk = (D - d_start) < TILE_WIDTH ? (D - d_start) : TILE_WIDTH;
            const uint32_t chunk_nbytes = d_chunk * ELEMENT_SIZE;
            const uint32_t chunk_offset = d_start * ELEMENT_SIZE;

            output_tile_cb.wait_front(1);
            const uint32_t tile_l1 = output_tile_cb.get_read_ptr();

            for (uint32_t r = 0; r < v_rows; ++r) {
                copy_tile_row_to_stick_chunk(scratch_l1, tile_l1, r, chunk_nbytes);

                CoreLocalMem<uint32_t> src(scratch_l1);
                noc.async_write(
                    src,
                    output_acc,
                    chunk_nbytes,
                    {.offset_bytes = chunk_offset},
                    {.page_id = start_id + r});
                noc.async_writes_flushed();
            }
            noc.async_write_barrier();
            output_tile_cb.pop_front(1);
        }
    }
}
