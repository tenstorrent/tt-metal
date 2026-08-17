// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// TILE-layout input reader (shared by the row-parallel and column-parallel
// factories; the per-core slice offset arrives as start_element, 0 on the
// row-parallel path). Reads each LLK chunk of a logical row straight out of
// the tile-padded input tensor into the same contiguous chunk layout the
// ROW_MAJOR readers (reader.cpp / reader_local.cpp) produce, so the compute
// kernels are untouched. This lets callers skip the untilize op entirely —
// for a single logical row it also skips untilize's 32x tile-padding read
// amplification (only the row's own face rows are pulled from DRAM).
//
// A 32x32 bf16 tile stores four 16x16 faces contiguously (f0 rows0-15/
// cols0-15, f1 rows0-15/cols16-31, f2, f3): one logical row contributes one
// 32-byte face row per 16 columns. Blackhole NoC DRAM READS require 64-byte
// congruence between the DRAM and L1 addresses ((l1 & 63) == (dram & 63));
// the chunk CB's 32-byte-stride slice destinations alternate 0/32 mod 64
// while a row's DRAM face-row offsets are constant mod 64, so no direct CB
// placement satisfies every slice. The reader therefore stages every slice:
//   phase A: one 64-byte read per 16-element slice, from the 64-aligned
//            offset at/below the face row into a 64-byte-stride staging slot
//            (both ends 64-aligned -> congruent; the extra 32 bytes are the
//            neighboring face row, discarded). Never crosses the tile page
//            end: the last face row starts 64 bytes before it.
//   phase B: one 32-byte local L1 copy per slice from slot + delta into the
//            chunk CB, delta = (face_row & 1) * 32 (constant per row; 16-byte
//            L1 congruence holds since delta is a multiple of 16).
//
// Tail chunks read only ceil(active_elements / 16) slices; the CB tail beyond
// them is stale, exactly like the ROW_MAJOR readers, and the compute masks by
// active element count. Tile padding COLUMNS inside a read slice are likewise
// masked by the compute (search width comes from the logical shape).

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_row = get_arg_val<uint32_t>(1);
    const uint32_t num_rows = get_arg_val<uint32_t>(2);
    const uint32_t num_chunks = get_arg_val<uint32_t>(3);
    const uint32_t tail_slices = get_arg_val<uint32_t>(4);
    const uint32_t w_tiles = get_arg_val<uint32_t>(5);         // padded input width / 32
    const uint32_t rows_2d = get_arg_val<uint32_t>(6);         // logical shape[-2] (1 for rank < 2)
    const uint32_t start_in_slice = get_arg_val<uint32_t>(7);  // start_row % rows_2d
    const uint32_t start_slice = get_arg_val<uint32_t>(8);     // start_row / rows_2d
    const uint32_t start_element = get_arg_val<uint32_t>(9);   // column-parallel slice offset (elements)

    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_stage = get_compile_time_arg_val(1);
    constexpr uint32_t slices_per_chunk = get_compile_time_arg_val(2);  // llk_k / 16
    constexpr uint32_t tiles_per_chunk = get_compile_time_arg_val(3);   // CB pages per chunk
    constexpr auto input_args = TensorAccessorArgs<4>();

    constexpr uint32_t input_tile_bytes = 32 * 32 * 2;  // bf16 tile page
    const auto input = TensorAccessor(input_args, src_addr, input_tile_bytes);
    CircularBuffer input_cb(cb_in);
    CircularBuffer stage_cb(cb_stage);
    Noc noc;

    // The staging slots must be 64-byte aligned for DRAM-read congruence; the
    // CB base itself is only guaranteed 16-byte aligned, so align up (the CB
    // is sized with 64 bytes of slack). Held reserved for the whole kernel.
    stage_cb.reserve_back(1);
    const uint32_t stage_base = (stage_cb.get_write_ptr() + 63u) & ~63u;

    const uint32_t noc_id = noc.get_noc_id();
    const auto local_src = [noc_id](uint32_t addr) {
        return noc_traits_t<UnicastEndpoint>::src_args_type{
            .noc_x = static_cast<uint32_t>(my_x[noc_id]), .noc_y = static_cast<uint32_t>(my_y[noc_id]), .addr = addr};
    };

    const uint32_t tiles_per_slice_h = (rows_2d + 31u) >> 5;
    uint32_t slice_idx = start_slice;
    uint32_t in_slice_row = start_in_slice;
    (void)start_row;

    for (uint32_t local_row = 0; local_row < num_rows; ++local_row) {
        const uint32_t tile_row = slice_idx * tiles_per_slice_h + (in_slice_row >> 5);
        const uint32_t in_tile_r = in_slice_row & 31u;
        const uint32_t face_row = in_tile_r & 15u;
        // Byte offset of this row's face-column-0 run within a tile, split into
        // the 64-aligned base and the constant sub-64 delta.
        const uint32_t row_face_bytes = (((in_tile_r >> 4) & 1u) * 512u + face_row * 16u) * 2u;
        const uint32_t delta = row_face_bytes & 63u;  // (face_row & 1) * 32
        const uint32_t aligned_base = row_face_bytes & ~63u;

        for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
            const uint32_t active_slices = (chunk + 1 == num_chunks) ? tail_slices : slices_per_chunk;
            input_cb.reserve_back(tiles_per_chunk);
            const uint32_t cb_base = input_cb.get_write_ptr();
            const uint32_t chunk_col = start_element + chunk * (slices_per_chunk * 16u);

            for (uint32_t s = 0; s < active_slices; ++s) {
                const uint32_t col = chunk_col + s * 16u;
                const uint32_t page = tile_row * w_tiles + (col >> 5);
                const uint32_t src_off = aligned_base + ((col >> 4) & 1u) * 512u;  // 64-aligned
                noc.async_read(
                    input,
                    CoreLocalMem<uint32_t>(stage_base + s * 64u),
                    64,
                    {.page_id = page, .offset_bytes = src_off},
                    {.offset_bytes = 0});
            }
            noc.async_read_barrier();

            noc.set_async_read_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                UnicastEndpoint{}, 32, local_src(stage_base));
            for (uint32_t s = 0; s < active_slices; ++s) {
                noc.async_read_with_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                    UnicastEndpoint{},
                    CoreLocalMem<uint32_t>(cb_base + s * 32u),
                    32,
                    local_src(stage_base + s * 64u + delta),
                    {.offset_bytes = 0});
            }
            noc.async_read_barrier();
            input_cb.push_back(tiles_per_chunk);
        }

        if (++in_slice_row == rows_2d) {
            in_slice_row = 0;
            ++slice_idx;
        }
    }
}
