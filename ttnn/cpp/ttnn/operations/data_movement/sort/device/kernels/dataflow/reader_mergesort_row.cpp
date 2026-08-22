// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

#include <cstdint>

/*
Mergesort row engine reader: streams one row's data to the compute kernel in
K=2048-element chunks (TILES_PER_RUN bf16 "tile" pages of 1024 consecutive
row elements each — the layout the TopK XL copy unpack consumes).

ROW_MAJOR input: one page per row, so a chunk is a single contiguous read.

TILE input: a row's data lives as two 16-element (32 B) face rows in every
tile of its tile-row. Blackhole DRAM reads require 64-byte alignment
(NOC_DRAM_READ_ALIGNMENT_BYTES; a 32 B-offset read returns the neighbouring
row's face row instead), so the gather reads the aligned 64 B window that
contains each face row into a scratch page and the RISC-V core compacts the
wanted halves into the staging pages.
*/
void kernel_main() {
    const uint32_t start_row = get_arg(args::start_row);
    const uint32_t num_rows = get_arg(args::num_rows);

    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t num_chunks = get_arg(args::num_chunks);

    constexpr uint32_t K = 2048;
    constexpr uint32_t TILES_PER_RUN = 2;
    constexpr uint32_t chunk_bytes = K * 2;  // bf16
    constexpr uint32_t TILE_H = 32;

    const auto input_accessor = TensorAccessor(tensor::input_tensor);
    DataflowBuffer input_stage_dfb(dfb::input_stage);
    Noc noc;

#ifdef IS_ROW_MAJOR
    for (uint32_t local_row = 0; local_row < num_rows; ++local_row) {
        const uint32_t row = start_row + local_row;
        for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
            input_stage_dfb.reserve_back(TILES_PER_RUN);
            // One RM page per row: read the chunk's contiguous span.
            noc.async_read(
                input_accessor,
                input_stage_dfb,
                chunk_bytes,
                {.page_id = row, .offset_bytes = chunk * chunk_bytes},
                {.offset_bytes = 0});
            noc.async_read_barrier();
            input_stage_dfb.push_back(TILES_PER_RUN);
        }
    }
#else
    DataflowBuffer reader_scratch_dfb(dfb::reader_scratch);
    constexpr uint32_t tiles_per_chunk = K / TILE_H;  // 64
    for (uint32_t local_row = 0; local_row < num_rows; ++local_row) {
        const uint32_t row = start_row + local_row;
        // Row `row` = tile-row row/32, in-tile row r = row%32. Within a tile,
        // in-tile row r's 32 elements are the two 32 B face rows at byte
        // offsets fo and fo+512, fo = (r >= 16 ? 1024 : 0) + (r % 16) * 32.
        const uint32_t tile_row = row / TILE_H;
        const uint32_t r = row % TILE_H;
        const uint32_t fo = ((r >= 16) ? 1024u : 0u) + (r % 16) * 32u;
        const uint32_t fo_aligned = fo & ~63u;  // 64 B window base
        const uint32_t half = fo & 32u;         // wanted 32 B within the window

        for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
            const uint32_t tile_base = tile_row * Wt + chunk * tiles_per_chunk;

            // Gather the aligned 64 B windows: two per input tile, 128 B of
            // scratch per tile.
            reader_scratch_dfb.reserve_back(1);
            for (uint32_t t = 0; t < tiles_per_chunk; ++t) {
                noc.async_read(
                    input_accessor,
                    reader_scratch_dfb,
                    64,
                    {.page_id = tile_base + t, .offset_bytes = fo_aligned},
                    {.offset_bytes = t * 128u});
                noc.async_read(
                    input_accessor,
                    reader_scratch_dfb,
                    64,
                    {.page_id = tile_base + t, .offset_bytes = fo_aligned + 512u},
                    {.offset_bytes = t * 128u + 64u});
            }
            noc.async_read_barrier();
            reader_scratch_dfb.push_back(1);
            reader_scratch_dfb.wait_front(1);

            // Compact the wanted 32 B halves into the contiguous chunk staging.
            input_stage_dfb.reserve_back(TILES_PER_RUN);
            volatile tt_l1_ptr uint32_t* src =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reader_scratch_dfb.get_read_ptr() + half);
            volatile tt_l1_ptr uint32_t* dst =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(input_stage_dfb.get_write_ptr());
            for (uint32_t t = 0; t < tiles_per_chunk; ++t) {
                const uint32_t s = t * 32;  // 128 B of scratch = 32 words per tile
                const uint32_t d = t * 16;  // 64 B of staging = 16 words per tile
                for (uint32_t w = 0; w < 8; ++w) {
                    dst[d + w] = src[s + w];
                    dst[d + 8 + w] = src[s + 16 + w];
                }
            }
            // Drain the RISC-V store queue before signalling the compute
            // kernel (RISC-V stores and NoC/compute are independent L1
            // clients with no program-order guarantee).
            __sync_synchronize();

            reader_scratch_dfb.pop_front(1);
            input_stage_dfb.push_back(TILES_PER_RUN);
        }
    }
#endif
}
