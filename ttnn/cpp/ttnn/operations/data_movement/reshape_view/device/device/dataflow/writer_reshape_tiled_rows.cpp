// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer of the tiled reshape that keeps the last dim. It walks the same runs as the reader
// (hostdevcommon/rows_common.hpp), copies each run's rows from its input tile into a scratch output
// tile, face row block by face row block, and writes the tile out. Rows of the output tile past the
// logical height are tile padding and are written as zeros, as the codegen reshape does.

#include "experimental/kernel_args.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/scratchpad.h"

#include "ttnn/operations/common/shared_with_host/runtime_length.hpp"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/operations/data_movement/reshape_view/device/hostdevcommon/rows_common.hpp"

using namespace tt::data_movement::common;
namespace rows = ttnn::prim::detail::reshape_rows;

// 32x32 tiles of four 16x16 faces: a tile row is one face row in the left face and one in the right.
constexpr uint32_t kFaceRows = 16;
constexpr uint32_t kFaceElements = 16 * 16;

void kernel_main() {
    const uint32_t core_idx = get_arg(args::core_idx);
    const uint32_t num_output_pages = get_arg(args::num_output_pages);
    const uint32_t in_rows = get_arg(args::in_rows);
    const uint32_t out_rows = get_arg(args::out_rows);

    constexpr uint32_t Tile_size_bytes = get_arg(args::Tile_size_bytes);
    constexpr uint32_t element_sz_bytes = get_arg(args::element_sz_bytes);
    constexpr uint32_t tiles_per_row = get_arg(args::tiles_per_row);
    constexpr uint32_t num_cores = get_arg(args::num_cores);
    constexpr uint32_t face_row_bytes = kFaceRows * element_sz_bytes;
    constexpr uint32_t face_bytes = kFaceElements * element_sz_bytes;

    const rows::Rows r{
        in_rows,
        ttnn::runtime_length::div_up(in_rows, rows::kTileRows),
        out_rows,
        ttnn::runtime_length::div_up(out_rows, rows::kTileRows),
        tiles_per_row};
    const uint32_t start = ttnn::runtime_length::balanced_start(core_idx, num_output_pages, num_cores);
    const uint32_t end = ttnn::runtime_length::balanced_end(core_idx, num_output_pages, num_cores);

    const auto output_addrgen = TensorAccessor(tensor::output);

    Noc noc;
    DataflowBuffer dfb_in_tiles(dfb::in_tiles);
    Scratchpad<uint8_t> working(scratch::working);
    const auto working_mem = working.local_mem();
    Scratchpad<uint32_t> working_words(scratch::working);  // the same scratch, to zero it word by word

    // Byte offset of tile row `row` in the left (half 0) or right (half 1) face of a tile.
    auto face_row_offset = [](uint32_t row, uint32_t half) {
        return ((((row / kFaceRows) * 2) + half) * face_bytes) + ((row % kFaceRows) * face_row_bytes);
    };

    for (uint32_t out_page = start; out_page < end; ++out_page) {
        // Zero the padding rows, which no run writes: in each face of the rows' half, from the first
        // padding row to the end of the face.
        const uint32_t rows_in_tile = rows::rows_in_tile(r, out_page);
        if (rows_in_tile < rows::kTileRows) {
            for (uint32_t row = rows_in_tile; row < rows::kTileRows;) {
                const uint32_t block = kFaceRows - (row % kFaceRows);
                for (uint32_t half = 0; half < 2; ++half) {
                    const uint32_t first_word = face_row_offset(row, half) / sizeof(uint32_t);
                    const uint32_t words = block * face_row_bytes / sizeof(uint32_t);
                    for (uint32_t i = 0; i < words; ++i) {
                        working_words[first_word + i] = 0;
                    }
                }
                row += block;
            }
            // The tile write below reads the scratch through the NOC: keep these stores before it.
            asm volatile("" ::: "memory");
        }
        rows::for_each_run(r, out_page, [&](const rows::Run& run) {
            dfb_in_tiles.wait_front(1);
            const uint32_t input_base_addr = dfb_in_tiles.get_read_ptr();
            uint32_t done = 0;
            while (done < run.count) {
                const uint32_t in_row = run.in_row + done;
                const uint32_t out_row = run.out_row + done;
                // Rows that stay within one face in both tiles are contiguous in both.
                uint32_t block = run.count - done;
                if (kFaceRows - (in_row % kFaceRows) < block) {
                    block = kFaceRows - (in_row % kFaceRows);
                }
                if (kFaceRows - (out_row % kFaceRows) < block) {
                    block = kFaceRows - (out_row % kFaceRows);
                }
                for (uint32_t half = 0; half < 2; ++half) {
                    tt_memmove<false, true, false, Tile_size_bytes>(
                        noc,
                        working_mem + face_row_offset(out_row, half),
                        CoreLocalMem<uint8_t>(input_base_addr + face_row_offset(in_row, half)),
                        block * face_row_bytes);
                }
                done += block;
            }
            noc.async_write_barrier();
            dfb_in_tiles.pop_front(1);
        });

        enhanced_noc_async_write<Tile_size_bytes, true>(
            noc, working_mem, output_addrgen.get_noc_addr(out_page), Tile_size_bytes);
        noc.async_write_barrier();
    }
}
