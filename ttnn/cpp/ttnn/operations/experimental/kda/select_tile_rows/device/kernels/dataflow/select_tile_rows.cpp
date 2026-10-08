// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include <tt-metalium/constants.hpp>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "select_rows_common.hpp"

namespace {

// A BF16 tile holds four 16x16 faces; one tile row is a 32-byte face row in each of two faces.
constexpr uint32_t face_rows = 16;
constexpr uint32_t face_row_bytes = 16 * sizeof(uint16_t);
constexpr uint32_t face_bytes = face_rows * face_row_bytes;
constexpr uint32_t tile_row_bytes = 2 * face_row_bytes;
// DRAM reads start on this alignment, so each face row is fetched within its aligned span.
constexpr uint32_t span_bytes = 64;

}  // namespace

// Gather this worker's column tiles of each indexed row into row-major output rows.
template <
    uint32_t rows,
    uint32_t rows_per_output,
    uint32_t input_row_tiles,
    uint32_t width_tiles,
    uint32_t tiles_per_core,
    uint32_t record,
    uint32_t has_actual_end,
    uint32_t sp_rank,
    uint32_t sp_size,
    uint32_t local_rows>
TT_KERNEL void dataflow(uint32_t first_tile) {
    const auto input = TensorAccessor(tensor::input);
    const auto output = TensorAccessor(tensor::output);
    const auto output_second = TensorAccessor(tensor::output_second);
    DataflowBuffer staging(dfb::staging);
    Noc noc;

    // Staging holds [indices | aligned spans | gathered rows].
    staging.reserve_back(1);
    const uint32_t base = staging.get_write_ptr();
    const uint32_t spans = base + span_bytes;
    const uint32_t gathered = spans + 2 * tiles_per_core * span_bytes;
    uint32_t words[rows];
    select_rows::resolve_rows<rows, record, has_actual_end, sp_rank, sp_size, local_rows>(noc, base, words);

    const uint32_t end_tile = first_tile + tiles_per_core < width_tiles ? first_tile + tiles_per_core : width_tiles;
    const uint32_t tiles = end_tile - first_tile;
    for (uint32_t index = 0; index < rows; ++index) {
        const uint32_t row = words[index];
        const uint32_t tile_row = row / tt::constants::TILE_HEIGHT;
        const uint32_t row_in_tile = row % tt::constants::TILE_HEIGHT;
        for (uint32_t tile = 0; tile < tiles; ++tile) {
            for (uint32_t column = 0; column < 2; ++column) {
                const uint32_t offset =
                    (row_in_tile / face_rows * 2 + column) * face_bytes + row_in_tile % face_rows * face_row_bytes;
                noc.async_read(
                    input,
                    CoreLocalMem<uint32_t>(spans + (2 * tile + column) * span_bytes),
                    span_bytes,
                    {.page_id = tile_row * input_row_tiles + first_tile + tile,
                     .offset_bytes = offset & ~(span_bytes - 1)},
                    {});
            }
        }
        noc.async_read_barrier();
        for (uint32_t tile = 0; tile < tiles; ++tile) {
            for (uint32_t column = 0; column < 2; ++column) {
                const uint32_t offset = row_in_tile % face_rows * face_row_bytes;
                noc_async_read(
                    get_noc_addr(spans + (2 * tile + column) * span_bytes + (offset & (span_bytes - 1))),
                    gathered + (index * tiles_per_core + tile) * tile_row_bytes + column * face_row_bytes,
                    face_row_bytes);
            }
        }
        noc.async_read_barrier();
    }
    for (uint32_t index = 0; index < rows; ++index) {
        // Rows past the first output's share go to the second output.
        const uint32_t source = gathered - base + index * tiles_per_core * tile_row_bytes;
        if (index >= rows_per_output) {
            noc.async_write(
                staging,
                output_second,
                tiles * tile_row_bytes,
                {.offset_bytes = source},
                {.page_id = index - rows_per_output, .offset_bytes = first_tile * tile_row_bytes});
        } else {
            noc.async_write(
                staging,
                output,
                tiles * tile_row_bytes,
                {.offset_bytes = source},
                {.page_id = index, .offset_bytes = first_tile * tile_row_bytes});
        }
    }
    noc.async_write_barrier();
    staging.push_back(1);
    staging.wait_front(1);
    staging.pop_front(1);
}
