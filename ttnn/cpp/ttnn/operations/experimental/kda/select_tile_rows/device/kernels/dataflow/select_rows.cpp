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

// Copy this worker's column slice of each selected row of a row-major BF16 table, page = row.
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
    // A tile-wide column slice of a BF16 row.
    constexpr uint32_t slice_unit_bytes = tt::constants::TILE_WIDTH * sizeof(uint16_t);
    const auto input = TensorAccessor(tensor::input);
    const auto output = TensorAccessor(tensor::output);
    const auto output_second = TensorAccessor(tensor::output_second);
    DataflowBuffer staging(dfb::staging);
    Noc noc;

    // Staging holds [indices | selected row slices].
    staging.reserve_back(1);
    const uint32_t base = staging.get_write_ptr();
    const uint32_t slices = base + 64;
    uint32_t words[rows];
    select_rows::resolve_rows<rows, record, has_actual_end, sp_rank, sp_size, local_rows>(noc, base, words);

    const uint32_t end_tile = first_tile + tiles_per_core < width_tiles ? first_tile + tiles_per_core : width_tiles;
    const uint32_t bytes = (end_tile - first_tile) * slice_unit_bytes;
    const uint32_t column_offset = first_tile * slice_unit_bytes;
    for (uint32_t index = 0; index < rows; ++index) {
        noc.async_read(
            input,
            CoreLocalMem<uint32_t>(slices + index * tiles_per_core * slice_unit_bytes),
            bytes,
            {.page_id = words[index], .offset_bytes = column_offset},
            {});
    }
    noc.async_read_barrier();
    for (uint32_t index = 0; index < rows; ++index) {
        // Rows past the first output's share go to the second output.
        const uint32_t source = slices - base + index * tiles_per_core * slice_unit_bytes;
        if (index >= rows_per_output) {
            noc.async_write(
                staging,
                output_second,
                bytes,
                {.offset_bytes = source},
                {.page_id = index - rows_per_output, .offset_bytes = column_offset});
        } else {
            noc.async_write(
                staging, output, bytes, {.offset_bytes = source}, {.page_id = index, .offset_bytes = column_offset});
        }
    }
    noc.async_write_barrier();
    staging.push_back(1);
    staging.wait_front(1);
    staging.pop_front(1);
}
