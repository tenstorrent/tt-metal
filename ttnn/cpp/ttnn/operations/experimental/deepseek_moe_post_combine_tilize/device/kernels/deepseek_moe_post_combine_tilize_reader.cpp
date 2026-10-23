// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "tt-metalium/constants.hpp"

void kernel_main() {
    Noc noc;

    constexpr uint32_t bytes_to_read_per_row = get_arg(args::bytes_to_read_per_row);

    const uint32_t intra_row_byte_offset = get_arg(args::intra_row_byte_offset);
    const uint32_t row_page_offset = get_arg(args::row_page_offset);

    const auto input_tensor_accessor = TensorAccessor(tensor::input);

    constexpr uint32_t tile_height = tt::constants::TILE_HEIGHT;

    DataflowBuffer dfb_tilize_input(dfb::tilize_input);

    dfb_tilize_input.reserve_back(tile_height);
    uint32_t l1_write_addr = dfb_tilize_input.get_write_ptr();

    uint32_t page_id = row_page_offset;
    for (uint32_t row = 0; row < tile_height; ++row) {
        noc.async_read(
            input_tensor_accessor,
            CoreLocalMem<uint32_t>(l1_write_addr),
            bytes_to_read_per_row,
            {.page_id = page_id, .offset_bytes = intra_row_byte_offset},
            {});

        l1_write_addr += bytes_to_read_per_row;
        page_id++;
    }

    noc.async_read_barrier();
    dfb_tilize_input.push_back(tile_height);
}
