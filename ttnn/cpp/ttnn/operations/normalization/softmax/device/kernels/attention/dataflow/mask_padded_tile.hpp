// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"
#include "../../../../../../kernel_helper_functions/pad_tile.hpp"

// Pushes the tile the compute adds to the last tile of every row when the last dimension is not a multiple of
// the tile width: 0 on the real columns, bfloat16 -inf on the num_datum_padded padding columns.
template <std::uint32_t num_datum_padded, std::uint32_t tile_hw>
void push_mask_padded_tile(DataflowBuffer& dfb_mask_padded) {
    // The value is the uint16 representation of bfloat16's -inf.
    constexpr std::uint16_t mask_val = 0xFF80;
    constexpr std::uint32_t num_datum_unpadded = 32 - num_datum_padded;

    dfb_mask_padded.reserve_back(1);
    const std::uint32_t ptr = dfb_mask_padded.get_write_ptr();
    volatile tt_l1_ptr std::uint16_t* zero_ptr = reinterpret_cast<volatile tt_l1_ptr std::uint16_t*>(ptr);
    for (std::uint32_t i = 0; i < tile_hw; i++) {
        zero_ptr[i] = 0;
    }
    fill_pad_tile<std::uint16_t, num_datum_unpadded, 32>(ptr, mask_val);
    dfb_mask_padded.push_back(1);
}

// Zeroes the padding columns [last_tile_w, 32) of a row's last tile, read into dfb_id at l1_addr. The -inf mask
// above turns NaN or +inf into NaN, which poisons the row max and sum, and implicit tile padding may hold either.
// last_tile_w == 0 means there is no padding, or the tile is not 32x32 and the mask does not apply.
template <std::uint32_t dfb_id, std::uint32_t last_tile_w>
void zero_last_tile_padding(std::uint32_t l1_addr) {
    if constexpr (last_tile_w > 0) {
        constexpr DataFormat data_format = get_dataformat(dfb_id);
        // pad_last_ktile zeroes the same columns of matmul's last K tile.
        pad_last_ktile<data_format, last_tile_w>(l1_addr);
    }
}
