// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"
#include "../../../../../../kernel_helper_functions/pad_tile.hpp"

// Pushes the tile the compute adds to the last tile of every row when the last dimension is not a multiple of
// the tile width: 0 on the real columns, bfloat16 -inf on the num_datum_padded padding columns.
// first_row_only (32x32 tiles): only row 0 of faces 0 and 1 is written, for a compute that broadcasts the mask by row.
template <std::uint32_t num_datum_padded, std::uint32_t tile_hw, bool first_row_only = false>
void push_mask_padded_tile(DataflowBuffer& dfb_mask_padded) {
    // The value is the uint16 representation of bfloat16's -inf.
    constexpr std::uint16_t mask_val = 0xFF80;
    constexpr std::uint32_t num_datum_unpadded = 32 - num_datum_padded;

    dfb_mask_padded.reserve_back(1);
    const std::uint32_t ptr = dfb_mask_padded.get_write_ptr();
    if constexpr (first_row_only && tile_hw == 32 * 32) {
        constexpr std::uint32_t face_words = 16 * 16 / 2;
        volatile tt_l1_ptr std::uint32_t* word_ptr = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(ptr);
        for (std::uint32_t c = 0; c < 32; c += 2) {
            const std::uint32_t lo = c >= num_datum_unpadded ? mask_val : 0;
            const std::uint32_t hi = c + 1 >= num_datum_unpadded ? mask_val : 0;
            word_ptr[(c / 16) * face_words + (c % 16) / 2] = lo | (hi << 16);
        }
        // Read back the last word written so the stores reach L1 before the push.
#if defined(ARCH_QUASAR)
        (void)word_ptr[face_words + 7];
#else
        (void)ckernel::load_blocking(&word_ptr[face_words + 7]);
#endif
        dfb_mask_padded.push_back(1);
        return;
    }
    volatile tt_l1_ptr std::uint16_t* zero_ptr = reinterpret_cast<volatile tt_l1_ptr std::uint16_t*>(ptr);
    for (std::uint32_t i = 0; i < tile_hw; i++) {
        zero_ptr[i] = 0;
    }
    fill_pad_tile<std::uint16_t, num_datum_unpadded, 32>(ptr, mask_val);
    dfb_mask_padded.push_back(1);
}
