// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/embedding/device/kernels/dataflow/embeddings_common.hpp"
#include "api/debug/dprint.h"

void kernel_main() {
    Noc noc;

    const auto tile_offset = get_arg(args::tile_offset);
    const auto face_offset = get_arg(args::face_offset);
    const auto num_rows = get_arg(args::num_rows);

    const auto curr_col = get_arg(args::curr_col);
    const auto starting_index = get_arg(args::starting_index);

    constexpr auto weight_stick_size = get_arg(args::weight_stick_size);
    constexpr auto row_length = get_arg(args::row_length);

    const auto input = TensorAccessor(tensor::input);
    const auto weights = TensorAccessor(tensor::weights);

    constexpr uint32_t face_size = 16;
    constexpr uint32_t tile_height = 32;
    constexpr uint32_t face_hw = face_size * face_size;

    prepare_local_cache(noc, weights, weight_stick_size);

    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);

    dfb_in1.reserve_back(1);
    uint32_t input_l1_addr = dfb_in1.get_write_ptr();
    volatile tt_l1_ptr input_token_t* input_l1_ptr = reinterpret_cast<volatile tt_l1_ptr input_token_t*>(input_l1_addr);

    const uint32_t input_page_size = input.get_aligned_page_size();

    auto read_block = [&](const uint32_t& token_idx, const uint32_t& width_size, const uint32_t& offset = 0) {
        dfb_in0.reserve_back(1);
        uint32_t weight_l1_addr = dfb_in0.get_write_ptr();
        input_token_t token = static_cast<input_token_t>(input_l1_ptr[token_idx + offset]);
        read_token_async(noc, token, weights, weight_l1_addr, width_size);
        noc.async_read_barrier();
        dfb_in0.push_back(1);
    };

    uint32_t curr_tile = tile_offset;
    uint32_t offset = face_offset;
    uint32_t index = starting_index;
    bool read_indices = true;
    uint32_t col_offset = curr_col;
    uint32_t tiles_per_row = (row_length + tile_height - 1) / tile_height;

    for (uint32_t i = 0; i < num_rows; ++i) {
        if (read_indices) {
            noc.async_read(input, CoreLocalMem<uint32_t>(input_l1_addr), input_page_size, {.page_id = curr_tile}, {});
            noc.async_read_barrier();
            read_indices = false;
        }
        read_block(index, weight_stick_size, offset);
        index++;
        col_offset++;
        if (index == face_size || col_offset == row_length) {
            index = 0;
            uint32_t face = offset / (face_size * face_size);
            if (col_offset == row_length) {
                read_indices = true;
                col_offset = 0;
                if (offset == tile_height * tile_height) {
                    curr_tile++;
                    offset = 0;
                } else {
                    curr_tile -= (tiles_per_row - 1);
                    if (face % 2 == 0) {
#if defined ONLY_ONE_FACE_COLUMN
                        // In this case, we need to ignore face 1 and face 3.
                        // If we are in the last row of face 0, we need to jump to the start of face 2
                        // If we are in the last row of face 2, we need to jump to the start of next tile (face 0 of
                        // next tile)
                        if (face == 0) {
                            uint32_t last_column_face_0 = (offset - (face_hw - face_size));
                            if (last_column_face_0 < (face_size)) {
                                offset += face_hw;
                            }
                            offset += face_size;
                        } else {
                            uint32_t last_column_face_2 = (offset - (face_hw * 3 - face_size));
                            if (last_column_face_2 < (face_size)) {
                                offset = 0;
                                curr_tile++;
                            } else {
                                offset += face_size;
                            }
                        }
#else
                        offset += face_size;
#endif
                    } else {
                        offset -= face_size * (face_size - 1);
                    }
                }
            } else if (face % 2 == 0) {
                offset += face_hw;
            } else {
                curr_tile++;
                offset -= face_hw;
                read_indices = true;
            }
        }
    }
    // dfb_in1 is reserved once as an index scratch buffer (no downstream consumer); commit the
    // reservation so the DFB is left balanced.
    dfb_in1.push_back(1);
}
