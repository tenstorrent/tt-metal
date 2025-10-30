// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "ttnn/operations/embedding/device/kernels/dataflow/embeddings_common.hpp"
#include "debug/dprint.h"

void kernel_main() {
    const std::uint32_t input_buffer_src_addr = get_arg_val<uint32_t>(0);
    const std::uint32_t weight_buffer_src_addr = get_arg_val<uint32_t>(1);
    const std::uint32_t tile_offset = get_arg_val<uint32_t>(2);
    const std::uint32_t face_offset = get_arg_val<uint32_t>(3);
    const std::uint32_t num_rows = get_arg_val<uint32_t>(4);

    const std::uint32_t curr_col = get_arg_val<uint32_t>(5);
    const std::uint32_t starting_index = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_in2 = get_compile_time_arg_val(2);

    constexpr uint32_t input_page_size = get_compile_time_arg_val(3);
    constexpr uint32_t weight_stick_size = get_compile_time_arg_val(4);
    constexpr uint32_t row_length = get_compile_time_arg_val(5);
    constexpr uint32_t input_block_size_bytes = get_compile_time_arg_val(6);

    constexpr auto input_args = TensorAccessorArgs<7>();
    constexpr auto weights_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    const auto input = TensorAccessor(input_args, input_buffer_src_addr, input_page_size);
    const auto weights = TensorAccessor(weights_args, weight_buffer_src_addr, weight_stick_size);

    constexpr uint32_t face_size = 16;
    constexpr uint32_t tile_height = 32;

    prepare_local_cache(cb_id_in2, weights, weight_stick_size, /*pad_token_arg_idx=*/6);

    cb_reserve_back(cb_id_in1, 1);
    uint32_t input_l1_addr = get_write_ptr(cb_id_in1);
    const uint32_t tile_size_bytes = get_tile_size(cb_id_in1);
    volatile tt_l1_ptr input_token_t* input_l1_ptr = reinterpret_cast<volatile tt_l1_ptr input_token_t*>(input_l1_addr);

    auto read_block = [&](const uint32_t& token_idx, const uint32_t& width_size, const uint32_t& offset = 0) {
        cb_reserve_back(cb_id_in0, 1);
        uint32_t weight_l1_addr = get_write_ptr(cb_id_in0);
        volatile tt_l1_ptr uint16_t* weight_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(weight_l1_addr);
        uint64_t src_noc_addr;
        uint32_t token = input_l1_ptr[token_idx + offset];

#if defined PADDED
        if (token == pad_token) {
            src_noc_addr = pad_noc_addr;
        } else {
            src_noc_addr = get_noc_addr(token, weights);
        }
#elif defined BINARY
        if (token == 0) {
            src_noc_addr = zero_noc_addr;
        } else {
            src_noc_addr = one_noc_addr;
        }
#else
#if defined BFP16
        union {
            float f;
            uint32_t u;
        } u;
        u.u = (uint32_t)input_l1_ptr[token_idx] << 16;
        uint32_t token_casted = static_cast<uint32_t>(u.f);
        src_noc_addr = get_noc_addr(token_casted, weights);
#else
        src_noc_addr = get_noc_addr(token, weights);
#endif
#endif
        noc_async_read(src_noc_addr, weight_l1_addr, width_size);
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, 1);
    };

    uint32_t curr_tile = tile_offset;
    uint32_t offset = face_offset;
    uint32_t index = starting_index;
    bool read_indices = true;
    uint32_t col_offset = curr_col;
    uint32_t tiles_per_row = (row_length + tile_height - 1) / tile_height;
    const auto s = TensorAccessor(input_args, input_buffer_src_addr, tile_size_bytes);

    for (uint32_t i = 0; i < num_rows; ++i) {
        if (read_indices) {
            uint64_t noc_input_src_addr = get_noc_addr(curr_tile, input) + (offset * sizeof(uint32_t));
            noc_async_read_tile(curr_tile, s, input_l1_addr);
            noc_async_read_barrier();
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
                    if (offset < 256) {
                        offset += face_size;
                    } else {
                        offset -= face_size * (face_size - 1);
                    }
                }
            } else if (face % 2 == 0) {
                offset += face_size * face_size;
            } else {
                curr_tile++;
                offset -= face_size * face_size;
                read_indices = true;
            }
        }
    }
}
