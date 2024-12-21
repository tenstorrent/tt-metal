// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    const std::uint32_t input_dram_buffer_src_addr = get_arg_val<uint32_t>(0);
    const std::uint32_t weights_dram_buffer_src_addr = get_arg_val<uint32_t>(1);
    const std::uint32_t tile_offset = get_arg_val<uint32_t>(2);
    const std::uint32_t face_offset = get_arg_val<uint32_t>(3);
    const std::uint32_t num_rows = get_arg_val<uint32_t>(4);

    const std::uint32_t curr_col = get_arg_val<uint32_t>(5);

#define in_is_dram get_compile_time_arg_val(0) == 1
#define in_stick_size_is_power_of_two get_compile_time_arg_val(1) == 1
    constexpr uint32_t input_page_size = get_compile_time_arg_val(2);
#if (in_stick_size_is_power_of_two)
    constexpr uint32_t log_base_2_of_input_page_size = get_compile_time_arg_val(3);
    const InterleavedPow2AddrGen<in_is_dram> input = {
        .bank_base_address = input_dram_buffer_src_addr,
        .log_base_2_of_page_size = log_base_2_of_input_page_size  // TODO(AP): refactor
    };
#else
    const InterleavedAddrGen<in_is_dram> input = {
        .bank_base_address = input_dram_buffer_src_addr, .page_size = input_page_size};
#endif

#define weights_is_dram get_compile_time_arg_val(4) == 1
#define weight_stick_size_is_power_of_two get_compile_time_arg_val(5) == 1
    constexpr uint32_t weight_stick_size = get_compile_time_arg_val(6);
#if (weight_stick_size_is_power_of_two)
    constexpr uint32_t log_base_2_of_weights_page_size = get_compile_time_arg_val(7);
    const InterleavedPow2AddrGen<weights_is_dram> weights = {
        .bank_base_address = weights_dram_buffer_src_addr,
        .log_base_2_of_page_size = log_base_2_of_weights_page_size  // TODO(AP): refactor
    };
#else
    const InterleavedAddrGen<weights_is_dram> weights = {
        .bank_base_address = weights_dram_buffer_src_addr, .page_size = weight_stick_size};
#endif

    constexpr uint32_t row_length = get_compile_time_arg_val(8);
    constexpr uint32_t input_block_size_bytes = get_compile_time_arg_val(9);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t cb_id_in2 = 2;

    constexpr uint32_t face_size = 16;
    constexpr uint32_t tile_height = 32;

#if defined PADDED
    const std::uint32_t pad_token = get_arg_val<uint32_t>(6);
    uint64_t pad_noc_addr;
    {
        cb_reserve_back(cb_id_in2, 1);
        uint32_t local_pad_addr = get_write_ptr(cb_id_in2);
        uint64_t src_noc_addr = get_noc_addr(pad_token, weights);
        noc_async_read(src_noc_addr, local_pad_addr, weight_stick_size);
        noc_async_read_barrier();
        pad_noc_addr = get_noc_addr(local_pad_addr);
    }
#elif defined BINARY
    uint64_t zero_noc_addr, one_noc_addr;
    {
        cb_reserve_back(cb_id_in2, 2);
        uint32_t local_write_addr = get_write_ptr(cb_id_in2);
        uint64_t src_noc_addr = get_noc_addr(0, weights);
        noc_async_read(src_noc_addr, local_write_addr, weight_stick_size);
        zero_noc_addr = get_noc_addr(local_write_addr);

        local_write_addr += weight_stick_size;
        src_noc_addr = get_noc_addr(1, weights);
        noc_async_read(src_noc_addr, local_write_addr, weight_stick_size);
        one_noc_addr = get_noc_addr(local_write_addr);

        noc_async_read_barrier();
    }
#endif

    cb_reserve_back(cb_id_in1, 1);
    uint32_t input_l1_addr = get_write_ptr(cb_id_in1);
    const uint32_t tile_size_bytes = get_tile_size(cb_id_in1);
    const DataFormat data_format = get_dataformat(cb_id_in1);
#if defined BFP16
    volatile tt_l1_ptr uint16_t* input_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(input_l1_addr);
#else
    volatile tt_l1_ptr uint32_t* input_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(input_l1_addr);
#endif
    // bool printed = false;
    auto read_block = [&](const uint32_t& token_idx, const uint32_t& width_size, const uint32_t& offset = 0) {
        cb_reserve_back(cb_id_in0, 1);
        uint32_t weight_l1_addr = get_write_ptr(cb_id_in0);
        volatile tt_l1_ptr uint16_t* weight_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(weight_l1_addr);
        uint64_t src_noc_addr;
        uint32_t token = input_l1_ptr[token_idx + offset];

        // I would like to keep this for future debugging purposes
        // This prints which token is being read, as well as one full tile per core
        // DPRINT << "Token: " << token << ENDL();

        // if (!printed) {
        //     for (int r_f = 0; r_f < 2; r_f++) {
        //         for (int i = 0; i < 105; i++) {
        //             DPRINT << "_";
        //         }
        //         DPRINT << ENDL();
        //         uint32_t r_f_offset = r_f * 2 * face_size * face_size;
        //         for (int r = 0; r < face_size; r++) {
        //             uint32_t r_offset = r * face_size;
        //             DPRINT << "[";
        //             for (int i = 0; i < face_size; i++) {
        //                 DPRINT << input_l1_ptr[i + r_f_offset + r_offset] << ", ";
        //             }
        //             DPRINT << "] | ";
        //             DPRINT << "[";
        //             for (int i = 0; i < face_size; i++) {
        //                 DPRINT << input_l1_ptr[i + r_f_offset + r_offset + (face_size * face_size)] << ", ";
        //             }
        //             DPRINT << "]" << ENDL();
        //         }
        //         for (int i = 0; i < 105; i++) {
        //             DPRINT << "_";
        //         }
        //         DPRINT << ENDL();
        //     }
        //     printed = true;
        // }
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
        // This prints the weight stick we are reading to the write location
        // DPRINT << "Weight: [" << ENDL();
        // for (uint32_t i = 0; i < weight_stick_size / 2; i++) {
        //     DPRINT << BF16(weight_l1_ptr[i]) << ", ";
        // }
        // DPRINT << "]" << ENDL();
        cb_push_back(cb_id_in0, 1);
    };

    uint32_t curr_tile = tile_offset;
    uint32_t offset = face_offset;
    uint32_t index = 0;
    bool read_indices = true;
    uint32_t col_offset = curr_col;
    uint32_t tiles_per_row = (row_length + tile_height - 1) / tile_height;
    const InterleavedAddrGenFast<true> s = {
        .bank_base_address = input_dram_buffer_src_addr,
        .page_size = tile_size_bytes,
        .data_format = data_format,
    };

    // DPRINT << "Starting tile: " << curr_tile << " offset: " << offset << ENDL();
    for (uint32_t i = 0; i < num_rows; ++i) {
        if (read_indices) {
            uint64_t noc_input_src_addr = get_noc_addr(curr_tile, input) + (offset * sizeof(uint32_t));
            // noc_async_read(noc_input_src_addr, input_l1_addr, 32*32*4);
            noc_async_read_tile(curr_tile, s, input_l1_addr);
            noc_async_read_barrier();
            read_indices = false;
        }
        read_block(index, weight_stick_size, offset);
        index++;
        col_offset++;
        if (index == face_size) {
            // DPRINT << "Read 16, offset: " << offset << ENDL();
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
                    offset -= face_size * (face_size - 1);
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
