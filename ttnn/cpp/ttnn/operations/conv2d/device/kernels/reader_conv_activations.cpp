// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
// #include "debug/dprint.h"

inline void pad_l1_buffer_with_zeroes(uint32_t l1_addr, uint32_t pad_size_bytes) {
    volatile std::uint32_t* dst = reinterpret_cast<volatile std::uint32_t*>(l1_addr);
    volatile std::uint32_t* end_dst = dst + (pad_size_bytes >> 2);  // Divide by 4 using right shift

    while (dst < end_dst) {
        *dst++ = 0;
    }

    uint32_t remainder = pad_size_bytes & 0x3;  // Get the remainder using bitwise AND
    if (remainder != 0) {
        volatile std::uint8_t* byte_dst = reinterpret_cast<volatile std::uint8_t*>(dst);
        for (uint32_t i = 0; i < remainder; ++i) {
            *byte_dst++ = 0;
        }
    }
}

void kernel_main() {
    uint32_t i = 0;
    uint32_t act_addr_dram_base  = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_dram_noc_x = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_dram_noc_y = get_arg_val<uint32_t>(i); i+=1;

    uint32_t conv_act_size_w = get_arg_val<uint32_t>(i); i+=1;
    uint32_t conv_act_size_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t conv_act_size_c = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weight_size_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weight_size_w = get_arg_val<uint32_t>(i); i+=1;
    uint32_t stride_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t stride_w = get_arg_val<uint32_t>(i); i+=1;
    uint32_t pad_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t pad_w = get_arg_val<uint32_t>(i); i+=1;
    uint32_t conv_output_size_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t conv_output_size_w = get_arg_val<uint32_t>(i); i+=1;
    uint32_t num_blocks_act_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t num_blocks_act_w = get_arg_val<uint32_t>(i); i+=1;
    uint32_t num_blocks_weight_w = get_arg_val<uint32_t>(i); i+=1;
    uint32_t num_groups = get_arg_val<uint32_t>(i); i+=1;

    uint32_t act_matrix_height_unpadded = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_matrix_width_unpadded = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_matrix_height = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_matrix_width = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_matrix_height_ntiles = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_matrix_width_ntiles = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_block_h_datums = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_block_w_datums = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_block_h_ntiles = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_block_w_ntiles = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_block_num_tiles = get_arg_val<uint32_t>(i); i+=1;
    uint32_t src_dram_act_buffer_size_bytes = get_arg_val<uint32_t>(i); i+=1;
    uint32_t dst_l1_act_buffer_size_bytes = get_arg_val<uint32_t>(i); i+=1;

    constexpr bool act_in_dram = get_compile_time_arg_val(0) == 1;

    constexpr uint32_t cb_id_act = 0;
    constexpr uint32_t tile_size_pow2_exponent = 11;
    const DataFormat data_format = get_dataformat(cb_id_act);
    uint32_t channel_stick_size = conv_act_size_c;
    uint32_t channel_stick_size_bytes = channel_stick_size << 1;
    const InterleavedAddrGen<act_in_dram> s_act = {
        .bank_base_address = act_addr_dram_base,
        .page_size = channel_stick_size_bytes
    };

    for(uint32_t group_idx = 0; group_idx < num_groups; group_idx++) {

        // Read activations for this group
        // Activations are in channels last layout in dram
        {
            cb_reserve_back(cb_id_act, act_block_num_tiles);
            uint32_t block_idx_h = (uint32_t) (group_idx / num_blocks_act_w) / (num_blocks_weight_w);
            uint32_t block_idx_w = (uint32_t) (group_idx % num_blocks_act_w);
            uint32_t block_idx = (block_idx_h * num_blocks_act_w) + block_idx_w;
            uint32_t start_block_2d_index_h = block_idx_h * act_block_h_datums;
            uint32_t start_block_2d_index_w = block_idx_w * act_block_w_datums;
            uint32_t start_block_2d_index = (start_block_2d_index_h * act_block_w_datums * num_blocks_act_w) + start_block_2d_index_w;
            uint32_t l1_write_addr_act = get_write_ptr(cb_id_act);
            // TODO (nshanker): add macro to disable checks
            if(start_block_2d_index_w >= act_matrix_width_unpadded) {
                //DPRINT << "Problem" << ENDL();
            }
            for(uint32_t h_b = 0; h_b < act_block_h_datums; h_b++) {
                uint32_t h = start_block_2d_index_h + h_b;
                uint32_t dst_address_offset_l1 = (h_b * act_block_w_datums)<<1;
                if (h >= act_matrix_height_unpadded) {
                    // pad (block shape padding for height dim)
                    uint32_t pad_size_bytes = act_block_w_datums<<1;
                    // TODO (nshanker): add macro to disable checks
                    if(dst_address_offset_l1 + (pad_size_bytes-1) >= dst_l1_act_buffer_size_bytes) {
                        //DPRINT << "Problem" << ENDL();
                    }
                    uint32_t dst_addr = l1_write_addr_act + dst_address_offset_l1;
                    pad_l1_buffer_with_zeroes(dst_addr, pad_size_bytes);
                }
                else {
                    uint32_t w = start_block_2d_index_w;
                    uint32_t end_block_2d_index_w = start_block_2d_index_w + act_block_w_datums - 1;
                    // TODO (nshanker): add macro to disable checks
                    if(end_block_2d_index_w >= act_matrix_width) {
                        //DPRINT << "Problem" << ENDL();
                    }
                    while (w <= end_block_2d_index_w) {
                        uint32_t src_address_offset_dram = 0;
                        uint32_t read_size_bytes = 0;
                        uint32_t pad = 0;
                        if (w >= act_matrix_width_unpadded) {
                            // pad (block shape padding for width dim)
                            // TODO (nshanker): add macro to disable checks
                            if(end_block_2d_index_w != act_matrix_width-1) {
                                //DPRINT << "Problem" << ENDL();
                            }
                            uint32_t pad_size_bytes = (end_block_2d_index_w - w + 1)<<1;
                            if(dst_address_offset_l1 + (pad_size_bytes-1) >= dst_l1_act_buffer_size_bytes) {
                                //DPRINT << "Problem" << ENDL();
                            }
                            uint32_t dst_addr = l1_write_addr_act + dst_address_offset_l1;
                            pad_l1_buffer_with_zeroes(dst_addr, pad_size_bytes);
                            read_size_bytes = pad_size_bytes;
                        }
                        else {
                            uint32_t channel_stick_offset = w % channel_stick_size;
                            uint32_t channel_stick_col_id = w / channel_stick_size;
                            uint32_t channel_stick_row_id = h;
                            if(channel_stick_offset % 16 != 0) { // DRAM read address must be aligned to 32 bytes
                                //DPRINT << "Problem" << ENDL();
                            }
                            uint32_t channel_stick_row_id_x = channel_stick_row_id % conv_output_size_w;
                            uint32_t channel_stick_row_id_y = channel_stick_row_id / conv_output_size_w;
                            uint32_t act_tensor_start_x = channel_stick_row_id_x * stride_w;
                            uint32_t act_tensor_start_y = channel_stick_row_id_y * stride_h;
                            uint32_t act_tensor_padded_x = act_tensor_start_x + (channel_stick_col_id % weight_size_w);
                            uint32_t act_tensor_padded_y = act_tensor_start_y + (channel_stick_col_id / weight_size_w);
                            if(w > end_block_2d_index_w) {
                                //DPRINT << "Problem" << ENDL();
                            }
                            uint32_t a = channel_stick_size - channel_stick_offset;
                            uint32_t b = (end_block_2d_index_w+1)-w;
                            uint32_t read_size = a < b ? a : b;
                            read_size_bytes = read_size << 1;
                            if(act_tensor_padded_x < pad_w || act_tensor_padded_x >= (pad_w + conv_act_size_w) || act_tensor_padded_y < pad_h || act_tensor_padded_y >= (pad_h + conv_act_size_h)) {
                                // pad (conv padding)
                                uint32_t dst_addr = l1_write_addr_act + dst_address_offset_l1;
                                uint32_t pad_size_bytes = read_size_bytes;
                                if(dst_address_offset_l1 + (pad_size_bytes-1) >= dst_l1_act_buffer_size_bytes) {
                                    //DPRINT << "Problem" << ENDL();
                                }
                                pad_l1_buffer_with_zeroes(dst_addr, pad_size_bytes);
                            }
                            else {
                                uint32_t act_tensor_x = act_tensor_padded_x - pad_w;
                                uint32_t act_tensor_y = act_tensor_padded_y - pad_h;
                                if(act_tensor_x >= conv_act_size_w || act_tensor_y >= conv_act_size_h) {
                                    //DPRINT << "Problem" << ENDL();
                                }
                                uint32_t act_tensor_channel_id = act_tensor_y * conv_act_size_w + act_tensor_x;
                                src_address_offset_dram = ((act_tensor_channel_id * channel_stick_size) + channel_stick_offset)<<1;
                                if(src_address_offset_dram % 32 != 0) { // DRAM read address must be aligned to 32 bytes
                                    //DPRINT << "Problem1" << ENDL();
                                }
                                if(src_address_offset_dram >= src_dram_act_buffer_size_bytes) {
                                    //DPRINT << "Problem2" << ENDL();
                                }
                                if(dst_address_offset_l1 + (read_size_bytes-1) >= dst_l1_act_buffer_size_bytes) {
                                    //DPRINT << "Problem3" << ENDL();
                                }
                                uint32_t src_addr = act_addr_dram_base + src_address_offset_dram;
                                uint32_t dst_addr = l1_write_addr_act + dst_address_offset_l1;
                                uint64_t act_noc_addr = get_noc_addr(act_tensor_channel_id, s_act, (channel_stick_offset<<1));
                                noc_async_read(act_noc_addr, dst_addr, read_size_bytes);
                            }
                        }
                        dst_address_offset_l1 += read_size_bytes;
                        w += (read_size_bytes>>1);
                        if(w > end_block_2d_index_w+1) {
                            //DPRINT << "Problem" << ENDL();
                        }
                    }
                }
            }
        }

        noc_async_read_barrier();
        cb_push_back(cb_id_act, act_block_num_tiles);
    }

}
