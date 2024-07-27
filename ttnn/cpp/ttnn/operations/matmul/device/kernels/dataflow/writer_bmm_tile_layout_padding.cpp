// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    // out tensor args
    uint32_t out_tensor_addr = get_arg_val<uint32_t>(0);
    uint32_t out_tensor_start_tile_id = get_arg_val<uint32_t>(1);
    uint32_t out_tensor_stride_w = get_arg_val<uint32_t>(2);
    uint32_t out_tensor_stride_h = get_arg_val<uint32_t>(3);
    uint32_t out_tensor_next_subblock_stride_w = get_arg_val<uint32_t>(4);
    uint32_t out_tensor_next_subblock_stride_h = get_arg_val<uint32_t>(5);

    // out subblock args
    uint32_t out_subblock_w = get_arg_val<uint32_t>(6);
    uint32_t out_subblock_h = get_arg_val<uint32_t>(7);
    uint32_t out_subblock_tile_count = get_arg_val<uint32_t>(8);
    uint32_t out_num_subblocks_w = get_arg_val<uint32_t>(9);
    uint32_t out_num_subblocks_h = get_arg_val<uint32_t>(10);

    // batch args
    uint32_t MtNt = get_arg_val<uint32_t>(11);  // if 0
    uint32_t batch = get_arg_val<uint32_t>(12);

    // padding args
    uint32_t out_num_nonzero_subblocks_h = get_arg_val<uint32_t>(13);
    uint32_t out_last_subblock_h = get_arg_val<uint32_t>(14);
    uint32_t padded_block_tiles_h_skip = get_arg_val<uint32_t>(15);
    uint32_t out_num_nonzero_subblocks_w = get_arg_val<uint32_t>(16);
    uint32_t out_last_subblock_w = get_arg_val<uint32_t>(17);
    uint32_t padded_subblock_tiles_addr_skip = get_arg_val<uint32_t>(18);
    uint32_t padded_block_tiles_w_skip = get_arg_val<uint32_t>(19);

    constexpr bool out_is_dram = get_compile_time_arg_val(0) == 1;

    constexpr uint32_t cb_id_out0 = 16;

    // single-tile
    const uint32_t single_tile_size_bytes = get_tile_size(cb_id_out0);
    const DataFormat data_format = get_dataformat(cb_id_out0);

    const InterleavedAddrGenFast<out_is_dram> s = {
        .bank_base_address = out_tensor_addr, .page_size = single_tile_size_bytes, .data_format = data_format};

    bool one_time_profile = true;
    for (uint32_t b = 0; b < batch; b++) {
        uint32_t out_tensor_sbh_start_tile_id = out_tensor_start_tile_id;
        for (uint32_t sbh = 0; sbh < out_num_nonzero_subblocks_h; sbh++) {
            uint32_t out_tensor_sbw_start_tile_id = out_tensor_sbh_start_tile_id;
            for (uint32_t sbw = 0; sbw < out_num_nonzero_subblocks_w; sbw++) {
                uint32_t out_tensor_sb_row_start_tile_id = out_tensor_sbw_start_tile_id;

                uint32_t out_subblock_h_ = out_subblock_h;
                uint32_t out_subblock_w_ = out_subblock_w;
                uint32_t subblock_tiles_addr_skip = 0;
                if (sbh == out_num_nonzero_subblocks_h - 1) {
                    out_subblock_h_ = out_last_subblock_h;
                }
                if (sbw == out_num_nonzero_subblocks_w - 1) {
                    out_subblock_w_ = out_last_subblock_w;
                    subblock_tiles_addr_skip = padded_subblock_tiles_addr_skip;
                }

                cb_wait_front(cb_id_out0, out_subblock_tile_count);
                uint32_t l1_read_addr = get_read_ptr(cb_id_out0);

                for (uint32_t h = 0; h < out_subblock_h_; h++) {
                    uint32_t out_tensor_tile_id = out_tensor_sb_row_start_tile_id;
                    for (uint32_t w = 0; w < out_subblock_w_; w++) {
                        noc_async_write_tile(out_tensor_tile_id, s, l1_read_addr);
                        l1_read_addr += single_tile_size_bytes;

                        out_tensor_tile_id += out_tensor_stride_w;
                    }
                    // Skip padded tiles in subblock along row
                    l1_read_addr += subblock_tiles_addr_skip;
                    out_tensor_sb_row_start_tile_id += out_tensor_stride_h;
                }

                noc_async_write_barrier();
                cb_pop_front(cb_id_out0, out_subblock_tile_count);
                out_tensor_sbw_start_tile_id += out_tensor_next_subblock_stride_w;
            }
            // Pop fully padded subblocks along the row
            cb_wait_front(cb_id_out0, padded_block_tiles_w_skip);
            cb_pop_front(cb_id_out0, padded_block_tiles_w_skip);
            out_tensor_sbh_start_tile_id += out_tensor_next_subblock_stride_h;
        }
        // Pop row(s) of fully padded subblocks
        cb_wait_front(cb_id_out0, padded_block_tiles_h_skip);
        cb_pop_front(cb_id_out0, padded_block_tiles_h_skip);
        out_tensor_start_tile_id += MtNt;
    }
}
