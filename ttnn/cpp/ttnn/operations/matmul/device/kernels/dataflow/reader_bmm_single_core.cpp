// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

// #include "debug/dprint.h"

#ifdef FUSE_BIAS
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/reader_bmm_single_core_bias.hpp"
#endif

void kernel_main() {
    bool one_time_profile = true;

    // in0 tensor args
    uint32_t in0_addr = get_arg_val<uint32_t>(0);
    uint32_t in0_num_blocks_h = get_arg_val<uint32_t>(1);
    uint32_t in0_num_blocks_w = get_arg_val<uint32_t>(2);
    uint32_t in0_stride_w = get_arg_val<uint32_t>(3);
    uint32_t in0_stride_h = get_arg_val<uint32_t>(4);
    uint32_t in0_next_block_stride = get_arg_val<uint32_t>(5);

    // in0 block args
    uint32_t in0_block_w = get_arg_val<uint32_t>(6);
    uint32_t in0_block_h = get_arg_val<uint32_t>(7);
    uint32_t in0_block_num_tiles = get_arg_val<uint32_t>(8);

    // in1 tensor args
    uint32_t in1_addr = get_arg_val<uint32_t>(9);
    uint32_t in1_num_blocks_w = get_arg_val<uint32_t>(10);
    uint32_t in1_start_tile_id = get_arg_val<uint32_t>(11);
    uint32_t in1_stride_w = get_arg_val<uint32_t>(12);
    uint32_t in1_stride_h = get_arg_val<uint32_t>(13);

    // in1 block args
    uint32_t in1_block_w = get_arg_val<uint32_t>(15);
    uint32_t in1_block_h = get_arg_val<uint32_t>(16);
    uint32_t in1_block_num_tiles = get_arg_val<uint32_t>(17);

    // in0 and in1 strides as number of tiles
    uint32_t in0_next_block_stride_h = get_arg_val<uint32_t>(18);
    uint32_t in0_next_block_stride_w = get_arg_val<uint32_t>(19);
    uint32_t in1_next_block_stride_h = get_arg_val<uint32_t>(20);
    uint32_t in1_next_block_stride_w = get_arg_val<uint32_t>(21);

    constexpr uint32_t in0_cb_id = tt::CB::c_in0;
    constexpr uint32_t in1_cb_id = tt::CB::c_in1;

    const uint32_t in0_tile_nbytes = get_tile_size(in0_cb_id);
    const uint32_t in1_tile_nbytes = get_tile_size(in1_cb_id);
    const DataFormat in0_df = get_dataformat(in0_cb_id);
    const DataFormat in1_df = get_dataformat(in1_cb_id);
    // const InterleavedAddrGenFast<true> s0 = {
    //     .bank_base_address = in0_addr,
    //     .page_size = in0_tile_nbytes,
    //     .data_format = in0_df
    // };
    // const InterleavedAddrGenFast<true> s1 = {
    //     .bank_base_address = in1_addr,
    //     .page_size = in1_tile_nbytes,
    //     .data_format = in1_df
    // };

    // constexpr uint32_t tile_size_pow2_exponent = 11;
    // const InterleavedPow2AddrGen<true> s0 = {
    //     .bank_base_address = in0_addr,
    //     .log_base_2_of_page_size = tile_size_pow2_exponent
    // };
    // const InterleavedPow2AddrGen<true> s1 = {
    //     .bank_base_address = in1_addr,
    //     .log_base_2_of_page_size = tile_size_pow2_exponent
    // };
    const InterleavedAddrGen<true> s0 = {.bank_base_address = in0_addr, .page_size = in0_tile_nbytes};
    const InterleavedAddrGen<true> s1 = {.bank_base_address = in1_addr, .page_size = in1_tile_nbytes};

// DPRINT << FIXED() << SETW(32) << SETPRECISION(2);

// read bias first if defined
#ifdef FUSE_BIAS
    constexpr uint32_t bias_cb_id = tt::CB::c_in2;
    uint32_t bias_addr = get_arg_val<uint32_t>(22);
    uint32_t bias_width_ntiles = get_arg_val<uint32_t>(23);
    uint32_t bias_log2_of_pagesize = get_arg_val<uint32_t>(24);
    uint32_t bias_pagesize = get_arg_val<uint32_t>(25);
    read_bias<true>(bias_addr, bias_width_ntiles, bias_cb_id, bias_log2_of_pagesize, bias_pagesize);
#endif

    uint32_t in0_start_tile_id = 0;
    // loop over in0 blocks along h
    for (uint32_t in0_block_h_i = 0; in0_block_h_i < in0_num_blocks_h; ++in0_block_h_i) {
        // Reset in1 start tile index
        uint32_t in1_start_tile_id = 0;
        // loop over in1 blocks along w
        for (uint32_t in1_block_w_i = 0; in1_block_w_i < in1_num_blocks_w; ++in1_block_w_i) {
            uint32_t in0_current_block_start_tile_id = in0_start_tile_id;
            uint32_t in1_current_block_start_tile_id = in1_start_tile_id;
            // loop over in0 blocks along w (in1 blocks along h)
            for (uint32_t in0_block_w_i = 0; in0_block_w_i < in0_num_blocks_w; ++in0_block_w_i) {
                // read in input data for current block
                // in0 DRAM -> L1 (activations in tiled form)
                // load block [in0_block_h_i, in0_block_w_i]
                cb_reserve_back(in0_cb_id, in0_block_num_tiles);
                uint32_t in0_write_l1_addr = get_write_ptr(in0_cb_id);
                uint32_t in0_row_start_tile_id =
                    in0_block_h_i * in0_next_block_stride_h + in0_block_w_i * in0_next_block_stride_w;
                // loop over in0 block tiles along h
                for (uint32_t in0_tile_h_i = 0; in0_tile_h_i < in0_block_h; ++in0_tile_h_i) {
                    uint32_t in0_tile_id = in0_row_start_tile_id;
                    // loop over in0 block tiles along w
                    for (uint32_t in0_tile_w_i = 0; in0_tile_w_i < in0_block_w; ++in0_tile_w_i) {
                        uint64_t in0_tile_noc_addr = get_noc_addr(in0_tile_id, s0);
                        noc_async_read(in0_tile_noc_addr, in0_write_l1_addr, in0_tile_nbytes);
                        in0_write_l1_addr += in0_tile_nbytes;
                        in0_tile_id += in0_stride_w;
                    }
                    in0_row_start_tile_id += in0_stride_h;
                }
                noc_async_read_barrier();
                in0_current_block_start_tile_id += in0_next_block_stride_w;
                cb_push_back(in0_cb_id, in0_block_num_tiles);

                // in1 DRAM -> L1 (weights in tiled form)
                cb_reserve_back(in1_cb_id, in1_block_num_tiles);
                uint32_t in1_write_l1_addr = get_write_ptr(in1_cb_id);
                uint32_t in1_row_start_tile_id =
                    in0_block_w_i * in1_next_block_stride_h + in1_block_w_i * in1_next_block_stride_w;
                // loop over in1 block tiles along h
                for (uint32_t in1_tile_h_i = 0; in1_tile_h_i < in1_block_h; ++in1_tile_h_i) {
                    uint32_t in1_tile_id = in1_row_start_tile_id;
                    // loop over in1 block tiles along w
                    for (uint32_t in1_tile_w_i = 0; in1_tile_w_i < in1_block_w; ++in1_tile_w_i) {
                        uint64_t in1_tile_noc_addr = get_noc_addr(in1_tile_id, s1);
                        noc_async_read(in1_tile_noc_addr, in1_write_l1_addr, in1_tile_nbytes);
                        in1_write_l1_addr += in1_tile_nbytes;
                        in1_tile_id += in1_stride_w;
                    }  // for in1_block_w
                    in1_row_start_tile_id += in1_stride_h;
                }  // for in1_block_h
                noc_async_read_barrier();
                in1_current_block_start_tile_id += in1_next_block_stride_h;  // in1_width_ntiles * in1_block_h
                cb_push_back(in1_cb_id, in1_block_num_tiles);
            }  // for in0_num_blocks_w
            in1_start_tile_id += in1_next_block_stride_w;  // in1_block_w
        }  // for in1_num_blocks_w
        in0_start_tile_id += in0_next_block_stride_h;
    }  // for in1_num_blocks_h
}  // kernel_main()
