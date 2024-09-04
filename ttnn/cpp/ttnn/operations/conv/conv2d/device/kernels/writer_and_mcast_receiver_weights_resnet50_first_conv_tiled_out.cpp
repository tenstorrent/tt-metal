// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"


void kernel_main() {
    uint32_t i = 0;
    uint32_t out_addr = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weight_addr_dram_base = get_arg_val<uint32_t>(i); i+=1;
    // Bias args. Unused if bias fusion is not enabled.
    const uint32_t bias_addr = get_arg_val<uint32_t>(i); i += 1;

    uint32_t out_next_tile_stride_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t out_next_tile_stride_w = get_arg_val<uint32_t>(i); i+=1;
    uint32_t out_next_subblock_stride_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t out_next_subblock_stride_w = get_arg_val<uint32_t>(i); i+=1;
    uint32_t out_next_block_stride_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t out_next_block_stride_w = get_arg_val<uint32_t>(i); i+=1;
    uint32_t out_subblock_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t out_subblock_w = get_arg_val<uint32_t>(i); i+=1;
    uint32_t out_subblock_tile_count = get_arg_val<uint32_t>(i); i+=1;
    uint32_t out_num_subblocks_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t out_num_subblocks_w = get_arg_val<uint32_t>(i); i+=1;
    uint32_t out_num_blocks_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t out_num_blocks_w = get_arg_val<uint32_t>(i); i+=1;
    uint32_t out_block_height_num_tiles = get_arg_val<uint32_t>(i); i+=1;
    uint32_t out_height_num_tiles = get_arg_val<uint32_t>(i); i+=1;
    uint32_t out_width_num_tiles = get_arg_val<uint32_t>(i); i+=1;
    uint32_t out_start_tile_id = get_arg_val<uint32_t>(i); i+=1;
    uint32_t out_start_tile_id_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t out_start_tile_id_w = get_arg_val<uint32_t>(i); i+=1;

    uint32_t num_blocks_weight_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weight_block_num_tiles = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weight_block_height_num_outer = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weight_block_height_ntiles = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weight_block_width_ntiles = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weight_stride_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weight_next_block_stride_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weight_next_block_stride_w = get_arg_val<uint32_t>(i); i+=1;

    // Bias args. Unused if bias fusion is not enabled.
    const uint32_t bias_ntiles = get_arg_val<uint32_t>(i); i += 1;
    const uint32_t bias_tile_offset = get_arg_val<uint32_t>(i); i += 1;

    uint32_t noop = get_arg_val<uint32_t>(i); i+=1;
    if(noop) {
        return;
    }

    // mcast args
    uint32_t weights_mcast_sender_noc_x           = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weights_mcast_sender_noc_y           = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weights_mcast_sender_semaphore_addr    = get_semaphore(get_arg_val<uint32_t>(i)); i+=1;
    uint32_t weights_mcast_receiver_semaphore_addr  = get_semaphore(get_arg_val<uint32_t>(i)); i+=1;


    constexpr bool out_in_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_weight = get_compile_time_arg_val(2);

    volatile tt_l1_ptr uint32_t* weights_mcast_receiver_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(weights_mcast_receiver_semaphore_addr);

    const uint32_t tile_nbytes = get_tile_size(cb_id_out0);
    const DataFormat out_df = get_dataformat(cb_id_out0);

    const InterleavedAddrGenFast<out_in_dram> s = {
        .bank_base_address = out_addr,
        .page_size = tile_nbytes,
        .data_format = out_df
    };

    // MCAST RECEIVE WEIGHTS
    // read weight blocks inner dim
    // read weight slice - 1 block of weights in width dim and full weight matrix height
    // read slice only once for all activation blocks
    for(uint32_t block_weight_h = 0; block_weight_h < num_blocks_weight_h; block_weight_h++) {
        cb_reserve_back(cb_id_weight, weight_block_num_tiles);
        // Set weights semaphore value to INVALID
        noc_semaphore_set(weights_mcast_receiver_semaphore_addr_ptr, INVALID);

        // Atomic increment source core counter
        uint64_t weights_mcast_sender_semaphore_noc_addr = get_noc_addr(weights_mcast_sender_noc_x, weights_mcast_sender_noc_y, weights_mcast_sender_semaphore_addr);
        noc_semaphore_inc(weights_mcast_sender_semaphore_noc_addr, 1);

        // wait on weights semaphore value to become VALID (set by mcast sender after it multicasts data)
        noc_semaphore_wait(weights_mcast_receiver_semaphore_addr_ptr, VALID);

        cb_push_back(cb_id_weight, weight_block_num_tiles);
    } // for num_blocks_weight_h

    // first read in bias if enabled (done only once for all blocks)
    #ifdef FUSE_BIAS
    constexpr uint32_t bias_cb_id = get_compile_time_arg_val(3);
    cb_reserve_back(bias_cb_id, bias_ntiles);

    // Set weights semaphore value to INVALID
    noc_semaphore_set(weights_mcast_receiver_semaphore_addr_ptr, INVALID);

    // Atomic increment source core counter
    uint64_t weights_mcast_sender_semaphore_noc_addr = get_noc_addr(weights_mcast_sender_noc_x, weights_mcast_sender_noc_y, weights_mcast_sender_semaphore_addr);
    noc_semaphore_inc(weights_mcast_sender_semaphore_noc_addr, 1);

    // wait on weights semaphore value to become VALID (set by mcast sender after it multicasts data)
    noc_semaphore_wait(weights_mcast_receiver_semaphore_addr_ptr, VALID);

    cb_push_back(bias_cb_id, bias_ntiles);
    #endif

    #ifndef SHARDED_OUT
    uint32_t out_block_h_start_tile_id = out_start_tile_id;
    uint32_t out_block_h_start_tile_id_h = out_start_tile_id_h;
    for(uint32_t bh = 0; bh < out_num_blocks_h; bh++) {
        uint32_t out_block_w_start_tile_id = out_block_h_start_tile_id;
        uint32_t out_block_w_start_tile_id_w = 0;
        for (uint32_t bw = 0; bw < out_num_blocks_w; bw++) {

            uint32_t out_sbh_start_tile_id = out_block_w_start_tile_id;
            uint32_t out_sbh_start_tile_id_h = out_block_h_start_tile_id_h;
            for(uint32_t sbh = 0; sbh < out_num_subblocks_h; sbh++) {
                uint32_t out_sbw_start_tile_id = out_sbh_start_tile_id;
                uint32_t out_sbw_start_tile_id_w = out_block_w_start_tile_id_w;
                for(uint32_t sbw = 0; sbw < out_num_subblocks_w; sbw++) {
                    uint32_t out_sb_row_start_tile_id = out_sbw_start_tile_id;
                    // wait for one subblock worth tiles
                    cb_wait_front(cb_id_out0, out_subblock_tile_count);
                    uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
                    for(uint32_t h = 0; h < out_subblock_h; h++) {
                        uint32_t out_tile_id = out_sb_row_start_tile_id;
                        uint32_t out_tile_id_h = out_sbh_start_tile_id_h + h;
                        if (out_tile_id_h >= out_height_num_tiles) { // block shape height padding
                            break;
                        }
                        for(uint32_t w = 0; w < out_subblock_w; w++) {
                            uint32_t out_tile_id_w = out_sbw_start_tile_id_w + w;
                            if (out_tile_id_w >= out_width_num_tiles) { // block shape width padding
                                l1_read_addr += tile_nbytes;
                            } else {
                                //DPRINT << "out_tile_id - " << out_tile_id << ENDL();
                                s.noc_async_write_tile(out_tile_id, l1_read_addr);
                                l1_read_addr += tile_nbytes;
                                out_tile_id += out_next_tile_stride_w;
                            }
                        } // out_subblock_w (ntiles)
                        out_sb_row_start_tile_id += out_next_tile_stride_h;
                    } // out_subblock_h (ntiles)
                    noc_async_write_barrier();
                    //DPRINT << "Done writing subblock." << ENDL();
                    cb_pop_front(cb_id_out0, out_subblock_tile_count);
                    out_sbw_start_tile_id += out_next_subblock_stride_w;
                    out_sbw_start_tile_id_w += out_subblock_w;
                } // out_num_subblocks_w
                out_sbh_start_tile_id += out_next_subblock_stride_h;
                out_sbh_start_tile_id_h += out_subblock_h;
            } // out_num_subblocks_h
            out_block_w_start_tile_id += out_next_block_stride_w;
            out_block_w_start_tile_id_w += weight_block_width_ntiles;
        } // out_num_blocks_w
        out_block_h_start_tile_id += out_next_block_stride_h;
        out_block_h_start_tile_id_h += out_block_height_num_tiles;
    } // out_num_blocks_h

    #else
    cb_wait_front(cb_id_out0, out_subblock_tile_count * out_num_subblocks_h * out_num_subblocks_w * out_num_blocks_w * out_num_blocks_h);
    #endif
}
