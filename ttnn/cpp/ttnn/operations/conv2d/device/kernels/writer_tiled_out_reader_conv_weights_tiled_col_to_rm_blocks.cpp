// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

// #include "debug/dprint.h"

#ifdef FUSE_BIAS
    #include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/reader_bmm_single_core_bias.hpp"
#endif

inline void read_weight_blocks_inner_h_dim(uint32_t cb_id_weight,
                        uint32_t num_blocks_weight_h,
                        uint32_t weight_block_num_tiles,
                        uint32_t weight_start_tile_id,
                        uint32_t weight_block_height_ntiles,
                        uint32_t weight_block_width_ntiles,
                        const InterleavedPow2AddrGen<true>& s_weight,
                        uint32_t weight_tile_nbytes,
                        uint32_t weight_stride_h,
                        uint32_t weight_next_block_stride_h) {
    // weight DRAM -> L1 (weights in tiled form)
    uint32_t weight_current_block_start_tile_id = weight_start_tile_id;
    for(uint32_t block_weight_h = 0; block_weight_h < num_blocks_weight_h; block_weight_h++) {
        cb_reserve_back(cb_id_weight, weight_block_num_tiles);
        uint32_t weight_write_l1_addr = get_write_ptr(cb_id_weight);
        uint32_t weight_row_start_tile_id = weight_current_block_start_tile_id;
        // loop over weight block tiles along h
        for(uint32_t weight_tile_h_i = 0; weight_tile_h_i < weight_block_height_ntiles; ++weight_tile_h_i) {
            uint32_t weight_tile_id = weight_row_start_tile_id;
            // loop over weight block tiles along w
            for(uint32_t weight_tile_w_i = 0; weight_tile_w_i < weight_block_width_ntiles; ++weight_tile_w_i) {
                uint64_t weight_tile_noc_addr = get_noc_addr(weight_tile_id, s_weight);
                //DPRINT << "weight_tile_id=" << weight_tile_id << ENDL();
                noc_async_read(weight_tile_noc_addr, weight_write_l1_addr, weight_tile_nbytes);
                weight_write_l1_addr += weight_tile_nbytes;
                weight_tile_id += 1;
            } // for weight_block_w
            weight_row_start_tile_id += weight_stride_h;
        } // for weight_block_h
        noc_async_read_barrier();
        weight_current_block_start_tile_id += weight_next_block_stride_h;
        cb_push_back(cb_id_weight, weight_block_num_tiles);
    } // for num_blocks_weight_h
}

void kernel_main() {
    // This writer is for output tensor in tile format
    uint32_t i = 0;
    uint32_t out_addr = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weight_addr_dram_base = get_arg_val<uint32_t>(i); i+=1;
    // Bias arg. Unused if bias fusion is not enabled.
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
    uint32_t weight_block_height_ntiles = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weight_block_width_ntiles = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weight_stride_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weight_next_block_stride_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weight_next_block_stride_w = get_arg_val<uint32_t>(i); i+=1;

    // Bias arg. Unused if bias fusion is not enabled.
    const uint32_t bias_ntiles = get_arg_val<uint32_t>(i); i += 1;
    const uint32_t bias_tile_offset = get_arg_val<uint32_t>(i); i += 1;

    uint32_t noop = get_arg_val<uint32_t>(i); i+=1;
    if(noop) {
        return;
    }

    constexpr bool out_in_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_weight = get_compile_time_arg_val(2);

    const uint32_t tile_nbytes = get_tile_size(cb_id_out0);
    const DataFormat out_df = get_dataformat(cb_id_out0);

    constexpr uint32_t tile_size_pow2_exponent = 11;    // == 2^11 = 2048 = 2 * 32 * 32 (assuming dtype = 2 bytes)
    const InterleavedPow2AddrGen<out_in_dram> s = {
        .bank_base_address = out_addr,
        .log_base_2_of_page_size = tile_size_pow2_exponent
    };

        // first read in bias if enabled (done only once for all batches)
    #ifdef FUSE_BIAS

        constexpr uint32_t bias_cb_id = get_compile_time_arg_val(3);
        constexpr uint32_t bias_log2_of_pagesize = get_compile_time_arg_val(4);
        constexpr uint32_t bias_pagesize = get_compile_time_arg_val(5);
        constexpr uint32_t bias_in_dram = get_compile_time_arg_val(6) == 1;

        read_bias_with_offset<bias_in_dram>(bias_addr, bias_tile_offset, bias_ntiles, bias_cb_id, bias_log2_of_pagesize, bias_pagesize);
    #endif

    // DPRINT << "tile_nbytes - " << tile_nbytes << ENDL();
    // DPRINT << "out_num_blocks_h - " << out_num_blocks_h << ENDL();
    // DPRINT << "out_num_blocks_w - " << out_num_blocks_w << ENDL();

    // DPRINT << "out_num_subblocks_h - " << out_num_subblocks_h << ENDL();
    // DPRINT << "out_num_subblocks_w - " << out_num_subblocks_w << ENDL();

    // DPRINT << "out_subblock_h - " << out_subblock_h << ENDL();
    // DPRINT << "out_subblock_w - " << out_subblock_w << ENDL();

    // DPRINT << "out_subblock_tile_count - " << out_subblock_tile_count << ENDL();

    // DPRINT << "num_blocks_weight_h - " << num_blocks_weight_h << ENDL();
    // DPRINT << "weight_block_height_ntiles - " << weight_block_height_ntiles << ENDL();
    // DPRINT << "weight_block_width_ntiles - " << weight_block_width_ntiles << ENDL();

    // DPRINT << "out_subblock_h - " << out_subblock_h << ENDL();
    // DPRINT << "out_subblock_w - " << out_subblock_w << ENDL();
    // DPRINT << "out_block_height_num_tiles - " << out_block_height_num_tiles << ENDL();
    // DPRINT << "out_height_num_tiles - " << out_height_num_tiles << ENDL();
    // DPRINT << "out_width_num_tiles - " << out_width_num_tiles << ENDL();

    const uint32_t weight_tile_nbytes = get_tile_size(cb_id_weight);
    const InterleavedPow2AddrGen<true> s_weight = {
        .bank_base_address = weight_addr_dram_base,
        .log_base_2_of_page_size = tile_size_pow2_exponent
    };

    // const InterleavedAddrGenFast<true> s = {
    //     .bank_base_address = out_addr,
    //     .page_size = tile_nbytes,
    //     .data_format = out_df
    // };

    // OUTER most loop is looping over out blocks in width dim because blocks from compute are in col major order.
    // Write out col major blocks in row major layout to output
    uint32_t out_block_w_start_tile_id = out_start_tile_id;
    //DPRINT << "out_start_tile_id=" << out_start_tile_id << ENDL();
    uint32_t out_block_w_start_tile_id_w = out_start_tile_id_w;
    uint32_t weight_start_tile_id = out_start_tile_id_w;
    //DPRINT << "weight_start_tile_id=" << weight_start_tile_id << ENDL();
    for (uint32_t bw = 0; bw < out_num_blocks_w; bw++) {
        uint32_t out_block_h_start_tile_id = out_block_w_start_tile_id;
        uint32_t out_block_h_start_tile_id_h = out_start_tile_id_h;
        for(uint32_t bh = 0; bh < out_num_blocks_h; bh++) {
            // read weight blocks inner dim
            read_weight_blocks_inner_h_dim(cb_id_weight,
                    num_blocks_weight_h,
                    weight_block_num_tiles,
                    weight_start_tile_id,
                    weight_block_height_ntiles,
                    weight_block_width_ntiles,
                    s_weight,
                    weight_tile_nbytes,
                    weight_stride_h,
                    weight_next_block_stride_h);

            uint32_t out_sbh_start_tile_id = out_block_h_start_tile_id;
            uint32_t out_sbh_start_tile_id_h = out_block_h_start_tile_id_h; //
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
                                uint64_t out_tile_noc_addr = get_noc_addr(out_tile_id, s);
                                //DPRINT << "out_tile_id=" << out_tile_id << ENDL();
                                noc_async_write(l1_read_addr, out_tile_noc_addr, tile_nbytes);
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
            out_block_h_start_tile_id += out_next_block_stride_h;
            out_block_h_start_tile_id_h += out_block_height_num_tiles;
        } // out_num_blocks_h
        out_block_w_start_tile_id += out_next_block_stride_w;
        out_block_w_start_tile_id_w += weight_block_width_ntiles;

        // Increment weight start tile id for next block in width dim
        weight_start_tile_id += weight_next_block_stride_w;
    } // out_num_blocks_w
}
