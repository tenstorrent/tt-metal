// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
// #include "debug/dprint.h"

FORCE_INLINE void read_weight_blocks_inner_h_dim(uint32_t cb_id_weight,
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

template <bool DRAM>
FORCE_INLINE void write_tiles_in_output_block(uint32_t cb_id_out0,
                        uint32_t block_height_ntiles,
                        uint32_t block_width_ntiles,
                        uint32_t block_start_row_id,
                        uint32_t block_row_offset,
                        uint32_t block_row_size,
                        uint32_t block_row_size_unpadded, // to remove padding from the last block in the row
                        uint32_t num_rows_unpadded,
                        const InterleavedPow2AddrGenFast<DRAM>& s) {
    constexpr uint32_t TILE_HEIGHT = 32;  // TODO: use common source of truth
    uint32_t block_row_id = block_start_row_id;
    for (uint32_t tile_row_id = 0; tile_row_id < block_height_ntiles; tile_row_id++) {
        // We reserve back an entire row of tiles in a block and issue a bunch of reads
        cb_wait_front(cb_id_out0, block_width_ntiles);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
        for (uint32_t j = 0; j < TILE_HEIGHT; j++) {
            if (block_row_id >= num_rows_unpadded) {
                break;
            }
            s.noc_async_write_page(block_row_id, l1_read_addr, block_row_size_unpadded, block_row_offset);
            l1_read_addr += block_row_size;
            block_row_id++;
        } // for tile_nrows
        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, block_width_ntiles);
    } // for block_height_ntiles
}

void kernel_main() {
    uint32_t i = 0;
    uint32_t dst_addr = get_arg_val<uint32_t>(i); i+=1;          // out_dram_addr
    uint32_t weight_addr_dram_base = get_arg_val<uint32_t>(i); i+=1;

    uint32_t num_rows_block = get_arg_val<uint32_t>(i); i+=1;
    uint32_t block_row_size = get_arg_val<uint32_t>(i); i+=1;     // in0_block_w * TILE_WIDTH * dtype_nbytes
    uint32_t batch = get_arg_val<uint32_t>(i); i+=1;
    uint32_t num_blocks_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t num_blocks_w = get_arg_val<uint32_t>(i); i+=1;
    uint32_t output_row_size = get_arg_val<uint32_t>(i); i+=1;    // output row size bytes
    uint32_t last_block_row_size_unpadded = get_arg_val<uint32_t>(i); i+=1; // unpadded last block width
    uint32_t num_output_rows_unpadded = get_arg_val<uint32_t>(i); i+=1;

    uint32_t num_blocks_weight_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weight_block_num_tiles = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weight_block_height_ntiles = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weight_block_width_ntiles = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weight_stride_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weight_next_block_stride_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weight_next_block_stride_w = get_arg_val<uint32_t>(i); i+=1;


    constexpr bool out_in_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_weight = get_compile_time_arg_val(2);
    constexpr uint32_t log_2_of_output_row_size = get_compile_time_arg_val(3);

    //DPRINT << "cb id weight " << cb_id_weight << ENDL();
    // NOTE: Row major layout only supports bfp16
    // TT_ASSERT(out_df != DataFormat::Bfp8_b);
    const DataFormat out_df = get_dataformat(cb_id_out0);

    constexpr uint32_t TILE_HEIGHT = 32;                    // TODO: use common source of truth

    const uint32_t block_width_ntiles = block_row_size >> 6; // Assuming 2 bytes per datum, there are 64 bytes per tile row
    const uint32_t block_height_ntiles = num_rows_block / TILE_HEIGHT;
    uint32_t block_start_row_id = 0;

    const InterleavedPow2AddrGenFast<out_in_dram> s = {
        .bank_base_address = dst_addr,
        .log_base_2_of_page_size = log_2_of_output_row_size
    };

    const uint32_t weight_tile_nbytes = get_tile_size(cb_id_weight);
    constexpr uint32_t tile_size_pow2_exponent = 11;
    const InterleavedPow2AddrGen<true> s_weight = {
        .bank_base_address = weight_addr_dram_base,
        .log_base_2_of_page_size = tile_size_pow2_exponent
    };

    for(uint32_t b = 0; b < batch; ++b) {
        for(uint32_t block_h = 0; block_h < num_blocks_h; block_h++) {
            uint32_t block_row_offset = 0;
            // Reset weight start tile index
            uint32_t weight_start_tile_id = 0;
            for(uint32_t block_w = 0; block_w < num_blocks_w; block_w++) {

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
                // Increment weight start tile id for next block in width dim
                weight_start_tile_id += weight_next_block_stride_w;

                uint32_t current_block_row_size_unpadded = block_row_size;
                if(block_w == (num_blocks_w - 1)) {
                    current_block_row_size_unpadded = last_block_row_size_unpadded;
                }
                write_tiles_in_output_block(cb_id_out0,
                        block_height_ntiles,
                        block_width_ntiles,
                        block_start_row_id,
                        block_row_offset,
                        block_row_size,
                        current_block_row_size_unpadded, // padding is only in the last block
                        num_output_rows_unpadded,
                        s);
                block_row_offset += block_row_size;
            } // for num_blocks_w
            block_start_row_id += num_rows_block;
        } // for num_blocks_h
    } // for batch
}
