// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
// #include "debug/dprint.h"

template <bool DRAM>
inline void write_tiles_in_block(
    uint32_t cb_id_out0,
    uint32_t block_height_ntiles,
    uint32_t block_width_ntiles,
    uint32_t block_start_row_id,
    uint32_t block_row_offset,
    uint32_t block_row_size,
    uint32_t block_row_size_unpadded,  // to remove padding from the last block in the row
    uint32_t num_rows_unpadded,
    const InterleavedAddrGen<DRAM>& s) {
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
            uint64_t dst_noc_addr = get_noc_addr(block_row_id, s, block_row_offset);
            noc_async_write(l1_read_addr, dst_noc_addr, block_row_size_unpadded);
            l1_read_addr += block_row_size;
            block_row_id++;
        }  // for tile_nrows
        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, block_width_ntiles);
    }  // for block_height_ntiles
}
void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);  // out_dram_addr
    uint32_t num_rows_block = get_arg_val<uint32_t>(1);
    uint32_t block_row_size = get_arg_val<uint32_t>(2);  // in0_block_w * TILE_WIDTH * dtype_nbytes
    uint32_t batch = get_arg_val<uint32_t>(3);
    uint32_t num_blocks_h = get_arg_val<uint32_t>(4);
    uint32_t num_blocks_w = get_arg_val<uint32_t>(5);
    uint32_t output_row_size = get_arg_val<uint32_t>(6);               // output row size bytes
    uint32_t last_block_row_size_unpadded = get_arg_val<uint32_t>(7);  // unpadded last block width
    uint32_t num_output_rows_unpadded = get_arg_val<uint32_t>(8);
    uint32_t block_start_row_id = get_arg_val<uint32_t>(9);
    uint32_t block_start_row_offset = get_arg_val<uint32_t>(10);

    constexpr bool out_in_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool FLOAT32_DTYPE = get_compile_time_arg_val(1) == 1;

    // NOTE: Row major layout only supports bfp16
    // TT_ASSERT(out_df != DataFormat::Bfp8_b);
    constexpr uint32_t cb_id_out0 = tt::CBIndex::c_16;
    const DataFormat out_df = get_dataformat(cb_id_out0);

    constexpr uint32_t TILE_HEIGHT = 32;  // TODO: use common source of truth

    const uint32_t block_width_ntiles =
        FLOAT32_DTYPE ? block_row_size >> 7
                      : block_row_size >> 6;  // Assuming 4/2 bytes per datum, there are 128/64 bytes per tile row
    const uint32_t block_height_ntiles = num_rows_block / TILE_HEIGHT;

    // const InterleavedAddrGenFast<true> s = {
    //     .bank_base_address = dst_addr,
    //     .page_size = output_row_size,
    //     .data_format = out_df
    // };
    const InterleavedAddrGen<out_in_dram> s = {.bank_base_address = dst_addr, .page_size = output_row_size};
    uint32_t num_rows_unpadded = num_output_rows_unpadded + block_start_row_id;
    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t block_h = 0; block_h < num_blocks_h; block_h++) {
            uint32_t block_row_offset = block_start_row_offset;
            for (uint32_t block_w = 0; block_w < num_blocks_w; block_w++) {
                uint32_t current_block_row_size_unpadded = block_row_size;
                if (block_w == (num_blocks_w - 1)) {
                    current_block_row_size_unpadded = last_block_row_size_unpadded;
                }
                write_tiles_in_block(
                    cb_id_out0,
                    block_height_ntiles,
                    block_width_ntiles,
                    block_start_row_id,
                    block_row_offset,
                    block_row_size,
                    current_block_row_size_unpadded,  // padding is only in the last block
                    num_rows_unpadded,
                    s);
                block_row_offset += block_row_size;
            }  // for num_blocks_w
            block_start_row_id += num_rows_block;
        }  // for num_blocks_h
    }  // for batch
}
