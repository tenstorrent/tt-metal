// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t i = 0;

    uint32_t has_work_for_q_heads = get_arg_val<uint32_t>(i++);
    if (has_work_for_q_heads == 0) {
        return;
    }

    uint32_t src0_addr = get_arg_val<uint32_t>(i++);
    uint32_t dst_addr = get_arg_val<uint32_t>(i++);
    uint32_t Mt = get_arg_val<uint32_t>(i++);
    uint32_t Kt = get_arg_val<uint32_t>(i++);
    uint32_t Nt = get_arg_val<uint32_t>(i++);
    uint32_t MtKt = get_arg_val<uint32_t>(i++);
    uint32_t blocks = get_arg_val<uint32_t>(i++);
    uint32_t in0_start_id = get_arg_val<uint32_t>(i++);
    uint32_t out_start_id = get_arg_val<uint32_t>(i++);

    // matmul params
    uint32_t in0_block_w = get_arg_val<uint32_t>(i++);
    uint32_t in1_num_subblocks = get_arg_val<uint32_t>(i++);
    uint32_t in1_num_blocks = get_arg_val<uint32_t>(i++);
    uint32_t out_num_tiles = get_arg_val<uint32_t>(i++);

    // constants
    uint32_t bfloat16_row_bytes = get_arg_val<uint32_t>(i++);
    uint32_t bfloat16_Nt_bytes = get_arg_val<uint32_t>(i++);
    uint32_t bfloat16_last_row_bytes_read = get_arg_val<uint32_t>(i++);

    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(2);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(3);
    constexpr uint32_t intermediate_num_tiles = get_compile_time_arg_val(3);

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_in1 =
        tt::CBIndex::c_1;  // mcast receive all kv_heads; compute chooses which kv_heads to use for matmul
    constexpr uint32_t cb_id_intermed0 = tt::CBIndex::c_3;
    constexpr uint32_t cb_id_intermed1 = tt::CBIndex::c_4;

    constexpr uint32_t onetile = 1;
    constexpr uint32_t num_rows_in_one_tile = 32;
    const uint32_t in1_tile_bytes = get_tile_size(cb_id_in1);

#ifndef IN0_SHARDED
    const uint32_t in0_tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat in0_data_format = get_dataformat(cb_id_in0);
    const InterleavedAddrGenFast<src0_is_dram> s0 = {
        .bank_base_address = src0_addr, .page_size = in0_tile_bytes, .data_format = in0_data_format};
#endif

#ifndef OUT_SHARDED
    const uint32_t out_tile_bytes = get_tile_size(cb_id_out);
    const DataFormat out_data_format = get_dataformat(cb_id_out);
    const InterleavedAddrGenFast<dst_is_dram> s = {
        .bank_base_address = dst_addr, .page_size = out_tile_bytes, .data_format = out_data_format};
#endif

#ifndef IN0_SHARDED
    // Only used for interleaved
    uint32_t in0_batch = in0_start_id;
    uint32_t in0_Mt;
    uint32_t in0_tensor_id;
#endif

#ifndef OUT_SHARDED
    uint32_t out_tensor_id = out_start_id;
#endif

    uint32_t bfloat16_row_bytes_read = bfloat16_row_bytes;

    for (uint32_t b = 0; b < blocks; b++) {  // TODO: Must be 1
#ifndef IN0_SHARDED
        in0_Mt = in0_batch;
#endif

        for (uint32_t m = 0; m < Mt; m++) {  // TODO: Must be 1; generalize to support batch > 32 (ie. Mt > 1)
            // TODO: Generalize to support inner dim blocking; in0 reads has to moved within num_rows_in_one_tile loop
            cb_reserve_back(cb_id_in0, in0_block_w);
#ifndef IN0_SHARDED
            in0_tensor_id = in0_Mt;
            uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
            for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                // Read in0 block
                noc_async_read_tile(in0_tensor_id, s0, l1_write_addr_in0);

                l1_write_addr_in0 += in0_tile_bytes;
                in0_tensor_id++;
            }
            noc_async_read_barrier();
#endif

            cb_push_back(cb_id_in0, in0_block_w);

            cb_reserve_back(cb_id_intermed1, out_num_tiles);
            uint32_t cb_intermed1_addr = get_write_ptr(cb_id_intermed1);
            for (uint32_t in1_block = 0; in1_block < in1_num_blocks; in1_block++) {
                const bool last_out = in1_block == in1_num_blocks - 1;
                if (last_out) {
                    bfloat16_row_bytes_read =
                        bfloat16_last_row_bytes_read;  // For padded subblocks, read partial untilized subblocks
                }

                uint32_t row_offset_bytes = 0;
                uint32_t cb_intermed1_addr_curr_block = cb_intermed1_addr;
                for (uint32_t tile_row_id = 0; tile_row_id < num_rows_in_one_tile; tile_row_id++) {
                    for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks;
                         in1_subblock++) {  // TODO: Must be 1; to generalize, need to handle untilizing + reading for
                                            // subblocks
                        // Read 32 untilized tiles and select correct rows to reconstruct single correct tile
                        cb_wait_front(cb_id_intermed0, intermediate_num_tiles);
                        noc_async_read(
                            get_noc_addr(get_read_ptr(cb_id_intermed0)) + row_offset_bytes,
                            cb_intermed1_addr_curr_block,
                            bfloat16_row_bytes_read);
                        noc_async_read_barrier();
                        cb_pop_front(cb_id_intermed0, intermediate_num_tiles);
                        row_offset_bytes += bfloat16_row_bytes;
                        cb_intermed1_addr_curr_block += bfloat16_Nt_bytes;
                    }  // in1_num_subblocks loop

#ifndef IN0_SHARDED
                    in0_Mt += Kt;
#endif
                }  // 32 tiles loop
                cb_intermed1_addr += bfloat16_row_bytes;

            }  // in1_num_blocks loop
            cb_push_back(cb_id_intermed1, out_num_tiles);

#ifndef OUT_SHARDED
            cb_wait_front(cb_id_out, out_num_tiles);
            uint32_t l1_read_addr_out = get_read_ptr(cb_id_out);
            for (uint32_t nt = 0; nt < Nt;
                 nt++) {  // TODO: Must be full MtNt; generalize to support Mt > 1 or blocks > 1
                noc_async_write_tile(out_tensor_id, s, l1_read_addr_out);
                l1_read_addr_out += out_tile_bytes;
                out_tensor_id++;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_id_out, out_num_tiles);
#endif
        }  // Mt loop

#ifndef IN0_SHARDED
        in0_batch += MtKt;
#endif
    }  // B loop

#ifdef OUT_SHARDED
    cb_wait_front(cb_id_out, out_num_tiles);
#endif
}
