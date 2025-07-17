// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"

void kernel_main() {
    // READER
    uint32_t rt_args_idx = 0;
    // in1 tensor args
    const uint32_t in1_tensor_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t in1_tensor_start_tile_id = get_arg_val<uint32_t>(rt_args_idx++);
    // in1 mcast args
    const uint32_t in1_mcast_dest_noc_start_x = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t in1_mcast_dest_noc_start_y = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t in1_mcast_dest_noc_end_x = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t in1_mcast_dest_noc_end_y = get_arg_val<uint32_t>(rt_args_idx++);

    // WRITER
    // out tensor args
    const uint32_t out_tensor_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t out_tensor_start_tile_id = get_arg_val<uint32_t>(rt_args_idx++);

    // padding args (READER)
    const uint32_t last_block_w = get_arg_val<uint32_t>(rt_args_idx++);
    // padding args (WRITER)
    const uint32_t out_num_nonzero_subblocks_h = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t out_last_subblock_h = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t padded_block_tiles_h_skip = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t out_num_nonzero_subblocks_w = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t out_last_num_nonzero_subblocks_w = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t out_last_subblock_w = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t padded_subblock_tiles_addr_skip = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t padded_block_tiles_w_skip = get_arg_val<uint32_t>(rt_args_idx++);

    // COMPILE TIME ARGS
    // interleaved accessor args
    constexpr bool in1_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool out_is_dram = get_compile_time_arg_val(1) == 1;

    // READER
    // in1 tensor args
    constexpr uint32_t in1_tensor_stride_w = get_compile_time_arg_val(2);
    constexpr uint32_t in1_tensor_stride_h = get_compile_time_arg_val(3);
    constexpr uint32_t in1_tensor_next_block_stride = get_compile_time_arg_val(4);
    constexpr uint32_t in1_tensor_next_w_dim_block_stride = get_compile_time_arg_val(5);
    // in1 block args
    constexpr uint32_t in1_block_w = get_compile_time_arg_val(6);
    constexpr uint32_t in1_block_h = get_compile_time_arg_val(7);
    constexpr uint32_t in1_block_num_tiles = get_compile_time_arg_val(8);
    // in0/in1 common args
    constexpr uint32_t num_blocks_inner_dim = get_compile_time_arg_val(9);
    constexpr uint32_t num_blocks_w_dim = get_compile_time_arg_val(10);
    constexpr uint32_t num_blocks_h_dim = get_compile_time_arg_val(11);

    // in1 mcast args
    uint32_t in1_mcast_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(12));
    uint32_t in1_mcast_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(13));
    constexpr uint32_t in1_mcast_num_dests = get_compile_time_arg_val(14);
    constexpr uint32_t in1_mcast_num_cores = get_compile_time_arg_val(15);
    // batch args
    constexpr uint32_t KtNt = get_compile_time_arg_val(16);
    constexpr uint32_t batchA = get_compile_time_arg_val(17);
    constexpr uint32_t batchB = get_compile_time_arg_val(18);

    // WRITER
    // out tensor args
    constexpr uint32_t out_tensor_stride_w = get_compile_time_arg_val(19);
    constexpr uint32_t out_tensor_stride_h = get_compile_time_arg_val(20);
    constexpr uint32_t out_tensor_next_subblock_stride_w = get_compile_time_arg_val(21);
    constexpr uint32_t out_tensor_next_subblock_stride_h = get_compile_time_arg_val(22);
    constexpr uint32_t out_tensor_next_w_dim_block_stride = get_compile_time_arg_val(23);
    constexpr uint32_t out_tensor_next_h_dim_block_stride = get_compile_time_arg_val(24);
    // out subblock args
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(25);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(26);
    constexpr uint32_t out_subblock_tile_count = get_compile_time_arg_val(27);
    // batch args
    constexpr uint32_t MtNt = get_compile_time_arg_val(28);  // if 0
    // Don't need batch; same as batch from READER args

    // sparsity args
    constexpr uint32_t sparsity_is_dram = get_compile_time_arg_val(29);
    constexpr uint32_t sparsity_log2_of_pagesize = get_compile_time_arg_val(30);

    rt_args_idx += 2;  // Skip over placeholders
    const uint32_t last_num_blocks_w_dim = get_arg_val<uint32_t>(rt_args_idx++);

    // sparsity args
    const uint32_t sparsity_addr = get_arg_val<uint32_t>(rt_args_idx++);

    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t in1_single_tile_size_bytes = get_tile_size(cb_id_in1);
    constexpr const uint32_t in1_tile_hw = get_tile_hw(cb_id_in1);
    constexpr uint32_t in1_block_size_bytes = in1_block_num_tiles * in1_single_tile_size_bytes;

    uint32_t l1_write_addr_in1;

    constexpr DataFormat in1_data_format = get_dataformat(cb_id_in1);
    const InterleavedAddrGenFast<in1_is_dram, in1_tile_hw> s1 = {
        .bank_base_address = in1_tensor_addr, .page_size = in1_single_tile_size_bytes, .data_format = in1_data_format};

    //  WRITER
    constexpr uint32_t cb_id_out0 = tt::CBIndex::c_4;
    constexpr uint32_t output_single_tile_size_bytes = get_tile_size(cb_id_out0);
    constexpr const uint32_t output_tile_hw = get_tile_hw(cb_id_out0);
    constexpr DataFormat output_data_format = get_dataformat(cb_id_out0);
    const InterleavedAddrGenFast<out_is_dram, output_tile_hw> s = {
        .bank_base_address = out_tensor_addr,
        .page_size = output_single_tile_size_bytes,
        .data_format = output_data_format};

    constexpr uint32_t cb_id_sparsity = tt::CBIndex::c_6;
    const InterleavedPow2AddrGenFast<sparsity_is_dram> s_sparsity = {
        .bank_base_address = sparsity_addr, .log_base_2_of_page_size = sparsity_log2_of_pagesize};

    cb_reserve_back(cb_id_sparsity, 1);
    uint32_t l1_write_addr_sparsity = get_write_ptr(cb_id_sparsity);
    for (uint32_t bA = 0; bA < batchA; ++bA) {
        uint32_t in1_batch_tile_id = in1_tensor_start_tile_id;
        noc_async_read_page(bA, s_sparsity, l1_write_addr_sparsity);
        noc_async_read_barrier();

        for (uint32_t bB = 0; bB < batchB; ++bB) {
            if (reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_write_addr_sparsity)[bB] != 0) {
                uint32_t in1_tensor_current_h_dim_block_tile_id = in1_batch_tile_id;
                uint32_t out_tensor_current_h_dim_block_tile_id = out_tensor_start_tile_id;
                for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh) {
                    uint32_t in1_tensor_current_w_dim_block_tile_id = in1_tensor_current_h_dim_block_tile_id;
                    uint32_t out_tensor_current_w_dim_block_tile_id = out_tensor_current_h_dim_block_tile_id;
                    for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw) {
                        uint32_t in1_tensor_current_inner_dim_block_start_tile_id =
                            in1_tensor_current_w_dim_block_tile_id;

                        for (uint32_t block = 0; block < num_blocks_inner_dim; ++block) {
                            // Operand 1
                            cb_reserve_back(cb_id_in1, in1_block_num_tiles);
                            l1_write_addr_in1 = get_write_ptr(cb_id_in1);
                            uint64_t in1_start_address =
                                l1_write_addr_in1;  // copy start address of block, to be used for mcasting

                            // Copy in1 block into CB, as the default kernel
                            uint32_t in1_tensor_row_start_tile_id = in1_tensor_current_inner_dim_block_start_tile_id;
                            for (uint32_t h = 0; h < in1_block_h; ++h) {
                                uint32_t in1_tensor_tile_id = in1_tensor_row_start_tile_id;
                                for (uint32_t w = 0; w < in1_block_w; ++w) {
                                    if (bw < num_blocks_w_dim - 1 || w < last_block_w) {
                                        noc_async_read_tile(in1_tensor_tile_id, s1, l1_write_addr_in1);
                                    }
                                    l1_write_addr_in1 += in1_single_tile_size_bytes;
                                    in1_tensor_tile_id += in1_tensor_stride_w;
                                }
                                in1_tensor_row_start_tile_id += in1_tensor_stride_h;
                            }
                            in1_tensor_current_inner_dim_block_start_tile_id += in1_tensor_next_block_stride;

                            // Barrier! make sure the reads are done
                            noc_async_read_barrier();

                            cb_push_back(cb_id_in1, in1_block_num_tiles);
                        }

                        // WRITER
                        uint32_t num_blocks_w_dim_ =
                            bw >= last_num_blocks_w_dim - 1 ? last_num_blocks_w_dim : num_blocks_w_dim;
                        uint32_t out_num_nonzero_subblocks_h_ = out_num_nonzero_subblocks_h;
                        uint32_t out_num_nonzero_subblocks_w_ = out_num_nonzero_subblocks_w;
                        if (bw == num_blocks_w_dim_ - 1) {
                            out_num_nonzero_subblocks_w_ = out_last_num_nonzero_subblocks_w;
                        }
                        uint32_t out_tensor_sbh_start_tile_id = out_tensor_current_w_dim_block_tile_id;
                        for (uint32_t sbh = 0; sbh < out_num_nonzero_subblocks_h_; ++sbh) {
                            uint32_t out_tensor_sbw_start_tile_id = out_tensor_sbh_start_tile_id;
                            for (uint32_t sbw = 0; sbw < out_num_nonzero_subblocks_w_; ++sbw) {
                                uint32_t out_tensor_sb_row_start_tile_id = out_tensor_sbw_start_tile_id;

                                uint32_t out_subblock_h_ = out_subblock_h;
                                uint32_t out_subblock_w_ = out_subblock_w;
                                uint32_t subblock_tiles_addr_skip = 0;
                                if (bh == num_blocks_h_dim - 1 && sbh == out_num_nonzero_subblocks_h - 1) {
                                    out_subblock_h_ = out_last_subblock_h;
                                }
                                if (bw == num_blocks_w_dim_ - 1 && sbw == out_num_nonzero_subblocks_w_ - 1) {
                                    out_subblock_w_ = out_last_subblock_w;
                                    subblock_tiles_addr_skip = padded_subblock_tiles_addr_skip;
                                }

                                cb_wait_front(cb_id_out0, out_subblock_tile_count);
                                uint32_t l1_read_addr = get_read_ptr(cb_id_out0);

                                for (uint32_t h = 0; h < out_subblock_h_; ++h) {
                                    uint32_t out_tensor_tile_id = out_tensor_sb_row_start_tile_id;
                                    for (uint32_t w = 0; w < out_subblock_w_; ++w) {
                                        if (bw < num_blocks_w_dim_) {
                                            noc_async_write_tile(out_tensor_tile_id, s, l1_read_addr);
                                        }

                                        l1_read_addr += output_single_tile_size_bytes;

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
                            if (bw == num_blocks_w_dim_ - 1) {
                                cb_wait_front(cb_id_out0, padded_block_tiles_w_skip);
                                cb_pop_front(cb_id_out0, padded_block_tiles_w_skip);
                            }
                            out_tensor_sbh_start_tile_id += out_tensor_next_subblock_stride_h;
                        }
                        // Pop row(s) of fully padded subblocks
                        if (bh == num_blocks_h_dim - 1) {
                            cb_wait_front(cb_id_out0, padded_block_tiles_h_skip);
                            cb_pop_front(cb_id_out0, padded_block_tiles_h_skip);
                        }

                        in1_tensor_current_w_dim_block_tile_id += in1_tensor_next_w_dim_block_stride;
                        out_tensor_current_w_dim_block_tile_id += out_tensor_next_w_dim_block_stride;
                    }
                    out_tensor_current_h_dim_block_tile_id += out_tensor_next_h_dim_block_stride;
                }
            }

            in1_batch_tile_id += KtNt;
            out_tensor_start_tile_id += MtNt;
        }
    }
}
