// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "ttnn/operations/kernel_helper_functions/pad_tile.hpp"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "api/debug/dprint.h"

void kernel_main() {
    // DPRINT << "HELLO FROM reader_bmm_tile_layout_in0_sender_padding KERNEL"<< ENDL();
    uint32_t rt_args_idx = 0;
    // in0 tensor args
    const uint32_t in0_tensor_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t in0_tensor_start_tile_id = get_arg_val<uint32_t>(rt_args_idx++);
    // in0 mcast args
    const uint32_t in0_mcast_dest_noc_start_x = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t in0_mcast_dest_noc_start_y = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t in0_mcast_dest_noc_end_x = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t in0_mcast_dest_noc_end_y = get_arg_val<uint32_t>(rt_args_idx++);

    // padding args
    const uint32_t last_block_h = get_arg_val<uint32_t>(rt_args_idx++);
    // sparsity args
    const uint32_t sparsity_addr = get_arg_val<uint32_t>(rt_args_idx++);

    // COMPILE TIME ARGS
    // in0 tensor args
    constexpr uint32_t in0_tensor_stride_w = get_compile_time_arg_val(0);
    constexpr uint32_t in0_tensor_stride_h = get_compile_time_arg_val(1);
    constexpr uint32_t in0_tensor_next_inner_dim_block_stride = get_compile_time_arg_val(2);
    constexpr uint32_t in0_tensor_next_h_dim_block_stride = get_compile_time_arg_val(3);
    // in0 block args
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(4);
    constexpr uint32_t in0_block_h = get_compile_time_arg_val(5);
    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t in0_last_ktile_w = get_compile_time_arg_val(7);
    constexpr uint32_t in0_last_ktile_h = get_compile_time_arg_val(8);

    constexpr bool extract_shard_sub_blocks = (bool)get_compile_time_arg_val(9);
    constexpr uint32_t shard_width_in_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t shard_height_in_tiles = get_compile_time_arg_val(11);
    // in0/in1 common args
    constexpr uint32_t num_blocks_inner_dim = get_compile_time_arg_val(12);
    constexpr uint32_t num_blocks_w_dim = get_compile_time_arg_val(13);
    constexpr uint32_t num_blocks_h_dim = get_compile_time_arg_val(14);
    // in0 mcast args
    uint32_t in0_mcast_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(15));
    uint32_t in0_mcast_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(16));
    constexpr uint32_t in0_mcast_num_dests = get_compile_time_arg_val(17);
    constexpr uint32_t in0_mcast_num_cores = get_compile_time_arg_val(18);
    // batch args
    constexpr uint32_t MtKt = get_compile_time_arg_val(19);  // if 0
    constexpr uint32_t in0_B = get_compile_time_arg_val(20);
    constexpr uint32_t in1_B = get_compile_time_arg_val(21);

    // sparsity args

    constexpr uint32_t batchB = get_compile_time_arg_val(22);
    constexpr uint32_t sparsity_pagesize = get_compile_time_arg_val(23);
    // Boolean that is set when input A is sparse. If set, both input A and B are assumed to be sparse.
    // Based on the sparsity tensor, the corresponding batch in input A and B are skipped.
    constexpr bool bcast_A = (bool)get_compile_time_arg_val(24);
    // This boolean is set when the number of batches is only known at runtime, typically based on a sparsity tensor.
    constexpr bool get_batch_from_reader = (bool)get_compile_time_arg_val(25);

    constexpr bool fuse_op = (bool)get_compile_time_arg_val(26);

    constexpr auto in0_args = TensorAccessorArgs<27>();
    constexpr auto sparsity_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();

    // 0 is used to specify "INVALID" state, i.e. when the multicasted data has not been received by the receiver.
    // 0x1 is used to specify "VALID" state, i.e. when the batch is valid.
    // 0x2 is used to specify "IGNORE_BATCH" state, i.e. when the batch is not valid.
    constexpr uint32_t IGNORE_BATCH = 0x2;

    // When sparsity is disabled, we just loop once
    constexpr uint32_t batchB_lim = batchB == 0 ? 1u : batchB;

    constexpr uint32_t cb_id_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t in0_single_tile_size_bytes = get_tile_size(cb_id_in0);
    constexpr uint32_t in0_block_size_bytes = in0_block_num_tiles * in0_single_tile_size_bytes;
    constexpr uint32_t one_tile = 1;

    const auto s0 = TensorAccessor(in0_args, in0_tensor_addr, in0_single_tile_size_bytes);

    // sparsity accessor
    constexpr uint32_t cb_id_sparsity = get_named_compile_time_arg_val("cb_sparsity");
    const auto s_sparsity = TensorAccessor(sparsity_args, sparsity_addr, sparsity_pagesize);

    uint32_t l1_write_addr_sparsity = 0;

    // DPRINT << "in0_tensor_start_tile_id: " << in0_tensor_start_tile_id << ENDL();
    // DPRINT << "num_blocks_h_dim: " << num_blocks_h_dim << ENDL();
    // DPRINT << "num_blocks_w_dim: " << num_blocks_w_dim << ENDL();
    // DPRINT << "num_blocks_inner_dim: " << num_blocks_inner_dim << ENDL();
    // DPRINT << "in0_block_w: " << in0_block_w << ENDL();
    // DPRINT << "in0_block_h: " << in0_block_h << ENDL();
    // DPRINT << "in0_block_num_tiles: " << in0_block_num_tiles << ENDL();

    constexpr uint32_t max_batch_size = (in0_B > in1_B) ? in0_B : in1_B;
    for (uint32_t b = 0; b < max_batch_size; ++b) {
        if (in0_B == 1 && in1_B > 1 && b > 0) {
            // DPRINT << "in 0 reader, batch " << b << "dummy reading " << in0_block_num_tiles << " tiles from DRAM" <<
            // ENDL();
            cb_reserve_back(cb_id_in0, in0_block_num_tiles);
            // SliceRange sr = SliceRange{.h0 = 0, .h1 = 1, .hs = 1, .w0 = 0, .w1 = 5, .ws = 1};
            // DPRINT_DATA1({ DPRINT << " in0 reader - batch: " << b << " data at read pointer: " << TileSlice(0, 0, sr,
            // TSLICE_OUTPUT_CB, TSLICE_WR_PTR, true,false) << ENDL(); });
            cb_push_back(cb_id_in0, in0_block_num_tiles);
        } else {
            uint32_t in0_tensor_current_h_dim_block_tile_id = in0_tensor_start_tile_id;
            for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh) {
                for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw) {
                    uint32_t in0_tensor_current_inner_dim_block_start_tile_id = in0_tensor_current_h_dim_block_tile_id;
                    for (uint32_t block = 0; block < num_blocks_inner_dim; ++block) {
                        // Operand 0
                        // Common for sharded and interleaved paths
                        cb_reserve_back(cb_id_in0, in0_block_num_tiles);
                        uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);

                        // Copy in0 block into CB, as the default kernel
                        uint32_t in0_tensor_row_start_tile_id = in0_tensor_current_inner_dim_block_start_tile_id;
                        uint32_t help1 = in0_tensor_row_start_tile_id;
                        uint32_t help2 = 0;
                        for (uint32_t h = 0; h < in0_block_h; ++h) {
                            uint32_t in0_tensor_tile_id = in0_tensor_row_start_tile_id;
                            for (uint32_t w = 0; w < in0_block_w; ++w) {
                                if (bh < num_blocks_h_dim - 1 || h < last_block_h) {
                                    noc_async_read_tile(in0_tensor_tile_id, s0, l1_write_addr_in0);
                                }

                                // Zero out padded regions for the very last tile
                                if constexpr (in0_last_ktile_w > 0) {
                                    if ((block == num_blocks_inner_dim - 1) && (w == in0_block_w - 1)) {
                                        noc_async_read_barrier();
                                        const DataFormat in0_data_format = get_dataformat(cb_id_in0);
                                        pad_last_ktile<in0_data_format, in0_last_ktile_w>(l1_write_addr_in0);
                                    }
                                }
                                if constexpr (in0_last_ktile_h > 0) {
                                    if ((block == num_blocks_inner_dim - 1) && (w == in0_block_w - 1)) {
                                        noc_async_read_barrier();
                                        const DataFormat in0_data_format = get_dataformat(cb_id_in0);
                                        pad_last_transposed_ktile<in0_data_format, in0_last_ktile_h>(l1_write_addr_in0);
                                    }
                                }

                                l1_write_addr_in0 += in0_single_tile_size_bytes;
                                in0_tensor_tile_id += in0_tensor_stride_w;
                                help2 = in0_tensor_tile_id;
                            }
                            in0_tensor_row_start_tile_id += in0_tensor_stride_h;
                        }
                        in0_tensor_current_inner_dim_block_start_tile_id += in0_tensor_next_inner_dim_block_stride;

                        // Barrier! make sure the reads are done
                        noc_async_read_barrier();
                        // SliceRange sr = SliceRange{.h0 = 0, .h1 = 1, .hs = 1, .w0 = 0, .w1 = 5, .ws = 1};
                        // DPRINT_DATA1({ DPRINT << " in0 reader - batch: " << b << ", block: " << block << " from DRAM
                        // " << in0_block_num_tiles << " tiles: (" << help1 << " - " << help2 << ")"  << TileSlice(0, 0,
                        // sr, TSLICE_OUTPUT_CB, TSLICE_WR_PTR, true,false) << ENDL(); });

                        // DPRINT << "finished reading " << in0_block_num_tiles << " tiles from DRAM to core" << ENDL();

                        // Common for sharded and interleaved paths
                        cb_push_back(cb_id_in0, in0_block_num_tiles);
                        // DPRINT << "PUSHED BACK " << in0_block_num_tiles << " tiles to CB0" << ENDL();
                    }
                }
                in0_tensor_current_h_dim_block_tile_id += in0_tensor_next_h_dim_block_stride;
            }
            // DPRINT << "finished reading " << in0_block_num_tiles << " in0 tiles from batch " << b << ENDL();
            if constexpr (!bcast_A) {
                in0_tensor_start_tile_id += MtKt;
            }
            if constexpr (bcast_A) {
                in0_tensor_start_tile_id += MtKt;
            }
        }
    }
    noc_async_write_barrier();
}
