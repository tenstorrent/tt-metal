// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "api/debug/dprint.h"

void kernel_main() {
    // DPRINT << "HELLO FROM reader_bmm_tile_layout_in1_sender_writer_padding KERNEL"<< ENDL();
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

    // sparsity args
    const uint32_t sparsity_addr = get_arg_val<uint32_t>(rt_args_idx++);

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
    // READER
    // in1 tensor args
    constexpr uint32_t in1_tensor_stride_w = get_compile_time_arg_val(0);
    constexpr uint32_t in1_tensor_stride_h = get_compile_time_arg_val(1);
    constexpr uint32_t in1_tensor_next_block_stride = get_compile_time_arg_val(2);
    constexpr uint32_t in1_tensor_next_w_dim_block_stride = get_compile_time_arg_val(3);
    // in1 block args
    constexpr uint32_t in1_block_w = get_compile_time_arg_val(4);
    constexpr uint32_t in1_block_h = get_compile_time_arg_val(5);
    constexpr uint32_t in1_block_num_tiles = get_compile_time_arg_val(6);
    // in0/in1 common args
    constexpr uint32_t num_blocks_inner_dim = get_compile_time_arg_val(7);
    constexpr uint32_t num_blocks_w_dim = get_compile_time_arg_val(8);
    constexpr uint32_t num_blocks_h_dim = get_compile_time_arg_val(9);

    // in1 mcast args
    uint32_t in1_mcast_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(10));
    uint32_t in1_mcast_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(11));
    constexpr uint32_t in1_mcast_num_dests = get_compile_time_arg_val(12);
    constexpr uint32_t in1_mcast_num_cores = get_compile_time_arg_val(13);
    // batch args
    constexpr uint32_t KtNt = get_compile_time_arg_val(14);
    constexpr uint32_t batch = get_compile_time_arg_val(15);
    constexpr uint32_t bcast_B = get_compile_time_arg_val(16);
    // sparsity args
    constexpr uint32_t batchB = get_compile_time_arg_val(17);
    constexpr uint32_t sparsity_pagesize = get_compile_time_arg_val(18);

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

    // When sparsity is disabled, we just loop once
    constexpr uint32_t batchB_lim = batchB == 0 ? 1u : batchB;

    constexpr uint32_t one_tile = 1;
    rt_args_idx += 2;  // Skip over placeholders
    const uint32_t last_num_blocks_w_dim = get_arg_val<uint32_t>(rt_args_idx++);
    constexpr bool fuse_op_all_gather = (bool)get_compile_time_arg_val(30);
    constexpr bool fuse_op_reduce_scatter = (bool)get_compile_time_arg_val(31);

    MatmulOpReceiver fused_op_receiver;
    OpSignaler op_signaler;
    if constexpr (fuse_op_all_gather) {
        fused_op_receiver = MatmulOpReceiver(
            false, /* wait_for_op_signal */
            rt_args_idx,
            num_blocks_inner_dim,
            in1_block_h /* tiles_per_block (in the same dimension */
        );
    } else if constexpr (fuse_op_reduce_scatter) {
        op_signaler = OpSignaler(rt_args_idx);
    }

    constexpr auto in1_args = TensorAccessorArgs<32>();
    constexpr auto sparsity_args = TensorAccessorArgs<in1_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<sparsity_args.next_compile_time_args_offset()>();
    constexpr auto after_bias_offset = out_args.next_compile_time_args_offset();

    constexpr uint32_t cb_id_in1 = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t in1_single_tile_size_bytes = get_tile_size(cb_id_in1);
    constexpr const uint32_t in1_tile_hw = get_tile_hw(cb_id_in1);
    constexpr uint32_t in1_block_size_bytes = in1_block_num_tiles * in1_single_tile_size_bytes;

    uint32_t l1_write_addr_in1;

    const auto s1 = TensorAccessor(in1_args, in1_tensor_addr, in1_single_tile_size_bytes);

    //  WRITER
    constexpr uint32_t cb_id_out0 = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t output_single_tile_size_bytes = get_tile_size(cb_id_out0);
    constexpr const uint32_t output_tile_hw = get_tile_hw(cb_id_out0);
    const auto s = TensorAccessor(out_args, out_tensor_addr, output_single_tile_size_bytes);

    // sparsity accessor
    constexpr uint32_t cb_id_sparsity = get_named_compile_time_arg_val("cb_sparsity");
    const auto s_sparsity = TensorAccessor(sparsity_args, sparsity_addr, sparsity_pagesize);

    // Set ur local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    volatile tt_l1_ptr uint32_t* in1_mcast_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_mcast_receiver_semaphore_addr);
    *(in1_mcast_receiver_semaphore_addr_ptr) = VALID;
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast
    volatile tt_l1_ptr uint32_t* in1_mcast_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_mcast_sender_semaphore_addr);

    const uint64_t in1_mcast_receiver_semaphore_noc_addr = get_noc_multicast_addr(
        in1_mcast_dest_noc_start_x,
        in1_mcast_dest_noc_start_y,
        in1_mcast_dest_noc_end_x,
        in1_mcast_dest_noc_end_y,
        in1_mcast_receiver_semaphore_addr);

    const uint64_t in1_multicast_data_noc = get_noc_multicast_addr(
        in1_mcast_dest_noc_start_x, in1_mcast_dest_noc_start_y, in1_mcast_dest_noc_end_x, in1_mcast_dest_noc_end_y, 0);

    uint32_t l1_write_addr_sparsity = 0;

    // DPRINT << "batch: " << batch << ENDL(); // HERE JAKSA - these are activation batches and not weight batches
    // DPRINT << "in1_tensor_start_tile_id: " << in1_tensor_start_tile_id << ENDL();
    // DPRINT << "out_tensor_start_tile_id: " << out_tensor_start_tile_id << ENDL();
    // DPRINT << "num_blocks_h_dim: " << num_blocks_h_dim << ENDL();
    // DPRINT << "num_blocks_w_dim: " << num_blocks_w_dim << ENDL();
    // DPRINT << "num_blocks_inner_dim: " << num_blocks_inner_dim << ENDL();
    // DPRINT << "in1_block_h: " << in1_block_h << ENDL();
    // DPRINT << "in1_block_w: " << in1_block_w << ENDL();
    // DPRINT << "in1_mcast_num_dests: " << in1_mcast_num_dests << ENDL();

    for (uint32_t b = 0; b < batch; ++b) {
        // DPRINT << "starting read of batch: " << b << ENDL();
        uint32_t in1_batch_tile_id = in1_tensor_start_tile_id;

        uint32_t in1_tensor_current_h_dim_block_tile_id = in1_batch_tile_id;
        uint32_t out_tensor_current_h_dim_block_tile_id = out_tensor_start_tile_id;
        for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh) {
            uint32_t in1_tensor_current_w_dim_block_tile_id = in1_tensor_current_h_dim_block_tile_id;
            uint32_t out_tensor_current_w_dim_block_tile_id = out_tensor_current_h_dim_block_tile_id;

            for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw) {
                uint32_t in1_tensor_current_inner_dim_block_start_tile_id = in1_tensor_current_w_dim_block_tile_id;

                for (uint32_t block = 0; block < num_blocks_inner_dim; ++block) {
                    if constexpr (fuse_op_all_gather) {
                        fused_op_receiver.update_current_block_start_tile_id(
                            block, in1_tensor_current_inner_dim_block_start_tile_id, in1_batch_tile_id);
                    }
                    // Operand 1 - interleaved
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

                    // DPRINT << "starting mcast of " << in1_block_num_tiles << " tiles from batch: " << b << ", block:
                    // " << block << ENDL();

                    // wait until all in1 mcast destinations have atomically incremented the in1 semaphore_addr
                    // (i.e. its value should be in0_mcast_num_dests), then reset the semaphore_addr value back to
                    // zero for the next block
                    noc_semaphore_wait(in1_mcast_sender_semaphore_addr_ptr, in1_mcast_num_dests);
                    noc_semaphore_set(in1_mcast_sender_semaphore_addr_ptr, 0);

                    // Now we have the block in the CB address, we can mcast to dests!
                    uint64_t in1_multicast_data_addr = in1_multicast_data_noc | in1_start_address;

                    // num_dests must not include source, since we are NOT really doing a local copy!
                    noc_async_write_multicast(
                        in1_start_address, in1_multicast_data_addr, in1_block_size_bytes, in1_mcast_num_cores, true);

                    // Note: no need for write barrier, since these two multicasts are done on the same noc id and
                    // same vc even though cmd bufs are different Also, this only works because we are setting VCs
                    // statically (using NOC_CMD_STATIC_VC).
                    // On Blackhole the flush is needed because NoC latency is higher than L1 <-> RISCV latency
                    // which means data could be changed before
                    //  write is issued.
                    noc_async_writes_flushed();

                    // We should also multicast the flag to destinations
                    // num_dests must not include source, since we are NOT really doing a local copy!
                    noc_semaphore_set_multicast(
                        in1_mcast_receiver_semaphore_addr, in1_mcast_receiver_semaphore_noc_addr, in1_mcast_num_cores);

                    cb_push_back(cb_id_in1, in1_block_num_tiles);
                }

                // WRITER
                uint32_t num_blocks_w_dim_ = bw >= last_num_blocks_w_dim - 1 ? last_num_blocks_w_dim : num_blocks_w_dim;
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
                        // DPRINT << "got " << out_subblock_tile_count << " tiles ready in out CB for block: " << b <<
                        // ENDL();
                        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);

                        // uint32_t help1 = out_tensor_sb_row_start_tile_id;
                        // uint32_t help2 = out_tensor_sb_row_start_tile_id;
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
                            // help2 = out_tensor_tile_id;
                        }
                        noc_async_write_barrier();
                        // DPRINT << "sent " << out_subblock_tile_count << " tiles from batch: " << b << " to DRAM to
                        // tile ids: " << help1 << " - " << help2 << ENDL(); DPRINT << "sent " <<
                        // out_subblock_tile_count << " tiles from out CB to DRAM for block: " << b << ENDL();

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
        out_tensor_start_tile_id += MtNt;
        in1_batch_tile_id += KtNt;
        if constexpr (bcast_B == 0) {
            // For height-sharded DRAM, tile IDs are relative within a batch;
            // batch offset is handled by switching DRAM banks
            in1_tensor_start_tile_id += KtNt;
        }

        if (fuse_op_reduce_scatter) {
            // Signal reduce_scatter to go
            op_signaler.synchronize_workers_and_signal_op(0);
        }
    }
    noc_async_write_barrier();
}
