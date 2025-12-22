// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
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

#ifdef FUSE_BIAS
    // in3 mcast args
    const uint32_t in3_tensor_addr = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t in3_tensor_start_tile_id = get_arg_val<uint32_t>(rt_args_idx++);

    constexpr uint32_t in3_tensor_stride_w = get_compile_time_arg_val(29);

    constexpr uint32_t cb_id_in3 = 3;
    constexpr uint32_t bias_single_tile_size_bytes = get_tile_size(cb_id_in3);
    constexpr const uint32_t in3_tile_hw = get_tile_hw(cb_id_in3);

#ifndef BIAS_SHARDED
    uint32_t l1_write_addr_in3;
    // Bias accessor will be defined later after TensorAccessor args
#endif  // BIAS_SHARDED
#else
    rt_args_idx += 2;  // Skip over placeholders
#endif  // FUSE_BIAS
#ifndef OUT_SHARDED
    const uint32_t last_num_blocks_w_dim = get_arg_val<uint32_t>(rt_args_idx++);
#endif  // OUT_SHARDED

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
#ifdef FUSE_BIAS
    constexpr auto bias_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto after_bias_offset = bias_args.next_compile_time_args_offset();
#else
    constexpr auto after_bias_offset = out_args.next_compile_time_args_offset();
#endif  // FUSE_BIAS

// RT and COMPILE TIME ARGS for DRAM sharded weights
#ifdef IN1_DRAM_SHARDED
    const uint32_t vc = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t num_dram_shards_to_read = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t dram_tensor_start_offset = get_arg_val<uint32_t>(rt_args_idx++);
    tt_l1_ptr uint32_t* in1_block_w_dram_stride_bytes = (tt_l1_ptr uint32_t*)get_arg_addr(rt_args_idx++);
    tt_l1_ptr uint32_t* current_dram_bank_id = (tt_l1_ptr uint32_t*)get_arg_addr(rt_args_idx++);

    constexpr uint32_t in1_dram_block_num_tiles = get_compile_time_arg_val(after_bias_offset);
    constexpr uint32_t in1_block_w_dram_bytes = get_compile_time_arg_val(after_bias_offset + 1);
#endif  // IN1_DRAM_SHARDED

#ifdef FUSE_BIAS
#ifndef BIAS_SHARDED
    const auto s3 = TensorAccessor(bias_args, in3_tensor_addr, bias_single_tile_size_bytes);
#endif  // BIAS_SHARDED
#endif  // FUSE_BIAS

    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t in1_single_tile_size_bytes = get_tile_size(cb_id_in1);
    constexpr const uint32_t in1_tile_hw = get_tile_hw(cb_id_in1);
    constexpr uint32_t in1_block_size_bytes = in1_block_num_tiles * in1_single_tile_size_bytes;

//  READER
#ifdef IN1_SHARDED
    cb_reserve_back(cb_id_in1, in1_block_num_tiles * num_blocks_inner_dim);
    cb_push_back(cb_id_in1, in1_block_num_tiles * num_blocks_inner_dim);
#else
    uint32_t l1_write_addr_in1;

    const auto s1 = TensorAccessor(in1_args, in1_tensor_addr, in1_single_tile_size_bytes);
#endif  // IN1_SHARDED

    //  WRITER
    constexpr uint32_t cb_id_out0 = tt::CBIndex::c_4;
    constexpr uint32_t output_single_tile_size_bytes = get_tile_size(cb_id_out0);
    constexpr const uint32_t output_tile_hw = get_tile_hw(cb_id_out0);
    const auto s = TensorAccessor(out_args, out_tensor_addr, output_single_tile_size_bytes);

    // sparsity accessor
    constexpr uint32_t cb_id_sparsity = tt::CBIndex::c_7;
    const auto s_sparsity = TensorAccessor(sparsity_args, sparsity_addr, sparsity_pagesize);

#ifndef SKIP_MCAST
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
#ifdef IN1_SHARDED
    uint64_t in1_start_address = get_write_ptr(cb_id_in1);
#endif  // IN1_SHARDED
#endif  // SKIP_MCAST

    uint32_t l1_write_addr_sparsity = 0;
    if constexpr (batchB > 0) {
        cb_reserve_back(cb_id_sparsity, 1);
        l1_write_addr_sparsity = get_write_ptr(cb_id_sparsity);
    }

#ifdef IN1_DRAM_SHARDED
    constexpr uint32_t in1_dram_block_size_bytes = in1_dram_block_num_tiles * in1_single_tile_size_bytes;
    uint32_t in1_block_w_bytes = in1_block_w * in1_single_tile_size_bytes;
#endif  // IN1_DRAM_SHARDED

    for (uint32_t b = 0; b < batch; ++b) {
        uint32_t in1_batch_tile_id = in1_tensor_start_tile_id;

        if constexpr (batchB > 0) {
            noc_async_read_page(b, s_sparsity, l1_write_addr_sparsity);
            noc_async_read_barrier();
        }

        for (uint32_t bB = 0; bB < batchB_lim; ++bB) {
            if constexpr (batchB > 0) {
                if (reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_addr_sparsity)[bB] == 0) {
                    out_tensor_start_tile_id += MtNt;
                    in1_batch_tile_id += KtNt;
                    continue;
                }
            }

#ifdef IN1_DRAM_SHARDED
            uint32_t l1_read_addr_in1_offset = 0;
#endif  // IN1_DRAM_SHARDED
            uint32_t in1_tensor_current_h_dim_block_tile_id = in1_batch_tile_id;
            uint32_t out_tensor_current_h_dim_block_tile_id = out_tensor_start_tile_id;
            for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh) {
                uint32_t in1_tensor_current_w_dim_block_tile_id = in1_tensor_current_h_dim_block_tile_id;
                uint32_t out_tensor_current_w_dim_block_tile_id = out_tensor_current_h_dim_block_tile_id;
#ifdef FUSE_BIAS
                uint32_t in3_tensor_current_w_dim_block_tile_id = in3_tensor_start_tile_id;
#endif  // FUSE_BIAS
                for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw) {
                    uint32_t in1_tensor_current_inner_dim_block_start_tile_id = in1_tensor_current_w_dim_block_tile_id;

                    for (uint32_t block = 0; block < num_blocks_inner_dim; ++block) {
                        if constexpr (fuse_op_all_gather) {
                            fused_op_receiver.update_current_block_start_tile_id(
                                block, in1_tensor_current_inner_dim_block_start_tile_id, in1_batch_tile_id);
                        }
#ifdef IN1_DRAM_SHARDED
                        // Operand 1
                        cb_reserve_back(cb_id_in1, in1_block_num_tiles);

                        uint64_t in1_start_address =
                            get_write_ptr(cb_id_in1);  // copy start address of block, to be used for mcasting

                        uint32_t l1_write_addr_in1_offset = 0;
                        uint32_t next_bank_id_and_dram_stride_index = 0;

                        for (uint32_t i = 0; i < num_dram_shards_to_read; ++i) {
                            uint64_t in1_base_addr = get_noc_addr_from_bank_id<true>(
                                current_dram_bank_id[next_bank_id_and_dram_stride_index], in1_tensor_addr);

                            if (i == 0) {
                                in1_base_addr += dram_tensor_start_offset;
                            }
                            noc_async_read_one_packet_set_state<true>(in1_base_addr, in1_single_tile_size_bytes, vc);

                            uint32_t l1_read_addr_in1 = l1_read_addr_in1_offset;
                            uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1) + l1_write_addr_in1_offset;
                            uint32_t in1_block_w_dram =
                                in1_block_w_dram_stride_bytes[next_bank_id_and_dram_stride_index] /
                                in1_single_tile_size_bytes;

                            for (uint32_t m = 0; m < in1_block_h; ++m) {
                                uint32_t l1_read_addr_in1_temp = l1_read_addr_in1;
                                uint32_t l1_write_addr_in1_temp = l1_write_addr_in1;
                                for (uint32_t w = 0; w < in1_block_w_dram; ++w) {
                                    noc_async_read_one_packet_with_state<true, true>(
                                        in1_base_addr + l1_read_addr_in1_temp, l1_write_addr_in1_temp, vc);
                                    l1_read_addr_in1_temp += in1_single_tile_size_bytes;
                                    l1_write_addr_in1_temp += in1_single_tile_size_bytes;
                                }
                                l1_read_addr_in1 += in1_block_w_dram_bytes;
                                l1_write_addr_in1 += in1_block_w_bytes;
                            }
                            l1_write_addr_in1_offset +=
                                in1_block_w_dram_stride_bytes[next_bank_id_and_dram_stride_index];
                            next_bank_id_and_dram_stride_index += 2;
                        }
                        l1_read_addr_in1_offset += in1_dram_block_size_bytes;
                        noc_async_read_barrier();
#else
#ifndef IN1_SHARDED
                        // Operand 1
                        cb_reserve_back(cb_id_in1, in1_block_num_tiles);
#ifdef INTERMEDIATE_CB_READ
                        constexpr uint32_t in1_intermediate_cb_index = tt::CBIndex::c_9;
                        cb_reserve_back(in1_intermediate_cb_index, one_tile);
                        uint32_t l1_write_addr_helper = get_write_ptr(in1_intermediate_cb_index);
#endif  // INTERMEDIATE_CB_READ
                        l1_write_addr_in1 = get_write_ptr(cb_id_in1);
                        uint64_t in1_start_address =
                            l1_write_addr_in1;  // copy start address of block, to be used for mcasting

                        // Copy in1 block into CB, as the default kernel
                        uint32_t in1_tensor_row_start_tile_id = in1_tensor_current_inner_dim_block_start_tile_id;
                        for (uint32_t h = 0; h < in1_block_h; ++h) {
                            uint32_t in1_tensor_tile_id = in1_tensor_row_start_tile_id;
                            for (uint32_t w = 0; w < in1_block_w; ++w) {
                                if (bw < num_blocks_w_dim - 1 || w < last_block_w) {
#ifndef INTERMEDIATE_CB_READ
                                    noc_async_read_tile(in1_tensor_tile_id, s1, l1_write_addr_in1);
#else
                                    noc_async_read_tile(in1_tensor_tile_id, s1, l1_write_addr_helper);
                                    noc_async_read_barrier();
                                    memcpy(
                                        /*dst=*/reinterpret_cast<void*>(l1_write_addr_in1),
                                        /*src=*/reinterpret_cast<const void*>(l1_write_addr_helper),
                                        /*size=*/in1_single_tile_size_bytes);
#endif  // INTERMEDIATE_CB_READ
                                }
                                l1_write_addr_in1 += in1_single_tile_size_bytes;
                                in1_tensor_tile_id += in1_tensor_stride_w;
                            }
                            in1_tensor_row_start_tile_id += in1_tensor_stride_h;
                        }
                        in1_tensor_current_inner_dim_block_start_tile_id += in1_tensor_next_block_stride;

                        // Barrier! make sure the reads are done
                        noc_async_read_barrier();
#endif  // IN1_SHARDED
#endif  // IN1_DRAM_SHARDED

#ifndef SKIP_MCAST
                        // wait until all in1 mcast destinations have atomically incremented the in1 semaphore_addr
                        // (i.e. its value should be in0_mcast_num_dests), then reset the semaphore_addr value back to
                        // zero for the next block
                        noc_semaphore_wait(in1_mcast_sender_semaphore_addr_ptr, in1_mcast_num_dests);
                        noc_semaphore_set(in1_mcast_sender_semaphore_addr_ptr, 0);

                        // Now we have the block in the CB address, we can mcast to dests!
                        uint64_t in1_multicast_data_addr = in1_multicast_data_noc | in1_start_address;

                        // num_dests must not include source, since we are NOT really doing a local copy!
                        noc_async_write_multicast(
                            in1_start_address,
                            in1_multicast_data_addr,
                            in1_block_size_bytes,
                            in1_mcast_num_cores,
                            true);

                        // Note: no need for write barrier, since these two multicasts are done on the same noc id and
                        // same vc even though cmd bufs are different Also, this only works because we are setting VCs
                        // statically (using NOC_CMD_STATIC_VC).
#ifdef ARCH_BLACKHOLE
                        // On Blackhole the flush is needed because NoC latency is higher than L1 <-> RISCV latency
                        // which means data could be changed before
                        //  write is issued.
                        noc_async_writes_flushed();
#endif  // ARCH_BLACKHOLE

                        // We should also multicast the flag to destinations
                        // num_dests must not include source, since we are NOT really doing a local copy!
                        noc_semaphore_set_multicast(
                            in1_mcast_receiver_semaphore_addr,
                            in1_mcast_receiver_semaphore_noc_addr,
                            in1_mcast_num_cores);
#endif  // SKIP_MCAST

#ifndef IN1_SHARDED
                        cb_push_back(cb_id_in1, in1_block_num_tiles);
#ifdef INTERMEDIATE_CB_READ
                        // Clean up helper CB
                        cb_push_back(in1_intermediate_cb_index, one_tile);
                        cb_wait_front(in1_intermediate_cb_index, one_tile);
                        cb_pop_front(in1_intermediate_cb_index, one_tile);
#endif  // INTERMEDIATE_CB_READ
#endif  // IN1_SHARDED
                    }
#ifdef FUSE_BIAS
                    // Only read bias on first batch, or we have multiple output blocks
                    if ((b == 0 && bh == 0) || num_blocks_w_dim > 1) {
                        // Operand 1
#ifndef BIAS_SHARDED
                        cb_reserve_back(cb_id_in3, in1_block_w);
                        l1_write_addr_in3 = get_write_ptr(cb_id_in3);

                        uint64_t in3_start_address =
                            l1_write_addr_in3;              // copy start address of block, to be used for mcasting
                        uint32_t in3_block_size_bytes = 0;  // can be optimized later, pass it to kernel

#ifdef IN1_DRAM_SHARDED
                        uint32_t l1_write_addr_in3_offset = 0;
                        uint32_t next_bank_id_and_dram_stride_index = 0;

                        for (uint32_t i = 0; i < num_dram_shards_to_read; ++i) {
                            uint64_t in3_base_addr = get_noc_addr_from_bank_id<true>(
                                current_dram_bank_id[next_bank_id_and_dram_stride_index], in3_tensor_addr);

                            if (i == 0) {
                                in3_base_addr += dram_tensor_start_offset;
                            }

                            noc_async_read_one_packet_set_state<true>(in3_base_addr, bias_single_tile_size_bytes, vc);

                            uint32_t l1_read_addr_in3 = 0;
                            uint32_t l1_write_addr_in3 = get_write_ptr(cb_id_in3) + l1_write_addr_in3_offset;
                            uint32_t in3_block_w_dram =
                                in1_block_w_dram_stride_bytes[next_bank_id_and_dram_stride_index] /
                                bias_single_tile_size_bytes;

                            for (uint32_t w = 0; w < in3_block_w_dram; ++w) {
                                noc_async_read_one_packet_with_state<true, true>(
                                    in3_base_addr + l1_read_addr_in3, l1_write_addr_in3, vc);
                                l1_read_addr_in3 += bias_single_tile_size_bytes;
                                l1_write_addr_in3 += bias_single_tile_size_bytes;
                                in3_block_size_bytes += bias_single_tile_size_bytes;
                            }
                            l1_write_addr_in3_offset +=
                                in1_block_w_dram_stride_bytes[next_bank_id_and_dram_stride_index];
                            next_bank_id_and_dram_stride_index += 2;
                        }
                        noc_async_read_barrier();
#else
                        // Copy in1 block into CB, as the default kernel
                        uint32_t in3_tensor_tile_id = in3_tensor_current_w_dim_block_tile_id;
                        for (uint32_t w = 0; w < in1_block_w; ++w) {
                            if (bw < num_blocks_w_dim - 1 || w < last_block_w) {
                                noc_async_read_tile(in3_tensor_tile_id, s3, l1_write_addr_in3);
                            }
                            l1_write_addr_in3 += bias_single_tile_size_bytes;
                            in3_tensor_tile_id += in3_tensor_stride_w;
                            in3_block_size_bytes += bias_single_tile_size_bytes;
                        }
                        // Barrier! make sure the reads are done
                        noc_async_read_barrier();
#endif  // IN1_DRAM_SHARDED

#ifndef SKIP_MCAST

                        // wait until all in1 mcast destinations have atomically incremented the in1 semaphore_addr
                        // (i.e. its value should be in0_mcast_num_dests), then reset the semaphore_addr value back to
                        // zero for the next block
                        noc_semaphore_wait(in1_mcast_sender_semaphore_addr_ptr, in1_mcast_num_dests);
                        noc_semaphore_set(in1_mcast_sender_semaphore_addr_ptr, 0);

                        // Now we have the block in the CB address, we can mcast to dests!
                        uint64_t in3_multicast_data_addr = in1_multicast_data_noc | in3_start_address;

                        // num_dests must not include source, since we are NOT really doing a local copy!
                        noc_async_write_multicast(
                            in3_start_address,
                            in3_multicast_data_addr,
                            in3_block_size_bytes,
                            in1_mcast_num_cores,
                            true);
                        // Note: no need for write barrier, since these two multicasts are done on the same noc id, same
                        // vc, same cmd_buf Also, this only works because we are setting VCs statically (using
                        // NOC_CMD_STATIC_VC).
#ifdef ARCH_BLACKHOLE
                        // On Blackhole the flush is needed because NoC latency is higherthan L1 <-> RISCV
                        // latency which means data could be changed before write is issued.
                        noc_async_writes_flushed();
#endif  // ARCH_BLACKHOLE

                        // We should also multicast the flag to destinations
                        // num_dests must not include source, since we are NOT really doing a local copy!
                        noc_semaphore_set_multicast(
                            in1_mcast_receiver_semaphore_addr,
                            in1_mcast_receiver_semaphore_noc_addr,
                            in1_mcast_num_cores);
#endif  // SKIP_MCAST

                        cb_push_back(cb_id_in3, in1_block_w);
#else
                        cb_reserve_back(cb_id_in3, in1_block_w);
                        cb_push_back(cb_id_in3, in1_block_w);
#endif  // BIAS_SHARDED
                    }
#endif  // FUSE_BIAS

#ifndef OUT_SHARDED
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

#endif
                    in1_tensor_current_w_dim_block_tile_id += in1_tensor_next_w_dim_block_stride;
                    out_tensor_current_w_dim_block_tile_id += out_tensor_next_w_dim_block_stride;
#ifdef FUSE_BIAS
                    in3_tensor_current_w_dim_block_tile_id += in1_block_w;
#endif
                }
                out_tensor_current_h_dim_block_tile_id += out_tensor_next_h_dim_block_stride;
            }
            out_tensor_start_tile_id += MtNt;
            in1_batch_tile_id += KtNt;
        }
        if constexpr (bcast_B == 0) {
            in1_tensor_start_tile_id += KtNt;
        }

        if (fuse_op_reduce_scatter) {
            // Signal reduce_scatter to go
            op_signaler.synchronize_workers_and_signal_op(0);
        }
    }

#if OUT_SHARDED
    cb_wait_front(
        cb_id_out0,
        batch * out_num_nonzero_subblocks_h * out_num_nonzero_subblocks_w * out_subblock_w * out_subblock_h);
#endif
    noc_async_write_barrier();
}
