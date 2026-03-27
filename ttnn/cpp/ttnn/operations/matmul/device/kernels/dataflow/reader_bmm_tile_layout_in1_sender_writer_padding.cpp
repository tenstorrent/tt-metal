// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/noc_semaphore.h"
#include "experimental/tensor.h"
#include "experimental/endpoints.h"
#include "experimental/core_local_mem.h"

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

    constexpr uint32_t cb_id_in3 = get_named_compile_time_arg_val("cb_bias");
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
#ifdef IN1_DRAM_WIDTH_SHARDED
    const uint32_t vc = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t num_dram_shards_to_read = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t dram_tensor_start_offset = get_arg_val<uint32_t>(rt_args_idx++);
    tt_l1_ptr uint32_t* in1_block_w_dram_stride_bytes = (tt_l1_ptr uint32_t*)get_arg_addr(rt_args_idx++);
    tt_l1_ptr uint32_t* current_dram_bank_id = (tt_l1_ptr uint32_t*)get_arg_addr(rt_args_idx++);

    constexpr uint32_t in1_dram_block_num_tiles = get_compile_time_arg_val(after_bias_offset);
    constexpr uint32_t in1_block_w_dram_bytes = get_compile_time_arg_val(after_bias_offset + 1);
#endif  // IN1_DRAM_WIDTH_SHARDED

#ifdef IN1_DRAM_HEIGHT_SHARDED
    constexpr uint32_t in1_KtNt_per_batch = get_compile_time_arg_val(after_bias_offset);        // K*N tiles per batch
    constexpr uint32_t in1_batches_per_bank = get_compile_time_arg_val(after_bias_offset + 1);  // batches per DRAM bank
#endif  // IN1_DRAM_HEIGHT_SHARDED

#ifdef FUSE_BIAS
#ifndef BIAS_SHARDED
    const auto s3 = TensorAccessor(bias_args, in3_tensor_addr, bias_single_tile_size_bytes);
#endif  // BIAS_SHARDED
#endif  // FUSE_BIAS

    constexpr uint32_t cb_id_in1 = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t in1_single_tile_size_bytes = get_tile_size(cb_id_in1);
    constexpr const uint32_t in1_tile_hw = get_tile_hw(cb_id_in1);
    constexpr uint32_t in1_block_size_bytes = in1_block_num_tiles * in1_single_tile_size_bytes;

    constexpr uint32_t cb_id_out0 = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t output_single_tile_size_bytes = get_tile_size(cb_id_out0);
    constexpr const uint32_t output_tile_hw = get_tile_hw(cb_id_out0);

    experimental::Noc noc;
    experimental::CircularBuffer cb_in1(cb_id_in1);
    experimental::CircularBuffer cb_out(cb_id_out0);
    experimental::Semaphore<> sender_sem(get_compile_time_arg_val(10));
    experimental::Semaphore<> receiver_sem(get_compile_time_arg_val(11));
#ifdef FUSE_BIAS
    experimental::CircularBuffer cb_in3(cb_id_in3);
#endif
#if !defined(IN1_SHARDED) && !defined(IN1_DRAM_WIDTH_SHARDED) && !defined(IN1_DRAM_HEIGHT_SHARDED)
#ifdef INTERMEDIATE_CB_READ
    constexpr uint32_t in1_intermediate_cb_index = get_named_compile_time_arg_val("cb_in1_intermediate");
    experimental::CircularBuffer cb_helper(in1_intermediate_cb_index);
#endif
#endif

//  READER
#ifdef IN1_SHARDED
    cb_in1.reserve_back(in1_block_num_tiles * num_blocks_inner_dim);
    cb_in1.push_back(in1_block_num_tiles * num_blocks_inner_dim);
#else
    uint32_t l1_write_addr_in1;

    const auto s1 = TensorAccessor(in1_args, in1_tensor_addr, in1_single_tile_size_bytes);
#endif  // IN1_SHARDED

    //  WRITER
    const auto s = TensorAccessor(out_args, out_tensor_addr, output_single_tile_size_bytes);

    // sparsity accessor
    constexpr uint32_t cb_id_sparsity = get_named_compile_time_arg_val("cb_sparsity");
    experimental::CircularBuffer cb_sparsity(cb_id_sparsity);
    const auto s_sparsity = TensorAccessor(sparsity_args, sparsity_addr, sparsity_pagesize);

#ifndef SKIP_MCAST
    // Set ur local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    receiver_sem.set(VALID);
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast

#ifdef IN1_SHARDED
    uint64_t in1_start_address = cb_in1.get_write_ptr();
#endif  // IN1_SHARDED
#endif  // SKIP_MCAST

    uint32_t l1_write_addr_sparsity = 0;
    if constexpr (batchB > 0) {
        cb_sparsity.reserve_back(1);
        l1_write_addr_sparsity = cb_sparsity.get_write_ptr();
    }

#ifdef IN1_DRAM_WIDTH_SHARDED
    constexpr uint32_t in1_dram_block_size_bytes = in1_dram_block_num_tiles * in1_single_tile_size_bytes;
    uint32_t in1_block_w_bytes = in1_block_w * in1_single_tile_size_bytes;
#endif  // IN1_DRAM_WIDTH_SHARDED

#ifdef IN1_DRAM_HEIGHT_SHARDED
    constexpr uint32_t in1_batch_stride_bytes = in1_KtNt_per_batch * in1_single_tile_size_bytes;
#endif  // IN1_DRAM_HEIGHT_SHARDED

    for (uint32_t b = 0; b < batch; ++b) {
        uint32_t in1_batch_tile_id = in1_tensor_start_tile_id;

#ifdef IN1_DRAM_HEIGHT_SHARDED
        // Compute DRAM bank and offset for this batch
        uint32_t in1_dram_bank_id = b / in1_batches_per_bank;
        uint32_t in1_batch_in_shard = b % in1_batches_per_bank;
        experimental::AllocatorBank<experimental::AllocatorBankType::DRAM> dram_src;
        uint32_t in1_dram_batch_offset = in1_batch_in_shard * in1_batch_stride_bytes;
#endif  // IN1_DRAM_HEIGHT_SHARDED

        if constexpr (batchB > 0) {
            noc.async_read(
                s_sparsity,
                experimental::CoreLocalMem<uint32_t>(l1_write_addr_sparsity),
                sparsity_pagesize,
                {.page_id = b},
                {});
            noc.async_read_barrier();
        }

        for (uint32_t bB = 0; bB < batchB_lim; ++bB) {
            if constexpr (batchB > 0) {
                if (reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_addr_sparsity)[bB] == 0) {
                    out_tensor_start_tile_id += MtNt;
                    in1_batch_tile_id += KtNt;
                    continue;
                }
            }

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
#ifdef IN1_DRAM_WIDTH_SHARDED
                    // Reset DRAM read offset for each bh block — the inner dim loop
                    // advances through K, and each output row block re-reads the same
                    // in1 columns from K=0. (bw is always 1 for DRAM-sharded senders.)
                    uint32_t l1_read_addr_in1_offset = 0;
#endif  // IN1_DRAM_WIDTH_SHARDED

                    for (uint32_t block = 0; block < num_blocks_inner_dim; ++block) {
                        if constexpr (fuse_op_all_gather) {
                            fused_op_receiver.update_current_block_start_tile_id(
                                block, in1_tensor_current_inner_dim_block_start_tile_id, in1_batch_tile_id);
                        }
#if defined(IN1_DRAM_WIDTH_SHARDED)
                        // Operand 1 - DRAM width sharded
                        cb_in1.reserve_back(in1_block_num_tiles);

                        uint64_t in1_start_address =
                            cb_in1.get_write_ptr();  // copy start address of block, to be used for mcasting

                        uint32_t l1_write_addr_in1_offset = 0;
                        uint32_t next_bank_id_and_dram_stride_index = 0;

                        experimental::AllocatorBank<experimental::AllocatorBankType::DRAM> dram_bank;
                        for (uint32_t i = 0; i < num_dram_shards_to_read; ++i) {
                            uint32_t shard_bank_id = current_dram_bank_id[next_bank_id_and_dram_stride_index];
                            uint32_t shard_base_offset = (i == 0) ? dram_tensor_start_offset : 0;

                            uint32_t l1_read_addr_in1 = l1_read_addr_in1_offset;
                            uint32_t cb_write_offset_in1 = l1_write_addr_in1_offset;
                            uint32_t in1_block_w_dram =
                                in1_block_w_dram_stride_bytes[next_bank_id_and_dram_stride_index] /
                                in1_single_tile_size_bytes;

                            for (uint32_t m = 0; m < in1_block_h; ++m) {
                                uint32_t l1_read_addr_in1_temp = l1_read_addr_in1;
                                uint32_t cb_write_offset_temp = cb_write_offset_in1;
                                for (uint32_t w = 0; w < in1_block_w_dram; ++w) {
                                    noc.async_read<Noc::TxnIdMode::DISABLED, in1_single_tile_size_bytes>(
                                        dram_bank,
                                        cb_in1,
                                        in1_single_tile_size_bytes,
                                        {.bank_id = shard_bank_id,
                                         .addr = in1_tensor_addr + shard_base_offset + l1_read_addr_in1_temp},
                                        {.offset_bytes = cb_write_offset_temp},
                                        vc);
                                    l1_read_addr_in1_temp += in1_single_tile_size_bytes;
                                    cb_write_offset_temp += in1_single_tile_size_bytes;
                                }
                                l1_read_addr_in1 += in1_block_w_dram_bytes;
                                cb_write_offset_in1 += in1_block_w_bytes;
                            }
                            l1_write_addr_in1_offset +=
                                in1_block_w_dram_stride_bytes[next_bank_id_and_dram_stride_index];
                            next_bank_id_and_dram_stride_index += 2;
                        }
                        l1_read_addr_in1_offset += in1_dram_block_size_bytes;
                        noc.async_read_barrier();
#elif defined(IN1_DRAM_HEIGHT_SHARDED)
                        // Operand 1 - DRAM height sharded (batched)
                        // Each DRAM bank holds batches_per_bank complete [K, N] matrices
                        // Bank and offset computed at start of batch loop
                        cb_in1.reserve_back(in1_block_num_tiles);

                        l1_write_addr_in1 = cb_in1.get_write_ptr();
                        uint64_t in1_start_address =
                            l1_write_addr_in1;  // copy start address of block, to be used for mcasting

                        // Read in1 block from the correct DRAM bank
                        // Tile layout within a batch: row-major [K, N], same strides as interleaved
                        uint32_t in1_tensor_row_start_tile_id = in1_tensor_current_inner_dim_block_start_tile_id;
                        for (uint32_t h = 0; h < in1_block_h; ++h) {
                            uint32_t in1_tensor_tile_id = in1_tensor_row_start_tile_id;
                            for (uint32_t w = 0; w < in1_block_w; ++w) {
                                if (bw < num_blocks_w_dim - 1 || w < last_block_w) {
                                    uint32_t tile_byte_offset =
                                        in1_dram_batch_offset + in1_tensor_tile_id * in1_single_tile_size_bytes;
                                    noc.async_read(
                                        dram_src,
                                        experimental::CoreLocalMem<uint32_t>(l1_write_addr_in1),
                                        in1_single_tile_size_bytes,
                                        {.bank_id = in1_dram_bank_id, .addr = in1_tensor_addr + tile_byte_offset},
                                        {});
                                }
                                l1_write_addr_in1 += in1_single_tile_size_bytes;
                                in1_tensor_tile_id += in1_tensor_stride_w;
                            }
                            in1_tensor_row_start_tile_id += in1_tensor_stride_h;
                        }
                        in1_tensor_current_inner_dim_block_start_tile_id += in1_tensor_next_block_stride;

                        // Barrier! make sure the reads are done
                        noc.async_read_barrier();
#elif !defined(IN1_SHARDED)
                        // Operand 1 - interleaved
                        cb_in1.reserve_back(in1_block_num_tiles);
#ifdef INTERMEDIATE_CB_READ
                        cb_helper.reserve_back(one_tile);
#endif  // INTERMEDIATE_CB_READ
                        uint32_t in1_write_offset = 0;
                        uint64_t in1_start_address =
                            cb_in1.get_write_ptr();  // copy start address of block, to be used for mcasting

                        // Copy in1 block into CB, as the default kernel
                        uint32_t in1_tensor_row_start_tile_id = in1_tensor_current_inner_dim_block_start_tile_id;
                        for (uint32_t h = 0; h < in1_block_h; ++h) {
                            uint32_t in1_tensor_tile_id = in1_tensor_row_start_tile_id;
                            for (uint32_t w = 0; w < in1_block_w; ++w) {
                                if (bw < num_blocks_w_dim - 1 || w < last_block_w) {
#ifndef INTERMEDIATE_CB_READ
                                    noc.async_read(
                                        s1,
                                        cb_in1,
                                        in1_single_tile_size_bytes,
                                        {.page_id = in1_tensor_tile_id},
                                        {.offset_bytes = in1_write_offset});
#else
                                    noc.async_read(
                                        s1,
                                        cb_helper,
                                        in1_single_tile_size_bytes,
                                        {.page_id = in1_tensor_tile_id},
                                        {.offset_bytes = 0});
                                    noc.async_read_barrier();
                                    memcpy(
                                        /*dst=*/reinterpret_cast<void*>(cb_in1.get_write_ptr() + in1_write_offset),
                                        /*src=*/reinterpret_cast<const void*>(cb_helper.get_write_ptr()),
                                        /*size=*/in1_single_tile_size_bytes);
#endif  // INTERMEDIATE_CB_READ
                                }
                                in1_write_offset += in1_single_tile_size_bytes;
                                in1_tensor_tile_id += in1_tensor_stride_w;
                            }
                            in1_tensor_row_start_tile_id += in1_tensor_stride_h;
                        }
                        in1_tensor_current_inner_dim_block_start_tile_id += in1_tensor_next_block_stride;

                        // Barrier! make sure the reads are done
                        noc.async_read_barrier();
#endif  // IN1_DRAM_WIDTH_SHARDED / IN1_DRAM_HEIGHT_SHARDED / IN1_SHARDED

#ifndef SKIP_MCAST
                        // wait until all in1 mcast destinations have atomically incremented the in1 semaphore_addr
                        // (i.e. its value should be in0_mcast_num_dests), then reset the semaphore_addr value back to
                        // zero for the next block
                        sender_sem.wait(in1_mcast_num_dests);
                        sender_sem.set(0);

                        // Now we have the block in the CB address, we can mcast to dests!
                        experimental::MulticastEndpoint mcast_dst;
                        // num_dests must not include source, since we are NOT really doing a local copy!
                        noc.async_write_multicast(
                            experimental::CoreLocalMem<uint32_t>(static_cast<uint32_t>(in1_start_address)),
                            mcast_dst,
                            in1_block_size_bytes,
                            in1_mcast_num_cores,
                            {},
                            {.noc_x_start = in1_mcast_dest_noc_start_x,
                             .noc_y_start = in1_mcast_dest_noc_start_y,
                             .noc_x_end = in1_mcast_dest_noc_end_x,
                             .noc_y_end = in1_mcast_dest_noc_end_y,
                             .addr = static_cast<uint32_t>(in1_start_address)},
                            true);

                        // Note: no need for write barrier, since these two multicasts are done on the same noc id and
                        // same vc even though cmd bufs are different Also, this only works because we are setting VCs
                        // statically (using NOC_CMD_STATIC_VC).
#ifdef ARCH_BLACKHOLE
                        // On Blackhole the flush is needed because NoC latency is higher than L1 <-> RISCV latency
                        // which means data could be changed before
                        //  write is issued.
                        noc.async_writes_flushed();
#endif  // ARCH_BLACKHOLE

                        // We should also multicast the flag to destinations
                        // num_dests must not include source, since we are NOT really doing a local copy!
                        receiver_sem.set_multicast(
                            noc,
                            in1_mcast_dest_noc_start_x,
                            in1_mcast_dest_noc_start_y,
                            in1_mcast_dest_noc_end_x,
                            in1_mcast_dest_noc_end_y,
                            in1_mcast_num_cores);
#endif  // SKIP_MCAST

#ifndef IN1_SHARDED
                        cb_in1.push_back(in1_block_num_tiles);
#ifdef INTERMEDIATE_CB_READ
                        // Clean up helper CB
                        cb_helper.push_back(one_tile);
                        cb_helper.wait_front(one_tile);
                        cb_helper.pop_front(one_tile);
#endif  // INTERMEDIATE_CB_READ
#endif  // IN1_SHARDED
                    }
#ifdef FUSE_BIAS
                    // Only read bias on first batch, or we have multiple output blocks
                    if ((b == 0 && bh == 0) || num_blocks_w_dim > 1) {
                        // Operand 1
#ifndef BIAS_SHARDED
                        cb_in3.reserve_back(in1_block_w);
                        uint32_t in3_write_offset = 0;

                        uint64_t in3_start_address =
                            cb_in3.get_write_ptr();         // copy start address of block, to be used for mcasting
                        uint32_t in3_block_size_bytes = 0;  // can be optimized later, pass it to kernel

#ifdef IN1_DRAM_WIDTH_SHARDED
                        uint32_t l1_write_addr_in3_offset = 0;
                        uint32_t next_bank_id_and_dram_stride_index = 0;

                        experimental::AllocatorBank<experimental::AllocatorBankType::DRAM> dram_bank_bias;
                        for (uint32_t i = 0; i < num_dram_shards_to_read; ++i) {
                            uint32_t bias_bank_id = current_dram_bank_id[next_bank_id_and_dram_stride_index];
                            // dram_tensor_start_offset is in in1 tile bytes; convert to
                            // bias tile bytes since bias_dtype may differ from in1_dtype.
                            uint32_t bias_base_offset = (i == 0)
                                                            ? (dram_tensor_start_offset / in1_single_tile_size_bytes) *
                                                                  bias_single_tile_size_bytes
                                                            : 0;

                            uint32_t l1_read_addr_in3 = 0;
                            uint32_t cb_write_offset_in3 = l1_write_addr_in3_offset;
                            // in1_block_w_dram_stride_bytes is in in1 tile bytes, so divide
                            // by in1_single_tile_size_bytes (not bias) to get the tile count.
                            uint32_t in3_block_w_dram =
                                in1_block_w_dram_stride_bytes[next_bank_id_and_dram_stride_index] /
                                in1_single_tile_size_bytes;

                            for (uint32_t w = 0; w < in3_block_w_dram; ++w) {
                                noc.async_read<Noc::TxnIdMode::DISABLED, bias_single_tile_size_bytes>(
                                    dram_bank_bias,
                                    cb_in3,
                                    bias_single_tile_size_bytes,
                                    {.bank_id = bias_bank_id,
                                     .addr = in3_tensor_addr + bias_base_offset + l1_read_addr_in3},
                                    {.offset_bytes = cb_write_offset_in3},
                                    vc);
                                l1_read_addr_in3 += bias_single_tile_size_bytes;
                                cb_write_offset_in3 += bias_single_tile_size_bytes;
                                in3_block_size_bytes += bias_single_tile_size_bytes;
                            }
                            // Advance L1 offset in bias tile bytes, not in1 stride bytes.
                            l1_write_addr_in3_offset += in3_block_w_dram * bias_single_tile_size_bytes;
                            next_bank_id_and_dram_stride_index += 2;
                        }
                        noc.async_read_barrier();
#else
                        // Copy in1 block into CB, as the default kernel
                        uint32_t in3_tensor_tile_id = in3_tensor_current_w_dim_block_tile_id;
                        for (uint32_t w = 0; w < in1_block_w; ++w) {
                            if (bw < num_blocks_w_dim - 1 || w < last_block_w) {
                                noc.async_read(
                                    s3,
                                    cb_in3,
                                    bias_single_tile_size_bytes,
                                    {.page_id = in3_tensor_tile_id},
                                    {.offset_bytes = in3_write_offset});
                            }
                            in3_write_offset += bias_single_tile_size_bytes;
                            in3_tensor_tile_id += in3_tensor_stride_w;
                            in3_block_size_bytes += bias_single_tile_size_bytes;
                        }
                        // Barrier! make sure the reads are done
                        noc.async_read_barrier();
#endif  // IN1_DRAM_WIDTH_SHARDED

#ifndef SKIP_MCAST

                        // wait until all in1 mcast destinations have atomically incremented the in1 semaphore_addr
                        // (i.e. its value should be in0_mcast_num_dests), then reset the semaphore_addr value back to
                        // zero for the next block
                        sender_sem.wait(in1_mcast_num_dests);
                        sender_sem.set(0);

                        // Now we have the block in the CB address, we can mcast to dests!
                        experimental::MulticastEndpoint mcast_dst;
                        // num_dests must not include source, since we are NOT really doing a local copy!
                        noc.async_write_multicast(
                            experimental::CoreLocalMem<uint32_t>(static_cast<uint32_t>(in3_start_address)),
                            mcast_dst,
                            in3_block_size_bytes,
                            in1_mcast_num_cores,
                            {},
                            {.noc_x_start = in1_mcast_dest_noc_start_x,
                             .noc_y_start = in1_mcast_dest_noc_start_y,
                             .noc_x_end = in1_mcast_dest_noc_end_x,
                             .noc_y_end = in1_mcast_dest_noc_end_y,
                             .addr = static_cast<uint32_t>(in3_start_address)},
                            true);
                        // Note: no need for write barrier, since these two multicasts are done on the same noc id, same
                        // vc, same cmd_buf Also, this only works because we are setting VCs statically (using
                        // NOC_CMD_STATIC_VC).
#ifdef ARCH_BLACKHOLE
                        // On Blackhole the flush is needed because NoC latency is higherthan L1 <-> RISCV
                        // latency which means data could be changed before write is issued.
                        noc.async_writes_flushed();
#endif  // ARCH_BLACKHOLE

                        // We should also multicast the flag to destinations
                        // num_dests must not include source, since we are NOT really doing a local copy!
                        receiver_sem.set_multicast(
                            noc,
                            in1_mcast_dest_noc_start_x,
                            in1_mcast_dest_noc_start_y,
                            in1_mcast_dest_noc_end_x,
                            in1_mcast_dest_noc_end_y,
                            in1_mcast_num_cores);
#endif  // SKIP_MCAST

                        cb_in3.push_back(in1_block_w);
#else
                        cb_in3.reserve_back(in1_block_w);
                        cb_in3.push_back(in1_block_w);
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

                            cb_out.wait_front(out_subblock_tile_count);
                            uint32_t out_read_offset = 0;

                            for (uint32_t h = 0; h < out_subblock_h_; ++h) {
                                uint32_t out_tensor_tile_id = out_tensor_sb_row_start_tile_id;
                                for (uint32_t w = 0; w < out_subblock_w_; ++w) {
                                    if (bw < num_blocks_w_dim_) {
                                        noc.async_write(
                                            experimental::use<experimental::CircularBuffer::AddrSelector::READ_PTR>(
                                                cb_out),
                                            s,
                                            output_single_tile_size_bytes,
                                            {.offset_bytes = out_read_offset},
                                            {.page_id = out_tensor_tile_id});
                                    }

                                    out_read_offset += output_single_tile_size_bytes;

                                    out_tensor_tile_id += out_tensor_stride_w;
                                }
                                // Skip padded tiles in subblock along row
                                out_read_offset += subblock_tiles_addr_skip;
                                out_tensor_sb_row_start_tile_id += out_tensor_stride_h;
                            }

                            noc.async_write_barrier();
                            cb_out.pop_front(out_subblock_tile_count);
                            out_tensor_sbw_start_tile_id += out_tensor_next_subblock_stride_w;
                        }
                        // Pop fully padded subblocks along the row
                        if (bw == num_blocks_w_dim_ - 1) {
                            cb_out.wait_front(padded_block_tiles_w_skip);
                            cb_out.pop_front(padded_block_tiles_w_skip);
                        }
                        out_tensor_sbh_start_tile_id += out_tensor_next_subblock_stride_h;
                    }
                    // Pop row(s) of fully padded subblocks
                    if (bh == num_blocks_h_dim - 1) {
                        cb_out.wait_front(padded_block_tiles_h_skip);
                        cb_out.pop_front(padded_block_tiles_h_skip);
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
#ifndef IN1_DRAM_HEIGHT_SHARDED
            // For height-sharded DRAM, tile IDs are relative within a batch;
            // batch offset is handled by switching DRAM banks
            in1_tensor_start_tile_id += KtNt;
#endif
        }

        if (fuse_op_reduce_scatter) {
            // Signal reduce_scatter to go
            op_signaler.synchronize_workers_and_signal_op(0);
        }
    }

#if OUT_SHARDED
    cb_out.wait_front(
        batch * out_num_nonzero_subblocks_h * out_num_nonzero_subblocks_w * out_subblock_w * out_subblock_h);
#endif
    noc.async_write_barrier();
}
