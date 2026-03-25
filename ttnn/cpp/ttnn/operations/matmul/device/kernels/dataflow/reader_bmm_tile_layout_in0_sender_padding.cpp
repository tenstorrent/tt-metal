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
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/noc_semaphore.h"
#include "experimental/tensor.h"
#include "experimental/endpoints.h"
#include "experimental/core_local_mem.h"

void kernel_main() {
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
    uint32_t in0_mcast_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(16));
    constexpr uint32_t in0_mcast_num_dests = get_compile_time_arg_val(17);
    constexpr uint32_t in0_mcast_num_cores = get_compile_time_arg_val(18);
    // batch args
    constexpr uint32_t MtKt = get_compile_time_arg_val(19);  // if 0
    constexpr uint32_t batch = get_compile_time_arg_val(20);

    // sparsity args

    constexpr uint32_t batchB = get_compile_time_arg_val(21);
    constexpr uint32_t sparsity_pagesize = get_compile_time_arg_val(22);
    // Boolean that is set when input A is sparse. If set, both input A and B are assumed to be sparse.
    // Based on the sparsity tensor, the corresponding batch in input A and B are skipped.
    constexpr bool bcast_A = (bool)get_compile_time_arg_val(23);
    // This boolean is set when the number of batches is only known at runtime, typically based on a sparsity tensor.
    constexpr bool get_batch_from_reader = (bool)get_compile_time_arg_val(24);

    constexpr bool fuse_op = (bool)get_compile_time_arg_val(25);

    constexpr auto in0_args = TensorAccessorArgs<26>();
    constexpr auto sparsity_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();

    // 0 is used to specify "INVALID" state, i.e. when the multicasted data has not been received by the receiver.
    // 0x1 is used to specify "VALID" state, i.e. when the batch is valid.
    // 0x2 is used to specify "IGNORE_BATCH" state, i.e. when the batch is not valid.
    constexpr uint32_t IGNORE_BATCH = 0x2;

    // When sparsity is disabled, we just loop once
    constexpr uint32_t batchB_lim = batchB == 0 ? 1u : batchB;

    MatmulOpReceiver fused_op_receiver;
    if constexpr (fuse_op) {
        fused_op_receiver = MatmulOpReceiver(
            true, /* wait_for_op_signal */
            rt_args_idx,
            num_blocks_inner_dim,
            in0_block_w /* tiles_per_block (in the same dimension as tensor slice) */
        );
    }

    constexpr uint32_t cb_id_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t in0_single_tile_size_bytes = get_tile_size(cb_id_in0);
    constexpr uint32_t in0_block_size_bytes = in0_block_num_tiles * in0_single_tile_size_bytes;
    constexpr uint32_t one_tile = 1;

    experimental::Noc noc;
    experimental::CircularBuffer cb_in0(cb_id_in0);
    experimental::Semaphore<> sender_sem(get_compile_time_arg_val(15));
    experimental::Semaphore<> receiver_sem(get_compile_time_arg_val(16));

#ifdef IN0_SHARDED
    // In case we need to send multiple blocks per shard, in0 sharded cb is cb2 and we extract the sub-blocks to cb0
    constexpr uint32_t shard_read_stride = shard_width_in_tiles * in0_single_tile_size_bytes;
    constexpr uint32_t shard_read_width = in0_single_tile_size_bytes * in0_block_w;
    constexpr uint32_t shard_num_tiles = shard_width_in_tiles * shard_height_in_tiles;
    constexpr uint32_t in0_tensor_next_h_dim_block_stride_bytes =
        in0_tensor_next_h_dim_block_stride * in0_single_tile_size_bytes;

    uint32_t noc_shard_read_start_addr = 0;
    if constexpr (extract_shard_sub_blocks) {
        constexpr uint32_t cb_id_in2 =
            get_named_compile_time_arg_val("cb_in0_sharded");  // in0 sharded cb if extract_shard_sub_blocks
        experimental::CircularBuffer cb_in2(cb_id_in2);
        noc_shard_read_start_addr = cb_in2.get_read_ptr();
    }

#else
    const auto s0 = TensorAccessor(in0_args, in0_tensor_addr, in0_single_tile_size_bytes);
#ifndef IN0_SHARDED
#ifdef INTERMEDIATE_CB_READ
    constexpr uint32_t in0_intermediate_cb_index = get_named_compile_time_arg_val("cb_in0_intermediate");
    experimental::CircularBuffer cb_helper(in0_intermediate_cb_index);
#endif  // INTERMEDIATE_CB_READ
#endif
#endif  // IN0_SHARDED

    // sparsity accessor
    constexpr uint32_t cb_id_sparsity = get_named_compile_time_arg_val("cb_sparsity");
    experimental::CircularBuffer cb_sparsity(cb_id_sparsity);
    const auto s_sparsity = TensorAccessor(sparsity_args, sparsity_addr, sparsity_pagesize);

#ifndef SKIP_MCAST
    // Set ur local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    receiver_sem.set(VALID);
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast

#ifdef IN0_SHARDED
    uint32_t in0_start_address = cb_in0.get_write_ptr();
#endif  // IN0_SHARDED
#endif  // SKIP_MCAST

    uint32_t l1_write_addr_sparsity = 0;
    if constexpr (batchB > 0) {
        cb_sparsity.reserve_back(1);
        l1_write_addr_sparsity = cb_sparsity.get_write_ptr();
    }

    for (uint32_t b = 0; b < batch; ++b) {
        if constexpr (batchB > 0) {
            noc_async_read_page(b, s_sparsity, l1_write_addr_sparsity);
            noc.async_read_barrier();
        }

        for (uint32_t bB = 0; bB < batchB_lim; ++bB) {
            if constexpr (batchB > 0) {
                volatile auto is_batch_valid =
                    ((reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_addr_sparsity))[bB]) != 0;

                if constexpr (get_batch_from_reader) {
#ifndef SKIP_MCAST
                    // First broadcast this to other cores
                    sender_sem.wait(in0_mcast_num_dests);
                    sender_sem.set(0);
                    receiver_sem.set(is_batch_valid ? VALID : IGNORE_BATCH);
                    receiver_sem.set_multicast(
                        noc,
                        in0_mcast_dest_noc_start_x,
                        in0_mcast_dest_noc_start_y,
                        in0_mcast_dest_noc_end_x,
                        in0_mcast_dest_noc_end_y,
                        in0_mcast_num_cores);
                    noc.async_writes_flushed();
                    // Reset the semaphore value to VALID
                    receiver_sem.set(VALID);
#endif  // SKIP_MCAST

                    // We need to pass the value to compute cores regardless of the value of is_batch_valid
                    ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, is_batch_valid);
                    ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, is_batch_valid);
                    ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, is_batch_valid);
                }

                if (!is_batch_valid) {
                    if constexpr (!bcast_A) {
                        in0_tensor_start_tile_id += MtKt;
                    }
                    continue;
                }
            }

#ifdef IN0_SHARDED
            uint32_t in0_tensor_current_h_dim_block_start_addr = noc_shard_read_start_addr;
#endif  // IN0_SHARDED
            uint32_t in0_tensor_current_h_dim_block_tile_id = in0_tensor_start_tile_id;
            for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh) {
                for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw) {
#ifdef IN0_SHARDED
                    uint32_t in0_tensor_current_inner_dim_block_start_addr = in0_tensor_current_h_dim_block_start_addr;
#endif  // IN0_SHARDED
                    uint32_t in0_tensor_current_inner_dim_block_start_tile_id = in0_tensor_current_h_dim_block_tile_id;
                    for (uint32_t block = 0; block < num_blocks_inner_dim; ++block) {
                        if constexpr (fuse_op) {
                            fused_op_receiver.update_current_block_start_tile_id(
                                block, in0_tensor_current_inner_dim_block_start_tile_id, in0_tensor_start_tile_id);
                        }

                        // Operand 0
                        // Common for sharded and interleaved paths
                        cb_in0.reserve_back(in0_block_num_tiles);
#ifndef IN0_SHARDED

#ifdef INTERMEDIATE_CB_READ
                        cb_helper.reserve_back(one_tile);
#endif  // INTERMEDIATE_CB_READ

                        uint32_t in0_write_offset = 0;

#ifndef SKIP_MCAST
                        uint32_t in0_start_address =
                            cb_in0.get_write_ptr();  // copy start address of block, to be used for mcasting
#endif                                               // SKIP_MCAST

                        // Copy in0 block into CB, as the default kernel
                        uint32_t in0_tensor_row_start_tile_id = in0_tensor_current_inner_dim_block_start_tile_id;
                        for (uint32_t h = 0; h < in0_block_h; ++h) {
                            uint32_t in0_tensor_tile_id = in0_tensor_row_start_tile_id;
                            for (uint32_t w = 0; w < in0_block_w; ++w) {
                                if (bh < num_blocks_h_dim - 1 || h < last_block_h) {
#ifndef INTERMEDIATE_CB_READ
                                    noc.async_read(
                                        s0,
                                        cb_in0,
                                        in0_single_tile_size_bytes,
                                        {.page_id = in0_tensor_tile_id},
                                        {.offset_bytes = in0_write_offset});
#else
                                    noc.async_read(
                                        s0,
                                        cb_helper,
                                        in0_single_tile_size_bytes,
                                        {.page_id = in0_tensor_tile_id},
                                        {.offset_bytes = 0});
                                    noc.async_read_barrier();
                                    memcpy(
                                        /*dst=*/reinterpret_cast<void*>(cb_in0.get_write_ptr() + in0_write_offset),
                                        /*src=*/reinterpret_cast<const void*>(cb_helper.get_write_ptr()),
                                        /*size=*/in0_single_tile_size_bytes);
#endif  // INTERMEDIATE_CB_READ
                                }

                                // Zero out padded regions for the very last tile
                                if constexpr (in0_last_ktile_w > 0) {
                                    if ((block == num_blocks_inner_dim - 1) && (w == in0_block_w - 1)) {
                                        noc.async_read_barrier();
                                        const DataFormat in0_data_format = get_dataformat(cb_id_in0);
                                        pad_last_ktile<in0_data_format, in0_last_ktile_w>(
                                            cb_in0.get_write_ptr() + in0_write_offset);
                                    }
                                }
                                if constexpr (in0_last_ktile_h > 0) {
                                    if ((block == num_blocks_inner_dim - 1) && (w == in0_block_w - 1)) {
                                        noc.async_read_barrier();
                                        const DataFormat in0_data_format = get_dataformat(cb_id_in0);
                                        pad_last_transposed_ktile<in0_data_format, in0_last_ktile_h>(
                                            cb_in0.get_write_ptr() + in0_write_offset);
                                    }
                                }

                                in0_write_offset += in0_single_tile_size_bytes;
                                in0_tensor_tile_id += in0_tensor_stride_w;
                            }
                            in0_tensor_row_start_tile_id += in0_tensor_stride_h;
                        }
                        in0_tensor_current_inner_dim_block_start_tile_id += in0_tensor_next_inner_dim_block_stride;

                        // Barrier! make sure the reads are done
                        noc.async_read_barrier();
#else
                        if constexpr (extract_shard_sub_blocks) {
                            uint32_t l1_write_addr_in0 = cb_in0.get_write_ptr();

#ifndef SKIP_MCAST
                            in0_start_address =
                                l1_write_addr_in0;  // copy start address of block, to be used for mcasting
#endif  // SKIP_MCAST

                            experimental::UnicastEndpoint self_ep;
                            uint32_t noc_shard_read_l1_addr = in0_tensor_current_inner_dim_block_start_addr;

                            for (uint32_t i = 0; i < in0_block_h; i++) {
                                noc.async_read(
                                    self_ep,
                                    experimental::CoreLocalMem<uint32_t>(l1_write_addr_in0),
                                    shard_read_width,
                                    {.noc_x = my_x[0], .noc_y = my_y[0], .addr = noc_shard_read_l1_addr},
                                    {});

                                l1_write_addr_in0 += shard_read_width;
                                noc_shard_read_l1_addr += shard_read_stride;
                            }

                            in0_tensor_current_inner_dim_block_start_addr += shard_read_width;
                            noc.async_read_barrier();
                        }
#endif  // IN0_SHARDED

#ifndef SKIP_MCAST
                        // wait until all in0 mcast destinations have atomically incremented the in0 semaphore_addr
                        // (i.e. its value should be in0_mcast_num_dests), then reset the semaphore_addr value back to
                        // zero for the next block
                        sender_sem.wait(in0_mcast_num_dests);
                        sender_sem.set(0);

                        // Now we have the block in the CB address, we can mcast to dests!
                        experimental::MulticastEndpoint mcast_dst;
                        // num_dests must not include source, since we are NOT really doing a local copy!
                        noc.async_write_multicast(
                            experimental::CoreLocalMem<uint32_t>(in0_start_address),
                            mcast_dst,
                            in0_block_size_bytes,
                            in0_mcast_num_cores,
                            {},
                            {.noc_x_start = in0_mcast_dest_noc_start_x,
                             .noc_y_start = in0_mcast_dest_noc_start_y,
                             .noc_x_end = in0_mcast_dest_noc_end_x,
                             .noc_y_end = in0_mcast_dest_noc_end_y,
                             .addr = in0_start_address},
                            true);

                        // Note: no need for write barrier, since these two multicasts are done on the same noc id, same
                        // vc, same cmd_buf Also, this only works because we are setting VCs statically (using
                        // NOC_CMD_STATIC_VC).
#ifdef ARCH_BLACKHOLE
                        // On Blackhole the flush is needed because NoC latency is higher than L1 <-> RISCV
                        // latency which means data could be changed before write is issued.
                        noc.async_writes_flushed();
#endif  // ARCH_BLACKHOLE

                        // We should also multicast the flag to destinations
                        // num_dests must not include source, since we are NOT really doing a local copy!
                        receiver_sem.set_multicast(
                            noc,
                            in0_mcast_dest_noc_start_x,
                            in0_mcast_dest_noc_start_y,
                            in0_mcast_dest_noc_end_x,
                            in0_mcast_dest_noc_end_y,
                            in0_mcast_num_cores);
#endif  // SKIP_MCAST

                        // Common for sharded and interleaved paths
                        cb_in0.push_back(in0_block_num_tiles);
#ifdef INTERMEDIATE_CB_READ
                        // Clean up helper CB
                        cb_helper.push_back(one_tile);
                        cb_helper.wait_front(one_tile);
                        cb_helper.pop_front(one_tile);
#endif  // INTERMEDIATE_CB_READ
                    }
                }
#ifdef IN0_SHARDED
                in0_tensor_current_h_dim_block_start_addr += in0_tensor_next_h_dim_block_stride_bytes;
#endif  // IN0_SHARDED
                in0_tensor_current_h_dim_block_tile_id += in0_tensor_next_h_dim_block_stride;
            }

            if constexpr (!bcast_A) {
                in0_tensor_start_tile_id += MtKt;
            }
        }

        if constexpr (bcast_A) {
            in0_tensor_start_tile_id += MtKt;
        }

        // this is an optimization for the case when in0 is [1, 1, M, K] and in1 is [1, H, K, N], i.e. when in0_B ==
        // 1 and in1_B > 1 in this case we originally had to replicate the in0 block for each batch, but with this
        // optimization we just read the tensor slice once into each core's L1 and keep it there for all weight
        // batches since the needed in0 data is already in L1 after batch 0, we can just move read pointer for this
        // CB so compute kernel thinks it has new data
    }
    noc.async_write_barrier();
    // For completeness, we empty the sparsity CB if it was reserved earlier
    if constexpr (batchB > 0) {
        cb_sparsity.push_back(1);
        cb_sparsity.wait_front(1);
        cb_sparsity.pop_front(1);
    }
}
