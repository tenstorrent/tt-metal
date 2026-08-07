// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "ttnn/operations/kernel_helper_functions/pad_tile.hpp"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"
void kernel_main() {
#ifdef MCAST_ARGS
    constexpr auto in0_mcast_args = dataflow_kernel_lib::McastArgs<15, 2>();
    constexpr uint32_t in0_post_mcast_ct_offset = in0_mcast_args.next_compile_time_args_offset();
    constexpr uint32_t in0_post_mcast_rt_offset = in0_mcast_args.next_runtime_args_offset();
#else
    constexpr uint32_t in0_post_mcast_ct_offset = 19;
    constexpr uint32_t in0_post_mcast_rt_offset = 6;
#endif

    uint32_t rt_args_idx = 0;
    // in0 tensor args
    const uint32_t in0_tensor_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t in0_tensor_start_tile_id = get_arg_val<uint32_t>(rt_args_idx++);
#ifdef MCAST_ARGS
    rt_args_idx = in0_post_mcast_rt_offset;
#else
    // Legacy in0 mcast args used by the non-multicast and sharded bindings.
    const uint32_t in0_mcast_dest_noc_start_x = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t in0_mcast_dest_noc_start_y = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t in0_mcast_dest_noc_end_x = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t in0_mcast_dest_noc_end_y = get_arg_val<uint32_t>(rt_args_idx++);
#endif

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
#ifndef MCAST_ARGS
    // Legacy in0 mcast args used by the non-multicast and sharded bindings.
    constexpr uint32_t in0_mcast_num_dests = get_compile_time_arg_val(17);
    constexpr uint32_t in0_mcast_num_cores = get_compile_time_arg_val(18);
#endif
    // batch args
    constexpr uint32_t MtKt = get_compile_time_arg_val(in0_post_mcast_ct_offset);  // if 0
    constexpr uint32_t in0_B = get_compile_time_arg_val(in0_post_mcast_ct_offset + 1);
    constexpr uint32_t in1_B = get_compile_time_arg_val(in0_post_mcast_ct_offset + 2);
    constexpr uint32_t in0_reuse_in_CB = get_compile_time_arg_val(in0_post_mcast_ct_offset + 3);

    // sparsity args

    constexpr uint32_t batchB = get_compile_time_arg_val(in0_post_mcast_ct_offset + 4);
    constexpr uint32_t sparsity_pagesize = get_compile_time_arg_val(in0_post_mcast_ct_offset + 5);
    // Boolean that is set when input A is sparse. If set, both input A and B are assumed to be sparse.
    // Based on the sparsity tensor, the corresponding batch in input A and B are skipped.
    constexpr bool bcast_A = (bool)get_compile_time_arg_val(in0_post_mcast_ct_offset + 6);
    // This boolean is set when the number of batches is only known at runtime, typically based on a sparsity tensor.
    constexpr bool get_batch_from_reader = (bool)get_compile_time_arg_val(in0_post_mcast_ct_offset + 7);

    constexpr bool fuse_op = (bool)get_compile_time_arg_val(in0_post_mcast_ct_offset + 8);

    constexpr auto in0_args = TensorAccessorArgs<in0_post_mcast_ct_offset + 9>();

    constexpr auto sparsity_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();

    // Validate the runtime nonzero count against the receiver/compute loop bound (issue #45943).
    [[maybe_unused]] constexpr uint32_t num_batch_compute =
        get_compile_time_arg_val(sparsity_args.next_compile_time_args_offset());

    // The flag channel carries INVALID(0), VALID(1), or IGNORE_BATCH(2).
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

    constexpr uint32_t dfb_id_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t in0_single_tile_size_bytes = get_tile_size(dfb_id_in0);
    // Tiles whose size is not a multiple of the DRAM alignment are padded to it in DRAM, and the
    // interleaved in0 CB pages are sized to match (see the program factory). The NOC reads the
    // unpadded tile of data into each padded slot, and tiles are laid out / multicast at the padded
    // stride. No-op when already aligned. The sharded path keeps the natural (unpadded) stride.
    constexpr uint32_t in0_aligned_tile_size_bytes =
        (in0_single_tile_size_bytes + (DRAM_ALIGNMENT - 1)) & ~(DRAM_ALIGNMENT - 1);
#ifdef IN0_SHARDED
    constexpr uint32_t in0_block_size_bytes = in0_block_num_tiles * in0_single_tile_size_bytes;
#else
    constexpr uint32_t in0_block_size_bytes = in0_block_num_tiles * in0_aligned_tile_size_bytes;
#endif

    Noc noc;
    DataflowBuffer dfb_in0(dfb_id_in0);
#ifndef MCAST_ARGS
    Semaphore<> sender_sem(get_compile_time_arg_val(15));
    Semaphore<> receiver_sem(get_compile_time_arg_val(16));
#endif

#ifdef IN0_SHARDED
    // In case we need to send multiple blocks per shard, in0 sharded cb is cb2 and we extract the sub-blocks to cb0
    constexpr uint32_t shard_read_stride = shard_width_in_tiles * in0_single_tile_size_bytes;
    constexpr uint32_t shard_read_width = in0_single_tile_size_bytes * in0_block_w;
    constexpr uint32_t shard_num_tiles = shard_width_in_tiles * shard_height_in_tiles;
    constexpr uint32_t in0_tensor_next_h_dim_block_stride_bytes =
        in0_tensor_next_h_dim_block_stride * in0_single_tile_size_bytes;

    uint32_t noc_shard_read_start_addr = 0;
    if constexpr (extract_shard_sub_blocks) {
        constexpr uint32_t dfb_id_in2 =
            get_named_compile_time_arg_val("cb_in0_sharded");  // in0 sharded cb if extract_shard_sub_blocks
        DataflowBuffer dfb_in2(dfb_id_in2);
        noc_shard_read_start_addr = dfb_in2.get_read_ptr();
    }

#else
    const auto s0 = TensorAccessor(in0_args, in0_tensor_addr);
#endif  // IN0_SHARDED

    // sparsity accessor
    constexpr uint32_t dfb_id_sparsity = get_named_compile_time_arg_val("cb_sparsity");
    DataflowBuffer dfb_sparsity(dfb_id_sparsity);
    const auto s_sparsity = TensorAccessor(sparsity_args, sparsity_addr);

#ifndef SKIP_MCAST
#ifdef MCAST_ARGS
    auto in0_pipe = in0_mcast_args.sender(noc);
#else
    receiver_sem.set(VALID);
#endif
#ifdef IN0_SHARDED
    uint32_t in0_start_address = dfb_in0.get_write_ptr();
#endif  // IN0_SHARDED
#endif  // SKIP_MCAST

    uint32_t l1_write_addr_sparsity = 0;
    if constexpr (batchB > 0) {
        dfb_sparsity.reserve_back(1);
        l1_write_addr_sparsity = dfb_sparsity.get_write_ptr();
    }

    // Count the data multicasts issued for the nnz contract.
    [[maybe_unused]] uint32_t num_valid_batches = 0;

    for (uint32_t b = 0; b < in0_B; ++b) {
        if constexpr (batchB > 0) {
            noc.async_read(s_sparsity, dfb_sparsity, sparsity_pagesize, {.page_id = b}, {.offset_bytes = 0});
            noc.async_read_barrier();
        }

        for (uint32_t bB = 0; bB < batchB_lim; ++bB) {
            if constexpr (batchB > 0) {
                volatile auto is_batch_valid =
                    ((reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_addr_sparsity))[bB]) != 0;

                if constexpr (get_batch_from_reader) {
#ifndef SKIP_MCAST
#ifdef MCAST_ARGS
                    in0_pipe.send_signal(is_batch_valid ? VALID : IGNORE_BATCH);
#else
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
#endif
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

                // Catch excess nonzero batches before the sender can wait on finished receivers.
                if constexpr (!get_batch_from_reader) {
                    ++num_valid_batches;
                    ASSERT(num_valid_batches <= num_batch_compute);
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
                        dfb_in0.reserve_back(in0_block_num_tiles);
#ifndef IN0_SHARDED

                        uint32_t in0_write_offset = 0;

#ifndef SKIP_MCAST
                        uint32_t in0_start_address =
                            dfb_in0.get_write_ptr();  // copy start address of block, to be used for mcasting
#endif                                               // SKIP_MCAST

                        // Copy in0 block into CB, as the default kernel
                        uint32_t in0_tensor_row_start_tile_id = in0_tensor_current_inner_dim_block_start_tile_id;
                        for (uint32_t h = 0; h < in0_block_h; ++h) {
                            uint32_t in0_tensor_tile_id = in0_tensor_row_start_tile_id;
                            for (uint32_t w = 0; w < in0_block_w; ++w) {
                                if (bh < num_blocks_h_dim - 1 || h < last_block_h) {
                                    noc.async_read(
                                        s0,
                                        dfb_in0,
                                        in0_single_tile_size_bytes,
                                        {.page_id = in0_tensor_tile_id},
                                        {.offset_bytes = in0_write_offset});
                                }

                                // Zero out padded regions for the very last tile
                                if constexpr (in0_last_ktile_w > 0) {
                                    if ((block == num_blocks_inner_dim - 1) && (w == in0_block_w - 1)) {
                                        noc.async_read_barrier();
                                        constexpr DataFormat in0_data_format = get_dataformat(dfb_id_in0);
                                        pad_last_ktile<in0_data_format, in0_last_ktile_w>(
                                            dfb_in0.get_write_ptr() + in0_write_offset);
                                    }
                                }
                                if constexpr (in0_last_ktile_h > 0) {
                                    if ((block == num_blocks_inner_dim - 1) && (w == in0_block_w - 1)) {
                                        noc.async_read_barrier();
                                        constexpr DataFormat in0_data_format = get_dataformat(dfb_id_in0);
                                        pad_last_transposed_ktile<in0_data_format, in0_last_ktile_h>(
                                            dfb_in0.get_write_ptr() + in0_write_offset);
                                    }
                                }

                                in0_write_offset += in0_aligned_tile_size_bytes;
                                in0_tensor_tile_id += in0_tensor_stride_w;
                            }
                            in0_tensor_row_start_tile_id += in0_tensor_stride_h;
                        }
                        in0_tensor_current_inner_dim_block_start_tile_id += in0_tensor_next_inner_dim_block_stride;

                        // Barrier! make sure the reads are done
                        noc.async_read_barrier();
#else
                        if constexpr (extract_shard_sub_blocks) {
                            uint32_t l1_write_addr_in0 = dfb_in0.get_write_ptr();

#ifndef SKIP_MCAST
                            in0_start_address =
                                l1_write_addr_in0;  // copy start address of block, to be used for mcasting
#endif  // SKIP_MCAST

                            UnicastEndpoint self_ep;
                            uint32_t noc_shard_read_l1_addr = in0_tensor_current_inner_dim_block_start_addr;

                            for (uint32_t i = 0; i < in0_block_h; i++) {
                                noc.async_read(
                                    self_ep,
                                    CoreLocalMem<uint32_t>(l1_write_addr_in0),
                                    shard_read_width,
                                    {.noc_x = my_x[0], .noc_y = my_y[0], .addr = noc_shard_read_l1_addr},
                                    {});

                                l1_write_addr_in0 += shard_read_width;
                                noc_shard_read_l1_addr += shard_read_stride;
                            }

                            in0_tensor_current_inner_dim_block_start_addr += shard_read_width;
                            noc.async_read_barrier();
                        }

                        {
                            constexpr DataFormat in0_data_format = get_dataformat(dfb_id_in0);
                            uint32_t in0_pad_base_addr = dfb_in0.get_write_ptr();
                            if constexpr (in0_last_ktile_w > 0) {
                                if ((block == num_blocks_inner_dim - 1)) {
                                    for (uint32_t h = 0; h < in0_block_h; ++h) {
                                        auto ptr = in0_pad_base_addr +
                                                   (h * in0_block_w + in0_block_w - 1) * in0_single_tile_size_bytes;
                                        pad_last_ktile<in0_data_format, in0_last_ktile_w>(ptr);
                                    }
                                }
                            }
                            if constexpr (in0_last_ktile_h > 0) {
                                if ((block == num_blocks_inner_dim - 1)) {
                                    for (uint32_t w = 0; w < in0_block_w; ++w) {
                                        auto ptr = in0_pad_base_addr +
                                                   ((in0_block_h - 1) * in0_block_w + w) * in0_single_tile_size_bytes;
                                        pad_last_transposed_ktile<in0_data_format, in0_last_ktile_h>(ptr);
                                    }
                                }
                            }
                        }
#endif  // IN0_SHARDED

#ifndef SKIP_MCAST
#ifdef MCAST_ARGS
                        in0_pipe.send(in0_start_address, in0_start_address, in0_block_size_bytes);
#else
                        sender_sem.wait(in0_mcast_num_dests);
                        sender_sem.set(0);

                        MulticastEndpoint mcast_dst;
                        noc.async_write_multicast(
                            CoreLocalMem<uint32_t>(in0_start_address),
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

#ifdef ARCH_BLACKHOLE
                        noc.async_writes_flushed();
#endif  // ARCH_BLACKHOLE

                        receiver_sem.set_multicast(
                            noc,
                            in0_mcast_dest_noc_start_x,
                            in0_mcast_dest_noc_start_y,
                            in0_mcast_dest_noc_end_x,
                            in0_mcast_dest_noc_end_y,
                            in0_mcast_num_cores);
#endif
#endif  // SKIP_MCAST

                        // Common for sharded and interleaved paths
                        dfb_in0.push_back(in0_block_num_tiles);
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

        // Re-publish a batch-broadcast activation already resident in L1.
        if (in0_reuse_in_CB) {
            for (uint32_t fake_batch = 0; fake_batch < in1_B - in0_B; ++fake_batch) {
                for (uint32_t blk = 0; blk < num_blocks_inner_dim; ++blk) {
                    dfb_in0.reserve_back(in0_block_num_tiles);
                    dfb_in0.push_back(in0_block_num_tiles);
                }
            }
        }
    }
    noc.async_write_barrier();

    // Catch too few nonzero batches after all possible multicasts have been issued.
    if constexpr (!get_batch_from_reader && batchB > 0) {
        ASSERT(num_valid_batches == num_batch_compute);
    }

    if constexpr (batchB > 0) {
        dfb_sparsity.push_back(1);
        dfb_sparsity.wait_front(1);
        dfb_sparsity.pop_front(1);
    }
}
