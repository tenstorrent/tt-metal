// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of reader_bmm_tile_layout_in0_sender_padding.cpp.
//
// Algorithm body is byte-for-byte identical to the legacy kernel; only the host-binding surface is
// converted to Metal 2.0:
//   - positional get_compile_time_arg_val(N)  -> named get_arg(args::name)
//   - positional get_arg_val<uint32_t>(i)     -> named get_arg(args::name)
//   - in0 tensor address RTA + TensorAccessorArgs -> tensor::in0 typed binding
//   - named CB-index CTAs ("cb_in0", "cb_sparsity") -> dfb:: tokens
//   - Semaphore<>(get_compile_time_arg_val(id)) -> Semaphore(sem::name)
// For resnet50 (mcast_in0 / mcast_in1) the sparsity and fused-op tails are inert (batchB == 0,
// fuse_op == false). The interleaved (non-IN0_SHARDED) path is the resnet50 path.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "ttnn/operations/kernel_helper_functions/pad_tile.hpp"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "experimental/kernel_args.h"
#include "api/debug/dprint.h"  // DEBUG: matmul layer3 hang localization (remove after)

void kernel_main() {
    DPRINT("IN0 start\n");  // DEBUG: matmul layer3 hang
    uint32_t rt_args_idx = 0;
    // in0 tensor args (in0_tensor_addr is now the tensor::in0 binding)
    uint32_t in0_tensor_start_tile_id = get_arg(args::in0_tensor_start_tile_id);
    // in0 mcast args
    const uint32_t in0_mcast_dest_noc_start_x = get_arg(args::in0_mcast_dest_noc_start_x);
    const uint32_t in0_mcast_dest_noc_start_y = get_arg(args::in0_mcast_dest_noc_start_y);
    const uint32_t in0_mcast_dest_noc_end_x = get_arg(args::in0_mcast_dest_noc_end_x);
    const uint32_t in0_mcast_dest_noc_end_y = get_arg(args::in0_mcast_dest_noc_end_y);

    // padding args
    const uint32_t last_block_h = get_arg(args::last_block_h);
    // sparsity args
    const uint32_t sparsity_addr = get_arg(args::sparsity_addr);

    // COMPILE TIME ARGS
    // in0 tensor args
    constexpr uint32_t in0_tensor_stride_w = get_arg(args::in0_tensor_stride_w);
    constexpr uint32_t in0_tensor_stride_h = get_arg(args::in0_tensor_stride_h);
    constexpr uint32_t in0_tensor_next_inner_dim_block_stride = get_arg(args::in0_tensor_next_inner_dim_block_stride);
    constexpr uint32_t in0_tensor_next_h_dim_block_stride = get_arg(args::in0_tensor_next_h_dim_block_stride);
    // in0 block args
    constexpr uint32_t in0_block_w = get_arg(args::in0_block_w);
    constexpr uint32_t in0_block_h = get_arg(args::in0_block_h);
    constexpr uint32_t in0_block_num_tiles = get_arg(args::in0_block_num_tiles);
    constexpr uint32_t in0_last_ktile_w = get_arg(args::in0_last_ktile_w);
    constexpr uint32_t in0_last_ktile_h = get_arg(args::in0_last_ktile_h);

    constexpr bool extract_shard_sub_blocks = (bool)get_arg(args::extract_shard_sub_blocks);
    constexpr uint32_t shard_width_in_tiles = get_arg(args::shard_width_in_tiles);
    constexpr uint32_t shard_height_in_tiles = get_arg(args::shard_height_in_tiles);
    // in0/in1 common args
    constexpr uint32_t num_blocks_inner_dim = get_arg(args::num_blocks_inner_dim);
    constexpr uint32_t num_blocks_w_dim = get_arg(args::num_blocks_w_dim);
    constexpr uint32_t num_blocks_h_dim = get_arg(args::num_blocks_h_dim);
    // in0 mcast args
    constexpr uint32_t in0_mcast_num_dests = get_arg(args::in0_mcast_num_dests);
    constexpr uint32_t in0_mcast_num_cores = get_arg(args::in0_mcast_num_cores);
    // batch args
    constexpr uint32_t MtKt = get_arg(args::MtKt);  // if 0
    constexpr uint32_t in0_B = get_arg(args::in0_B);
    constexpr uint32_t in1_B = get_arg(args::in1_B);
    constexpr uint32_t in0_reuse_in_CB = get_arg(args::in0_reuse_in_CB);

    // sparsity args
    constexpr uint32_t batchB = get_arg(args::batchB);
    [[maybe_unused]] constexpr uint32_t sparsity_pagesize = get_arg(args::sparsity_pagesize);
    // Boolean that is set when input A is sparse. If set, both input A and B are assumed to be sparse.
    constexpr bool bcast_A = (bool)get_arg(args::bcast_A);
    // This boolean is set when the number of batches is only known at runtime, typically based on a sparsity tensor.
    constexpr bool get_batch_from_reader = (bool)get_arg(args::get_batch_from_reader);

    constexpr bool fuse_op = (bool)get_arg(args::fuse_op);

    // Number of valid (non-zero sparsity) batches; inert for resnet50 (batchB == 0).
    [[maybe_unused]] constexpr uint32_t num_batch_compute = get_arg(args::num_batch_compute);

    // 0 is used to specify "INVALID" state, 0x1 "VALID", 0x2 "IGNORE_BATCH".
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

    constexpr uint32_t cb_id_in0 = dfb::cb_in0;
    constexpr uint32_t in0_single_tile_size_bytes = get_tile_size(cb_id_in0);
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
    DataflowBuffer cb_in0(dfb::cb_in0);
    Semaphore sender_sem(sem::in0_sender);
    Semaphore receiver_sem(sem::in0_receiver);

#ifdef IN0_SHARDED
    // In case we need to send multiple blocks per shard, in0 sharded cb is cb2 and we extract the sub-blocks to cb0
    constexpr uint32_t shard_read_stride = shard_width_in_tiles * in0_single_tile_size_bytes;
    constexpr uint32_t shard_read_width = in0_single_tile_size_bytes * in0_block_w;
    constexpr uint32_t shard_num_tiles = shard_width_in_tiles * shard_height_in_tiles;
    constexpr uint32_t in0_tensor_next_h_dim_block_stride_bytes =
        in0_tensor_next_h_dim_block_stride * in0_single_tile_size_bytes;

    uint32_t noc_shard_read_start_addr = 0;
#ifdef EXTRACT_SHARD_SUB_BLOCKS
    // The resident in0 shard is reached by L1 base address from a local TensorAccessor over the in0
    // tensor (no borrowed self-loop CB, which Metal 2.0 forbids on DM kernels).
    noc_shard_read_start_addr = (uint32_t)NOC_LOCAL_ADDR_OFFSET(TensorAccessor(tensor::in0).get_noc_addr(0));
#endif

#else
    const auto s0 = TensorAccessor(tensor::in0);
#endif  // IN0_SHARDED

    // sparsity accessor. cb_sparsity is an inert DMA-landing scratch used only when sparsity is
    // enabled (batchB > 0). As a single-kernel self-loop DFB (PRODUCER+CONSUMER) it is rejected by
    // the Metal 2.0 DM-kernel self-loop validator, so it is gated behind SPARSITY — never defined by
    // the non-sparse mcast factories that build this kernel (batchB is always 0 here). tensor::sparsity
    // stays referenced so the factory's inert sparsity tensor binding remains valid.
    [[maybe_unused]] const auto s_sparsity = TensorAccessor(tensor::sparsity);
#ifdef SPARSITY
    constexpr uint32_t cb_id_sparsity = dfb::cb_sparsity;
    DataflowBuffer cb_sparsity(cb_id_sparsity);
#endif

#ifndef SKIP_MCAST
    // Set ur local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    receiver_sem.set(VALID);
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast

#ifdef IN0_SHARDED
    uint32_t in0_start_address = cb_in0.get_write_ptr();
#endif  // IN0_SHARDED
#endif  // SKIP_MCAST

    [[maybe_unused]] uint32_t l1_write_addr_sparsity = 0;
#ifdef SPARSITY
    if constexpr (batchB > 0) {
        cb_sparsity.reserve_back(1);
        l1_write_addr_sparsity = cb_sparsity.get_write_ptr();
    }
#endif

    [[maybe_unused]] uint32_t num_valid_batches = 0;

    for (uint32_t b = 0; b < in0_B; ++b) {
#ifdef SPARSITY
        if constexpr (batchB > 0) {
            noc.async_read(s_sparsity, cb_sparsity, sparsity_pagesize, {.page_id = b}, {.offset_bytes = 0});
            noc.async_read_barrier();
        }
#endif

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
                        cb_in0.reserve_back(in0_block_num_tiles);
#ifndef IN0_SHARDED

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
                                    noc.async_read(
                                        s0,
                                        cb_in0,
                                        in0_single_tile_size_bytes,
                                        {.page_id = in0_tensor_tile_id},
                                        {.offset_bytes = in0_write_offset});
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
                            uint32_t l1_write_addr_in0 = cb_in0.get_write_ptr();

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
                            constexpr DataFormat in0_data_format = get_dataformat(cb_id_in0);
                            uint32_t in0_pad_base_addr = cb_in0.get_write_ptr();
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
                        // wait until all in0 mcast destinations have atomically incremented the in0 semaphore_addr
                        // (i.e. its value should be in0_mcast_num_dests), then reset the semaphore_addr value back to
                        // zero for the next block
                        sender_sem.wait(in0_mcast_num_dests);
                        sender_sem.set(0);

                        // Now we have the block in the CB address, we can mcast to dests!
                        MulticastEndpoint mcast_dst;
                        // num_dests must not include source, since we are NOT really doing a local copy!
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

        // in0 reuse-in-CB optimization for [1,1,M,K] x [1,H,K,N]
        if (in0_reuse_in_CB) {
            for (uint32_t fake_batch = 0; fake_batch < in1_B - in0_B; ++fake_batch) {
                for (uint32_t blk = 0; blk < num_blocks_inner_dim; ++blk) {
                    cb_in0.reserve_back(in0_block_num_tiles);
                    cb_in0.push_back(in0_block_num_tiles);
                }
            }
        }
    }
    noc.async_write_barrier();

    if constexpr (!get_batch_from_reader && batchB > 0) {
        ASSERT(num_valid_batches == num_batch_compute);
    }

#ifdef SPARSITY
    if constexpr (batchB > 0) {
        cb_sparsity.push_back(1);
        cb_sparsity.wait_front(1);
        cb_sparsity.pop_front(1);
    }
#endif
    // Drain outstanding NOC writes AND atomics before returning (Metal 2.0 FW epilogue does not).
    noc.async_full_barrier();
    DPRINT("IN0 end\n");  // DEBUG: matmul layer3 hang
}
