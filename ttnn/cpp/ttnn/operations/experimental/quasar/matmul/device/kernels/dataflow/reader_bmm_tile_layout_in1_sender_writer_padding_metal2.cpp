// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of reader_bmm_tile_layout_in1_sender_writer_padding.cpp.
//
// Algorithm body matches the legacy kernel; only the host-binding surface is converted to Metal 2.0:
//   - positional get_compile_time_arg_val(N)  -> named get_arg(args::name)
//   - positional get_arg_val<uint32_t>(i)     -> named get_arg(args::name)
//   - in1 / out / bias tensor address RTAs + TensorAccessorArgs -> tensor:: typed bindings
//   - named CB-index CTAs ("cb_in1","cb_out","cb_bias","cb_sparsity") -> dfb:: tokens
//   - Semaphore<>(get_compile_time_arg_val(id)) -> Semaphore(sem::name)
//
// The DRAM-width/height-sharded reader paths of the legacy kernel are NOT ported here: those configs
// stay on the legacy original (dram_sharded / batched_hs factories), so IN1_DRAM_*_SHARDED are never
// emitted for this fork. For resnet50 the sparsity / fused-op tails are inert.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "ttnn/operations/kernel_helper_functions/pad_tile.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "experimental/kernel_args.h"
#include "api/debug/dprint.h"  // DEBUG: matmul mcast hang diagnosis (remove after)

void kernel_main() {
    DPRINT("WSM enter\n");  // DEBUG: matmul pre-kernel_main confirmation (remove after)
    // READER
    uint32_t rt_args_idx = 0;
    // in1 tensor args (in1_tensor_addr is now the tensor::in1 binding)
    uint32_t in1_tensor_start_tile_id = get_arg(args::in1_tensor_start_tile_id);
    // in1 mcast args
    const uint32_t in1_mcast_dest_noc_start_x = get_arg(args::in1_mcast_dest_noc_start_x);
    const uint32_t in1_mcast_dest_noc_start_y = get_arg(args::in1_mcast_dest_noc_start_y);
    const uint32_t in1_mcast_dest_noc_end_x = get_arg(args::in1_mcast_dest_noc_end_x);
    const uint32_t in1_mcast_dest_noc_end_y = get_arg(args::in1_mcast_dest_noc_end_y);

    // sparsity args
    const uint32_t sparsity_addr = get_arg(args::sparsity_addr);

    // WRITER
    // out tensor args (out_tensor_addr is now the tensor::out binding)
    uint32_t out_tensor_start_tile_id = get_arg(args::out_tensor_start_tile_id);

    // padding args (READER)
    const uint32_t last_block_w = get_arg(args::last_block_w);
    // padding args (WRITER)
    const uint32_t out_num_nonzero_subblocks_h = get_arg(args::out_num_nonzero_subblocks_h);
    const uint32_t out_last_subblock_h = get_arg(args::out_last_subblock_h);
    const uint32_t padded_block_tiles_h_skip = get_arg(args::padded_block_tiles_h_skip);
    const uint32_t out_num_nonzero_subblocks_w = get_arg(args::out_num_nonzero_subblocks_w);
    const uint32_t out_last_num_nonzero_subblocks_w = get_arg(args::out_last_num_nonzero_subblocks_w);
    const uint32_t out_last_subblock_w = get_arg(args::out_last_subblock_w);
    const uint32_t padded_subblock_tiles_addr_skip = get_arg(args::padded_subblock_tiles_addr_skip);
    const uint32_t padded_block_tiles_w_skip = get_arg(args::padded_block_tiles_w_skip);

    // COMPILE TIME ARGS
    // READER
    // in1 tensor args
    constexpr uint32_t in1_tensor_stride_w = get_arg(args::in1_tensor_stride_w);
    constexpr uint32_t in1_tensor_stride_h = get_arg(args::in1_tensor_stride_h);
    constexpr uint32_t in1_tensor_next_block_stride = get_arg(args::in1_tensor_next_block_stride);
    constexpr uint32_t in1_tensor_next_w_dim_block_stride = get_arg(args::in1_tensor_next_w_dim_block_stride);
    // in1 block args
    constexpr uint32_t in1_block_w = get_arg(args::in1_block_w);
    constexpr uint32_t in1_block_h = get_arg(args::in1_block_h);
    constexpr uint32_t in1_block_num_tiles = get_arg(args::in1_block_num_tiles);
    // in0/in1 common args
    constexpr uint32_t num_blocks_inner_dim = get_arg(args::num_blocks_inner_dim);
    constexpr uint32_t num_blocks_w_dim = get_arg(args::num_blocks_w_dim);
    constexpr uint32_t num_blocks_h_dim = get_arg(args::num_blocks_h_dim);

    // in1 mcast args
    constexpr uint32_t in1_mcast_num_dests = get_arg(args::in1_mcast_num_dests);
    constexpr uint32_t in1_mcast_num_cores = get_arg(args::in1_mcast_num_cores);
    // batch args
    constexpr uint32_t KtNt = get_arg(args::KtNt);
    constexpr uint32_t batch = get_arg(args::batch);
    constexpr uint32_t bcast_B = get_arg(args::bcast_B);
    // sparsity args
    constexpr uint32_t batchB = get_arg(args::batchB);
    [[maybe_unused]] constexpr uint32_t sparsity_pagesize = get_arg(args::sparsity_pagesize);

    // WRITER
    // out tensor args
    constexpr uint32_t out_tensor_stride_w = get_arg(args::out_tensor_stride_w);
    constexpr uint32_t out_tensor_stride_h = get_arg(args::out_tensor_stride_h);
    constexpr uint32_t out_tensor_next_subblock_stride_w = get_arg(args::out_tensor_next_subblock_stride_w);
    constexpr uint32_t out_tensor_next_subblock_stride_h = get_arg(args::out_tensor_next_subblock_stride_h);
    constexpr uint32_t out_tensor_next_w_dim_block_stride = get_arg(args::out_tensor_next_w_dim_block_stride);
    constexpr uint32_t out_tensor_next_h_dim_block_stride = get_arg(args::out_tensor_next_h_dim_block_stride);
    // out subblock args
    constexpr uint32_t out_subblock_w = get_arg(args::out_subblock_w);
    constexpr uint32_t out_subblock_h = get_arg(args::out_subblock_h);
    constexpr uint32_t out_subblock_tile_count = get_arg(args::out_subblock_tile_count);
    // batch args
    constexpr uint32_t MtNt = get_arg(args::MtNt);  // if 0
    // Don't need batch; same as batch from READER args

    // When sparsity is disabled, we just loop once
    constexpr uint32_t batchB_lim = batchB == 0 ? 1u : batchB;

#ifdef FUSE_BIAS
    // in3 mcast args (in3_tensor_addr is now the tensor::bias binding)
    const uint32_t in3_tensor_start_tile_id = get_arg(args::in3_tensor_start_tile_id);

    constexpr uint32_t in3_tensor_stride_w = get_arg(args::in3_tensor_stride_w);

    constexpr uint32_t cb_id_in3 = dfb::cb_bias;
    constexpr uint32_t bias_single_tile_size_bytes = get_tile_size(cb_id_in3);

#ifndef BIAS_SHARDED
    uint32_t l1_write_addr_in3;
#endif  // BIAS_SHARDED
#endif  // FUSE_BIAS
#ifndef OUT_SHARDED
    const uint32_t last_num_blocks_w_dim = get_arg(args::last_num_blocks_w_dim);
#endif  // OUT_SHARDED

    constexpr bool fuse_op_all_gather = (bool)get_arg(args::fuse_op_all_gather);
    constexpr bool fuse_op_reduce_scatter = (bool)get_arg(args::fuse_op_reduce_scatter);

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

#ifdef FUSE_BIAS
#ifndef BIAS_SHARDED
    const auto s3 = TensorAccessor(tensor::bias);
#endif  // BIAS_SHARDED
#endif  // FUSE_BIAS

    constexpr uint32_t cb_id_in1 = dfb::cb_in1;
    constexpr uint32_t in1_single_tile_size_bytes = get_tile_size(cb_id_in1);
    // Tiles whose size is not a multiple of the DRAM alignment are padded to it in DRAM, and the
    // interleaved in1 CB pages are sized to match (see the program factory). The NOC reads the
    // unpadded tile of data into each padded slot and tiles are laid out / multicast at the padded
    // stride. No-op when already aligned. The sharded path keeps the natural (unpadded) stride.
    constexpr uint32_t in1_aligned_tile_size_bytes =
        (in1_single_tile_size_bytes + (DRAM_ALIGNMENT - 1)) & ~(DRAM_ALIGNMENT - 1);
#if !defined(IN1_SHARDED)
    constexpr uint32_t in1_block_size_bytes = in1_block_num_tiles * in1_aligned_tile_size_bytes;
#else
    constexpr uint32_t in1_block_size_bytes = in1_block_num_tiles * in1_single_tile_size_bytes;
#endif

    constexpr uint32_t cb_id_out0 = dfb::cb_out;
    constexpr uint32_t output_single_tile_size_bytes = get_tile_size(cb_id_out0);

    Noc noc;
    // DEBUG: matmul mcast hang — the sender reports its NoC coords, expected ack count, and the
    // multicast rectangle it sets receiver_sem=VALID over. If the rectangle doesn't cover every
    // receiver (ragged/L-shaped grid), uncovered receivers hang at their in1 wait.
    DPRINT(
        "SEND core x={} y={} num_dests={} num_cores={} mcast=[{},{}]..[{},{}]\n",
        (uint32_t)my_x[noc.get_noc_id()],
        (uint32_t)my_y[noc.get_noc_id()],
        in1_mcast_num_dests,
        in1_mcast_num_cores,
        in1_mcast_dest_noc_start_x,
        in1_mcast_dest_noc_start_y,
        in1_mcast_dest_noc_end_x,
        in1_mcast_dest_noc_end_y);
    DataflowBuffer cb_in1(dfb::cb_in1);
    DataflowBuffer cb_out(dfb::cb_out);
    Semaphore sender_sem(sem::in1_sender);
    Semaphore receiver_sem(sem::in1_receiver);
#ifdef FUSE_BIAS
    DataflowBuffer cb_in3(dfb::cb_bias);
#endif

//  READER
#ifdef IN1_SHARDED
    cb_in1.reserve_back(in1_block_num_tiles * num_blocks_inner_dim);
    cb_in1.push_back(in1_block_num_tiles * num_blocks_inner_dim);
#else
    const auto s1 = TensorAccessor(tensor::in1);
#endif  // IN1_SHARDED

    //  WRITER
    // Used only when the output-write path below is compiled in (some mcast configs write via DFB
    // instead), so mark maybe_unused to avoid -Wunused-but-set-variable, matching s_sparsity below.
    [[maybe_unused]] const auto s = TensorAccessor(tensor::out);

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
#ifdef IN1_SHARDED
    uint64_t in1_start_address = cb_in1.get_write_ptr();
#endif  // IN1_SHARDED
#endif  // SKIP_MCAST

    [[maybe_unused]] uint32_t l1_write_addr_sparsity = 0;
#ifdef SPARSITY
    if constexpr (batchB > 0) {
        cb_sparsity.reserve_back(1);
        l1_write_addr_sparsity = cb_sparsity.get_write_ptr();
    }
#endif

    for (uint32_t b = 0; b < batch; ++b) {
        uint32_t in1_batch_tile_id = in1_tensor_start_tile_id;

#ifdef SPARSITY
        if constexpr (batchB > 0) {
            noc.async_read(s_sparsity, cb_sparsity, sparsity_pagesize, {.page_id = b}, {.offset_bytes = 0});
            noc.async_read_barrier();
        }
#endif

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

                    for (uint32_t block = 0; block < num_blocks_inner_dim; ++block) {
                        if constexpr (fuse_op_all_gather) {
                            fused_op_receiver.update_current_block_start_tile_id(
                                block, in1_tensor_current_inner_dim_block_start_tile_id, in1_batch_tile_id);
                        }
#if !defined(IN1_SHARDED)
                        // Operand 1 - interleaved
                        cb_in1.reserve_back(in1_block_num_tiles);
                        uint32_t in1_write_offset = 0;
                        uint64_t in1_start_address =
                            cb_in1.get_write_ptr();  // copy start address of block, to be used for mcasting

                        // Copy in1 block into CB, as the default kernel
                        uint32_t in1_tensor_row_start_tile_id = in1_tensor_current_inner_dim_block_start_tile_id;
                        for (uint32_t h = 0; h < in1_block_h; ++h) {
                            uint32_t in1_tensor_tile_id = in1_tensor_row_start_tile_id;
                            for (uint32_t w = 0; w < in1_block_w; ++w) {
                                if (bw < num_blocks_w_dim - 1 || w < last_block_w) {
                                    noc.async_read(
                                        s1,
                                        cb_in1,
                                        in1_single_tile_size_bytes,
                                        {.page_id = in1_tensor_tile_id},
                                        {.offset_bytes = in1_write_offset});
                                }
                                in1_write_offset += in1_aligned_tile_size_bytes;
                                in1_tensor_tile_id += in1_tensor_stride_w;
                            }
                            in1_tensor_row_start_tile_id += in1_tensor_stride_h;
                        }
                        in1_tensor_current_inner_dim_block_start_tile_id += in1_tensor_next_block_stride;

                        // Barrier! make sure the reads are done
                        noc.async_read_barrier();
#endif  // IN1_SHARDED

#ifndef SKIP_MCAST
                        // wait until all in1 mcast destinations have atomically incremented the in1 semaphore_addr
                        sender_sem.wait(in1_mcast_num_dests);
                        sender_sem.set(0);
                        // DEBUG: matmul mcast hang — reached iff all in1 acks received (first block).
                        if (b == 0 && bh == 0 && bw == 0 && block == 0) {
                            DPRINT("SEND in1 acked (got {} dests)\n", in1_mcast_num_dests);
                        }

                        // Now we have the block in the CB address, we can mcast to dests!
                        MulticastEndpoint mcast_dst;
                        // num_dests must not include source, since we are NOT really doing a local copy!
                        noc.async_write_multicast(
                            CoreLocalMem<uint32_t>(static_cast<uint32_t>(in1_start_address)),
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

#if defined(ARCH_BLACKHOLE) || defined(ARCH_QUASAR)
                        // Flush the DATA multicast before the VALID-semaphore multicast. On Quasar, without this
                        // barrier the back-to-back in1-then-bias mcasts let the bias VALID semaphore write
                        // race/drop on the NoC -> the receiver's bias wait(VALID) hangs (flaky: 1x1 256->128
                        // flaked, bottleneck conv2 hung deterministically). Sender sends+acks bias but the
                        // receiver never sees bias VALID. Matches the BH ordering requirement.
                        noc.async_writes_flushed();
#endif  // ARCH_BLACKHOLE || ARCH_QUASAR

                        // We should also multicast the flag to destinations
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

#ifndef SKIP_MCAST
                        sender_sem.wait(in1_mcast_num_dests);
                        sender_sem.set(0);
                        // DEBUG: matmul mcast hang — reached iff all BIAS acks received (first block).
                        // If "SEND in1 acked" prints but this does not, the bias mcast handshake is the
                        // stuck point (receivers never reached the bias up() because in1 VALID didn't reach them).
                        if (b == 0 && bh == 0 && bw == 0) {
                            DPRINT("SEND bias acked (got {} dests)\n", in1_mcast_num_dests);
                        }

                        MulticastEndpoint mcast_dst;
                        noc.async_write_multicast(
                            CoreLocalMem<uint32_t>(static_cast<uint32_t>(in3_start_address)),
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
#if defined(ARCH_BLACKHOLE) || defined(ARCH_QUASAR)
                        // Flush the DATA multicast before the VALID-semaphore multicast. On Quasar, without this
                        // barrier the back-to-back in1-then-bias mcasts let the bias VALID semaphore write
                        // race/drop on the NoC -> the receiver's bias wait(VALID) hangs (flaky: 1x1 256->128
                        // flaked, bottleneck conv2 hung deterministically). Sender sends+acks bias but the
                        // receiver never sees bias VALID. Matches the BH ordering requirement.
                        noc.async_writes_flushed();
#endif  // ARCH_BLACKHOLE || ARCH_QUASAR

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
                                        // A DataflowBuffer used as a NoC write source resolves to
                                        // get_read_ptr() + offset_bytes, matching the legacy
                                        // use<CircularBuffer::AddrSelector::READ_PTR>(cb_out) semantics.
                                        noc.async_write(
                                            cb_out,
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
            in1_tensor_start_tile_id += KtNt;
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
    // [DEBUG #47797] Dump SW NoC issued-counters vs HW completion just before the drain. in1 hangs
    // in the nonposted-writes-sent wait (NIU_MST_NONPOSTED_WR_REQ_SENT == noc_nonposted_writes_num_issued).
    // If npw_issued exceeds the actual HW sends (npw_sent==0), a metal2 primitive (likely the in1
    // multicast wrapper counting issued += num_dests while the NIU counts one request) over-incremented.
    DPRINT(
        "[in1-dev] PREBARRIER noc={} reads_issued={} npw_issued={} reads_flushed={} npw_sent={}\n",
        (uint32_t)noc_index,
        (uint32_t)noc_reads_num_issued[noc_index],
        (uint32_t)noc_nonposted_writes_num_issued[noc_index],
        (uint32_t)ncrisc_noc_reads_flushed(noc_index),
        (uint32_t)ncrisc_noc_nonposted_writes_sent(noc_index));

    // Drain outstanding NOC writes AND atomics before returning (Metal 2.0 FW epilogue does not).
    noc.async_full_barrier();
    DPRINT("WSM end\n");  // DEBUG: matmul layer3 hang
}
