// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of reader_bmm_tile_layout_in1_sender_writer_padding.cpp, which lives beside it.
// Factories ported to Metal 2.0 bind this fork; the original serves the consumers still on the
// legacy ProgramDescriptor API. Until the last of them migrates and the original is retired, changes
// to either copy likely belong in the other too.
//
// The binding and argument names below are this fork's interface: every factory that later ports
// onto it inherits them and cannot rename them.
//
// THREE REGIONS OF THE LEGACY KERNEL ARE NOT CARRIED HERE. Each is selected by a define that no
// Metal 2.0 consumer can currently set, and each would need work that cannot be written or tested
// from this port. A factory that needs one of them is not yet portable onto this fork.
//
//   * ENABLE_GLOBAL_CB - the legacy "remote CB" path. A GlobalCircularBuffer is a user-managed
//     buffer whose Metal 2.0 analog, GlobalDataflowBuffer, is not implemented; it is NOT a
//     DataflowBuffer. Carrying the region would mean carrying a raw CB index (c_31) into a kernel
//     that has no CB indices. Only the legacy mcast-1d MeshWorkload paths and the llama all-gather
//     matmul fusion set this define, and neither can bind a Metal 2.0 fork.
//   * IN1_DRAM_WIDTH_SHARDED / IN1_DRAM_HEIGHT_SHARDED - the DRAM-sharded weight readers, set only
//     by the (still unported) mcast-2d factory. They use the in1 base address as a raw pointer with
//     explicit bank arithmetic, which in Metal 2.0 must come from TensorAccessor's
//     get_bank_base_address() bridge, and they walk per-bank stride/id lists that become runtime
//     varargs. Converting them blind, with no consumer able to exercise them, would ship untested
//     address arithmetic; the mcast-2d porter should add them here when that factory ports.
//
// Two further regions are gated behind defines rather than compile-time args, because each needs a
// resource that only exists when the feature is on:
//   * SPARSITY - the sparsity dataflow buffer and its tensor accessor. A binding the host does not
//     declare produces no dfb::/tensor:: token at all, so the references must not reach C++ name
//     lookup; the legacy kernel guarded them with `if constexpr (batchB > 0)`, which still looks
//     names up in the discarded branch.
//   * FUSE_OP_ALL_GATHER / FUSE_OP_REDUCE_SCATTER - MatmulOpReceiver and OpSignaler consume
//     *positional* runtime args through an index they advance by reference, and they live outside
//     this op's directory (ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp), so they cannot
//     be fed from named arguments without changing a file this port may not touch.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "hostdevcommon/common_values.hpp"
#if defined(FUSE_OP_ALL_GATHER) || defined(FUSE_OP_REDUCE_SCATTER)
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#endif
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "experimental/kernel_args.h"

#ifdef ENABLE_PREFETCHER_PIPE
#ifdef ARCH_QUASAR
#error "PrefetcherPipe weight delivery into this matmul needs matmul_block_in1_at, which Quasar lacks"
#endif
#include "api/dataflow/prefetcher_pipe.h"
#endif

void kernel_main() {
    // READER
#if defined(FUSE_OP_ALL_GATHER) || defined(FUSE_OP_REDUCE_SCATTER)
    uint32_t rt_args_idx = 0;
#endif
    // in1 tensor args
    uint32_t in1_tensor_start_tile_id = get_arg(args::in1_tensor_start_tile_id);
    // in1 mcast args
    const uint32_t in1_mcast_dest_noc_start_x = get_arg(args::in1_mcast_dest_noc_start_x);
    const uint32_t in1_mcast_dest_noc_start_y = get_arg(args::in1_mcast_dest_noc_start_y);
    const uint32_t in1_mcast_dest_noc_end_x = get_arg(args::in1_mcast_dest_noc_end_x);
    const uint32_t in1_mcast_dest_noc_end_y = get_arg(args::in1_mcast_dest_noc_end_y);

    // WRITER
    // out tensor args
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
    constexpr auto in1_tensor_stride_w = get_arg(args::in1_tensor_stride_w);
    constexpr auto in1_tensor_stride_h = get_arg(args::in1_tensor_stride_h);
    constexpr auto in1_tensor_next_block_stride = get_arg(args::in1_tensor_next_block_stride);
    constexpr auto in1_tensor_next_w_dim_block_stride = get_arg(args::in1_tensor_next_w_dim_block_stride);
    // in1 block args
    constexpr auto in1_block_w = get_arg(args::in1_block_w);
    constexpr auto in1_block_h = get_arg(args::in1_block_h);
    constexpr auto in1_block_num_tiles = get_arg(args::in1_block_num_tiles);
    // in0/in1 common args
    constexpr auto num_blocks_inner_dim = get_arg(args::num_blocks_inner_dim);
    constexpr auto num_blocks_w_dim = get_arg(args::num_blocks_w_dim);
    constexpr auto num_blocks_h_dim = get_arg(args::num_blocks_h_dim);

    // in1 mcast args
    constexpr auto in1_mcast_num_dests = get_arg(args::in1_mcast_num_dests);
    constexpr auto in1_mcast_num_cores = get_arg(args::in1_mcast_num_cores);
    // batch args
    constexpr auto KtNt = get_arg(args::KtNt);
    constexpr auto batch = get_arg(args::batch);
    constexpr auto bcast_B = get_arg(args::bcast_B);
    // sparsity args
    constexpr auto batchB = get_arg(args::batchB);
#ifdef SPARSITY
    constexpr auto sparsity_pagesize = get_arg(args::sparsity_pagesize);
#endif

    // WRITER
    // out tensor args
    constexpr auto out_tensor_stride_w = get_arg(args::out_tensor_stride_w);
    constexpr auto out_tensor_stride_h = get_arg(args::out_tensor_stride_h);
    constexpr auto out_tensor_next_subblock_stride_w = get_arg(args::out_tensor_next_subblock_stride_w);
    constexpr auto out_tensor_next_subblock_stride_h = get_arg(args::out_tensor_next_subblock_stride_h);
    constexpr auto out_tensor_next_w_dim_block_stride = get_arg(args::out_tensor_next_w_dim_block_stride);
    constexpr auto out_tensor_next_h_dim_block_stride = get_arg(args::out_tensor_next_h_dim_block_stride);
    // out subblock args
    constexpr auto out_subblock_w = get_arg(args::out_subblock_w);
    constexpr auto out_subblock_h = get_arg(args::out_subblock_h);
    constexpr auto out_subblock_tile_count = get_arg(args::out_subblock_tile_count);
    // batch args
    constexpr auto MtNt = get_arg(args::MtNt);  // if 0
    // Don't need batch; same as batch from READER args
    constexpr bool compact_output = get_arg(args::compact_output);

    // When sparsity is disabled, we just loop once
    constexpr uint32_t batchB_lim = batchB == 0 ? 1u : batchB;

    // Indexed/gather mode: iterate only the num_active sparse groups named by the caller's `indices`
    // operand. Each iteration gathers the weights of group indices[i] and writes its result to COMPACT
    // output slot i, so there is no sparsity scan and no skipped slot.
    //
    // The id list rides in the sparsity operand's plumbing (tensor binding, sparsity DataflowBuffer):
    // the sparsity mask itself is never read in this mode, so reusing those slots keeps this shared
    // kernel's argument surface unchanged.
    //
    // Every factory that builds this kernel passes "num_active"; only the sparse matmul factory ever
    // sets it non-zero. 0 means not indexed, i.e. the unchanged dense sparsity-scan path.
    constexpr auto num_active = get_arg(args::num_active);
    constexpr bool use_indices = num_active > 0;
    constexpr uint32_t batch_loop_lim = use_indices ? num_active : batchB_lim;

    const Noc noc;
    // in1 is filled here (from DRAM or the local shard) and drained by the compute kernel; out is
    // filled by the compute kernel's packer and drained here.
    DataflowBuffer dfb_in1(dfb::in1);
    DataflowBuffer dfb_out(dfb::out);
    Semaphore sender_sem(sem::in1_mcast_sender);
    Semaphore receiver_sem(sem::in1_mcast_receiver);

#ifdef FUSE_BIAS
    // in3 mcast args
    const uint32_t in3_tensor_start_tile_id = get_arg(args::in3_tensor_start_tile_id);

    constexpr auto in3_tensor_stride_w = get_arg(args::in3_tensor_stride_w);

    // bias is filled here from DRAM (or resident when sharded) and consumed by the compute kernel's
    // bias add.
    DataflowBuffer dfb_in3(dfb::bias);
    // Use the buffer's entry size (padded to the DRAM alignment by the factory) for DRAM reads
    // and L1 write strides, NOT the raw tile size. On Blackhole, the DRAM read alignment
    // is 64B, so a sub-64B tile (e.g. 32B for a (1,16) bf16 bias tile) cannot be read
    // directly from DRAM, and 32B-strided L1 writes land at non-64B-aligned addresses
    // that disagree with the 64B-aligned DRAM source. The factory pads the entry to
    // 64B; the unpacker still reads the actual 32B tile from the padded entry via tile
    // dims. For tiles already >= dram_alignment (e.g. 32x32 bf16 = 2048B), the entry size
    // equals the tile size, so this is a no-op. Mirrors how in0/in1 readers walk at the
    // aligned stride.
    const uint32_t bias_single_tile_size_bytes = dfb_in3.get_entry_size();

#ifndef BIAS_SHARDED
    const auto s3 = TensorAccessor(tensor::bias);
#endif  // BIAS_SHARDED
#endif  // FUSE_BIAS

#ifndef OUT_SHARDED
    const uint32_t last_num_blocks_w_dim = get_arg(args::last_num_blocks_w_dim);
#endif  // OUT_SHARDED

#ifdef FUSE_OP_ALL_GATHER
    MatmulOpReceiver fused_op_receiver = MatmulOpReceiver(
        false, /* wait_for_op_signal */
        rt_args_idx,
        num_blocks_inner_dim,
        in1_block_h /* tiles_per_block (in the same dimension */
    );
#elif defined(FUSE_OP_REDUCE_SCATTER)
    OpSignaler op_signaler = OpSignaler(rt_args_idx);
#endif

    constexpr uint32_t in1_single_tile_size_bytes = get_tile_size(dfb::in1);
    // Tiles whose size is not a multiple of the DRAM alignment are padded to it in DRAM, and the
    // interleaved in1 buffer entries are sized to match (see the program factory). On the plain
    // interleaved path the NOC reads the unpadded tile of data into each padded slot and tiles are
    // laid out / multicast at the padded stride. No-op when already aligned. The sharded path keeps
    // its natural (unpadded) stride.
    constexpr uint32_t in1_aligned_tile_size_bytes =
        (in1_single_tile_size_bytes + (DRAM_ALIGNMENT - 1)) & ~(DRAM_ALIGNMENT - 1);
#ifndef IN1_SHARDED
    constexpr uint32_t in1_block_size_bytes = in1_block_num_tiles * in1_aligned_tile_size_bytes;
#else
    constexpr uint32_t in1_block_size_bytes = in1_block_num_tiles * in1_single_tile_size_bytes;
#endif

    constexpr uint32_t output_single_tile_size_bytes = get_tile_size(dfb::out);

//  READER
#if defined(ENABLE_PREFETCHER_PIPE)
    // in1 is a relay laid over this worker's PrefetcherPipe ring, so the prefetcher's K-blocks arrive
    // already in place: this kernel only turns a delivered entry into in1 credit for compute and, once
    // compute is done with it, that entry's credit back into an ack to the sender. One accessor names
    // every pipe; the one present on this worker is the one bound here. bind_relay() aligns in1 to the
    // pipe's durable cursor (firmware resets it at launch) and makes pop_front wait for compute. The
    // pipe lives to the end of kernel_main; its destructor stores the cursor back.
    experimental::PrefetcherPipe pipe(pipe::in1);
    auto in1_relay = pipe.bind_relay();
#elif defined(IN1_SHARDED)
    dfb_in1.reserve_back(in1_block_num_tiles * num_blocks_inner_dim);
    dfb_in1.push_back(in1_block_num_tiles * num_blocks_inner_dim);
#else
    [[maybe_unused]] const auto s1 = TensorAccessor(tensor::in1);
#endif  // ENABLE_PREFETCHER_PIPE / IN1_SHARDED

    //  WRITER
    const auto s = TensorAccessor(tensor::out);
    // `s` is only consumed inside the `#ifndef OUT_SHARDED` write path below; mark it used so
    // sharded builds don't warn (-Wunused-but-set-variable).
    (void)s;

    // sparsity accessor
#ifdef SPARSITY
    DataflowBuffer dfb_sparsity(dfb::sparsity);
    const auto s_sparsity = TensorAccessor(tensor::sparsity);
#endif

#ifndef SKIP_MCAST
    // Set ur local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    receiver_sem.set(VALID);
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast

#ifdef IN1_SHARDED
    uint64_t in1_start_address = dfb_in1.get_write_ptr();
#endif  // IN1_SHARDED
#endif  // SKIP_MCAST

#ifdef SPARSITY
    uint32_t l1_write_addr_sparsity = 0;
    if constexpr (batchB > 0) {
        dfb_sparsity.reserve_back(1);
        l1_write_addr_sparsity = dfb_sparsity.get_write_ptr();
    }

    if constexpr (use_indices) {
        // Indexed/gather mode: the sparsity operand slot carries the active-group id list instead of
        // the mask. It is a single ROW_MAJOR stick (validated on the host), so one page-0 read pulls
        // in the whole list, once, for every outer batch.
        noc.async_read(s_sparsity, dfb_sparsity, sparsity_pagesize, {.page_id = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
    }
#endif  // SPARSITY

    for (uint32_t b = 0; b < batch; ++b) {
        uint32_t in1_batch_tile_id = in1_tensor_start_tile_id;

#ifdef SPARSITY
        if constexpr (batchB > 0 && !use_indices) {
            noc.async_read(s_sparsity, dfb_sparsity, sparsity_pagesize, {.page_id = b}, {.offset_bytes = 0});
            noc.async_read_barrier();
        }
#endif

        // Indexed/gather mode writes to compact output slots, so capture this outer batch's output
        // base and index it by the compact slot (the loop counter) each iteration.
        [[maybe_unused]] const uint32_t out_base_tile_id = out_tensor_start_tile_id;

        for (uint32_t bB = 0; bB < batch_loop_lim; ++bB) {
#ifdef SPARSITY
            if constexpr (use_indices) {
                // Gather: jump straight to group indices[bB]'s weight block, scatter its result to
                // compact output slot bB. Every iterated group is active, so nothing is skipped.
                const uint32_t group_id = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_addr_sparsity)[bB];
                // The ids are device-resident, so the host can only bound their count, not their
                // values. An out-of-range id would silently read an unrelated weight block; assert
                // loudly (under watcher) instead, as the in0 sender does for the exact-nnz contract.
                ASSERT(group_id < batchB);
                in1_batch_tile_id = in1_tensor_start_tile_id + group_id * KtNt;
                out_tensor_start_tile_id = out_base_tile_id + bB * MtNt;
            } else if constexpr (batchB > 0) {
                if (reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_addr_sparsity)[bB] == 0) {
                    if constexpr (!compact_output) {
                        out_tensor_start_tile_id += MtNt;
                    }
                    in1_batch_tile_id += KtNt;
                    continue;
                }
            }
#endif  // SPARSITY

            const uint32_t in1_tensor_current_h_dim_block_tile_id = in1_batch_tile_id;
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
#ifdef FUSE_OP_ALL_GATHER
                        fused_op_receiver.update_current_block_start_tile_id(
                            block, in1_tensor_current_inner_dim_block_start_tile_id, in1_batch_tile_id);
#endif
#if defined(ENABLE_PREFETCHER_PIPE)
                        // One K-block of lookahead over the pipe: publish this block to compute, then
                        // hand the previous block's entry back to the sender once compute has drained
                        // it. One in1 entry is one K-block, which is also one pipe entry.
                        in1_relay.reserve_back(1);
                        pipe.wait_front(block == 0 ? 1u : 2u);
#elif !defined(IN1_SHARDED)
                        // Operand 1 - interleaved
                        dfb_in1.reserve_back(in1_block_num_tiles);
                        uint32_t in1_write_offset = 0;
                        const uint64_t in1_start_address =
                            dfb_in1.get_write_ptr();  // copy start address of block, to be used for mcasting

                        // Copy in1 block into the in1 buffer, as the default kernel
                        uint32_t in1_tensor_row_start_tile_id = in1_tensor_current_inner_dim_block_start_tile_id;
                        for (uint32_t h = 0; h < in1_block_h; ++h) {
                            uint32_t in1_tensor_tile_id = in1_tensor_row_start_tile_id;
                            for (uint32_t w = 0; w < in1_block_w; ++w) {
                                if (bw < num_blocks_w_dim - 1 || w < last_block_w) {
                                    noc.async_read(
                                        s1,
                                        dfb_in1,
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
#endif  // ENABLE_PREFETCHER_PIPE / IN1_SHARDED

#ifndef SKIP_MCAST
                        // wait until all in1 mcast destinations have atomically incremented the in1 semaphore_addr
                        // (i.e. its value should be in0_mcast_num_dests), then reset the semaphore_addr value back to
                        // zero for the next block
                        sender_sem.wait(in1_mcast_num_dests);
                        sender_sem.set(0);

                        // Now we have the block in the buffer's address, we can mcast to dests!
                        const MulticastEndpoint mcast_dst;
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

#if defined(ENABLE_PREFETCHER_PIPE)
                        // pop_front waits for compute to have popped that block out of in1 before
                        // acking it, so no free-space spin is needed. Publish only through the relay
                        // view: pushing dfb_in1 as well would double the credit compute sees.
                        in1_relay.push_back(1);
                        if (block >= 1) {
                            pipe.pop_front(1, noc);
                        }
#elif !defined(IN1_SHARDED)
                        dfb_in1.push_back(in1_block_num_tiles);
#endif  // ENABLE_PREFETCHER_PIPE / IN1_SHARDED
                    }
#ifdef ENABLE_PREFETCHER_PIPE
                    if (num_blocks_inner_dim > 0) {
                        pipe.pop_front(1, noc);
                    }
#endif
#ifdef FUSE_BIAS
                    // Only read bias on first batch, or we have multiple output blocks
                    if ((b == 0 && bh == 0) || num_blocks_w_dim > 1) {
                        // Operand 1
#ifndef BIAS_SHARDED
                        dfb_in3.reserve_back(in1_block_w);
                        uint32_t in3_write_offset = 0;

                        const uint64_t in3_start_address =
                            dfb_in3.get_write_ptr();        // copy start address of block, to be used for mcasting
                        uint32_t in3_block_size_bytes = 0;  // can be optimized later, pass it to kernel

                        // Copy in1 block into the bias buffer, as the default kernel
                        uint32_t in3_tensor_tile_id = in3_tensor_current_w_dim_block_tile_id;
                        for (uint32_t w = 0; w < in1_block_w; ++w) {
                            if (bw < num_blocks_w_dim - 1 || w < last_block_w) {
                                noc.async_read(
                                    s3,
                                    dfb_in3,
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

                        // wait until all in1 mcast destinations have atomically incremented the in1 semaphore_addr
                        // (i.e. its value should be in0_mcast_num_dests), then reset the semaphore_addr value back to
                        // zero for the next block
                        sender_sem.wait(in1_mcast_num_dests);
                        sender_sem.set(0);

                        // Now we have the block in the buffer's address, we can mcast to dests!
                        const MulticastEndpoint mcast_dst;
                        // num_dests must not include source, since we are NOT really doing a local copy!
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

                        dfb_in3.push_back(in1_block_w);
#else
                        dfb_in3.reserve_back(in1_block_w);
                        dfb_in3.push_back(in1_block_w);
#endif  // BIAS_SHARDED
                    }
#endif  // FUSE_BIAS

#ifndef OUT_SHARDED
                    // WRITER
                    const uint32_t num_blocks_w_dim_ =
                        bw >= last_num_blocks_w_dim - 1 ? last_num_blocks_w_dim : num_blocks_w_dim;
                    const uint32_t out_num_nonzero_subblocks_h_ = out_num_nonzero_subblocks_h;
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

                            dfb_out.wait_front(out_subblock_tile_count);
                            uint32_t out_read_offset = 0;

                            for (uint32_t h = 0; h < out_subblock_h_; ++h) {
                                uint32_t out_tensor_tile_id = out_tensor_sb_row_start_tile_id;
                                for (uint32_t w = 0; w < out_subblock_w_; ++w) {
                                    if (bw < num_blocks_w_dim_) {
                                        noc.async_write(
                                            dfb_out,
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
                            dfb_out.pop_front(out_subblock_tile_count);
                            out_tensor_sbw_start_tile_id += out_tensor_next_subblock_stride_w;
                        }
                        // Pop fully padded subblocks along the row
                        if (bw == num_blocks_w_dim_ - 1) {
                            dfb_out.wait_front(static_cast<uint16_t>(padded_block_tiles_w_skip));
                            dfb_out.pop_front(static_cast<uint16_t>(padded_block_tiles_w_skip));
                        }
                        out_tensor_sbh_start_tile_id += out_tensor_next_subblock_stride_h;
                    }
                    // Pop row(s) of fully padded subblocks
                    if (bh == num_blocks_h_dim - 1) {
                        dfb_out.wait_front(static_cast<uint16_t>(padded_block_tiles_h_skip));
                        dfb_out.pop_front(static_cast<uint16_t>(padded_block_tiles_h_skip));
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

#ifdef FUSE_OP_REDUCE_SCATTER
        // Signal reduce_scatter to go
        op_signaler.synchronize_workers_and_signal_op(0);
#endif
    }

#ifdef OUT_SHARDED
    dfb_out.wait_front(static_cast<uint16_t>(
        batch * out_num_nonzero_subblocks_h * out_num_nonzero_subblocks_w * out_subblock_w * out_subblock_h));
#endif
    // #53329: this kernel issues non-posted NOC atomics (multicast semaphore increments, and
    // OpSignaler in the fused reduce-scatter path). Flushing only writes lets the kernel retire
    // with atomics still unacknowledged -> watcher reports an inter-kernel data race under load.
    // Barrier both, unconditionally, mirroring the CCL reader fix in #53595.
    noc.async_atomic_barrier();
    noc.async_write_barrier();
}
