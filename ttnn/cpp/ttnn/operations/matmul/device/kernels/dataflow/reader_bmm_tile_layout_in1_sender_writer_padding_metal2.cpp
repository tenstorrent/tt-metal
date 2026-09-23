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
// Optional resource families are gated on preprocessor defines rather than on compile-time argument
// values, because a Metal 2.0 binding token only exists when the host actually binds it:
//   FUSE_BIAS / BIAS_SHARDED  -- dfb::bias, and tensor::bias on the non-sharded bias path
//   IN1_SHARDED               -- in1 arrives in the resident dfb::in1 shard; no tensor binding
//   IN1_DRAM_WIDTH_SHARDED    -- in1 is read bank-by-bank from DRAM using tensor::in1's base address
//   IN1_DRAM_HEIGHT_SHARDED   -- ditto, one complete [K, N] matrix per bank
//   SPARSITY                  -- the sparsity operand (dfb::sparsity + tensor::sparsity) is bound
//   OUT_SHARDED               -- output stays resident in dfb::out; the writer loop is compiled out
//
// The DRAM-width-sharded path takes its per-bank list as runtime varargs, walked two at a time:
// entry 2*k is bank k's stride in bytes, entry 2*k+1 is its bank id.
//
// TWO REGIONS ARE PRESERVED VERBATIM FROM THE LEGACY KERNEL AND ARE NOT CONVERTED. Each is selected
// by a define that no Metal 2.0 factory may set, and each would need work that cannot be written or
// tested from a port:
//
//   * ENABLE_GLOBAL_CB - the legacy "remote CB" path. A GlobalCircularBuffer is a user-managed
//     buffer whose Metal 2.0 analog, GlobalDataflowBuffer, is not implemented; it is NOT a
//     DataflowBuffer, so the region keeps its raw CB index (c_31) in a kernel that otherwise has no
//     CB indices. Only the legacy mcast-1d MeshWorkload paths and the llama all-gather matmul
//     fusion set this define, and neither binds this fork.
//   * FUSE_OP_ALL_GATHER / FUSE_OP_REDUCE_SCATTER - MatmulOpReceiver and OpSignaler consume
//     *positional* runtime args through an index they advance by reference, and they live outside
//     this op's directory (ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp), so they cannot
//     be fed from named arguments without changing a file this port may not touch.
//
// SPARSITY is likewise define-gated rather than arg-gated: the sparsity dataflow buffer and its
// tensor accessor produce no dfb::/tensor:: token at all when the host does not declare them, so the
// references must not reach C++ name lookup. The legacy kernel guarded them with
// `if constexpr (batchB > 0)`, which still looks the names up in the discarded branch.

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
#ifdef ENABLE_GLOBAL_CB
#include "api/remote_circular_buffer.h"
#endif
#include "api/tensor/noc_traits.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "experimental/kernel_args.h"

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
    // kernel's binding set unchanged.
    //
    // Every factory that builds this kernel passes "num_active"; only the sparse matmul factory ever
    // sets it non-zero. 0 means not indexed, i.e. the unchanged dense sparsity-scan path.
    constexpr auto num_active = get_arg(args::num_active);
    constexpr bool use_indices = num_active > 0;
    constexpr uint32_t batch_loop_lim = use_indices ? num_active : batchB_lim;

    const Noc noc;
    DataflowBuffer dfb_in1(dfb::in1);
    DataflowBuffer dfb_out(dfb::out);
    Semaphore sender_sem(sem::in1_mcast_sender);
    Semaphore receiver_sem(sem::in1_mcast_receiver);

#ifdef FUSE_BIAS
    // in3 mcast args
    const uint32_t in3_tensor_start_tile_id = get_arg(args::in3_tensor_start_tile_id);

    constexpr auto in3_tensor_stride_w = get_arg(args::in3_tensor_stride_w);

    DataflowBuffer dfb_in3(dfb::bias);
    // Use the DFB entry size (padded to the DRAM alignment by the factory) for DRAM reads
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

// NOT CONVERTED TO METAL 2.0 -- both blocks are preserved verbatim from the legacy kernel and are
// unreachable here: no Metal 2.0 factory may define either. MatmulOpReceiver and OpSignaler consume
// runtime arguments positionally through the `uint32_t& rt_args_idx` cursor declared above, while
// Metal 2.0 kernels address their arguments by name, so a factory that set one of these defines
// would be feeding them a cursor into an argument block it does not populate. See the header note.
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

// RT and COMPILE TIME ARGS for DRAM sharded weights
#ifdef IN1_DRAM_WIDTH_SHARDED
    const uint32_t vc = get_arg(args::vc);
    const uint32_t num_dram_shards_to_read = get_arg(args::num_dram_shards_to_read);
    const uint32_t dram_tensor_start_offset = get_arg(args::dram_tensor_start_offset);

    constexpr auto in1_dram_block_num_tiles = get_arg(args::in1_dram_block_num_tiles);
    constexpr auto in1_block_w_dram_bytes = get_arg(args::in1_block_w_dram_bytes);
#endif  // IN1_DRAM_WIDTH_SHARDED

#ifdef IN1_DRAM_HEIGHT_SHARDED
    constexpr auto in1_KtNt_per_batch = get_arg(args::in1_KtNt_per_batch);      // K*N tiles per batch
    constexpr auto in1_batches_per_bank = get_arg(args::in1_batches_per_bank);  // batches per DRAM bank
#endif                                                                          // IN1_DRAM_HEIGHT_SHARDED

    constexpr uint32_t in1_single_tile_size_bytes = get_tile_size(dfb::in1);
    // Tiles whose size is not a multiple of the DRAM alignment are padded to it in DRAM, and the
    // interleaved in1 buffer entries are sized to match (see the program factory). On the plain interleaved
    // path the NOC reads the unpadded tile of data into each padded slot and tiles are laid out /
    // multicast at the padded stride. No-op when already aligned. The sharded / DRAM-sharded paths
    // keep their natural (unpadded) stride.
    constexpr uint32_t in1_aligned_tile_size_bytes =
        (in1_single_tile_size_bytes + (DRAM_ALIGNMENT - 1)) & ~(DRAM_ALIGNMENT - 1);
#if !defined(IN1_SHARDED) && !defined(IN1_DRAM_WIDTH_SHARDED) && !defined(IN1_DRAM_HEIGHT_SHARDED) && \
    !defined(ENABLE_GLOBAL_CB)
    constexpr uint32_t in1_block_size_bytes = in1_block_num_tiles * in1_aligned_tile_size_bytes;
#else
    constexpr uint32_t in1_block_size_bytes = in1_block_num_tiles * in1_single_tile_size_bytes;
#endif

    constexpr uint32_t output_single_tile_size_bytes = get_tile_size(dfb::out);

//  READER
#ifdef IN1_SHARDED
    dfb_in1.reserve_back(in1_block_num_tiles * num_blocks_inner_dim);
    dfb_in1.push_back(in1_block_num_tiles * num_blocks_inner_dim);
#elif !defined(ENABLE_GLOBAL_CB)
    [[maybe_unused]] const auto s1 = TensorAccessor(tensor::in1);
#if defined(IN1_DRAM_WIDTH_SHARDED) || defined(IN1_DRAM_HEIGHT_SHARDED)
    // The DRAM-sharded paths below address banks directly rather than paging through the accessor,
    // so they need in1's raw base address. It comes off the binding, never through a runtime arg.
    const uint32_t in1_tensor_addr = s1.get_bank_base_address();
#endif  // IN1_DRAM_WIDTH_SHARDED / IN1_DRAM_HEIGHT_SHARDED
#endif  // IN1_SHARDED / ENABLE_GLOBAL_CB

#ifdef ENABLE_GLOBAL_CB
    // NOT CONVERTED TO METAL 2.0 -- a GlobalCircularBuffer ("remote CB") is not a DataflowBuffer and
    // has no Metal 2.0 analog yet (GlobalDataflowBuffer is unimplemented), so this tensor-prefetcher
    // path is preserved verbatim and no Metal 2.0 factory may define ENABLE_GLOBAL_CB.
    constexpr uint32_t remote_cb_id = tt::CBIndex::c_31;
    const uint32_t in1_fifo_tiles = dfb_in1.get_total_num_entries();
#endif

    //  WRITER
    const auto s = TensorAccessor(tensor::out);
    // `s` is only consumed inside the `#ifndef OUT_SHARDED` write path below; mark it used so
    // sharded builds don't warn (-Wunused-but-set-variable).
    (void)s;

#ifdef SPARSITY
    // sparsity accessor
    DataflowBuffer dfb_sparsity(dfb::sparsity);
    const auto s_sparsity = TensorAccessor(tensor::sparsity);
#endif  // SPARSITY

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
        AllocatorBank<AllocatorBankType::DRAM> dram_src;
        uint32_t in1_dram_batch_offset = in1_batch_in_shard * in1_batch_stride_bytes;
#endif  // IN1_DRAM_HEIGHT_SHARDED

#ifdef SPARSITY
        if constexpr (batchB > 0 && !use_indices) {
            noc.async_read(s_sparsity, dfb_sparsity, sparsity_pagesize, {.page_id = b}, {.offset_bytes = 0});
            noc.async_read_barrier();
        }
#endif  // SPARSITY

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
#ifdef IN1_DRAM_WIDTH_SHARDED
                    // Reset DRAM read offset for each bh block — the inner dim loop
                    // advances through K, and each output row block re-reads the same
                    // in1 columns from K=0. (bw is always 1 for DRAM-sharded senders.)
                    uint32_t l1_read_addr_in1_offset = 0;
#endif  // IN1_DRAM_WIDTH_SHARDED

                    for (uint32_t block = 0; block < num_blocks_inner_dim; ++block) {
#ifdef FUSE_OP_ALL_GATHER
                        fused_op_receiver.update_current_block_start_tile_id(
                            block, in1_tensor_current_inner_dim_block_start_tile_id, in1_batch_tile_id);
#endif  // FUSE_OP_ALL_GATHER
#if defined(ENABLE_GLOBAL_CB)
                        // The tensor prefetcher pushes this receiver's K-blocks in natural order.
                        // Keep one block of lookahead: publish the current block to compute, then
                        // wait for the unpack engine to drain the previous block before returning
                        // its remote-CB credit to the prefetcher.
                        dfb_in1.reserve_back(in1_block_num_tiles);
                        experimental::remote_cb_wait_front(remote_cb_id, block == 0 ? 1u : 2u);
#elif defined(IN1_DRAM_WIDTH_SHARDED)
                        // Operand 1 - DRAM width sharded
                        dfb_in1.reserve_back(in1_block_num_tiles);

                        uint64_t in1_start_address =
                            dfb_in1.get_write_ptr();  // copy start address of block, to be used for mcasting

                        uint32_t l1_write_addr_in1_offset = 0;
                        uint32_t next_bank_id_and_dram_stride_index = 0;

                        AllocatorBank<AllocatorBankType::DRAM> dram_bank;
                        for (uint32_t i = 0; i < num_dram_shards_to_read; ++i) {
                            uint32_t shard_bank_id = get_vararg(next_bank_id_and_dram_stride_index + 1);
                            uint32_t shard_base_addr = in1_tensor_addr;
                            if (i == 0) {
                                shard_base_addr += dram_tensor_start_offset;
                            }
                            noc.set_async_read_state<NocOptions::CUSTOM_VC, NOC_MAX_BURST_SIZE>(
                                dram_bank,
                                in1_single_tile_size_bytes,
                                {.bank_id = shard_bank_id, .addr = shard_base_addr},
                                NocOptVals{.vc = vc});

                            uint32_t l1_read_addr_in1 = l1_read_addr_in1_offset;
                            uint32_t l1_write_addr_in1 = dfb_in1.get_write_ptr() + l1_write_addr_in1_offset;
                            uint32_t in1_block_w_dram =
                                get_vararg(next_bank_id_and_dram_stride_index) / in1_single_tile_size_bytes;

                            for (uint32_t m = 0; m < in1_block_h; ++m) {
                                uint32_t l1_read_addr_in1_temp = l1_read_addr_in1;
                                uint32_t l1_write_addr_in1_temp = l1_write_addr_in1;
                                for (uint32_t w = 0; w < in1_block_w_dram; ++w) {
                                    noc.async_read_with_state<NocOptions::CUSTOM_VC, NOC_MAX_BURST_SIZE>(
                                        dram_bank,
                                        CoreLocalMem<uint32_t>(l1_write_addr_in1_temp),
                                        in1_single_tile_size_bytes,
                                        {.bank_id = shard_bank_id, .addr = shard_base_addr + l1_read_addr_in1_temp},
                                        {},
                                        NocOptVals{.vc = vc});
                                    l1_read_addr_in1_temp += in1_single_tile_size_bytes;
                                    l1_write_addr_in1_temp += in1_single_tile_size_bytes;
                                }
                                l1_read_addr_in1 += in1_block_w_dram_bytes;
                                l1_write_addr_in1 += in1_block_w_bytes;
                            }
                            l1_write_addr_in1_offset += get_vararg(next_bank_id_and_dram_stride_index);
                            next_bank_id_and_dram_stride_index += 2;
                        }
                        l1_read_addr_in1_offset += in1_dram_block_size_bytes;
                        noc.async_read_barrier();
#elif defined(IN1_DRAM_HEIGHT_SHARDED)
                        // Operand 1 - DRAM height sharded (batched)
                        // Each DRAM bank holds batches_per_bank complete [K, N] matrices
                        // Bank and offset computed at start of batch loop
                        dfb_in1.reserve_back(in1_block_num_tiles);

                        uint32_t l1_write_addr_in1 = dfb_in1.get_write_ptr();
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
                                        CoreLocalMem<uint32_t>(l1_write_addr_in1),
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
                        dfb_in1.reserve_back(in1_block_num_tiles);
                        uint32_t in1_write_offset = 0;
                        const uint64_t in1_start_address =
                            dfb_in1.get_write_ptr();  // copy start address of block, to be used for mcasting

                        // Copy in1 block into the buffer, as the default kernel
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
#endif  // IN1_DRAM_WIDTH_SHARDED / IN1_DRAM_HEIGHT_SHARDED / IN1_SHARDED

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

#ifndef IN1_SHARDED
                        dfb_in1.push_back(in1_block_num_tiles);
#endif  // IN1_SHARDED
#ifdef ENABLE_GLOBAL_CB
                        if (block >= 1) {
                            while (!dfb_in1.pages_reservable_at_back(in1_fifo_tiles - in1_block_num_tiles)) {
                                invalidate_l1_cache();
                            }
                            experimental::remote_cb_pop_front(remote_cb_id, 1);
                        }
#endif
                    }
#ifdef ENABLE_GLOBAL_CB
                    if (num_blocks_inner_dim > 0) {
                        while (!dfb_in1.pages_reservable_at_back(in1_fifo_tiles)) {
                            invalidate_l1_cache();
                        }
                        experimental::remote_cb_pop_front(remote_cb_id, 1);
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

#ifdef IN1_DRAM_WIDTH_SHARDED
                        uint32_t l1_write_addr_in3_offset = 0;
                        uint32_t next_bank_id_and_dram_stride_index = 0;

                        // Bank-direct reads need bias's raw base address; it comes off the binding.
                        const uint32_t in3_tensor_addr = s3.get_bank_base_address();

                        AllocatorBank<AllocatorBankType::DRAM> bias_dram_bank;
                        for (uint32_t i = 0; i < num_dram_shards_to_read; ++i) {
                            uint32_t bias_shard_bank_id = get_vararg(next_bank_id_and_dram_stride_index + 1);
                            uint32_t bias_shard_base_addr = in3_tensor_addr;
                            if (i == 0) {
                                // dram_tensor_start_offset is in in1 tile bytes; convert to
                                // bias tile bytes since bias_dtype may differ from in1_dtype.
                                bias_shard_base_addr += (dram_tensor_start_offset / in1_single_tile_size_bytes) *
                                                        bias_single_tile_size_bytes;
                            }

                            noc.set_async_read_state<NocOptions::CUSTOM_VC, NOC_MAX_BURST_SIZE>(
                                bias_dram_bank,
                                bias_single_tile_size_bytes,
                                {.bank_id = bias_shard_bank_id, .addr = bias_shard_base_addr},
                                NocOptVals{.vc = vc});

                            uint32_t l1_read_addr_in3 = 0;
                            uint32_t l1_write_addr_in3 = dfb_in3.get_write_ptr() + l1_write_addr_in3_offset;
                            // the stride vararg is in in1 tile bytes, so divide
                            // by in1_single_tile_size_bytes (not bias) to get the tile count.
                            uint32_t in3_block_w_dram =
                                get_vararg(next_bank_id_and_dram_stride_index) / in1_single_tile_size_bytes;

                            for (uint32_t w = 0; w < in3_block_w_dram; ++w) {
                                noc.async_read_with_state<NocOptions::CUSTOM_VC, NOC_MAX_BURST_SIZE>(
                                    bias_dram_bank,
                                    CoreLocalMem<uint32_t>(l1_write_addr_in3),
                                    bias_single_tile_size_bytes,
                                    {.bank_id = bias_shard_bank_id, .addr = bias_shard_base_addr + l1_read_addr_in3},
                                    {},
                                    NocOptVals{.vc = vc});
                                l1_read_addr_in3 += bias_single_tile_size_bytes;
                                l1_write_addr_in3 += bias_single_tile_size_bytes;
                                in3_block_size_bytes += bias_single_tile_size_bytes;
                            }
                            // Advance L1 offset in bias tile bytes, not in1 stride bytes.
                            l1_write_addr_in3_offset += in3_block_w_dram * bias_single_tile_size_bytes;
                            next_bank_id_and_dram_stride_index += 2;
                        }
                        noc.async_read_barrier();
#else
                        // Copy in1 block into the buffer, as the default kernel
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
#endif  // IN1_DRAM_WIDTH_SHARDED

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
#ifndef IN1_DRAM_HEIGHT_SHARDED
            // For height-sharded DRAM, tile IDs are relative within a batch;
            // batch offset is handled by switching DRAM banks
            in1_tensor_start_tile_id += KtNt;
#endif
        }

#ifdef FUSE_OP_REDUCE_SCATTER
        // Signal reduce_scatter to go
        op_signaler.synchronize_workers_and_signal_op(0);
#endif  // FUSE_OP_REDUCE_SCATTER
    }

#ifdef OUT_SHARDED
    dfb_out.wait_front(static_cast<uint16_t>(
        batch * out_num_nonzero_subblocks_h * out_num_nonzero_subblocks_w * out_subblock_w * out_subblock_h));
#endif
#ifdef ENABLE_GLOBAL_CB
    experimental::update_remote_cb_config_in_l1(remote_cb_id);
#endif
    // #53329: this kernel issues non-posted NOC atomics (multicast semaphore increments, and
    // OpSignaler in the fused reduce-scatter path). Flushing only writes lets the kernel retire
    // with atomics still unacknowledged -> watcher reports an inter-kernel data race under load.
    // Barrier both, unconditionally, mirroring the CCL reader fix in #53595.
    noc.async_atomic_barrier();
    noc.async_write_barrier();
}
