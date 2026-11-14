// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "hostdevcommon/common_values.hpp"
#if defined(FUSE_OP_ALL_GATHER) || defined(FUSE_OP_REDUCE_SCATTER)
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#endif
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#ifdef ENABLE_GLOBAL_CB
#include "api/remote_circular_buffer.h"
#endif
#include "api/tensor/noc_traits.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args_metal2.hpp"

#ifdef ENABLE_PREFETCHER_PIPE
#include "ttnn/operations/matmul/device/kernels/dataflow/prefetcher_pipe_in1_window.hpp"
#endif

template <
    uint32_t BLOCK_H,
    uint32_t BLOCK_W,
    uint32_t STRIDE_W,
    uint32_t STRIDE_H,
    uint32_t TILE_BYTES,
    uint32_t ALIGNED_TILE_BYTES,
    typename Accessor>
__attribute__((noinline)) void read_in1_block(
    const Noc noc, const Accessor s1, uint32_t l1_write_addr, uint32_t start_tile_id, uint32_t block_w_to_read) {
    uint32_t row_start_tile_id = start_tile_id;
    for (uint32_t h = 0; h < BLOCK_H; ++h) {
        uint32_t tile_id = row_start_tile_id;
        uint32_t tile_l1_addr = l1_write_addr;
        for (uint32_t w = 0; w < block_w_to_read; ++w) {
            noc.async_read(s1, CoreLocalMem<uint32_t>(tile_l1_addr), TILE_BYTES, {.page_id = tile_id}, {});
            tile_l1_addr += ALIGNED_TILE_BYTES;
            tile_id += STRIDE_W;
        }
        l1_write_addr += BLOCK_W * ALIGNED_TILE_BYTES;
        row_start_tile_id += STRIDE_H;
    }
}

void kernel_main() {
#if defined(FUSE_OP_ALL_GATHER) || defined(FUSE_OP_REDUCE_SCATTER)
    uint32_t rt_args_idx = 0;
#endif
    uint32_t in1_tensor_start_tile_id = get_arg(args::in1_tensor_start_tile_id);

    uint32_t out_tensor_start_tile_id = get_arg(args::out_tensor_start_tile_id);

    const uint32_t last_block_w = get_arg(args::last_block_w);
    const uint32_t out_num_nonzero_subblocks_h = get_arg(args::out_num_nonzero_subblocks_h);
    const uint32_t out_last_subblock_h = get_arg(args::out_last_subblock_h);
    const uint32_t padded_block_tiles_h_skip = get_arg(args::padded_block_tiles_h_skip);
    const uint32_t out_num_nonzero_subblocks_w = get_arg(args::out_num_nonzero_subblocks_w);
    const uint32_t out_last_num_nonzero_subblocks_w = get_arg(args::out_last_num_nonzero_subblocks_w);
    const uint32_t out_last_subblock_w = get_arg(args::out_last_subblock_w);
    const uint32_t padded_subblock_tiles_addr_skip = get_arg(args::padded_subblock_tiles_addr_skip);
    const uint32_t padded_block_tiles_w_skip = get_arg(args::padded_block_tiles_w_skip);

    constexpr auto in1_tensor_stride_w = get_arg(args::in1_tensor_stride_w);
    constexpr auto in1_tensor_stride_h = get_arg(args::in1_tensor_stride_h);
    constexpr auto in1_tensor_next_block_stride = get_arg(args::in1_tensor_next_block_stride);
    constexpr auto in1_tensor_next_w_dim_block_stride = get_arg(args::in1_tensor_next_w_dim_block_stride);
    constexpr auto in1_block_w = get_arg(args::in1_block_w);
    constexpr auto in1_block_h = get_arg(args::in1_block_h);
    constexpr auto in1_block_num_tiles = get_arg(args::in1_block_num_tiles);
    constexpr auto num_blocks_inner_dim = get_arg(args::num_blocks_inner_dim);
    constexpr auto num_blocks_w_dim = get_arg(args::num_blocks_w_dim);
    constexpr auto num_blocks_h_dim = get_arg(args::num_blocks_h_dim);

    constexpr auto KtNt = get_arg(args::KtNt);
    constexpr auto batch = get_arg(args::batch);
    constexpr auto bcast_B = get_arg(args::bcast_B);
    constexpr auto batchB = get_arg(args::batchB);
#ifdef SPARSITY
    constexpr auto sparsity_pagesize = get_arg(args::sparsity_pagesize);
#endif

    constexpr auto out_tensor_stride_w = get_arg(args::out_tensor_stride_w);
    constexpr auto out_tensor_stride_h = get_arg(args::out_tensor_stride_h);
    constexpr auto out_tensor_next_subblock_stride_w = get_arg(args::out_tensor_next_subblock_stride_w);
    constexpr auto out_tensor_next_subblock_stride_h = get_arg(args::out_tensor_next_subblock_stride_h);
    constexpr auto out_tensor_next_w_dim_block_stride = get_arg(args::out_tensor_next_w_dim_block_stride);
    constexpr auto out_tensor_next_h_dim_block_stride = get_arg(args::out_tensor_next_h_dim_block_stride);
    constexpr auto out_subblock_w = get_arg(args::out_subblock_w);
    constexpr auto out_subblock_h = get_arg(args::out_subblock_h);
    constexpr auto out_subblock_tile_count = get_arg(args::out_subblock_tile_count);
    constexpr auto MtNt = get_arg(args::MtNt);
    constexpr bool compact_output = get_arg(args::compact_output) != 0;

    constexpr uint32_t batchB_lim = batchB == 0 ? 1u : batchB;

    constexpr auto num_active = get_arg(args::num_active);
    constexpr bool use_indices = num_active > 0;
    constexpr uint32_t batch_loop_lim = use_indices ? num_active : batchB_lim;

    const Noc noc;
    DataflowBuffer dfb_in1(dfb::in1);
#ifndef OUT_WRITER_ON_IN0
    DataflowBuffer dfb_out(dfb::out);
#endif
    constexpr auto in1_mcast = MCAST_ARGS(in1_mcast);
    auto sender = in1_mcast.optional_sender(noc);
#if defined(ENABLE_PREFETCHER_PIPE) || defined(ENABLE_GLOBAL_CB)
    static_assert(!in1_mcast.active, "in1 lands on every worker directly, so this kernel never multicasts it");
#endif

#ifdef FUSE_BIAS
    const uint32_t in3_tensor_start_tile_id = get_arg(args::in3_tensor_start_tile_id);
#ifdef BIAS_PER_GROUP
    constexpr bool bias_per_group = true;
#else
    constexpr bool bias_per_group = false;
#endif

    constexpr auto in3_tensor_stride_w = get_arg(args::in3_tensor_stride_w);

    DataflowBuffer dfb_in3(dfb::bias);
    const uint32_t bias_single_tile_size_bytes = dfb_in3.get_entry_size();

#ifndef BIAS_SHARDED
    const auto s3 = TensorAccessor(tensor::bias);
#endif
#endif
#ifndef OUT_SHARDED
    const uint32_t last_num_blocks_w_dim = get_arg(args::last_num_blocks_w_dim);
#endif

#ifdef FUSE_OP_ALL_GATHER
    MatmulOpReceiver fused_op_receiver = MatmulOpReceiver(false, rt_args_idx, num_blocks_inner_dim, in1_block_h);
#elif defined(FUSE_OP_REDUCE_SCATTER)
    OpSignaler op_signaler = OpSignaler(rt_args_idx);
#endif

#ifdef IN1_DRAM_WIDTH_SHARDED
    const uint32_t vc = get_arg(args::vc);
    const uint32_t num_dram_shards_to_read = get_arg(args::num_dram_shards_to_read);
    const uint32_t dram_tensor_start_offset = get_arg(args::dram_tensor_start_offset);

    constexpr auto in1_dram_block_num_tiles = get_arg(args::in1_dram_block_num_tiles);
    constexpr auto in1_block_w_dram_bytes = get_arg(args::in1_block_w_dram_bytes);
#endif

#ifdef IN1_DRAM_HEIGHT_SHARDED
    constexpr auto in1_KtNt_per_batch = get_arg(args::in1_KtNt_per_batch);
    constexpr auto in1_batches_per_bank = get_arg(args::in1_batches_per_bank);
#endif

    constexpr uint32_t in1_single_tile_size_bytes = get_tile_size(dfb::in1);
    constexpr uint32_t in1_aligned_tile_size_bytes =
        (in1_single_tile_size_bytes + (DRAM_ALIGNMENT - 1)) & ~(DRAM_ALIGNMENT - 1);
#if !defined(IN1_SHARDED) && !defined(IN1_DRAM_WIDTH_SHARDED) && !defined(IN1_DRAM_HEIGHT_SHARDED) && \
    !defined(ENABLE_GLOBAL_CB)
    constexpr uint32_t in1_block_size_bytes = in1_block_num_tiles * in1_aligned_tile_size_bytes;
#else
    constexpr uint32_t in1_block_size_bytes = in1_block_num_tiles * in1_single_tile_size_bytes;
#endif

#ifndef OUT_WRITER_ON_IN0
    constexpr uint32_t output_single_tile_size_bytes = get_tile_size(dfb::out);
#endif

#if defined(ENABLE_PREFETCHER_PIPE)
    experimental::PrefetcherPipe pipe(pipe::in1);
    auto in1_relay = pipe.bind_relay();
#elif defined(IN1_SHARDED)
    dfb_in1.reserve_back(in1_block_num_tiles * num_blocks_inner_dim);
    dfb_in1.push_back(in1_block_num_tiles * num_blocks_inner_dim);
#elif !defined(ENABLE_GLOBAL_CB)
    [[maybe_unused]] const auto s1 = TensorAccessor(tensor::in1);
#endif

#ifdef ENABLE_GLOBAL_CB
    constexpr uint32_t remote_cb_id = tt::CBIndex::c_31;
    const uint32_t in1_fifo_tiles = dfb_in1.get_total_num_entries();
#endif

    const auto s = TensorAccessor(tensor::out);
    (void)s;

#ifdef SPARSITY
    DataflowBuffer dfb_sparsity(dfb::sparsity);
    const auto s_sparsity = TensorAccessor(tensor::sparsity);
#endif

#ifdef IN1_SHARDED
    uint64_t in1_start_address = dfb_in1.get_write_ptr();
#endif

#ifdef SPARSITY
    uint32_t l1_write_addr_sparsity = 0;
    if constexpr (batchB > 0) {
        dfb_sparsity.reserve_back(1);
        l1_write_addr_sparsity = dfb_sparsity.get_write_ptr();
    }

    if constexpr (use_indices) {
        noc.async_read(s_sparsity, dfb_sparsity, sparsity_pagesize, {.page_id = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
    }
#endif

#ifdef IN1_DRAM_WIDTH_SHARDED
    constexpr uint32_t in1_dram_block_size_bytes = in1_dram_block_num_tiles * in1_single_tile_size_bytes;
    uint32_t in1_block_w_bytes = in1_block_w * in1_single_tile_size_bytes;
#endif

#ifdef IN1_DRAM_HEIGHT_SHARDED
    constexpr uint32_t in1_batch_stride_bytes = in1_KtNt_per_batch * in1_single_tile_size_bytes;
#endif

    for (uint32_t b = 0; b < batch; ++b) {
        uint32_t in1_batch_tile_id = in1_tensor_start_tile_id;

#ifdef IN1_DRAM_HEIGHT_SHARDED
        uint32_t in1_dram_bank_id = b / in1_batches_per_bank;
        uint32_t in1_batch_in_shard = b % in1_batches_per_bank;
        const BankView dram_src(s1);
        uint32_t in1_dram_batch_offset = in1_batch_in_shard * in1_batch_stride_bytes;
#endif

#ifdef SPARSITY
        if constexpr (batchB > 0 && !use_indices) {
            noc.async_read(s_sparsity, dfb_sparsity, sparsity_pagesize, {.page_id = b}, {.offset_bytes = 0});
            noc.async_read_barrier();
        }
#endif

        [[maybe_unused]] const uint32_t out_base_tile_id = out_tensor_start_tile_id;

        for (uint32_t bB = 0; bB < batch_loop_lim; ++bB) {
#ifdef BIAS_PER_GROUP
            uint32_t in3_group_tile_offset = 0;
#endif
#ifdef SPARSITY
            if constexpr (use_indices) {
                const uint32_t group_id = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_addr_sparsity)[bB];
#ifdef BIAS_PER_GROUP
                in3_group_tile_offset = group_id * out_tensor_stride_h;
#endif
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
#endif

            const uint32_t in1_tensor_current_h_dim_block_tile_id = in1_batch_tile_id;
            uint32_t out_tensor_current_h_dim_block_tile_id = out_tensor_start_tile_id;
            for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh) {
                uint32_t in1_tensor_current_w_dim_block_tile_id = in1_tensor_current_h_dim_block_tile_id;
                uint32_t out_tensor_current_w_dim_block_tile_id = out_tensor_current_h_dim_block_tile_id;
#ifdef FUSE_BIAS
#ifdef BIAS_PER_GROUP
                uint32_t in3_tensor_current_w_dim_block_tile_id = in3_tensor_start_tile_id + in3_group_tile_offset;
#else
                uint32_t in3_tensor_current_w_dim_block_tile_id = in3_tensor_start_tile_id;
#endif
#endif
                for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw) {
                    uint32_t in1_tensor_current_inner_dim_block_start_tile_id = in1_tensor_current_w_dim_block_tile_id;
#ifdef IN1_DRAM_WIDTH_SHARDED
                    uint32_t l1_read_addr_in1_offset = 0;
#endif

                    for (uint32_t block = 0; block < num_blocks_inner_dim; ++block) {
#ifndef IN1_SHARDED
                        uint64_t in1_start_address = 0;
#endif
#ifdef FUSE_OP_ALL_GATHER
                        fused_op_receiver.update_current_block_start_tile_id(
                            block, in1_tensor_current_inner_dim_block_start_tile_id, in1_batch_tile_id);
#endif
#if defined(ENABLE_PREFETCHER_PIPE)
                        publish_in1_block_from_pipe(pipe, in1_relay, noc, block, in1_block_num_tiles);
#elif defined(ENABLE_GLOBAL_CB)
                        dfb_in1.reserve_back(in1_block_num_tiles);
                        experimental::remote_cb_wait_front(remote_cb_id, block == 0 ? 1u : 2u);
#elif defined(IN1_DRAM_WIDTH_SHARDED) && defined(IN1_DRAM_SHARDED_PER_N_BLOCK)
                        dfb_in1.reserve_back(in1_block_num_tiles);
                        in1_start_address = dfb_in1.get_write_ptr();
                        {
                            const BankView n_block_bank(s1);
                            const uint32_t block_col_begin = bw * in1_block_w;
                            const uint32_t block_col_end = block_col_begin + in1_block_w;
                            uint32_t shard_col_begin = 0;
                            uint32_t vararg_index = 0;
                            for (uint32_t i = 0; i < num_dram_shards_to_read; ++i) {
                                const uint32_t n_block_bank_id = get_vararg(vararg_index + 1);
                                const uint32_t shard_cols = get_vararg(vararg_index) / in1_single_tile_size_bytes;
                                const uint32_t read_col_begin =
                                    shard_col_begin > block_col_begin ? shard_col_begin : block_col_begin;
                                const uint32_t read_col_end = (shard_col_begin + shard_cols) < block_col_end
                                                                  ? (shard_col_begin + shard_cols)
                                                                  : block_col_end;
                                if (read_col_begin < read_col_end) {
                                    const uint32_t shard_offset =
                                        (i == 0 ? dram_tensor_start_offset : 0) +
                                        (read_col_begin - shard_col_begin) * in1_single_tile_size_bytes;
                                    uint32_t read_offset = l1_read_addr_in1_offset;
                                    uint32_t write_addr =
                                        static_cast<uint32_t>(in1_start_address) +
                                        (read_col_begin - block_col_begin) * in1_single_tile_size_bytes;
                                    for (uint32_t m = 0; m < in1_block_h; ++m) {
                                        noc.async_read(
                                            n_block_bank,
                                            CoreLocalMem<uint32_t>(write_addr),
                                            (read_col_end - read_col_begin) * in1_single_tile_size_bytes,
                                            {.bank_id = n_block_bank_id, .offset_bytes = shard_offset + read_offset},
                                            {});
                                        read_offset += in1_block_w_dram_bytes;
                                        write_addr += in1_block_w_bytes;
                                    }
                                }
                                shard_col_begin += shard_cols;
                                vararg_index += 2;
                            }
                        }
                        l1_read_addr_in1_offset += in1_dram_block_size_bytes;
                        noc.async_read_barrier();
#elif defined(IN1_DRAM_WIDTH_SHARDED)
                        dfb_in1.reserve_back(in1_block_num_tiles);

                        in1_start_address = dfb_in1.get_write_ptr();

                        uint32_t l1_write_addr_in1_offset = 0;
                        uint32_t next_bank_id_and_dram_stride_index = 0;

                        const BankView dram_bank(s1);
                        for (uint32_t i = 0; i < num_dram_shards_to_read; ++i) {
                            uint32_t shard_bank_id = get_vararg(next_bank_id_and_dram_stride_index + 1);
                            uint32_t shard_base_offset = 0;
                            if (i == 0) {
                                shard_base_offset += dram_tensor_start_offset;
                            }
                            noc.set_async_read_state<NocOptions::CUSTOM_VC, NOC_MAX_BURST_SIZE>(
                                dram_bank,
                                in1_single_tile_size_bytes,
                                {.bank_id = shard_bank_id, .offset_bytes = shard_base_offset},
                                NocOptVals{.vc = vc});

                            uint32_t l1_read_addr_in1 = l1_read_addr_in1_offset;
                            uint32_t l1_write_addr_in1 = dfb_in1.get_write_ptr() + l1_write_addr_in1_offset;
                            uint32_t in1_block_w_dram =
                                get_vararg(next_bank_id_and_dram_stride_index) / in1_single_tile_size_bytes;

                            for (uint32_t m = 0; m < in1_block_h; ++m) {
                                uint32_t l1_read_addr_in1_temp = l1_read_addr_in1;
                                uint32_t l1_write_addr_in1_temp = l1_write_addr_in1;
#ifdef IN1_COALESCED_READ
                                noc.async_read(
                                    dram_bank,
                                    CoreLocalMem<uint32_t>(l1_write_addr_in1_temp),
                                    in1_block_w_dram * in1_single_tile_size_bytes,
                                    {.bank_id = shard_bank_id,
                                     .offset_bytes = shard_base_offset + l1_read_addr_in1_temp},
                                    {});
#else
                                for (uint32_t w = 0; w < in1_block_w_dram; ++w) {
                                    noc.async_read_with_state<NocOptions::CUSTOM_VC, NOC_MAX_BURST_SIZE>(
                                        dram_bank,
                                        CoreLocalMem<uint32_t>(l1_write_addr_in1_temp),
                                        in1_single_tile_size_bytes,
                                        {.bank_id = shard_bank_id,
                                         .offset_bytes = shard_base_offset + l1_read_addr_in1_temp},
                                        {},
                                        NocOptVals{.vc = vc});
                                    l1_read_addr_in1_temp += in1_single_tile_size_bytes;
                                    l1_write_addr_in1_temp += in1_single_tile_size_bytes;
                                }
#endif
                                l1_read_addr_in1 += in1_block_w_dram_bytes;
                                l1_write_addr_in1 += in1_block_w_bytes;
                            }
                            l1_write_addr_in1_offset += get_vararg(next_bank_id_and_dram_stride_index);
                            next_bank_id_and_dram_stride_index += 2;
                        }
                        l1_read_addr_in1_offset += in1_dram_block_size_bytes;
                        noc.async_read_barrier();
#elif defined(IN1_DRAM_HEIGHT_SHARDED)
                        dfb_in1.reserve_back(in1_block_num_tiles);

                        uint32_t l1_write_addr_in1 = dfb_in1.get_write_ptr();
                        in1_start_address = l1_write_addr_in1;

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
                                        {.bank_id = in1_dram_bank_id, .offset_bytes = tile_byte_offset},
                                        {});
                                }
                                l1_write_addr_in1 += in1_single_tile_size_bytes;
                                in1_tensor_tile_id += in1_tensor_stride_w;
                            }
                            in1_tensor_row_start_tile_id += in1_tensor_stride_h;
                        }
                        in1_tensor_current_inner_dim_block_start_tile_id += in1_tensor_next_block_stride;

                        noc.async_read_barrier();
#elif !defined(IN1_SHARDED)
                        dfb_in1.reserve_back(in1_block_num_tiles);
                        in1_start_address = dfb_in1.get_write_ptr();

#ifdef IN1_COALESCED_READ
                        uint32_t in1_tensor_row_start_tile_id = in1_tensor_current_inner_dim_block_start_tile_id;
                        uint32_t in1_write_offset = 0;
                        {
                            const uint32_t run_write_base = dfb_in1.get_write_ptr();
                            uint64_t run_src = 0;
                            uint32_t run_dst = 0, run_len = 0;
                            for (uint32_t h = 0; h < in1_block_h; ++h) {
                                uint32_t in1_tensor_tile_id = in1_tensor_row_start_tile_id;
                                for (uint32_t w = 0; w < in1_block_w; ++w) {
                                    if (bw < num_blocks_w_dim - 1 || w < last_block_w) {
                                        const uint64_t a = s1.get_noc_addr(in1_tensor_tile_id);
                                        const uint32_t d = run_write_base + in1_write_offset;
                                        if (run_len != 0 && a == run_src + run_len && d == run_dst + run_len) {
                                            run_len += in1_single_tile_size_bytes;
                                        } else {
                                            if (run_len != 0) {
                                                noc_async_read(run_src, run_dst, run_len);
                                            }
                                            run_src = a;
                                            run_dst = d;
                                            run_len = in1_single_tile_size_bytes;
                                        }
                                    } else if (run_len != 0) {
                                        noc_async_read(run_src, run_dst, run_len);
                                        run_len = 0;
                                    }
                                    in1_write_offset += in1_aligned_tile_size_bytes;
                                    in1_tensor_tile_id += in1_tensor_stride_w;
                                }
                                in1_tensor_row_start_tile_id += in1_tensor_stride_h;
                            }
                            if (run_len != 0) {
                                noc_async_read(run_src, run_dst, run_len);
                            }
                            noc_async_read_barrier();
                        }
#else
                        read_in1_block<
                            in1_block_h,
                            in1_block_w,
                            in1_tensor_stride_w,
                            in1_tensor_stride_h,
                            in1_single_tile_size_bytes,
                            in1_aligned_tile_size_bytes>(
                            noc,
                            s1,
                            static_cast<uint32_t>(in1_start_address),
                            in1_tensor_current_inner_dim_block_start_tile_id,
                            bw < num_blocks_w_dim - 1 ? in1_block_w : last_block_w);
#endif
                        in1_tensor_current_inner_dim_block_start_tile_id += in1_tensor_next_block_stride;

                        noc.async_read_barrier();
#endif

                        if constexpr (in1_mcast.active) {
                            sender->send(
                                static_cast<uint32_t>(in1_start_address),
                                static_cast<uint32_t>(in1_start_address),
                                in1_block_size_bytes);
                        }

#if !defined(ENABLE_PREFETCHER_PIPE) && !defined(IN1_SHARDED)
                        dfb_in1.push_back(in1_block_num_tiles);
#endif
#ifdef ENABLE_GLOBAL_CB
                        if (block >= 1) {
                            while (!dfb_in1.pages_reservable_at_back(in1_fifo_tiles - in1_block_num_tiles)) {
                                invalidate_l1_cache();
                            }
                            experimental::remote_cb_pop_front(remote_cb_id, 1);
                        }
#endif
                    }
#if defined(ENABLE_PREFETCHER_PIPE)
                    drain_in1_pipe_window(pipe, noc, num_blocks_inner_dim);
#elif defined(ENABLE_GLOBAL_CB)
                    if (num_blocks_inner_dim > 0) {
                        while (!dfb_in1.pages_reservable_at_back(in1_fifo_tiles)) {
                            invalidate_l1_cache();
                        }
                        experimental::remote_cb_pop_front(remote_cb_id, 1);
                    }
#endif
#ifdef FUSE_BIAS
                    if (bias_per_group || (b == 0 && bh == 0) || num_blocks_w_dim > 1) {
#ifndef BIAS_SHARDED
                        dfb_in3.reserve_back(in1_block_w);
                        uint32_t in3_write_offset = 0;

                        const uint64_t in3_start_address = dfb_in3.get_write_ptr();
                        uint32_t in3_block_size_bytes = 0;

#ifdef IN1_DRAM_WIDTH_SHARDED
                        uint32_t l1_write_addr_in3_offset = 0;
                        uint32_t next_bank_id_and_dram_stride_index = 0;

                        const BankView bias_dram_bank(s3);
                        for (uint32_t i = 0; i < num_dram_shards_to_read; ++i) {
                            uint32_t bias_shard_bank_id = get_vararg(next_bank_id_and_dram_stride_index + 1);
                            uint32_t bias_shard_base_offset = 0;
                            if (i == 0) {
                                bias_shard_base_offset += (dram_tensor_start_offset / in1_single_tile_size_bytes) *
                                                          bias_single_tile_size_bytes;
                            }

                            noc.set_async_read_state<NocOptions::CUSTOM_VC, NOC_MAX_BURST_SIZE>(
                                bias_dram_bank,
                                bias_single_tile_size_bytes,
                                {.bank_id = bias_shard_bank_id, .offset_bytes = bias_shard_base_offset},
                                NocOptVals{.vc = vc});

                            uint32_t l1_read_addr_in3 = 0;
                            uint32_t l1_write_addr_in3 = dfb_in3.get_write_ptr() + l1_write_addr_in3_offset;
                            uint32_t in3_block_w_dram =
                                get_vararg(next_bank_id_and_dram_stride_index) / in1_single_tile_size_bytes;

                            for (uint32_t w = 0; w < in3_block_w_dram; ++w) {
                                noc.async_read_with_state<NocOptions::CUSTOM_VC, NOC_MAX_BURST_SIZE>(
                                    bias_dram_bank,
                                    CoreLocalMem<uint32_t>(l1_write_addr_in3),
                                    bias_single_tile_size_bytes,
                                    {.bank_id = bias_shard_bank_id,
                                     .offset_bytes = bias_shard_base_offset + l1_read_addr_in3},
                                    {},
                                    NocOptVals{.vc = vc});
                                l1_read_addr_in3 += bias_single_tile_size_bytes;
                                l1_write_addr_in3 += bias_single_tile_size_bytes;
                                in3_block_size_bytes += bias_single_tile_size_bytes;
                            }
                            l1_write_addr_in3_offset += in3_block_w_dram * bias_single_tile_size_bytes;
                            next_bank_id_and_dram_stride_index += 2;
                        }
                        noc.async_read_barrier();
#else
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
                        noc.async_read_barrier();
#endif

                        if constexpr (in1_mcast.active) {
                            sender->send(
                                static_cast<uint32_t>(in3_start_address),
                                static_cast<uint32_t>(in3_start_address),
                                in3_block_size_bytes);
                        }

                        dfb_in3.push_back(in1_block_w);
#else
                        dfb_in3.reserve_back(in1_block_w);
                        dfb_in3.push_back(in1_block_w);
#endif
                    }
#endif

#if defined(OUT_WRITER_ON_IN0)
                    (void)out_num_nonzero_subblocks_h;
                    (void)out_last_subblock_h;
                    (void)padded_block_tiles_h_skip;
                    (void)out_num_nonzero_subblocks_w;
                    (void)out_last_num_nonzero_subblocks_w;
                    (void)out_last_subblock_w;
                    (void)padded_subblock_tiles_addr_skip;
                    (void)padded_block_tiles_w_skip;
                    (void)last_num_blocks_w_dim;
#elif !defined(OUT_SHARDED)
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
                                out_read_offset += subblock_tiles_addr_skip;
                                out_tensor_sb_row_start_tile_id += out_tensor_stride_h;
                            }

#ifdef OUT_FLUSH_PER_BLOCK
                            noc.async_writes_flushed();
#else
                            noc.async_write_barrier();
#endif
                            dfb_out.pop_front(out_subblock_tile_count);
                            out_tensor_sbw_start_tile_id += out_tensor_next_subblock_stride_w;
                        }
                        if (bw == num_blocks_w_dim_ - 1) {
                            dfb_out.wait_front(static_cast<uint16_t>(padded_block_tiles_w_skip));
                            dfb_out.pop_front(static_cast<uint16_t>(padded_block_tiles_w_skip));
                        }
                        out_tensor_sbh_start_tile_id += out_tensor_next_subblock_stride_h;
                    }
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
            in1_tensor_start_tile_id += KtNt;
#endif
        }

#ifdef FUSE_OP_REDUCE_SCATTER
        op_signaler.synchronize_workers_and_signal_op(0);
#endif
    }

#ifdef OUT_SHARDED
    constexpr auto out_block_num_tiles = get_arg(args::out_block_num_tiles);
    const uint16_t out_num_tiles =
        static_cast<uint16_t>(batch * num_blocks_h_dim * num_blocks_w_dim * out_block_num_tiles);
    dfb_out.wait_front(out_num_tiles);
    dfb_out.pop_front(out_num_tiles);
#endif
#ifdef ENABLE_GLOBAL_CB
    experimental::update_remote_cb_config_in_l1(remote_cb_id);
#endif
    noc.async_atomic_barrier();
    noc.async_write_barrier();

#ifdef IN1_DRAM_WIDTH_SHARDED
    const uint32_t noc_rd_cmd_field =
        NOC_CMD_CPY | NOC_CMD_RD | NOC_CMD_RESP_MARKED | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(1);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_CTRL, noc_rd_cmd_field);
#endif

#ifdef SPARSITY
    if constexpr (batchB > 0) {
        dfb_sparsity.push_back(1);
    }
#endif
}
