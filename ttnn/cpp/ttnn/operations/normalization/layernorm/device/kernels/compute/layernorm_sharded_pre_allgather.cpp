// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#ifdef DO_COL_MASK
#include "ttnn/operations/normalization/kernel_util/compute/col_mask.h"
#endif

// SPLIT REDUCE across Cores
void kernel_main() {
    constexpr auto num_blocks_first_stage = get_arg(args::num_blocks_first_stage);
    constexpr auto block_w = get_arg(args::block_w);
    constexpr auto block_h_const = get_arg(args::block_h);
    volatile uint32_t block_h_volatile = get_arg(args::block_h);
    constexpr auto subblock_w_const = get_arg(args::subblock_w);
    volatile uint32_t subblock_w_volatile = get_arg(args::subblock_w);
    constexpr auto num_subblocks_w = get_arg(args::num_subblocks_w);
    constexpr auto num_tiles_per_block = get_arg(args::num_tiles_per_block);
    constexpr bool FLOAT32_DTYPE = get_arg(args::float32_dtype) == 1;
    constexpr auto num_blocks_second_stage = get_arg(args::num_blocks_second_stage);

    const uint32_t num_reduce_tiles_per_block_h = get_arg(
        args::num_reduce_tiles_per_block_h);  // This value is the same for all cores, except ones that have
                                              // padding tiles in it. In that case, skip reduce for padding tiles.
    // Only the cores that gather run the cross-core combine. They alone read its runtime arguments and
    // write its two possible destinations, so the distinction is a compile-time one and everything the
    // combine needs lives inside it.
#ifdef IS_ALLGATHER_WORKER
    const uint32_t num_tiles_per_allgather_worker = get_arg(args::num_rows_per_all_to_all_worker);
    const bool use_two_stage_reduce = get_arg(args::use_two_stage_reduce) == 1;
    const bool is_second_stage_reader = get_arg(args::is_second_stage_reader) == 1;

    uint32_t num_blocks_reduce;
    if (is_second_stage_reader) {
        num_blocks_reduce = num_blocks_first_stage + num_blocks_second_stage - 1;
    } else {
        num_blocks_reduce = num_blocks_first_stage;
    }
#endif

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;
    constexpr uint32_t scaler0 = 0;

    constexpr uint32_t dfb_in0 = dfb::in0;
#ifdef FUSE_PRE_ADD
    constexpr uint32_t dfb_in1 = dfb::in1;
    // The pre-add result is written back over the input tensor's own shard.
    constexpr uint32_t dfb_in_id = dfb::in_pre_add;
#else
    constexpr uint32_t dfb_in_id = dfb_in0;
#endif
    DataflowBuffer dfb_in(dfb_in_id);
    constexpr uint32_t dfb_scaler_id = dfb::scaler;
    constexpr uint32_t dfb_scaler_global_id = dfb::scaler_global;
    constexpr uint32_t dfb_x = dfb::x;     // x minus mean
    constexpr uint32_t dfb_x2_id = dfb_x;  // x^2

    constexpr uint32_t dfb_ex_partial2_id = dfb::ex_partial2;    // E[x^2] partial reduce
    constexpr uint32_t dfb_ex_external2_id = dfb::ex_external2;  // E[x^2] partials received from other cores
#ifdef IS_ALLGATHER_WORKER
    // Where the combine writes its result: the output when this core produces the final statistics,
    // the Var[x] buffer when a second stage still has to reduce them.
    constexpr uint32_t dfb_ex2 = dfb::ex2;
    constexpr uint32_t dfb_out = dfb::out;
    const uint32_t dfb_reduction_out = (!use_two_stage_reduce or is_second_stage_reader) ? dfb_out : dfb_ex2;
#endif
#ifdef DO_COL_MASK
    // Writer-generated column mask (1.0 valid / 0.0 padding)
    constexpr uint32_t dfb_col_mask_packed_id = dfb::col_mask;
    DataflowBuffer dfb_col_mask_packed(dfb_col_mask_packed_id);
#endif

    DataflowBuffer dfb_scaler(dfb_scaler_id);
    DataflowBuffer dfb_x2(dfb_x2_id);
    DataflowBuffer dfb_ex_partial2(dfb_ex_partial2_id);
    DataflowBuffer dfb_scaler_global(dfb_scaler_global_id);
    DataflowBuffer dfb_ex_external2(dfb_ex_external2_id);

    // set block_h to volatile to disable automatically unroll of the loops, avoid code overflow
    const uint32_t block_h = (block_w == 1) ? block_h_volatile : block_h_const;
    const uint32_t subblock_w = (block_w <= 2) ? subblock_w_volatile : subblock_w_const;

    int index_subblock_w_offset = 0;
    int index_h_offset = 0;
    int index = 0;

    uint32_t num_tiles_per_partial_result = 2;
#ifdef RMSNORM
    num_tiles_per_partial_result = 1;
#endif

// pre-add x + y
#ifdef FUSE_PRE_ADD
    compute_kernel_hw_startup(dfb_in0, dfb_in1, dfb_in_id);
    add_init(dfb_in0, dfb_in1);
    dfb_in.reserve_back(num_tiles_per_block);
    for (uint32_t i = 0; i < block_h; i++) {
        index_subblock_w_offset = 0;
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset + index_h_offset;
                add_tiles(dfb_in0, dfb_in1, index, index, w);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < subblock_w; i++) {
                pack_tile(i, dfb_in_id);
            }
            tile_regs_release();
            index_subblock_w_offset += subblock_w;
        }
        index_h_offset += block_w;
    }
    dfb_in.push_back(num_tiles_per_block);
    dfb_in.wait_front(num_tiles_per_block);
    pack_reconfig_data_format(dfb_in_id, dfb_x2_id);
#else
    // TODO(#52395): compute_kernel_hw_startup is a call-once API; this mid-kernel re-init (preserving the pre-cleanup full-init behaviour) should become a targeted DST re-arm.
    compute_kernel_hw_startup(dfb_in_id, dfb_in_id, dfb_x2_id);
#endif

#ifdef DO_COL_MASK
    // The column mask has block_w tiles, one per tile across the shard width.
    // Wait once for it here; the masking sites below read it by tile index without
    // re-waiting (it is reused across all rows and masking sites).
    // It is popped once at the end of the kernel so the buffer is left balanced.
    dfb_col_mask_packed.wait_front(block_w);
#endif

#ifndef RMSNORM
    dfb_scaler.wait_front(1);
#ifdef FUSE_PRE_ADD
    reconfig_data_format(dfb_in0, dfb_in_id, dfb_in1, dfb_scaler_id);
#else
    reconfig_data_format_srcb(dfb_in_id, dfb_scaler_id);
#endif
#ifdef DO_COL_MASK
    // Non-tile-aligned width: the E[x] reduce must average over the logical width, so mask any
    // padding columns out of the input first. The masked copy goes to the x^2 scratch,
    // not back into the input, because the X^2 pass below re-reads the input (which is also a
    // buffer-backed input that must not be mutated). The X^2 pass masks its own result separately, on
    // the squares (the DO_COL_MASK block after the X^2 loop), so both statistics end up reduced over
    // the logical width only. The column mask is the writer-generated mask (1.0 valid / 0.0 padding),
    // waited on above and read by tile index.
    reconfig_data_format(dfb_in_id, dfb_col_mask_packed_id);
    mul_init(dfb_in_id, dfb_col_mask_packed_id);
    dfb_x2.reserve_back(num_tiles_per_block);
    index_h_offset = 0;
    for (uint32_t i = 0; i < block_h; i++) {
        for (uint32_t wt = 0; wt < block_w; wt++) {
            tile_regs_acquire();
            mul_tiles(dfb_in_id, dfb_col_mask_packed_id, wt + index_h_offset, wt, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dfb_x2_id);
            tile_regs_release();
        }
        index_h_offset += block_w;
    }
    dfb_x2.push_back(num_tiles_per_block);
    dfb_x2.wait_front(num_tiles_per_block);
    // E[x] over the masked input.
    compute_kernel_lib::reduce<
        PoolType::AVG,
        ReduceDim::REDUCE_ROW,
        dfb_x2_id,
        dfb_scaler_id,
        dfb_ex_partial2_id,
        compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT>(
        compute_kernel_lib::ReduceInputBlockShape::of(block_h, num_reduce_tiles_per_block_h),
        compute_kernel_lib::ReduceInputMemoryLayout::with_row_stride(block_w));
    dfb_x2.pop_front(num_tiles_per_block);
    reconfig_data_format(dfb_in_id, dfb_in_id);
#else
    // E[x],
    compute_kernel_lib::reduce<
        PoolType::AVG,
        ReduceDim::REDUCE_ROW,
        dfb_in_id,
        dfb_scaler_id,
        dfb_ex_partial2_id,
        compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT>(
        compute_kernel_lib::ReduceInputBlockShape::of(block_h, num_reduce_tiles_per_block_h),
        compute_kernel_lib::ReduceInputMemoryLayout::with_row_stride(block_w));
    reconfig_data_format(dfb_in_id, dfb_in_id);
#endif  // DO_COL_MASK
#else
#ifdef FUSE_PRE_ADD
    reconfig_data_format(dfb_in0, dfb_in_id, dfb_in1, dfb_in_id);
#endif
#endif  // not RMSNORM

    // X^2
    mul_init(dfb_in0, dfb_in0);
    index_h_offset = 0;
    dfb_x2.reserve_back(num_tiles_per_block);
    for (uint32_t i = 0; i < block_h; i++) {
        index_subblock_w_offset = 0;
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset + index_h_offset;
                mul_tiles(dfb_in_id, dfb_in_id, index, index, w);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < subblock_w; i++) {
                pack_tile(i, dfb_x2_id);
            }
            tile_regs_release();
            index_subblock_w_offset += subblock_w;
        }
        index_h_offset += block_w;
    }
    dfb_x2.push_back(num_tiles_per_block);

#ifdef FUSE_PRE_ADD
    // The fused-add result (a + b) lives in a kernel-local scratch buffer that was reserved, pushed,
    // waited on, and read by tile index through the E[x] and X^2 passes above. The X^2 loop is its last
    // read, so pop it here to leave the buffer balanced. On the non-fused path the intake name aliases
    // the buffer-backed input, which is read by index and never waited or popped.
    dfb_in.pop_front(num_tiles_per_block);
#endif

#ifdef DO_COL_MASK
    // The mean-of-squares reduce (RMSNorm's statistic, and LayerNorm's E[x^2]) squares the raw input,
    // which leaves the padding columns holding (pad_value)^2; zero them in place before the reduce so
    // they do not enter the mean of squares. The writer-generated mask carries
    // each block's own validity (full, partial, or all-padding tiles). It was waited on near the
    // top of the kernel and is read by tile index here (never popped).
    reconfig_data_format(dfb_x2_id, dfb_col_mask_packed_id);
    norm::kernel_util::compute::mask_block_in_place(dfb_x2, dfb_col_mask_packed_id, num_tiles_per_block, block_w);
#endif

    // E(x^2)
    dfb_x2.wait_front(num_tiles_per_block);
#ifdef RMSNORM
    dfb_scaler.wait_front(1);
#endif  // RMSNORM

    // RMS E(x2) #Layernorm //E(x) and E(x^2)
    compute_kernel_lib::reduce<
        PoolType::AVG,
        ReduceDim::REDUCE_ROW,
        dfb_x2_id,
        dfb_scaler_id,
        dfb_ex_partial2_id,
        compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop>(
        compute_kernel_lib::ReduceInputBlockShape::of(block_h, num_reduce_tiles_per_block_h),
        compute_kernel_lib::ReduceInputMemoryLayout::with_row_stride(block_w));
    reconfig_data_format(dfb_x2_id, dfb_scaler_id);
    dfb_x2.pop_front(num_tiles_per_block);

    // global reduce, the combine destination <-- dfb_ex_external2_id, dfb_ex_partial2_id
#ifdef IS_ALLGATHER_WORKER
    {
        dfb_scaler_global.wait_front(1);
        reconfig_data_format(dfb_scaler_global_id, dfb_ex_external2_id);
        pack_reconfig_data_format(dfb_reduction_out);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(dfb_ex_external2_id, dfb_scaler_global_id, dfb_reduction_out);
        DataflowBuffer(dfb_reduction_out).reserve_back(num_tiles_per_partial_result * num_tiles_per_allgather_worker);

        for (uint32_t i = 0; i < num_tiles_per_allgather_worker; i++) {  // loops over height
            tile_regs_acquire();
            for (uint32_t w = 0; w < num_tiles_per_partial_result * num_blocks_reduce;
                 w++) {  // Need to read this interleaved now, we have SUM(X) and SUM(X^2) interleaved
                dfb_ex_external2.wait_front(1);
                reduce_tile<PoolType::AVG, ReduceDim::REDUCE_ROW>(
                    dfb_ex_external2_id,
                    dfb_scaler_global_id,
                    0,
                    scaler0,
                    w % num_tiles_per_partial_result);  // E(x) and E(x^2) interleaved so we reduce each one into
                                                        // different dest reg
                dfb_ex_external2.pop_front(1);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, dfb_reduction_out);
#ifndef RMSNORM
            pack_tile(dst1, dfb_reduction_out);
#endif
            tile_regs_release();
        }
        reduce_uninit();
        DataflowBuffer(dfb_reduction_out).push_back(num_tiles_per_partial_result * num_tiles_per_allgather_worker);
        // The global-reduce scaler tile is pushed once (only on all-gather worker cores) and read by
        // tile index throughout the global reduce above without being popped. Pop it once here, inside
        // the same guard that gated the wait, so the buffer is left balanced on every core.
        dfb_scaler_global.pop_front(1);
    }
#endif
    // The single scaler tile is waited once (by the E[x] reduce on the LayerNorm path or the E[x^2]
    // reduce on the RMSNorm path) but never popped; pop it once here so the buffer is left balanced.
    dfb_scaler.pop_front(1);
#ifdef DO_COL_MASK
    // The column mask is waited once near the top of the kernel (on every core) and read by tile index
    // at every masking site; pop its block_w tiles once here so the buffer is left balanced.
    dfb_col_mask_packed.pop_front(block_w);
#endif
}
