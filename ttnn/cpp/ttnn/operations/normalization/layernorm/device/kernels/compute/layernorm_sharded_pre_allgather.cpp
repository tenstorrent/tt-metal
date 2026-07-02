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
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#ifdef DO_COL_MASK
#include "ttnn/operations/normalization/kernel_util/compute/col_mask.h"
#endif

// SPLIT REDUCE across Cores
void kernel_main() {
    constexpr uint32_t num_blocks_first_stage = get_compile_time_arg_val(3);
    constexpr uint32_t block_w = get_compile_time_arg_val(5);
    constexpr uint32_t block_h_const = get_compile_time_arg_val(4);
    volatile uint32_t block_h_volatile = get_compile_time_arg_val(4);
    constexpr uint32_t subblock_w_const = get_compile_time_arg_val(6);
    volatile uint32_t subblock_w_volatile = get_compile_time_arg_val(6);
    constexpr uint32_t num_subblocks_w = get_compile_time_arg_val(7);
    const bool is_allgather_worker = get_compile_time_arg_val(8) == 1;
    constexpr uint32_t num_tiles_per_block = get_compile_time_arg_val(9);
    constexpr bool FLOAT32_DTYPE = get_compile_time_arg_val(10) == 1;
    constexpr bool FLOAT32_REDUCTION = get_compile_time_arg_val(11) == 1;
    // LEGACY_RSQRT at index 12 is not used but needed for consistency across sharded compute kernels
    constexpr uint32_t num_blocks_second_stage = get_compile_time_arg_val(13);

    const uint32_t num_reduce_tiles_per_block_h =
        get_arg_val<uint32_t>(0);  // This value is the same for all cores, except ones that have padding tiles in it.
                                   // In that case, skip reduce for padding tiles.
    const uint32_t num_tiles_per_allgather_worker = is_allgather_worker ? get_arg_val<uint32_t>(1) : 0;
    const bool use_two_stage_reduce = is_allgather_worker ? get_arg_val<uint32_t>(2) == 1 : false;
    const bool is_second_stage_reader = is_allgather_worker ? get_arg_val<uint32_t>(3) == 1 : false;

    uint32_t num_blocks_reduce;
    if (is_second_stage_reader) {
        num_blocks_reduce = num_blocks_first_stage + num_blocks_second_stage - 1;
    } else {
        num_blocks_reduce = num_blocks_first_stage;
    }

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;
    constexpr uint32_t scaler0 = 0;

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
#ifdef FUSE_PRE_ADD
    constexpr uint32_t cb_in_id = tt::CBIndex::c_14;
#else
    constexpr uint32_t cb_in_id = cb_in0;
#endif
    DataflowBuffer cb_in(cb_in_id);
    constexpr uint32_t cb_scaler_id = tt::CBIndex::c_2;
    constexpr uint32_t cb_scaler_global_id = tt::CBIndex::c_4;
    constexpr uint32_t cb_x = tt::CBIndex::c_24;  // x minus mean
    constexpr uint32_t cb_ex = tt::CBIndex::c_9;  // E[x] global reduce

    constexpr uint32_t cb_ex2 = tt::CBIndex::c_12;  // E[(x-E[x])^2] global reduce
    constexpr uint32_t cb_x2_id = cb_x;             // x^2
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t cb_ex_partial2_id = tt::CBIndex::c_11;   // E[x^2] partial reduce
    constexpr uint32_t cb_ex_external2_id = tt::CBIndex::c_13;  // E[x^2] partials received from other cores
    const uint32_t cb_reduction_out = (!use_two_stage_reduce or is_second_stage_reader) ? cb_out : cb_ex2;
#ifdef DO_COL_MASK
    // Writer-generated column mask (1.0 valid / 0.0 padding)
    constexpr uint32_t cb_col_mask_packed_id = tt::CBIndex::c_19;
    DataflowBuffer cb_col_mask_packed(cb_col_mask_packed_id);
#endif

    DataflowBuffer cb_scaler(cb_scaler_id);
    DataflowBuffer cb_x2(cb_x2_id);
    DataflowBuffer cb_ex_partial2(cb_ex_partial2_id);
    DataflowBuffer cb_scaler_global(cb_scaler_global_id);
    DataflowBuffer cb_ex_external2(cb_ex_external2_id);

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
    binary_op_init_common(cb_in0, cb_in1, cb_in_id);
    add_tiles_init(cb_in0, cb_in1);
    cb_in.reserve_back(num_tiles_per_block);
    for (uint32_t i = 0; i < block_h; i++) {
        index_subblock_w_offset = 0;
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset + index_h_offset;
                add_tiles(cb_in0, cb_in1, index, index, w);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < subblock_w; i++) {
                pack_tile(i, cb_in_id);
            }
            tile_regs_release();
            index_subblock_w_offset += subblock_w;
        }
        index_h_offset += block_w;
    }
    cb_in.push_back(num_tiles_per_block);
    cb_in.wait_front(num_tiles_per_block);
    pack_reconfig_data_format(cb_in_id, cb_x2_id);
#else
    binary_op_init_common(cb_in_id, cb_in_id, cb_x2_id);
#endif

#ifdef DO_COL_MASK
    // The column mask has block_w tiles, one per tile across the shard width.
    // Wait once for it here; the masking sites below read it by tile index without
    // re-waiting (it is reused across all rows and masking sites).
    // It is popped once at the end of the kernel so the CB is left balanced.
    cb_col_mask_packed.wait_front(block_w);
#endif

#ifndef RMSNORM
    cb_scaler.wait_front(1);
#ifdef FUSE_PRE_ADD
    reconfig_data_format(cb_in0, cb_in_id, cb_in1, cb_scaler_id);
#else
    reconfig_data_format_srcb(cb_in_id, cb_scaler_id);
#endif
#ifdef DO_COL_MASK
    // Non-tile-aligned width: the E[x] reduce must average over the logical width, so mask any
    // padding columns out of the input first. The masked copy goes to a scratch (cb_x2),
    // not back into cb_in, because the X^2 pass below re-reads cb_in (which is also a buffer-backed
    // input CB that must not be mutated). The X^2 pass masks its own result separately, on the squares
    // (the DO_COL_MASK block after the X^2 loop), so both statistics end up reduced over the logical
    // width only. cb_col_mask_packed is the writer-generated mask (1.0 valid / 0.0 padding),
    // waited on above and read by tile index.
    reconfig_data_format(cb_in_id, cb_col_mask_packed_id);
    mul_tiles_init(cb_in_id, cb_col_mask_packed_id);
    cb_x2.reserve_back(num_tiles_per_block);
    index_h_offset = 0;
    for (uint32_t i = 0; i < block_h; i++) {
        for (uint32_t wt = 0; wt < block_w; wt++) {
            tile_regs_acquire();
            mul_tiles(cb_in_id, cb_col_mask_packed_id, wt + index_h_offset, wt, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_x2_id);
            tile_regs_release();
        }
        index_h_offset += block_w;
    }
    cb_x2.push_back(num_tiles_per_block);
    cb_x2.wait_front(num_tiles_per_block);
    // E[x] over the masked input.
    compute_kernel_lib::reduce<
        PoolType::AVG,
        ReduceDim::REDUCE_ROW,
        cb_x2_id,
        cb_scaler_id,
        cb_ex_partial2_id,
        compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT>(
        compute_kernel_lib::ReduceInputBlockShape::of(block_h, num_reduce_tiles_per_block_h),
        compute_kernel_lib::ReduceInputMemoryLayout::with_row_stride(block_w));
    cb_x2.pop_front(num_tiles_per_block);
    reconfig_data_format(cb_in_id, cb_in_id);
#else
    // E[x],
    compute_kernel_lib::reduce<
        PoolType::AVG,
        ReduceDim::REDUCE_ROW,
        cb_in_id,
        cb_scaler_id,
        cb_ex_partial2_id,
        compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT>(
        compute_kernel_lib::ReduceInputBlockShape::of(block_h, num_reduce_tiles_per_block_h),
        compute_kernel_lib::ReduceInputMemoryLayout::with_row_stride(block_w));
    reconfig_data_format(cb_in_id, cb_in_id);
#endif  // DO_COL_MASK
#else
#ifdef FUSE_PRE_ADD
    reconfig_data_format(cb_in0, cb_in_id, cb_in1, cb_in_id);
#endif
#endif  // not RMSNORM

    // X^2
    mul_tiles_init(cb_in0, cb_in0);
    index_h_offset = 0;
    cb_x2.reserve_back(num_tiles_per_block);
    for (uint32_t i = 0; i < block_h; i++) {
        index_subblock_w_offset = 0;
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset + index_h_offset;
                mul_tiles(cb_in_id, cb_in_id, index, index, w);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < subblock_w; i++) {
                pack_tile(i, cb_x2_id);
            }
            tile_regs_release();
            index_subblock_w_offset += subblock_w;
        }
        index_h_offset += block_w;
    }
    cb_x2.push_back(num_tiles_per_block);

#ifdef FUSE_PRE_ADD
    // The fused-add result (a + b) lives in cb_in, a kernel-local scratch CB that was reserved, pushed,
    // waited on, and read by tile index through the E[x] and X^2 passes above. The X^2 loop is its last
    // read, so pop it here to leave the CB balanced. On the non-fused path cb_in aliases the
    // buffer-backed input CB, which is read by index and never waited or popped.
    cb_in.pop_front(num_tiles_per_block);
#endif

#ifdef DO_COL_MASK
    // The mean-of-squares reduce (RMSNorm's statistic, and LayerNorm's E[x^2]) squares the raw input,
    // which leaves the padding columns holding (pad_value)^2; zero them in place before the reduce so
    // they do not enter the mean of squares. The writer-generated mask (cb_col_mask_packed) carries
    // each block's own validity (full, partial, or all-padding tiles). It was waited on near the
    // top of the kernel and is read by tile index here (never popped).
    reconfig_data_format(cb_x2_id, cb_col_mask_packed_id);
    norm::kernel_util::compute::mask_block_in_place(cb_x2, cb_col_mask_packed_id, num_tiles_per_block, block_w);
#endif

    // E(x^2)
    cb_x2.wait_front(num_tiles_per_block);
#ifdef RMSNORM
    cb_scaler.wait_front(1);
#endif  // RMSNORM

    // RMS E(x2) #Layernorm //E(x) and E(x^2)
    compute_kernel_lib::reduce<
        PoolType::AVG,
        ReduceDim::REDUCE_ROW,
        cb_x2_id,
        cb_scaler_id,
        cb_ex_partial2_id,
        compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop>(
        compute_kernel_lib::ReduceInputBlockShape::of(block_h, num_reduce_tiles_per_block_h),
        compute_kernel_lib::ReduceInputMemoryLayout::with_row_stride(block_w));
    reconfig_data_format(cb_x2_id, cb_scaler_id);
    cb_x2.pop_front(num_tiles_per_block);

    // global reduce, cb_ex <-- cb_ex_external2_id, cb_ex_partial2_id
    if constexpr (is_allgather_worker) {
        cb_scaler_global.wait_front(1);
        reconfig_data_format(cb_scaler_global_id, cb_ex_external2_id);
        pack_reconfig_data_format(cb_reduction_out);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_ex_external2_id, cb_scaler_global_id, cb_reduction_out);
        DataflowBuffer(cb_reduction_out)
            .reserve_back(num_tiles_per_partial_result * num_tiles_per_allgather_worker);

        for (uint32_t i = 0; i < num_tiles_per_allgather_worker; i++) {  // loops over height
            tile_regs_acquire();
            for (uint32_t w = 0; w < num_tiles_per_partial_result * num_blocks_reduce;
                 w++) {  // Need to read this interleaved now, we have SUM(X) and SUM(X^2) interleaved
                cb_ex_external2.wait_front(1);
                reduce_tile<PoolType::AVG, ReduceDim::REDUCE_ROW>(
                    cb_ex_external2_id,
                    cb_scaler_global_id,
                    0,
                    scaler0,
                    w % num_tiles_per_partial_result);  // E(x) and E(x^2) interleaved so we reduce each one into
                                                        // different dest reg
                cb_ex_external2.pop_front(1);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, cb_reduction_out);
#ifndef RMSNORM
            pack_tile(dst1, cb_reduction_out);
#endif
            tile_regs_release();
        }
        reduce_uninit();
        DataflowBuffer(cb_reduction_out)
            .push_back(num_tiles_per_partial_result * num_tiles_per_allgather_worker);
        // The global-reduce scaler tile is pushed once (only on all-gather worker cores) and read by
        // tile index throughout the global reduce above without being popped. Pop it once here, inside
        // the same guard that gated the wait, so the CB is left balanced on every core.
        cb_scaler_global.pop_front(1);
    }
    // The single scaler tile is waited once (by the E[x] reduce on the LayerNorm path or the E[x^2]
    // reduce on the RMSNorm path) but never popped; pop it once here so the CB is left balanced.
    cb_scaler.pop_front(1);
#ifdef DO_COL_MASK
    // The column mask is waited once near the top of the kernel (on every core) and read by tile index
    // at every masking site; pop its block_w tiles once here so the CB is left balanced.
    cb_col_mask_packed.pop_front(block_w);
#endif
}
