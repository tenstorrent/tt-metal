// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#ifdef DO_COL_MASK
#include "ttnn/operations/normalization/kernel_util/compute/col_mask.h"
#endif

// SPLIT REDUCE across Cores
void kernel_main() {
    // An idle core sits in a hole of a non-rectangular shard grid. It carries this program's dataflow
    // buffers so the reduction's multicast has somewhere to land, and does no work of its own, so its
    // whole body is compiled out.
#ifndef IDLE_CORE

    constexpr auto num_blocks_first_stage = get_arg(args::num_blocks_first_stage);
    constexpr auto block_w = get_arg(args::block_w);
    constexpr auto block_h_const = get_arg(args::block_h);
    volatile uint32_t block_h_volatile = get_arg(args::block_h);
    constexpr auto subblock_w_const = get_arg(args::subblock_w);
    volatile uint32_t subblock_w_volatile = get_arg(args::subblock_w);
    constexpr auto num_subblocks_w = get_arg(args::num_subblocks_w);
    constexpr auto num_tiles_per_block = get_arg(args::num_tiles_per_block);
    constexpr bool FLOAT32_DTYPE = get_arg(args::float32_dtype) == 1;
    constexpr bool FP32_DEST_ACC = compute_kernel_lib::get_fp32_dest_acc_enabled();
    constexpr bool LEGACY_RSQRT = get_arg(args::legacy_rsqrt) == 1;
    constexpr auto num_blocks_second_stage = get_arg(args::num_blocks_second_stage);
    // gamma and beta each gate a buffer that only exists when their tensor was supplied, so the flag
    // has to reach the preprocessor as well as `if constexpr`.
#ifdef FUSE_GAMMA
    constexpr bool do_gamma = true;
#else
    constexpr bool do_gamma = false;
#endif
#ifdef FUSE_BETA
    constexpr bool do_beta = true;
#else
    constexpr bool do_beta = false;
#endif
    // Only the cores that gather read the three cross-core reduce arguments below, so the distinction
    // is a compile-time one: their runtime-argument schemas differ.
#ifdef IS_ALLGATHER_WORKER
    constexpr bool is_allgather_worker = true;
#else
    constexpr bool is_allgather_worker = false;
#endif

    const uint32_t num_reduce_tiles_per_block_h = get_arg(
        args::num_reduce_tiles_per_block_h);  // This value is the same for all cores, except ones that have
                                              // padding tiles in it. In that case, skip reduce for padding tiles.
#ifdef IS_ALLGATHER_WORKER
    const uint32_t num_tiles_per_allgather_worker = get_arg(args::num_rows_per_all_to_all_worker);
    const bool use_two_stage_reduce = get_arg(args::use_two_stage_reduce) == 1;
    const bool is_second_stage_reader = get_arg(args::is_second_stage_reader) == 1;
#else
    const uint32_t num_tiles_per_allgather_worker = 0;
    const bool use_two_stage_reduce = false;
    const bool is_second_stage_reader = false;
#endif

    uint32_t num_blocks_reduce;
    if (is_second_stage_reader) {
        num_blocks_reduce = num_blocks_first_stage + num_blocks_second_stage - 1;
    } else {
        num_blocks_reduce = num_blocks_first_stage;
    }

    bool enable_sqrt;
    if (use_two_stage_reduce and not is_second_stage_reader) {
        enable_sqrt = false;
    } else {
        enable_sqrt = true;
    }

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t scaler0 = 0;

    constexpr uint32_t dfb_in0 = dfb::in0;
    constexpr uint32_t dfb_scaler_id = dfb::scaler;
    constexpr uint32_t dfb_eps = dfb::eps;
    constexpr uint32_t dfb_scaler_global_id = dfb::scaler_global;
    constexpr uint32_t dfb_x = dfb::x;  // x minus mean
#ifdef FUSE_PRE_ADD
    constexpr uint32_t dfb_in1 = dfb::in1;
#endif
#ifdef FUSE_GAMMA
    constexpr uint32_t dfb_gamma_id = dfb::gamma;
#endif
#ifdef FUSE_BETA
    constexpr uint32_t dfb_beta_id = dfb::beta;
#endif
#if defined RMSNORM and not defined FUSE_PRE_ADD
    constexpr uint32_t dfb_xmm_id = dfb_in0;  // x minus mean
#else
    constexpr uint32_t dfb_xmm_id = dfb::xmm;  // x minus mean
#endif
    // RMSNorm normalizes by the mean of squares, so it has no mean to reduce and the E[x] chain's
    // buffers are not declared.
#ifndef RMSNORM
    constexpr uint32_t dfb_ex_partial_id = dfb::ex_partial;    // E[x] partial reduce
    constexpr uint32_t dfb_ex_id = dfb::ex;                    // E[x] global reduce
    constexpr uint32_t dfb_ex_external_id = dfb::ex_external;  // partial E[x] from the other cores
#endif
    constexpr uint32_t dfb_ex_partial2_id = dfb::ex_partial2;    // E[(x-E[x])^2] partial reduce
    constexpr uint32_t dfb_ex2_id = dfb::ex2;                    // E[(x-E[x])^2] global reduce
    constexpr uint32_t dfb_ex_external2_id = dfb::ex_external2;  // partial Var[x] from the other cores
    constexpr uint32_t dfb_ex_global_id = dfb::ex_global;        // E[x] global reduce
    constexpr uint32_t dfb_xmm2_id = dfb_x;                      // xmm^2
    constexpr uint32_t dfb_ex2pe_id = dfb::ex2pe;                // E[(x-E[x])^2]+eps
    constexpr uint32_t dfb_fusion_id = dfb::xmm;                 // stream gamma/beta (alias of dfb_xmm_id)
    constexpr uint32_t dfb_out_id = dfb::out;
#ifdef DO_COL_MASK
#ifndef RMSNORM
    // Scratch buffer holding the input with any padding columns zeroed, so those
    // columns contribute 0 to the E[x] sum. The input buffer stays intact for the (x - E[x]) pass;
    // the masking itself uses the column mask below.
    // RMSNorm needs no such copy: it masks the squared input in place (a fresh intermediate,
    // not the input), so the input is never overwritten in the first place.
    constexpr uint32_t dfb_mask_scratch_id = dfb::mask_scratch;
    DataflowBuffer dfb_mask_scratch(dfb_mask_scratch_id);
#endif
    // Multiplicative mask that keeps tile padding out of the statistics. Each row of the input is
    // normalized only over its real elements (logical width). Input is stored in 32-column-wide tiles,
    // and this core's block is block_w tiles wide, so it has block_w * 32 columns. When the
    // logical width is not a multiple of 32, the columns beyond the logical width hold padding
    // that needs to be ignored. This buffer holds the mask: block_w tiles, one for each input data tile.
    // The writer fills this buffer; this kernel waits for it once below, reads tiles by index during the
    // body (the same mask is reused at every masking site), and pops it once at the end so the buffer is
    // left balanced.
    constexpr uint32_t dfb_col_mask_packed_id = dfb::col_mask;
    DataflowBuffer dfb_col_mask_packed(dfb_col_mask_packed_id);
#endif

    DataflowBuffer dfb_scaler(dfb_scaler_id);
    DataflowBuffer dfb_scaler_global(dfb_scaler_global_id);
#ifdef FUSE_GAMMA
    DataflowBuffer dfb_gamma(dfb_gamma_id);
#endif
#ifdef FUSE_BETA
    DataflowBuffer dfb_beta(dfb_beta_id);
#endif
    DataflowBuffer dfb_xmm(dfb_xmm_id);
#ifndef RMSNORM
    DataflowBuffer dfb_ex_partial(dfb_ex_partial_id);
    DataflowBuffer dfb_ex(dfb_ex_id);
    DataflowBuffer dfb_ex_external(dfb_ex_external_id);
#endif
    DataflowBuffer dfb_ex_partial2(dfb_ex_partial2_id);
    DataflowBuffer dfb_ex2(dfb_ex2_id);
    DataflowBuffer dfb_ex_external2(dfb_ex_external2_id);
    DataflowBuffer dfb_ex_global(dfb_ex_global_id);
    DataflowBuffer dfb_xmm2(dfb_xmm2_id);
    DataflowBuffer dfb_ex2pe(dfb_ex2pe_id);
    DataflowBuffer dfb_fusion(dfb_fusion_id);
    DataflowBuffer dfb_out(dfb_out_id);

    compute_kernel_hw_startup(dfb_in0, dfb_in0, dfb_x);

#ifdef DO_COL_MASK
    // The column mask has block_w tiles, one per tile across the shard width.
    // Wait once for it here; the masking sites below read it by tile index without
    // re-waiting (it is reused across all rows and masking sites).
    // It is popped once at the end of the kernel so the buffer is left balanced.
    dfb_col_mask_packed.wait_front(block_w);
#endif

    // set block_h to volatile to disable automatically unroll of the loops, avoid code overflow
    const uint32_t block_h = (block_w == 1) ? block_h_volatile : block_h_const;
    const uint32_t subblock_w = (block_w <= 2) ? subblock_w_volatile : subblock_w_const;

    int index_subblock_w_offset = 0;
    int index_h_offset = 0;
    int index = 0;

#ifdef FUSE_PRE_ADD
#ifdef RMSNORM
    constexpr uint32_t dfb_in_id = dfb_xmm_id;
#else
    constexpr uint32_t dfb_in_id = dfb_x;
#endif
#else
    constexpr uint32_t dfb_in_id = dfb_in0;
#endif
    DataflowBuffer dfb_in(dfb_in_id);
    constexpr uint32_t dfb_im_id = do_gamma ? dfb_x : (do_beta ? dfb_fusion_id : dfb_out_id);
    DataflowBuffer dfb_im(dfb_im_id);
    constexpr uint32_t dfb_outgamma_id = do_beta ? dfb_fusion_id : dfb_out_id;
    DataflowBuffer dfb_outgamma(dfb_outgamma_id);

// pre-add x + y
#ifdef FUSE_PRE_ADD
    reconfig_data_format_srcb(dfb_in0, dfb_in1);
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
#ifndef RMSNORM
    reconfig_data_format(dfb_in0, dfb_in_id, dfb_in1, dfb_scaler_id);
#else
    reconfig_data_format(dfb_in0, dfb_in_id, dfb_in1, dfb_in_id);
#endif
    dfb_in.wait_front(num_tiles_per_block);
#else
#ifndef RMSNORM
    reconfig_data_format_srcb(dfb_in0, dfb_scaler_id);
#endif  // RMSNORM
#endif  // FUSE_PRE_ADD

#ifndef RMSNORM
#ifdef DO_COL_MASK
    // Zero any padding columns of the input into the mask scratch so they do not contribute
    // to E[x]; the reduce below consumes the masked copy instead of the input.
    // The input itself is left intact for the (x - E[x]) pass that follows. The column mask is the
    // writer-generated mask (1.0 valid / 0.0 padding), already waited on above and read by
    // tile index.
    reconfig_data_format(dfb_in_id, dfb_col_mask_packed_id);
    mul_init(dfb_in_id, dfb_col_mask_packed_id);
    dfb_mask_scratch.reserve_back(num_tiles_per_block);
    index_h_offset = 0;
    for (uint32_t i = 0; i < block_h; i++) {
        for (uint32_t wt = 0; wt < block_w; wt++) {
            tile_regs_acquire();
            mul_tiles(dfb_in_id, dfb_col_mask_packed_id, wt + index_h_offset, wt, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dfb_mask_scratch_id);
            tile_regs_release();
        }
        index_h_offset += block_w;
    }
    dfb_mask_scratch.push_back(num_tiles_per_block);
    dfb_mask_scratch.wait_front(num_tiles_per_block);
    reconfig_data_format_srcb(dfb_col_mask_packed_id, dfb_scaler_id);
    constexpr uint32_t dfb_ex_reduce_input = dfb_mask_scratch_id;
#else
    constexpr uint32_t dfb_ex_reduce_input = dfb_in_id;
#endif
    // E[x],
    compute_kernel_lib::reduce<
        PoolType::AVG,
        ReduceDim::REDUCE_ROW,
        dfb_ex_reduce_input,
        dfb_scaler_id,
        dfb_ex_partial_id,
        compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT>(
        compute_kernel_lib::ReduceInputBlockShape::of(block_h, num_reduce_tiles_per_block_h, 1),
        compute_kernel_lib::ReduceInputMemoryLayout::with_row_stride(block_w));
#ifdef DO_COL_MASK
    dfb_mask_scratch.pop_front(num_tiles_per_block);
#endif
    reconfig_data_format(dfb_ex_external_id, dfb_scaler_id);

    // global reduce, dfb_ex_id <-- dfb_ex_external_id, dfb_ex_partial_id
    if constexpr (is_allgather_worker) {
        reconfig_data_format(dfb_scaler_global_id, dfb_ex_external_id);
        reduce_init<PoolType::AVG, ReduceDim::REDUCE_ROW>(dfb_ex_external_id, dfb_scaler_global_id, dfb_ex_id);
        dfb_ex.reserve_back(num_tiles_per_allgather_worker);

        for (uint32_t i = 0; i < num_tiles_per_allgather_worker; i++) {
            dfb_scaler_global.wait_front(1);
            tile_regs_acquire();
            for (uint32_t w = 0; w < num_blocks_reduce; w++) {
                dfb_ex_external.wait_front(1);
                reduce_tile<PoolType::AVG, ReduceDim::REDUCE_ROW>(
                    dfb_ex_external_id, dfb_scaler_global_id, 0, scaler0, dst0);
                dfb_ex_external.pop_front(1);
            }
            if (use_two_stage_reduce && !is_second_stage_reader) {
                dfb_ex_external.wait_front(static_cast<uint16_t>(num_blocks_second_stage - 1));
                dfb_ex_external.pop_front(static_cast<uint16_t>(num_blocks_second_stage - 1));
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, dfb_ex_id);
            tile_regs_release();
        }
        reduce_uninit();
        dfb_ex.push_back(num_tiles_per_allgather_worker);
        reconfig_data_format(dfb_ex_external_id, dfb_scaler_global_id);
        dfb_ex.wait_front(num_tiles_per_allgather_worker);
    }

    // x - E[x]
    if constexpr (FLOAT32_DTYPE) {
        reconfig_data_format(dfb_in_id, dfb_ex_global_id);
    }
    index_h_offset = 0;
    reconfig_data_format_srca(dfb_ex_external_id, dfb_in_id);
    sub_bcast_cols_init(dfb_in_id, dfb_ex_global_id);
    dfb_xmm.reserve_back(num_tiles_per_block);
    for (uint32_t i = 0; i < block_h; i++) {
        index_subblock_w_offset = 0;
        dfb_ex_global.wait_front(1);
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset;
                sub_tiles_bcast_cols(dfb_in_id, dfb_ex_global_id, index, 0, w);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < subblock_w; i++) {
                pack_tile(i, dfb_xmm_id);
            }
            tile_regs_release();
            index_subblock_w_offset += subblock_w;
        }
        dfb_ex_global.pop_front(1);
        dfb_in.pop_front(block_w);
    }
    dfb_xmm.push_back(num_tiles_per_block);
#ifndef FUSE_PRE_ADD
    reconfig_data_format_srca(dfb_in_id, dfb_xmm_id);
#endif
    dfb_xmm.wait_front(num_tiles_per_block);
#endif

#if defined(DO_COL_MASK) && !defined(RMSNORM)
    // Zero the padding columns of (x - E[x]) so the variance excludes them, using the writer-generated
    // mask (1.0 in valid columns, 0.0 in padding).
    // Applied in place by re-circulating the (x - E[x]) buffer (which also zeroes the padding for the
    // final (x - E[x]) * 1/sqrt(var+eps); that padding output is discarded). Mask tile index tracks wt
    // (the tile's position across the width). The column mask was waited on once near the top of
    // the kernel and is read by tile index here (never popped).
    // SrcA already holds the (x - E[x]) tiles' format; only SrcB changes for this multiply, to read the
    // mask in its own data format (the mask format need not match the compute tiles).
    reconfig_data_format_srcb(dfb_col_mask_packed_id);
    norm::kernel_util::compute::mask_block_in_place(dfb_xmm, dfb_col_mask_packed_id, num_tiles_per_block, block_w);
    dfb_xmm.wait_front(num_tiles_per_block);
    // The masking above reads the column mask on SrcB, leaving SrcB configured for the mask's data format,
    // which need not match the (x - E[x]) tiles. Restore SrcB to their format for the squaring below.
    reconfig_data_format_srcb(dfb_xmm_id);
#endif

    // (x - E[x])^2, dfb_xmm2 <-- dfb_xmm_id
    mul_init(dfb_xmm_id, dfb_xmm_id);
    index_h_offset = 0;
    dfb_xmm2.reserve_back(num_tiles_per_block);
    for (uint32_t i = 0; i < block_h; i++) {
        index_subblock_w_offset = 0;
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset + index_h_offset;
                mul_tiles(dfb_xmm_id, dfb_xmm_id, index, index, w);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < subblock_w; i++) {
                pack_tile(i, dfb_xmm2_id);
            }
            tile_regs_release();
            index_subblock_w_offset += subblock_w;
        }
        index_h_offset += block_w;
    }
    dfb_xmm2.push_back(num_tiles_per_block);

#if defined(RMSNORM) && defined(DO_COL_MASK)
    // RMSNorm has no mean-subtraction stage, so its statistic is the mean of squares of the input
    // (the raw input, or the fused residual sum a + b). Squaring it leaves the padding columns holding
    // (pad_value)^2; zero them in place before the reduce so they do not enter the mean of squares.
    // The writer-generated mask carries each block's own validity (full, partial,
    // or all-padding tiles). It was waited on once near the top of the kernel and is read by tile index
    // here (never popped).
    reconfig_data_format(dfb_xmm2_id, dfb_col_mask_packed_id);
    norm::kernel_util::compute::mask_block_in_place(dfb_xmm2, dfb_col_mask_packed_id, num_tiles_per_block, block_w);
    dfb_xmm2.wait_front(num_tiles_per_block);
#endif

#if defined RMSNORM and not defined FUSED_PRE_ADD
    reconfig_data_format(dfb_xmm_id, dfb_xmm2_id, dfb_xmm_id, dfb_scaler_id);
#else
    if constexpr (FLOAT32_DTYPE) {
        reconfig_data_format(dfb_xmm_id, dfb_xmm2_id, dfb_xmm_id, dfb_scaler_id);
    }
#endif

    dfb_xmm2.wait_front(num_tiles_per_block);

    // Var(x)
    compute_kernel_lib::reduce<
        PoolType::AVG,
        ReduceDim::REDUCE_ROW,
        dfb_xmm2_id,
        dfb_scaler_id,
        dfb_ex_partial2_id,
        compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT>(
        compute_kernel_lib::ReduceInputBlockShape::of(block_h, num_reduce_tiles_per_block_h, 1),
        compute_kernel_lib::ReduceInputMemoryLayout::with_row_stride(block_w));
    reconfig_data_format(dfb_xmm2_id, dfb_scaler_id);
    dfb_xmm2.pop_front(num_tiles_per_block);

    // global reduce, dfb_ex2_id <-- dfb_ex_external2_id, dfb_ex_partial2_id
    if constexpr (is_allgather_worker) {
        reconfig_data_format(dfb_scaler_global_id, dfb_ex_external2_id);
        reduce_init<PoolType::AVG, ReduceDim::REDUCE_ROW>(dfb_ex_external2_id, dfb_scaler_global_id, dfb_ex2_id);
        dfb_ex2.reserve_back(num_tiles_per_allgather_worker);

        for (uint32_t i = 0; i < num_tiles_per_allgather_worker; i++) {
            dfb_scaler_global.wait_front(1);

            tile_regs_acquire();
            for (uint32_t w = 0; w < num_blocks_reduce; w++) {
                dfb_ex_external2.wait_front(1);
                reduce_tile<PoolType::AVG, ReduceDim::REDUCE_ROW>(
                    dfb_ex_external2_id, dfb_scaler_global_id, 0, scaler0, dst0);
                dfb_ex_external2.pop_front(1);
            }
            if (use_two_stage_reduce && !is_second_stage_reader) {
                dfb_ex_external2.wait_front(static_cast<uint16_t>(num_blocks_second_stage - 1));
                dfb_ex_external2.pop_front(static_cast<uint16_t>(num_blocks_second_stage - 1));
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, dfb_ex2_id);
            tile_regs_release();
        }
        reduce_uninit();
        dfb_ex2.push_back(num_tiles_per_allgather_worker);
        reconfig_data_format(dfb_xmm2_id, dfb_scaler_id);

        if (enable_sqrt) {
            for (uint32_t i = 0; i < num_tiles_per_allgather_worker; i++) {
                // 1/[sqrt(Var + eps)],
                dfb_ex2.wait_front(1);
                dfb_ex2pe.reserve_back(1);
                tile_regs_acquire();
                add_init(dfb_ex2_id, dfb_eps);
                add_tiles(dfb_ex2_id, dfb_eps, i, 0, dst0);
                tile_regs_wait();
                rsqrt_tile_init<LEGACY_RSQRT>();
                rsqrt_tile<LEGACY_RSQRT>(dst0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(dst0, dfb_ex2pe_id);
                dfb_ex2pe.push_back(1);
                tile_regs_release();
            }
        }
    }

    if constexpr (do_gamma == 0 && do_beta == 0) {
        pack_reconfig_data_format(dfb_out_id);
    }
// (x - Ex) * 1/[sqrt(Var + eps)]
#if defined RMSNORM and not defined FUSE_PRE_ADD
    if constexpr (FLOAT32_DTYPE) {
        reconfig_data_format(dfb_xmm_id, dfb_ex_global_id);
    } else {
        reconfig_data_format_srca(dfb_ex2_id, dfb_xmm_id);
    }
#else
    if constexpr (FLOAT32_DTYPE) {
        reconfig_data_format(dfb_xmm_id, dfb_ex_global_id);
    }
#endif
    mul_bcast_cols_init(dfb_xmm_id, dfb_ex_global_id);
    index_h_offset = 0;
    dfb_im.reserve_back(num_tiles_per_block);
    for (uint32_t i = 0; i < block_h; i++) {
        index_subblock_w_offset = 0;
        dfb_ex_global.wait_front(1);
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset + index_h_offset;
                mul_tiles_bcast_cols(dfb_xmm_id, dfb_ex_global_id, index, 0, w);

#ifdef SFPU_OP_INIT_ACTIVATION
                // Activation must be applied last. If do_gamma != 0 or do_beta != 0 then
                // activation will be applied after the gamma/beta multiplication/addition.
                // Otherwise, we can apply the activation here.
                if constexpr (!(do_gamma == 1 || do_beta == 1)) {
                    SFPU_OP_INIT_ACTIVATION
                    SFPU_OP_FUNC_ACTIVATION
                }
#endif
            }
            tile_regs_commit();

            tile_regs_wait();
            for (uint32_t i = 0; i < subblock_w; i++) {
                pack_tile(i, dfb_im_id);
            }
            tile_regs_release();

            index_subblock_w_offset += subblock_w;
        }
        index_h_offset += block_w;
        dfb_ex_global.pop_front(1);
    }
    dfb_im.push_back(num_tiles_per_block);

    dfb_xmm.pop_front(num_tiles_per_block);
    dfb_im.wait_front(num_tiles_per_block);

#ifdef FUSE_GAMMA
    {
        reconfig_data_format(dfb_im_id, dfb_gamma_id);
        if constexpr (do_beta == 0) {
            pack_reconfig_data_format(dfb_out_id);
        }
        mul_bcast_rows_init(dfb_im_id, dfb_gamma_id);
        dfb_gamma.wait_front(block_w);
        index_h_offset = 0;
        dfb_outgamma.reserve_back(num_tiles_per_block);
        for (uint32_t i = 0; i < block_h; i++) {
            index_subblock_w_offset = 0;
            for (uint32_t j = 0; j < num_subblocks_w; j++) {
                tile_regs_acquire();
                for (uint32_t w = 0; w < subblock_w; w++) {
                    index = w + index_subblock_w_offset;
                    mul_tiles_bcast_rows(dfb_im_id, dfb_gamma_id, index + index_h_offset, index, w);
#ifdef SFPU_OP_INIT_ACTIVATION
                    // Activation must be applied last. If do_beta != 0 then
                    // activation will be applied after the beta addition.
                    // Otherwise, we can apply the activation here.
                    if constexpr (!do_beta) {
                        SFPU_OP_INIT_ACTIVATION
                        SFPU_OP_FUNC_ACTIVATION
                    }
#endif
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < subblock_w; i++) {
                    pack_tile(i, dfb_outgamma_id);
                }
                tile_regs_release();
                index_subblock_w_offset += subblock_w;
            }
            index_h_offset += block_w;
        }
        dfb_outgamma.push_back(num_tiles_per_block);
        dfb_im.pop_front(num_tiles_per_block);
        dfb_outgamma.wait_front(num_tiles_per_block);
    }
#endif

#ifdef FUSE_BETA
    {
        reconfig_data_format(dfb_fusion_id, dfb_beta_id);
        pack_reconfig_data_format(dfb_out_id);
        add_bcast_rows_init(dfb_fusion_id, dfb_beta_id);
        dfb_beta.wait_front(block_w);
        index_h_offset = 0;
        dfb_out.reserve_back(num_tiles_per_block);
        for (uint32_t i = 0; i < block_h; i++) {
            index_subblock_w_offset = 0;
            for (uint32_t j = 0; j < num_subblocks_w; j++) {
                tile_regs_acquire();
                for (uint32_t w = 0; w < subblock_w; w++) {
                    index = w + index_subblock_w_offset;
                    add_tiles_bcast_rows(dfb_fusion_id, dfb_beta_id, index + index_h_offset, index, w);
#ifdef SFPU_OP_INIT_ACTIVATION
                    SFPU_OP_INIT_ACTIVATION
                    SFPU_OP_FUNC_ACTIVATION
#endif
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < subblock_w; i++) {
                    pack_tile(i, dfb_out_id);
                }
                tile_regs_release();
                index_subblock_w_offset += subblock_w;
            }
            index_h_offset += block_w;
        }
        dfb_out.push_back(num_tiles_per_block);
        dfb_fusion.pop_front(num_tiles_per_block);
        dfb_out.wait_front(num_tiles_per_block);
    }
#endif
    // The single scaler tile is waited by both reductions (E[x] and Var[x]) but never popped;
    // pop it once at the end so the buffer is left balanced.
    dfb_scaler.pop_front(1);
    if constexpr (is_allgather_worker) {
        // The global-reduce scaler tile is pushed once (only on all-gather worker cores) and read by
        // tile index across the E[x] and Var[x] global reductions without being popped. Pop it once
        // here, under the same guard that gated the waits, so the buffer is left balanced on every core.
        dfb_scaler_global.pop_front(1);
    }
#ifdef DO_COL_MASK
    // The column mask is waited once near the top of the kernel and read by tile index at every masking
    // site; pop its block_w tiles once here so the buffer is left balanced.
    dfb_col_mask_packed.pop_front(block_w);
#endif

#endif  // IDLE_CORE
}
