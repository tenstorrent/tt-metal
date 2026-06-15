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
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

// SPLIT REDUCE across Cores
void kernel_main() {
    constexpr uint32_t is_top_row = get_compile_time_arg_val(0);
    constexpr uint32_t do_gamma = get_compile_time_arg_val(1);
    constexpr uint32_t do_beta = get_compile_time_arg_val(2);
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
    constexpr bool FP32_DEST_ACC = compute_kernel_lib::get_fp32_dest_acc_enabled();
    constexpr bool LEGACY_RSQRT = get_compile_time_arg_val(12) == 1;
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

    bool enable_sqrt;
    if (use_two_stage_reduce and not is_second_stage_reader) {
        enable_sqrt = false;
    } else {
        enable_sqrt = true;
    }

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t scaler0 = 0;

    constexpr uint32_t cb_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t cb_in1 = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t cb_scaler_id = get_named_compile_time_arg_val("cb_scaler");
    constexpr uint32_t cb_eps = get_named_compile_time_arg_val("cb_eps");
    constexpr uint32_t cb_scaler_global_id = get_named_compile_time_arg_val("cb_scaler_global");
    constexpr uint32_t cb_gamma_id = get_named_compile_time_arg_val("cb_gamma");
    constexpr uint32_t cb_beta_id = get_named_compile_time_arg_val("cb_beta");
    constexpr uint32_t cb_x = get_named_compile_time_arg_val("cb_x");  // x minus mean
#if defined RMSNORM and not defined FUSE_PRE_ADD
    constexpr uint32_t cb_xmm_id = cb_in0;  // x minus mean
#else
    constexpr uint32_t cb_xmm_id = get_named_compile_time_arg_val("cb_xmm");  // x minus mean
#endif
    constexpr uint32_t cb_ex_partial_id = get_named_compile_time_arg_val("cb_ex_partial");  // E[x] partial reduce
    constexpr uint32_t cb_ex_id = get_named_compile_time_arg_val("cb_ex");                  // E[x] global reduce
    constexpr uint32_t cb_ex_external_id = get_named_compile_time_arg_val("cb_ex_external");
    constexpr uint32_t cb_ex_partial2_id =
        get_named_compile_time_arg_val("cb_ex_partial2");                     // E[(x-E[x])^2] partial reduce
    constexpr uint32_t cb_ex2_id = get_named_compile_time_arg_val("cb_ex2");  // E[(x-E[x])^2] global reduce
    constexpr uint32_t cb_ex_external2_id = get_named_compile_time_arg_val("cb_ex_external2");
    constexpr uint32_t cb_ex_global_id = get_named_compile_time_arg_val("cb_ex_global");  // E[x] global reduce
    constexpr uint32_t cb_xmm2_id = cb_x;                                                 // xmm^2
    constexpr uint32_t cb_ex2pe_id = get_named_compile_time_arg_val("cb_ex2pe");          // E[(x-E[x])^2]+eps
    constexpr uint32_t cb_fusion_id =
        get_named_compile_time_arg_val("cb_xmm");  // stream gamma/beta (alias of cb_xmm_id)
    constexpr uint32_t cb_out_id = get_named_compile_time_arg_val("cb_out");

    CircularBuffer cb_scaler(cb_scaler_id);
    CircularBuffer cb_scaler_global(cb_scaler_global_id);
    CircularBuffer cb_gamma(cb_gamma_id);
    CircularBuffer cb_beta(cb_beta_id);
    CircularBuffer cb_xmm(cb_xmm_id);
    CircularBuffer cb_ex_partial(cb_ex_partial_id);
    CircularBuffer cb_ex(cb_ex_id);
    CircularBuffer cb_ex_external(cb_ex_external_id);
    CircularBuffer cb_ex_partial2(cb_ex_partial2_id);
    CircularBuffer cb_ex2(cb_ex2_id);
    CircularBuffer cb_ex_external2(cb_ex_external2_id);
    CircularBuffer cb_ex_global(cb_ex_global_id);
    CircularBuffer cb_xmm2(cb_xmm2_id);
    CircularBuffer cb_ex2pe(cb_ex2pe_id);
    CircularBuffer cb_fusion(cb_fusion_id);
    CircularBuffer cb_out(cb_out_id);

    binary_op_init_common(cb_in0, cb_in0, cb_x);

    // set block_h to volatile to disable automatically unroll of the loops, avoid code overflow
    const uint32_t block_h = (block_w == 1) ? block_h_volatile : block_h_const;
    const uint32_t subblock_w = (block_w <= 2) ? subblock_w_volatile : subblock_w_const;

    int index_subblock_w_offset = 0;
    int index_h_offset = 0;
    int index = 0;

#ifdef FUSE_PRE_ADD
#ifdef RMSNORM
    constexpr uint32_t cb_in_id = cb_xmm_id;
#else
    constexpr uint32_t cb_in_id = cb_x;
#endif
#else
    constexpr uint32_t cb_in_id = cb_in0;
#endif
    CircularBuffer cb_in(cb_in_id);
    constexpr uint32_t cb_im_id = do_gamma ? cb_x : (do_beta ? cb_fusion_id : cb_out_id);
    CircularBuffer cb_im(cb_im_id);
    constexpr uint32_t cb_outgamma_id = do_beta ? cb_fusion_id : cb_out_id;
    CircularBuffer cb_outgamma(cb_outgamma_id);

// pre-add x + y
#ifdef FUSE_PRE_ADD
    reconfig_data_format_srcb(cb_in0, cb_in1);
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
#ifndef RMSNORM
    reconfig_data_format(cb_in0, cb_in_id, cb_in1, cb_scaler_id);
#else
    reconfig_data_format(cb_in0, cb_in_id, cb_in1, cb_in_id);
#endif
    cb_in.wait_front(num_tiles_per_block);
#else
#ifndef RMSNORM
    reconfig_data_format_srcb(cb_in0, cb_scaler_id);
#endif  // RMSNORM
#endif  // FUSE_PRE_ADD

#ifndef RMSNORM
    // E[x],
    compute_kernel_lib::reduce<
        PoolType::AVG,
        ReduceDim::REDUCE_ROW,
        cb_in_id,
        cb_scaler_id,
        cb_ex_partial_id,
        compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT>(
        compute_kernel_lib::ReduceInputBlockShape::of(block_h, num_reduce_tiles_per_block_h, 1),
        compute_kernel_lib::ReduceInputMemoryLayout::with_row_stride(block_w));
    reconfig_data_format(cb_ex_external_id, cb_scaler_id);

    // global reduce, cb_ex_id <-- cb_ex_external_id, cb_ex_partial_id
    if constexpr (is_allgather_worker) {
        reduce_init<PoolType::AVG, ReduceDim::REDUCE_ROW, FP32_DEST_ACC>(
            cb_ex_external_id, cb_scaler_global_id, cb_ex_id);
        cb_ex.reserve_back(num_tiles_per_allgather_worker);

        for (uint32_t i = 0; i < num_tiles_per_allgather_worker; i++) {
            cb_scaler_global.wait_front(1);
            tile_regs_acquire();
            for (uint32_t w = 0; w < num_blocks_reduce; w++) {
                cb_ex_external.wait_front(1);
                reduce_tile<PoolType::AVG, ReduceDim::REDUCE_ROW, FP32_DEST_ACC>(
                    cb_ex_external_id, cb_scaler_global_id, 0, scaler0, dst0);
                cb_ex_external.pop_front(1);
            }
            if (use_two_stage_reduce && !is_second_stage_reader) {
                cb_ex_external.wait_front(num_blocks_second_stage - 1);
                cb_ex_external.pop_front(num_blocks_second_stage - 1);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, cb_ex_id);
            tile_regs_release();
        }
        reduce_uninit<FP32_DEST_ACC>();
        cb_ex.push_back(num_tiles_per_allgather_worker);
        cb_ex.wait_front(num_tiles_per_allgather_worker);
    }

    // x - E[x]
    if constexpr (FLOAT32_DTYPE) {
        reconfig_data_format(cb_in_id, cb_ex_global_id);
    }
    index_h_offset = 0;
    reconfig_data_format_srca(cb_ex_external_id, cb_in_id);
    sub_bcast_cols_init_short(cb_in_id, cb_ex_global_id);
    cb_xmm.reserve_back(num_tiles_per_block);
    for (uint32_t i = 0; i < block_h; i++) {
        index_subblock_w_offset = 0;
        cb_ex_global.wait_front(1);
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset;
                sub_tiles_bcast_cols(cb_in_id, cb_ex_global_id, index, 0, w);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < subblock_w; i++) {
                pack_tile(i, cb_xmm_id);
            }
            tile_regs_release();
            index_subblock_w_offset += subblock_w;
        }
        cb_ex_global.pop_front(1);
        cb_in.pop_front(block_w);
    }
    cb_xmm.push_back(num_tiles_per_block);
#ifndef FUSE_PRE_ADD
    reconfig_data_format_srca(cb_in_id, cb_xmm_id);
#endif
    cb_xmm.wait_front(num_tiles_per_block);
#endif

    // (x - E[x])^2, cb_mm2 <-- cb_xmm_id
    mul_tiles_init(cb_xmm_id, cb_xmm_id);
    index_h_offset = 0;
    cb_xmm2.reserve_back(num_tiles_per_block);
    for (uint32_t i = 0; i < block_h; i++) {
        index_subblock_w_offset = 0;
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset + index_h_offset;
                mul_tiles(cb_xmm_id, cb_xmm_id, index, index, w);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < subblock_w; i++) {
                pack_tile(i, cb_xmm2_id);
            }
            tile_regs_release();
            index_subblock_w_offset += subblock_w;
        }
        index_h_offset += block_w;
    }
    cb_xmm2.push_back(num_tiles_per_block);

#if defined RMSNORM and not defined FUSED_PRE_ADD
    reconfig_data_format(cb_xmm_id, cb_xmm2_id, cb_xmm_id, cb_scaler_id);
#else
    if constexpr (FLOAT32_DTYPE) {
        reconfig_data_format(cb_xmm_id, cb_xmm2_id, cb_xmm_id, cb_scaler_id);
    }
#endif

    cb_xmm2.wait_front(num_tiles_per_block);

    // Var(x)
    compute_kernel_lib::reduce<
        PoolType::AVG,
        ReduceDim::REDUCE_ROW,
        cb_xmm2_id,
        cb_scaler_id,
        cb_ex_partial2_id,
        compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT>(
        compute_kernel_lib::ReduceInputBlockShape::of(block_h, num_reduce_tiles_per_block_h, 1),
        compute_kernel_lib::ReduceInputMemoryLayout::with_row_stride(block_w));
    reconfig_data_format(cb_xmm2_id, cb_scaler_id);
    cb_xmm2.pop_front(num_tiles_per_block);

    // global reduce, cb_ex_id <-- cb_ex_external_id, cb_ex_partial_id
    if constexpr (is_allgather_worker) {
        reduce_init<PoolType::AVG, ReduceDim::REDUCE_ROW, FP32_DEST_ACC>(
            cb_ex_external2_id, cb_scaler_global_id, cb_ex2_id);
        cb_ex2.reserve_back(num_tiles_per_allgather_worker);

        for (uint32_t i = 0; i < num_tiles_per_allgather_worker; i++) {
            cb_scaler_global.wait_front(1);

            tile_regs_acquire();
            for (uint32_t w = 0; w < num_blocks_reduce; w++) {
                cb_ex_external2.wait_front(1);
                reduce_tile<PoolType::AVG, ReduceDim::REDUCE_ROW, FP32_DEST_ACC>(
                    cb_ex_external2_id, cb_scaler_global_id, 0, scaler0, dst0);
                cb_ex_external2.pop_front(1);
            }
            if (use_two_stage_reduce && !is_second_stage_reader) {
                cb_ex_external2.wait_front(num_blocks_second_stage - 1);
                cb_ex_external2.pop_front(num_blocks_second_stage - 1);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, cb_ex2_id);
            tile_regs_release();
        }
        reduce_uninit<FP32_DEST_ACC>();
        cb_ex2.push_back(num_tiles_per_allgather_worker);

        if (enable_sqrt) {
            for (uint32_t i = 0; i < num_tiles_per_allgather_worker; i++) {
                // 1/[sqrt(Var + eps)],
                cb_ex2.wait_front(1);
                cb_ex2pe.reserve_back(1);
                tile_regs_acquire();
                add_tiles_init(cb_ex2_id, cb_eps);
                add_tiles(cb_ex2_id, cb_eps, i, 0, dst0);
                tile_regs_wait();
                rsqrt_tile_init<LEGACY_RSQRT>();
                rsqrt_tile<LEGACY_RSQRT>(dst0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(dst0, cb_ex2pe_id);
                cb_ex2pe.push_back(1);
                tile_regs_release();
            }
        }
    }

    if constexpr (do_gamma == 0 && do_beta == 0) {
        pack_reconfig_data_format(cb_out_id);
    }
// (x - Ex) * 1/[sqrt(Var + eps)]
#if defined RMSNORM and not defined FUSE_PRE_ADD
    if constexpr (FLOAT32_DTYPE) {
        reconfig_data_format(cb_xmm_id, cb_ex_global_id);
    } else {
        reconfig_data_format_srca(cb_ex2_id, cb_xmm_id);
    }
#else
    if constexpr (FLOAT32_DTYPE) {
        reconfig_data_format(cb_xmm_id, cb_ex_global_id);
    }
#endif
    mul_bcast_cols_init_short(cb_xmm_id, cb_ex_global_id);
    index_h_offset = 0;
    cb_im.reserve_back(num_tiles_per_block);
    for (uint32_t i = 0; i < block_h; i++) {
        index_subblock_w_offset = 0;
        cb_ex_global.wait_front(1);
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset + index_h_offset;
                mul_tiles_bcast_cols(cb_xmm_id, cb_ex_global_id, index, 0, w);

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
                pack_tile(i, cb_im_id);
            }
            tile_regs_release();

            index_subblock_w_offset += subblock_w;
        }
        index_h_offset += block_w;
        cb_ex_global.pop_front(1);
    }
    cb_im.push_back(num_tiles_per_block);

    cb_xmm.pop_front(num_tiles_per_block);
    cb_im.wait_front(num_tiles_per_block);

    if constexpr (do_gamma) {
        reconfig_data_format(cb_im_id, cb_gamma_id);
        if constexpr (do_beta == 0) {
            pack_reconfig_data_format(cb_out_id);
        }
        mul_bcast_rows_init_short(cb_im_id, cb_gamma_id);
        cb_gamma.wait_front(block_w);
        index_h_offset = 0;
        cb_outgamma.reserve_back(num_tiles_per_block);
        for (uint32_t i = 0; i < block_h; i++) {
            index_subblock_w_offset = 0;
            for (uint32_t j = 0; j < num_subblocks_w; j++) {
                tile_regs_acquire();
                for (uint32_t w = 0; w < subblock_w; w++) {
                    index = w + index_subblock_w_offset;
                    mul_tiles_bcast_rows(cb_im_id, cb_gamma_id, index + index_h_offset, index, w);
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
                    pack_tile(i, cb_outgamma_id);
                }
                tile_regs_release();
                index_subblock_w_offset += subblock_w;
            }
            index_h_offset += block_w;
        }
        cb_outgamma.push_back(num_tiles_per_block);
        cb_im.pop_front(num_tiles_per_block);
        cb_outgamma.wait_front(num_tiles_per_block);
    }

    if constexpr (do_beta) {
        reconfig_data_format(cb_fusion_id, cb_beta_id);
        pack_reconfig_data_format(cb_out_id);
        add_bcast_rows_init_short(cb_fusion_id, cb_beta_id);
        cb_beta.wait_front(block_w);
        index_h_offset = 0;
        cb_out.reserve_back(num_tiles_per_block);
        for (uint32_t i = 0; i < block_h; i++) {
            index_subblock_w_offset = 0;
            for (uint32_t j = 0; j < num_subblocks_w; j++) {
                tile_regs_acquire();
                for (uint32_t w = 0; w < subblock_w; w++) {
                    index = w + index_subblock_w_offset;
                    add_tiles_bcast_rows(cb_fusion_id, cb_beta_id, index + index_h_offset, index, w);
#ifdef SFPU_OP_INIT_ACTIVATION
                    SFPU_OP_INIT_ACTIVATION
                    SFPU_OP_FUNC_ACTIVATION
#endif
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < subblock_w; i++) {
                    pack_tile(i, cb_out_id);
                }
                tile_regs_release();
                index_subblock_w_offset += subblock_w;
            }
            index_h_offset += block_w;
        }
        cb_out.push_back(num_tiles_per_block);
        cb_fusion.pop_front(num_tiles_per_block);
        cb_out.wait_front(num_tiles_per_block);
    }
}
