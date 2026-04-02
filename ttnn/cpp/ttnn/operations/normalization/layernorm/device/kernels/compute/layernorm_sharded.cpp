// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "experimental/circular_buffer.h"

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
    constexpr uint32_t cb_scaler = get_named_compile_time_arg_val("cb_scaler");
    constexpr uint32_t cb_eps = get_named_compile_time_arg_val("cb_eps");
    constexpr uint32_t cb_scaler_global = get_named_compile_time_arg_val("cb_scaler_global");
    constexpr uint32_t cb_gamma = get_named_compile_time_arg_val("cb_gamma");
    constexpr uint32_t cb_beta = get_named_compile_time_arg_val("cb_beta");
    constexpr uint32_t cb_x = get_named_compile_time_arg_val("cb_x");  // x minus mean
#if defined RMSNORM and not defined FUSE_PRE_ADD
    constexpr uint32_t cb_xmm = cb_in0;  // x minus mean
#else
    constexpr uint32_t cb_xmm = get_named_compile_time_arg_val("cb_xmm");  // x minus mean
#endif
    constexpr uint32_t cb_ex_partial = get_named_compile_time_arg_val("cb_ex_partial");  // E[x] partial reduce
    constexpr uint32_t cb_ex = get_named_compile_time_arg_val("cb_ex");                  // E[x] global reduce
    constexpr uint32_t cb_ex_external = get_named_compile_time_arg_val("cb_ex_external");
    constexpr uint32_t cb_ex_partial2 =
        get_named_compile_time_arg_val("cb_ex_partial2");                  // E[(x-E[x])^2] partial reduce
    constexpr uint32_t cb_ex2 = get_named_compile_time_arg_val("cb_ex2");  // E[(x-E[x])^2] global reduce
    constexpr uint32_t cb_ex_external2 = get_named_compile_time_arg_val("cb_ex_external2");
    constexpr uint32_t cb_ex_global = get_named_compile_time_arg_val("cb_ex_global");  // E[x] global reduce
    constexpr uint32_t cb_xmm2 = cb_x;                    // xmm^2
    constexpr uint32_t cb_ex2pe = get_named_compile_time_arg_val("cb_ex2pe");  // E[(x-E[x])^2]+eps
    constexpr uint32_t cb_fusion = get_named_compile_time_arg_val("cb_xmm");   // stream gamma/beta (alias of cb_xmm)
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");

    experimental::CircularBuffer cb_scaler_obj(cb_scaler);
    experimental::CircularBuffer cb_scaler_global_obj(cb_scaler_global);
    experimental::CircularBuffer cb_gamma_obj(cb_gamma);
    experimental::CircularBuffer cb_beta_obj(cb_beta);
    experimental::CircularBuffer cb_xmm_obj(cb_xmm);
    experimental::CircularBuffer cb_ex_partial_obj(cb_ex_partial);
    experimental::CircularBuffer cb_ex_obj(cb_ex);
    experimental::CircularBuffer cb_ex_external_obj(cb_ex_external);
    experimental::CircularBuffer cb_ex_partial2_obj(cb_ex_partial2);
    experimental::CircularBuffer cb_ex2_obj(cb_ex2);
    experimental::CircularBuffer cb_ex_external2_obj(cb_ex_external2);
    experimental::CircularBuffer cb_ex_global_obj(cb_ex_global);
    experimental::CircularBuffer cb_xmm2_obj(cb_xmm2);
    experimental::CircularBuffer cb_ex2pe_obj(cb_ex2pe);
    experimental::CircularBuffer cb_fusion_obj(cb_fusion);
    experimental::CircularBuffer cb_out_obj(cb_out);

    binary_op_init_common(cb_in0, cb_in0, cb_x);

    // set block_h to volatile to disable automatically unroll of the loops, avoid code overflow
    const uint32_t block_h = (block_w == 1) ? block_h_volatile : block_h_const;
    const uint32_t subblock_w = (block_w <= 2) ? subblock_w_volatile : subblock_w_const;

    int index_subblock_w_offset = 0;
    int index_h_offset = 0;
    int index = 0;

#ifdef FUSE_PRE_ADD
#ifdef RMSNORM
    constexpr uint32_t cb_in = cb_xmm;
#else
    constexpr uint32_t cb_in = cb_x;
#endif
#else
    constexpr uint32_t cb_in = cb_in0;
#endif
    experimental::CircularBuffer cb_in_obj(cb_in);
    constexpr uint32_t cb_im = do_gamma ? cb_x : (do_beta ? cb_fusion : cb_out);
    experimental::CircularBuffer cb_im_obj(cb_im);
    constexpr uint32_t cb_outgamma = do_beta ? cb_fusion : cb_out;
    experimental::CircularBuffer cb_outgamma_obj(cb_outgamma);

// pre-add x + y
#ifdef FUSE_PRE_ADD
    reconfig_data_format_srcb(cb_in0, cb_in1);
    add_tiles_init(cb_in0, cb_in1);
    cb_in_obj.reserve_back(num_tiles_per_block);
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
                pack_tile(i, cb_in);
            }
            tile_regs_release();
            index_subblock_w_offset += subblock_w;
        }
        index_h_offset += block_w;
    }
    cb_in_obj.push_back(num_tiles_per_block);
#ifndef RMSNORM
    reconfig_data_format(cb_in0, cb_in, cb_in1, cb_scaler);
#else
    reconfig_data_format(cb_in0, cb_in, cb_in1, cb_in);
#endif
    cb_in_obj.wait_front(num_tiles_per_block);
#else
#ifndef RMSNORM
    reconfig_data_format_srcb(cb_in0, cb_scaler);
#endif  // RMSNORM
#endif  // FUSE_PRE_ADD

#ifndef RMSNORM
    // E[x],
    index_h_offset = 0;
    reduce_init<REDUCE_OP, REDUCE_DIM, FLOAT32_REDUCTION>(cb_in, cb_scaler, cb_ex_partial);
    cb_scaler_obj.wait_front(1);
    cb_ex_partial_obj.reserve_back(block_h);
    for (uint32_t i = 0; i < block_h; i++) {
        tile_regs_acquire();
        for (uint32_t w = 0; w < num_reduce_tiles_per_block_h; w++) {
            reduce_tile<REDUCE_OP, REDUCE_DIM, FLOAT32_REDUCTION>(cb_in, cb_scaler, w + index_h_offset, scaler0, dst0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_ex_partial);
        tile_regs_release();
        index_h_offset += block_w;
    }
    reduce_uninit();
    cb_ex_partial_obj.push_back(block_h);

    reconfig_data_format_srca(cb_in, cb_ex_external);

    // global reduce, cb_ex <-- cb_ex_external, cb_ex_partial
    if constexpr (is_allgather_worker) {
        reduce_init<REDUCE_OP, REDUCE_DIM, FLOAT32_REDUCTION>(cb_ex_external, cb_scaler_global, cb_ex);
        cb_ex_obj.reserve_back(num_tiles_per_allgather_worker);

        for (uint32_t i = 0; i < num_tiles_per_allgather_worker; i++) {
            cb_scaler_global_obj.wait_front(1);
            tile_regs_acquire();
            for (uint32_t w = 0; w < num_blocks_reduce; w++) {
                cb_ex_external_obj.wait_front(1);
                reduce_tile<REDUCE_OP, REDUCE_DIM, FLOAT32_REDUCTION>(
                    cb_ex_external, cb_scaler_global, 0, scaler0, dst0);
                cb_ex_external_obj.pop_front(1);
            }
            if (use_two_stage_reduce && !is_second_stage_reader) {
                cb_ex_external_obj.wait_front(num_blocks_second_stage - 1);
                cb_ex_external_obj.pop_front(num_blocks_second_stage - 1);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, cb_ex);
            tile_regs_release();
        }
        reduce_uninit();
        cb_ex_obj.push_back(num_tiles_per_allgather_worker);
        cb_ex_obj.wait_front(num_tiles_per_allgather_worker);
    }

    // x - E[x]
    if constexpr (FLOAT32_DTYPE) {
        reconfig_data_format(cb_in, cb_ex_global);
    }
    index_h_offset = 0;
    reconfig_data_format_srca(cb_ex_external, cb_in);
    sub_bcast_cols_init_short(cb_in, cb_ex_global);
    cb_xmm_obj.reserve_back(num_tiles_per_block);
    for (uint32_t i = 0; i < block_h; i++) {
        index_subblock_w_offset = 0;
        cb_ex_global_obj.wait_front(1);
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset;
                sub_tiles_bcast_cols(cb_in, cb_ex_global, index, 0, w);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < subblock_w; i++) {
                pack_tile(i, cb_xmm);
            }
            tile_regs_release();
            index_subblock_w_offset += subblock_w;
        }
        cb_ex_global_obj.pop_front(1);
        cb_in_obj.pop_front(block_w);
    }
    cb_xmm_obj.push_back(num_tiles_per_block);
#ifndef FUSE_PRE_ADD
    reconfig_data_format_srca(cb_in, cb_xmm);
#endif
    cb_xmm_obj.wait_front(num_tiles_per_block);
#endif

    // (x - E[x])^2, cb_mm2 <-- cb_xmm
    mul_tiles_init(cb_xmm, cb_xmm);
    index_h_offset = 0;
    cb_xmm2_obj.reserve_back(num_tiles_per_block);
    for (uint32_t i = 0; i < block_h; i++) {
        index_subblock_w_offset = 0;
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset + index_h_offset;
                mul_tiles(cb_xmm, cb_xmm, index, index, w);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < subblock_w; i++) {
                pack_tile(i, cb_xmm2);
            }
            tile_regs_release();
            index_subblock_w_offset += subblock_w;
        }
        index_h_offset += block_w;
    }
    cb_xmm2_obj.push_back(num_tiles_per_block);

#if defined RMSNORM and not defined FUSED_PRE_ADD
    reconfig_data_format(cb_xmm, cb_xmm2, cb_xmm, cb_scaler);
#else
    if constexpr (FLOAT32_DTYPE) {
        reconfig_data_format(cb_xmm, cb_xmm2, cb_xmm, cb_scaler);
    }
#endif

    cb_xmm2_obj.wait_front(num_tiles_per_block);

// Var(x)
#ifdef RMSNORM
    cb_scaler_obj.wait_front(1);
#endif
    cb_ex_partial2_obj.reserve_back(block_h);
    reduce_init<REDUCE_OP, REDUCE_DIM, FLOAT32_REDUCTION>(cb_xmm2, cb_scaler, cb_ex_partial2);
    index_h_offset = 0;
    for (uint32_t i = 0; i < block_h; i++) {
        tile_regs_acquire();
        for (uint32_t w = 0; w < num_reduce_tiles_per_block_h; w++) {
            reduce_tile<REDUCE_OP, REDUCE_DIM, FLOAT32_REDUCTION>(
                cb_xmm2, cb_scaler, w + index_h_offset, scaler0, dst0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_ex_partial2);
        tile_regs_release();
        index_h_offset += block_w;
    }
    reduce_uninit();
    cb_xmm2_obj.pop_front(num_tiles_per_block);
    cb_ex_partial2_obj.push_back(block_h);

    // global reduce, cb_ex <-- cb_ex_external, cb_ex_partial
    if constexpr (is_allgather_worker) {
        reduce_init<REDUCE_OP, REDUCE_DIM, FLOAT32_REDUCTION>(cb_ex_external2, cb_scaler_global, cb_ex2);
        cb_ex2_obj.reserve_back(num_tiles_per_allgather_worker);

        for (uint32_t i = 0; i < num_tiles_per_allgather_worker; i++) {
            cb_scaler_global_obj.wait_front(1);

            tile_regs_acquire();
            for (uint32_t w = 0; w < num_blocks_reduce; w++) {
                cb_ex_external2_obj.wait_front(1);
                reduce_tile<REDUCE_OP, REDUCE_DIM, FLOAT32_REDUCTION>(
                    cb_ex_external2, cb_scaler_global, 0, scaler0, dst0);
                cb_ex_external2_obj.pop_front(1);
            }
            if (use_two_stage_reduce && !is_second_stage_reader) {
                cb_ex_external2_obj.wait_front(num_blocks_second_stage - 1);
                cb_ex_external2_obj.pop_front(num_blocks_second_stage - 1);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, cb_ex2);
            tile_regs_release();
        }
        reduce_uninit();
        cb_ex2_obj.push_back(num_tiles_per_allgather_worker);

        if (enable_sqrt) {
            for (uint32_t i = 0; i < num_tiles_per_allgather_worker; i++) {
                // 1/[sqrt(Var + eps)],
                cb_ex2_obj.wait_front(1);
                cb_ex2pe_obj.reserve_back(1);
                tile_regs_acquire();
                add_tiles_init(cb_ex2, cb_eps);
                add_tiles(cb_ex2, cb_eps, i, 0, dst0);
                tile_regs_wait();
                rsqrt_tile_init<LEGACY_RSQRT>();
                rsqrt_tile<LEGACY_RSQRT>(dst0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(dst0, cb_ex2pe);
                cb_ex2pe_obj.push_back(1);
                tile_regs_release();
            }
        }
    }

    if constexpr (do_gamma == 0 && do_beta == 0) {
        pack_reconfig_data_format(cb_out);
    }
// (x - Ex) * 1/[sqrt(Var + eps)]
#if defined RMSNORM and not defined FUSE_PRE_ADD
    if constexpr (FLOAT32_DTYPE) {
        reconfig_data_format(cb_xmm, cb_ex_global);
    } else {
        reconfig_data_format_srca(cb_ex2, cb_xmm);
    }
#else
    if constexpr (FLOAT32_DTYPE) {
        reconfig_data_format(cb_xmm, cb_ex_global);
    }
#endif
    mul_bcast_cols_init_short(cb_xmm, cb_ex_global);
    index_h_offset = 0;
    cb_im_obj.reserve_back(num_tiles_per_block);
    for (uint32_t i = 0; i < block_h; i++) {
        index_subblock_w_offset = 0;
        cb_ex_global_obj.wait_front(1);
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset + index_h_offset;
                mul_tiles_bcast_cols(cb_xmm, cb_ex_global, index, 0, w);

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
                pack_tile(i, cb_im);
            }
            tile_regs_release();

            index_subblock_w_offset += subblock_w;
        }
        index_h_offset += block_w;
        cb_ex_global_obj.pop_front(1);
    }
    cb_im_obj.push_back(num_tiles_per_block);

    cb_xmm_obj.pop_front(num_tiles_per_block);
    cb_im_obj.wait_front(num_tiles_per_block);

    if constexpr (do_gamma) {
        reconfig_data_format(cb_im, cb_gamma);
        if constexpr (do_beta == 0) {
            pack_reconfig_data_format(cb_out);
        }
        mul_bcast_rows_init_short(cb_im, cb_gamma);
        cb_gamma_obj.wait_front(block_w);
        index_h_offset = 0;
        cb_outgamma_obj.reserve_back(num_tiles_per_block);
        for (uint32_t i = 0; i < block_h; i++) {
            index_subblock_w_offset = 0;
            for (uint32_t j = 0; j < num_subblocks_w; j++) {
                tile_regs_acquire();
                for (uint32_t w = 0; w < subblock_w; w++) {
                    index = w + index_subblock_w_offset;
                    mul_tiles_bcast_rows(cb_im, cb_gamma, index + index_h_offset, index, w);
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
                    pack_tile(i, cb_outgamma);
                }
                tile_regs_release();
                index_subblock_w_offset += subblock_w;
            }
            index_h_offset += block_w;
        }
        cb_outgamma_obj.push_back(num_tiles_per_block);
        cb_im_obj.pop_front(num_tiles_per_block);
        cb_outgamma_obj.wait_front(num_tiles_per_block);
    }

    if constexpr (do_beta) {
        reconfig_data_format(cb_fusion, cb_beta);
        pack_reconfig_data_format(cb_out);
        add_bcast_rows_init_short(cb_fusion, cb_beta);
        cb_beta_obj.wait_front(block_w);
        index_h_offset = 0;
        cb_out_obj.reserve_back(num_tiles_per_block);
        for (uint32_t i = 0; i < block_h; i++) {
            index_subblock_w_offset = 0;
            for (uint32_t j = 0; j < num_subblocks_w; j++) {
                tile_regs_acquire();
                for (uint32_t w = 0; w < subblock_w; w++) {
                    index = w + index_subblock_w_offset;
                    add_tiles_bcast_rows(cb_fusion, cb_beta, index + index_h_offset, index, w);
#ifdef SFPU_OP_INIT_ACTIVATION
                    SFPU_OP_INIT_ACTIVATION
                    SFPU_OP_FUNC_ACTIVATION
#endif
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < subblock_w; i++) {
                    pack_tile(i, cb_out);
                }
                tile_regs_release();
                index_subblock_w_offset += subblock_w;
            }
            index_h_offset += block_w;
        }
        cb_out_obj.push_back(num_tiles_per_block);
        cb_fusion_obj.pop_front(num_tiles_per_block);
        cb_out_obj.wait_front(num_tiles_per_block);
    }
}
