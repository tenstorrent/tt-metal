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

void kernel_main() {
#ifndef IDLE_CORE

    constexpr auto num_blocks_first_stage = get_arg(args::num_blocks_first_stage);
    constexpr auto block_w = get_arg(args::block_w);
    constexpr auto block_h_const = get_arg(args::block_h);
    const volatile uint32_t block_h_volatile = get_arg(args::block_h);
    constexpr auto subblock_w_const = get_arg(args::subblock_w);
    const volatile uint32_t subblock_w_volatile = get_arg(args::subblock_w);
    constexpr auto num_subblocks_w = get_arg(args::num_subblocks_w);
    constexpr auto num_tiles_per_block = get_arg(args::num_tiles_per_block);
    constexpr bool FLOAT32_DTYPE = get_arg(args::float32_dtype) == 1;
    constexpr bool FP32_DEST_ACC = compute_kernel_lib::get_fp32_dest_acc_enabled();
    constexpr auto num_blocks_second_stage = get_arg(args::num_blocks_second_stage);
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
#ifdef IS_ALLGATHER_WORKER
    constexpr bool is_allgather_worker = true;
#else
    constexpr bool is_allgather_worker = false;
#endif

    const uint32_t num_reduce_tiles_per_block_h = get_arg(args::num_reduce_tiles_per_block_h);
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

    const bool enable_sqrt = not(use_two_stage_reduce and not is_second_stage_reader);

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t scaler0 = 0;

    constexpr uint32_t dfb_in0 = dfb::in0;
    constexpr uint32_t dfb_scaler_id = dfb::scaler;
    constexpr uint32_t dfb_eps = dfb::eps;
    constexpr uint32_t dfb_scaler_global_id = dfb::scaler_global;
    constexpr uint32_t dfb_x = dfb::x;
    constexpr uint32_t dfb_in1 = dfb::in1;
#ifdef FUSE_GAMMA
    constexpr uint32_t dfb_gamma_id = dfb::gamma;
#endif
#ifdef FUSE_BETA
    constexpr uint32_t dfb_beta_id = dfb::beta;
#endif
    constexpr uint32_t dfb_xmm_id = dfb::xmm;
    constexpr uint32_t dfb_ex_partial2_id = dfb::ex_partial2;
#ifdef TWO_STAGE_REDUCE
    constexpr uint32_t dfb_ex2_id = dfb::ex2;
#endif
    constexpr uint32_t dfb_ex2_local_id = dfb::ex2_local;
    constexpr uint32_t dfb_ex_external2_id = dfb::ex_external2;
    constexpr uint32_t dfb_ex_global_id = dfb::ex_global;
    constexpr uint32_t dfb_xmm2_id = dfb_x;
    constexpr uint32_t dfb_ex2pe_id = dfb::ex2pe;
    constexpr uint32_t dfb_fusion_id = do_gamma ? dfb::xmm : dfb_x;
    constexpr uint32_t dfb_out_id = dfb::out;
#ifdef DO_COL_MASK
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
    DataflowBuffer dfb_ex_partial2(dfb_ex_partial2_id);
#ifdef TWO_STAGE_REDUCE
    const uint32_t dfb_var_id = enable_sqrt ? dfb_ex2_local_id : dfb_ex2_id;
#else
    constexpr uint32_t dfb_var_id = dfb_ex2_local_id;
#endif
    DataflowBuffer dfb_var(dfb_var_id);
    DataflowBuffer dfb_ex_external2(dfb_ex_external2_id);
    DataflowBuffer dfb_ex_global(dfb_ex_global_id);
    DataflowBuffer dfb_xmm2(dfb_xmm2_id);
    DataflowBuffer dfb_ex2pe(dfb_ex2pe_id);
    DataflowBuffer dfb_fusion(dfb_fusion_id);
    DataflowBuffer dfb_out(dfb_out_id);
    DataflowBuffer dfb_eps_obj(dfb_eps);

    compute_kernel_hw_startup(dfb_in0, dfb_in0, dfb_x);

#ifdef DO_COL_MASK
    dfb_col_mask_packed.wait_front(block_w);
#endif

    const uint32_t block_h = (block_w == 1) ? block_h_volatile : block_h_const;
    const uint32_t subblock_w = (block_w <= 2) ? subblock_w_volatile : subblock_w_const;

    uint32_t index_subblock_w_offset = 0;
    uint32_t index_h_offset = 0;
    uint32_t index = 0;

    constexpr uint32_t dfb_im_no_gamma_id = do_beta ? dfb_fusion_id : dfb_out_id;
    constexpr uint32_t dfb_im_id = do_gamma ? dfb_x : dfb_im_no_gamma_id;
    DataflowBuffer dfb_im(dfb_im_id);
    constexpr uint32_t dfb_outgamma_id = do_beta ? dfb_fusion_id : dfb_out_id;
    DataflowBuffer dfb_outgamma(dfb_outgamma_id);

    DataflowBuffer dfb_in0_obj(dfb_in0);
    DataflowBuffer dfb_in1_obj(dfb_in1);
    DataflowBuffer dfb_hout(dfb::hout);
    dfb_in0_obj.wait_front(num_tiles_per_block);
    dfb_in1_obj.wait_front(num_tiles_per_block);
    reconfig_data_format_srcb(dfb_in0, dfb_in1);
    add_init(dfb_in0, dfb_in1);
    pack_reconfig_data_format(dfb_xmm_id);
#ifdef ARCH_QUASAR
    pack_init(dfb_xmm_id);
#endif
    dfb_xmm.reserve_back(num_tiles_per_block);
    dfb_hout.reserve_back(num_tiles_per_block);
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
                pack_tile(i, dfb_xmm_id);
            }
            for (uint32_t i = 0; i < subblock_w; i++) {
                pack_tile(i, dfb::hout);
            }
            tile_regs_release();
            index_subblock_w_offset += subblock_w;
        }
        index_h_offset += block_w;
    }
    dfb_xmm.push_back(num_tiles_per_block);
    dfb_hout.push_back(num_tiles_per_block);
    dfb_in0_obj.pop_front(num_tiles_per_block);
    dfb_in1_obj.pop_front(num_tiles_per_block);
    pack_reconfig_data_format(dfb_x);
    reconfig_data_format(dfb_in0, dfb_xmm_id, dfb_in1, dfb_xmm_id);
    dfb_xmm.wait_front(num_tiles_per_block);

    mul_init(dfb_xmm_id, dfb_xmm_id);
    index_h_offset = 0;
#ifdef ARCH_QUASAR
    pack_init(dfb_xmm2_id);
#endif
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
            for (uint32_t dst_i = 0; dst_i < subblock_w; dst_i++) {
                pack_tile(dst_i, dfb_xmm2_id);
            }
            tile_regs_release();
            index_subblock_w_offset += subblock_w;
        }
        index_h_offset += block_w;
    }
    dfb_xmm2.push_back(num_tiles_per_block);

#ifdef DO_COL_MASK
    reconfig_data_format(dfb_xmm2_id, dfb_col_mask_packed_id);
    norm::kernel_util::compute::mask_block_in_place(dfb_xmm2, dfb_col_mask_packed_id, num_tiles_per_block, block_w);
    dfb_xmm2.wait_front(num_tiles_per_block);
#endif

    if constexpr (FLOAT32_DTYPE) {
        reconfig_data_format(dfb_xmm_id, dfb_xmm2_id, dfb_xmm_id, dfb_scaler_id);
    }

    dfb_xmm2.wait_front(num_tiles_per_block);

#ifdef ARCH_QUASAR
    pack_init(dfb_ex_partial2_id);
#endif
    compute_kernel_lib::reduce<
        PoolType::AVG,
        ReduceDim::REDUCE_ROW,
        dfb_xmm2_id,
        dfb_scaler_id,
        dfb_ex_partial2_id,
        compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT>(
        dfb_xmm2,
        dfb_scaler,
        dfb_ex_partial2,
        compute_kernel_lib::ReduceInputBlockShape::of(block_h, num_reduce_tiles_per_block_h, 1),
        compute_kernel_lib::ReduceInputMemoryLayout::with_row_stride(block_w));
    reconfig_data_format(dfb_xmm2_id, dfb_scaler_id);
    dfb_xmm2.pop_front(num_tiles_per_block);

    if constexpr (is_allgather_worker) {
        reconfig_data_format(dfb_scaler_global_id, dfb_ex_external2_id);
        reduce_init<PoolType::AVG, ReduceDim::REDUCE_ROW>(dfb_ex_external2_id, dfb_scaler_global_id, dfb_var_id);
#ifdef ARCH_QUASAR
        pack_init(dfb_var_id);
#endif
        dfb_var.reserve_back(static_cast<uint16_t>(num_tiles_per_allgather_worker));

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
                dummy_unpack(dfb_ex_external2_id);
                dfb_ex_external2.pop_front(static_cast<uint16_t>(num_blocks_second_stage - 1));
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, dfb_var_id);
            tile_regs_release();
        }
        reduce_uninit();
        dfb_var.push_back(static_cast<uint16_t>(num_tiles_per_allgather_worker));
        reconfig_data_format(dfb_xmm2_id, dfb_scaler_id);

        if (enable_sqrt) {
#ifdef ARCH_QUASAR
            pack_init(dfb_ex2pe_id);
#endif
            dfb_var.wait_front(static_cast<uint16_t>(num_tiles_per_allgather_worker));
            dfb_eps_obj.wait_front(1);
            for (uint32_t i = 0; i < num_tiles_per_allgather_worker; i++) {
                dfb_ex2pe.reserve_back(1);
                tile_regs_acquire();
                add_init(dfb_ex2_local_id, dfb_eps);
                add_tiles(dfb_ex2_local_id, dfb_eps, i, 0, dst0);
                tile_regs_wait();
                rsqrt_tile_init();
#ifdef ARCH_WORMHOLE
                rsqrt_tile(dst0, VectorMode::C);
#else
                rsqrt_tile(dst0);
#endif
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(dst0, dfb_ex2pe_id);
                dfb_ex2pe.push_back(1);
                tile_regs_release();
            }
            dfb_var.pop_front(static_cast<uint16_t>(num_tiles_per_allgather_worker));
        }
    }

    if constexpr (!do_gamma && !do_beta) {
        pack_reconfig_data_format(dfb_out_id);
    }
    if constexpr (FLOAT32_DTYPE) {
        reconfig_data_format(dfb_xmm_id, dfb_ex_global_id);
    }
    mul_bcast_cols_init(dfb_xmm_id, dfb_ex_global_id);
    index_h_offset = 0;
#ifdef ARCH_QUASAR
    pack_init(dfb_im_id);
#endif
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
                if constexpr (!do_gamma && !do_beta) {
                    SFPU_OP_INIT_ACTIVATION
                    SFPU_OP_FUNC_ACTIVATION
                }
#endif
            }
            tile_regs_commit();

            tile_regs_wait();
            for (uint32_t dst_i = 0; dst_i < subblock_w; dst_i++) {
                pack_tile(dst_i, dfb_im_id);
            }
            tile_regs_release();

            index_subblock_w_offset += subblock_w;
        }
        index_h_offset += block_w;
        dfb_ex_global.pop_front(1);
    }
    dfb_im.push_back(num_tiles_per_block);

    dfb_xmm.pop_front(num_tiles_per_block);

#ifdef FUSE_GAMMA
    {
        dfb_im.wait_front(num_tiles_per_block);
        reconfig_data_format(dfb_im_id, dfb_gamma_id);
        if constexpr (!do_beta) {
            pack_reconfig_data_format(dfb_out_id);
        }
        mul_bcast_rows_init(dfb_im_id, dfb_gamma_id);
        dfb_gamma.wait_front(block_w);
        index_h_offset = 0;
#ifdef ARCH_QUASAR
        pack_init(dfb_outgamma_id);
#endif
        dfb_outgamma.reserve_back(num_tiles_per_block);
        for (uint32_t i = 0; i < block_h; i++) {
            index_subblock_w_offset = 0;
            for (uint32_t j = 0; j < num_subblocks_w; j++) {
                tile_regs_acquire();
                for (uint32_t w = 0; w < subblock_w; w++) {
                    index = w + index_subblock_w_offset;
                    mul_tiles_bcast_rows(dfb_im_id, dfb_gamma_id, index + index_h_offset, index, w);
#ifdef SFPU_OP_INIT_ACTIVATION
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
    }
#endif

#ifdef FUSE_BETA
    {
        dfb_fusion.wait_front(num_tiles_per_block);
        reconfig_data_format(dfb_fusion_id, dfb_beta_id);
        pack_reconfig_data_format(dfb_out_id);
        add_bcast_rows_init(dfb_fusion_id, dfb_beta_id);
        dfb_beta.wait_front(block_w);
        index_h_offset = 0;
#ifdef ARCH_QUASAR
        pack_init(dfb_out_id);
#endif
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
    }
#endif
#ifdef FUSE_GAMMA
    dfb_gamma.pop_front(block_w);
#endif
#ifdef FUSE_BETA
    dfb_beta.pop_front(block_w);
#endif
    dfb_scaler.pop_front(1);
    if constexpr (is_allgather_worker) {
        dfb_scaler_global.pop_front(1);
    }
#ifdef DO_COL_MASK
    dfb_col_mask_packed.pop_front(block_w);
#endif
    dfb_eps_obj.wait_front(1);
    dummy_unpack(dfb_eps);
    dfb_eps_obj.pop_front(1);
#ifdef OUT_SELF_LOOP
    dfb_out.wait_front(num_tiles_per_block);
    dummy_unpack(dfb_out_id);
    dfb_out.pop_front(num_tiles_per_block);
#endif

#endif
}
