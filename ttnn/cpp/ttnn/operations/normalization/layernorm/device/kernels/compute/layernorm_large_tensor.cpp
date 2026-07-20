// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/compute_kernel_api.h"
#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/operations/normalization/kernel_util/compute/numeric.h"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "api/dataflow/circular_buffer.h"

#include "layernorm_compute_utils.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_bcast.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"  // Square
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"

namespace ckl = compute_kernel_lib;

namespace kutil = norm::kernel_util;
namespace numeric = kutil::compute::numeric;
namespace policies = kutil::compute::policies;
namespace generic = kutil::generic;

void kernel_main() {
    uint32_t NCHt = get_arg_val<uint32_t>(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t block_size = get_compile_time_arg_val(1);
    constexpr uint32_t do_gamma = get_compile_time_arg_val(2);
    constexpr uint32_t do_beta = get_compile_time_arg_val(3);
    constexpr bool FLOAT32_DTYPE = get_compile_time_arg_val(4) == 1;
    constexpr bool FLOAT32_REDUCTION = get_compile_time_arg_val(5) == 1;
    constexpr bool LEGACY_RSQRT = get_compile_time_arg_val(6) == 1;
    constexpr uint32_t W = get_compile_time_arg_val(7);
    constexpr uint32_t tile_width = get_compile_time_arg_val(8);

    constexpr uint32_t onetile = 1;

    // CB indices - configurable via named compile-time args for kernel chaining support
    constexpr auto cb_scaler = get_named_compile_time_arg_val("cb_scaler");
    constexpr auto cb_eps = get_named_compile_time_arg_val("cb_eps");
    constexpr auto cb_in = get_named_compile_time_arg_val("cb_in");
    constexpr auto cb_inb = get_named_compile_time_arg_val("cb_inb");
    constexpr auto cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr auto cb_gamma = get_named_compile_time_arg_val("cb_gamma");
    constexpr auto cb_beta = get_named_compile_time_arg_val("cb_beta");
    constexpr uint32_t cb_xmm = get_named_compile_time_arg_val("cb_xmm");
    constexpr auto cb_ex = get_named_compile_time_arg_val("cb_ex");
    constexpr auto cb_ex2 = get_named_compile_time_arg_val("cb_ex2");
    constexpr auto cb_xmm2 = get_named_compile_time_arg_val("cb_xmm2");
    constexpr auto cb_ex2pe = get_named_compile_time_arg_val("cb_ex2pe");
    uint32_t cb_fusion = get_named_compile_time_arg_val("cb_fusion");  // stream gamma/beta
    constexpr auto scaler0 = 0;
    constexpr auto cb_accumulate = get_named_compile_time_arg_val("cb_accumulate");

    constexpr auto cb_in_rm =
        get_named_compile_time_arg_val("cb_in_rm");  // input row-major (if row-major input, otherwise unused)

#ifdef RMSNORM
    constexpr bool is_rmsnorm = true;
#else
    constexpr bool is_rmsnorm = false;
#endif
#ifdef FUSE_PRE_ADD
    constexpr bool do_fuse_pre_add = true;
#else
    constexpr bool do_fuse_pre_add = false;
#endif

#ifdef FUSE_PRE_ADD
#ifdef RMSNORM
    constexpr uint32_t cb_x = cb_xmm;
#else
    constexpr uint32_t cb_x = get_named_compile_time_arg_val("cb_x");
#endif
#else
    constexpr uint32_t cb_x = cb_in;
#endif

    CircularBuffer cb_eps_obj(cb_eps);
    CircularBuffer cb_scaler_obj(cb_scaler);
    CircularBuffer cb_in_obj(cb_in);
    CircularBuffer cb_in_rm_obj(cb_in_rm);
    CircularBuffer cb_inb_obj(cb_inb);
    CircularBuffer cb_out_obj(cb_out);
    CircularBuffer cb_gamma_obj(cb_gamma);
    CircularBuffer cb_beta_obj(cb_beta);
    CircularBuffer cb_xmm_obj(cb_xmm);
    CircularBuffer cb_ex_obj(cb_ex);
    CircularBuffer cb_ex2_obj(cb_ex2);
    CircularBuffer cb_xmm2_obj(cb_xmm2);
    CircularBuffer cb_ex2pe_obj(cb_ex2pe);
    CircularBuffer cb_accumulate_obj(cb_accumulate);

#ifdef FUSE_PRE_ADD
    binary_op_init_common(cb_in, cb_inb, cb_x);
#else
    // Always call binary_op_init_common regardless of TILIZE_IN.
    // This initializes llk_pack_dest_init, which sets up the MATH-PACK DST semaphore
    // in the "available for MATH" state.  Without it, the first tilize_block call's
    // internal llk_math_wait_for_dest_available() spins forever (deadlock).
#ifdef RMSNORM
    binary_op_init_common(cb_in, cb_scaler, cb_xmm2);
#else
    binary_op_init_common(cb_in, cb_scaler, cb_ex);
#endif
#endif
    cb_eps_obj.wait_front(1);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        constexpr int onetile = 1;
        constexpr int dst0 = 0;
#ifndef RMSNORM
        // Start of
        //  E[x]
        //  aka   ∑(x)
        //      --------
        //         n
#ifdef FUSE_PRE_ADD
        numeric::row_wise_mean_with_pre_add<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            FLOAT32_REDUCTION,
            policies::FullBlockWithPopPolicy>(
            cb_in_obj, cb_inb_obj, cb_scaler_obj, cb_ex_obj, W, Wt, block_size, tile_width);
#else
        numeric::
            row_wise_mean<PoolType::SUM, ReduceDim::REDUCE_ROW, FLOAT32_REDUCTION, policies::FullBlockWithPopPolicy>(
                cb_in_obj, cb_scaler_obj, cb_ex_obj, W, Wt, block_size, tile_width);
#endif
#endif  // !RMS ifdef end
        // Start of
        // Var Calculation
        // Var(X) = ∑(x-E[x])^2
        //         -----------
        //              n
        const bool last_tile_is_partial = W % tile_width > 0;
        for (auto block : generic::blocks(Wt, block_size)) {
#ifdef TILIZE_IN
            tilize_row_major_block(cb_in_rm_obj, cb_in_obj, block_size, block);
#ifdef RMSNORM
            binary_op_init_common(cb_in, cb_scaler, cb_xmm2);
#else
            binary_op_init_common(cb_in, cb_scaler, cb_ex);
#endif
#endif
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(block.full_block_size(), block.full_block_size()),
                ckl::OptionalChainElement<
                    is_rmsnorm,  // RMSNORM: copy x (no mean subtraction)
                    ckl::CopyTile<
                        cb_in,
                        ckl::Dst::D0,
                        ckl::input(ckl::InputLifecycle::Bulk, ckl::OperandKind::Block)>>{},
                ckl::OptionalChainElement<
                    !is_rmsnorm,  // LayerNorm: x - E[x] (reads cb_ex; stripped under RMSNORM)
                    ckl::BinaryFpu<
                        cb_in,
                        cb_ex,
                        ckl::BinaryFpuOp::Sub,
                        ckl::BroadcastDim::Col,
                        ckl::input(ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
                        ckl::input(ckl::InputLifecycle::CallerManaged)>>{},
                ckl::OptionalChainElement<
                    do_fuse_pre_add,  // FUSE_PRE_ADD: + b (DEST-reuse), else stripped
                    ckl::DestReuseBinary<
                        cb_inb,
                        ckl::BinaryFpuOp::Add,
                        ckl::DestReuseType::DEST_TO_SRCB,
                        ckl::input(ckl::InputLifecycle::Bulk, ckl::OperandKind::Block)>>{},
                ckl::Square<ckl::Dst::D0>{},
                ckl::PackTile<cb_xmm2, ckl::output(ckl::OutputLifecycle::Bulk)>{});

            tile_regs_acquire();
            if (!block.is_first()) {
                cb_accumulate_obj.wait_front(onetile);
                reconfig_data_format_srca(cb_accumulate);
                copy_tile_init(cb_accumulate);
                copy_tile(cb_accumulate, 0, dst0);
                cb_accumulate_obj.pop_front(onetile);
            }
            cb_xmm2_obj.wait_front(block.full_block_size());

            // Accumulate (x-E[x])^2
            // main #47311 dropped the FLOAT32_REDUCTION template arg (fp32 now via FP32_DEST_ACC_EN)
            // and uses the scaler-first reconfig convention documented in reduce.h.
            reconfig_data_format(cb_scaler, cb_xmm2);
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_xmm2, cb_scaler, cb_accumulate);
            for (auto i : block.local()) {
                const auto scaler_tile_idx = block.to_global(i) == Wt - 1 && last_tile_is_partial ? 1 : 0;
                reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_xmm2, cb_scaler, i, scaler_tile_idx, dst0);
            }

            cb_pop_front(cb_xmm2, block.full_block_size());

            const auto final_iter = block.last() == Wt;
            const auto pack_cb = final_iter ? cb_ex2 : cb_accumulate;
            if (final_iter) {
                // Divide by W
                binop_with_scalar_tile_init();
                mul_unary_tile(dst0, generic::bit_cast<uint32_t>(1.0f / W));
            }

            reduce_uninit();
            tile_regs_commit();
            tile_regs_wait();

            CircularBuffer(pack_cb).reserve_back(onetile);
            pack_reconfig_data_format(pack_cb);
            pack_tile(dst0, pack_cb);
            tile_regs_release();
            CircularBuffer(pack_cb).push_back(onetile);
        }

        // End of
        // Var Calculation
        // Var(X) = ∑(x-E[x])^2
        //         -----------

        // Start of
        // Calculation
        //                     1
        //               √(Var(X) + ε)
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(onetile),
            ckl::BinaryFpu<
                cb_ex2,
                cb_eps,
                ckl::BinaryFpuOp::Add,
                ckl::BroadcastDim::None,
                ckl::input(),
                ckl::input(ckl::InputLifecycle::CallerManaged)>{},
            ckl::Rsqrt<ckl::Approx::Exact, LEGACY_RSQRT ? ckl::Legacy::On : ckl::Legacy::Off, ckl::Dst::D0>{},
            ckl::PackTile<cb_ex2pe, ckl::output(ckl::OutputLifecycle::ReserveNonePushEnd)>{});

        ckl::unary_bcast<ckl::BroadcastDim::Col, cb_ex2pe, cb_ex2pe>(ckl::EltwiseShape::tiles(onetile));
        cb_ex2pe_obj.wait_front(onetile);

        // End of
        // Calculation
        //                     1
        //               √(Var(X) + ε)

        // Start of
        // Final Val Calc
        //    x-E[X]
        //(---------------*𝛄)+ß
        //  √(Var(X)+ε)
        for (auto block : generic::blocks(Wt, block_size)) {
#ifdef TILIZE_IN
            // Reader supplies this second pass of data after the variance data.
            tilize_row_major_block(cb_in_rm_obj, cb_in_obj, block_size, block);

#ifdef RMSNORM
            binary_op_init_common(cb_in, cb_scaler, cb_xmm2);
#else
            binary_op_init_common(cb_in, cb_scaler, cb_ex);
#endif
#endif
#ifndef RMSNORM
            cb_ex_obj.wait_front(1);
#endif
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(block.full_block_size(), block.full_block_size()),
                ckl::OptionalChainElement<
                    is_rmsnorm,  // RMSNORM: copy x (no mean subtraction)
                    ckl::CopyTile<
                        cb_in,
                        ckl::Dst::D0,
                        ckl::input(ckl::InputLifecycle::Bulk, ckl::OperandKind::Block)>>{},
                ckl::OptionalChainElement<
                    !is_rmsnorm,  // LayerNorm: x - E[x] (reads cb_ex; stripped under RMSNORM)
                    ckl::BinaryFpu<
                        cb_in,
                        cb_ex,
                        ckl::BinaryFpuOp::Sub,
                        ckl::BroadcastDim::Col,
                        ckl::input(ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
                        ckl::input(ckl::InputLifecycle::CallerManaged)>>{},
                ckl::OptionalChainElement<
                    do_fuse_pre_add,  // FUSE_PRE_ADD: + b (DEST-reuse), else stripped
                    ckl::DestReuseBinary<
                        cb_inb,
                        ckl::BinaryFpuOp::Add,
                        ckl::DestReuseType::DEST_TO_SRCB,
                        ckl::input(ckl::InputLifecycle::Bulk, ckl::OperandKind::Block)>>{},
                ckl::PackTile<cb_xmm, ckl::output(ckl::OutputLifecycle::Bulk)>{});

            cb_xmm_obj.wait_front(block.full_block_size());
            reconfig_data_format(cb_xmm, cb_ex2pe);
            tile_regs_acquire();

            mul_tiles_init(cb_xmm, cb_ex2pe);
            for (auto i : block.local()) {
                mul_tiles(cb_xmm, cb_ex2pe, i, 0, i);
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

            if constexpr (!(do_gamma == 1 or do_beta == 1)) {
                cb_fusion = cb_out;
            }
            CircularBuffer(cb_fusion).reserve_back(block.full_block_size());
            pack_reconfig_data_format(cb_fusion);
            for (auto i : block.local()) {
                pack_tile(i, cb_fusion);
            }
            tile_regs_release();
            CircularBuffer(cb_fusion).push_back(block.full_block_size());
            cb_xmm_obj.pop_front(block.full_block_size());

            if constexpr (do_gamma == 1) {
                tile_regs_acquire();
                tile_regs_wait();
                reconfig_data_format(cb_fusion, cb_gamma);
                if constexpr (!do_beta) {
                    pack_reconfig_data_format(cb_out);
                }
                cb_gamma_obj.wait_front(block.full_block_size());
                CircularBuffer(cb_fusion).wait_front(block.full_block_size());
                mul_bcast_rows_init_short(cb_fusion, cb_gamma);
                for (auto i : block.local()) {
                    mul_tiles_bcast_rows(cb_fusion, cb_gamma, i, i, i);
#ifdef SFPU_OP_INIT_ACTIVATION
                    // Activation must be applied last. If do_beta != 0 then
                    // activation will be applied after the beta addition.
                    // Otherwise, we can apply the activation here.
                    if constexpr (!(do_beta == 1)) {
                        SFPU_OP_INIT_ACTIVATION
                        SFPU_OP_FUNC_ACTIVATION
                    }
#endif
                }
                tile_regs_commit();
                cb_gamma_obj.pop_front(block.full_block_size());
                CircularBuffer(cb_fusion).pop_front(block.full_block_size());
                if constexpr (!do_beta) {
                    cb_out_obj.reserve_back(block.full_block_size());
                    for (auto i : block.local()) {
                        pack_tile(i, cb_out);
                    }
                    cb_out_obj.push_back(block.full_block_size());
                } else {
                    CircularBuffer(cb_fusion).reserve_back(block.full_block_size());
                    for (auto i : block.local()) {
                        pack_tile(i, cb_fusion);
                    }
                    CircularBuffer(cb_fusion).push_back(block.full_block_size());
                }

                tile_regs_release();
            }
            if constexpr (do_beta == 1) {
                tile_regs_acquire();
                tile_regs_wait();
                reconfig_data_format(cb_fusion, cb_beta);
                pack_reconfig_data_format(cb_out);
                cb_beta_obj.wait_front(block.full_block_size());
                CircularBuffer(cb_fusion).wait_front(block.full_block_size());
                add_bcast_rows_init_short(cb_fusion, cb_beta);
                for (auto i : block.local()) {
                    add_tiles_bcast_rows(cb_fusion, cb_beta, i, i, i);
#ifdef SFPU_OP_INIT_ACTIVATION
                    SFPU_OP_INIT_ACTIVATION
                    SFPU_OP_FUNC_ACTIVATION
#endif
                }
                tile_regs_commit();
                cb_beta_obj.pop_front(block.full_block_size());
                CircularBuffer(cb_fusion).pop_front(block.full_block_size());
                cb_out_obj.reserve_back(block.full_block_size());
                for (auto i : block.local()) {
                    pack_tile(i, cb_out);
                }
                tile_regs_release();
                cb_out_obj.push_back(block.full_block_size());
            }

#ifdef UNTILIZE_OUT
            constexpr auto cb_out_rm = get_named_compile_time_arg_val("cb_out_rm");
            CircularBuffer cb_out_rm_obj(cb_out_rm);
            untilize_row_major_block<decltype(block), block_size>(cb_out_obj, cb_out_rm_obj, block);
#endif
        }  // block loop
        // End of
        // Final Val Calc
        //    x-E[X]
        //(---------------*𝛄)+ß
        //  √(Var(X)+ε)
#ifndef RMSNORM
        cb_ex_obj.pop_front(onetile);
#endif
        cb_ex2pe_obj.pop_front(onetile);
    }  // NCHt loop
}
