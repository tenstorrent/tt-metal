// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/softmax.h"
#include "api/compute/reduce.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"

namespace ckl = compute_kernel_lib;

#if defined(BH_ACC)
#if defined(BH_SUM_FIDELITY)
#define BH_ACC_REDUCE(CB, I)                                                                                    \
    do {                                                                                                     \
        UNPACK((llk_unpack_AB_reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB, dfb::sum_scaler, I, 0)));     \
        MATH((llk_math_reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, DST_ACCUM_MODE,                          \
                              static_cast<MathFidelity>(BH_SUM_FIDELITY)>(CB, dfb::sum_scaler, 0)));               \
    } while (0)
#else
#define BH_ACC_REDUCE(CB, I) reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB, dfb::sum_scaler, I, 0, 0)
#endif
#endif

void kernel_main() {
    const std::uint32_t T = get_arg(args::T);
    const std::uint32_t c = get_arg(args::c);
    const std::uint32_t C = get_arg(args::C);
    const std::uint32_t num_rows = get_arg(args::num_rows);
    const std::uint32_t ndst = get_arg(args::blk);
    if (T == 0 || num_rows == 0) {
        return;
    }
    static constexpr auto dfb_in0 = dfb::in0;
    static constexpr auto dfb_max_scaler = dfb::max_scaler;
    static constexpr auto dfb_sum_scaler = dfb::sum_scaler;
    static constexpr auto dfb_exps = dfb::exps;
    [[maybe_unused]] static constexpr auto dfb_pmax = dfb::pmax;
    [[maybe_unused]] static constexpr auto dfb_mcb = dfb::mcb;
    static constexpr auto dfb_out0 = dfb::out0;
    static constexpr auto dfb_rcb = dfb::rcb;
    DataflowBuffer in0(dfb_in0);
    DataflowBuffer max_scaler(dfb_max_scaler);
    DataflowBuffer sum_scaler(dfb_sum_scaler);
    DataflowBuffer exps(dfb_exps);
    DataflowBuffer pmax(dfb_pmax);
    DataflowBuffer rmax(dfb::rmax);
    DataflowBuffer mout(dfb::mout);
    DataflowBuffer mcb(dfb_mcb);
    DataflowBuffer pacc(dfb::pacc);
    DataflowBuffer racc(dfb::racc);
    DataflowBuffer rout(dfb::rout);
    DataflowBuffer rcb(dfb_rcb);
    DataflowBuffer out0(dfb_out0);
#if defined(BH_ACC)
    DataflowBuffer exs(dfb::exs);
#endif

    compute_kernel_hw_startup(dfb_in0, dfb_max_scaler, dfb_exps);
    max_scaler.wait_front(1);
    sum_scaler.wait_front(1);
#if defined(BH_ACC)
    [[maybe_unused]] const bool sm_send = c > 0;
#endif
#if defined(BH_ACC)
    const bool is_last = c == 0;
#else
    const bool is_last = c + 1 == C;
#endif
    const std::uint32_t sum_out =
        is_last ? static_cast<std::uint32_t>(dfb::rout) : static_cast<std::uint32_t>(dfb::pacc);

    for (std::uint32_t r = 0; r < num_rows; ++r) {
#if defined(NUMERIC_STABLE)
        {
        reconfig_data_format(dfb_max_scaler, dfb_in0);
        pack_reconfig_data_format(dfb_pmax);
        reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(dfb_in0, dfb_max_scaler, dfb_pmax);
        tile_regs_acquire();
        for (std::uint32_t t = 0; t < T; ++t) {
            in0.wait_front(static_cast<uint16_t>(t + 1));
            reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(dfb_in0, dfb_max_scaler, t, 0, 0);
        }
        pmax.reserve_back(1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb_pmax);
        tile_regs_release();
        pmax.push_back(1);
        reduce_uninit();
        }

        if (c == 0) {
            rmax.wait_front(static_cast<uint16_t>(C));
            reconfig_data_format_srca(dfb::rmax);
            pack_reconfig_data_format(dfb::mout);
            tile_regs_acquire();
            copy_init(dfb::rmax);
            copy_tile(dfb::rmax, 0, 0);
            for (std::uint32_t k = 1; k < C && k < 4; ++k) {
                copy_tile(dfb::rmax, k, k);
            }
            binary_max_tile_init();
            for (std::uint32_t i = 1; i < C; ++i) {
                binary_max_tile(0, 1 + (i - 1) % 3, 0);
                if (i % 3 == 0 && i + 1 < C) {
                    copy_init(dfb::rmax);
                    for (std::uint32_t k = i + 1; k < C && k < i + 4; ++k) {
                        copy_tile(dfb::rmax, k, 1 + (k - 1) % 3);
                    }
                    binary_max_tile_init();
                }
            }
            tile_regs_commit();
            mout.reserve_back(1);
            tile_regs_wait();
            pack_tile(0, dfb::mout);
            tile_regs_release();
            mout.push_back(1);
            rmax.pop_front(static_cast<uint16_t>(C));
        }

        {
#if defined(BH_ACC)
        reconfig_data_format_srca(dfb::in0);
        pack_reconfig_data_format(dfb::exps);
        reconfig_data_format_srcb(dfb::mcb);
        in0.wait_front(static_cast<uint16_t>(T));
        mcb.wait_front(1);
        sub_bcast_cols_init(dfb::in0, dfb::mcb);
        exp_tile_init<EXP_APPROX>();
        for (std::uint32_t b = 0; b < T; b += ndst) {
            const std::uint32_t rem = (b + ndst > T) ? (T - b) : ndst;
            exps.reserve_back(static_cast<uint16_t>(rem));
#if defined(BH_ACC)
            if (sm_send) {
                exs.reserve_back(static_cast<uint16_t>(rem));
            }
#endif
            tile_regs_acquire();
            for (std::uint32_t d = 0; d < rem; ++d) {
                sub_tiles_bcast_cols(dfb::in0, dfb::mcb, b + d, 0, d);
            }
            for (std::uint32_t d = 0; d < rem; ++d) {
                exp_tile<EXP_APPROX>(d);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (std::uint32_t d = 0; d < rem; ++d) {
                pack_tile(d, dfb::exps);
            }
#if defined(BH_ACC)
            if (sm_send) {
                for (std::uint32_t d = 0; d < rem; ++d) {
                    pack_tile(d, dfb::exs);
                }
            }
#endif
            tile_regs_release();
            exps.push_back(static_cast<uint16_t>(rem));
#if defined(BH_ACC)
            if (sm_send) {
                exs.push_back(static_cast<uint16_t>(rem));
            }
#endif
        }
        in0.pop_front(static_cast<uint16_t>(T));
        mcb.pop_front(1);
#else
        reconfig_data_format(dfb_in0, dfb_mcb);
        pack_reconfig_data_format(dfb_exps);
        ckl::eltwise_chain(
            ckl::IterationShape::tiles(T).block_size(ndst),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Sub,
                ckl::input(
                    dfb_in0,
                    ckl::WaitPolicy::Upfront,
                    ckl::PopPolicy::AtEnd,
                    ckl::InputTileMapping::Block,
                    ckl::DataFormatReconfig::Disabled),
                ckl::input(dfb_mcb, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd)>{in0, mcb},
            ckl::Exp<static_cast<ckl::Approx>(EXP_APPROX), ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(
                dfb_exps,
                ckl::ReservePolicy::PerBlockSize,
                ckl::PushPolicy::PerBlockSize,
                ckl::DataFormatReconfig::Disabled)>{exps});
#endif
        }
#else
        reconfig_data_format(dfb_in0, dfb_in0);
        pack_reconfig_data_format(dfb_exps);
        copy_init(dfb_in0);
        exp_tile_init<EXP_APPROX>();
        ckl::unary<
            ckl::Exp<static_cast<ckl::Approx>(EXP_APPROX), ckl::Dst::D0>,
            ckl::input(dfb_in0, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::InputTileMapping::Block),
            ckl::output(
                dfb_exps,
                ckl::ReservePolicy::PerBlockSize,
                ckl::PushPolicy::PerBlockSize,
                ckl::DataFormatReconfig::Disabled)>(ckl::IterationShape::tiles(T).block_size(ndst), in0, exps);
#endif
        exps.wait_front(static_cast<uint16_t>(T));

#if defined(BH_ACC)
        if (c == 0) {
            DataflowBuffer exr(dfb::exr);
            const std::uint32_t sm_wt = get_arg(args::Wt);
            reconfig_data_format(dfb_sum_scaler, dfb_exps);
            pack_reconfig_data_format(dfb::rout);
            rout.reserve_back(1);
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(dfb_exps, dfb_sum_scaler, sum_out);
            tile_regs_acquire();
#if defined(BH_SUM_FIDELITY)
            MATH((llk_math_reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW, DST_ACCUM_MODE, static_cast<MathFidelity>(BH_SUM_FIDELITY)>(
                dfb_exps, dfb_sum_scaler)));
#endif
            for (std::uint32_t t = 0; t < T; ++t) {
                BH_ACC_REDUCE(dfb_exps, t);
            }
            for (std::uint32_t b = T; b < sm_wt; b += ndst) {
                const std::uint32_t rem = (b + ndst > sm_wt) ? (sm_wt - b) : ndst;
                exr.wait_front(static_cast<uint16_t>(b + rem));
                for (std::uint32_t d = 0; d < rem; ++d) {
                    BH_ACC_REDUCE(dfb::exr, b + d);
                }
            }
            if constexpr (DST_ACCUM_MODE) {
                recip_tile_init<ReciprocalDestAcc::FP32, ReciprocalApproxMode::Precise>();
                recip_tile<ReciprocalDestAcc::FP32, ReciprocalApproxMode::Precise>(0);
            } else {
                recip_tile_init();
                recip_tile(0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dfb::rout);
            tile_regs_release();
            rout.push_back(1);
            reduce_uninit();
            exr.pop_front(static_cast<uint16_t>(sm_wt));
        }
#else
        reconfig_data_format(dfb_sum_scaler, dfb_exps);
        const bool sm_chain = c > 0;
        if (sm_chain) {
            reconfig_data_format_srca(dfb::racc);
            copy_init(dfb::racc);
        }
        if (sm_chain) {
            racc.wait_front(1);
        }
        {
        if (!sm_chain) {
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(dfb_exps, dfb_sum_scaler, sum_out);
        }
        tile_regs_acquire();
        if (sm_chain) {
            copy_tile(dfb::racc, 0, 0);
            reconfig_data_format(dfb_sum_scaler, dfb_exps);
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(dfb_exps, dfb_sum_scaler, sum_out);
        }
#if defined(BH_SUM_FIDELITY)
        MATH((llk_math_reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW, DST_ACCUM_MODE, static_cast<MathFidelity>(BH_SUM_FIDELITY)>(
            dfb_exps, dfb_sum_scaler)));
#endif
        for (std::uint32_t t = 0; t < T; ++t) {
#if defined(BH_SUM_FIDELITY)
            UNPACK((llk_unpack_AB_reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(dfb_exps, dfb_sum_scaler, t, 0)));
            MATH((llk_math_reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, DST_ACCUM_MODE, static_cast<MathFidelity>(BH_SUM_FIDELITY)>(
                dfb_exps, dfb_sum_scaler, 0)));
#else
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(dfb_exps, dfb_sum_scaler, t, 0, 0);
#endif
        }
        if (is_last) {
            if constexpr (DST_ACCUM_MODE) {
                recip_tile_init<ReciprocalDestAcc::FP32, ReciprocalApproxMode::Precise>();
                recip_tile<ReciprocalDestAcc::FP32, ReciprocalApproxMode::Precise>(0);
            } else {
                recip_tile_init();
                recip_tile(0);
            }
            pack_reconfig_data_format(dfb::rout);
            rout.reserve_back(1);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dfb::rout);
            tile_regs_release();
            rout.push_back(1);
            reduce_uninit();
        } else {
            pack_reconfig_data_format(dfb::pacc);
            pacc.reserve_back(1);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dfb::pacc);
            tile_regs_release();
            pacc.push_back(1);
            reduce_uninit();
        }
        }
        if (sm_chain) {
            racc.pop_front(1);
        }
#endif

        ckl::mul<
            ckl::input(dfb_exps, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block),
            ckl::input(dfb_rcb, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd),
            ckl::output(dfb_out0, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
            ckl::IterationShape::tiles(T).block_size(ndst), exps, rcb, out0);
    }
}
