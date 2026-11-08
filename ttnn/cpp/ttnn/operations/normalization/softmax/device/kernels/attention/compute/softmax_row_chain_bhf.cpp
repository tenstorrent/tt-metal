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

#ifdef BHF_PROF
#include "api/debug/kernel_profiler.hpp"
#define BHFZ(n) DeviceZoneScopedN(n)
#else
#define BHFZ(n)
#endif

namespace ckl = compute_kernel_lib;
#if defined(BHF_COMBCOL)
#define BHF_VM_COMB , VectorMode::C
#else
#define BHF_VM_COMB
#endif
#if defined(BHF_RECIPCOL)
#define BHF_VM_RECIP VectorMode::C
#else
#define BHF_VM_RECIP VectorMode::RC
#endif
#if defined(BHF_ADDCOL)
#define BHF_VM_ADD , VectorMode::C
#else
#define BHF_VM_ADD
#endif

#if defined(BHF_SUMLOFI)
#define BHF_SUMFID 0
#elif defined(BHF_SUMHIFI2)
#define BHF_SUMFID 2
#endif
#ifdef BHF_SUMFID
#define BHF_SUM_INIT(a, b, o)                                                                                        do {                                                                                                                reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(a, b, o);                                                     MATH((llk_math_reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW, DST_ACCUM_MODE,                                                           static_cast<ckernel::MathFidelity>(BHF_SUMFID)>(a, b)));                          } while (0)
#define BHF_SUM_TILE(a, b, t, d)                                                                                     do {                                                                                                                MATH((llk_math_reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, DST_ACCUM_MODE,                                                           static_cast<ckernel::MathFidelity>(BHF_SUMFID)>(a, b, d)));                                UNPACK((llk_unpack_AB_reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(a, b, t, 0)));                           } while (0)
#else
#define BHF_SUM_INIT(a, b, o) reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(a, b, o)
#define BHF_SUM_TILE(a, b, t, d) reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(a, b, t, 0, d)
#endif
#ifdef BHF_MAXLOFI
#define BHF_MAX_INIT(a, b, o)                                                                                        do {                                                                                                                reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(a, b, o);                                                     MATH((llk_math_reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW, DST_ACCUM_MODE,                                                           static_cast<ckernel::MathFidelity>(0)>(a, b)));                                  } while (0)
#define BHF_MAX_TILE(a, b, t, d)                                                                                     do {                                                                                                                MATH((llk_math_reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, DST_ACCUM_MODE,                                                           static_cast<ckernel::MathFidelity>(0)>(a, b, d)));                                        UNPACK((llk_unpack_AB_reduce<PoolType::MAX, ReduceDim::REDUCE_ROW>(a, b, t, 0)));                           } while (0)
#else
#define BHF_MAX_INIT(a, b, o) reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(a, b, o)
#define BHF_MAX_TILE(a, b, t, d) reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(a, b, t, 0, d)
#endif

void kernel_main() {
    const std::uint32_t T = get_arg(args::T);
    const std::uint32_t c = get_arg(args::c);
    const std::uint32_t C = get_arg(args::C);
    const std::uint32_t num_rows = get_arg(args::num_rows);
    const std::uint32_t ndst = get_arg(args::blk);
    [[maybe_unused]] const std::uint32_t th = get_arg(args::th);
    [[maybe_unused]] const std::uint32_t gm = get_arg(args::gm);
    [[maybe_unused]] const std::uint32_t gi = get_arg(args::gi);
    [[maybe_unused]] const std::uint32_t gt = get_arg(args::gt);
    [[maybe_unused]] const std::uint32_t am = get_arg(args::am);
    [[maybe_unused]] const std::uint32_t bi = get_arg(args::bi);
    [[maybe_unused]] const std::uint32_t nb = get_arg(args::nb);
    if ((T == 0 && am != 2) || num_rows == 0) {
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
    [[maybe_unused]] DataflowBuffer gexp(dfb::gexp);

    compute_kernel_hw_startup(dfb_in0, dfb_max_scaler, dfb_exps);
    max_scaler.wait_front(1);
    sum_scaler.wait_front(1);
#if defined(BHF_STREAM) && defined(BHF_BLOCKACC)
    if (am == 2) {
        BHFZ("A_SUM");
        reconfig_data_format(dfb_sum_scaler, dfb::gexp);
        if (bi == 0) {
            pack_reconfig_data_format(dfb_exps);
            BHF_SUM_INIT(dfb::gexp, dfb_sum_scaler, dfb_exps);
        } else {
            pack_reconfig_data_format(dfb::pacc);
            BHF_SUM_INIT(dfb::gexp, dfb_sum_scaler, dfb::pacc);
        }
        tile_regs_acquire();
        for (std::uint32_t t = 0; t < gt; ++t) {
            gexp.wait_front(static_cast<uint16_t>(t + 1));
            BHF_SUM_TILE(dfb::gexp, dfb_sum_scaler, t, 0);
        }
        tile_regs_commit();
        if (bi == 0) {
            exps.reserve_back(1);
            tile_regs_wait();
            pack_tile(0, dfb_exps);
            tile_regs_release();
            exps.push_back(1);
        } else {
            pacc.reserve_back(1);
            tile_regs_wait();
            pack_tile(0, dfb::pacc);
            tile_regs_release();
            pacc.push_back(1);
        }
        reduce_uninit();
        gexp.pop_front(static_cast<uint16_t>(gt));
        if (bi != 0) {
            return;
        }
        pack_reconfig_data_format(dfb_exps);
        for (std::uint32_t k = 1; k < nb; ++k) {
            racc.wait_front(static_cast<uint16_t>(k));
            tile_regs_acquire();
            reconfig_data_format_srca(dfb::racc);
            copy_init(dfb::racc);
            copy_tile(dfb::racc, k - 1, 0);
            exps.wait_front(1);
            reconfig_data_format_srca(dfb_exps);
            copy_init(dfb_exps);
            copy_tile(dfb_exps, 0, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            exps.pop_front(1);
            tile_regs_commit();
            exps.reserve_back(1);
            tile_regs_wait();
            pack_tile(0, dfb_exps);
            tile_regs_release();
            exps.push_back(1);
        }
        if (nb > 1) {
            racc.pop_front(static_cast<uint16_t>(nb - 1));
        }
        exps.wait_front(1);
        reconfig_data_format_srca(dfb_exps);
        pack_reconfig_data_format(dfb::rout);
        tile_regs_acquire();
        copy_init(dfb_exps);
        copy_tile(dfb_exps, 0, 0);
        exps.pop_front(1);
        if constexpr (DST_ACCUM_MODE) {
            recip_tile_init<ReciprocalDestAcc::FP32, ReciprocalApproxMode::Precise>();
            recip_tile<ReciprocalDestAcc::FP32, ReciprocalApproxMode::Precise>(0);
        } else {
            recip_tile_init();
            recip_tile(0);
        }
        tile_regs_commit();
        tile_regs_wait();
        rout.reserve_back(1);
        pack_tile(0, dfb::rout);
        rout.push_back(1);
        tile_regs_release();
        return;
    }
#elif defined(BHF_STREAM)
    if (am == 2) {
        BHFZ("A_SUM");
#ifdef BHF_ACCSEQ
        const std::uint32_t tb = gi == 0 ? 0u : th;
        const std::uint32_t te = gi == 0 ? th : gt;
        const std::uint32_t sout = gi == 0 ? static_cast<std::uint32_t>(dfb::pacc) : static_cast<std::uint32_t>(dfb::rout);
        if (gi != 0) {
            reconfig_data_format_srca(dfb::racc);
            copy_init(dfb::racc);
            racc.wait_front(1);
        }
        reconfig_data_format(dfb_sum_scaler, dfb::gexp);
        pack_reconfig_data_format(sout);
        if (gi == 0) {
            BHF_SUM_INIT(dfb::gexp, dfb_sum_scaler, sout);
        }
        tile_regs_acquire();
        if (gi != 0) {
            copy_tile(dfb::racc, 0, 0);
            reconfig_data_format(dfb_sum_scaler, dfb::gexp);
            BHF_SUM_INIT(dfb::gexp, dfb_sum_scaler, sout);
        }
        for (std::uint32_t t = tb; t < te; ++t) {
            gexp.wait_front(static_cast<uint16_t>(t - tb + 1));
            BHF_SUM_TILE(dfb::gexp, dfb_sum_scaler, t, 0);
        }
        if (gi == 0) {
            pacc.reserve_back(1);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dfb::pacc);
            tile_regs_release();
            pacc.push_back(1);
            reduce_uninit();
            gexp.pop_front(static_cast<uint16_t>(te - tb));
            return;
        }
        racc.pop_front(1);
        const std::uint32_t bhf_npop = te - tb;
#else
        reconfig_data_format(dfb_sum_scaler, dfb::gexp);
        pack_reconfig_data_format(dfb::rout);
        BHF_SUM_INIT(dfb::gexp, dfb_sum_scaler, dfb::rout);
        tile_regs_acquire();
        for (std::uint32_t t = 0; t < gt; ++t) {
            gexp.wait_front(static_cast<uint16_t>(t + 1));
            BHF_SUM_TILE(dfb::gexp, dfb_sum_scaler, t, 0);
        }
        const std::uint32_t bhf_npop = gt;
#endif
        if constexpr (DST_ACCUM_MODE) {
            recip_tile_init<ReciprocalDestAcc::FP32, ReciprocalApproxMode::Precise>();
            recip_tile<ReciprocalDestAcc::FP32, ReciprocalApproxMode::Precise>(0);
        } else {
            recip_tile_init();
            recip_tile(0);
        }
        rout.reserve_back(1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::rout);
        tile_regs_release();
        rout.push_back(1);
        reduce_uninit();
        gexp.pop_front(static_cast<uint16_t>(bhf_npop));
        return;
    }
#endif
#ifdef BHF_GROUP
    [[maybe_unused]] const bool is_member = gi != 0;
    const bool is_last = !is_member && c + gm >= C;
#else
    [[maybe_unused]] const bool is_member = false;
    const bool is_last = c + 1 == C;
#endif
    const std::uint32_t sum_out =
        is_last ? static_cast<std::uint32_t>(dfb::rout) : static_cast<std::uint32_t>(dfb::pacc);

    for (std::uint32_t r = 0; r < num_rows; ++r) {
#ifdef NUMERIC_STABLE
        {
        BHFZ("C_MAX");
#ifdef BHF_MAXSTREAM
        reconfig_data_format(dfb_max_scaler, dfb_in0);
        pack_reconfig_data_format(dfb_pmax);
        BHF_MAX_INIT(dfb_in0, dfb_max_scaler, dfb_pmax);
        tile_regs_acquire();
        for (std::uint32_t t = 0; t < T; ++t) {
            in0.wait_front(static_cast<uint16_t>(t + 1));
            BHF_MAX_TILE(dfb_in0, dfb_max_scaler, t, 0);
        }
        pmax.reserve_back(1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb_pmax);
        tile_regs_release();
        pmax.push_back(1);
        reduce_uninit();
#else
        compute_kernel_lib::reduce<
            PoolType::MAX,
            ReduceDim::REDUCE_ROW,
            dfb_in0,
            dfb_max_scaler,
            dfb_pmax,
            compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
            in0, max_scaler, pmax, compute_kernel_lib::ReduceInputBlockShape::row(T));
#endif
        }

#if !defined(BHF_ABL_NOMAXX) && !defined(BHF_MAXRISC) && !defined(BHF_MAXAG) && !defined(BHF_MAXAGC)
#ifdef BHF_TREE2
        const std::uint32_t bhf_nc = c == 0 ? th + 1 : C - th;
        if (c == 0 || c == th) {
#else
        const std::uint32_t bhf_nc = C;
        if (c == 0) {
#endif
            BHFZ("C_COMB");
#ifdef BHF_COMBSTREAM
            rmax.wait_front(1);
            reconfig_data_format_srca(dfb::rmax);
            pack_reconfig_data_format(dfb::mout);
            tile_regs_acquire();
            copy_init(dfb::rmax);
            copy_tile(dfb::rmax, 0, 0);
            for (std::uint32_t i = 1; i < bhf_nc; ++i) {
                rmax.wait_front(static_cast<uint16_t>(i + 1));
                copy_init(dfb::rmax);
                copy_tile(dfb::rmax, i, 1);
                binary_max_tile_init();
                binary_max_tile(0, 1, 0 BHF_VM_COMB);
            }
            tile_regs_commit();
            mout.reserve_back(1);
            tile_regs_wait();
            pack_tile(0, dfb::mout);
            tile_regs_release();
            mout.push_back(1);
            rmax.pop_front(static_cast<uint16_t>(bhf_nc));
        }
        if (false) {
#endif
            rmax.wait_front(static_cast<uint16_t>(bhf_nc));
            reconfig_data_format_srca(dfb::rmax);
            pack_reconfig_data_format(dfb::mout);
            tile_regs_acquire();
#ifdef BHF_COMBFAST
            copy_init(dfb::rmax);
            copy_tile(dfb::rmax, 0, 0);
            for (std::uint32_t k = 1; k < bhf_nc && k < 4; ++k) {
                copy_tile(dfb::rmax, k, k);
            }
            binary_max_tile_init();
            for (std::uint32_t i = 1; i < bhf_nc; ++i) {
                binary_max_tile(0, 1 + (i - 1) % 3, 0 BHF_VM_COMB);
                if (i % 3 == 0 && i + 1 < bhf_nc) {
                    copy_init(dfb::rmax);
                    for (std::uint32_t k = i + 1; k < bhf_nc && k < i + 4; ++k) {
                        copy_tile(dfb::rmax, k, 1 + (k - 1) % 3);
                    }
                    binary_max_tile_init();
                }
            }
#else
            copy_init(dfb::rmax);
            copy_tile(dfb::rmax, 0, 0);
            for (std::uint32_t i = 1; i < bhf_nc; ++i) {
                copy_init(dfb::rmax);
                copy_tile(dfb::rmax, i, 1);
                binary_max_tile_init();
                binary_max_tile(0, 1, 0 BHF_VM_COMB);
            }
#endif
            tile_regs_commit();
            mout.reserve_back(1);
            tile_regs_wait();
            pack_tile(0, dfb::mout);
            tile_regs_release();
            mout.push_back(1);
            rmax.pop_front(static_cast<uint16_t>(bhf_nc));
        }
#endif

#if (defined(BHF_GROUP) && !defined(BHF_GROUPMUL)) || (defined(BHF_STREAM) && defined(BHF_DIRECT))
        if (is_member) {
            reconfig_data_format(dfb_in0, dfb_mcb);
            pack_reconfig_data_format(dfb::pacc);
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
                    dfb::pacc,
                    ckl::ReservePolicy::PerBlockSize,
                    ckl::PushPolicy::PerBlockSize,
                    ckl::DataFormatReconfig::Disabled)>{pacc});
#if defined(BHF_STREAM) && defined(BHF_DIRECT)
            BHFZ("C_MUL");
            ckl::mul<
                ckl::input(dfb::gexp, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block),
                ckl::input(dfb_rcb, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd),
                ckl::output(dfb_out0, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
                ckl::IterationShape::tiles(T).block_size(ndst), gexp, rcb, out0);
#endif
            continue;
        }
#endif
        {
        BHFZ("C_EXP");
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
#ifndef BHF_ABL_NOEXP
            ckl::Exp<static_cast<ckl::Approx>(EXP_APPROX), ckl::Dst::D0>{},
#endif
            ckl::PackTile<ckl::output(
                dfb_exps,
                ckl::ReservePolicy::PerBlockSize,
                ckl::PushPolicy::PerBlockSize,
                ckl::DataFormatReconfig::Disabled)>{exps});
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
#ifdef BHF_GROUPMUL
        if (is_member) {
            BHFZ("P_COPY");
            reconfig_data_format_srca(dfb_exps);
            pack_reconfig_data_format(dfb::pacc);
            copy_init(dfb_exps);
            for (std::uint32_t t = 0; t < T; ++t) {
                tile_regs_acquire();
                copy_tile(dfb_exps, t, 0);
                tile_regs_commit();
                pacc.reserve_back(1);
                tile_regs_wait();
                pack_tile(0, dfb::pacc);
                tile_regs_release();
                pacc.push_back(1);
            }
        }
        if (!is_member) {
#else
        {
#endif

        reconfig_data_format(dfb_sum_scaler, dfb_exps);
#ifdef BHF_ABL_NOSUM
        const bool bhf_chain = false;
#else
        const bool bhf_chain = c > 0;
#endif
#ifdef BHF_PRESTAGE
        if (bhf_chain) {
            reconfig_data_format_srca(dfb::racc);
            copy_init(dfb::racc);
        }
#endif
        if (bhf_chain) {
            BHFZ("C_WACC");
            racc.wait_front(1);
        }
        {
        BHFZ("C_SUM");
#ifdef BHF_PRESTAGE
        if (!bhf_chain) {
            BHF_SUM_INIT(dfb_exps, dfb_sum_scaler, sum_out);
        }
#else
        BHF_SUM_INIT(dfb_exps, dfb_sum_scaler, sum_out);
#endif
        tile_regs_acquire();
        if (bhf_chain) {
#ifndef BHF_PRESTAGE
            reconfig_data_format_srca(dfb::racc);
            copy_init(dfb::racc);
#endif
            copy_tile(dfb::racc, 0, 0);
            reconfig_data_format(dfb_sum_scaler, dfb_exps);
            BHF_SUM_INIT(dfb_exps, dfb_sum_scaler, sum_out);
        }
        for (std::uint32_t t = 0; t < T; ++t) {
#ifdef BHF_ABL_SUMIND
            BHF_SUM_TILE(dfb_exps, dfb_sum_scaler, t, t & 1);
#else
            BHF_SUM_TILE(dfb_exps, dfb_sum_scaler, t, 0);
#endif
        }
#ifdef BHF_GROUP
        if (gt != 0) {
            gexp.wait_front(static_cast<uint16_t>(gt));
            for (std::uint32_t t = 0; t < gt; ++t) {
                BHF_SUM_TILE(dfb::gexp, dfb_sum_scaler, t, 0);
            }
        }
#endif
        if (is_last) {
            if constexpr (DST_ACCUM_MODE) {
                recip_tile_init<ReciprocalDestAcc::FP32, ReciprocalApproxMode::Precise>();
                recip_tile<ReciprocalDestAcc::FP32, ReciprocalApproxMode::Precise>(0, BHF_VM_RECIP);
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
        if (bhf_chain) {
            racc.pop_front(1);
        }
#ifdef BHF_GROUPMUL
        if (gt != 0) {
            gexp.pop_front(static_cast<uint16_t>(gt));
        }
#endif
        }

        BHFZ("C_MUL");
#if defined(BHF_GROUP) && !defined(BHF_GROUPMUL)
        if (gt != 0) {
            ckl::mul<
                ckl::input(dfb_exps, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block),
                ckl::input(dfb_rcb, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None),
                ckl::output(dfb_out0, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
                ckl::IterationShape::tiles(T).block_size(ndst), exps, rcb, out0);
            ckl::mul<
                ckl::input(dfb::gexp, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block),
                ckl::input(dfb_rcb, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd),
                ckl::output(dfb_out0, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
                ckl::IterationShape::tiles(gt).block_size(ndst), gexp, rcb, out0);
            continue;
        }
#endif
        ckl::mul<
            ckl::input(dfb_exps, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block),
            ckl::input(dfb_rcb, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd),
            ckl::output(dfb_out0, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
            ckl::IterationShape::tiles(T).block_size(ndst), exps, rcb, out0);
    }
}
