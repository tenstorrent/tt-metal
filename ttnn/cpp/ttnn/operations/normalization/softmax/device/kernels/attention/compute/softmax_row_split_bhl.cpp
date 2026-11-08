// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/binary_max_min.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/softmax.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

#ifdef BHL_PROF
#include "api/debug/kernel_profiler.hpp"
#define BHLZ(n) DeviceZoneScopedN(n)
#else
#define BHLZ(n)
#endif

#ifdef BHL_LRMAXC
#define BHL_MAXVM VectorMode::C
#else
#define BHL_MAXVM VectorMode::RC
#endif

#if defined(BHL_LRADD) && defined(TRISC_MATH)
namespace ckernel::sfpu {
// dst[out] = dst[in_new] + tf32(dst[in_run]); MODE 1 clears the 13 low mantissa bits of the running sum
// (the value main's PREV round trip through srcA keeps), MODE 2 adds it unchanged.
template <int ITERATIONS, int MODE>
inline void bhl_trunc_add(const std::uint32_t in_new, const std::uint32_t in_run, const std::uint32_t out) {
    for (int d = 0; d < ITERATIONS; d++) {
        constexpr std::uint32_t ts = 32;
        sfpi::vFloat a = sfpi::dst_reg[in_new * ts];
        sfpi::vFloat b = sfpi::dst_reg[in_run * ts];
        if constexpr (MODE == 1) {
            b = sfpi::as<sfpi::vFloat>((sfpi::as<sfpi::vUInt>(b) >> 13) << 13);
        } else if constexpr (MODE == 3) {
            sfpi::vUInt u = sfpi::as<sfpi::vUInt>(b);
            u = u + ((u >> 13) & 1) + 0xFFF;
            b = sfpi::as<sfpi::vFloat>((u >> 13) << 13);
        }
        sfpi::dst_reg[out * ts] = a + b;
        sfpi::dst_reg++;
    }
}
// running sum over n compacted chunk-sum rows (one SFPU row = one chunk's 32 values), in chunk order:
// run = new + trunc(run); result truncated and left in the last row
template <int ITERATIONS, int MODE>
inline void bhl_cmp_add(const std::uint32_t, const std::uint32_t, const std::uint32_t, const std::uint32_t n) {
    sfpi::vFloat run = sfpi::dst_reg[0];
    for (std::uint32_t k = 1; k < n; ++k) {
        sfpi::dst_reg++;
        sfpi::dst_reg++;
        sfpi::vFloat b = sfpi::as<sfpi::vFloat>((sfpi::as<sfpi::vUInt>(run) >> 13) << 13);
        run = sfpi::dst_reg[0] + b;
    }
    run = sfpi::as<sfpi::vFloat>((sfpi::as<sfpi::vUInt>(run) >> 13) << 13);
    sfpi::dst_reg[0] = run;
}
template <int ITERATIONS, int MODE>
inline void bhl_trunc_only(const std::uint32_t in0, const std::uint32_t in1, const std::uint32_t out) {
    for (int d = 0; d < ITERATIONS; d++) {
        constexpr std::uint32_t ts = 32;
        sfpi::vFloat b = sfpi::dst_reg[in0 * ts];
        if constexpr (MODE == 1) {
            b = sfpi::as<sfpi::vFloat>((sfpi::as<sfpi::vUInt>(b) >> 13) << 13);
        } else if constexpr (MODE == 3) {
            sfpi::vUInt u = sfpi::as<sfpi::vUInt>(b);
            u = u + ((u >> 13) & 1) + 0xFFF;
            b = sfpi::as<sfpi::vFloat>((u >> 13) << 13);
        }
        sfpi::dst_reg[out * ts] = b;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

// merge n max tiles from cb into dst 0 (first: dst 0 is loaded from tile 0)
ALWI void bhl_maxmerge(std::uint32_t cb, std::uint32_t n, bool first) {
    reconfig_data_format_srca(cb);
    std::uint32_t i = 0;
    if (first) {
        copy_init(cb);
        copy_tile(cb, 0, 0);
        i = 1;
    }
    while (i < n) {
        const std::uint32_t b = (n - i) < 3 ? (n - i) : 3;
        copy_init(cb);
        for (std::uint32_t j = 0; j < b; ++j) {
            copy_tile(cb, i + j, 1 + j);
        }
        binary_max_tile_init();
        for (std::uint32_t j = 0; j < b; ++j) {
            binary_max_tile(0, 1 + j, 0, BHL_MAXVM);
        }
        i += b;
    }
}

#ifdef BHL_LRMSM
// same merge as bhl_maxmerge(first=true) but waits for the partial maxima one batch at a time
template <typename RmDFB>
ALWI void bhl_maxmerge_s(RmDFB& rm, std::uint32_t cb, std::uint32_t n) {
    reconfig_data_format_srca(cb);
    rm.wait_front(1);
    copy_init(cb);
    copy_tile(cb, 0, 0);
    for (std::uint32_t i = 1; i < n;) {
        const std::uint32_t b = (n - i) < 3 ? (n - i) : 3;
        rm.wait_front(static_cast<uint16_t>(i + b));
        copy_init(cb);
        for (std::uint32_t j = 0; j < b; ++j) {
            copy_tile(cb, i + j, 1 + j);
        }
        binary_max_tile_init();
        for (std::uint32_t j = 0; j < b; ++j) {
            binary_max_tile(0, 1 + j, 0, BHL_MAXVM);
        }
        i += b;
    }
}
#endif

template <typename InDFB, typename OutDFB>
ALWI void exp_range(InDFB& /*in0*/, OutDFB& exps, std::uint32_t start, std::uint32_t len, std::uint32_t blk) {
    reconfig_data_format_srca(dfb::in0);
    pack_reconfig_data_format(dfb::exps);
    reconfig_data_format_srcb(dfb::mcb);
    sub_bcast_cols_init(dfb::in0, dfb::mcb);
    exp_tile_init<EXP_APPROX>();
    for (std::uint32_t b = 0; b < len; b += blk) {
        const std::uint32_t rem = (b + blk > len) ? (len - b) : blk;
        exps.reserve_back(static_cast<uint16_t>(rem));
        tile_regs_acquire();
        for (std::uint32_t d = 0; d < rem; ++d) {
            sub_tiles_bcast_cols(dfb::in0, dfb::mcb, start + b + d, 0, d);
        }
        for (std::uint32_t d = 0; d < rem; ++d) {
            exp_tile<EXP_APPROX>(d);
        }
        tile_regs_wait();
        tile_regs_commit();
        for (std::uint32_t d = 0; d < rem; ++d) {
            pack_tile(d, dfb::exps);
        }
        exps.push_back(static_cast<uint16_t>(rem));
        tile_regs_release();
    }
}

template <typename ExpsDFB, typename OutDFB>
ALWI void mul_range(ExpsDFB& exps, OutDFB& out0, std::uint32_t len, std::uint32_t blk) {
    reconfig_data_format(dfb::exps, dfb::rcb);
    pack_reconfig_data_format(dfb::out0);
    mul_bcast_cols_init(dfb::exps, dfb::rcb);
    for (std::uint32_t b = 0; b < len; b += blk) {
        const std::uint32_t rem = (b + blk > len) ? (len - b) : blk;
        exps.wait_front(static_cast<uint16_t>(rem));
        tile_regs_acquire();
        tile_regs_wait();
        for (std::uint32_t d = 0; d < rem; ++d) {
            mul_tiles_bcast_cols(dfb::exps, dfb::rcb, d, 0, d);
        }
        tile_regs_commit();
        out0.reserve_back(static_cast<uint16_t>(rem));
        for (std::uint32_t d = 0; d < rem; ++d) {
            pack_tile(d, dfb::out0);
        }
        exps.pop_front(static_cast<uint16_t>(rem));
        out0.push_back(static_cast<uint16_t>(rem));
        tile_regs_release();
    }
}

#ifdef BHL_LRSEG
template <typename ExpsDFB, typename RaccDFB, typename PaccDFB, typename PsumDFB>
ALWI void bhl_seg_reduce(
    ExpsDFB& exps, RaccDFB& racc, PaccDFB& pacc, PsumDFB& psum, std::uint32_t base, std::uint32_t len, std::uint32_t fl) {
    const bool first = (fl & 1u) != 0;
    const bool last = (fl & 2u) != 0;
    const std::uint32_t outcb = last ? static_cast<std::uint32_t>(dfb::psum) : static_cast<std::uint32_t>(dfb::pacc);
    if (!first) {
        racc.wait_front(1);
    }
    reconfig_data_format(dfb::sum_scaler, dfb::exps);
    pack_reconfig_data_format(outcb);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(dfb::exps, dfb::sum_scaler, outcb);
    tile_regs_acquire();
    if (!first) {
        reconfig_data_format_srca(dfb::racc);
        copy_init(dfb::racc);
        copy_tile(dfb::racc, 0, 0);
        reconfig_data_format(dfb::sum_scaler, dfb::exps);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(dfb::exps, dfb::sum_scaler, outcb);
    }
    exps.wait_front(static_cast<uint16_t>(base + len));
    for (std::uint32_t t = 0; t < len; ++t) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(dfb::exps, dfb::sum_scaler, base + t, 0, 0);
    }
    if (last) {
        psum.reserve_back(1);
    } else {
        pacc.reserve_back(1);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, outcb);
    tile_regs_release();
    if (last) {
        psum.push_back(1);
    } else {
        pacc.push_back(1);
    }
    reduce_uninit();
    if (!first) {
        racc.pop_front(1);
    }
}
#endif

void kernel_main() {
    const std::uint32_t T = get_arg(args::T);
    const std::uint32_t nck = get_arg(args::nck);
    const std::uint32_t ck0 = get_arg(args::ck0);
    const std::uint32_t nchunk = get_arg(args::nchunk);
    const std::uint32_t L = get_arg(args::L);
    const std::uint32_t last_len = get_arg(args::last_len);
    const bool is_comb = get_arg(args::is_comb) == 1;
    const std::uint32_t active = get_arg(args::active);
    const std::uint32_t num_rows = get_arg(args::num_rows);
    const std::uint32_t blk = get_arg(args::blk);
    const std::uint32_t pad = get_arg(args::pad);
    [[maybe_unused]] const std::uint32_t cj = get_arg(args::cj);
    [[maybe_unused]] const std::uint32_t cS = get_arg(args::cS);
    [[maybe_unused]] const std::uint32_t nseg = get_arg(args::nseg);
    [[maybe_unused]] const std::uint32_t s0o = get_arg(args::s0o);
    [[maybe_unused]] const std::uint32_t s0n = get_arg(args::s0n);
    [[maybe_unused]] const std::uint32_t s0f = get_arg(args::s0f);
    [[maybe_unused]] const std::uint32_t s1o = get_arg(args::s1o);
    [[maybe_unused]] const std::uint32_t s1n = get_arg(args::s1n);
    [[maybe_unused]] const std::uint32_t s1f = get_arg(args::s1f);
    [[maybe_unused]] const bool is_lead = get_arg(args::is_lead) == 1;
    [[maybe_unused]] const std::uint32_t gsz = get_arg(args::gsz);
    [[maybe_unused]] const std::uint32_t ngrp = get_arg(args::ngrp);
    if (T == 0) {
        return;
    }

    DataflowBuffer in0(dfb::in0);
    DataflowBuffer max_scaler(dfb::max_scaler);
    DataflowBuffer sum_scaler(dfb::sum_scaler);
    DataflowBuffer exps(dfb::exps);
    DataflowBuffer pmax(dfb::pmax);
    DataflowBuffer rmax(dfb::rmax);
    DataflowBuffer mout(dfb::mout);
    DataflowBuffer mcb(dfb::mcb);
    DataflowBuffer psum(dfb::psum);
    DataflowBuffer rsum(dfb::rsum);
    DataflowBuffer prev(dfb::prev);
    DataflowBuffer rout(dfb::rout);
    DataflowBuffer rcb(dfb::rcb);
    DataflowBuffer out0(dfb::out0);
    [[maybe_unused]] DataflowBuffer pacc(dfb::pacc);
    [[maybe_unused]] DataflowBuffer racc(dfb::racc);
    [[maybe_unused]] DataflowBuffer rmax2(dfb::rmax2);
    [[maybe_unused]] DataflowBuffer f5fl(dfb::f5fl);

    compute_kernel_hw_startup(dfb::in0, dfb::max_scaler, dfb::exps);
    max_scaler.wait_front(1);
    sum_scaler.wait_front(1);

    for (std::uint32_t r = 0; r < num_rows; ++r) {
        {
        BHLZ("S_MAX");
#ifdef BHL_LRMSTR
        reconfig_data_format(dfb::max_scaler, dfb::in0);
        pack_reconfig_data_format(dfb::pmax);
        reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(dfb::in0, dfb::max_scaler, dfb::pmax);
        tile_regs_acquire();
        for (std::uint32_t t = 0; t < T; ++t) {
            in0.wait_front(static_cast<uint16_t>(t + 1));
            reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(dfb::in0, dfb::max_scaler, t, 0, 0);
        }
        pmax.reserve_back(1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::pmax);
        tile_regs_release();
        pmax.push_back(1);
        reduce_uninit();
#else
        compute_kernel_lib::reduce<
            PoolType::MAX,
            ReduceDim::REDUCE_ROW,
            dfb::in0,
            dfb::max_scaler,
            dfb::pmax,
            compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
            in0, max_scaler, pmax, compute_kernel_lib::ReduceInputBlockShape::row(T));
#endif
        }

#if defined(BHL_LRTREE) || defined(BHL_LRMAXC)
#ifdef BHL_LRTREE
        const std::uint32_t bhl_gn = gsz;
        const bool bhl_merge_here = is_lead;
#else
        const std::uint32_t bhl_gn = active;
        const bool bhl_merge_here = is_comb;
#endif
        if (bhl_merge_here) {
            BHLZ("S_COMB");
#ifdef BHL_LRMSM
            pack_reconfig_data_format(dfb::mout);
            tile_regs_acquire();
            bhl_maxmerge_s(rmax, static_cast<std::uint32_t>(dfb::rmax), bhl_gn);
#else
            rmax.wait_front(static_cast<uint16_t>(bhl_gn));
            pack_reconfig_data_format(dfb::mout);
            tile_regs_acquire();
            bhl_maxmerge(static_cast<std::uint32_t>(dfb::rmax), bhl_gn, true);
#endif
#ifdef BHL_LRTREE
            if (is_comb && ngrp > 1) {
                rmax2.wait_front(static_cast<uint16_t>(ngrp - 1));
                bhl_maxmerge(static_cast<std::uint32_t>(dfb::rmax2), ngrp - 1, false);
            }
#endif
            tile_regs_commit();
            mout.reserve_back(1);
            tile_regs_wait();
            pack_tile(0, dfb::mout);
            tile_regs_release();
            mout.push_back(1);
            rmax.pop_front(static_cast<uint16_t>(bhl_gn));
#ifdef BHL_LRTREE
            if (is_comb && ngrp > 1) {
                rmax2.pop_front(static_cast<uint16_t>(ngrp - 1));
            }
#endif
        }
#else
        if (is_comb) {
            BHLZ("S_COMB");
            rmax.wait_front(static_cast<uint16_t>(active));
            reconfig_data_format_srca(dfb::rmax);
            pack_reconfig_data_format(dfb::mout);
            tile_regs_acquire();
#ifdef BHL_COMBFAST
            copy_init(dfb::rmax);
            copy_tile(dfb::rmax, 0, 0);
            for (std::uint32_t k = 1; k < active && k < 4; ++k) {
                copy_tile(dfb::rmax, k, k);
            }
            binary_max_tile_init();
            for (std::uint32_t i = 1; i < active; ++i) {
                binary_max_tile(0, 1 + (i - 1) % 3, 0);
                if (i % 3 == 0 && i + 1 < active) {
                    copy_init(dfb::rmax);
                    for (std::uint32_t k = i + 1; k < active && k < i + 4; ++k) {
                        copy_tile(dfb::rmax, k, 1 + (k - 1) % 3);
                    }
                    binary_max_tile_init();
                }
            }
#else
            copy_init(dfb::rmax);
            copy_tile(dfb::rmax, 0, 0);
            for (std::uint32_t i = 1; i < active; ++i) {
                copy_init(dfb::rmax);
                copy_tile(dfb::rmax, i, 1);
                binary_max_tile_init();
                binary_max_tile(0, 1, 0);
            }
#endif
            tile_regs_commit();
            mout.reserve_back(1);
            tile_regs_wait();
            pack_tile(0, dfb::mout);
            tile_regs_release();
            mout.push_back(1);
            rmax.pop_front(static_cast<uint16_t>(active));
        }
#endif

        mcb.wait_front(1);
        {
        BHLZ("S_EXPSUM");
#if defined(BHL_LRSEG)
        exp_range(in0, exps, 0, T, blk);
        if (nseg == 2) {
            bhl_seg_reduce(exps, racc, pacc, psum, s1o, s1n, s1f);
            bhl_seg_reduce(exps, racc, pacc, psum, s0o, s0n, s0f);
        } else {
            bhl_seg_reduce(exps, racc, pacc, psum, s0o, s0n, s0f);
        }
#elif defined(BHL_KEEPEXP)
        std::uint32_t bhl_base = 0;
        for (std::uint32_t k = 0; k < nck; ++k) {
            const std::uint32_t len = (ck0 + k == nchunk - 1) ? last_len : L;
            exp_range(in0, exps, k * L, len, blk);
#ifdef BHL_CCHAIN
            const bool bhl_first = cj == 0;
            const bool bhl_last = cj + 1 == cS;
            const std::uint32_t bhl_out =
                bhl_last ? static_cast<std::uint32_t>(dfb::psum) : static_cast<std::uint32_t>(dfb::pacc);
            if (!bhl_first) {
                racc.wait_front(1);
            }
            reconfig_data_format(dfb::sum_scaler, dfb::exps);
            pack_reconfig_data_format(bhl_out);
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(dfb::exps, dfb::sum_scaler, bhl_out);
            tile_regs_acquire();
            if (!bhl_first) {
                reconfig_data_format_srca(dfb::racc);
                copy_init(dfb::racc);
                copy_tile(dfb::racc, 0, 0);
                reconfig_data_format(dfb::sum_scaler, dfb::exps);
                reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(dfb::exps, dfb::sum_scaler, bhl_out);
            }
            for (std::uint32_t t = 0; t < len; ++t) {
                exps.wait_front(static_cast<uint16_t>(bhl_base + t + 1));
                reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(dfb::exps, dfb::sum_scaler, bhl_base + t, 0, 0);
            }
            if (bhl_last) {
                psum.reserve_back(1);
            } else {
                pacc.reserve_back(1);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, bhl_out);
            tile_regs_release();
            if (bhl_last) {
                psum.push_back(1);
            } else {
                pacc.push_back(1);
            }
            reduce_uninit();
            if (!bhl_first) {
                racc.pop_front(1);
            }
#else
            reconfig_data_format(dfb::sum_scaler, dfb::exps);
            pack_reconfig_data_format(dfb::psum);
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(dfb::exps, dfb::sum_scaler, dfb::psum);
            tile_regs_acquire();
            for (std::uint32_t t = 0; t < len; ++t) {
                exps.wait_front(static_cast<uint16_t>(bhl_base + t + 1));
                reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(dfb::exps, dfb::sum_scaler, bhl_base + t, 0, 0);
            }
            psum.reserve_back(1);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dfb::psum);
            tile_regs_release();
            psum.push_back(1);
            reduce_uninit();
#endif
            bhl_base += len;
        }
#else
        for (std::uint32_t k = 0; k < nck; ++k) {
            const std::uint32_t len = (ck0 + k == nchunk - 1) ? last_len : L;
            exp_range(in0, exps, k * L, len, blk);
            compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, dfb::exps, dfb::sum_scaler, dfb::psum>(
                exps, sum_scaler, psum, compute_kernel_lib::ReduceInputBlockShape::row(len));
        }
#endif
        }

#if defined(BHL_LRACMP)
        if (is_comb) {
            BHLZ("S_ADD");
            DataflowBuffer f5cp(dfb::f5cp);
            DataflowBuffer f5co(dfb::f5co);
            const std::uint32_t ncomp = (nchunk + 15) / 16;
            f5cp.wait_front(static_cast<uint16_t>(ncomp));
            reconfig_data_format_srca(dfb::f5cp);
            pack_reconfig_data_format(dfb::f5co);
            tile_regs_acquire();
            copy_init(dfb::f5cp);
            for (std::uint32_t t = 0; t < ncomp; ++t) {
                copy_tile(dfb::f5cp, t, t);
            }
            add_binary_tile_init();
            MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, bhl_cmp_add, (8, 1), 0, 0, 0, VectorMode::RC_custom, nchunk)));
            tile_regs_commit();
            f5co.reserve_back(1);
            tile_regs_wait();
            pack_tile((nchunk - 1) / 16, dfb::f5co);
            tile_regs_release();
            f5co.push_back(1);
            f5cp.pop_front(static_cast<uint16_t>(ncomp));
            rsum.wait_front(1);
            reconfig_data_format_srca(dfb::rsum);
            pack_reconfig_data_format(dfb::rout);
            tile_regs_acquire();
            copy_init(dfb::rsum);
            copy_tile(dfb::rsum, 0, 0);
            recip_tile_init<ReciprocalDestAcc::FP32, ReciprocalApproxMode::Precise>();
            recip_tile<ReciprocalDestAcc::FP32, ReciprocalApproxMode::Precise>(0, VectorMode::C);
            tile_regs_commit();
            tile_regs_wait();
            rout.reserve_back(1);
            pack_tile(0, dfb::rout);
            rout.push_back(1);
            tile_regs_release();
            rsum.pop_front(1);
        }
#elif defined(BHL_LRADD)
        if (is_comb) {
            BHLZ("S_ADD");
#ifdef BHL_LRASTR
            rsum.wait_front(1);
#else
            rsum.wait_front(static_cast<uint16_t>(nchunk));
#endif
            reconfig_data_format_srca(dfb::rsum);
            pack_reconfig_data_format(dfb::rout);
            tile_regs_acquire();
            copy_init(dfb::rsum);
            copy_tile(dfb::rsum, 0, 0);
            for (std::uint32_t k = 1; k < nchunk;) {
                const std::uint32_t b = (nchunk - k) < 3 ? (nchunk - k) : 3;
#ifdef BHL_LRASTR
                rsum.wait_front(static_cast<uint16_t>(k + b));
#endif
                copy_init(dfb::rsum);
                for (std::uint32_t j = 0; j < b; ++j) {
                    copy_tile(dfb::rsum, k + j, 1 + j);
                }
                add_binary_tile_init();
                for (std::uint32_t j = 0; j < b; ++j) {
                    MATH((SFPU_BINARY_CALL(
                        DST_SYNC_MODE, DST_ACCUM_MODE, bhl_trunc_add, (8, BHL_LRADD), 1 + j, 0, 0, VectorMode::C)));
                }
                k += b;
            }
            MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, bhl_trunc_only, (8, BHL_LRADD), 0, 0, 0, VectorMode::C)));
            recip_tile_init<ReciprocalDestAcc::FP32, ReciprocalApproxMode::Precise>();
            recip_tile<ReciprocalDestAcc::FP32, ReciprocalApproxMode::Precise>(0, VectorMode::C);
            tile_regs_commit();
            tile_regs_wait();
            rout.reserve_back(1);
            pack_tile(0, dfb::rout);
            rout.push_back(1);
            tile_regs_release();
            rsum.pop_front(static_cast<uint16_t>(nchunk));
        }
#else
        if (is_comb) {
            BHLZ("S_ADD");
            rsum.wait_front(static_cast<uint16_t>(nchunk));
            pack_reconfig_data_format(dfb::prev);
            for (std::uint32_t k = 0; k < nchunk; ++k) {
                tile_regs_acquire();
                reconfig_data_format_srca(dfb::rsum);
                copy_init(dfb::rsum);
                copy_tile(dfb::rsum, k, 0);
                if (k > 0) {
                    prev.wait_front(1);
                    reconfig_data_format_srca(dfb::prev);
                    copy_init(dfb::prev);
                    copy_tile(dfb::prev, 0, 1);
                    add_binary_tile_init();
                    add_binary_tile(0, 1, 0);
                    prev.pop_front(1);
                }
                tile_regs_commit();
                prev.reserve_back(1);
                tile_regs_wait();
                pack_tile(0, dfb::prev);
                tile_regs_release();
                prev.push_back(1);
            }
            rsum.pop_front(static_cast<uint16_t>(nchunk));

            prev.wait_front(1);
            reconfig_data_format_srca(dfb::prev);
            pack_reconfig_data_format(dfb::prev, dfb::rout);
            tile_regs_acquire();
            copy_init(dfb::prev);
            copy_tile(dfb::prev, 0, 0);
            prev.pop_front(1);
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
        }
#endif

        BHLZ("S_EXPMUL");
        rcb.wait_front(1);
#ifdef BHL_LRSEG
        mul_range(exps, out0, T, blk);
#else
        for (std::uint32_t k = 0; k < nck; ++k) {
            const std::uint32_t len = (ck0 + k == nchunk - 1) ? last_len : L;
#ifndef BHL_KEEPEXP
            exp_range(in0, exps, k * L, len, blk);
#endif
            mul_range(exps, out0, len, blk);
        }
#endif
        rcb.pop_front(1);
        mcb.pop_front(1);
        if (pad > 0) {
            in0.wait_front(static_cast<uint16_t>(T + pad));
        }
        in0.pop_front(static_cast<uint16_t>(T + pad));
    }
}
