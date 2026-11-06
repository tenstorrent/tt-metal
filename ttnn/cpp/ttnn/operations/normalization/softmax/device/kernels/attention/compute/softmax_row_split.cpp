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

    compute_kernel_hw_startup(dfb::in0, dfb::max_scaler, dfb::exps);
    max_scaler.wait_front(1);
    sum_scaler.wait_front(1);

    for (std::uint32_t r = 0; r < num_rows; ++r) {
        compute_kernel_lib::reduce<
            PoolType::MAX,
            ReduceDim::REDUCE_ROW,
            dfb::in0,
            dfb::max_scaler,
            dfb::pmax,
            compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
            in0, max_scaler, pmax, compute_kernel_lib::ReduceInputBlockShape::row(T));

        if (is_comb) {
            rmax.wait_front(static_cast<uint16_t>(active));
            reconfig_data_format_srca(dfb::rmax);
            pack_reconfig_data_format(dfb::mout);
            tile_regs_acquire();
            copy_init(dfb::rmax);
            copy_tile(dfb::rmax, 0, 0);
            for (std::uint32_t i = 1; i < active; ++i) {
                copy_init(dfb::rmax);
                copy_tile(dfb::rmax, i, 1);
                binary_max_tile_init();
                binary_max_tile(0, 1, 0);
            }
            tile_regs_commit();
            mout.reserve_back(1);
            tile_regs_wait();
            pack_tile(0, dfb::mout);
            tile_regs_release();
            mout.push_back(1);
            rmax.pop_front(static_cast<uint16_t>(active));
        }

        mcb.wait_front(1);
        for (std::uint32_t k = 0; k < nck; ++k) {
            const std::uint32_t len = (ck0 + k == nchunk - 1) ? last_len : L;
            exp_range(in0, exps, k * L, len, blk);
            compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, dfb::exps, dfb::sum_scaler, dfb::psum>(
                exps, sum_scaler, psum, compute_kernel_lib::ReduceInputBlockShape::row(len));
        }

        if (is_comb) {
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

        rcb.wait_front(1);
        for (std::uint32_t k = 0; k < nck; ++k) {
            const std::uint32_t len = (ck0 + k == nchunk - 1) ? last_len : L;
            exp_range(in0, exps, k * L, len, blk);
            mul_range(exps, out0, len, blk);
        }
        rcb.pop_front(1);
        mcb.pop_front(1);
        if (pad > 0) {
            in0.wait_front(static_cast<uint16_t>(T + pad));
        }
        in0.pop_front(static_cast<uint16_t>(T + pad));
    }
}
