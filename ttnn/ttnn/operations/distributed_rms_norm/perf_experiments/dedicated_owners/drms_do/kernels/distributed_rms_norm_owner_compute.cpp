// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// dedicated_owners experiment: the OWNER compute of one row-group, on a core that reads no x and scales nothing.
// For every (wave, owner slot) with rows ("slot", (w, k) order): chip_sum (the baseline's reduce<SUM, REDUCE_COL,
// AccumulateViaAdd, Skip> over the G_W gathered partials -> cb_chip_partial); then for every slot the baseline's
// cross_device_finalize (copy_tile with UnpackToDestFp32 + SFPU add_binary_tile in group-position order, the row-0
// finalize) -> cb_rms_root. Same arithmetic, same fold order as distributed_rms_norm_compute.cpp: bit-identical.
// Order: every chip sum first (its inputs land before any wave's records can return), then the folds in slot order.
// Raw LLK: the same fold / finalize sequence the baseline compute uses (no new bypass; see that file's header).

#include <stdint.h>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

#if defined(DRMS_PERF_ZONES) && DRMS_PERF_ZONES
#include "tools/profiler/kernel_profiler.hpp"
#define DRMS_ZONE(name) DeviceZoneScopedN(name)
#else
#define DRMS_ZONE(name)
#endif
#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"
#include "sfpu/ckernel_sfpu_converter.h"
#endif

#ifndef DRMS_OWNER_PUBLISH_ONLY
#define DRMS_OWNER_PUBLISH_ONLY 0
#endif

namespace ckl = compute_kernel_lib;

namespace {
#ifdef TRISC_MATH
sfpi_inline void finalize_row0_body(const uint32_t inv_w_bits, const uint32_t eps_bits) {
#pragma GCC unroll 0
    for (int v = 0; v < 2; v++) {
        const sfpi::vFloat mean_sq = sfpi::dst_reg[0] * ckernel::sfpu::Converter::as_float(inv_w_bits) +
                                     ckernel::sfpu::Converter::as_float(eps_bits);
        sfpi::vFloat t = ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, false>(mean_sq);
        if constexpr (!DST_ACCUM_MODE) {
            t = sfpi::convert<sfpi::vFloat16b>(t, sfpi::RoundMode::Nearest);
        }
        v_if(t == 0.0f) { t = sfpi::setsgn(t, 1); }
        v_endif;
        sfpi::dst_reg[0] = t;
        sfpi::dst_reg++;
    }
}
#endif

constexpr uint32_t cb_reduce_scaler = 2;
constexpr uint32_t cb_gather = 10;
constexpr uint32_t cb_rms_root = 11;
constexpr uint32_t cb_chip_partial = 14;
constexpr uint32_t cb_stats = 15;
}  // namespace

void kernel_main() {
    constexpr uint32_t NW = get_compile_time_arg_val(7);
    constexpr uint32_t Q_W = get_compile_time_arg_val(8);
    constexpr uint32_t OWNERS = get_compile_time_arg_val(9);
    constexpr uint32_t Q = get_compile_time_arg_val(10);
    constexpr uint32_t G = get_compile_time_arg_val(11);
    constexpr uint32_t G_W = get_compile_time_arg_val(12);

    const uint32_t rows = get_arg_val<uint32_t>(0);
    const uint32_t inv_w_bits = get_arg_val<uint32_t>(1);
    const uint32_t eps_bits = get_arg_val<uint32_t>(2);

    compute_kernel_hw_startup(cb_gather, cb_gather, cb_chip_partial);

    auto n_wave = [&](uint32_t w) -> uint32_t {
        const uint32_t r0 = w * Q_W;
        return r0 >= rows ? 0 : ((rows - r0) < Q_W ? rows - r0 : Q_W);
    };
    auto n_own = [&](uint32_t w, uint32_t k) -> uint32_t {
        const uint32_t n = n_wave(w);
        const uint32_t first = k * Q;
        return first >= n ? 0 : ((n - first) < Q ? n - first : Q);
    };

    for (uint32_t w = 0; w < NW; ++w) {
        for (uint32_t k = 0; k < OWNERS; ++k) {
            const uint32_t no = n_own(w, k);
            if (no == 0) {
                continue;
            }
            {
                DRMS_ZONE("ocm_wait_gather");
                cb_wait_front(cb_gather, G_W * Q);
            }
            DRMS_ZONE("ocm_chipsum");
            ckl::reduce<
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_COL,
                cb_gather,
                cb_reduce_scaler,
                cb_chip_partial,
                ckl::ReduceInputPolicy::BulkWaitBulkPop,
                ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                ReduceFp32Mode::Fast,
                ckl::ReduceAlgorithm::AccumulateViaAdd,
                ckl::ReduceWithinTile::Skip>(
                ckl::ReduceInputBlockShape::of(G_W, no),
                ckl::ReduceInputMemoryLayout::with_row_stride(Q),
                ckl::NoAccumulation{},
                ckl::NoOp{});
        }
    }

    for (uint32_t w = 0; w < NW && !DRMS_OWNER_PUBLISH_ONLY; ++w) {
        for (uint32_t k = 0; k < OWNERS; ++k) {
            const uint32_t no = n_own(w, k);
            if (no == 0) {
                continue;
            }
            {
                DRMS_ZONE("ocm_wait_stats");
                cb_wait_front(cb_stats, G * Q);
            }
            DRMS_ZONE("ocm_fold");
            reconfig_data_format_srca(cb_stats);
            pack_reconfig_data_format(cb_rms_root);
            cb_reserve_back(cb_rms_root, no);
            copy_init(cb_stats);
            add_binary_tile_init();
            rsqrt_tile_init();
            for (uint32_t r = 0; r < no; ++r) {
                tile_regs_acquire();
                copy_tile(cb_stats, r, 0);
                for (uint32_t pos = 1; pos < G; ++pos) {
                    copy_tile(cb_stats, pos * Q + r, 1);
                    add_binary_tile(0, 1, 0);
                }
                MATH((_llk_math_eltwise_unary_sfpu_params_(
                    finalize_row0_body, 0, ckernel::VectorMode::R, inv_w_bits, eps_bits)));
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_rms_root);
                tile_regs_release();
            }
            cb_push_back(cb_rms_root, no);
            cb_pop_front(cb_stats, G * Q);
        }
    }
    cb_pop_front(cb_reduce_scaler, 1);
}
