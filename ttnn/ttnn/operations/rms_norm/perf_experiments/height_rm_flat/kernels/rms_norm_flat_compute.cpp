// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm FLAT compute (TRISC0/1/2) — HEIGHT_SHARDED ROW_MAJOR, few sticks per Tensix core.
//
// Each stick is ceil(W / 1024) "flat tiles" (1024 consecutive datums each; see rms_norm_flat_reader.cpp). Per stick:
//   pass 1     sum_j x_j * x_j, DEST-accumulated over the stick's flat tiles (FPU ELWMUL) -> cb_sumsq (1 tile)
//   collapse   reduce<SUM, REDUCE_SCALAR, ReduceTile> (FPU, 1.0 scaler) + finalize (rsqrt(sum / W + eps), column 0
//              incl. element (0, 0)) -> cb_inv_rms (1 tile, the stick's inv_rms at (0, 0))
//   pass 2     out_j = (x_j * bcast_scalar(inv)) [* gamma_j via DEST reuse] -> cb_out (zero-copy: the output shard)
// gamma_j = flat tile j of the RM gamma page (same datum order as x_j). Zero padding past W (staged slots / tail
// slot) adds 0 to sum x^2; the output datums past W are never written back.
// No tilize / untilize, no 32x row padding: work per stick is 3 * ceil(W / 1024) tile ops + one scalar reduce,
// against 5 * W / 32 tile ops per (up to 32-stick) tile-row on the tilized path.
//
// Raw compute API (documented helper bypass): eltwise_chain addresses an input by tile index with a unit column step
// (Strided{base, row_stride}); the zero-copy cb_x has pages of gcd(stick pitch, tile bytes), so flat tile j of stick
// s is page s * stick_pages + j * tile_pages (non-unit column step), and a stick's last flat tile may come from a
// second CB (cb_x_rm / cb_out_rm tail slot). Capability gap (addressing), not ergonomics. The finalize is the op's
// column-0 SFPU body (FinalizeInvRms, see rms_norm_compute.cpp).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "rms_norm_common.hpp"
#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"
#include "sfpu/ckernel_sfpu_converter.h"
#endif

using namespace compute_kernel_lib;
using namespace rms_norm_cb;

constexpr uint32_t n_body = get_compile_time_arg_val(0);        // flat tiles per stick in cb_x
constexpr bool has_tail_cb = get_compile_time_arg_val(1) != 0;  // + 1 tail flat tile per stick in cb_x_rm / cb_out_rm
constexpr uint32_t stick_pages = get_compile_time_arg_val(2);   // cb_x / cb_out pages per stick
constexpr uint32_t tile_pages = get_compile_time_arg_val(3);    // cb_x / cb_out pages per flat tile
constexpr uint32_t per_stick = n_body + (has_tail_cb ? 1 : 0);  // flat tiles per stick (= gamma flat tiles)
// a CB whose page is a whole tile (hw startup / format inits read the page size as the tile size)
constexpr uint32_t cb_xa = has_tail_cb ? cb_x_rm : cb_x;
constexpr uint32_t cb_outa = has_tail_cb ? cb_out_rm : cb_out;

#ifdef TRISC_MATH
// rsqrt(sum * (1/W) + eps) on column 0 of DST[dst] (even-parity vectors of faces 0/2) — identical to the op's.
sfpi_inline void finalize_col0_body(const uint32_t inv_w_bits, const uint32_t eps_bits) {
#pragma GCC unroll 0
    for (int rg = 0; rg < 4; rg++) {
        const sfpi::vFloat inv_w = ckernel::sfpu::Converter::as_float(inv_w_bits);
        const sfpi::vFloat eps = ckernel::sfpu::Converter::as_float(eps_bits);
        sfpi::vFloat y =
            ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, false>(sfpi::dst_reg[0] * inv_w + eps);
        if constexpr (!DST_ACCUM_MODE) {
            y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = y;
        sfpi::dst_reg += 2;
    }
}
// Flat: the stick's sum sits at element (0, 0) only (REDUCE_SCALAR) and pass 2 broadcasts element (0, 0) only
// (BroadcastType::SCALAR) -> finalize the ONE SFPU vector holding it (face 0, first vector) instead of column 0's 8.
sfpi_inline void finalize_scalar_body(const uint32_t inv_w_bits, const uint32_t eps_bits) {
    const sfpi::vFloat inv_w = ckernel::sfpu::Converter::as_float(inv_w_bits);
    const sfpi::vFloat eps = ckernel::sfpu::Converter::as_float(eps_bits);
    sfpi::vFloat y =
        ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, false>(sfpi::dst_reg[0] * inv_w + eps);
    if constexpr (!DST_ACCUM_MODE) {
        y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
    }
    sfpi::dst_reg[0] = y;
}
#endif

struct FinalizeInvRms {
    uint32_t inv_w_bits;
    uint32_t eps_bits;
    ALWI void operator()(uint32_t dst) const {
        rsqrt_tile_init();
#ifdef RMS_FLAT_FIN_COL0
        MATH((_llk_math_eltwise_unary_sfpu_params_(finalize_col0_body, dst, VectorMode::C, inv_w_bits, eps_bits)));
#else
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            finalize_scalar_body, dst, VectorMode::RC_custom, inv_w_bits, eps_bits)));
#endif
    }
};

ALWI uint32_t body_page(uint32_t s, uint32_t j) { return s * stick_pages + j * tile_pages; }

void kernel_main() {
#ifdef RMS_ABLATE_EMPTY
    return;
#endif
    const uint32_t sticks = get_arg_val<uint32_t>(0);
    const FinalizeInvRms finalize{get_arg_val<uint32_t>(1), get_arg_val<uint32_t>(2)};
    constexpr uint32_t batch = DEST_AUTO_LIMIT;

    compute_kernel_hw_startup(cb_xa, cb_xa, cb_sumsq_tiles);

    {
        MaybeDeviceZoneScope("cmp_wait_x");
        cb_wait_front(cb_x, sticks * stick_pages);
        if constexpr (has_tail_cb) {
            cb_wait_front(cb_x_rm, sticks);
        }
    }
    // ---- pass 1: per stick, sum_j x_j^2 accumulated in one DEST tile (ELWMUL accumulates into DEST) ----
    {
        MaybeDeviceZoneScope("cmp_pass1");
        reconfig_data_format(cb_xa, cb_xa);
        pack_reconfig_data_format(cb_sumsq_tiles);
        mul_init(cb_xa, cb_xa, true);
        for (uint32_t s0 = 0; s0 < sticks; s0 += batch) {
            const uint32_t n = (sticks - s0) < batch ? (sticks - s0) : batch;
            cb_reserve_back(cb_sumsq_tiles, n);
            tile_regs_acquire();
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t s = s0 + i;
                for (uint32_t j = 0; j < n_body; ++j) {
                    const uint32_t p = body_page(s, j);
                    mul_tiles(cb_x, cb_x, p, p, i);
                }
                if constexpr (has_tail_cb) {
                    mul_tiles(cb_x_rm, cb_x_rm, s, s, i);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < n; ++i) {
                pack_tile(i, cb_sumsq_tiles);
            }
            tile_regs_release();
            cb_push_back(cb_sumsq_tiles, n);
        }
    }
    // ---- collapse: whole-tile sum (FPU reduce SCALAR) + finalize -> inv_rms at (0, 0) ----
    {
        MaybeDeviceZoneScope("cmp_collapse");
        reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_SCALAR,
            cb_sumsq_tiles,
            cb_ones,
            cb_inv_rms,
            ReduceInputPolicy::BulkWaitBulkPop,
            ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
            ReduceFp32Mode::Fast,
            ReduceAlgorithm::ReduceTile>(
            ReduceInputBlockShape::of(1, 1, sticks), ReduceInputMemoryLayout::contiguous(), NoAccumulation{}, finalize);
    }
    cb_wait_front(cb_inv_rms, sticks);
#ifdef RMS_HAS_GAMMA
    {
        MaybeDeviceZoneScope("cmp_wait_gamma");  // flat gamma (writer load / multicast)
        cb_wait_front(cb_gamma, per_stick);
    }
#endif
    // ---- pass 2: out_j = x_j * inv (bcast scalar) [* gamma_j, DEST reuse], packed in place ----
    {
        MaybeDeviceZoneScope("cmp_pass2");
        cb_reserve_back(cb_out, sticks * stick_pages);
        if constexpr (has_tail_cb) {
            cb_reserve_back(cb_out_rm, sticks);
        }
        pack_reconfig_data_format(cb_outa);
        const uint32_t total = sticks * per_stick;
        for (uint32_t t0 = 0; t0 < total; t0 += batch) {
            const uint32_t n = (total - t0) < batch ? (total - t0) : batch;
            tile_regs_acquire();
            reconfig_data_format(cb_xa, cb_inv_rms);
            mul_bcast_scalar_init(cb_xa, cb_inv_rms);
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t s = (t0 + i) / per_stick;
                const uint32_t j = (t0 + i) - s * per_stick;
                if (j < n_body) {
                    mul_tiles_bcast_scalar(cb_x, cb_inv_rms, body_page(s, j), s, i);
                } else {
                    mul_tiles_bcast_scalar(cb_x_rm, cb_inv_rms, s, s, i);
                }
            }
#ifdef RMS_HAS_GAMMA
            reconfig_data_format_srcb(cb_inv_rms, cb_gamma);
            mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_gamma);
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t s = (t0 + i) / per_stick;
                const uint32_t j = (t0 + i) - s * per_stick;
                mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_gamma, j, i);
            }
#endif
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t s = (t0 + i) / per_stick;
                const uint32_t j = (t0 + i) - s * per_stick;
                if (j < n_body) {
                    pack_tile<true>(i, cb_out, body_page(s, j));
                } else {
                    pack_tile<true>(i, cb_out_rm, s);
                }
            }
            tile_regs_release();
        }
        cb_push_back(cb_out, sticks * stick_pages);
        if constexpr (has_tail_cb) {
            cb_push_back(cb_out_rm, sticks);
        }
    }
    cb_pop_front(cb_x, sticks * stick_pages);
    if constexpr (has_tail_cb) {
        cb_pop_front(cb_x_rm, sticks);
    }
    cb_pop_front(cb_inv_rms, sticks);
#ifdef RMS_HAS_GAMMA
    cb_pop_front(cb_gamma, per_stick);
#endif
}
