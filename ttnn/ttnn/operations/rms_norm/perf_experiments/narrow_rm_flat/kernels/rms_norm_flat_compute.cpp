// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm flat compute (TRISC0/1/2) — ROW_MAJOR WIDTH / BLOCK_SHARDED shards of width w_s | 16 datums, computed on
// "flat tiles" (1024 consecutive shard datums; a stick = w_s consecutive logical columns of one logical row).
//
//   pass 1 (cmp_pass1)  per unit u: sq_f = x_f * x_f (FPU, DEST batches -> cb_sq), then
//                       FLAT_CMP : P_u = sum_{j < n_u} sq_{uR+j} @ S_j  (one DEST tile, matmul accumulation)
//                       broadcast: P_f = sq_f @ B
//                       -> cb_partial (the writer ships unit u to its owner u % k)
//   combine (owner)     pipelined into pass 1: unit o is combined right after this member's pass 1 of unit o + lag
//                       (members progress in lockstep, so the k partials of o are in by then; the rest after pass 1).
//                       reduce<SUM, ROW, AccumulateViaAdd, Skip>(1 x k) + FULL-tile finalize rsqrt(sum / W + eps)
//                       -> cb_inv_owned (the writer returns it into slot o of every member's cb_inv)
//   pass 2 (cmp_pass2)  FLAT_CMP : DEST[0..n) = inv_u @ [E_j ...] (one matmul_block, in0 reused; gamma folded into
//                       E_j), DEST *= x_f (DEST reuse)
//                       broadcast: DEST = x_f * inv_f, DEST *= gamma_f
//                       -> cb_out (shard-backed: packed straight into the output shard) / cb_outtail (last tile)
//                       cb_inv is waited per unit (the writer pushes each unit as its owner's flag arrives).
//
// Raw LLK / raw compute API (documented helper bypasses):
//  * matmul_tiles / matmul_block — the per-stick reduction is a right-multiplication of each flat tile by a 0/1
//    matrix. compute_kernel_lib::matmul_block is built around K-blocked activations x weights with its own CB
//    lifecycle (in0 / in1 wait+pop per K-block, subblock pack); here in1 is a resident constant indexed per flat
//    tile (S_j / E_j), in0 is the per-unit x^2 batch (pass 1) or ONE inv tile reused for R outputs (pass 2), and the
//    pass-2 output must stay in DEST for the x multiply before the pack. Capability gap (no DEST-resident epilogue
//    with a CB operand, no per-tile in1 index), not ergonomics.
//  * pass 2 eltwise_chain cannot start from a matmul result in DEST, so the DEST-reuse multiply is raw
//    mul_reuse_dest_tiles<DEST_TO_SRCA> (the CB side's srcB reconfigured explicitly, as DestReuseBinary does).
//  * FinalizeFull: rsqrt(x * 1/W + eps) as one SFPU pass over the whole tile (stock binop_with_scalar + rsqrt_tile
//    would be three full-tile passes) — same body as the op's column-0 finalize, VectorMode::RC.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "rms_norm_flat_common.hpp"
#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"
#include "sfpu/ckernel_sfpu_converter.h"
#endif

using namespace compute_kernel_lib;
using namespace rms_norm_flat_cb;

constexpr uint32_t F_full = get_compile_time_arg_val(0);
constexpr bool has_tail = get_compile_time_arg_val(1) != 0;
constexpr uint32_t R = get_compile_time_arg_val(2);  // flat tiles per unit (FLAT_CMP: w_s; broadcast: 1)
constexpr uint32_t U = get_compile_time_arg_val(3);  // units per member
constexpr uint32_t K = get_compile_time_arg_val(4);  // members per group
constexpr uint32_t LAG = get_compile_time_arg_val(5);
constexpr uint32_t Q = get_compile_time_arg_val(6);   // exchange sub-blocks per unit tile
constexpr uint32_t KT = get_compile_time_arg_val(7);  // gathered tiles per owned sub-unit (k contributions, Q per tile)
constexpr uint32_t V = U * Q;
constexpr uint32_t F = F_full + (has_tail ? 1 : 0);
constexpr uint32_t DB = compute_kernel_lib::DEST_AUTO_LIMIT;
constexpr uint32_t cb_x0 = F_full > 0 ? cb_x : cb_xtail;  // format reference of x (both CBs share x's dtype)
constexpr uint32_t cb_out0 = F_full > 0 ? cb_out : cb_outtail;

FORCE_INLINE uint32_t x_cb(uint32_t f) { return f < F_full ? cb_x : cb_xtail; }
FORCE_INLINE uint32_t x_idx(uint32_t f) { return f < F_full ? f : 0; }

// Fold + finalize (owner): DEST holds the sum of the KT gathered tiles; sub-block s (1024 / Q datums = 32 / Q SFPU rows
// of the tile: dst_reg index i covers DEST rows 2i.., tile = 32 indices) carries the partial sums of members
// m == s (mod Q). out[r] = rsqrt((sum_s dst[s * NV + r]) * (1/W) + eps) for the NV = 32 / Q rows of sub-block 0
// (fp32 LREG sum, one store per row). VectorMode::None: one call over the whole tile.
#ifdef TRISC_MATH
sfpi_inline void fold_finalize_body(const uint32_t inv_w_bits, const uint32_t eps_bits) {
    constexpr uint32_t NV = 32 / Q;
#pragma GCC unroll 0
    for (uint32_t r = 0; r < NV; r++) {
        sfpi::vFloat acc = sfpi::dst_reg[0];
#pragma GCC unroll 16
        for (uint32_t q = 1; q < Q; q++) {
            acc += sfpi::dst_reg[q * NV];
        }
        const sfpi::vFloat inv_w = ckernel::sfpu::Converter::as_float(inv_w_bits);
        const sfpi::vFloat eps = ckernel::sfpu::Converter::as_float(eps_bits);
        sfpi::vFloat y = ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, false>(acc * inv_w + eps);
        if constexpr (!DST_ACCUM_MODE) {
            y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = y;
        sfpi::dst_reg++;
    }
}
#endif

struct FoldFinalize {
    uint32_t inv_w_bits;
    uint32_t eps_bits;
    ALWI void operator()(uint32_t dst) const {
        rsqrt_tile_init();
        MATH((_llk_math_eltwise_unary_sfpu_params_(fold_finalize_body, dst, VectorMode::None, inv_w_bits, eps_bits)));
    }
};

// sq_{f0+i} = x * x for i < n into cb_sq (DEST batches of DB).
ALWI void squares(uint32_t f0, uint32_t n) {
    reconfig_data_format(cb_x0, cb_x0);
    pack_reconfig_data_format(cb_sq);
    mul_init(cb_x0, cb_x0, true);
    for (uint32_t b0 = 0; b0 < n; b0 += DB) {
        const uint32_t nb = (n - b0) < DB ? (n - b0) : DB;
        cb_reserve_back(cb_sq, nb);
        tile_regs_acquire();
        for (uint32_t i = 0; i < nb; ++i) {
            const uint32_t f = f0 + b0 + i;
            mul_tiles(x_cb(f), x_cb(f), x_idx(f), x_idx(f), i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < nb; ++i) {
            pack_tile(i, cb_sq);
        }
        tile_regs_release();
        cb_push_back(cb_sq, nb);
    }
}

// pass 1 of unit u -> cb_partial (FLAT_CMP: one compressed tile; broadcast: R == 1 -> one broadcast-form tile).
ALWI uint32_t unit_tiles(uint32_t u) { return (F - u * R) < R ? (F - u * R) : R; }

ALWI void squares_unit(uint32_t u) { squares(u * R, unit_tiles(u)); }

// compress: unit u's squares (the oldest n pages of cb_sq) -> one partial tile. Software-pipelined behind the
// squares of unit u + 1 (FLAT_SQ_AHEAD): the FPU computes those while the packer drains unit u's squares, so the
// x^2 pack -> unpack round trip no longer idles the math thread.
ALWI void compress_unit(uint32_t u) {
    const uint32_t n = unit_tiles(u);
    cb_wait_front(cb_sq, n);
    cb_reserve_back(cb_partial, 1);
    reconfig_data_format(cb_mms, cb_sq);  // srcA <- in1 (S_j / B), srcB <- in0 (x^2)
    pack_reconfig_data_format(cb_partial);
    matmul_init(cb_sq, cb_mms);
    tile_regs_acquire();
    for (uint32_t j = 0; j < n; ++j) {
        matmul_tiles(cb_sq, cb_mms, j, R > 1 ? j : 0, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_partial);
    tile_regs_release();
    cb_push_back(cb_partial, 1);
    cb_pop_front(cb_sq, n);
}

ALWI void combine_one(FoldFinalize finalize) {
    reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_ROW,
        cb_gathered,
        cb_mms,  // scaler slot: never read (AccumulateViaAdd, Skip, no partial)
        cb_inv_owned,
        ReduceInputPolicy::BulkWaitBulkPop,
        ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Fast,
        ReduceAlgorithm::AccumulateViaAdd,
        ReduceWithinTile::Skip>(
        ReduceInputBlockShape::of(1, KT), ReduceInputMemoryLayout::contiguous(), NoAccumulation{}, finalize);
}

void kernel_main() {
    const uint32_t member_idx = get_arg_val<uint32_t>(0);
    [[maybe_unused]] const uint32_t owned = get_arg_val<uint32_t>(1);
    const FoldFinalize finalize{get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(3)};

    compute_kernel_hw_startup(cb_x0, cb_x0, cb_sq);

    if constexpr (F_full > 0) {
        cb_wait_front(cb_x, F_full);
    }
    if constexpr (has_tail) {
        cb_wait_front(cb_xtail, 1);
    }
#ifdef FLAT_CMP
    constexpr uint32_t n_mms = R < F ? R : F;  // S_j / E_j built for j < min(R, F)
#else
    constexpr uint32_t n_mms = 1;
#endif
    cb_wait_front(cb_mms, n_mms);

    // ---------------- pass 1 (+ pipelined owner combines) ----------------
    // owned sub-units v = member_idx + j * K, combined in order after pass 1 of unit v / Q + LAG
    uint32_t next_v = member_idx;
    {
        MaybeDeviceZoneScope("cmp_pass1");
#ifdef FLAT_NO_SQ_AHEAD
        constexpr uint32_t ahead = 0;
#else
        constexpr uint32_t ahead = 1;
#endif
        for (uint32_t u = 0; u < ahead && u < U; ++u) {
            squares_unit(u);
        }
        for (uint32_t u = 0; u < U; ++u) {
            if (u + ahead < U) {
                squares_unit(u + ahead);
            }
            compress_unit(u);
            while (next_v < V && next_v / Q + LAG <= u) {
                combine_one(finalize);
                next_v += K;
            }
        }
    }
    // Owned sub-units of the last LAG units (combine waits on the slowest member's pass 1): deferred into pass 2,
    // run just before pass 2 of the first such unit (U - LAG) — pass 2 of the earlier units hides the wait.
    auto combine_tail = [&]() {
        MaybeDeviceZoneScope("cmp_combine_tail");
        for (; next_v < V; next_v += K) {
            combine_one(finalize);
        }
    };
    constexpr uint32_t tail_unit = U > LAG ? U - LAG : 0;

    // ---------------- pass 2 ----------------
    MaybeDeviceZoneScope("cmp_pass2");  // incl. the per-unit waits on cb_inv
#if defined(RMS_HAS_GAMMA) && !defined(FLAT_CMP)
    cb_wait_front(cb_gamma, 1);
#endif
#ifdef FLAT_CMP
    cb_wait_front(cb_mme, n_mms);
#endif
    if constexpr (F_full > 0) {
        cb_reserve_back(cb_out, F_full);
    }
    pack_reconfig_data_format(cb_out0);
    uint32_t inv_ready = 0;
    for (uint32_t f0 = 0; f0 < F;) {
        // a batch never straddles two units (in0 = one inv tile per matmul_block)
        const uint32_t u = f0 / R;
        uint32_t n = R - f0 % R;
        n = n < DB ? n : DB;
        n = (F - f0) < n ? (F - f0) : n;
        if (u == tail_unit && f0 % R == 0 && next_v < V) {
            combine_tail();
            pack_reconfig_data_format(cb_out0);
        }
        if (u + 1 > inv_ready) {
            cb_wait_front(cb_inv, u + 1);
            inv_ready = u + 1;
        }
        tile_regs_acquire();
#ifdef FLAT_CMP
        reconfig_data_format(cb_mme, cb_inv);  // srcA <- E_j, srcB <- inv_u
        matmul_block_init(cb_inv, cb_mme, 0, n, 1, 1);
        matmul_block(cb_inv, cb_mme, u, f0 % R, 0, 0, n, 1, 1);
        reconfig_data_format_srcb(cb_inv, cb_x0);
        mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_x0);
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t f = f0 + i;
            mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(x_cb(f), x_idx(f), i);
        }
#else
        reconfig_data_format(cb_x0, cb_inv);
        mul_init(cb_x0, cb_inv, true);
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t f = f0 + i;
            mul_tiles(x_cb(f), cb_inv, x_idx(f), f, i);
        }
#ifdef RMS_HAS_GAMMA
        reconfig_data_format_srcb(cb_inv, cb_gamma);
        mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_gamma);
        for (uint32_t i = 0; i < n; ++i) {
            mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_gamma, 0, i);
        }
#endif
#endif
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t f = f0 + i;
            if (has_tail && f == F_full) {
                cb_reserve_back(cb_outtail, 1);
                pack_tile(i, cb_outtail);
                cb_push_back(cb_outtail, 1);
            } else {
                pack_tile(i, cb_out);
            }
        }
        tile_regs_release();
        f0 += n;
    }
    if constexpr (F_full > 0) {
        cb_push_back(cb_out, F_full);
        cb_pop_front(cb_x, F_full);
    }
    if constexpr (has_tail) {
        cb_pop_front(cb_xtail, 1);
    }
    cb_pop_front(cb_inv, U);
    cb_pop_front(cb_mms, n_mms);
#ifdef FLAT_CMP
    cb_pop_front(cb_mme, n_mms);
#elif defined(RMS_HAS_GAMMA)
    cb_pop_front(cb_gamma, 1);
#endif
}
