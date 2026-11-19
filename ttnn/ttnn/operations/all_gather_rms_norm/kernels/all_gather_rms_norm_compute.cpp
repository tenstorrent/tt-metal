// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// all_gather_rms_norm compute (TRISC0/1/2). One block per member per call: the member's whole assignment
// (rows x core_w tiles of cb_x, row pitch `pitch`), resident across every pass and popped at the end.
//
// rms_norm (ported from base_ops/rms_norm):
//   member_moments_block   raw: x*x DEST-accumulated (FPU) per tile-row -> cb_moment_a;
//                          reduce<SUM, ROW, ReduceTile> (cb_ones) -> cb_rowstat (k == 1, + finalize) / cb_partial
//   combine_finalize_block reduce<SUM, ROW, AccumulateViaAdd, Skip>(rows x k) + finalize -> cb_rowstat     (k > 1)
//   normalize_block        eltwise_chain BinaryFpu(x * inv_rms, bcast Col) [-> MulGammaRowBcast] -> cb_out
// layer_norm (pivot-shifted member moments + Chan combine, op_design.md -> Block schedule):
//   pass A        sum x per tile-row (SFPU, fp32 DEST) -> cb_moment_a; reduce ROW + * 1/n_m -> K_m = bf16(mean)
//                 (packed to the bf16 cb_km: the pack IS the rounding to a bf16 pivot)
//   pass B        c' = x - K_m (SFPU, exact for every input dtype; slice-last tiles masked to their valid
//                 columns), D1 = sum c', D2 = sum c'^2 accumulated in fp32 DEST -> cb_moment_b; reduce ROW -> cb_mstat
//   member final  mean_m = K_m + D1 / n_m, M2_m = D2 - D1^2 / n_m -> cb_partial (2 tiles per tile-row)
//   Chan combine  mu = sum n_m mean_m / W, M2 = sum M2_m + sum n_m (mean_m - mu)^2, inv_std = rsqrt(M2 / W + eps),
//                 K_g = bf16(mu), shift = (K_g - mu) * inv_std -> cb_kg, cb_rowstat, cb_shift
//   normalize     ((x - K_g) * inv_std + shift) [* gamma] [+ beta] -> cb_out (eltwise_chain, SFPU binaries)
//
// Raw API (documented exceptions):
//  * MulGammaRowBcast (rms normalize): base op's dest-reuse multiply with a ROW-broadcast CB operand
//    (DestReuseBinary has no BroadcastDim).
//  * FinalizeInvRms (rms): base op's column-0-scoped SFPU finalize through the reduce() post_reduce_op hook.
//  * rms pass 1: base op's DEST-accumulated x*x (FPU) per tile-row.
//  * ln pass A / pass B / member finalize / Chan combine: eltwise_chain keeps no SFPU accumulator in DEST across the
//    tiles of a walk (DestAccumulation is FPU-only), and the FPU path would truncate fp32 x to tf32 before the
//    pivot subtraction (big-offset precision). These are short raw loops of copy_tile / unary_bcast /
//    *_binary_tile / *_unary_tile calls, each with its own init, inside one DEST window per tile-row.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "all_gather_rms_norm_common.hpp"
#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"
#include "sfpu/ckernel_sfpu_converter.h"
#endif

using namespace compute_kernel_lib;
using namespace agrms_cb;

constexpr uint32_t w_split_cores = get_compile_time_arg_val(0);
constexpr uint32_t pitch = get_compile_time_arg_val(1);
constexpr uint32_t n_stat = get_compile_time_arg_val(2);

FORCE_INLINE uint32_t arg(uint32_t i) { return get_arg_val<uint32_t>(i); }

// ---------------------------------------------------------------------------------------------------------------
// rms_norm (base op)
// ---------------------------------------------------------------------------------------------------------------
#ifndef AGRMS_LN
#ifdef TRISC_MATH
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
#endif

struct FinalizeInvRms {
    uint32_t inv_w_bits;
    uint32_t eps_bits;
    ALWI void operator()(uint32_t dst) const {
        rsqrt_tile_init();
        MATH((_llk_math_eltwise_unary_sfpu_params_(finalize_col0_body, dst, VectorMode::C, inv_w_bits, eps_bits)));
    }
};

// sum_j x[r][j]^2 accumulated in DEST, one tile per tile-row (DEST_AUTO_LIMIT tile-rows per acquire).
ALWI void sum_squares_block(uint32_t rows, uint32_t core_w) {
    constexpr uint32_t batch = compute_kernel_lib::DEST_AUTO_LIMIT;
    reconfig_data_format(cb_x, cb_x);
    pack_reconfig_data_format(cb_moment_a);
    mul_init(cb_x, cb_x, true);
    for (uint32_t r0 = 0; r0 < rows; r0 += batch) {
        const uint32_t n = (rows - r0) < batch ? (rows - r0) : batch;
        cb_reserve_back(cb_moment_a, n);
        tile_regs_acquire();
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t base = (r0 + i) * pitch;
            for (uint32_t j = 0; j < core_w; ++j) {
                mul_tiles(cb_x, cb_x, base + j, base + j, i);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(i, cb_moment_a);
        }
        tile_regs_release();
        cb_push_back(cb_moment_a, n);
    }
}

template <uint32_t out_cb, typename PostOp>
ALWI void collapse_block(uint32_t rows, PostOp post_op) {
    reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_ROW,
        cb_moment_a,
        cb_ones,
        out_cb,
        ReduceInputPolicy::BulkWaitBulkPop,
        ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Fast,
        ReduceAlgorithm::ReduceTile>(
        ReduceInputBlockShape::of(rows, 1), ReduceInputMemoryLayout::contiguous(), NoAccumulation{}, post_op);
}

ALWI void combine_block(uint32_t rows, FinalizeInvRms finalize) {
    reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_ROW,
        cb_gathered,
        cb_x,  // scaler slot: never read under Skip
        cb_rowstat,
        ReduceInputPolicy::BulkWaitBulkPop,
        ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Fast,
        ReduceAlgorithm::AccumulateViaAdd,
        ReduceWithinTile::Skip>(
        ReduceInputBlockShape::of(rows, w_split_cores),
        ReduceInputMemoryLayout::contiguous(),
        NoAccumulation{},
        finalize);
}

#ifdef AGRMS_HAS_GAMMA
// MulGammaRowBcast (base op): DEST[d] *= row-broadcast(cb tile), dest-reuse with a ROW-broadcast srcB operand.
template <InputSpec In>
struct MulGammaRowBcast : DestReuseBinary<BinaryFpuOp::Mul, In, DestReuseType::DEST_TO_SRCA> {
    using Base = DestReuseBinary<BinaryFpuOp::Mul, In, DestReuseType::DEST_TO_SRCA>;
    static constexpr uint32_t GCb = In.cb_id;
    constexpr MulGammaRowBcast() noexcept = default;
    constexpr explicit MulGammaRowBcast(uint32_t base) noexcept : Base(base) {}

    static ALWI void init() {
        state_configure(GCb, __builtin_LINE());
        UNPACK((llk_unpack_A_init<BroadcastType::ROW, true, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(0, 0, GCb)));
        MATH((llk_math_eltwise_binary_init<
              EltwiseBinaryType::ELWMUL,
              BroadcastType::ROW,
              MATH_FIDELITY,
              EltwiseBinaryReuseDestType::DEST_TO_SRCA>(GCb, GCb, 0)));
    }

    ALWI void exec(uint32_t i_flat, uint32_t ht, uint32_t wt, uint32_t slot_offset) const {
        const uint32_t in_idx =
            compute_kernel_lib::tile_base_value<Base::Addressing>(this->tile_base) +
            compute_kernel_lib::detail::input_idx<Base::Mapping, Base::Wait, Base::Pop, Base::Addressing>(
                i_flat, ht, wt, this->row_stride);
        const uint32_t dst = to_u32(Base::DstSlot) + slot_offset;
        UNPACK((llk_unpack_A<BroadcastType::ROW, true, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(GCb, in_idx)));
        MATH((llk_math_eltwise_binary<
              EltwiseBinaryType::ELWMUL,
              BroadcastType::ROW,
              DST_ACCUM_MODE,
              MATH_FIDELITY,
              EltwiseBinaryReuseDestType::DEST_TO_SRCA>(GCb, GCb, dst, true)));
    }
};
#endif

ALWI void normalize_block(uint32_t rows, uint32_t core_w) {
    constexpr auto x_in = input(
        cb_x,
        WaitPolicy::None,
        PopPolicy::None,
        InputTileMapping::Block,
        DataFormatReconfig::Enabled,
        TileAddressing::Strided);
    constexpr auto inv_in =
        input(cb_rowstat, BroadcastDim::Col, WaitPolicy::None, PopPolicy::None, InputTileMapping::Col);
    using XTimesInv = BinaryFpu<BinaryFpuOp::Mul, x_in, inv_in>;
#ifdef AGRMS_HAS_GAMMA
    constexpr auto gamma_in = input(cb_gamma, WaitPolicy::None, PopPolicy::None, InputTileMapping::Row);
    eltwise_chain(
        IterationShape::grid(rows, core_w),
        XTimesInv{StridedTileRange{0, pitch}},
        MulGammaRowBcast<gamma_in>{},
        PackTile<output(cb_out)>{});
#else
    eltwise_chain(
        IterationShape::grid(rows, core_w), XTimesInv{StridedTileRange{0, pitch}}, PackTile<output(cb_out)>{});
#endif
}

void kernel_main() {
    const uint32_t rows = arg(0);
    const uint32_t core_w = arg(1);
    const FinalizeInvRms finalize{arg(3), arg(4)};

    compute_kernel_hw_startup(cb_x, cb_x, cb_out);
    cb_wait_front(cb_x, rows * pitch);
    sum_squares_block(rows, core_w);
    if constexpr (w_split_cores == 1) {
        collapse_block<cb_rowstat>(rows, finalize);
    } else {
        collapse_block<cb_partial>(rows, NoOp{});
        combine_block(rows, finalize);
    }
    cb_wait_front(cb_rowstat, rows);
#ifdef AGRMS_HAS_GAMMA
    cb_wait_front(cb_gamma, pitch);
#endif
    normalize_block(rows, core_w);
    cb_pop_front(cb_x, rows * pitch);
    cb_pop_front(cb_rowstat, rows);
#ifdef AGRMS_HAS_GAMMA
    cb_pop_front(cb_gamma, pitch);
#endif
}

#else  // AGRMS_LN
// ---------------------------------------------------------------------------------------------------------------
// layer_norm
// ---------------------------------------------------------------------------------------------------------------

ALWI void copy_to(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, tile, dst);
}

ALWI void bcast_col_to(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format_srcb(cb);
    unary_bcast_init<BroadcastType::COL>(cb);
    unary_bcast<BroadcastType::COL>(cb, tile, dst);
}

ALWI void add_d(uint32_t a, uint32_t b, uint32_t o) {
    add_binary_tile_init();
    add_binary_tile(a, b, o);
}
ALWI void sub_d(uint32_t a, uint32_t b, uint32_t o) {
    sub_binary_tile_init();
    sub_binary_tile(a, b, o);
}
ALWI void mul_d(uint32_t a, uint32_t b, uint32_t o) {
    mul_binary_tile_init();
    mul_binary_tile(a, b, o);
}
ALWI void mul_s(uint32_t d, uint32_t bits) {
    binop_with_scalar_tile_init();
    mul_unary_tile(d, bits);
}
ALWI void copy_d(uint32_t from, uint32_t to) {
    copy_dest_values_init();
    copy_dest_values<DST_ACCUM_MODE ? DataFormat::Float32 : DataFormat::Float16_b>(from, to);
}

// pass A: per tile-row sum of the member's x tiles (fp32 DEST) -> cb_moment_a
ALWI void sum_x_block(uint32_t rows, uint32_t core_w) {
    pack_reconfig_data_format(cb_moment_a);
    for (uint32_t r = 0; r < rows; ++r) {
        cb_reserve_back(cb_moment_a, 1);
        tile_regs_acquire();
        copy_to(cb_x, r * pitch, 0);
        for (uint32_t j = 1; j < core_w; ++j) {
            copy_to(cb_x, r * pitch + j, 1);
            add_d(0, 1, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_moment_a);
        tile_regs_release();
        cb_push_back(cb_moment_a, 1);
    }
}

// pass B: per tile-row D1 = sum c', D2 = sum c'^2 with c' = x - K_m (masked on slice-last tiles) -> cb_moment_b
ALWI void centered_moments_block(uint32_t rows, uint32_t n_segs, uint32_t segs_idx) {
    pack_reconfig_data_format(cb_moment_b);
    for (uint32_t r = 0; r < rows; ++r) {
        cb_reserve_back(cb_moment_b, 2);
        tile_regs_acquire();
        bool first = true;
        uint32_t col = 0;
        for (uint32_t s = 0; s < n_segs; ++s) {
            const uint32_t w = arg(segs_idx + 2 * s);
            const bool masked = arg(segs_idx + 2 * s + 1) != 0;
            for (uint32_t j = 0; j < w; ++j, ++col) {
                const uint32_t c = first ? 0 : 2;
                copy_to(cb_x, r * pitch + col, c);
                bcast_col_to(cb_km, r, 3);
                sub_d(c, 3, c);
#ifdef AGRMS_MASK
                if (masked && j + 1 == w) {
                    reconfig_data_format_srcb(cb_mask);
                    unary_bcast_init<BroadcastType::ROW>(cb_mask);
                    unary_bcast<BroadcastType::ROW>(cb_mask, 0, 3);
                    mul_d(c, 3, c);
                }
#endif
                if (first) {
                    copy_d(0, 1);
                    square_tile_init();
                    square_tile(1);
                    first = false;
                } else {
                    add_d(0, 2, 0);
                    square_tile_init();
                    square_tile(2);
                    add_d(1, 2, 1);
                }
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_moment_b);
        pack_tile(1, cb_moment_b);
        tile_regs_release();
        cb_push_back(cb_moment_b, 2);
    }
}

template <uint32_t in_cb, uint32_t out_cb, typename PostOp>
ALWI void row_reduce(uint32_t tiles, PostOp post_op) {
    reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_ROW,
        in_cb,
        cb_ones,
        out_cb,
        ReduceInputPolicy::BulkWaitBulkPop,
        ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Fast,
        ReduceAlgorithm::ReduceTile>(
        ReduceInputBlockShape::of(tiles, 1), ReduceInputMemoryLayout::contiguous(), NoAccumulation{}, post_op);
}

// member finalize: mean_m = K_m + D1 / n_m, M2_m = D2 - D1^2 / n_m -> cb_partial (mean_m, M2_m per tile-row)
ALWI void member_finalize_block(uint32_t rows, uint32_t inv_n_bits) {
    cb_wait_front(cb_mstat, 2 * rows);
    cb_wait_front(cb_km, rows);
    pack_reconfig_data_format(cb_partial);
    for (uint32_t r = 0; r < rows; ++r) {
        cb_reserve_back(cb_partial, 2);
        tile_regs_acquire();
        copy_to(cb_mstat, 2 * r, 0);      // D1
        copy_to(cb_mstat, 2 * r + 1, 1);  // D2
        copy_to(cb_km, r, 2);             // K_m
        copy_d(0, 3);                     // D1
        mul_s(0, inv_n_bits);             // t = D1 / n_m
        mul_d(3, 0, 3);                   // D1^2 / n_m
        sub_d(1, 3, 1);                   // M2_m
        add_d(2, 0, 2);                   // mean_m
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(2, cb_partial);
        pack_tile(1, cb_partial);
        tile_regs_release();
        cb_push_back(cb_partial, 2);
    }
    cb_pop_front(cb_mstat, 2 * rows);
    cb_pop_front(cb_km, rows);
}

// Chan combine of the k members' (mean_m, M2_m) -> K_g (cb_kg), inv_std (cb_rowstat), shift (cb_shift)
template <uint32_t src_cb>
ALWI void combine_block(uint32_t rows, uint32_t inv_w_bits, uint32_t eps_bits, uint32_t nm_idx) {
    constexpr uint32_t k = w_split_cores;
    cb_wait_front(src_cb, 2 * k * rows);
    for (uint32_t r = 0; r < rows; ++r) {
        // window 1: mu, M2, inv_std
        cb_reserve_back(cb_mu, 1);
        cb_reserve_back(cb_kg, 1);
        cb_reserve_back(cb_rowstat, 1);
        tile_regs_acquire();
        for (uint32_t m = 0; m < k; ++m) {
            const uint32_t t = (r * k + m) * 2;
            copy_to(src_cb, t, m == 0 ? 0 : 1);
            mul_s(m == 0 ? 0 : 1, arg(nm_idx + m));
            if (m > 0) {
                add_d(0, 1, 0);
            }
        }
        mul_s(0, inv_w_bits);  // mu
        for (uint32_t m = 0; m < k; ++m) {
            const uint32_t t = (r * k + m) * 2;
            copy_to(src_cb, t, 1);
            sub_d(1, 0, 1);
            square_tile_init();
            square_tile(1);
            mul_s(1, arg(nm_idx + m));
            copy_to(src_cb, t + 1, 3);
            add_d(1, 3, 1);
            if (m == 0) {
                copy_d(1, 2);
            } else {
                add_d(2, 1, 2);
            }
        }
        mul_s(2, inv_w_bits);
        binop_with_scalar_tile_init();
        add_unary_tile(2, eps_bits);
        rsqrt_tile_init();
        rsqrt_tile(2);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_mu);
        pack_tile(0, cb_mu);
        pack_reconfig_data_format(cb_kg);
        pack_tile(0, cb_kg);
        pack_reconfig_data_format(cb_rowstat);
        pack_tile(2, cb_rowstat);
        tile_regs_release();
        cb_push_back(cb_mu, 1);
        cb_push_back(cb_kg, 1);
        cb_push_back(cb_rowstat, 1);
        // window 2: shift = (K_g - mu) * inv_std
        cb_wait_front(cb_mu, 1);
        cb_wait_front(cb_kg, r + 1);
        cb_wait_front(cb_rowstat, r + 1);
        cb_reserve_back(cb_shift, 1);
        tile_regs_acquire();
        copy_to(cb_mu, 0, 0);
        copy_to(cb_kg, r, 1);
        copy_to(cb_rowstat, r, 2);
        sub_d(1, 0, 1);
        mul_d(1, 2, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_shift);
        pack_tile(1, cb_shift);
        tile_regs_release();
        cb_push_back(cb_shift, 1);
        cb_pop_front(cb_mu, 1);
    }
    cb_pop_front(src_cb, 2 * k * rows);
}

ALWI void normalize_block(uint32_t rows, uint32_t core_w) {
    constexpr auto x_in = input(
        cb_x,
        WaitPolicy::None,
        PopPolicy::None,
        InputTileMapping::Block,
        DataFormatReconfig::Enabled,
        TileAddressing::Strided);
    constexpr auto kg_in = input(cb_kg, WaitPolicy::None, PopPolicy::None, InputTileMapping::Col);
    constexpr auto inv_in = input(cb_rowstat, WaitPolicy::None, PopPolicy::None, InputTileMapping::Col);
    constexpr auto shift_in = input(cb_shift, WaitPolicy::None, PopPolicy::None, InputTileMapping::Col);
#ifdef AGRMS_HAS_GAMMA
    constexpr auto gamma_in = input(cb_gamma, WaitPolicy::None, PopPolicy::None, InputTileMapping::Row);
#endif
#ifdef AGRMS_HAS_BETA
    constexpr auto beta_in = input(cb_beta, WaitPolicy::None, PopPolicy::None, InputTileMapping::Row);
#endif
    eltwise_chain(
        IterationShape::grid(rows, core_w),
        CopyTile<x_in, Dst::D0>{StridedTileRange{0, pitch}},
        UnaryBcast<BroadcastDim::Col, kg_in, Dst::D1>{},
        SubBinary<Dst::D0, Dst::D1, Dst::D0>{},
        UnaryBcast<BroadcastDim::Col, inv_in, Dst::D1>{},
        MulBinary<Dst::D0, Dst::D1, Dst::D0>{},
        UnaryBcast<BroadcastDim::Col, shift_in, Dst::D1>{},
        AddBinary<Dst::D0, Dst::D1, Dst::D0>{},
#ifdef AGRMS_HAS_GAMMA
        UnaryBcast<BroadcastDim::Row, gamma_in, Dst::D1>{},
        MulBinary<Dst::D0, Dst::D1, Dst::D0>{},
#endif
#ifdef AGRMS_HAS_BETA
        UnaryBcast<BroadcastDim::Row, beta_in, Dst::D1>{},
        AddBinary<Dst::D0, Dst::D1, Dst::D0>{},
#endif
        PackTile<output(cb_out), Dst::D0>{});
}

struct MulScalar {
    uint32_t bits;
    ALWI void operator()(uint32_t dst) const {
        binop_with_scalar_tile_init();
        mul_unary_tile(dst, bits);
    }
};

void kernel_main() {
    const uint32_t rows = arg(0);
    const uint32_t core_w = arg(1);
    const uint32_t inv_w_bits = arg(3);
    const uint32_t eps_bits = arg(4);
    const uint32_t inv_n_bits = arg(5);
    const uint32_t n_segs = arg(6);
    const uint32_t segs_idx = 7;
    const uint32_t nm_idx = segs_idx + 2 * n_segs;

    compute_kernel_hw_startup(cb_x, cb_x, cb_out);
    cb_wait_front(cb_x, rows * pitch);
    // pass A -> pivot K_m
    sum_x_block(rows, core_w);
    row_reduce<cb_moment_a, cb_km>(rows, MulScalar{inv_n_bits});
    cb_wait_front(cb_km, rows);
    // pass B -> centered moments
    centered_moments_block(rows, n_segs, segs_idx);
    row_reduce<cb_moment_b, cb_mstat>(2 * rows, NoOp{});
    member_finalize_block(rows, inv_n_bits);
    if constexpr (w_split_cores > 1) {
        combine_block<cb_gathered>(rows, inv_w_bits, eps_bits, nm_idx);
    } else {
        combine_block<cb_partial>(rows, inv_w_bits, eps_bits, nm_idx);
    }
#ifdef AGRMS_HAS_GAMMA
    cb_wait_front(cb_gamma, pitch);
#endif
#ifdef AGRMS_HAS_BETA
    cb_wait_front(cb_beta, pitch);
#endif
    cb_wait_front(cb_shift, rows);
    normalize_block(rows, core_w);
    cb_pop_front(cb_x, rows * pitch);
    cb_pop_front(cb_kg, rows);
    cb_pop_front(cb_rowstat, rows);
    cb_pop_front(cb_shift, rows);
#ifdef AGRMS_HAS_GAMMA
    cb_pop_front(cb_gamma, pitch);
#endif
#ifdef AGRMS_HAS_BETA
    cb_pop_front(cb_beta, pitch);
#endif
}
#endif
