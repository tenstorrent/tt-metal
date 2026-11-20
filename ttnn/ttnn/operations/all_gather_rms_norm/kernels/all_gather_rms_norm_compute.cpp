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
//  * owner combine (k > 8, base op R3 / Perf 2 row partial): the partials travel in ROW form, which no helper
//    produces or consumes — compute_kernel_lib::reduce has no row-form output and its post-op cannot transpose
//    (transpose_dest reprograms the math MOP / address mods), and there is no "column-reduce gather tiles, transpose,
//    finalize" combine helper. rms: member collapse = matmul ones-row x sumsq^T (unpacker transpose; base op
//    collapse_rows_to_row0), owner = reduce_tile COL over the gather tiles + transpose_dest + finalize (base op
//    combine_rows_block). ln: member finalize transposes (mean_m, M2_m) in DEST before the pack; the owner's Chan
//    combine is the raw SFPU loop of the symmetric path plus a ones-row matmul for the sums over members.
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
#ifdef AGRMS_OWNER
#include "api/compute/reduce.h"
#include "api/compute/matmul.h"
#endif
#include "api/compute/transpose_dest.h"
#ifdef AGRMS_LN
#include "api/compute/transpose.h"
#include "api/compute/welford.h"
#endif
#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"
#include "sfpu/ckernel_sfpu_converter.h"
#endif

using namespace compute_kernel_lib;
using namespace agrms_cb;

#ifdef AGRMS_HAS_GAMMA
constexpr bool kHasGamma = true;
#else
constexpr bool kHasGamma = false;
#endif
#ifdef AGRMS_HAS_BETA
constexpr bool kHasBeta = true;
#else
constexpr bool kHasBeta = false;
#endif

constexpr uint32_t w_split_cores = get_compile_time_arg_val(0);
constexpr uint32_t pitch = get_compile_time_arg_val(1);
constexpr uint32_t n_stat = get_compile_time_arg_val(2);
constexpr uint32_t max_fin = get_compile_time_arg_val(3);  // dense: foreign slots per tile-row
constexpr uint32_t gather_pages = get_compile_time_arg_val(5);  // owner combine: gather tiles per owned row
#ifdef AGRMS_DENSE
#include "api/compute/matmul.h"
constexpr uint32_t norm_out_cb = cb_norm;  // the member block is assembled into dense output tiles afterwards
#else
constexpr uint32_t norm_out_cb = cb_out;
#endif

FORCE_INLINE uint32_t arg(uint32_t i) { return get_arg_val<uint32_t>(i); }

#ifdef AGRMS_DENSE
// assemble_dense_block (w_non_aligned): output tile j of a tile-row = sum over its source runs of
// (normalized member tile) x S (0/1 shift matrix, S[e][e + delta] = 1) [+ peers' partial tiles x identity].
// Raw matmul_tiles (documented helper bypass): matmul_block would multiply all member tiles by a dense K, while an
// output tile needs <= 2-3 banded sources; one DEST tile accumulates them.
// RT layout at d: n_mats, id_mat, n_home, per home: (n_src | n_for << 8), src (lj | mat << 16)..., slot...;
//                 n_fout, per contribution: n_src, src...
ALWI void mm_src(uint32_t in0_cb, uint32_t in0_tile, uint32_t mat) {
    reconfig_data_format(cb_shift_mats, in0_cb);  // matmul: in1 -> srcA, in0 -> srcB
    matmul_init(in0_cb, cb_shift_mats);
    matmul_tiles(in0_cb, cb_shift_mats, in0_tile, mat, 0);
}

ALWI void dense_assemble_block(uint32_t rows, uint32_t core_w, uint32_t d) {
    const uint32_t n_mats = arg(d);
    const uint32_t id_mat = arg(d + 1);
    const uint32_t n_home = arg(d + 2);
    uint32_t fo = d + 3;  // first word after the home list
    for (uint32_t h = 0; h < n_home; ++h) {
        const uint32_t hdr = arg(fo);
        fo += 1 + (hdr & 0xFF) + (hdr >> 8);
    }
    const uint32_t n_fout = arg(fo);
    cb_wait_front(cb_norm, rows * core_w);
    cb_wait_front(cb_shift_mats, n_mats);
    // contributions to peers' home tiles
    if (n_fout > 0) {
        pack_reconfig_data_format(cb_contrib);
        for (uint32_t r = 0; r < rows; ++r) {
            uint32_t a = fo + 1;
            for (uint32_t f = 0; f < n_fout; ++f) {
                const uint32_t n_src = arg(a);
                cb_reserve_back(cb_contrib, 1);
                tile_regs_acquire();
                for (uint32_t i = 0; i < n_src; ++i) {
                    const uint32_t w = arg(a + 1 + i);
                    mm_src(cb_norm, r * core_w + (w & 0xFFFF), w >> 16);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_contrib);
                tile_regs_release();
                cb_push_back(cb_contrib, 1);
                a += 1 + n_src;
            }
        }
    }
    // home tiles
    if constexpr (max_fin > 0) {
        bool any_foreign = false;
        uint32_t a = d + 3;
        for (uint32_t h = 0; h < n_home; ++h) {
            const uint32_t hdr = arg(a);
            any_foreign |= (hdr >> 8) != 0;
            a += 1 + (hdr & 0xFF) + (hdr >> 8);
        }
        if (any_foreign) {
            cb_wait_front(cb_foreign, rows * max_fin);
        }
    }
    pack_reconfig_data_format(cb_out);
    for (uint32_t r = 0; r < rows; ++r) {
        uint32_t a = d + 3;
        for (uint32_t h = 0; h < n_home; ++h) {
            const uint32_t hdr = arg(a);
            const uint32_t n_src = hdr & 0xFF, n_for = hdr >> 8;
            cb_reserve_back(cb_out, 1);
            tile_regs_acquire();
            for (uint32_t i = 0; i < n_src; ++i) {
                const uint32_t w = arg(a + 1 + i);
                mm_src(cb_norm, r * core_w + (w & 0xFFFF), w >> 16);
            }
            for (uint32_t i = 0; i < n_for; ++i) {
                mm_src(cb_foreign, r * max_fin + arg(a + 1 + n_src + i), id_mat);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, 1);
            a += 1 + n_src + n_for;
        }
    }
    cb_pop_front(cb_norm, rows * core_w);
    cb_pop_front(cb_shift_mats, n_mats);
    if constexpr (max_fin > 0) {
        bool any_foreign = false;
        uint32_t a = d + 3;
        for (uint32_t h = 0; h < n_home; ++h) {
            const uint32_t hdr = arg(a);
            any_foreign |= (hdr >> 8) != 0;
            a += 1 + (hdr & 0xFF) + (hdr >> 8);
        }
        if (any_foreign) {
            cb_pop_front(cb_foreign, rows * max_fin);
        }
    }
}
#endif

// ---------------------------------------------------------------------------------------------------------------
// rms_norm (base op)
// ---------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------
// shared block operations
// ---------------------------------------------------------------------------------------------------------------

// sum_j a[r][j]^2 accumulated in DEST (FPU ELWMUL acc-to-dest), one tile per tile-row of `in_cb` (row stride
// `stride`), DEST_AUTO_LIMIT tile-rows per acquire -> out_cb (base op pass 1).
template <uint32_t in_cb, uint32_t out_cb>
ALWI void sum_squares_block(uint32_t rows, uint32_t core_w, uint32_t stride) {
    constexpr uint32_t batch = compute_kernel_lib::DEST_AUTO_LIMIT;
    reconfig_data_format(in_cb, in_cb);
    pack_reconfig_data_format(out_cb);
    mul_init(in_cb, in_cb, true);
    for (uint32_t r0 = 0; r0 < rows; r0 += batch) {
        const uint32_t n = (rows - r0) < batch ? (rows - r0) : batch;
        cb_reserve_back(out_cb, n);
        tile_regs_acquire();
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t base = (r0 + i) * stride;
            for (uint32_t j = 0; j < core_w; ++j) {
                mul_tiles(in_cb, in_cb, base + j, base + j, i);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(i, out_cb);
        }
        tile_regs_release();
        cb_push_back(out_cb, n);
    }
}

// DestReuseBcast: DEST[d] = DEST[d] (op) broadcast(cb tile), dest-reuse with a broadcast srcB operand.
// Generalizes the base op's MulGammaRowBcast (compute_kernel_lib::DestReuseBinary has no BroadcastDim): ROW reads
// row 0 of faces 0/1 (gamma / beta pages), COL reads column 0 (per-row statistics). Inherits every chain trait
// (waits, srcB reconfig fold, tile addressing) from DestReuseBinary; init/exec replaced.
// SrcaIn / SrcaOut: DEST -> srcA moves the intermediate in the CURRENT math-side srcA format, i.e. the chain head's
// (cb_x: bf16 / bfloat8_b). Left alone, every dest-reuse step truncates the fp32 DEST value to that 16-bit format
// (measured: -0.15 % per step, a -0.6 % per-row scale bias on the 4-step LN normalize chain, -0.3 % on rms). The first
// dest-reuse step of a chain (SrcaIn) switches the math-side srcA format to the DEST width (cb_rowstat = T_acc) and the
// last one (SrcaOut) restores the chain head's (cb_x) for the next tile (unpack + math: a math-only switch leaves
// the bias in place, measured); a no-op when the formats agree (fp32 input, or 16-bit DEST).
template <EltwiseBinaryType Op, BroadcastType BT, InputSpec In, bool SrcaIn = true, bool SrcaOut = true>
struct DestReuseBcast : DestReuseBinary<
                            (Op == EltwiseBinaryType::ELWMUL ? BinaryFpuOp::Mul : BinaryFpuOp::Add),
                            In,
                            DestReuseType::DEST_TO_SRCA> {
    using Base = DestReuseBinary<
        (Op == EltwiseBinaryType::ELWMUL ? BinaryFpuOp::Mul : BinaryFpuOp::Add),
        In,
        DestReuseType::DEST_TO_SRCA>;
    static constexpr uint32_t GCb = In.cb_id;
    constexpr DestReuseBcast() noexcept = default;
    constexpr explicit DestReuseBcast(uint32_t base) noexcept : Base(base) {}

    static ALWI void init() {
        state_configure(GCb, __builtin_LINE());
        UNPACK((llk_unpack_A_init<BT, true, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(0, 0, GCb)));
        MATH((llk_math_eltwise_binary_init<Op, BT, MATH_FIDELITY, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
            GCb, GCb, 0)));
    }

    ALWI void exec(uint32_t i_flat, uint32_t ht, uint32_t wt, uint32_t slot_offset) const {
        const uint32_t in_idx =
            compute_kernel_lib::tile_base_value<Base::Addressing>(this->tile_base) +
            compute_kernel_lib::detail::input_idx<Base::Mapping, Base::Wait, Base::Pop, Base::Addressing>(
                i_flat, ht, wt, this->row_stride);
        const uint32_t dst = to_u32(Base::DstSlot) + slot_offset;
        if constexpr (SrcaIn) {
            reconfig_data_format_srca(cb_x, cb_rowstat);
        }
        UNPACK((llk_unpack_A<BT, true, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(GCb, in_idx)));
        MATH((llk_math_eltwise_binary<Op, BT, DST_ACCUM_MODE, MATH_FIDELITY, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
            GCb, GCb, dst, true)));
        if constexpr (SrcaOut) {
            reconfig_data_format_srca(cb_rowstat, cb_x);
        }
    }
};
template <InputSpec In>
using MulGammaRowBcast = DestReuseBcast<EltwiseBinaryType::ELWMUL, BroadcastType::ROW, In>;

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

#ifdef AGRMS_OWNER
// collapse_rows_to_row0 (base op): per tile-row, DST row 0 = ones-row x sumsq^T (in1 transposed by the unpacker) =
// the 32 row sums in ROW form (face 0 row 0 + face 1 row 0) -> cb_partial
ALWI void collapse_rows_to_row0(uint32_t rows) {
    constexpr uint32_t batch = compute_kernel_lib::DEST_AUTO_LIMIT;
    cb_wait_front(cb_ones, 1);
    for (uint32_t r0 = 0; r0 < rows; r0 += batch) {
        const uint32_t n = (rows - r0) < batch ? (rows - r0) : batch;
        cb_wait_front(cb_moment_a, n);
        cb_reserve_back(cb_partial, n);
        reconfig_data_format(cb_moment_a, cb_ones);  // srcA = in1 (sumsq), srcB = in0 (ones)
        pack_reconfig_data_format(cb_partial);
        matmul_init(cb_ones, cb_moment_a, 1 /*transpose in1*/);
        tile_regs_acquire();
        for (uint32_t i = 0; i < n; ++i) {
            matmul_tiles(cb_ones, cb_moment_a, 0, i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(i, cb_partial);
        }
        tile_regs_release();
        cb_push_back(cb_partial, n);
        cb_pop_front(cb_moment_a, n);
    }
}

// combine_rows_block (base op): per owned row, column-reduce its gather tiles (1.0 scaler: row 0 = sum over the
// members), transpose row 0 -> column 0 in DEST, finalize column 0 -> cb_own_ret (the writer returns it)
ALWI void combine_rows_block(uint32_t owned, FinalizeInvRms finalize) {
    cb_wait_front(cb_gathered, owned * gather_pages);
    cb_wait_front(cb_ones, 1);
    for (uint32_t j = 0; j < owned; ++j) {
        cb_reserve_back(cb_own_ret, 1);
        reconfig_data_format(cb_gathered, cb_ones);  // REDUCE_COL: data -> srcA, scaler -> srcB
        pack_reconfig_data_format(cb_own_ret);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_gathered, cb_ones, cb_own_ret);
        tile_regs_acquire();
        for (uint32_t g = 0; g < gather_pages; ++g) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_gathered, cb_ones, j * gather_pages + g, 0, 0);
        }
        reduce_uninit();
        transpose_dest_init<DST_ACCUM_MODE>();
        transpose_dest<DST_ACCUM_MODE>(0);
        finalize(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_own_ret);
        tile_regs_release();
        cb_push_back(cb_own_ret, 1);
    }
    cb_pop_front(cb_gathered, owned * gather_pages);
}
#endif

#ifdef AGRMS_XG_FIRST
// OVERLAP (Refinement 4): the statistics-independent half of the normalize. x * gamma (gamma ROW-broadcast) of the
// whole member block -> cb_xg (accumulator format), run while the member waits for its group's partials, so only
// the one multiply by inv_rms is left on the chain after the combine (normalize_block below).
ALWI void xg_block(uint32_t rows, uint32_t core_w) {
    constexpr auto x_in = input(
        cb_x,
        WaitPolicy::None,
        PopPolicy::None,
        InputTileMapping::Block,
        DataFormatReconfig::Enabled,
        TileAddressing::Strided);
    constexpr auto g_in = input(cb_gamma, BroadcastDim::Row, WaitPolicy::None, PopPolicy::None, InputTileMapping::Row);
    eltwise_chain(
        IterationShape::grid(rows, core_w),
        BinaryFpu<BinaryFpuOp::Mul, x_in, g_in>{StridedTileRange{0, pitch}},
        PackTile<output(cb_xg)>{});
}
#endif

ALWI void normalize_block(uint32_t rows, uint32_t core_w) {
    constexpr auto inv_in =
        input(cb_rowstat, BroadcastDim::Col, WaitPolicy::None, PopPolicy::None, InputTileMapping::Col);
#ifdef AGRMS_XG_FIRST
    constexpr auto xg_in = input(cb_xg, WaitPolicy::None, PopPolicy::None, InputTileMapping::Block);
    eltwise_chain(
        IterationShape::grid(rows, core_w),
        BinaryFpu<BinaryFpuOp::Mul, xg_in, inv_in>{},
        PackTile<output(norm_out_cb)>{});
    return;
#endif
    constexpr auto x_in = input(
        cb_x,
        WaitPolicy::None,
        PopPolicy::None,
        InputTileMapping::Block,
        DataFormatReconfig::Enabled,
        TileAddressing::Strided);
    using XTimesInv = BinaryFpu<BinaryFpuOp::Mul, x_in, inv_in>;
#ifdef AGRMS_HAS_GAMMA
    constexpr auto gamma_in = input(cb_gamma, WaitPolicy::None, PopPolicy::None, InputTileMapping::Row);
    eltwise_chain(
        IterationShape::grid(rows, core_w),
        XTimesInv{StridedTileRange{0, pitch}},
        MulGammaRowBcast<gamma_in>{},
        PackTile<output(norm_out_cb)>{});
#else
    eltwise_chain(
        IterationShape::grid(rows, core_w), XTimesInv{StridedTileRange{0, pitch}}, PackTile<output(norm_out_cb)>{});
#endif
}

void kernel_main() {
    const uint32_t rows = arg(0);
    const uint32_t core_w = arg(1);
    const FinalizeInvRms finalize{arg(3), arg(4)};

    compute_kernel_hw_startup(cb_x, cb_x, cb_out);
    {
        AGZ("c_waitx");
        cb_wait_front(cb_x, rows * pitch);
    }
    {
        AGZ("c_mom");
        sum_squares_block<cb_x, cb_moment_a>(rows, core_w, pitch);
#ifdef AGRMS_OWNER
        collapse_rows_to_row0(rows);
#else
        if constexpr (w_split_cores == 1) {
            collapse_block<cb_rowstat>(rows, finalize);
        } else {
            collapse_block<cb_partial>(rows, NoOp{});
        }
#endif
    }
#ifdef AGRMS_XG_FIRST
    {
        AGZ("c_xg");
        cb_wait_front(cb_gamma, pitch);
        xg_block(rows, core_w);
        cb_wait_front(cb_xg, rows * core_w);
    }
#endif
#ifdef AGRMS_OWNER
    {
        AGZ("c_comb");
        const uint32_t owned = agrms_owner::owned_rows(rows, arg(2), w_split_cores);
        if (owned > 0) {
            combine_rows_block(owned, finalize);
        }
    }
#else
    if constexpr (w_split_cores > 1) {
        AGZ("c_comb");
        combine_block(rows, finalize);
    }
#endif
    {
        AGZ("c_waitg");
        cb_wait_front(cb_rowstat, rows);
#ifdef AGRMS_HAS_GAMMA
        cb_wait_front(cb_gamma, pitch);
#endif
    }
    {
        AGZ("c_norm");
        normalize_block(rows, core_w);
    }
#ifdef AGRMS_XG_FIRST
    cb_pop_front(cb_xg, rows * core_w);
#endif
    cb_pop_front(cb_x, rows * pitch);
    cb_pop_front(cb_rowstat, rows);
#ifdef AGRMS_HAS_GAMMA
    cb_pop_front(cb_gamma, pitch);
#endif
#ifdef AGRMS_DENSE
    dense_assemble_block(rows, core_w, 7 + 2 * arg(6) + w_split_cores);
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

// ---- 16-bit DEST member moments (Refinement 2) ----------------------------------------------------------------
// With fp32_dest_acc_en=False every DEST write is bf16 and an SFPU store TRUNCATES (round toward zero): accumulating
// sum c'^2 tile by tile in DEST (FPU or SFPU) under-estimates the variance by ~1-3 % (measured per-row scale
// +0.3..1.3 %). Here the per-row sums never touch DEST: each x tile is transposed into DEST (original row rho ->
// column rho), loaded four transposed rows (= four original columns) at a time into LREG0-3 (the Welford LLK's
// SFPTRANSP load pattern: one LREG lane per original row), and accumulated in fp32 LREGs that persist across the
// member's tiles: LREG6 = K (shift = the row's first sample), LREG4 = S1 = sum (x - K), LREG5 = S2 = sum (x - K)^2.
// The finalize forms mean_m = K + S1/n, M2_m = S2 - S1^2/n in fp32 and stores them once, rounded to nearest-even, in
// ROW form (row 0 of two DEST tiles). Raw TTI (documented helper bypass): no helper / SFPI construct keeps a
// per-row accumulator in LREGs across tiles, and the cross-lane load needs SFPTRANSP.
#ifdef TRISC_MATH
template <uint32_t I, uint32_t J>
sfpi_inline void m16_load_block() {
    constexpr uint32_t off = I * 32 + 4 * J;  // transposed rows I*16 + J*4 .. +3 of the tile at the dst base
    TTI_SFPTRANSP(0, 0, 0, 0);                // the two TRANSPs leave LREG4-7 (the state) unchanged
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, sfpi::SFPLOAD_MOD0_FMT_SRCB, ckernel::ADDR_MOD_7, off);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG1, sfpi::SFPLOAD_MOD0_FMT_SRCB, ckernel::ADDR_MOD_7, off + 2);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG2, sfpi::SFPLOAD_MOD0_FMT_SRCB, ckernel::ADDR_MOD_7, off + 16);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG3, sfpi::SFPLOAD_MOD0_FMT_SRCB, ckernel::ADDR_MOD_7, off + 18);
    TTI_SFPTRANSP(0, 0, 0, 0);
}

template <uint32_t L>
sfpi_inline void m16_sample() {
    TTI_SFPMAD(ckernel::p_sfpu::LREG11 /*-1*/, ckernel::p_sfpu::LREG6, L, ckernel::p_sfpu::LREG7, 0);  // c = x - K
    TTI_SFPMAD(ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG4, 0);
    TTI_SFPMAD(ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG5, 0);
}

// samples base .. base+3 of the tile (original columns), the first `nvalid - base` of them valid
template <uint32_t I, uint32_t J>
sfpi_inline void m16_block(uint32_t nvalid, bool first) {
    constexpr uint32_t base = I * 16 + J * 4;
    if (nvalid <= base) {
        return;
    }
    m16_load_block<I, J>();
    if (first && base == 0) {  // shift = the row's first sample; S1 = S2 = 0
        TTI_SFPMOV(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG6, 0);
        TTI_SFPLOADI(ckernel::p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_FLOATB, 0);
        TTI_SFPLOADI(ckernel::p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_FLOATB, 0);
    }
    const uint32_t n = nvalid - base;
    m16_sample<ckernel::p_sfpu::LREG0>();
    if (n > 1) {
        m16_sample<ckernel::p_sfpu::LREG1>();
    }
    if (n > 2) {
        m16_sample<ckernel::p_sfpu::LREG2>();
    }
    if (n > 3) {
        m16_sample<ckernel::p_sfpu::LREG3>();
    }
}

sfpi_inline void m16_accumulate_tile(uint32_t nvalid, uint32_t first) {
    m16_block<0, 0>(nvalid, first);
    m16_block<0, 1>(nvalid, first);
    m16_block<0, 2>(nvalid, first);
    m16_block<0, 3>(nvalid, first);
    m16_block<1, 0>(nvalid, first);
    m16_block<1, 1>(nvalid, first);
    m16_block<1, 2>(nvalid, first);
    m16_block<1, 3>(nvalid, first);
}

// mean_m -> row 0 of the tile at the dst base, M2_m -> row 0 of the next tile (both RNE bf16)
sfpi_inline void m16_finalize(uint32_t inv_n_bits) {
    using namespace ckernel;
    TT_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_UPPER, inv_n_bits >> 16);
    TT_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_LOWER, inv_n_bits & 0xFFFF);
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG7, p_sfpu::LCONST_0, p_sfpu::LREG7, 0);   // d = S1 / n
    TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LCONST_1, p_sfpu::LREG6, p_sfpu::LREG0, 0);   // mean = K + d
    TTI_SFPMAD(p_sfpu::LREG11, p_sfpu::LREG7, p_sfpu::LCONST_0, p_sfpu::LREG7, 0);  // -d
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG7, p_sfpu::LREG5, p_sfpu::LREG4, 0);      // M2 = S2 - S1 d
    TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG4, p_sfpu::LREG4, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPLOADI(p_sfpu::LREG1, 0, 0);
    TTI_SFPLOADI(p_sfpu::LREG2, 0, 0);
    TTI_SFPLOADI(p_sfpu::LREG3, 0, 0);
    TTI_SFPLOADI(p_sfpu::LREG5, 0, 0);
    TTI_SFPLOADI(p_sfpu::LREG6, 0, 0);
    TTI_SFPLOADI(p_sfpu::LREG7, 0, 0);
    TTI_SFPTRANSP(0, 0, 0, 0);  // lane rho of LREG0 / LREG4 -> row 0, column rho
    TTI_SFPSTORE(p_sfpu::LREG0, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, 0);
    TTI_SFPSTORE(p_sfpu::LREG1, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, 2);
    TTI_SFPSTORE(p_sfpu::LREG2, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, 16);
    TTI_SFPSTORE(p_sfpu::LREG3, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, 18);
    TTI_SFPSTORE(p_sfpu::LREG4, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, 64);
    TTI_SFPSTORE(p_sfpu::LREG5, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, 64 + 2);
    TTI_SFPSTORE(p_sfpu::LREG6, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, 64 + 16);
    TTI_SFPSTORE(p_sfpu::LREG7, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, 64 + 18);
}

// rows 0..3 of the tile at dst-base offset T (64 per tile) -> LREG0..3 (one row per LREG, lane = column);
// LREG4-7 unchanged (double TRANSP)
template <uint32_t T>
sfpi_inline void c16_load_rows() {
    TTI_SFPTRANSP(0, 0, 0, 0);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, sfpi::SFPLOAD_MOD0_FMT_SRCB, ckernel::ADDR_MOD_7, T);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG1, sfpi::SFPLOAD_MOD0_FMT_SRCB, ckernel::ADDR_MOD_7, T + 2);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG2, sfpi::SFPLOAD_MOD0_FMT_SRCB, ckernel::ADDR_MOD_7, T + 16);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG3, sfpi::SFPLOAD_MOD0_FMT_SRCB, ckernel::ADDR_MOD_7, T + 18);
    TTI_SFPTRANSP(0, 0, 0, 0);
}

// LREG0..3 -> rows 0..3 of the tile at offset TA, LREG4..7 -> rows 0..3 of the tile at offset TB
template <uint32_t TA, uint32_t TB, bool StoreB>
sfpi_inline void c16_store_rows() {
    TTI_SFPTRANSP(0, 0, 0, 0);
    TTI_SFPSTORE(ckernel::p_sfpu::LREG0, sfpi::SFPSTORE_MOD0_FMT_SRCB, ckernel::ADDR_MOD_7, TA);
    TTI_SFPSTORE(ckernel::p_sfpu::LREG1, sfpi::SFPSTORE_MOD0_FMT_SRCB, ckernel::ADDR_MOD_7, TA + 2);
    TTI_SFPSTORE(ckernel::p_sfpu::LREG2, sfpi::SFPSTORE_MOD0_FMT_SRCB, ckernel::ADDR_MOD_7, TA + 16);
    TTI_SFPSTORE(ckernel::p_sfpu::LREG3, sfpi::SFPSTORE_MOD0_FMT_SRCB, ckernel::ADDR_MOD_7, TA + 18);
    if constexpr (StoreB) {
        TTI_SFPSTORE(ckernel::p_sfpu::LREG4, sfpi::SFPSTORE_MOD0_FMT_SRCB, ckernel::ADDR_MOD_7, TB);
        TTI_SFPSTORE(ckernel::p_sfpu::LREG5, sfpi::SFPSTORE_MOD0_FMT_SRCB, ckernel::ADDR_MOD_7, TB + 2);
        TTI_SFPSTORE(ckernel::p_sfpu::LREG6, sfpi::SFPSTORE_MOD0_FMT_SRCB, ckernel::ADDR_MOD_7, TB + 16);
        TTI_SFPSTORE(ckernel::p_sfpu::LREG7, sfpi::SFPSTORE_MOD0_FMT_SRCB, ckernel::ADDR_MOD_7, TB + 18);
    }
}

// rows of the tile at runtime offset `off` (a 4-row block, 64 per tile) -> LREG0..3; LREG4-7 unchanged
sfpi_inline void c16_load_rows_rt(uint32_t off) {
    TTI_SFPTRANSP(0, 0, 0, 0);
    TT_SFPLOAD(ckernel::p_sfpu::LREG0, sfpi::SFPLOAD_MOD0_FMT_SRCB, ckernel::ADDR_MOD_7, off);
    TT_SFPLOAD(ckernel::p_sfpu::LREG1, sfpi::SFPLOAD_MOD0_FMT_SRCB, ckernel::ADDR_MOD_7, off + 2);
    TT_SFPLOAD(ckernel::p_sfpu::LREG2, sfpi::SFPLOAD_MOD0_FMT_SRCB, ckernel::ADDR_MOD_7, off + 16);
    TT_SFPLOAD(ckernel::p_sfpu::LREG3, sfpi::SFPLOAD_MOD0_FMT_SRCB, ckernel::ADDR_MOD_7, off + 18);
    TTI_SFPTRANSP(0, 0, 0, 0);
}

// Chan combine, one member: its mean_m is row `rho` of the tile at the dst base, its M2_m row `rho` of the next tile
// (symmetric: rho = 0 of transposed partials; owner: rho = the member's row of a gather tile). Pivoted one-pass sums
// in fp32 LREGs: LREG4 = P (member 0's mean), LREG5 = A = sum n_m (mean_m - P),
// LREG6 = B = sum M2_m + n_m (mean_m - P)^2
template <uint32_t Li>
sfpi_inline void c16_member_lane(uint32_t off, uint32_t n_bits, uint32_t first) {
    using namespace ckernel;
    c16_load_rows_rt(off);
    TTI_SFPMOV(0, Li, p_sfpu::LREG7, 0);  // mean_m (kept across the next load)
    c16_load_rows_rt(off + 64);
    if constexpr (Li != p_sfpu::LREG0) {
        TTI_SFPMOV(0, Li, p_sfpu::LREG0, 0);  // M2_m
    }
    if (first) {
        TTI_SFPMOV(0, p_sfpu::LREG7, p_sfpu::LREG4, 0);
        TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_FLOATB, 0);
        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG6, 0);
        return;
    }
    TT_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_UPPER, n_bits >> 16);
    TT_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_LOWER, n_bits & 0xFFFF);
    TTI_SFPMAD(p_sfpu::LREG11, p_sfpu::LREG4, p_sfpu::LREG7, p_sfpu::LREG7, 0);    // d = mean_m - P
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG7, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);  // n_m d
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG5, 0);  // A += n_m d
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG6, p_sfpu::LREG6, 0);  // B += M2_m
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG7, p_sfpu::LREG6, p_sfpu::LREG6, 0);     // B += n_m d^2
}

sfpi_inline void c16_member(uint32_t rho, uint32_t n_bits, uint32_t first) {
    const uint32_t off = (rho >> 4) * 32 + ((rho & 15) >> 2) * 4;  // 4-row block holding row rho
    switch (rho & 3) {
        case 0: c16_member_lane<ckernel::p_sfpu::LREG0>(off, n_bits, first); break;
        case 1: c16_member_lane<ckernel::p_sfpu::LREG1>(off, n_bits, first); break;
        case 2: c16_member_lane<ckernel::p_sfpu::LREG2>(off, n_bits, first); break;
        default: c16_member_lane<ckernel::p_sfpu::LREG3>(off, n_bits, first); break;
    }
}

// finalize (fp32): delta = A / W, mu = P + delta, M2 = B - A delta, inv_std = rsqrt(M2 / W + eps), K_g = bf16(mu),
// shift = (K_g - mu) inv_std; RNE-rounded, ROW form: K_g -> tile 0 (row 1 = shift, scratch), inv_std -> tile 1,
// shift -> tile 2 (offsets from the dst base)
sfpi_inline void c16_finalize_values(uint32_t inv_w_bits, uint32_t eps_bits) {
    const sfpi::vFloat p = sfpi::l_reg[sfpi::LRegs::LReg4];
    const sfpi::vFloat a = sfpi::l_reg[sfpi::LRegs::LReg5];
    const sfpi::vFloat b = sfpi::l_reg[sfpi::LRegs::LReg6];
    const sfpi::vFloat inv_w = ckernel::sfpu::Converter::as_float(inv_w_bits);
    const sfpi::vFloat eps = ckernel::sfpu::Converter::as_float(eps_bits);
    const sfpi::vFloat delta = a * inv_w;
    const sfpi::vFloat mu = p + delta;
    const sfpi::vFloat m2 = b - a * delta;
    sfpi::vFloat inv_std = ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, false>(m2 * inv_w + eps);
    sfpi::vFloat kg = sfpi::convert<sfpi::vFloat16b>(mu, sfpi::RoundMode::Nearest);
    sfpi::vFloat shift = (kg - mu) * inv_std;
    inv_std = sfpi::convert<sfpi::vFloat16b>(inv_std, sfpi::RoundMode::Nearest);
    shift = sfpi::convert<sfpi::vFloat16b>(shift, sfpi::RoundMode::Nearest);
    const sfpi::vFloat zero = 0.0f;
    sfpi::l_reg[sfpi::LRegs::LReg0] = kg;
    sfpi::l_reg[sfpi::LRegs::LReg1] = shift;
    sfpi::l_reg[sfpi::LRegs::LReg2] = zero;
    sfpi::l_reg[sfpi::LRegs::LReg3] = zero;
    sfpi::l_reg[sfpi::LRegs::LReg4] = inv_std;
    sfpi::l_reg[sfpi::LRegs::LReg5] = zero;
    sfpi::l_reg[sfpi::LRegs::LReg6] = zero;
    sfpi::l_reg[sfpi::LRegs::LReg7] = zero;
}

sfpi_inline void c16_finalize_store() {
    using namespace ckernel;
    c16_store_rows<0, 64, true>();  // tile 0: K_g (row 0), shift (row 1); tile 1: inv_std
    c16_load_rows<0>();             // LREG1 = shift row
    TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
    TTI_SFPLOADI(p_sfpu::LREG1, 0, 0);
    TTI_SFPLOADI(p_sfpu::LREG2, 0, 0);
    TTI_SFPLOADI(p_sfpu::LREG3, 0, 0);
    c16_store_rows<128, 0, false>();  // tile 2: shift
}
#endif

// Row form -> column form for 16-bit DEST: the row-form DEST tiles [0, n) are packed to cb_mu (caller pushed) and
// transposed back on unpack (transpose_tile), packing tile i to outs[i] — the generic Welford reduction's pattern
// (it keeps transpose_dest's replay / ALU-format reprogramming out of the per-row SFPU-config sequence).
template <uint32_t n>
ALWI void mu_rows_to_cols(const uint32_t (&outs)[n]) {
    cb_wait_front(cb_mu, n);
    reconfig_data_format_srca(cb_mu);
    transpose_init(cb_mu);
    tile_regs_acquire();
    for (uint32_t i = 0; i < n; ++i) {
        transpose_tile(cb_mu, i, i);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < n; ++i) {
        pack_reconfig_data_format(outs[i]);
        pack_tile(i, outs[i]);
    }
    tile_regs_release();
    cb_pop_front(cb_mu, n);
}

// pack the row-form DEST tiles [d, d + n) to cb_mu (inside the caller's DEST window)
ALWI void pack_rows_to_mu(uint32_t d, uint32_t n) {
    pack_reconfig_data_format(cb_mu);
    for (uint32_t i = 0; i < n; ++i) {
        pack_tile(d + i, cb_mu);
    }
}

// Chan combine for 16-bit DEST (k members' column-form (mean_m, M2_m) in src_cb) -> cb_kg, cb_rowstat, cb_shift.
// The members' tiles are transposed on unpack (row form: one LREG per statistic), combined in fp32 LREGs and stored
// once, rounded to nearest-even (the 16-bit SFPU op chain of combine_block truncates at every step).
template <uint32_t src_cb>
ALWI void combine16_block(uint32_t rows, uint32_t inv_w_bits, uint32_t eps_bits, uint32_t nm_idx) {
    constexpr uint32_t k = w_split_cores;
    cb_wait_front(src_cb, 2 * k * rows);
    for (uint32_t r = 0; r < rows; ++r) {
        cb_reserve_back(cb_kg, 1);
        cb_reserve_back(cb_rowstat, 1);
        cb_reserve_back(cb_shift, 1);
        welford_init();
        reconfig_data_format_srca(src_cb);
        transpose_init(src_cb);
        tile_regs_acquire();
        for (uint32_t m = 0; m < k; ++m) {
            const uint32_t t = (r * k + m) * 2;
            transpose_tile(src_cb, t, 0);
            transpose_tile(src_cb, t + 1, 1);
            MATH((_llk_math_welfords_sfpu_params_(c16_member, 0, 0u, arg(nm_idx + m), m == 0 ? 1u : 0u)));
        }
        rsqrt_tile_init();  // the fp32 rsqrt body's programmable constants (never left to a previous kernel)
        MATH((_llk_math_welfords_sfpu_params_(c16_finalize_values, 2, inv_w_bits, eps_bits)));
        MATH((_llk_math_welfords_sfpu_params_(c16_finalize_store, 2)));
        cb_reserve_back(cb_mu, 3);
        tile_regs_commit();
        tile_regs_wait();
        pack_rows_to_mu(2, 3);
        tile_regs_release();
        cb_push_back(cb_mu, 3);
        mu_rows_to_cols<3>({cb_kg, cb_rowstat, cb_shift});
        cb_push_back(cb_kg, 1);
        cb_push_back(cb_rowstat, 1);
        cb_push_back(cb_shift, 1);
    }
    cb_pop_front(src_cb, 2 * k * rows);
    cb_wait_front(cb_kg, rows);
    cb_wait_front(cb_rowstat, rows);
}

#ifdef AGRMS_OWNER
// Owner Chan combine for 16-bit DEST (k > 8): per owned row j, gather tile g of stat s holds member m = 32 g + rho's
// (mean_m, M2_m) in row rho (row form, lane = block row). Same fp32 LREG combine as combine16_block, one member row at
// a time -> (inv_std, shift) -> cb_own_ret, K_g -> cb_own_kg (column form; the writer returns them).
ALWI void owner16_ln(uint32_t owned, uint32_t inv_w_bits, uint32_t eps_bits, uint32_t nm_idx) {
    constexpr uint32_t gp = gather_pages;
    constexpr uint32_t k = w_split_cores;
    cb_wait_front(cb_gathered, owned * 2 * gp);
    for (uint32_t j = 0; j < owned; ++j) {
        cb_reserve_back(cb_own_ret, 2);
        cb_reserve_back(cb_own_kg, 1);
        welford_init();
        tile_regs_acquire();
        for (uint32_t g = 0; g < gp; ++g) {
            copy_to(cb_gathered, (j * 2) * gp + g, 0);
            copy_to(cb_gathered, (j * 2 + 1) * gp + g, 1);
            for (uint32_t rho = 0; rho < 32 && 32 * g + rho < k; ++rho) {
                const uint32_t m = 32 * g + rho;
                MATH((_llk_math_welfords_sfpu_params_(c16_member, 0, rho, arg(nm_idx + m), m == 0 ? 1u : 0u)));
            }
        }
        rsqrt_tile_init();  // the fp32 rsqrt body's programmable constants (never left to a previous kernel)
        MATH((_llk_math_welfords_sfpu_params_(c16_finalize_values, 2, inv_w_bits, eps_bits)));
        MATH((_llk_math_welfords_sfpu_params_(c16_finalize_store, 2)));
        cb_reserve_back(cb_mu, 3);
        tile_regs_commit();
        tile_regs_wait();
        pack_rows_to_mu(2, 3);
        tile_regs_release();
        cb_push_back(cb_mu, 3);
        mu_rows_to_cols<3>({cb_own_kg, cb_own_ret, cb_own_ret});
        cb_push_back(cb_own_ret, 2);
        cb_push_back(cb_own_kg, 1);
    }
    cb_pop_front(cb_gathered, owned * 2 * gp);
}
#endif

// member_moments_block for 16-bit DEST: per tile-row, (mean_m, M2_m) -> cb_partial (column form; owner: row form)
ALWI void moments16_block(uint32_t rows, uint32_t n_segs, uint32_t segs_idx, uint32_t inv_n_bits) {
    pack_reconfig_data_format(cb_partial);
    for (uint32_t r = 0; r < rows; ++r) {
        cb_reserve_back(cb_partial, 2);
        welford_init();  // SFPU config (LREG11 = -1) + ADDR_MOD_7; re-armed per row (transpose_dest reprograms)
        reconfig_data_format_srca(cb_x);
        transpose_init(cb_x);
        tile_regs_acquire();
        uint32_t col = 0;
        bool first = true;
        for (uint32_t s = 0; s < n_segs; ++s) {
            const uint32_t w = arg(segs_idx + 2 * s);
            const uint32_t tail_valid = arg(segs_idx + 2 * s + 1);  // valid columns of the slice-last tile, 0 = all
            for (uint32_t j = 0; j < w; ++j, ++col) {
                const uint32_t nvalid = (tail_valid != 0 && j + 1 == w) ? tail_valid : 32;
                transpose_tile(cb_x, r * pitch + col, 0);
                MATH((_llk_math_welfords_sfpu_params_(m16_accumulate_tile, 0, nvalid, first ? 1u : 0u)));
                first = false;
            }
        }
        MATH((_llk_math_welfords_sfpu_params_(m16_finalize, 1, inv_n_bits)));
#ifdef AGRMS_OWNER
        // owner combine: the partials travel in ROW form, as produced
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_partial);
        pack_tile(1, cb_partial);
        pack_tile(2, cb_partial);
        tile_regs_release();
#else
        // symmetric / k = 1: column form (column 0 = the row statistics)
        cb_reserve_back(cb_mu, 2);
        tile_regs_commit();
        tile_regs_wait();
        pack_rows_to_mu(1, 2);
        tile_regs_release();
        cb_push_back(cb_mu, 2);
        mu_rows_to_cols<2>({cb_partial, cb_partial});
#endif
        cb_push_back(cb_partial, 2);
    }
}

// reduce() post-op: DEST tile *= scalar (fp32 bits), e.g. sum x -> mean with 1/n_m
struct MulScalar {
    uint32_t bits;
    ALWI void operator()(uint32_t dst) const {
        binop_with_scalar_tile_init();
        mul_unary_tile(dst, bits);
    }
};

// k == 1 with fp32 DEST: one member holds the whole row, so the member finalize and the Chan combine collapse into
// one column-0 SFPU window (ln_finalize_k1_block) instead of ~15 full-tile SFPU init + op pairs in two windows.
//  * kK1Direct (block pivot): K_m = bf16(member mean) = bf16(row mean) IS the normalize pivot K_g: the pivot reduce
//    packs straight to cb_kg and the finalize forms only (inv_std, shift) — no bf16 re-rounding of mu.
//  * own-slice pivot (AGRMS_LN_OWN_PIVOT): K_m is the own slice's mean, so K_g = bf16_rne(mu) is formed in the finalize
//    (the normalize keeps its row-mean pivot: x - K_g stays small for the DEST -> srcA tf32 steps); K_m is staged
//    into cb_mstat ahead of (D1, D2) before the landing (stage_pivot_block).
#ifdef AGRMS_LN_OWN_PIVOT
constexpr bool kOwnPivot = true;
#else
constexpr bool kOwnPivot = false;
#endif
constexpr bool kFusedK1 = w_split_cores == 1 && DST_ACCUM_MODE;
constexpr bool kK1Direct = kFusedK1 && !kOwnPivot;
constexpr uint32_t cb_pivot = kK1Direct ? cb_kg : cb_km;
constexpr uint32_t kStagedK = (kFusedK1 && kOwnPivot) ? 1 : 0;  // K_m tiles per row ahead of (D1, D2) in cb_mstat

// k == 1 finalize, column-0 SFPU scope (the base op's finalize_col0_body pattern: faces 0/2, the even dst rows =
// columns 0-7, where a REDUCE_ROW result lives). DEST tile 0 = D1 = sum (x - K), tile 1 = D2 = sum (x - K)^2
// [, tile 2 = K (RoundKg)] ->
//   t = D1 / n, M2 = D2 - D1 t, inv_std = rsqrt(M2 / W + eps) -> tile 1
//   Direct : shift = (K - mu) inv_std = -t inv_std -> tile 0
//   RoundKg: mu = K + t, K_g = bf16_rne(mu) -> tile 2, shift = (K_g - mu) inv_std -> tile 0
// Same statistics as member_finalize_block + combine_block (k = 1), all fp32 in SFPU registers.
#ifdef TRISC_MATH
template <bool RoundKg>
sfpi_inline void ln_finalize_k1_col0_body(uint32_t inv_n_bits, uint32_t inv_w_bits, uint32_t eps_bits) {
#pragma GCC unroll 0
    for (int rg = 0; rg < 4; rg++) {
        const sfpi::vFloat inv_n = ckernel::sfpu::Converter::as_float(inv_n_bits);
        const sfpi::vFloat inv_w = ckernel::sfpu::Converter::as_float(inv_w_bits);
        const sfpi::vFloat eps = ckernel::sfpu::Converter::as_float(eps_bits);
        const sfpi::vFloat d1 = sfpi::dst_reg[0];
        const sfpi::vFloat d2 = sfpi::dst_reg[32];
        const sfpi::vFloat t = d1 * inv_n;
        const sfpi::vFloat m2 = d2 - d1 * t;
        const sfpi::vFloat inv_std =
            ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, false>(m2 * inv_w + eps);
        sfpi::dst_reg[32] = inv_std;
        if constexpr (RoundKg) {
            const sfpi::vFloat mu = sfpi::dst_reg[64] + t;
            const sfpi::vFloat kg = sfpi::convert<sfpi::vFloat16b>(mu, sfpi::RoundMode::Nearest);
            sfpi::dst_reg[0] = (kg - mu) * inv_std;
            sfpi::dst_reg[64] = kg;
        } else {
            sfpi::dst_reg[0] = -t * inv_std;
        }
        sfpi::dst_reg += 2;
    }
}
#endif

// own-slice pivot, k == 1: K_m (bf16 cb_km) -> fp32 cb_mstat AHEAD of (D1, D2), so the finalize copies its three inputs
// in one format (a bf16 copy among the fp32 UnpackToDestFp32 ones cost ~0.27 us on the indexer tail); runs before
// the landing.
ALWI void stage_pivot_block(uint32_t rows) {
    cb_wait_front(cb_km, rows);
    pack_reconfig_data_format(cb_mstat);
    reconfig_data_format_srca(cb_km);
    copy_init(cb_km);
    for (uint32_t r = 0; r < rows; ++r) {
        cb_reserve_back(cb_mstat, 1);
        tile_regs_acquire();
        copy_tile(cb_km, r, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_mstat);
        tile_regs_release();
        cb_push_back(cb_mstat, 1);
    }
}

// k == 1 (fp32 DEST): cb_mstat = [K_m x rows (staged)][(D1, D2) per tile-row] -> shift (cb_shift), inv_std
// (cb_rowstat) [, K_g (cb_kg)]
ALWI void ln_finalize_k1_block(uint32_t rows, uint32_t inv_n_bits, uint32_t inv_w_bits, uint32_t eps_bits) {
    constexpr uint32_t n_out = 2 + kStagedK;
    const uint32_t s0 = kStagedK * rows;  // first D tile
    cb_wait_front(cb_mstat, s0 + 2 * rows);
    for (uint32_t r = 0; r < rows; ++r) {
        cb_reserve_back(cb_shift, 1);
        cb_reserve_back(cb_rowstat, 1);
        if constexpr (kStagedK) {
            cb_reserve_back(cb_kg, 1);
        }
        tile_regs_acquire();
        reconfig_data_format_srca(cb_mstat);
        copy_init(cb_mstat);
#ifdef AGRMS_LN_FPU
        copy_tile(cb_mstat, s0 + r, 0);         // D1 (D1 block, then D2 block)
        copy_tile(cb_mstat, s0 + rows + r, 1);  // D2
#else
        copy_tile(cb_mstat, s0 + 2 * r, 0);      // D1
        copy_tile(cb_mstat, s0 + 2 * r + 1, 1);  // D2
#endif
        if constexpr (kStagedK) {
            copy_tile(cb_mstat, r, 2);  // K_m
        }
        rsqrt_tile_init();
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            ln_finalize_k1_col0_body < kStagedK != 0 >, 0, VectorMode::C, inv_n_bits, inv_w_bits, eps_bits)));
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_shift);
        pack_tile(0, cb_shift);
        pack_reconfig_data_format(cb_rowstat);
        pack_tile(1, cb_rowstat);
        if constexpr (kStagedK) {
            pack_reconfig_data_format(cb_kg);
            pack_tile(2, cb_kg);
        }
        tile_regs_release();
        cb_push_back(cb_shift, 1);
        cb_push_back(cb_rowstat, 1);
        if constexpr (kStagedK) {
            cb_push_back(cb_kg, 1);
        }
    }
    cb_pop_front(cb_mstat, s0 + 2 * rows);
    if constexpr (kStagedK) {
        cb_pop_front(cb_km, rows);
    }
    static_assert(n_out <= 3);
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
                bcast_col_to(cb_pivot, r, 3);
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
#ifdef AGRMS_LN_FPU
        copy_to(cb_mstat, r, 0);         // D1 (D1 block, then D2 block)
        copy_to(cb_mstat, rows + r, 1);  // D2
#else
        copy_to(cb_mstat, 2 * r, 0);      // D1
        copy_to(cb_mstat, 2 * r + 1, 1);  // D2
#endif
        copy_to(cb_km, r, 2);             // K_m
        copy_d(0, 3);                     // D1
        mul_s(0, inv_n_bits);             // t = D1 / n_m
        mul_d(3, 0, 3);                   // D1^2 / n_m
        sub_d(1, 3, 1);                   // M2_m
        add_d(2, 0, 2);                   // mean_m
#ifdef AGRMS_OWNER
        // owner combine: the partials travel in ROW form (column 0 -> row 0)
        transpose_dest_init<DST_ACCUM_MODE>();
        transpose_dest<DST_ACCUM_MODE>(2);
        transpose_dest<DST_ACCUM_MODE>(1);
#endif
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
        if constexpr (k == 1) {  // one member holds the whole row: mu = mean_1, M2 = M2_1
            copy_to(src_cb, 2 * r, 0);
            copy_to(src_cb, 2 * r + 1, 2);
        } else {
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

#ifdef AGRMS_OWNER
ALWI void bcast_row_to(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format_srcb(cb);
    unary_bcast_init<BroadcastType::ROW>(cb);
    unary_bcast<BroadcastType::ROW>(cb, tile, dst);
}

// Owner Chan combine in ROW form (k > 8). Per owned row j, gather tile g of stat s = page (j * 2 + s) * gp + g holds
// member m = 32 g + rho's (mean_m, M2_m) in row rho (32 columns = the block's 32 rows of tile-row r; unused rows
// zero). Pivot K = bf16(mean of member 0) (row 0 of gather tile 0; the pack to bf16 cb_km IS the rounding), so every
// FPU operand is centered:
//   W1 (per g): d = mean - K (SFPU, fp32), a = n_m d, b = M2_m + n_m d^2 -> cb_dev        (n_m: cb_wcol bcast COL)
//   W2: S1 = sum_m a, S2 = sum_m b (ones-row matmul, row 0); delta = S1 / W, M2 = S2 - S1 delta,
//       inv_std = rsqrt(M2 / W + eps), mu = K + delta; transposed to column form -> cb_mu (mu, inv_std),
//       K_g = bf16(mu) -> cb_km
//   W3: shift = (K_g - mu) inv_std -> cb_own_ret (inv_std, shift), K_g -> cb_own_kg (the writer returns them)
ALWI void owner_combine_ln(uint32_t owned, uint32_t inv_w_bits, uint32_t eps_bits) {
    constexpr uint32_t gp = gather_pages;
    cb_wait_front(cb_gathered, owned * 2 * gp);
    cb_wait_front(cb_wcol, gp);
    cb_wait_front(cb_ones, 1);
    for (uint32_t j = 0; j < owned; ++j) {
        const uint32_t mean0 = (j * 2) * gp, m2_0 = (j * 2 + 1) * gp;
        // W0: pivot K = bf16(member 0's means)
        cb_reserve_back(cb_km, 1);
        tile_regs_acquire();
        copy_to(cb_gathered, mean0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_km);
        pack_tile(0, cb_km);
        tile_regs_release();
        cb_push_back(cb_km, 1);
        cb_wait_front(cb_km, 1);
        // W1: centered, weighted member terms -> cb_dev
        pack_reconfig_data_format(cb_dev);
        for (uint32_t g = 0; g < gp; ++g) {
            cb_reserve_back(cb_dev, 2);
            tile_regs_acquire();
            copy_to(cb_gathered, mean0 + g, 0);
            bcast_row_to(cb_km, 0, 1);
            sub_d(0, 1, 0);  // d = mean_m - K
            bcast_col_to(cb_wcol, g, 1);
            mul_d(0, 1, 2);  // a = n_m d
            mul_d(0, 2, 0);  // n_m d^2
            copy_to(cb_gathered, m2_0 + g, 1);
            add_d(0, 1, 0);  // b = M2_m + n_m d^2
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(2, cb_dev);
            pack_tile(0, cb_dev);
            tile_regs_release();
            cb_push_back(cb_dev, 2);
        }
        // W2: sums over the members (row 0), finalize, transpose to column form
        cb_wait_front(cb_dev, 2 * gp);
        cb_reserve_back(cb_mu, 2);
        cb_reserve_back(cb_km, 1);
        reconfig_data_format(cb_dev, cb_ones);  // srcA = in1 (dev), srcB = in0 (ones)
        matmul_init(cb_ones, cb_dev, 0);
        tile_regs_acquire();
        for (uint32_t g = 0; g < gp; ++g) {
            matmul_tiles(cb_ones, cb_dev, 0, 2 * g, 0);      // S1
            matmul_tiles(cb_ones, cb_dev, 0, 2 * g + 1, 1);  // S2
        }
        copy_d(0, 2);
        mul_s(2, inv_w_bits);  // delta = S1 / W
        mul_d(0, 2, 0);        // S1 delta
        sub_d(1, 0, 1);        // M2
        mul_s(1, inv_w_bits);
        binop_with_scalar_tile_init();
        add_unary_tile(1, eps_bits);
        rsqrt_tile_init();
        rsqrt_tile(1);  // inv_std
        bcast_row_to(cb_km, 0, 0);
        add_d(0, 2, 0);  // mu = K + delta
        transpose_dest_init<DST_ACCUM_MODE>();
        transpose_dest<DST_ACCUM_MODE>(0);
        transpose_dest<DST_ACCUM_MODE>(1);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_mu);
        pack_tile(0, cb_mu);
        pack_tile(1, cb_mu);
        pack_reconfig_data_format(cb_km);
        pack_tile(0, cb_km);
        tile_regs_release();
        cb_push_back(cb_mu, 2);
        cb_push_back(cb_km, 1);
        cb_pop_front(cb_dev, 2 * gp);
        cb_pop_front(cb_km, 1);  // K
        // W3: shift = (K_g - mu) * inv_std; return (inv_std, shift) + K_g
        cb_wait_front(cb_mu, 2);
        cb_wait_front(cb_km, 1);
        cb_reserve_back(cb_own_ret, 2);
        cb_reserve_back(cb_own_kg, 1);
        tile_regs_acquire();
        copy_to(cb_mu, 0, 0);  // mu
        copy_to(cb_km, 0, 1);  // K_g
        copy_to(cb_mu, 1, 2);  // inv_std
        copy_d(1, 3);
        sub_d(1, 0, 1);
        mul_d(1, 2, 1);  // shift
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_own_ret);
        pack_tile(2, cb_own_ret);
        pack_tile(1, cb_own_ret);
        pack_reconfig_data_format(cb_own_kg);
        pack_tile(3, cb_own_kg);
        tile_regs_release();
        cb_push_back(cb_own_ret, 2);
        cb_push_back(cb_own_kg, 1);
        cb_pop_front(cb_mu, 2);
        cb_pop_front(cb_km, 1);
    }
    cb_pop_front(cb_gathered, owned * 2 * gp);
}
#endif

#ifdef AGRMS_LN_FPU
// ---- FPU fast path (tile-aligned bf16 / bfloat8_b input: x and the bf16 pivots are exact FPU operands, so
// x - K is exact in DEST; no slice tail to mask). Same statistics as the SFPU path, a fraction of the inits. ----
// pass A: K_m = bf16(sum x / n_m) — one reduce over the resident block (row stride = pitch, x stays resident)
ALWI void pivot_block_fpu(uint32_t rows, uint32_t core_w, uint32_t inv_n_bits) {
    reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_ROW,
        cb_x,
        cb_ones,
        cb_pivot,
        ReduceInputPolicy::NoWaitNoPop,
        ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Fast,
        ReduceAlgorithm::ReduceTile>(
        ReduceInputBlockShape::of(rows, core_w),
        ReduceInputMemoryLayout::with_row_stride(pitch),
        NoAccumulation{},
        MulScalar{inv_n_bits});
}

#ifdef AGRMS_LN_OWN_PIVOT
#ifdef AGRMS_DENSE
#error "AGRMS_LN_OWN_PIVOT: the own-slice arguments follow n_m, where the dense assembly's arguments start"
#endif
// OVERLAP: K_m = bf16(mean of the member's OWN slice), reduced from cb_own (the reader's copy of the own slice, pushed
// with the own read), so the pivot is formed while the remote slices are still in flight. Any bf16 value near the
// member mean is a valid pivot (M2 = D2 - D1^2 / n is exact algebra); an own-slice pivot bounds the cancellation in
// D2 by n_m / n_own and leaves K_g (= bf16(mu), formed by the finalize / combine) unchanged.
ALWI void own_pivot_block(uint32_t rows, uint32_t own_n, uint32_t inv_own_bits) {
    reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_ROW,
        cb_own,
        cb_ones,
        cb_km,
        ReduceInputPolicy::BulkWaitBulkPop,
        ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Fast,
        ReduceAlgorithm::ReduceTile>(
        ReduceInputBlockShape::of(rows, own_n),
        ReduceInputMemoryLayout::contiguous(),
        NoAccumulation{},
        MulScalar{inv_own_bits});
}
#endif

// pass B: c' = x - K_m (FPU, bcast col) -> cb_cent (accumulator format); D1 = row-reduce(c') (block reduce),
// D2 = row-reduce(sum_j c'_j^2) (DEST-accumulated squares) -> cb_mstat [D1 x rows, D2 x rows]
ALWI void centered_moments_block_fpu(uint32_t rows, uint32_t core_w) {
    constexpr auto x_in = input(
        cb_x,
        WaitPolicy::None,
        PopPolicy::None,
        InputTileMapping::Block,
        DataFormatReconfig::Enabled,
        TileAddressing::Strided);
    constexpr auto km_in = input(cb_pivot, BroadcastDim::Col, WaitPolicy::None, PopPolicy::None, InputTileMapping::Col);
    eltwise_chain(
        IterationShape::grid(rows, core_w),
        BinaryFpu<BinaryFpuOp::Sub, x_in, km_in>{StridedTileRange{0, pitch}},
        PackTile<output(cb_cent)>{});
    reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_ROW,
        cb_cent,
        cb_ones,
        cb_mstat,
        ReduceInputPolicy::WaitUpfrontNoPop,
        ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Fast,
        ReduceAlgorithm::ReduceTile>(ReduceInputBlockShape::of(rows, core_w));
    sum_squares_block<cb_cent, cb_moment_b>(rows, core_w, core_w);
    reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_ROW,
        cb_moment_b,
        cb_ones,
        cb_mstat,
        ReduceInputPolicy::BulkWaitBulkPop,
        ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Fast,
        ReduceAlgorithm::ReduceTile>(ReduceInputBlockShape::of(rows, 1));
    cb_pop_front(cb_cent, rows * core_w);
}

// normalize: (x - K_g) [FPU, exact] * inv_std (bcast col) + shift (bcast col) [* gamma] [+ beta] (dest reuse)
ALWI void normalize_block(uint32_t rows, uint32_t core_w) {
    constexpr auto x_in = input(
        cb_x,
        WaitPolicy::None,
        PopPolicy::None,
        InputTileMapping::Block,
        DataFormatReconfig::Enabled,
        TileAddressing::Strided);
    constexpr auto kg_in = input(cb_kg, BroadcastDim::Col, WaitPolicy::None, PopPolicy::None, InputTileMapping::Col);
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
        BinaryFpu<BinaryFpuOp::Sub, x_in, kg_in>{StridedTileRange{0, pitch}},
        // srcA switched to the DEST width once (first step) and restored once (last step), see DestReuseBcast
        DestReuseBcast<EltwiseBinaryType::ELWMUL, BroadcastType::COL, inv_in, true, false>{},
        DestReuseBcast < EltwiseBinaryType::ELWADD,
        BroadcastType::COL,
        shift_in,
        false,
        !kHasGamma && !kHasBeta > {},
#ifdef AGRMS_HAS_GAMMA
        DestReuseBcast<EltwiseBinaryType::ELWMUL, BroadcastType::ROW, gamma_in, false, !kHasBeta>{},
#endif
#ifdef AGRMS_HAS_BETA
        DestReuseBcast<EltwiseBinaryType::ELWADD, BroadcastType::ROW, beta_in, false, true>{},
#endif
        PackTile<output(norm_out_cb)>{});
}
#else
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
        PackTile<output(norm_out_cb), Dst::D0>{});
}
#endif  // AGRMS_LN_FPU

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
#ifdef AGRMS_LN_OWN_PIVOT
    const uint32_t own_n = arg(nm_idx + w_split_cores);
    if (own_n > 0) {
        AGZ("c_opiv");
        own_pivot_block(rows, own_n, arg(nm_idx + w_split_cores + 1));
        if constexpr (kStagedK) {
            stage_pivot_block(rows);
        }
    }
#else
    constexpr uint32_t own_n = 0;
#endif
    {
        AGZ("c_waitx");
        cb_wait_front(cb_x, rows * pitch);
    }
    {
        AGZ("c_mom");
        if constexpr (!DST_ACCUM_MODE) {
            moments16_block(rows, n_segs, segs_idx, inv_n_bits);
        } else {
#ifdef AGRMS_LN_FPU
            if (own_n == 0) {
                pivot_block_fpu(rows, core_w, inv_n_bits);
                if constexpr (kStagedK) {
                    stage_pivot_block(rows);
                }
            }
            cb_wait_front(cb_pivot, rows);
            centered_moments_block_fpu(rows, core_w);
#else
            // pass A -> pivot K_m
            sum_x_block(rows, core_w);
            row_reduce<cb_moment_a, cb_pivot>(rows, MulScalar{inv_n_bits});
            cb_wait_front(cb_pivot, rows);
            // pass B -> centered moments
            centered_moments_block(rows, n_segs, segs_idx);
            row_reduce<cb_moment_b, cb_mstat>(2 * rows, NoOp{});
#endif
            if constexpr (!kFusedK1) {
                member_finalize_block(rows, inv_n_bits);
            }
        }
    }
    {
        AGZ("c_comb");
#ifdef AGRMS_OWNER
        const uint32_t owned = agrms_owner::owned_rows(rows, arg(2), w_split_cores);
        if (owned > 0) {
            if constexpr (!DST_ACCUM_MODE) {
                owner16_ln(owned, inv_w_bits, eps_bits, nm_idx);
            } else {
                owner_combine_ln(owned, inv_w_bits, eps_bits);
            }
        }
#else
        constexpr uint32_t src_cb = w_split_cores > 1 ? cb_gathered : cb_partial;
        if constexpr (!DST_ACCUM_MODE) {
            combine16_block<src_cb>(rows, inv_w_bits, eps_bits, nm_idx);
        } else if constexpr (kFusedK1) {
            ln_finalize_k1_block(rows, inv_n_bits, inv_w_bits, eps_bits);
        } else {
            combine_block<src_cb>(rows, inv_w_bits, eps_bits, nm_idx);
        }
#endif
    }
    {
        AGZ("c_waitg");
#ifdef AGRMS_HAS_GAMMA
        cb_wait_front(cb_gamma, pitch);
#endif
#ifdef AGRMS_HAS_BETA
        cb_wait_front(cb_beta, pitch);
#endif
        cb_wait_front(cb_shift, rows);
        cb_wait_front(cb_kg, rows);  // owner: returned by the row owners (writer)
        cb_wait_front(cb_rowstat, rows);
    }
    {
        AGZ("c_norm");
        normalize_block(rows, core_w);
    }
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
#ifdef AGRMS_DENSE
    dense_assemble_block(rows, core_w, nm_idx + w_split_cores);
#endif
}
#endif
