// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_post compute (regimes R1 plain and R2 fused-gather).
//
// build_kpages (row start): the row's 1 + n "K pages" (fp32, cb_kpages) from the column-0 coefficient pages: page s
// (s = F: post_j; s = X_i: comb[i][j]) holds coef(s, j)[token R] at (R, C), j = 2*(C >= 16) + (C & 1), built per page
// by 4 column broadcasts + one SFPU move + one fp32 pack (exact).
// mix_block, one dst-sync window per UNIT (fused operand-outer SFPU sweep; DEST: A_0..A_3, operand, K page, [Dsq]):
//     for s in {F, X_0..X_3}: D4 = operand s (unpacked once), D5 = K page s; ONE sfpi sweep updates all 4 streams:
//       A_j = c_{F,j} * F, then A_j = fma(c_{i,j}, X_i, A_j)       (same fp32 op order per element as before)
//     [R2] last sweep: Dsq = reload(cb_sumsq_acc); Dsq = fma(A_j, A_j, Dsq), j = 0..3  (unit-major fold)
//     pack A_j -> cb_mixed_out_tiles [R2: and cb_mixed_bf16] at k = j*B + u (out of order, pass window reserved)
// Every coefficient product is fp32 SFPU arithmetic; the FPU never multiplies a coefficient (tf32 bias).
// A ragged plain pass runs unit by unit (u-major pages); a ragged R2 pass computes zero filler units.
//
// R2 per token-tile row (project_block): ONE matmul_block call whose K-blocks are the row's column passes. Its
// PreKBlockFn runs mix_block (and, collapse, collapse_block: y_u = sum_i P_i * X'bf16_i[u] -> cb_y_out) of the
// pass, so the bf16 X' of a pass is consumed by the matmul as soon as it is packed and never re-read from DRAM
// (regime X2 forbidden); W is the resident (or streamed) in1. The K order of cb_mixed_bf16 is the pack order:
// k = j*B + u (the reader lays W out the same way). A ragged pass runs the full block on zeroed filler units.
// finalize_row: cb_sumsq_acc --SFPU row sum--> column 0 of cb_sumsq_col (the writer merges it into col 24).
// combine_row: fixed slot-order SFPU fold — level 1 eager at the row boundary on the row's leader, level 2 on the
// row's root deferred past the root's own rows (or, norm with root_lag > 0, lagged inside the row loop).
// hc_step_row (collapse, root): the DeepSeek-V4 hc step as one custom SFPU program in lane layout.
// norm: collapse packs y_b into cb_y_hold (held until rstd returns) and accumulates sum y_b^2 of the bf16 y_b
// (reloaded, exact) into cb_ysq_acc; finalize_row row-sums it into a second cb_sumsq_col page (partial column
// ng+1). The root's hc step also computes rstd = rsqrt(sum y^2 / C + norm_eps) in lane vector ng+1. norm_block:
// y = (y_b * rstd) * w (fp32 SFPU, one bf16 rounding) -> cb_y_out, after the roots (default) or norm_lag rows after
// the row (lagged knob).
//
// Raw-LLK deviation (documented per the helper-first policy): the combine fold sums a RUNTIME number of slot
// tiles (chunk size / chunk count) in one DEST window; eltwise_chain needs a compile-time element list, so the
// fold is copy_tile + add_binary_tile + pack_tile (the same LLKs the chain's CopyTile / AddBinary / PackTile
// elements issue, with the same inits / reconfigs). The row-sum and the hc step are custom chain elements.

#include <cstdint>
#include <tuple>
#include <utility>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/ternary/ternary.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/generators/fill.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

#ifdef TRISC_MATH
#include "sfpi.h"
#include "ckernel_sfpu_exp.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckl = compute_kernel_lib;

namespace {

constexpr uint32_t cb_sublayer_tiles = get_compile_time_arg_val(0);
constexpr uint32_t cb_stream_tiles = get_compile_time_arg_val(1);
constexpr uint32_t cb_coef_cols = get_compile_time_arg_val(2);
constexpr uint32_t cb_mixed_out_tiles = get_compile_time_arg_val(3);
constexpr uint32_t n = get_compile_time_arg_val(4);
constexpr uint32_t block_rows = get_compile_time_arg_val(5);
constexpr uint32_t block_cols = get_compile_time_arg_val(6);
constexpr uint32_t mix_fixed_slots = get_compile_time_arg_val(7);  // host MIX_FIXED_SLOTS[_STATS]
constexpr bool has_w = get_compile_time_arg_val(8) != 0;
constexpr bool collapse = get_compile_time_arg_val(9) != 0;
constexpr bool w_resident = get_compile_time_arg_val(10) != 0;
constexpr uint32_t w_pages = get_compile_time_arg_val(11);
constexpr uint32_t group_q = get_compile_time_arg_val(12);  // Q: members of a row group
constexpr uint32_t fanin = get_compile_time_arg_val(13);
constexpr uint32_t cb_mixed_bf16 = get_compile_time_arg_val(14);
constexpr uint32_t cb_w_slice = get_compile_time_arg_val(15);
constexpr uint32_t cb_sumsq_acc = get_compile_time_arg_val(16);
constexpr uint32_t cb_proj_interm = get_compile_time_arg_val(17);
constexpr uint32_t cb_proj_out = get_compile_time_arg_val(18);
constexpr uint32_t cb_sumsq_col = get_compile_time_arg_val(19);
constexpr uint32_t cb_leader_slots = get_compile_time_arg_val(20);
constexpr uint32_t cb_chunk_sum = get_compile_time_arg_val(21);
constexpr uint32_t cb_root_slots = get_compile_time_arg_val(22);
constexpr uint32_t cb_stats = get_compile_time_arg_val(23);
constexpr uint32_t cb_hc_in = get_compile_time_arg_val(24);
constexpr uint32_t cb_hc_out = get_compile_time_arg_val(25);
constexpr uint32_t cb_y_out = get_compile_time_arg_val(26);
constexpr bool norm = get_compile_time_arg_val(27) != 0;
constexpr uint32_t cb_y_hold = get_compile_time_arg_val(28);
constexpr uint32_t cb_ysq_acc = get_compile_time_arg_val(29);
constexpr uint32_t cb_rstd_col = get_compile_time_arg_val(30);
constexpr uint32_t cb_norm_w = get_compile_time_arg_val(31);
// norm schedule knob (Refinement 3). root_lag == 0: the roots are deferred past every row, norm_block runs after them
// and cb_y_hold holds every row (the default). root_lag > 0 (lagged): the root work of row t runs root_lag rows after
// t and norm_block(t) norm_lag rows after t, inside the row loop, so rstd / the y writes / the feed publication stream
// with the rows; cb_y_hold holds norm_lag + 1 rows.
constexpr uint32_t root_lag = get_compile_time_arg_val(32);
constexpr uint32_t norm_lag = get_compile_time_arg_val(33);
constexpr bool lagged = norm && root_lag > 0;
static_assert(!lagged || norm_lag > root_lag, "lagged norm: 1 <= root_lag < norm_lag");
constexpr uint32_t hold_pages = get_compile_time_arg_val(34);    // norm: cb_y_hold capacity
constexpr uint32_t cb_rstd_lane = get_compile_time_arg_val(35);  // norm: the root's rstd in lane layout
constexpr uint32_t cb_y_collapse = norm ? cb_y_hold : cb_y_out;  // collapse_block's destination

constexpr uint32_t n_mix_coef = n + n * n;
constexpr uint32_t n_coef = n_mix_coef + (collapse ? n : 0);
constexpr uint32_t cb_kpages = get_compile_time_arg_val(36);  // fused mix: the row's 1 + n K pages
constexpr uint32_t n_kpages = 1 + n;
constexpr uint32_t out_ring = get_compile_time_arg_val(37);  // cb_mixed_out_tiles capacity (pages)
static_assert(n == 4, "fused mix: a K page carries the 4 output streams' coefficients");
constexpr uint32_t pass_k_tiles = n * block_cols;  // matmul K tiles per column pass
constexpr uint32_t num_chunks = (group_q + fanin - 1) / fanin;
constexpr uint32_t ONE_F32 = 0x3f800000u;  // Addcmul value: A + 1.0 * (B * C) == fma(B, C, A)

static_assert(mix_fixed_slots == (has_w ? 3u : 2u), "mix_block DEST layout: Dc, Do, [Dsq], accumulators after");
static_assert(
    block_cols + mix_fixed_slots <= ckl::DEST_AUTO_LIMIT, "mix_block: block_cols + fixed DEST slots must fit");

constexpr ckl::Dst Dc = ckl::Dst::D0;  // coefficient (column-broadcast): collapse_block
constexpr ckl::Dst Do = ckl::Dst::D1;  // operand: collapse_block / ysq

// Caller-managed, offset-addressed inputs: the page index (from the CB front) is the element's runtime base.
constexpr ckl::InputSpec caller_managed(uint32_t cb) {
    return ckl::input(
        cb,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Scalar,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Offset);
}
constexpr ckl::OutputSpec out_of_order(uint32_t cb) {
    return ckl::output(
        cb,
        ckl::ReservePolicy::None,
        ckl::PushPolicy::None,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Offset);
}

using CoefBcast = ckl::UnaryBcast<ckl::BroadcastDim::Col, caller_managed(cb_coef_cols), Dc>;

// ---------------- mix_block: fused operand-outer SFPU sweep, one dst-sync window per unit ----------------
// DEST: D0..D3 = A_0..A_3 (the unit's n = 4 output streams), D4 = operand, D5 = K page, D6 = [R2] Dsq.
// For s in {F, X_0..X_3}: D4 = operand s (unpacked ONCE per unit), D5 = K page s, then ONE custom SFPU sweep:
//   s = F: A_j = c_{F,j} * F ; s = X_i: A_j = fma(c_{i,j}, X_i, A_j) ; [R2, last] Dsq = fma(A_j, A_j, Dsq), j = 0..3.
// Same fp32 op sequence per element as the j-major chain it replaces (A is bit-identical); Dsq folds unit-major.
// K page s (built by build_kpages on this core, fp32): element (R, C) = coef(s, j)[token R], j = 2*(C >= 16) + (C & 1):
// the 4 SFPU vectors of a row group (offsets {0, 1, 8, 9} from its base: even / odd columns of the left / right face)
// then hold exactly coef(s, 0..3) of the group's 4 tokens, so one coefficient load serves the 4 vectors of every tile.
// Raw-sfpi deviation (helper-first policy): MulBinary / Addcmul each walk one output tile with every operand a full
// DEST tile; updating four output tiles from one operand load and a 4-coefficient page is not expressible as chain
// elements, so the sweep is one custom SFPU program (a chain element). Copies and packs stay chain elements.
// Measured (Perf 1, BH, focus T160 C5120 collapse+norm): helper chain 25.9 us -> sweep 14.0 us a 4-unit pass (1.85x);
// whole op 123.3 -> 111.5 us fastest chip. Do not revert to the per-stream MulBinary / Addcmul chain.
constexpr uint32_t S_OP = 4;
constexpr uint32_t S_K = 5;
constexpr uint32_t S_SQ = 6;
static_assert(S_SQ < ckl::DEST_AUTO_LIMIT, "fused mix: 7 fp32 DEST slots (full-sync DEST)");

#ifdef TRISC_MATH
namespace mix {
using sfpi::dst_reg;
using sfpi::vFloat;
constexpr uint32_t T = 32;  // sfpi vectors per fp32 DEST tile

// two accumulators in flight (the MAD latency of one hidden behind the other)
template <uint32_t MODE, uint32_t A0, uint32_t A1>
sfpi_inline void acc_pair(vFloat c0, vFloat c1, vFloat x, vFloat& q) {
    vFloat a = dst_reg[A0];
    vFloat b = dst_reg[A1];
    a = c0 * x + a;
    b = c1 * x + b;
    dst_reg[A0] = a;
    dst_reg[A1] = b;
    if constexpr (MODE == 2) {
        q = a * a + q;
        q = b * b + q;
    }
}
// MODE 0: A_j = c_j * x. MODE 1: A_j = fma(c_j, x, A_j). MODE 2: MODE 1 + Dsq = fma(A_j, A_j, Dsq), j = 0..3.
template <uint32_t MODE, uint32_t OFF>
sfpi_inline void vec_at(vFloat c0, vFloat c1, vFloat c2, vFloat c3) {
    vFloat x = dst_reg[S_OP * T + OFF];
    if constexpr (MODE == 0) {
        dst_reg[0 * T + OFF] = c0 * x;
        dst_reg[1 * T + OFF] = c1 * x;
        dst_reg[2 * T + OFF] = c2 * x;
        dst_reg[3 * T + OFF] = c3 * x;
    } else {
        vFloat q;  // live in MODE 2 only
        if constexpr (MODE == 2) {
            q = dst_reg[S_SQ * T + OFF];
        }
        acc_pair<MODE, 0 * T + OFF, 1 * T + OFF>(c0, c1, x, q);
        acc_pair<MODE, 2 * T + OFF, 3 * T + OFF>(c2, c3, x, q);
        if constexpr (MODE == 2) {
            dst_reg[S_SQ * T + OFF] = q;
        }
    }
}
// The DEST pointer walks the 8 row groups (+= 2 inside a face pair, += 8 between the pairs).
template <uint32_t MODE>
inline void sweep() {
#pragma GCC unroll 1
    for (uint32_t h = 0; h < 2; ++h) {
#pragma GCC unroll 1
        for (uint32_t rg = 0; rg < 4; ++rg) {
            vFloat c0 = dst_reg[S_K * T + 0];
            vFloat c1 = dst_reg[S_K * T + 1];
            vFloat c2 = dst_reg[S_K * T + 8];
            vFloat c3 = dst_reg[S_K * T + 9];
            vec_at<MODE, 0>(c0, c1, c2, c3);
            vec_at<MODE, 1>(c0, c1, c2, c3);
            vec_at<MODE, 8>(c0, c1, c2, c3);
            vec_at<MODE, 9>(c0, c1, c2, c3);
            dst_reg += 2;
        }
        dst_reg += 3;
        dst_reg += 3;
        dst_reg += 2;
    }
}
}  // namespace mix
#endif

template <uint32_t MODE>
struct MixSweep : ckl::UnaryOp<MixSweep<MODE>, static_cast<ckl::Dst>(S_SQ)> {
    static ALWI void init() { MATH((SFPU_UNARY_INIT(unused))); }
    static ALWI void exec_impl(uint32_t /*slot_offset*/) {
        MATH((_llk_math_eltwise_unary_sfpu_params_(mix::sweep<MODE>, 0, VectorMode::RC_custom)));
    }
};
constexpr uint32_t LAST_MODE = has_w ? 2 : 1;

using OpF = ckl::CopyTile<caller_managed(cb_sublayer_tiles), static_cast<ckl::Dst>(S_OP)>;
using OpX = ckl::CopyTile<caller_managed(cb_stream_tiles), static_cast<ckl::Dst>(S_OP)>;
using KPage = ckl::CopyTile<caller_managed(cb_kpages), static_cast<ckl::Dst>(S_K)>;
template <uint32_t J>
using PackOut = ckl::PackTile<out_of_order(cb_mixed_out_tiles), static_cast<ckl::Dst>(J)>;
template <uint32_t J>
using PackBf = ckl::PackTile<out_of_order(cb_mixed_bf16), static_cast<ckl::Dst>(J)>;

// cb_mixed_out_tiles ring position of the write pointer (pages pushed mod out_ring): a ragged plain pass publishes
// fewer than n*block_cols pages, so a later reserved window may wrap the ring end. Out-of-order pack addresses
// wr_ptr + page * index in uint32 arithmetic, so a page past the end is addressed as index - out_ring (wrapped).
uint32_t out_slot = 0;
ALWI uint32_t out_index(uint32_t k) { return out_slot + k >= out_ring ? k - out_ring : k; }

ALWI auto mix_packs(uint32_t B, uint32_t u) {
    if constexpr (has_w) {
        return std::make_tuple(
            PackOut<0>{out_index(0 * B + u)},
            PackOut<1>{out_index(1 * B + u)},
            PackOut<2>{out_index(2 * B + u)},
            PackOut<3>{out_index(3 * B + u)},
            PackBf<0>{0 * B + u},
            PackBf<1>{1 * B + u},
            PackBf<2>{2 * B + u},
            PackBf<3>{3 * B + u},
            ckl::PackTile<ckl::output(cb_sumsq_acc), static_cast<ckl::Dst>(S_SQ)>{});
    } else {
        return std::make_tuple(
            PackOut<0>{out_index(0 * B + u)},
            PackOut<1>{out_index(1 * B + u)},
            PackOut<2>{out_index(2 * B + u)},
            PackOut<3>{out_index(3 * B + u)});
    }
}
ALWI auto sumsq_reload() {
    if constexpr (has_w) {  // R2: the running sum of squares, reloaded before the last sweep (zeroed at row start)
        return std::make_tuple(ckl::CopyTile<ckl::input(cb_sumsq_acc), static_cast<ckl::Dst>(S_SQ)>{});
    } else {
        return std::make_tuple();
    }
}

// one unit of the waited window: F page w, X pages w*n + i; outputs packed at j*B + u of the reserved window
ALWI void mix_unit(uint32_t w, uint32_t B, uint32_t u) {
#ifdef MHC_ABLATE_MIX  // breakdown ablation: the mix payload stubbed, the packs (and their CB sync) kept
    auto elems = std::tuple_cat(sumsq_reload(), mix_packs(B, u));
#else
    auto elems = std::tuple_cat(
        std::make_tuple(OpF{w}, KPage{0}, MixSweep<0>{}),
        std::make_tuple(OpX{w * n + 0}, KPage{1}, MixSweep<1>{}),
        std::make_tuple(OpX{w * n + 1}, KPage{2}, MixSweep<1>{}),
        std::make_tuple(OpX{w * n + 2}, KPage{3}, MixSweep<1>{}),
        sumsq_reload(),
        std::make_tuple(OpX{w * n + 3}, KPage{4}, MixSweep<LAST_MODE>{}),
        mix_packs(B, u));
#endif
    std::apply([](auto... e) { ckl::eltwise_chain(ckl::IterationShape::one_tile(), e...); }, elems);
}

// ---------------- build_kpages (row start): K page s from the column-0 coefficient pages ----------------
// D0..D3 = bcast_col(coef(s, j)) (the column-broadcast tile of each output stream j's coefficient), then one SFPU pass
// moves, per row group, the vector of D_j at offset {0, 1, 8, 9}[j] into D0 (offset 0 already holds D0's), and D0 is
// packed (fp32, exact) as K page s. s = 0: post_j (page j); s = 1 + i: comb[i][j] (page n + i*n + j).
// Raw-sfpi deviation (measured, Perf 1): no chain element moves vectors across DEST tiles, so the merge is a custom
// SFPU element (KMerge); the broadcasts and the pack stay helper elements. The K pages buy the fused sweep's 1.85x
// (mix 25.9 -> 14.0 us a 4-unit pass, isolated; whole op 123.3 -> 111.5 us) for 4.1 us a row.
#ifdef TRISC_MATH
namespace mix {
inline void kmerge() {
#pragma GCC unroll 1
    for (uint32_t h = 0; h < 2; ++h) {
#pragma GCC unroll 1
        for (uint32_t rg = 0; rg < 4; ++rg) {
            vFloat b1 = dst_reg[1 * T + 1];
            vFloat b2 = dst_reg[2 * T + 8];
            vFloat b3 = dst_reg[3 * T + 9];
            dst_reg[0 * T + 1] = b1;
            dst_reg[0 * T + 8] = b2;
            dst_reg[0 * T + 9] = b3;
            dst_reg += 2;
        }
        dst_reg += 3;
        dst_reg += 3;
        dst_reg += 2;
    }
}
}  // namespace mix
#endif
struct KMerge : ckl::UnaryOp<KMerge, ckl::Dst::D3> {
    static ALWI void init() { MATH((SFPU_UNARY_INIT(unused))); }
    static ALWI void exec_impl(uint32_t /*slot_offset*/) {
        MATH((_llk_math_eltwise_unary_sfpu_params_(mix::kmerge, 0, VectorMode::RC_custom)));
    }
};
ALWI void build_kpages() {
    MaybeDeviceZoneScope("c_kbuild");
    cb_reserve_back(cb_kpages, n_kpages);
    for (uint32_t s = 0; s < n_kpages; ++s) {
        const uint32_t p0 = s == 0 ? 0 : n + (s - 1) * n;  // page of coef(s, j = 0); j at p0 + j
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::UnaryBcast<ckl::BroadcastDim::Col, caller_managed(cb_coef_cols), ckl::Dst::D0>{p0 + 0},
            ckl::UnaryBcast<ckl::BroadcastDim::Col, caller_managed(cb_coef_cols), ckl::Dst::D1>{p0 + 1},
            ckl::UnaryBcast<ckl::BroadcastDim::Col, caller_managed(cb_coef_cols), ckl::Dst::D2>{p0 + 2},
            ckl::UnaryBcast<ckl::BroadcastDim::Col, caller_managed(cb_coef_cols), ckl::Dst::D3>{p0 + 3},
            KMerge{},
            ckl::PackTile<out_of_order(cb_kpages), ckl::Dst::D0>{s});
    }
    cb_push_back(cb_kpages, n_kpages);
    cb_wait_front(cb_kpages, n_kpages);
}

ALWI void reserve_mix(uint32_t pages) {
    cb_reserve_back(cb_mixed_out_tiles, pages);
    if constexpr (has_w) {
        cb_reserve_back(cb_mixed_bf16, pages);
    }
}
ALWI void push_mix(uint32_t pages) {
    cb_push_back(cb_mixed_out_tiles, pages);
    out_slot = out_slot + pages >= out_ring ? out_slot + pages - out_ring : out_slot + pages;
    if constexpr (has_w) {
        cb_push_back(cb_mixed_bf16, pages);
    }
}

// mix_block of one column pass: waits / pops the pass's F and X block (block_cols ring slots, filler on a
// ragged pass). A full pass packs j-major (k = j*B + u, the order the writer / matmul / collapse read); a ragged
// plain pass packs unit by unit (u-major), as the per-unit blocks it replaces did.
ALWI void mix_pass(uint32_t cols_in_pass) {
    {
        MaybeDeviceZoneScope("c_wait_in");
        cb_wait_front(cb_sublayer_tiles, block_cols);
        cb_wait_front(cb_stream_tiles, block_cols * n);
    }
    MaybeDeviceZoneScope("c_mix");
    if (has_w || cols_in_pass == block_cols) {  // R2: a ragged pass computes zero filler units
        reserve_mix(n * block_cols);
        for (uint32_t u = 0; u < block_cols; ++u) {
            mix_unit(u, block_cols, u);
        }
        push_mix(n * block_cols);
    } else {
        for (uint32_t u = 0; u < cols_in_pass; ++u) {
            reserve_mix(n);
            mix_unit(u, 1, 0);
            push_mix(n);
        }
    }
    cb_pop_front(cb_sublayer_tiles, block_cols);
    cb_pop_front(cb_stream_tiles, block_cols * n);
}

// ---------------- collapse_block: y_u = sum_i P_i * X'_i[u] (bf16 X' as written, fp32 MADs, one rounding) ----
constexpr ckl::Dst Dy0 = ckl::Dst::D2;  // y accumulators after Dc, Do
template <size_t U>
constexpr ckl::Dst y_slot() {
    return static_cast<ckl::Dst>(static_cast<uint32_t>(Dy0) + U);
}
using MixedCopy = ckl::CopyTile<caller_managed(cb_mixed_bf16), Do>;

// X'_i[u] sits at i*block_cols + u in cb_mixed_bf16 (j-major pack of the full block).
template <uint32_t I, size_t... U>
ALWI auto collapse_term(std::index_sequence<U...>) {
    if constexpr (I == 0) {
        return std::tuple_cat(
            std::make_tuple(CoefBcast{n_mix_coef + I}),
            std::make_tuple(MixedCopy{I * block_cols + U}, ckl::MulBinary<Dc, Do, y_slot<U>()>{})...);
    } else {
        return std::tuple_cat(
            std::make_tuple(CoefBcast{n_mix_coef + I}),
            std::make_tuple(
                MixedCopy{I * block_cols + U},
                ckl::Addcmul<DataFormat::Float32, y_slot<U>(), Dc, Do, y_slot<U>()>{ONE_F32})...);
    }
}

template <size_t... I, size_t... U>
ALWI void collapse_units_impl(std::index_sequence<I...>, std::index_sequence<U...>) {
#ifdef MHC_ABLATE_COLLAPSE
    auto elems = std::make_tuple(ckl::PackTile<ckl::output(cb_y_collapse), y_slot<U>()>{}...);
#else
    auto elems = std::tuple_cat(
        collapse_term<I>(std::index_sequence<U...>{})...,
        std::make_tuple(ckl::PackTile<ckl::output(cb_y_collapse), y_slot<U>()>{}...));
#endif
    std::apply([](auto... e) { ckl::eltwise_chain(ckl::IterationShape::one_tile(), e...); }, elems);
}

// collapse_block of one pass (full block, filler units give y = 0, dropped by the writer): reads the pass's bf16
// X' (already in cb_mixed_bf16, not popped: the matmul pops it).
ALWI void collapse_pass() {
    if constexpr (collapse) {
        cb_wait_front(cb_mixed_bf16, n * block_cols);
        MaybeDeviceZoneScope("c_collapse");
        collapse_units_impl(std::make_index_sequence<n>{}, std::make_index_sequence<block_cols>{});
    }
}

// norm: Dsq_y += y_b[u]^2 for the pass's units — the bf16 y_b as packed (cb_y_hold pages base..base+B-1 from the
// front, whose ring position is `front`). The held rows wrap the ring: a page past its end is addressed as
// base + u - hold_pages (the unpacker forms rd_ptr + page * index in uint32 arithmetic, so this is the wrapped page).
using HoldCopy = ckl::CopyTile<caller_managed(cb_y_hold), ckl::Dst::D1>;
ALWI uint32_t hold_index(uint32_t front, uint32_t idx) { return front + idx >= hold_pages ? idx - hold_pages : idx; }
template <size_t... U>
ALWI void ysq_pass_impl(uint32_t front, uint32_t base, std::index_sequence<U...>) {
    auto elems = std::tuple_cat(
        std::make_tuple(ckl::CopyTile<ckl::input(cb_ysq_acc), ckl::Dst::D0>{}),
        std::make_tuple(
            HoldCopy{hold_index(front, base + U)},
            ckl::Addcmul<DataFormat::Float32, ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D1, ckl::Dst::D0>{ONE_F32})...,
        std::make_tuple(ckl::PackTile<ckl::output(cb_ysq_acc), ckl::Dst::D0>{}));
    std::apply([](auto... e) { ckl::eltwise_chain(ckl::IterationShape::one_tile(), e...); }, elems);
}

// ---------------- norm_block: y_u = (y_b[u] * rstd) * w[u] for the pass's units, one dst-sync window ----------
// DEST: D0 = rstd (column broadcast, once per pass), D1 = w operand, the units' y in D2.. (block_cols <= 6).
constexpr ckl::Dst Dn0 = ckl::Dst::D2;
template <size_t U>
constexpr ckl::Dst norm_slot() {
    return static_cast<ckl::Dst>(static_cast<uint32_t>(Dn0) + U);
}
using RstdBcast = ckl::UnaryBcast<ckl::BroadcastDim::Col, caller_managed(cb_rstd_col), ckl::Dst::D0>;
using NormWBcast = ckl::UnaryBcast<ckl::BroadcastDim::Row, caller_managed(cb_norm_w), ckl::Dst::D1>;
template <size_t U>
using HoldToSlot = ckl::CopyTile<caller_managed(cb_y_hold), norm_slot<U>()>;
template <size_t... U>
ALWI void norm_pass(uint32_t w_base, std::index_sequence<U...>) {
#ifdef MHC_ABLATE_NORM
    auto elems = std::make_tuple(ckl::PackTile<ckl::output(cb_y_out), norm_slot<U>()>{}...);
#else
    auto elems = std::tuple_cat(
        std::make_tuple(RstdBcast{0}),
        std::make_tuple(HoldToSlot<U>{U}, ckl::MulBinary<norm_slot<U>(), ckl::Dst::D0, norm_slot<U>()>{})...,
        std::make_tuple(NormWBcast{w_base + U}, ckl::MulBinary<norm_slot<U>(), ckl::Dst::D1, norm_slot<U>()>{})...,
        std::make_tuple(ckl::PackTile<ckl::output(cb_y_out), norm_slot<U>()>{}...));
#endif
    std::apply([](auto... e) { ckl::eltwise_chain(ckl::IterationShape::one_tile(), e...); }, elems);
}
static_assert(!norm || block_cols + 2 <= ckl::DEST_AUTO_LIMIT, "norm_block: rstd + w + block_cols units in DEST");

// ---------------- project_block hooks ----------------
struct PassCols {
    uint32_t c_count;
    ALWI uint32_t cols(uint32_t pass) const {
        const uint32_t cb0 = pass * block_cols;
        return (c_count - cb0) < block_cols ? (c_count - cb0) : block_cols;
    }
};

// PreKBlockFn: mix (+ collapse) the pass whose bf16 X' the K-block multiplies (always a full block: the filler
// units of a ragged pass are zeros, their X' is 0 and their W rows are zeroed by the reader).
struct MixPassFn {
    PassCols pc;
    uint32_t hold_base;   // norm: cb_y_hold page (from the front) of this row's first unit
    uint32_t hold_front;  // norm: ring position of the front page
    ALWI void operator()(uint32_t pass, uint32_t /*num_k_blocks*/, bool /*is_last*/) const {
        mix_pass(pc.cols(pass));
        collapse_pass();
        if constexpr (norm) {
            const uint32_t base = hold_base + pass * block_cols;
            cb_wait_front(cb_y_hold, base + block_cols);
            MaybeDeviceZoneScope("c_ysq");
            ysq_pass_impl(hold_front, base, std::make_index_sequence<block_cols>{});
        }
        if constexpr (w_resident) {  // resident W lands behind the first block: wait only before the first K-block
            MaybeDeviceZoneScope("c_wait_w");
            cb_wait_front(cb_w_slice, w_pages);
        }
    }
};
struct PassIn1Offset {
    ALWI uint32_t operator()(uint32_t pass) const { return w_resident ? pass * pass_k_tiles : 0; }
};

// ---------------- custom chain elements ----------------
// finalize_row: per-token (row) sum of the 32 columns of the sum-of-squares tile, fp32 SFPU, into column 0.
template <ckl::Dst Slot>
struct RowSumF32 : ckl::UnaryOp<RowSumF32<Slot>, Slot> {
    static ALWI void init() { sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(ckl::to_u32(Slot) + slot_offset);
    }
};

// hc step parameters (bit-cast fp32 runtime args, re-sent every call).
struct HcParams {
    float a_pre, a_post, a_res;
    float base[4 * 6];
    float inv_norm_width, norm_eps, hc_eps;
    uint32_t iters;
    float inv_c, y_norm_eps;  // norm: rstd = rsqrt(sum y^2 * inv_c + y_norm_eps)
};

#ifdef TRISC_MATH
// ---- hc_step_row, lane layout: SFPU vector v (dst_reg[v]) holds stats value v for the 32 tokens of the row ----
// v = 0..n-1 pre mix, n..2n-1 post mix, 2n..2n+n^2-1 comb (row-major j*n+k), 2n+n^2 = sum X'^2.
// Output in place: v = 0..NG-1 = [pre | post | comb]. All fp32, approximation off.
namespace hc {
using sfpi::dst_reg;
using sfpi::vFloat;

constexpr uint32_t NG = n * (n + 2);
constexpr uint32_t V_COMB = 2 * n;
constexpr uint32_t V_SUMSQ = NG;

// x^-1: hardware estimate + 2 Newton steps (no programmable constants: they are shared with other SFPU ops).
sfpi_inline vFloat recip(vFloat x) {
    vFloat y = sfpi::approx_recip(x);
    vFloat t = -x * y + 2.0f;
    y = y * t;
    t = -x * y + 2.0f;
    y = y * t;
    return y;
}
// x^-1/2: bit-trick seed + 3 Newton steps (seed 3.4e-2 -> 1.7e-3 -> 4.4e-6 -> 3e-11 + rounding).
sfpi_inline vFloat rsqrt(vFloat x) {
    sfpi::vInt i = sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(x) >> 1);
    sfpi::vInt magic = 0x5f3759df;
    vFloat y = sfpi::as<vFloat>(magic - i);
    vFloat hx = x * 0.5f;
#pragma GCC unroll 3
    for (int it = 0; it < 3; ++it) {
        vFloat t = -(hx * y) * y + 1.5f;
        y = y * t;
    }
    return y;
}
sfpi_inline vFloat sigmoid(vFloat x) { return recip(ckernel::sfpu::_sfpu_exp_fp32_accurate_<false>(-x) + 1.0f); }

// The DEST pointer walks the value vectors at runtime (dst_reg += k; clear_dst_reg_addr() returns to the tile
// base) so the step is a few small loops, not a fully unrolled program (kernel-config buffer budget).
// dst_reg += K in steps the INCRWC immediate can encode (at most 3 vectors per step)
template <uint32_t K>
sfpi_inline void advance() {
    if constexpr (K > 3) {
        dst_reg += 3;
        advance<K - 3>();
    } else if constexpr (K > 0) {
        dst_reg += static_cast<int>(K);
    }
}
template <uint32_t V>
sfpi_inline void to_vector() {
    ckernel::math::clear_dst_reg_addr();
    advance<V>();
}

// softmax over the row of the logits at the pointer, then + hc_eps
sfpi_inline void softmax_row(vFloat hc_eps) {
    vFloat m = dst_reg[0];
#pragma GCC unroll 4
    for (uint32_t k = 1; k < n; ++k) {
        vFloat v = dst_reg[k];
        v_if(v > m) { m = v; }
        v_endif;
    }
    vFloat sum = 0.0f;
#pragma GCC unroll 4
    for (uint32_t k = 0; k < n; ++k) {
        vFloat e = ckernel::sfpu::_sfpu_exp_fp32_accurate_<false>(vFloat(dst_reg[k]) - m);
        dst_reg[k] = e;
        sum = sum + e;
    }
    vFloat r = recip(sum);
#pragma GCC unroll 4
    for (uint32_t k = 0; k < n; ++k) {
        dst_reg[k] = vFloat(dst_reg[k]) * r + hc_eps;
    }
}
// the n values at the pointer with stride `stride` (a row: 1, a column: n) /= (their sum + eps)
template <uint32_t stride>
sfpi_inline void normalize(vFloat hc_eps) {
    vFloat sum = dst_reg[0];
#pragma GCC unroll 4
    for (uint32_t k = 1; k < n; ++k) {
        sum = sum + dst_reg[k * stride];
    }
    vFloat r = recip(sum + hc_eps);
#pragma GCC unroll 4
    for (uint32_t k = 0; k < n; ++k) {
        dst_reg[k * stride] = vFloat(dst_reg[k * stride]) * r;
    }
}
// M[j][:] /= (row sum + eps) for every row j  (rows = n consecutive vectors)
sfpi_inline void rows_step(vFloat e) {
    to_vector<V_COMB>();
#pragma GCC unroll 1
    for (uint32_t j = 0; j < n; ++j) {
        normalize<1>(e);
        advance<n>();
    }
}
// M[:][k] /= (column sum + eps) for every column k  (a column = vectors k, k+n, ...)
sfpi_inline void cols_step(vFloat e) {
    to_vector<V_COMB>();
#pragma GCC unroll 1
    for (uint32_t k = 0; k < n; ++k) {
        normalize<n>(e);
        dst_reg += 1;
    }
}

inline void hc_step(const HcParams* p) {
    const vFloat e = p->hc_eps;  // runtime scalar materialised once, kept in a vector register
    const vFloat s = rsqrt(vFloat(dst_reg[V_SUMSQ]) * p->inv_norm_width + p->norm_eps);
    // pre_k = sigmoid(a_pre z_k + b_k) + eps ; post_k = 2 sigmoid(a_post z_{n+k} + b_{n+k})
#pragma GCC unroll 1
    for (uint32_t k = 0; k < n; ++k) {
        vFloat z = vFloat(dst_reg[0]) * s;
        dst_reg[0] = sigmoid(z * p->a_pre + p->base[k]) + e;
        dst_reg += 1;
    }
#pragma GCC unroll 1
    for (uint32_t k = n; k < 2 * n; ++k) {
        vFloat z = vFloat(dst_reg[0]) * s;
        vFloat sg = sigmoid(z * p->a_post + p->base[k]);
        dst_reg[0] = sg + sg;
        dst_reg += 1;
    }
    // logits L_jk = a_res z + b, then the row softmax (+ eps) and the first column step
#pragma GCC unroll 1
    for (uint32_t jk = V_COMB; jk < NG; ++jk) {
        vFloat z = vFloat(dst_reg[0]) * s;
        dst_reg[0] = z * p->a_res + p->base[jk];
        dst_reg += 1;
    }
    to_vector<V_COMB>();
#pragma GCC unroll 1
    for (uint32_t j = 0; j < n; ++j) {
        softmax_row(e);
        advance<n>();
    }
    cols_step(e);
#pragma GCC unroll 1
    for (uint32_t it = 1; it < p->iters; ++it) {
        rows_step(e);
        cols_step(e);
    }
    ckernel::math::clear_dst_reg_addr();
}
// norm: rstd of y in vector V_SUMSQ + 1 (in place of sum y_b^2) — its own step, run (and sent) before the hc step
inline void rstd_step(const HcParams* p) {
    to_vector<V_SUMSQ + 1>();
    dst_reg[0] = rsqrt(vFloat(dst_reg[0]) * p->inv_c + p->y_norm_eps);
    ckernel::math::clear_dst_reg_addr();
}
}  // namespace hc
#endif

template <ckl::Dst Slot>
struct RstdStep : ckl::UnaryOp<RstdStep<Slot>, Slot> {
    const HcParams* params;
    static ALWI void init() { MATH((SFPU_UNARY_INIT(unused))); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const {
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            hc::rstd_step, ckl::to_u32(Slot) + slot_offset, VectorMode::RC_custom, params)));
    }
};
template <ckl::Dst Slot>
struct HcStep : ckl::UnaryOp<HcStep<Slot>, Slot> {
    const HcParams* params;
    static ALWI void init() { MATH((SFPU_UNARY_INIT(unused))); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const {
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            hc::hc_step, ckl::to_u32(Slot) + slot_offset, VectorMode::RC_custom, params)));
    }
};

// ---------------- combine_row: fixed-order fold of `count` slot tiles -> one tile (raw LLK, see header) ----
// The slot CB is always waited / popped whole (`pages`) so its FIFO pointer stays on the base address the remote
// writers target; only the first `count` slots carry a partial.
ALWI void fold_slots(uint32_t cb_slots, uint32_t pages, uint32_t count, uint32_t cb_out) {
    cb_wait_front(cb_slots, pages);
    reconfig_data_format_srca(cb_slots);
    pack_reconfig_data_format(cb_out);
    cb_reserve_back(cb_out, 1);
    tile_regs_acquire();
    copy_init(cb_slots);
    copy_tile(cb_slots, 0, 0);
    for (uint32_t k = 1; k < count; ++k) {
        copy_init(cb_slots);
        copy_tile(cb_slots, k, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
    }
    tile_regs_commit();
    cb_pop_front(cb_slots, pages);
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
}

}  // namespace

void kernel_main() {
    const uint32_t t_start = get_arg_val<uint32_t>(0);
    const uint32_t t_count = get_arg_val<uint32_t>(1);
    const uint32_t c_count = get_arg_val<uint32_t>(2);
    [[maybe_unused]] const uint32_t member_q = get_arg_val<uint32_t>(3);

    compute_kernel_hw_startup(cb_stream_tiles, cb_coef_cols, cb_mixed_out_tiles);

    [[maybe_unused]] HcParams hcp{};
    if constexpr (collapse) {
        auto f = [](uint32_t i) {
            union {
                uint32_t u;
                float f;
            } c{get_common_arg_val<uint32_t>(i)};
            return c.f;
        };
        hcp.a_pre = f(0);
        hcp.a_post = f(1);
        hcp.a_res = f(2);
        for (uint32_t k = 0; k < n * (n + 2); ++k) {
            hcp.base[k] = f(3 + k);
        }
        hcp.inv_norm_width = f(3 + n * (n + 2));
        hcp.norm_eps = f(4 + n * (n + 2));
        hcp.hc_eps = f(5 + n * (n + 2));
        hcp.iters = get_common_arg_val<uint32_t>(6 + n * (n + 2));
        hcp.inv_c = f(7 + n * (n + 2));
        hcp.y_norm_eps = f(8 + n * (n + 2));
    }

    // has_w && w_resident: W is resident for the whole kernel (never popped), waited in MixPassFn
    // my chunk (combine tree level 1) and its size
    [[maybe_unused]] const uint32_t my_chunk = member_q / fanin;
    [[maybe_unused]] const uint32_t chunk_size =
        (group_q - my_chunk * fanin) < fanin ? (group_q - my_chunk * fanin) : fanin;

    [[maybe_unused]] const uint32_t num_passes_all = (c_count + block_cols - 1) / block_cols;
    // combine_row level 1 of row index i (local), lagged one row: I lead my chunk for it
    [[maybe_unused]] auto lead_fold = [&](uint32_t i) __attribute__((always_inline)) {
        if (member_q == my_chunk * fanin + ((t_start + i) % chunk_size)) {
            MaybeDeviceZoneScope("c_lead_fold");
            fold_slots(cb_leader_slots, fanin, chunk_size, cb_chunk_sum);
        }
    };
    // combine_row level 2 + hc_step_row of a row I root (the writer hands its slot region over in order)
    [[maybe_unused]] auto root_step = [&]() __attribute__((always_inline)) {
        {
            MaybeDeviceZoneScope("c_root_fold");
            fold_slots(cb_root_slots, num_chunks, num_chunks, cb_stats);
        }
        if constexpr (norm) {
            MaybeDeviceZoneScope("c_rstd");
            // rstd first (it needs sum y_b^2 only): the writer sends it to the members while the hc step runs; the
            // lane tile stays in cb_hc_in for the hc step (which pops it)
            cb_wait_front(cb_hc_in, 1);
            ckl::eltwise_chain(
                ckl::IterationShape::one_tile(),
                ckl::CopyTile<caller_managed(cb_hc_in), ckl::Dst::D0>{0},
                RstdStep<ckl::Dst::D0>{{}, &hcp},
                ckl::PackTile<ckl::output(cb_rstd_lane), ckl::Dst::D0>{});
        }
        if constexpr (collapse) {
            // hc_step_row: the writer permutes the stats tile into lane layout (cb_hc_in)
            MaybeDeviceZoneScope("c_hc");
            ckl::eltwise_chain(
                ckl::IterationShape::one_tile(),
                ckl::CopyTile<ckl::input(cb_hc_in), ckl::Dst::D0>{},
                HcStep<ckl::Dst::D0>{{}, &hcp},
                ckl::PackTile<ckl::output(cb_hc_out), ckl::Dst::D0>{});
        }
    };
    // norm_block of the oldest held row: y = (y_b * rstd) * w per pass (filler units included: the writer drops
    // them); rstd arrives in column 0 of a cb_rstd_col page (the writer expands the root's unicast)
    [[maybe_unused]] auto norm_row = [&]() __attribute__((always_inline)) {
        {
            MaybeDeviceZoneScope("c_wait_rstd");
            cb_wait_front(cb_norm_w, num_passes_all * block_cols);  // resident, never popped
            cb_wait_front(cb_rstd_col, 1);
        }
        MaybeDeviceZoneScope("c_norm");
        for (uint32_t p = 0; p < num_passes_all; ++p) {
            cb_wait_front(cb_y_hold, block_cols);
            norm_pass(p * block_cols, std::make_index_sequence<block_cols>{});
            cb_pop_front(cb_y_hold, block_cols);
        }
        cb_pop_front(cb_rstd_col, 1);
    };

    // Virtual rows: rows 0..t_count-1 do their row work; with norm the loop runs norm_lag rows further, and every
    // virtual row i then also runs the root work of row i - root_lag (if I root it) and norm_block(i - norm_lag).
    // The writer walks the same order (lead / root / partial / norm writes).
    const uint32_t vrows = t_count + (lagged ? norm_lag : 0);
    for (uint32_t i = 0; i < vrows; ++i) {
        if (i < t_count) {
            MaybeDeviceZoneScope("c_row");
            {
                MaybeDeviceZoneScope("c_wait_coef");
                cb_wait_front(cb_coef_cols, n_coef);  // the row's expanded coefficients, held for every block
            }
            build_kpages();
            if constexpr (!has_w) {
                for (uint32_t cb0 = 0; cb0 < c_count; cb0 += block_cols) {
                    const uint32_t cols_in_pass = (c_count - cb0) < block_cols ? (c_count - cb0) : block_cols;
                    mix_pass(cols_in_pass);
                }
                cb_pop_front(cb_coef_cols, n_coef);
                cb_pop_front(cb_kpages, n_kpages);
            } else {
                const PassCols pc{c_count};
                const uint32_t num_passes = num_passes_all;
                if constexpr (norm) {  // the row's sums of squares start at 0
                    ckl::eltwise_chain(
                        ckl::IterationShape::one_tile(),
                        ckl::FillScalar<ckl::Dst::D0>{0.0f},
                        ckl::PackTile<ckl::output(cb_sumsq_acc), ckl::Dst::D0>{},
                        ckl::PackTile<ckl::output(cb_ysq_acc), ckl::Dst::D0>{});
                } else {
                    ckl::eltwise_chain(
                        ckl::IterationShape::one_tile(),
                        ckl::FillScalar<ckl::Dst::D0>{0.0f},
                        ckl::PackTile<ckl::output(cb_sumsq_acc), ckl::Dst::D0>{});
                }
                // project_block: K-blocks = column passes; each pass is mixed by the PreKBlockFn
                ::CircularBuffer in0_buf(cb_mixed_bf16), in1_buf(cb_w_slice), out_buf(cb_proj_out),
                    interm_buf(cb_proj_interm);
                ckl::matmul_block<
                    false,
                    false,
                    ckl::LastBlockTarget::Out,
                    ckl::OutputCBLayout::SubblockMajor,
                    ckl::matmul_config::InitMode::ShortAfterPreKBlock,
                    ckl::InputPolicy::WaitAndPopPerKBlock,
                    w_resident ? ckl::InputPolicy::NoWaitNoPop : ckl::InputPolicy::WaitAndPopPerKBlock,
                    ckl::NoPostCompute,
                    MixPassFn,
                    ckl::NoPostKBlock,
                    0,
                    ckl::NoKBlockInnerDimFn,
                    ckl::NoIn0Source,
                    PassIn1Offset>(
                    in0_buf,
                    in1_buf,
                    out_buf,
                    interm_buf,
                    ckl::MatmulBlockShape::of(1, 1, 1, 1, pass_k_tiles, num_passes),
                    ckl::NoPostCompute{},
                    MixPassFn{
                        pc,
                        (!lagged || i < norm_lag ? i : norm_lag) * num_passes * block_cols,  // held rows ahead of i
                        lagged ? ((i < norm_lag ? 0 : i - norm_lag) * num_passes * block_cols) % hold_pages : 0},
                    1,
                    0,
                    ckl::NoPostKBlock{},
                    ckl::NoKBlockInnerDimFn{},
                    ckl::NoIn0Source{},
                    PassIn1Offset{});
                cb_pop_front(cb_coef_cols, n_coef);
                cb_pop_front(cb_kpages, n_kpages);

                // finalize_row: sum of squares -> column 0 (the writer merges it into the partial's col 24)
                MaybeDeviceZoneScope("c_finalize");
                ckl::eltwise_chain(
                    ckl::IterationShape::one_tile(),
                    ckl::CopyTile<ckl::input(cb_sumsq_acc), ckl::Dst::D0>{},
                    RowSumF32<ckl::Dst::D0>{},
                    ckl::PackTile<ckl::output(cb_sumsq_col), ckl::Dst::D0>{});
                if constexpr (norm) {  // sum y_b^2 -> column 0 of the row's second cb_sumsq_col page (col ng+1)
                    ckl::eltwise_chain(
                        ckl::IterationShape::one_tile(),
                        ckl::CopyTile<ckl::input(cb_ysq_acc), ckl::Dst::D0>{},
                        RowSumF32<ckl::Dst::D0>{},
                        ckl::PackTile<ckl::output(cb_sumsq_col), ckl::Dst::D0>{});
                }

                // combine_row level 1, lagged one row (rotating leaders spread the folds; the lag means the chunk's
                // partials have landed by the time I get here)
                if (i > 0) {
                    lead_fold(i - 1);
                }
            }
        } else if constexpr (has_w) {
            if (i == t_count && t_count > 0) {  // the lagged level-1 fold of my last row
                lead_fold(t_count - 1);
            }
        }
        if constexpr (lagged) {
            if (i >= root_lag && i - root_lag < t_count && (i - root_lag) % group_q == member_q) {
                root_step();
            }
            if (i >= norm_lag) {
                norm_row();
            }
        }
    }

    if constexpr (has_w && !lagged) {
        if (t_count > 0) {  // the lagged level-1 fold of my last row
            lead_fold(t_count - 1);
        }
        // combine_row level 2 + hc_step_row, DEFERRED past my own rows (no core's row loop waits on a root): every
        // row t I root (t - t_start = member_q + k*Q) has its own slot region, handed over by the writer in order k.
        const uint32_t rooted = member_q < t_count ? (t_count - member_q + group_q - 1) / group_q : 0;
        for (uint32_t k = 0; k < rooted; ++k) {
            root_step();
        }
    }
    if constexpr (norm && !lagged) {  // deferred: every row's rstd returns once the roots have run
        for (uint32_t r = 0; r < t_count; ++r) {
            norm_row();
        }
    }
}
