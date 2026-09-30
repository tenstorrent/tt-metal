// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_pre compute kernel. Per block (block_token_tiles token tile-rows of this rank's K slice):
//
//   w_split_block       ONCE, before block 0, fp32 W only: each fp32 W tile k of cb_weight is rewritten
//                        in place (aliased cb_weight_split) into the exact bf16 pair [W_hi(k), W_lo(k)]
//   project_block       bf16 W:  matmul_block helper: mix partial = X_blk @ W_slice -> cb_partial [mix rows]
//                        (in0 WaitAndRetainOnLastBlock, num_k_blocks = 1: the X block stays resident)
//                        fp32 W:  project_sumsq_streamed: per K chunk (x_stream_chunks per row, as the reader
//                        publishes them) X @ W_hi + X @ W_lo on the exactly reloaded fp32 running mix, and
//                        the chunk's sum x^2 partial -> cb_sq_acc (both stream under the X burst)
//   sumsq_block         eltwise_chain Mul x*x DEST-accumulated over K (bf16 W: after the projection)
//                        + reduce<SUM, REDUCE_ROW, Accurate> (fp32 W: over the chunk partials) -> cb_partial
//                        [sumsq rows]; cb_partial is [mix rows | sumsq rows]
//   combine_block       root only: rank-ordered fp32 SFPU fold of cb_gathered -> cb_combined
//   coefficients_block  S row-major (cb_coef_in, multicast by the root) -> transpose_tile -> SFPU subvector
//                        gather into coefficient-major -> custom SFPU coefficients (r, pre) -> pre_i tiles
//                        (subvector scatter + transpose_dest) -> cb_pre_cols
//   ymix_block          eltwise_chain Mul x_i * bcast_col(pre_i), DEST-accumulated over i -> cb_y_out
//   owned_block         owned rows, fused in front of coefficients_block: gather + coefficients, Sinkhorn,
//                        post / comb row-major tiles (subvector scatter + transpose_dest), in one DEST window
//                        -> cb_comb_coef; the coefficient tile -> cb_coef_keep (the pre tiles reload it)
//
// Block schedule (Perf 1, cross-block pipeline). Step b (proj(b) done on entry):
//   pipelined step (pipe_at(b)): proj(b+1) (X(b+1) sits behind X(b) in cb_x_resident, XView), the root's fold of
//     b (if not yet) and -- after the coefficients of b -- of b+1, then tail(b) = coefficients + y-mix of b. The
//     group's round trip for b+1 thus overlaps every rank's tail(b), and proj(b+1) runs while S(b) is in flight.
//   serial step: [fold b]; tail(b); proj(b+1).
//   pipe_at(b) = b+1 < num_blocks and (x_block_depth >= 3, or b + x_block_depth >= num_blocks): holding X(b) past
//   proj(b+1) delays the reader's X(b+depth); with depth 2 only the last step (nothing left to prefetch) pipelines.
//   depth 1 is always serial. Measured (BH, bf16 X / fp32 W): 1280x4096 148.3 -> 138.2 us, 2048x5120 313 -> 302,
//   4096x1792 238 -> 228.
//
// fp32 X (x_pieces == 3, Refinement 2; compile-time gated, the bf16-X path is unchanged):
//   w_grid_split_block  ONCE: max|W| -> grid; W -> [W0 on a 2^-W_GRID_BITS grid, W - W0] (bf16, in place)
//   x_stats_block       per token row-tile: exact SFPU lane-wise sum x^2 -> cb_sq_acc (reduced as before) and
//                        its max -> reduce<MAX, REDUCE_SCALAR> -> the row-tile's x grid (cb_grid)
//   project_block_split per K chunk: x_split_window (x -> [x0 on grid, x1_hi, x1_mid], bf16) + ONE DEST window:
//                        exact fp32 reload of the running mix, sum_{q+p<=2} x_q @ W_p on top -> cb_mix_run /
//                        cb_partial. x0 @ W0 is exact in-tile (on-grid products); the rest is 2^-4 smaller.
//
// Raw-LLK deviations (helper considered and rejected; see op_design.md "Helpers considered and rejected"):
//   * project_block_split (fp32 X): matmul_block cannot take a per-chunk exact reload + several (in0, in1)
//     piece pairs into one DEST window; realized over matmul_tiles like project_block_pieces.
//   * x_split_window / w_grid_split_block / stats_pass: copy_tile (UnpackToDestFp32) -> custom SFPI -> packs
//     of several DEST tiles per input tile (the chain has one pack terminal per element).
//   * grid_block: unary_bcast<SCALAR> (no chain element broadcasts one CB tile into DEST); the max itself
//     is the reduce helper.
//   * project_block_pieces (fp32 W): matmul_block cannot accumulate two in1 operands (W_hi, W_lo) into one
//     DEST window — a second call would pack and FPU-reload the fp32 partial (tf32 truncation). Realized
//     as a thin block op over matmul_tiles: per output sub-block, all K x pieces products accumulate in
//     DEST; the X block is waited but never popped (same retention contract as the helper path).
//   * project_sumsq_streamed (bf16 X / fp32 W, Refinement 4): the project_block_pieces realization split into
//     K-chunk DEST windows so it streams under the X burst, the fp32 running mix reloaded exactly between
//     windows (UnpackToDestFp32 copy, as project_block_split); matmul_block has no exact partial reload.
//     Its sum x^2 half is the sumsq_row chain + the reduce helper (one partial tile per chunk).
//   * project_block_pieces<1> (bf16 X / bf16 W, pipelined proj(b+1) only): matmul_block reads in0 from the CB
//     front and has no tile-index base, so it cannot project a block that sits behind the resident X(b)
//     (bitwise identical to the helper's DEST accumulation; the front-block projection keeps the helper).
//   * w_split_block: copy_tile (UnpackToDestFp32) -> SFPU bit mask / subtract -> two packs per tile; the
//     chain has one pack terminal per element, and the in-place alias rewrite needs explicit page indices.
//   * combine_block: reduce<AccumulateViaAdd> reads the fp32 partials through the FPU (tf32 truncation);
//     eltwise_chain cannot express a runtime group_cores-deep fold in one DEST window. Realized with the
//     chain's own two primitives, copy_tile (UnpackToDestFp32) + add_binary_tile (SFPU), in a loop.
//   * coefficient layout transforms (gather_coef_major / pre_tiles / post_comb_tiles): a data permutation
//     (coefficient columns <-> per-token lanes) no helper expresses; built from transpose_tile /
//     transpose_dest (exact fp32) + the SFPI subvector transpose.
//   * coefficients_block / sinkhorn_block: the math runs across slots WITHIN one tile (row / column sums
//     over the n x n matrix, per-token softmax); chain SFPU elements are whole-tile elementwise. Realized as
//     custom SFPI functions over one DEST tile that call the non-approximate SFPI exp primitive, with a
//     Newton-refined reciprocal / rsqrt (no bare approximate reciprocal).
//   * sinkhorn_block iterations (Perf round 2): hand-scheduled SFPLOADMACRO passes in raw TTI (load / multiply /
//     write-back of an entry in one issued instruction); SFPLOADMACRO has no sfpi or kernel_lib expression, and its
//     scheduled sub-unit instructions have no interlocks, so the issue order is a generated, rule-checked static
//     schedule. Bit-identical to the plain row_norm / col_norm formulation with fused Newton MADs.

#include <stdint.h>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/bcast.h"
#include "api/compute/transpose.h"
#include "api/compute/transpose_dest.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "perf_instrumentation.hpp"

// Stage zones (permanent; opt-in via the KERNEL_PERF_ZONES define, see perf_instrumentation.hpp). Only the
// unpack thread blocks on cb_wait_front and only pack on cb_reserve_back; math-thread numbers are occupancy.
//   c_w_split / c_w_publish   W all-gather: split the own share / wait for the whole slice
//   c_x_wait                  per K chunk: waiting for the reader's X chunk (unpack)
//   c_proj / c_sumsq          per K chunk: projection window / sum x^2 window (bf16 X, fp32 W)
//   c_sq_reduce               per block: REDUCE_ROW over the sum x^2 partials
//   c_gather_wait / c_combine root: all ranks' partials landed / the fp32 fold
//   c_coef_wait               the multicast S landed in cb_coef_in
//   c_owned / c_pre           owned row: coefficients + Sinkhorn + post/comb window / pre-column window
//   c_ymix                    the y-mix (includes cb_y_out back-pressure on pack)
// Ablation (perf tournaments only; outputs are garbage, sync scaffolding kept): MHC_ABLATE_PROJ (projection
// + sum x^2 math), MHC_ABLATE_COEF (coefficient / Sinkhorn / layout SFPU math), MHC_ABLATE_YMIX (y-mix math).

// DEST tiles of the coefficient layout transforms (visible on every TRISC).
constexpr uint32_t mhc_t_mix = 2;  // the transposed mix tile of S
constexpr uint32_t mhc_t_sq = 3;   // the transposed sum(x^2) tile of S (row 0 = sum x^2 per token)

#ifdef TRISC_MATH
#include "sfpi.h"
#include "ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#endif

constexpr uint32_t cb_x_resident = get_compile_time_arg_val(0);
constexpr uint32_t cb_weight = get_compile_time_arg_val(1);
constexpr uint32_t cb_bias_coef = get_compile_time_arg_val(2);
constexpr uint32_t cb_reduce_scaler = get_compile_time_arg_val(3);
constexpr uint32_t cb_sq_acc = get_compile_time_arg_val(4);
constexpr uint32_t cb_partial = get_compile_time_arg_val(5);
constexpr uint32_t cb_gathered = get_compile_time_arg_val(6);
constexpr uint32_t cb_combined = get_compile_time_arg_val(7);
constexpr uint32_t cb_coef_in = get_compile_time_arg_val(8);
// Owned rows: the coefficient-major tile (after coefficients + Sinkhorn; the pre slots are untouched) is parked
// here (fp32, exact reload) so the pre columns are built without a second S gather + coefficients pass.
constexpr uint32_t cb_coef_keep = get_compile_time_arg_val(9);
// CT arg 10 is unused (formerly a writer-scattered coefficient CB).
constexpr uint32_t cb_comb_coef = get_compile_time_arg_val(11);
constexpr uint32_t cb_pre_cols = get_compile_time_arg_val(12);
constexpr uint32_t cb_y_out = get_compile_time_arg_val(13);
constexpr uint32_t n_streams = get_compile_time_arg_val(14);
constexpr uint32_t block_token_tiles = get_compile_time_arg_val(15);
constexpr uint32_t core_k_tiles_max = get_compile_time_arg_val(16);
constexpr uint32_t group_cores = get_compile_time_arg_val(17);
constexpr uint32_t cb_weight_split = get_compile_time_arg_val(18);
constexpr uint32_t w_pieces = get_compile_time_arg_val(19);       // 1: bf16 W (exact); 2: fp32 W -> hi/lo
constexpr uint32_t w_chunk_tiles = get_compile_time_arg_val(20);  // reader's W push quantum
// Math fidelity of the W_lo products (pieces p >= 1). W_lo is tiny (<= 2^-8 |W|) and, for a tf32-valued W,
// carries <= 3 significant bits, so a low fidelity costs ~nothing in precision and saves FPU phases.
constexpr auto w_lo_fidelity = static_cast<ckernel::MathFidelity>(get_compile_time_arg_val(21));
// == MATH_FIDELITY, but visible on every TRISC (the unpack order depends on whether the lo pass is separate).
constexpr auto w_main_fidelity = static_cast<ckernel::MathFidelity>(get_compile_time_arg_val(22));
constexpr uint32_t cb_w_matmul = w_pieces > 1 ? cb_weight_split : cb_weight;
// fp32 X (x_pieces == 3): exact-grid projection / exact sum x^2 path (Refinement 2).
//   cb_x_fp32     aliases cb_x_resident (same allocation, UnpackToDestFp32): exact fp32 X for the SFPU
//   cb_x_pieces   one K chunk window of bf16 X pieces [x0, x1_hi, x1_mid] (streams K)
//   cb_mix_run    fp32 running mix between chunk windows (reloaded exactly)
//   cb_max_lanes  lane-wise max|v| tile -> reduce<MAX, REDUCE_SCALAR> -> cb_max_scalar
//   cb_grid       per token row-tile (and once for W) rounding-constant tile c = 1.5 * 2^23 * grid
constexpr uint32_t cb_x_fp32 = get_compile_time_arg_val(23);
constexpr uint32_t cb_x_pieces = get_compile_time_arg_val(24);
constexpr uint32_t cb_mix_run = get_compile_time_arg_val(25);
constexpr uint32_t x_pieces = get_compile_time_arg_val(26);         // 1: bf16 X (exact); 3: fp32 X grid split
constexpr uint32_t x_chunk_k_tiles = get_compile_time_arg_val(27);  // K tiles per piece window
constexpr uint32_t x_piece_rows = get_compile_time_arg_val(28);     // piece window rows (>= sub-block height)
constexpr uint32_t cb_max_lanes = get_compile_time_arg_val(29);
constexpr uint32_t cb_max_scalar = get_compile_time_arg_val(30);
constexpr uint32_t cb_grid = get_compile_time_arg_val(31);
constexpr uint32_t cb_max_scaler = get_compile_time_arg_val(32);
constexpr uint32_t x_grid_bits = get_compile_time_arg_val(33);        // x0 = k * 2^(E - bits), |k| <= 2^bits
constexpr uint32_t w_grid_bits = get_compile_time_arg_val(34);        // W0 likewise (x_grid_bits + w_grid_bits <= 10)
constexpr uint32_t product_order_max = get_compile_time_arg_val(35);  // keep piece products with q + p <= this
constexpr uint32_t product_lo_order = get_compile_time_arg_val(36);   // products with q + p >= this at x_lo_fidelity
constexpr auto x_lo_fidelity = static_cast<ckernel::MathFidelity>(get_compile_time_arg_val(37));
// W column all-gather with the fp32-W hi/lo split done per share (see w_split_own_share).
constexpr uint32_t cb_w_own_ready = get_compile_time_arg_val(38);
constexpr uint32_t cb_w_own_split = get_compile_time_arg_val(39);
constexpr bool w_presplit = get_compile_time_arg_val(40) != 0;
// bf16 X / fp32 W: projection and sum x^2 stream together over this many K chunks per token row (the reader's
// X_STREAM_CHUNKS publish granularity); 1 = the whole row in one window each.
constexpr uint32_t x_stream_chunks = get_compile_time_arg_val(41);
constexpr uint32_t x_block_depth = get_compile_time_arg_val(42);                // cb_x_resident ring depth (blocks)
constexpr uint32_t x_window_pages = x_pieces * x_chunk_k_tiles * x_piece_rows;  // nominal push per window
constexpr bool x_grid_split = x_pieces > 1;
// fp32 X + fp32 W: the W hi/lo split is replaced by the W grid split (same 2 bf16 pieces, same alias).

#ifdef TRISC_MATH
namespace mhc_sfpu {

using namespace sfpi;
using ckernel::sfpu::Converter;

constexpr int N = static_cast<int>(n_streams);
constexpr int MIX = N * (N + 2);  // slots 0..MIX-1 hold the mixes, slot MIX holds sum(x^2)
constexpr int LOGIT0 = 2 * N;     // first logit / comb slot
constexpr int SCRATCH0 = 31;      // unused slots (MIX+1 .. 31) serve as per-lane scratch
constexpr int SCRATCH1 = 30;
constexpr int TILE_SLOTS = 32;  // dst_reg stride of one DEST tile

// 1/x for x > 0 finite: hardware seed + two Newton steps (~fp32 accurate).
sfpi_inline vFloat recip_pos(vFloat x) {
    vFloat y = approx_recip(x);
    vFloat t = 2.0f - x * y;
    y = y * t;
    t = 2.0f - x * y;
    y = y * t;
    return y;
}

// 1/sqrt(x) for x > 0 finite: bit-trick seed + four Newton steps.
sfpi_inline vFloat rsqrt_pos(vFloat x) {
    vInt i = as<vInt>(as<vUInt>(x) >> 1);
    vInt magic = 0x5f3759df;
    vFloat y = as<vFloat>(magic - i);
    vFloat half_x = x * 0.5f;
#pragma GCC unroll 0
    for (int it = 0; it < 4; ++it) {
        y = y * (1.5f - half_x * y * y);
    }
    return y;
}

// sigmoid(x) = 1 / (1 + exp(-x)); exp argument clamped so 1 + exp stays finite.
sfpi_inline vFloat sigmoid_acc(vFloat x) {
    vFloat z = -x;
    v_if(z > 80.0f) { z = 80.0f; }
    v_endif;
    vFloat e = ckernel::sfpu::_sfpu_exp_fp32_accurate_<false>(z);
    return recip_pos(e + 1.0f);
}

// w_split_block on one DEST tile pair: tile 0 = fp32 W in, -> tile 0 = W_hi (bf16-truncated), tile 1 =
// W_lo = W - W_hi (exact in fp32; exactly bf16 for a tf32-valued W).
void w_split_hi_lo() {
    vUInt hi_mask = 0xFFFF0000;
#pragma GCC unroll 8
    for (int k = 0; k < TILE_SLOTS; ++k) {
        vFloat v = dst_reg[k];
        vFloat hi = as<vFloat>(as<vUInt>(v) & hi_mask);
        dst_reg[k] = hi;
        dst_reg[TILE_SLOTS + k] = v - hi;
    }
}

// ---- Exact-grid split (fp32 X). The FPU's in-tile (32-long) dot product rounds its sum to ~11 bits below
// the largest product, but sums of products that all lie on one power-of-two grid within that window are
// exact (probes 013-015). So each operand gets a leading piece on a per-tile grid g = 2^(E - bits), |v| < 2^E:
// v0 = round-to-grid(v) (|v0 / g| <= 2^bits, exactly bf16), and bf16 pieces of the exact remainder v - v0.
// The x0 @ W0 product (the whole leading part of the mix) is then exact; the remainder products are
// 2^-bits smaller, and so is their in-tile rounding noise.

// v0 = (v + c) - c with c = 1.5 * 2^23 * g: the add lands on a float whose ulp is g (|v| < 2^22 g).
sfpi_inline vFloat round_to_grid(vFloat v, vFloat c) {
    vFloat t = v + c;
    return t - c;
}

// DEST tile 0: bcast bound m (every lane) -> rounding constant c = 1.5 * 2^(E - BITS + 23) with |v| < 2^E.
// SQ_BOUND: m = max over lanes of a lane-wise sum of squares, so |v| <= sqrt(m) < 2^ceil((e2 + 1) / 2)
// (e2 = unbiased exponent of m); in biased terms E + 127 = ((eb2 + 3) >> 1) + 63. Otherwise m = max|v|:
// E = exp(m) + 1.
template <int BITS, bool SQ_BOUND>
void grid_constant() {
#pragma GCC unroll 4
    for (int k = 0; k < TILE_SLOTS; ++k) {
        vFloat m = dst_reg[k];
        vInt e = as<vInt>(as<vUInt>(m) >> 23);  // m >= 0: biased exponent
        if constexpr (SQ_BOUND) {
            e = e + 3;
            e = as<vInt>(as<vUInt>(e) >> 1);
            e = e + (86 - BITS);  // (E + 127) + (23 - BITS)
        } else {
            e = e + (24 - BITS);
        }
        v_if(e < 1) { e = 1; }
        v_endif;
        v_if(e > 254) { e = 254; }
        v_endif;
        vUInt cb = (as<vUInt>(e) << 23) | vUInt(0x400000);
        dst_reg[k] = as<vFloat>(cb);
    }
}

// x split, DEST tile 0 = fp32 x, tile 3 = c -> tiles 0..2 = [x0, x1_hi = trunc(x - x0), x1_mid = the rest].
// x1_mid (<= 2^-(bits+8) |x|) is left in fp32: the bf16 pack's conversion error on it is ~2^-(bits+16) |x|.
void x_split_grid() {
    vUInt hi_mask = 0xFFFF0000;
    vFloat c = dst_reg[3 * TILE_SLOTS];  // uniform over the tile (scalar broadcast)
#pragma GCC unroll 4
    for (int k = 0; k < TILE_SLOTS; ++k) {
        vFloat v = dst_reg[k];
        vFloat v0 = round_to_grid(v, c);
        vFloat r = v - v0;
        vFloat rh = as<vFloat>(as<vUInt>(r) & hi_mask);
        dst_reg[k] = v0;
        dst_reg[TILE_SLOTS + k] = rh;
        dst_reg[2 * TILE_SLOTS + k] = r - rh;
    }
}

// W split, DEST tile 0 = fp32 W, tile 3 = c -> tiles 0, 1 = [W0, W - W0]. W - W0 is left in fp32: the bf16
// pack converts it (its conversion error, on the <= 2^-(bits+1) |W| remainder, is below the target).
void w_split_grid() {
    vFloat c = dst_reg[3 * TILE_SLOTS];  // uniform over the tile (scalar broadcast)
#pragma GCC unroll 4
    for (int k = 0; k < TILE_SLOTS; ++k) {
        vFloat v = dst_reg[k];
        vFloat v0 = round_to_grid(v, c);
        dst_reg[k] = v0;
        dst_reg[TILE_SLOTS + k] = v - v0;
    }
}

// Lane-wise statistic over COUNT source tiles at DEST tiles 1.. into DEST tile 0: SQ -> sum v^2 (exact fp32
// SFPU MADs), else max|v|. FIRST initialises the accumulator.
template <int COUNT, bool FIRST, bool SQ>
void stats_accumulate() {
#pragma GCC unroll 4
    for (int k = 0; k < TILE_SLOTS; ++k) {
        vFloat acc = 0.0f;
        if constexpr (!FIRST) {
            acc = dst_reg[k];
        }
#pragma GCC unroll 8
        for (int j = 0; j < COUNT; ++j) {
            vFloat v = dst_reg[TILE_SLOTS * (1 + j) + k];
            if constexpr (SQ) {
                acc = acc + v * v;
            } else {
                vFloat a = sfpi::abs(v);
                v_if(a > acc) { acc = a; }
                v_endif;
            }
        }
        dst_reg[k] = acc;
    }
}

// Round-to-nearest-even to tf32 (10 explicit mantissa bits): the FPU then reads the value losslessly.
sfpi_inline vFloat rne_tf32(vFloat v) {
    vUInt u = as<vUInt>(v);
    vUInt lsb = (u << 18) >> 31;
    u = u + 0xFFF;
    u = u + lsb;
    u = (u >> 13) << 13;
    return as<vFloat>(u);
}

// coefficients_block on one coefficient-major tile: DEST tile 0 = raw sums, DEST tile 1 = bias.
void coefficients(
    uint32_t a_pre_bits,
    uint32_t a_post_bits,
    uint32_t a_res_bits,
    uint32_t eps_bits,
    uint32_t norm_eps_bits,
    uint32_t inv_nc_bits) {
    {
        vFloat ssq = dst_reg[MIX];
        vFloat inv_nc = Converter::as_float(inv_nc_bits);
        vFloat neps = Converter::as_float(norm_eps_bits);
        vFloat r = rsqrt_pos(ssq * inv_nc + neps);
        dst_reg[MIX] = r;
    }
#pragma GCC unroll 32
    for (int k = 0; k < MIX; ++k) {
        const uint32_t a_bits = k < N ? a_pre_bits : (k < 2 * N ? a_post_bits : a_res_bits);
        vFloat r = dst_reg[MIX];
        vFloat a = Converter::as_float(a_bits);
        vFloat z = dst_reg[k] * r;
        z = z * a + dst_reg[TILE_SLOTS + k];
        if (k < N) {
            vFloat s = sigmoid_acc(z);
            vFloat e = Converter::as_float(eps_bits);
            dst_reg[k] = rne_tf32(s + e);
        } else if (k < 2 * N) {
            vFloat s = sigmoid_acc(z);
            dst_reg[k] = s + s;
        } else {
            dst_reg[k] = z;
        }
    }
}

// ---- sinkhorn_block (Perf round 2, E2): bitwise identical to the plain formulation
// (m = softmax_j(L) + eps; col_norm; (iters - 1) x {row_norm, col_norm}; sums in index order, recip_pos with two
// Newton steps), restructured for the SFPU, which issues ~1 instruction per cycle (no visible dependency stalls),
// so its cost is the issued-instruction count:
//   * 2.0 (Newton) and eps sit in the programmable constant registers L12 / L13 (no SFPLOADI / injected runtime
//     immediates per use);
//   * softmax: j loops unrolled (fixed dst addresses), row max via SFPSWAP (sfpi::max), max / sum kept in LREGs;
//   * the (iters - 1) iterations are hand-scheduled SFPLOADMACRO passes (raw TTI, see below): each pass scales
//     the 16 entries and accumulates the other direction's sums (pass C: m *= rc_j + row sums; pass R: m *= rr_i +
//     column sums), with the load / multiply / store of every entry issued as ONE SFPLOADMACRO (its MAD and store
//     sub-units run the multiply and the write-back), so only the sums' adds are issued besides.
// Every entry sees exactly the same multiplications (SFPMUL = MAD with VC = L9 = 0, as the compiler emits) and
// every sum the same operand order as the plain formulation, with the Newton residual 2 - x*y as ONE fused SFPMAD,
// so the comb is bit-identical to the plain source compiled that way (bench: 4.97 -> 2.72 us per Sinkhorn, 20
// iterations). Note: sfpi (-ffast-math) compiled the plain recip_pos in this kernel as SFPMUL + SFPADDI (two
// roundings) or as SFPMAD depending on unrelated code (e.g. KERNEL_PERF_ZONES flips it), so the previous build's
// comb differed from this one by <= 2.4e-7 (<= 12 ulp); the fused form here is pinned (raw TTI / an L12 operand)
// and is as close or closer to fp64. Scratch: slots SK_SCR .. SK_SCR + 3 (> MIX, unused). n = 4 only.
//
// Raw-LLK deviation (no helper / sfpi equivalent): SFPLOADMACRO has no sfpi or kernel_lib wrapper; its scheduled
// MAD / store have no hardware interlocks, so the issue order below is a verified static schedule (generator +
// rule checker: perf_experiments/sinkhorn_sfpu_fast/bench/gen_sinkhorn_lm.py). Registers are pinned (the sfpi
// allocator does not see them): L0..L2 macro temps (SFPLOADMACRO VD must be L0..L3 for even DEST addresses),
// L3 the sum accumulator, L4..L7 the four multipliers, L9 = 0, L10 = 1, L12 = 2.0, L13 = eps.
constexpr int SK_SCR = MIX + 1;
static_assert(N == 4 && LOGIT0 == 8 && SK_SCR == 25, "the scheduled Sinkhorn passes are generated for n = 4");
constexpr int sk_m(int i, int j) { return LOGIT0 + i * N + j; }
constexpr uint32_t SK_SCRA = 2 * SK_SCR;  // DEST address of scratch slot SK_SCR (dst_reg[k] <-> address 2k)

// recip_pos with the Newton constant from L12 (same instruction sequence)
sfpi_inline vFloat sk_recip(vFloat x) {
    vFloat y = approx_recip(x);
    vFloat t = vConstFloatPrgm0 - x * y;
    y = y * t;
    t = vConstFloatPrgm0 - x * y;
    y = y * t;
    return y;
}

// SFPLOADMACRO config: macro q (q = 0..3): MAD sub-unit = InstructionTemplate[q] = SFPMUL(VA = L(4+q), VB <- the
// loaded VD, VC = L9 = 0), delay 0; store sub-unit = SFPSTORE of VD to the loaded address, 2 issued instructions
// later (UnitDelayKind = WaitForElapsedInstructions for MAD + store; the store uses the load's Mod0).
sfpi_inline void sk_lm_config() {
    TTI_SFPMUL(4, 0, 9, 12, 0);  // VD = 12 + q: backdoor write of InstructionTemplate[q]
    TTI_SFPMUL(5, 0, 9, 13, 0);
    TTI_SFPMUL(6, 0, 9, 14, 0);
    TTI_SFPMUL(7, 0, 9, 15, 0);
    constexpr uint32_t store_bits = (2 << 3) | 3;
#define MHC_SK_SEQ(q)                                                                     \
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, ((0x80 | (0 << 3) | (4 + (q))) << 8) | 0); \
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | 0);                    \
    TTI_SFPCONFIG(0, 4 + (q), 0);
    MHC_SK_SEQ(0)
    MHC_SK_SEQ(1)
    MHC_SK_SEQ(2)
    MHC_SK_SEQ(3)
#undef MHC_SK_SEQ
    TTI_SFPCONFIG(0xAF0, 8, 1);  // Misc: UsesLoadMod0ForStore = macros 0..3, UnitDelayKind = MAD + store
}

// L4..L7 <- 1 / (x_k + eps): recip_pos (approx_recip + two Newton steps), two chains interleaved (temps L0..L3).
// FROM_ACC: x_3 is the last group sum, still in L3 (the pass does not store it); x_0..x_2 (x_0..x_3 otherwise) come
// from scratch, >= 8 issued instructions after their stores: a Dst write cannot be read back for 4 cycles and
// SFPLOAD is not interlocked (BH ISA, Dst.md "Instruction scheduling").
#define MHC_SK_RECIP2(a, b, EPS_B)   \
    TTI_SFPADD(10, a, 13, a, 0);     \
    if (EPS_B) {                     \
        TTI_SFPADD(10, b, 13, b, 0); \
    }                                \
    TTI_SFPARECIP(0, a, 0, 0);       \
    TTI_SFPARECIP(0, b, 1, 0);       \
    TTI_SFPMAD(a, 0, 12, 2, 1);      \
    TTI_SFPMAD(b, 1, 12, 3, 1);      \
    TTI_SFPMUL(0, 2, 9, 0, 0);       \
    TTI_SFPMUL(1, 3, 9, 1, 0);       \
    TTI_SFPMAD(a, 0, 12, 2, 1);      \
    TTI_SFPMAD(b, 1, 12, 3, 1);      \
    TTI_SFPMUL(0, 2, 9, a, 0);       \
    TTI_SFPMUL(1, 3, 9, b, 0);
template <bool FROM_ACC>
sfpi_inline void sk_lm_recips() {
    if constexpr (FROM_ACC) {
        TTI_SFPADD(10, 3, 13, 7, 0);  // L7 = acc + eps, before L3 is reused as a temp
    }
    TTI_SFPLOAD(4, 0, 7, SK_SCRA + 0);
    TTI_SFPLOAD(5, 0, 7, SK_SCRA + 2);
    TTI_SFPLOAD(6, 0, 7, SK_SCRA + 4);
    if constexpr (!FROM_ACC) {
        TTI_SFPLOAD(7, 0, 7, SK_SCRA + 6);
    }
    MHC_SK_RECIP2(4, 5, true)
    MHC_SK_RECIP2(6, 7, !FROM_ACC)
}
#undef MHC_SK_RECIP2

// ---- generated by perf_experiments/sinkhorn_sfpu_fast/bench/gen_sinkhorn_lm.py (static schedule, rule-checked) ----
// pass C: m[i][j] *= rc_j (L4 + j); row sums (j order) -> scratch slot SCR + i
// 40 issue slots
sfpi_inline void sk_pass_c_sums() {
    TTI_SFPLOADMACRO((0 << 2) | 0, 0, 7, 16);  // m[0][0] *= L4
    TTI_SFPLOADMACRO((1 << 2) | 1, 0, 7, 18);  // m[0][1] *= L5
    TTI_SFPLOADMACRO((2 << 2) | 2, 0, 7, 20);  // m[0][2] *= L6
    TTI_SFPNOP;
    TTI_SFPADD(10, 0, 1, 3, 0);                // acc = e0 + e1
    TTI_SFPLOADMACRO((3 << 2) | 0, 0, 7, 22);  // m[0][3] *= L7
    TTI_SFPLOADMACRO((0 << 2) | 1, 0, 7, 24);  // m[1][0] *= L4
    TTI_SFPNOP;
    TTI_SFPADD(10, 3, 2, 3, 0);                // acc += e2
    TTI_SFPLOADMACRO((1 << 2) | 2, 0, 7, 26);  // m[1][1] *= L5
    TTI_SFPNOP;
    TTI_SFPADD(10, 3, 0, 3, 0);                // acc += e3
    TTI_SFPLOADMACRO((2 << 2) | 0, 0, 7, 28);  // m[1][2] *= L6
    TTI_SFPSTORE(3, 0, 7, 50);                 // group sum 0 -> scratch
    TTI_SFPADD(10, 1, 2, 3, 0);                // acc = e4 + e5
    TTI_SFPLOADMACRO((3 << 2) | 1, 0, 7, 30);  // m[1][3] *= L7
    TTI_SFPLOADMACRO((0 << 2) | 2, 0, 7, 32);  // m[2][0] *= L4
    TTI_SFPNOP;
    TTI_SFPADD(10, 3, 0, 3, 0);                // acc += e6
    TTI_SFPLOADMACRO((1 << 2) | 0, 0, 7, 34);  // m[2][1] *= L5
    TTI_SFPNOP;
    TTI_SFPADD(10, 3, 1, 3, 0);                // acc += e7
    TTI_SFPLOADMACRO((2 << 2) | 1, 0, 7, 36);  // m[2][2] *= L6
    TTI_SFPSTORE(3, 0, 7, 52);                 // group sum 1 -> scratch
    TTI_SFPADD(10, 2, 0, 3, 0);                // acc = e8 + e9
    TTI_SFPLOADMACRO((3 << 2) | 2, 0, 7, 38);  // m[2][3] *= L7
    TTI_SFPLOADMACRO((0 << 2) | 0, 0, 7, 40);  // m[3][0] *= L4
    TTI_SFPNOP;
    TTI_SFPADD(10, 3, 1, 3, 0);                // acc += e10
    TTI_SFPLOADMACRO((1 << 2) | 1, 0, 7, 42);  // m[3][1] *= L5
    TTI_SFPNOP;
    TTI_SFPADD(10, 3, 2, 3, 0);                // acc += e11
    TTI_SFPLOADMACRO((2 << 2) | 2, 0, 7, 44);  // m[3][2] *= L6
    TTI_SFPSTORE(3, 0, 7, 54);                 // group sum 2 -> scratch
    TTI_SFPADD(10, 0, 1, 3, 0);                // acc = e12 + e13
    TTI_SFPLOADMACRO((3 << 2) | 0, 0, 7, 46);  // m[3][3] *= L7
    TTI_SFPNOP;
    TTI_SFPADD(10, 3, 2, 3, 0);  // acc += e14
    TTI_SFPNOP;
    TTI_SFPADD(10, 3, 0, 3, 0);  // acc += e15
}

// pass R: m[i][j] *= rr_i (L4 + i); column sums (i order) -> scratch slot SCR + j
// 40 issue slots
sfpi_inline void sk_pass_r_sums() {
    TTI_SFPLOADMACRO((0 << 2) | 0, 0, 7, 16);  // m[0][0] *= L4
    TTI_SFPLOADMACRO((1 << 2) | 1, 0, 7, 24);  // m[1][0] *= L5
    TTI_SFPLOADMACRO((2 << 2) | 2, 0, 7, 32);  // m[2][0] *= L6
    TTI_SFPNOP;
    TTI_SFPADD(10, 0, 1, 3, 0);                // acc = e0 + e1
    TTI_SFPLOADMACRO((3 << 2) | 0, 0, 7, 40);  // m[3][0] *= L7
    TTI_SFPLOADMACRO((0 << 2) | 1, 0, 7, 18);  // m[0][1] *= L4
    TTI_SFPNOP;
    TTI_SFPADD(10, 3, 2, 3, 0);                // acc += e2
    TTI_SFPLOADMACRO((1 << 2) | 2, 0, 7, 26);  // m[1][1] *= L5
    TTI_SFPNOP;
    TTI_SFPADD(10, 3, 0, 3, 0);                // acc += e3
    TTI_SFPLOADMACRO((2 << 2) | 0, 0, 7, 34);  // m[2][1] *= L6
    TTI_SFPSTORE(3, 0, 7, 50);                 // group sum 0 -> scratch
    TTI_SFPADD(10, 1, 2, 3, 0);                // acc = e4 + e5
    TTI_SFPLOADMACRO((3 << 2) | 1, 0, 7, 42);  // m[3][1] *= L7
    TTI_SFPLOADMACRO((0 << 2) | 2, 0, 7, 20);  // m[0][2] *= L4
    TTI_SFPNOP;
    TTI_SFPADD(10, 3, 0, 3, 0);                // acc += e6
    TTI_SFPLOADMACRO((1 << 2) | 0, 0, 7, 28);  // m[1][2] *= L5
    TTI_SFPNOP;
    TTI_SFPADD(10, 3, 1, 3, 0);                // acc += e7
    TTI_SFPLOADMACRO((2 << 2) | 1, 0, 7, 36);  // m[2][2] *= L6
    TTI_SFPSTORE(3, 0, 7, 52);                 // group sum 1 -> scratch
    TTI_SFPADD(10, 2, 0, 3, 0);                // acc = e8 + e9
    TTI_SFPLOADMACRO((3 << 2) | 2, 0, 7, 44);  // m[3][2] *= L7
    TTI_SFPLOADMACRO((0 << 2) | 0, 0, 7, 22);  // m[0][3] *= L4
    TTI_SFPNOP;
    TTI_SFPADD(10, 3, 1, 3, 0);                // acc += e10
    TTI_SFPLOADMACRO((1 << 2) | 1, 0, 7, 30);  // m[1][3] *= L5
    TTI_SFPNOP;
    TTI_SFPADD(10, 3, 2, 3, 0);                // acc += e11
    TTI_SFPLOADMACRO((2 << 2) | 2, 0, 7, 38);  // m[2][3] *= L6
    TTI_SFPSTORE(3, 0, 7, 54);                 // group sum 2 -> scratch
    TTI_SFPADD(10, 0, 1, 3, 0);                // acc = e12 + e13
    TTI_SFPLOADMACRO((3 << 2) | 0, 0, 7, 46);  // m[3][3] *= L7
    TTI_SFPNOP;
    TTI_SFPADD(10, 3, 2, 3, 0);  // acc += e14
    TTI_SFPNOP;
    TTI_SFPADD(10, 3, 0, 3, 0);  // acc += e15
}

// last pass C: m[i][j] *= rc_j, no sums
// 24 issue slots
sfpi_inline void sk_pass_c_final() {
    TTI_SFPLOADMACRO((0 << 2) | 0, 0, 7, 16);  // m[0][0] *= L4
    TTI_SFPLOADMACRO((1 << 2) | 1, 0, 7, 18);  // m[0][1] *= L5
    TTI_SFPLOADMACRO((2 << 2) | 2, 0, 7, 20);  // m[0][2] *= L6
    TTI_SFPNOP;
    TTI_SFPLOADMACRO((3 << 2) | 0, 0, 7, 22);  // m[0][3] *= L7
    TTI_SFPLOADMACRO((0 << 2) | 1, 0, 7, 24);  // m[1][0] *= L4
    TTI_SFPLOADMACRO((1 << 2) | 2, 0, 7, 26);  // m[1][1] *= L5
    TTI_SFPNOP;
    TTI_SFPLOADMACRO((2 << 2) | 0, 0, 7, 28);  // m[1][2] *= L6
    TTI_SFPLOADMACRO((3 << 2) | 1, 0, 7, 30);  // m[1][3] *= L7
    TTI_SFPLOADMACRO((0 << 2) | 2, 0, 7, 32);  // m[2][0] *= L4
    TTI_SFPNOP;
    TTI_SFPLOADMACRO((1 << 2) | 0, 0, 7, 34);  // m[2][1] *= L5
    TTI_SFPLOADMACRO((2 << 2) | 1, 0, 7, 36);  // m[2][2] *= L6
    TTI_SFPLOADMACRO((3 << 2) | 2, 0, 7, 38);  // m[2][3] *= L7
    TTI_SFPNOP;
    TTI_SFPLOADMACRO((0 << 2) | 0, 0, 7, 40);  // m[3][0] *= L4
    TTI_SFPLOADMACRO((1 << 2) | 1, 0, 7, 42);  // m[3][1] *= L5
    TTI_SFPLOADMACRO((2 << 2) | 2, 0, 7, 44);  // m[3][2] *= L6
    TTI_SFPNOP;
    TTI_SFPLOADMACRO((3 << 2) | 0, 0, 7, 46);  // m[3][3] *= L7
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// sinkhorn_block on one coefficient-major tile (DEST tile 0): logits -> comb, all iterations in DEST.
void sinkhorn(uint32_t eps_bits, uint32_t iters) {
    vConstFloatPrgm0 = 2.0f;
    vConstFloatPrgm1 = Converter::as_float(eps_bits);
    // m = softmax_j(L) + eps, row max subtracted (overflow-safe).
#pragma GCC unroll 4
    for (int i = 0; i < N; ++i) {
        vFloat mx = dst_reg[sk_m(i, 0)];
#pragma GCC unroll 4
        for (int j = 1; j < N; ++j) {
            mx = sfpi::max(mx, vFloat(dst_reg[sk_m(i, j)]));
        }
        vFloat sum = 0.0f;
#pragma GCC unroll 4
        for (int j = 0; j < N; ++j) {
            vFloat d = dst_reg[sk_m(i, j)] - mx;
            vFloat ex = ckernel::sfpu::_sfpu_exp_fp32_accurate_<false>(d);
            dst_reg[sk_m(i, j)] = ex;
            sum = sum + ex;
        }
        vFloat rs = sk_recip(sum);
#pragma GCC unroll 4
        for (int j = 0; j < N; ++j) {
            dst_reg[sk_m(i, j)] = dst_reg[sk_m(i, j)] * rs + vConstFloatPrgm1;
        }
    }
    // column sums of the softmax (in row order) -> scratch
    {
        vFloat c0 = dst_reg[sk_m(0, 0)];
        vFloat c1 = dst_reg[sk_m(0, 1)];
        vFloat c2 = dst_reg[sk_m(0, 2)];
        vFloat c3 = dst_reg[sk_m(0, 3)];
#pragma GCC unroll 4
        for (int i = 1; i < N; ++i) {
            c0 = c0 + dst_reg[sk_m(i, 0)];
            c1 = c1 + dst_reg[sk_m(i, 1)];
            c2 = c2 + dst_reg[sk_m(i, 2)];
            c3 = c3 + dst_reg[sk_m(i, 3)];
        }
        dst_reg[SK_SCR + 0] = c0;
        dst_reg[SK_SCR + 1] = c1;
        dst_reg[SK_SCR + 2] = c2;
        dst_reg[SK_SCR + 3] = c3;
    }
    // col_norm, then (iters - 1) x {row_norm, col_norm}: raw from here on (pinned LREGs).
    sk_lm_config();
    sk_lm_recips<false>();  // rc_j of the first col_norm -> L4..L7
#pragma GCC unroll 0
    for (uint32_t it = 1; it < iters; ++it) {
        sk_pass_c_sums();      // m *= rc_j, row sums (3 -> scratch, the last in L3)
        sk_lm_recips<true>();  // rr_i -> L4..L7
        sk_pass_r_sums();      // m *= rr_i, column sums
        sk_lm_recips<true>();  // rc_j -> L4..L7
    }
    sk_pass_c_final();  // the closing col_norm scaling
    TTI_SFPNOP;         // the last macro stores land > 4 cycles before any following Dst read
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// ---- Layout transforms (Refinement 4): row-major S <-> coefficient-major, all in DEST ----
// A row-major tile transposed in DEST ("T layout": row = coefficient, col = token) holds coefficient row r of
// face F + h (F = (r >> 4) * 2, h = token >= 16) in the dst_reg vectors 8(F + h) + 2rb + o (rb = (r & 15) >> 2,
// o = token parity), subvector g = r & 3, lane c <-> token 16h + 2c + o. subvec_transp of the four vectors
// (h, o) of one row block therefore yields the four coefficients of that block as whole vectors, lane
// l <-> token 16(l >> 4) + 2(l & 7) + ((l >> 3) & 1) (mhc_layout::lane_token), and it is its own inverse.
// Every per-token op is lane-wise, so this fixed token permutation is invisible except at these transforms.
constexpr int T_MIX = static_cast<int>(mhc_t_mix);
constexpr int T_SQ = static_cast<int>(mhc_t_sq);

constexpr int t_vec(int tile, int row_block, int h, int o) {
    return tile * TILE_SLOTS + 8 * (((4 * row_block) >> 4) * 2 + h) + 2 * (((4 * row_block) & 15) >> 2) + o;
}

// DEST tile 0 slots [0, MIX) <- T_MIX rows [0, MIX); slot MIX <- T_SQ row 0.
void gather_coef_major() {
#pragma GCC unroll 8
    for (int b = 0; 4 * b < MIX; ++b) {
        vFloat u0 = dst_reg[t_vec(T_MIX, b, 0, 0)];
        vFloat u1 = dst_reg[t_vec(T_MIX, b, 0, 1)];
        vFloat u2 = dst_reg[t_vec(T_MIX, b, 1, 0)];
        vFloat u3 = dst_reg[t_vec(T_MIX, b, 1, 1)];
        subvec_transp(u0, u1, u2, u3);
        dst_reg[4 * b] = u0;
        if (4 * b + 1 < MIX) {
            dst_reg[4 * b + 1] = u1;
        }
        if (4 * b + 2 < MIX) {
            dst_reg[4 * b + 2] = u2;
        }
        if (4 * b + 3 < MIX) {
            dst_reg[4 * b + 3] = u3;
        }
    }
    vFloat u0 = dst_reg[t_vec(T_SQ, 0, 0, 0)];
    vFloat u1 = dst_reg[t_vec(T_SQ, 0, 0, 1)];
    vFloat u2 = dst_reg[t_vec(T_SQ, 0, 1, 0)];
    vFloat u3 = dst_reg[t_vec(T_SQ, 0, 1, 1)];
    subvec_transp(u0, u1, u2, u3);
    dst_reg[MIX] = u0;
}

sfpi_inline void zero_tile(int tile) {
#pragma GCC unroll 32
    for (int k = 0; k < TILE_SLOTS; ++k) {
        dst_reg[tile * TILE_SLOTS + k] = 0.0f;
    }
}

// Pre-column tile: coefficient-major slot I of tile 0 -> T-layout row 0 of tile DST (rest 0); transposed by
// the caller into column 0 = pre_I per token row (the y-mix's bcast-column operand). DST may be 0 (the slot
// is read before the tile is cleared).
template <int I, int DST>
void pre_tile() {
    vFloat v = dst_reg[I];
    zero_tile(DST);
    vFloat z1 = 0.0f, z2 = 0.0f, z3 = 0.0f;
    subvec_transp(v, z1, z2, z3);
    dst_reg[t_vec(DST, 0, 0, 0)] = v;
    dst_reg[t_vec(DST, 0, 0, 1)] = z1;
    dst_reg[t_vec(DST, 0, 1, 0)] = z2;
    dst_reg[t_vec(DST, 0, 1, 1)] = z3;
}

// All n pre-column tiles: pre_i -> DEST tile (i + 1) % 4 (tile 1 = the consumed bias, 2 / 3 = the consumed
// transposed S tiles, 0 = the coefficient tile itself, last).
void pre_tiles() {
    static_assert(N >= 1 && N <= 4, "n*(n+2)+1 <= 32 coefficient slots bounds n to 4");
    pre_tile<0, 1>();
    if constexpr (N > 1) {
        pre_tile<1, 2>();
    }
    if constexpr (N > 2) {
        pre_tile<2, 3>();
    }
    if constexpr (N > 3) {
        pre_tile<3, 0>();
    }
}
// Row-major output tile: coefficient-major slots [SRC0, SRC0 + COUNT) of tile 0 -> T-layout rows [0, COUNT)
// of tile DST (!= 0; rest 0); transposed by the caller into columns [0, COUNT) per token row.
template <int SRC0, int COUNT, int DST>
void rows_tile() {
    static_assert(DST != 0 && COUNT <= 16, "rows stay in faces 0/1 of a tile other than the source");
    zero_tile(DST);
#pragma GCC unroll 4
    for (int b = 0; 4 * b < COUNT; ++b) {
        vFloat u0 = dst_reg[SRC0 + 4 * b];
        vFloat u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;
        if (4 * b + 1 < COUNT) {
            u1 = dst_reg[SRC0 + 4 * b + 1];
        }
        if (4 * b + 2 < COUNT) {
            u2 = dst_reg[SRC0 + 4 * b + 2];
        }
        if (4 * b + 3 < COUNT) {
            u3 = dst_reg[SRC0 + 4 * b + 3];
        }
        subvec_transp(u0, u1, u2, u3);
        dst_reg[t_vec(DST, b, 0, 0)] = u0;
        dst_reg[t_vec(DST, b, 0, 1)] = u1;
        dst_reg[t_vec(DST, b, 1, 0)] = u2;
        dst_reg[t_vec(DST, b, 1, 1)] = u3;
    }
}

// Owned rows, after the Sinkhorn: post -> tile 2, comb -> tile 3 (T layout).
void post_comb_tiles() {
    rows_tile<N, N, 2>();
    rows_tile<2 * N, N * N, 3>();
}

}  // namespace mhc_sfpu
#endif

// One-time SFPU init for the custom block ops (config reg + addr mods + counters); no programmable
// constants are used by the custom ops.
ALWI void custom_sfpu_init() { MATH((ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>())); }

// combine_block (root): S[j] = sum_rank gathered[rank][j], fixed rank order, fp32 SFPU adds.
ALWI void combine_block() {
    constexpr uint32_t slot_tiles = 2 * block_token_tiles;
    cb_wait_front(cb_gathered, group_cores * slot_tiles);
    cb_reserve_back(cb_combined, slot_tiles);
    reconfig_data_format_srca(cb_gathered);
    pack_reconfig_data_format(cb_combined);
    copy_tile_to_dst_init_short(cb_gathered);
    add_binary_tile_init();
    for (uint32_t j = 0; j < slot_tiles; ++j) {
        tile_regs_acquire();
        copy_tile(cb_gathered, j, 0);
        for (uint32_t r = 1; r < group_cores; ++r) {
            copy_tile(cb_gathered, r * slot_tiles + j, 1);
            add_binary_tile(0, 1, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_combined);
        tile_regs_release();
    }
    cb_push_back(cb_combined, slot_tiles);
    cb_pop_front(cb_gathered, group_cores * slot_tiles);
}

// w_split_block (once, before block 0): cb_weight fp32 tile k (bytes [4096k, 4096k+4096)) -> cb_weight_split
// bf16 pages 2k (W_hi) and 2k+1 (W_lo), the SAME bytes. In place is safe: tile k is fully unpacked into DEST
// before its pair is packed, and later unpacks only touch tiles > k.
// Splits W tiles [p0, p1) in place. CHUNK_WAITS: the tiles arrive in w_chunk_tiles chunks (cumulative waits,
// R1 DRAM fill); otherwise the caller guarantees they are in L1 (W_ROLE_SPREAD own share, token-guarded).
template <bool CHUNK_WAITS>
ALWI void w_split_range(uint32_t p0, uint32_t p1, uint32_t core_k_tiles) {
    constexpr uint32_t pair_limit = compute_kernel_lib::DEST_AUTO_LIMIT / 2;  // one DEST pair per W tile
    constexpr uint32_t tiles_per_window = pair_limit < w_chunk_tiles ? pair_limit : w_chunk_tiles;
    static_assert(w_chunk_tiles % tiles_per_window == 0, "a DEST window must not straddle a W chunk");
    reconfig_data_format_srca(cb_weight);
    pack_reconfig_data_format(cb_weight_split);
    copy_tile_to_dst_init_short(cb_weight);
    custom_sfpu_init();
    for (uint32_t k0 = p0; k0 < p1; k0 += tiles_per_window) {
        const uint32_t nt = (p1 - k0) < tiles_per_window ? (p1 - k0) : tiles_per_window;
        if constexpr (CHUNK_WAITS) {
            if (k0 % w_chunk_tiles == 0) {
                // W arrives in chunks; waits are cumulative (cb_weight is never popped).
                const uint32_t upto = k0 + w_chunk_tiles;
                cb_wait_front(cb_weight, upto < core_k_tiles ? upto : core_k_tiles);
            }
        }
        tile_regs_acquire();
        for (uint32_t j = 0; j < nt; ++j) {
            copy_tile(cb_weight, k0 + j, 2 * j);
            MATH((_llk_math_eltwise_unary_sfpu_params_(mhc_sfpu::w_split_hi_lo, 2 * j, VectorMode::None)));
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < 2 * nt; ++j) {
            pack_tile<true>(j, cb_weight_split, 2 * k0 + j);
        }
        tile_regs_release();
    }
}

ALWI void w_split_block(uint32_t core_k_tiles) {
    cb_reserve_back(cb_weight_split, 2 * core_k_tiles);
    w_split_range<true>(0, core_k_tiles, core_k_tiles);
    cb_push_back(cb_weight_split, 2 * core_k_tiles);
    cb_wait_front(cb_weight_split, 2 * core_k_tiles);  // resident for the whole kernel, never popped
}

// W_ROLE_SPREAD (column all-gather): split only this core's share [p0, p1) — the writer multicasts it split,
// and the other rows' shares land already split. The copy_tile reads of the own share run before cb_weight is
// published: the writer's cb_w_own_ready token guarantees they landed. The whole slice is waited for only
// right before the first projection (w_publish_split), so the W-free sumsq of block 0 runs under the exchange.
ALWI void w_split_own_share(uint32_t p0, uint32_t p1, uint32_t core_k_tiles) {
    cb_reserve_back(cb_weight_split, 2 * core_k_tiles);
    cb_wait_front(cb_w_own_ready, 1);
    w_split_range<false>(p0, p1, core_k_tiles);
    cb_pop_front(cb_w_own_ready, 1);
    cb_reserve_back(cb_w_own_split, 1);  // packs above are complete when the push lands (pack thread order)
    cb_push_back(cb_w_own_split, 1);
}
ALWI void w_publish_split(uint32_t core_k_tiles) {
    cb_wait_front(cb_weight, core_k_tiles);
    cb_push_back(cb_weight_split, 2 * core_k_tiles);
    cb_wait_front(cb_weight_split, 2 * core_k_tiles);  // resident for the whole kernel, never popped
}

// A block's view into cb_x_resident (and its fp32 alias): `wait0` pages sit at the front ahead of it (the
// current block, when the next one is projected), `off` is added to every tile index -- modular: the next block
// may sit at the ring's start, BEHIND the read pointer (the unpacker's uint32 address arithmetic wraps).
struct XView {
    uint32_t wait0;
    uint32_t off;
};

// project_block_pieces: mix[t] = sum_k sum_p X[t][k] @ Wp[k] (Wp page k*PIECES + p), every product of an
// output sub-block accumulated in one DEST window, then packed once (fp32) to cb_partial. Piece 0 (W_hi, or
// the bf16 W itself) runs at MATH_FIDELITY; pieces >= 1 (W_lo) at w_lo_fidelity (math re-init only; same DEST).
template <uint32_t PIECES>
ALWI void project_block_pieces(uint32_t extent, uint32_t core_k_tiles, uint32_t sb_h, XView v = {0, 0}) {
    // The reader publishes the block in K chunks: wait per K tile (cumulative, never popped here — the block is
    // retained for sumsq and the y-mix), so the projection runs under the rest of the X burst.
    uint32_t waited = 0;
    reconfig_data_format(cb_w_matmul, cb_x_resident);  // matmul: srca = in1, srcb = in0
    pack_reconfig_data_format(cb_partial);
    matmul_init(cb_x_resident, cb_w_matmul);
    for (uint32_t r0 = 0; r0 < extent; r0 += sb_h) {
        constexpr bool lo_reinit = PIECES > 1 && w_lo_fidelity != w_main_fidelity;
        tile_regs_acquire();
        if constexpr (lo_reinit) {
            if (r0 > 0) {
                MATH((llk_math_matmul_init<w_main_fidelity, MM_THROTTLE>(cb_x_resident, cb_w_matmul)));
            }
        }
        for (uint32_t k = 0; k < core_k_tiles; ++k) {
            const uint32_t need = v.wait0 + (r0 + sb_h - 1) * core_k_tiles + k + 1;
            if (need > waited) {
                cb_wait_front(cb_x_resident, need);
                waited = need;
            }
            for (uint32_t r = 0; r < sb_h; ++r) {
                const uint32_t x_idx = v.off + (r0 + r) * core_k_tiles + k;
                matmul_tiles(cb_x_resident, cb_w_matmul, x_idx, k * PIECES, r);
                if constexpr (!lo_reinit) {
                    for (uint32_t p = 1; p < PIECES; ++p) {
                        matmul_tiles(cb_x_resident, cb_w_matmul, x_idx, k * PIECES + p, r);
                    }
                }
            }
        }
        if constexpr (lo_reinit) {
            MATH((llk_math_matmul_init<w_lo_fidelity, MM_THROTTLE>(cb_x_resident, cb_w_matmul)));
            for (uint32_t k = 0; k < core_k_tiles; ++k) {
                for (uint32_t r = 0; r < sb_h; ++r) {
                    const uint32_t x_idx = v.off + (r0 + r) * core_k_tiles + k;
                    for (uint32_t p = 1; p < PIECES; ++p) {
                        UNPACK((llk_unpack_AB_matmul(cb_x_resident, cb_w_matmul, x_idx, k * PIECES + p)));
                        MATH((llk_math_matmul<w_lo_fidelity, MM_THROTTLE>(r)));
                    }
                }
            }
        }
        tile_regs_commit();
        cb_reserve_back(cb_partial, sb_h);
        tile_regs_wait();
        for (uint32_t r = 0; r < sb_h; ++r) {
            pack_tile(r, cb_partial);
        }
        tile_regs_release();
        cb_push_back(cb_partial, sb_h);
    }
    cb_wait_front(cb_x_resident, v.wait0 + extent * core_k_tiles);  // whole block landed (no-op after the last)
}

#ifdef TRISC_MATH
template <int C, bool SQ>
ALWI void stats_run(bool first) {
    if (first) {
        _llk_math_eltwise_unary_sfpu_params_(mhc_sfpu::stats_accumulate<C, true, SQ>, 0, VectorMode::None);
    } else {
        _llk_math_eltwise_unary_sfpu_params_(mhc_sfpu::stats_accumulate<C, false, SQ>, 0, VectorMode::None);
    }
}
// runtime tile count (ragged last round) -> compile-time SFPU body
template <uint32_t MAX_COUNT, bool SQ, int C = 1>
ALWI void stats_dispatch(uint32_t count, bool first) {
    if constexpr (C < static_cast<int>(MAX_COUNT)) {
        if (count != static_cast<uint32_t>(C)) {
            stats_dispatch<MAX_COUNT, SQ, C + 1>(count, first);
            return;
        }
    }
    stats_run<C, SQ>(first);
}
#endif

// stats_pass: one DEST window over `count` fp32 tiles of `cb_src` (UnpackToDestFp32) starting at page `base`,
// lane-wise into DEST tile 0. SQ: sum v^2 (exact fp32) -> cb_sq_acc AND cb_max_lanes (the grid bound is
// sqrt(max of it): no per-element compare). Else: max|v| -> cb_max_lanes.
template <uint32_t cb_src, bool SQ>
ALWI void stats_pass(uint32_t base, uint32_t count) {
    constexpr uint32_t per_round = compute_kernel_lib::DEST_AUTO_LIMIT - 1;
    static_assert(per_round >= 1 && per_round <= 7, "stats dispatch covers 1..7 tiles per round");
    if constexpr (SQ) {
        cb_reserve_back(cb_sq_acc, 1);
    }
    cb_reserve_back(cb_max_lanes, 1);
    reconfig_data_format_srca(cb_src);
    pack_reconfig_data_format(cb_max_lanes);  // cb_sq_acc: same (fp32) format
    copy_tile_to_dst_init_short(cb_src);
    custom_sfpu_init();
    tile_regs_acquire();
    for (uint32_t k0 = 0; k0 < count; k0 += per_round) {
        const uint32_t nt = (count - k0) < per_round ? (count - k0) : per_round;
        for (uint32_t j = 0; j < nt; ++j) {
            copy_tile(cb_src, base + k0 + j, 1 + j);
        }
        MATH((stats_dispatch<per_round, SQ>(nt, k0 == 0)));
    }
    tile_regs_commit();
    tile_regs_wait();
    if constexpr (SQ) {
        pack_tile(0, cb_sq_acc);
    }
    pack_tile(0, cb_max_lanes);
    tile_regs_release();
    if constexpr (SQ) {
        cb_push_back(cb_sq_acc, 1);
    }
    cb_push_back(cb_max_lanes, 1);
}

// grid_block: cb_max_lanes -> reduce<MAX, REDUCE_SCALAR> (m = max|v| over the tile, at element (0, 0)) ->
// scalar-broadcast into DEST -> SFPU rounding constant for a BITS-bit grid -> one cb_grid page.
// Raw LLK: unary_bcast<SCALAR> (no chain element broadcasts a single CB tile into DEST).
template <int BITS, bool SQ_BOUND>
ALWI void grid_block() {
    compute_kernel_lib::reduce<
        PoolType::MAX,
        ReduceDim::REDUCE_SCALAR,
        cb_max_lanes,
        cb_max_scaler,
        cb_max_scalar,
        compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
        compute_kernel_lib::ReduceInputBlockShape::single());
    cb_wait_front(cb_max_scalar, 1);
    cb_reserve_back(cb_grid, 1);
    reconfig_data_format_srca(cb_max_scalar);
    pack_reconfig_data_format(cb_grid);
    unary_bcast_init<BroadcastType::SCALAR>(cb_max_scalar);
    custom_sfpu_init();
    tile_regs_acquire();
    unary_bcast<BroadcastType::SCALAR>(cb_max_scalar, 0, 0);
    MATH((_llk_math_eltwise_unary_sfpu_params_(mhc_sfpu::grid_constant<BITS, SQ_BOUND>, 0, VectorMode::None)));
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_grid);
    tile_regs_release();
    unary_bcast_uninit<BroadcastType::SCALAR>(cb_max_scalar);
    cb_push_back(cb_grid, 1);
    cb_pop_front(cb_max_scalar, 1);
}

// w_grid_split_block (once, fp32 X + fp32 W): global max|W| of the resident slice -> W grid, then each fp32
// W tile k is rewritten in place (aliased cb_weight_split, pages 2k / 2k+1) into [W0(k), rne(W - W0)(k)].
// One grid for the whole slice: a per-W-chunk grid (which would pipeline with the chunked W read) measured
// slower (640x7168 fp32: 613 vs 585 us) — the extra reduce / broadcast / init phases per chunk cost more
// than the W read they would hide.
ALWI void w_grid_split_block(uint32_t core_k_tiles) {
    cb_wait_front(cb_weight, core_k_tiles);
    stats_pass<cb_weight, false>(0, core_k_tiles);
    grid_block<static_cast<int>(w_grid_bits), false>();
    cb_wait_front(cb_grid, 1);
    cb_reserve_back(cb_weight_split, 2 * core_k_tiles);
    reconfig_data_format_srca(cb_weight);
    pack_reconfig_data_format(cb_weight_split);
    copy_tile_to_dst_init_short(cb_weight);  // cb_grid: same fp32 UnpackToDestFp32 format
    custom_sfpu_init();
    for (uint32_t k = 0; k < core_k_tiles; ++k) {
        tile_regs_acquire();
        copy_tile(cb_weight, k, 0);
        copy_tile(cb_grid, 0, 3);
        MATH((_llk_math_eltwise_unary_sfpu_params_(mhc_sfpu::w_split_grid, 0, VectorMode::None)));
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb_weight_split, 2 * k);
        pack_tile<true>(1, cb_weight_split, 2 * k + 1);
        tile_regs_release();
    }
    cb_push_back(cb_weight_split, 2 * core_k_tiles);
    cb_wait_front(cb_weight_split, 2 * core_k_tiles);  // resident for the whole kernel, never popped
    cb_pop_front(cb_grid, 1);
}

// x_stats_block (fp32 X, per block, before the projection): per token row-tile t, ONE pass over its K slice
// gives the exact lane-wise sum x^2 (-> cb_sq_acc, reduced after the projection) and max|x| (-> the x grid
// constant, cb_grid page t).
ALWI void x_stats_block(uint32_t extent, uint32_t core_k_tiles, uint32_t xoff) {
    for (uint32_t t = 0; t < extent; ++t) {
        stats_pass<cb_x_fp32, true>(xoff + t * core_k_tiles, core_k_tiles);
        grid_block<static_cast<int>(x_grid_bits), true>();
    }
    cb_wait_front(cb_grid, extent);
}

// x_split_window: the bf16 pieces of X rows [r0, r0 + rows) x K tiles [k0, k0 + kc) -> one cb_x_pieces
// window, page (r * kc + kk) * x_pieces + q. Recomputed per K chunk from the resident fp32 block (read via
// the UnpackToDestFp32 alias), so the pieces never exist as a second resident copy of the block.
ALWI void x_split_window(uint32_t r0, uint32_t rows, uint32_t k0, uint32_t kc, uint32_t core_k_tiles, uint32_t xoff) {
    static_assert(compute_kernel_lib::DEST_AUTO_LIMIT >= 4, "x split uses DEST tiles 0..3");
    cb_reserve_back(cb_x_pieces, x_window_pages);
    reconfig_data_format_srca(cb_x_fp32);
    pack_reconfig_data_format(cb_x_pieces);
    copy_tile_to_dst_init_short(cb_x_fp32);  // cb_grid: same fp32 UnpackToDestFp32 format
    custom_sfpu_init();
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t kk = 0; kk < kc; ++kk) {
            tile_regs_acquire();
            copy_tile(cb_x_fp32, xoff + (r0 + r) * core_k_tiles + k0 + kk, 0);
            copy_tile(cb_grid, r0 + r, 3);
            MATH((_llk_math_eltwise_unary_sfpu_params_(mhc_sfpu::x_split_grid, 0, VectorMode::None)));
            tile_regs_commit();
            tile_regs_wait();
            const uint32_t page0 = (r * kc + kk) * x_pieces;
            for (uint32_t q = 0; q < x_pieces; ++q) {
                pack_tile<true>(q, cb_x_pieces, page0 + q);
            }
            tile_regs_release();
        }
    }
    cb_push_back(cb_x_pieces, x_window_pages);
}

// project_block_split (fp32 X): mix[t] = sum_k sum_{q + p <= product_order_max} x_q[t][k] @ W_p[k], streamed
// over K in x_chunk_k_tiles windows. Per (sub-block, chunk): x_split_window, then ONE DEST window that
// reloads the fp32 running partial exactly (UnpackToDestFp32 copy; an FPU reload would truncate it),
// accumulates the chunk's piece products on top (all at MATH_FIDELITY: the remainder pieces carry up to 8
// bits), and packs the new running partial (cb_mix_run) or, on the last chunk, the mix partial (cb_partial).
ALWI void project_block_split(uint32_t extent, uint32_t core_k_tiles, uint32_t sb_h, uint32_t xoff) {
    static_assert(!x_grid_split || x_pieces == 3, "the fp32-X grid split is [x0, x1_hi, x1_mid]");
    constexpr uint32_t sb_max = block_token_tiles < compute_kernel_lib::DEST_AUTO_LIMIT
                                    ? block_token_tiles
                                    : compute_kernel_lib::DEST_AUTO_LIMIT;
    static_assert(
        !x_grid_split || sb_max <= x_piece_rows, "cb_x_pieces / cb_mix_run must hold a full projection sub-block");
    for (uint32_t r0 = 0; r0 < extent; r0 += sb_h) {
        for (uint32_t k0 = 0; k0 < core_k_tiles; k0 += x_chunk_k_tiles) {
            const uint32_t kc = (core_k_tiles - k0) < x_chunk_k_tiles ? (core_k_tiles - k0) : x_chunk_k_tiles;
            const bool first = k0 == 0;
            const bool last = k0 + kc >= core_k_tiles;
            x_split_window(r0, sb_h, k0, kc, core_k_tiles, xoff);

            cb_wait_front(cb_x_pieces, x_window_pages);
            tile_regs_acquire();
            if (!first) {
                cb_wait_front(cb_mix_run, x_piece_rows);
                reconfig_data_format_srca(cb_mix_run);
                copy_tile_to_dst_init_short(cb_mix_run);
                for (uint32_t r = 0; r < sb_h; ++r) {
                    copy_tile(cb_mix_run, r, r);
                }
            }
            reconfig_data_format(cb_w_matmul, cb_x_pieces);  // matmul: srca = in1, srcb = in0
            matmul_init(cb_x_pieces, cb_w_matmul);
            // products of order q + p < product_lo_order at MATH_FIDELITY; the (much smaller) higher-order ones
            // at x_lo_fidelity (math re-init only, same DEST window)
            for (uint32_t kk = 0; kk < kc; ++kk) {
                for (uint32_t r = 0; r < sb_h; ++r) {
                    for (uint32_t q = 0; q < x_pieces; ++q) {
                        const uint32_t x_idx = (r * kc + kk) * x_pieces + q;
                        for (uint32_t p = 0; p < w_pieces && q + p < product_lo_order; ++p) {
                            matmul_tiles(cb_x_pieces, cb_w_matmul, x_idx, (k0 + kk) * w_pieces + p, r);
                        }
                    }
                }
            }
            if constexpr (product_lo_order <= product_order_max) {
                MATH((llk_math_matmul_init<x_lo_fidelity, MM_THROTTLE>(cb_x_pieces, cb_w_matmul)));
                for (uint32_t kk = 0; kk < kc; ++kk) {
                    for (uint32_t r = 0; r < sb_h; ++r) {
                        for (uint32_t q = 0; q < x_pieces; ++q) {
                            const uint32_t x_idx = (r * kc + kk) * x_pieces + q;
                            for (uint32_t p = 0; p < w_pieces && q + p <= product_order_max; ++p) {
                                if (q + p < product_lo_order) {
                                    continue;
                                }
                                UNPACK(
                                    (llk_unpack_AB_matmul(cb_x_pieces, cb_w_matmul, x_idx, (k0 + kk) * w_pieces + p)));
                                MATH((llk_math_matmul<x_lo_fidelity, MM_THROTTLE>(r)));
                            }
                        }
                    }
                }
            }
            tile_regs_commit();
            cb_pop_front(cb_x_pieces, x_window_pages);
            if (!first) {
                cb_pop_front(cb_mix_run, x_piece_rows);
            }
            const uint32_t cb_dst = last ? cb_partial : cb_mix_run;
            const uint32_t dst_pages = last ? sb_h : x_piece_rows;  // cb_mix_run: nominal (never wraps mid-window)
            cb_reserve_back(cb_dst, dst_pages);
            pack_reconfig_data_format(cb_dst);
            tile_regs_wait();
            for (uint32_t r = 0; r < sb_h; ++r) {
                pack_tile<true>(r, cb_dst, r);
            }
            tile_regs_release();
            cb_push_back(cb_dst, dst_pages);
        }
    }
}

namespace mhc_sfpu_tiles {
constexpr uint32_t pre_tile_index(uint32_t i) { return (i + 1) % 4; }  // mirrors mhc_sfpu::pre_tiles
}  // namespace mhc_sfpu_tiles

// Row t of the landed S block (cb_coef_in: [mix x block_token_tiles | sum(x^2) x block_token_tiles]) -> DEST
// tile 0 = coefficient-major raw sums, tile 1 = bias, then the coefficient op: tile 0 = r / pre / post /
// logits. Caller holds the DEST window.
struct CoefScalars {
    uint32_t a_pre, a_post, a_res, eps, norm_eps, inv_nc;
};
ALWI void load_coef_major(uint32_t t, const CoefScalars& c) {
    reconfig_data_format_srca(cb_coef_in);  // the previous phase may have left bf16 X formats (y-mix)
    transpose_init(cb_coef_in);
    transpose_tile(cb_coef_in, t, mhc_t_mix);
    transpose_tile(cb_coef_in, block_token_tiles + t, mhc_t_sq);
    copy_tile_to_dst_init_short(cb_bias_coef);
    copy_tile(cb_bias_coef, 0, 1);
    custom_sfpu_init();
#ifndef MHC_ABLATE_COEF
    MATH((_llk_math_eltwise_unary_sfpu_params_(mhc_sfpu::gather_coef_major, 0, VectorMode::None)));
    MATH((_llk_math_eltwise_unary_sfpu_params_(
        mhc_sfpu::coefficients, 0, VectorMode::None, c.a_pre, c.a_post, c.a_res, c.eps, c.norm_eps, c.inv_nc)));
#endif
}

// sumsq_row: Q = sum_k x_k * x_k over one token row of the resident X block (DEST-accumulated) -> cb_sq_acc.
template <compute_kernel_lib::WaitPolicy WAIT, compute_kernel_lib::TileAddressing ADDR>
ALWI void sumsq_row(uint32_t core_k_tiles, uint32_t base) {
    using namespace compute_kernel_lib;
    eltwise_chain(
        IterationShape::tiles(core_k_tiles),
        BinaryFpu<
            BinaryFpuOp::Mul,
            input(cb_x_resident, WAIT, PopPolicy::None, InputTileMapping::Block, DataFormatReconfig::Enabled, ADDR),
            input(cb_x_resident, WAIT, PopPolicy::None, InputTileMapping::Block, DataFormatReconfig::Enabled, ADDR),
            Dst::D0,
            DestAccumulation::WholeShape>{base, base},
        PackTile<output(
            cb_sq_acc,
            ReservePolicy::OneUpfront,
            PushPolicy::OneAtEnd,
            DataFormatReconfig::Enabled,
            TileAddressing::Direct,
            DestAccumulation::WholeShape)>{});
}

// project_sumsq_streamed (bf16 X, fp32 W pieces): per token row, per K chunk [k0, k0 + kc) (waited cumulatively,
// so both run under the X burst instead of after it):
//   projection window: exact fp32 reload of the running mix (cb_mix_run, UnpackToDestFp32; not on the first
//     chunk), the chunk's X @ W_p products on top (as project_block_pieces), pack -> cb_mix_run, or on the
//     row's last chunk the mix partial -> cb_partial [mix rows];
//   sum x^2 window: Q_chunk = sum_k x_k*x_k (DEST-accumulated) -> one fp32 partial tile in cb_sq_acc.
// Then (all mix rows pushed first: cb_partial is [mix rows | sumsq rows]) per row: reduce<SUM, REDUCE_ROW> over
// its x_stream_chunks partial tiles -> cb_partial [sumsq rows].
template <uint32_t PIECES>
ALWI void project_sumsq_streamed(uint32_t extent, uint32_t core_k_tiles, XView v) {
    using namespace compute_kernel_lib;
    constexpr bool lo_reinit = PIECES > 1 && w_lo_fidelity != w_main_fidelity;
    const uint32_t kc_len = (core_k_tiles + x_stream_chunks - 1) / x_stream_chunks;
    uint32_t num_chunks = 0;
    for (uint32_t t = 0; t < extent; ++t) {
        const uint32_t base = v.off + t * core_k_tiles;
        const uint32_t wbase = v.wait0 + t * core_k_tiles;
        num_chunks = 0;
        for (uint32_t k0 = 0; k0 < core_k_tiles; k0 += kc_len, ++num_chunks) {
            const uint32_t kc = (core_k_tiles - k0) < kc_len ? (core_k_tiles - k0) : kc_len;
            const bool first = k0 == 0;
            const bool last = k0 + kc >= core_k_tiles;
            {
                MaybeDeviceZoneScope("c_x_wait");
                cb_wait_front(cb_x_resident, wbase + k0 + kc);
            }
            {
                MaybeDeviceZoneScope("c_proj");
                tile_regs_acquire();
                if (!first) {
                    cb_wait_front(cb_mix_run, 1);
                    reconfig_data_format_srca(cb_mix_run);
                    copy_tile_to_dst_init_short(cb_mix_run);
                    copy_tile(cb_mix_run, 0, 0);
                }
                reconfig_data_format(cb_w_matmul, cb_x_resident);  // matmul: srca = in1, srcb = in0
                matmul_init(cb_x_resident, cb_w_matmul);
#ifndef MHC_ABLATE_PROJ
                for (uint32_t k = k0; k < k0 + kc; ++k) {
                    matmul_tiles(cb_x_resident, cb_w_matmul, base + k, k * PIECES, 0);
                    if constexpr (!lo_reinit) {
                        for (uint32_t p = 1; p < PIECES; ++p) {
                            matmul_tiles(cb_x_resident, cb_w_matmul, base + k, k * PIECES + p, 0);
                        }
                    }
                }
                if constexpr (lo_reinit) {
                    MATH((llk_math_matmul_init<w_lo_fidelity, MM_THROTTLE>(cb_x_resident, cb_w_matmul)));
                    for (uint32_t k = k0; k < k0 + kc; ++k) {
                        for (uint32_t p = 1; p < PIECES; ++p) {
                            UNPACK((llk_unpack_AB_matmul(cb_x_resident, cb_w_matmul, base + k, k * PIECES + p)));
                            MATH((llk_math_matmul<w_lo_fidelity, MM_THROTTLE>(0)));
                        }
                    }
                }
#endif
                tile_regs_commit();
                if (!first) {
                    cb_pop_front(cb_mix_run, 1);
                }
                const uint32_t cb_dst = last ? cb_partial : cb_mix_run;
                cb_reserve_back(cb_dst, 1);
                pack_reconfig_data_format(cb_dst);
                tile_regs_wait();
                pack_tile(0, cb_dst);
                tile_regs_release();
                cb_push_back(cb_dst, 1);
            }
            {
                MaybeDeviceZoneScope("c_sumsq");
#ifndef MHC_ABLATE_PROJ
                sumsq_row<WaitPolicy::None, TileAddressing::Offset>(kc, base + k0);
#else
                cb_reserve_back(cb_sq_acc, 1);
                pack_reconfig_data_format(cb_sq_acc);
                tile_regs_acquire();
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_sq_acc);
                tile_regs_release();
                cb_push_back(cb_sq_acc, 1);
#endif
            }
        }
    }
    MaybeDeviceZoneScope("c_sq_reduce");
    for (uint32_t t = 0; t < extent; ++t) {
        reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            cb_sq_acc,
            cb_reduce_scaler,
            cb_partial,
            ReduceInputPolicy::WaitAndPopPerTile,
            ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
            ReduceFp32Mode::Accurate>(ReduceInputBlockShape::row(num_chunks));
    }
}

void kernel_main() {
    const uint32_t num_blocks = get_arg_val<uint32_t>(0);
    const uint32_t core_token_tiles = get_arg_val<uint32_t>(1);
    const uint32_t core_c_tiles = get_arg_val<uint32_t>(2);
    const uint32_t rank = get_arg_val<uint32_t>(3);
    const uint32_t a_pre_bits = get_arg_val<uint32_t>(4);
    const uint32_t a_post_bits = get_arg_val<uint32_t>(5);
    const uint32_t a_res_bits = get_arg_val<uint32_t>(6);
    const uint32_t eps_bits = get_arg_val<uint32_t>(7);
    const uint32_t norm_eps_bits = get_arg_val<uint32_t>(8);
    const uint32_t inv_nc_bits = get_arg_val<uint32_t>(9);
    const uint32_t sinkhorn_iters = get_arg_val<uint32_t>(10);
    const CoefScalars coef_scalars{a_pre_bits, a_post_bits, a_res_bits, eps_bits, norm_eps_bits, inv_nc_bits};

    using namespace compute_kernel_lib;

    const uint32_t core_k_tiles = n_streams * core_c_tiles;
    constexpr uint32_t x_block_pages = block_token_tiles * core_k_tiles_max;  // nominal (matches reader)

    CircularBuffer x_buf(cb_x_resident);
    CircularBuffer w_buf(cb_weight);
    CircularBuffer partial_buf(cb_partial);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x_resident, cb_w_matmul, cb_partial);

    if constexpr (x_grid_split && w_pieces > 1) {
        w_grid_split_block(core_k_tiles);
    } else if constexpr (w_presplit) {
        MaybeDeviceZoneScope("c_w_split");
        w_split_own_share(get_arg_val<uint32_t>(11), get_arg_val<uint32_t>(12), core_k_tiles);
    } else if constexpr (w_pieces > 1) {
        w_split_block(core_k_tiles);
    }

    auto extent_of = [&](uint32_t block_idx) {
        const uint32_t row0 = block_idx * block_token_tiles;
        return (core_token_tiles - row0) < block_token_tiles ? (core_token_tiles - row0) : block_token_tiles;
    };

    // ---- proj phase: everything of block `block_idx` that needs only its own X (projection + sum x^2 ->
    // cb_partial [mix rows | sumsq rows]); v = its view into cb_x_resident ----
    auto proj_phase = [&](uint32_t block_idx, XView v) {
        const uint32_t extent = extent_of(block_idx);
        // projection out sub-block height = largest divisor of the extent <= DEST
        uint32_t sb_h = 1;
        for (uint32_t h = DEST_AUTO_LIMIT; h > 1; --h) {
            if (extent % h == 0) {
                sb_h = h;
                break;
            }
        }
        if constexpr (x_grid_split) {
            if constexpr (w_pieces == 1) {
                // bf16 W feeds the split projection's matmul unsplit: nothing else waits for it (the fp32-W grid
                // split does, before block 0). Cumulative wait, never popped (resident); a no-op after block 0.
                cb_wait_front(cb_weight, core_k_tiles);
            }
            // cb_x_fp32 tracks cb_x_resident page for page (compute is its producer and consumer; the
            // reader's data is guaranteed by the cb_x_resident wait).
            cb_reserve_back(cb_x_fp32, x_block_pages);
            cb_push_back(cb_x_fp32, x_block_pages);
            cb_wait_front(cb_x_resident, v.wait0 + extent * core_k_tiles);
            cb_wait_front(cb_x_fp32, v.wait0 + x_block_pages);
            x_stats_block(extent, core_k_tiles, v.off);
        }
        {
            // ---- project_block (first: it waits per K tile, so it streams under the X burst; the longest
            // per-block compute phase must not start only after the last X tile): mix partial = X_blk @ W_slice
            // -> cb_partial [mix rows] ----
            if constexpr (x_grid_split) {
                project_block_split(extent, core_k_tiles, sb_h, v.off);
                cb_pop_front(cb_grid, extent);  // the grid is read only by the projection's x split
            } else if constexpr (w_pieces > 1) {
                if constexpr (w_presplit) {
                    if (block_idx == 0) {
                        MaybeDeviceZoneScope("c_w_publish");
                        w_publish_split(core_k_tiles);
                    }
                }
                // sum x^2 streams with it (pushes cb_partial [mix rows | sumsq rows] itself)
                project_sumsq_streamed<w_pieces>(extent, core_k_tiles, v);
            } else if (v.wait0 != 0) {
                // the next block sits behind the current one: indexed tile reads (matmul_block reads the front).
                // The bf16 W slice is waited here (the helper's in1 wait did it): cumulative, never popped.
                cb_wait_front(cb_weight, core_k_tiles);
                project_block_pieces<1>(extent, core_k_tiles, sb_h, v);
            } else {
                matmul_block<
                    false,
                    false,
                    LastBlockTarget::Out,
                    OutputCBLayout::SubblockMajor,
                    matmul_config::InitMode::Short,
                    InputPolicy::WaitAndRetainOnLastBlock,
                    InputPolicy::WaitAndRetainOnLastBlock>(
                    x_buf,
                    w_buf,
                    partial_buf,
                    partial_buf,
                    MatmulBlockShape::of(extent / sb_h, 1, sb_h, 1, core_k_tiles, 1));
            }
        }
        if constexpr (x_grid_split || w_pieces == 1) {  // bf16 X / fp32 W: streamed with the projection
            // ---- sumsq_block (after the projection: the X block is resident by now): per row,
            // Q = sum_k x_k*x_k (DEST-accumulated), then fp32 row-collapse -> cb_partial [sumsq rows].
            for (uint32_t t = 0; t < extent; ++t) {
                const uint32_t base = t * core_k_tiles;
                if constexpr (!x_grid_split) {  // fp32 X: cb_sq_acc was filled exactly by x_stats_block
                    cb_wait_front(cb_x_resident, v.wait0 + base + core_k_tiles);  // no-op: the projection waited
                    sumsq_row<WaitPolicy::None, TileAddressing::Offset>(core_k_tiles, v.off + base);
                }
                reduce<
                    PoolType::SUM,
                    ReduceDim::REDUCE_ROW,
                    cb_sq_acc,
                    cb_reduce_scaler,
                    cb_partial,
                    ReduceInputPolicy::WaitAndPopPerTile,
                    ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                    ReduceFp32Mode::Accurate>(ReduceInputBlockShape::single());
            }
        }
    };

    // ---- combine_block (root only; a no-op on the other ranks) ----
    auto combine_phase = [&]() {
        if (rank == 0) {
            {
                MaybeDeviceZoneScope("c_gather_wait");
                cb_wait_front(cb_gathered, group_cores * 2 * block_token_tiles);
            }
            MaybeDeviceZoneScope("c_combine");
            combine_block();
        }
    };

    // ---- coefficients_block: S (landed row-major in cb_coef_in, block b at its front) -> pre-column tiles ----
    //   transpose S -> gather into coefficient-major -> coefficients (r, pre) -> pre_i into T-layout row 0
    //   of its own tile -> transpose -> column 0 = pre_i per token row -> cb_pre_cols.
    // ---- owned_block (owned rows), fused in front of it: coefficients (post, logits) + Sinkhorn + the
    //   post / comb row-major tiles in ONE DEST window -> cb_comb_coef [post, comb] (the writer stores both
    //   pages as they are); the coefficient tile itself -> cb_coef_keep, reloaded exactly for the pre tiles.
    auto coef_phase = [&](uint32_t block_idx) {
        const uint32_t row0 = block_idx * block_token_tiles;
        const uint32_t extent = extent_of(block_idx);
        {
            MaybeDeviceZoneScope("c_coef_wait");
            cb_wait_front(cb_coef_in, 2 * block_token_tiles);
        }
        cb_wait_front(cb_bias_coef, 1);  // resident constant (writer-produced): waited, never popped
        cb_reserve_back(cb_pre_cols, n_streams * extent);
        for (uint32_t t = 0; t < extent; ++t) {
            const bool owned = ((row0 + t) % group_cores) == rank;
            if (owned) {
                MaybeDeviceZoneScope("c_owned");
                cb_reserve_back(cb_comb_coef, 2);
                cb_reserve_back(cb_coef_keep, 1);
                pack_reconfig_data_format(cb_comb_coef);  // cb_coef_keep: same (fp32) format
                tile_regs_acquire();
                load_coef_major(t, coef_scalars);
#ifndef MHC_ABLATE_COEF
                MATH((_llk_math_eltwise_unary_sfpu_params_(
                    mhc_sfpu::sinkhorn, 0, VectorMode::None, eps_bits, sinkhorn_iters)));
                MATH((_llk_math_eltwise_unary_sfpu_params_(mhc_sfpu::post_comb_tiles, 0, VectorMode::None)));
#endif
                transpose_dest_init<true, true>();
                transpose_dest<true, true>(2);
                transpose_dest<true, true>(3);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(2, cb_comb_coef);
                pack_tile(3, cb_comb_coef);
                pack_tile(0, cb_coef_keep);
                tile_regs_release();
                cb_push_back(cb_comb_coef, 2);
                cb_push_back(cb_coef_keep, 1);
                cb_wait_front(cb_coef_keep, 1);
            }
            MaybeDeviceZoneScope("c_pre");
            pack_reconfig_data_format(cb_pre_cols);
            tile_regs_acquire();
            if (owned) {
                reconfig_data_format_srca(cb_coef_keep);
                copy_tile_to_dst_init_short(cb_coef_keep);
                copy_tile(cb_coef_keep, 0, 0);
                custom_sfpu_init();
            } else {
                load_coef_major(t, coef_scalars);
            }
#ifndef MHC_ABLATE_COEF
            MATH((_llk_math_eltwise_unary_sfpu_params_(mhc_sfpu::pre_tiles, 0, VectorMode::None)));
#endif
            transpose_dest_init<true, true>();
            for (uint32_t i = 0; i < n_streams; ++i) {
                transpose_dest<true, true>(mhc_sfpu_tiles::pre_tile_index(i));
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < n_streams; ++i) {
                pack_tile(mhc_sfpu_tiles::pre_tile_index(i), cb_pre_cols);
            }
            tile_regs_release();
            if (owned) {
                cb_pop_front(cb_coef_keep, 1);
            }
        }
        cb_push_back(cb_pre_cols, n_streams * extent);
    };

    // ---- ymix_block: y[c] = sum_i x[i][c] * bcast_col(pre_i), n-deep DEST accumulation per output; frees X(b)
    // (the reader may load block b + depth) and S(b) ----
    auto ymix_phase = [&](uint32_t block_idx) {
        const uint32_t extent = extent_of(block_idx);
        cb_wait_front(cb_pre_cols, n_streams * extent);
        {
            MaybeDeviceZoneScope("c_ymix");
            for (uint32_t t = 0; t < extent; ++t) {
#ifdef MHC_ABLATE_YMIX
                pack_reconfig_data_format(cb_y_out);
                for (uint32_t c = 0; c < core_c_tiles; ++c) {
                    cb_reserve_back(cb_y_out, 1);
                    tile_regs_acquire();
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(0, cb_y_out);
                    tile_regs_release();
                    cb_push_back(cb_y_out, 1);
                }
                continue;
#endif
                eltwise_chain(
                    IterationShape::grid(core_c_tiles, n_streams),
                    BinaryFpu<
                        BinaryFpuOp::Mul,
                        input(
                            cb_x_resident,
                            WaitPolicy::None,
                            PopPolicy::None,
                            InputTileMapping::Block,
                            DataFormatReconfig::Enabled,
                            TileAddressing::Offset),
                        input(
                            input(
                                cb_pre_cols,
                                WaitPolicy::None,
                                PopPolicy::None,
                                InputTileMapping::Row,
                                DataFormatReconfig::Enabled,
                                TileAddressing::Offset),
                            BroadcastDim::Col),
                        Dst::D0,
                        DestAccumulation::PerRow>{t * core_k_tiles, t * n_streams},
                    PackTile<output(
                        cb_y_out,
                        ReservePolicy::PerOuter,
                        PushPolicy::PerOuter,
                        DataFormatReconfig::Enabled,
                        TileAddressing::Direct,
                        DestAccumulation::PerRow)>{});
            }
        }
        cb_pop_front(cb_pre_cols, n_streams * extent);
        if constexpr (x_grid_split) {
            cb_pop_front(cb_x_fp32, x_block_pages);  // alias kept in lockstep
        }
        cb_pop_front(cb_x_resident, x_block_pages);       // X block freed: the reader may load block + depth
        cb_pop_front(cb_coef_in, 2 * block_token_tiles);  // S consumed
    };

    // ---- block schedule (see the header): mirrored step for step by the writer ----
    auto pipe_at = [&](uint32_t b) {
        return b + 1 < num_blocks && (x_block_depth >= 3 || b + x_block_depth >= num_blocks);
    };
    bool combined = false;  // the root folded block b already (in the previous, pipelined step)
    if (num_blocks > 0) {
        proj_phase(0, XView{0, 0});
    }
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        if (pipe_at(block_idx)) {
            // X(b+1) ring slot relative to X(b) at the front: the next slot, or the ring start (wrap)
            const uint32_t slot = block_idx % x_block_depth;
            const uint32_t next_off =
                slot + 1 < x_block_depth ? x_block_pages : static_cast<uint32_t>(0u - slot * x_block_pages);
            proj_phase(block_idx + 1, XView{x_block_pages, next_off});
            if (!combined) {
                combine_phase();  // b
            }
            // root: the coefficients / Sinkhorn of b while the group's partials of b+1 arrive, the fold of b+1
            // (its S then multicasts under everyone's tail(b)), then the y-mix of b
            coef_phase(block_idx);
            combine_phase();  // b+1
            ymix_phase(block_idx);
            combined = true;
        } else {
            if (!combined) {
                combine_phase();
            }
            coef_phase(block_idx);
            ymix_phase(block_idx);
            if (block_idx + 1 < num_blocks) {
                proj_phase(block_idx + 1, XView{0, 0});  // X(b) popped: X(b+1) is at the front
            }
            combined = false;
        }
    }
}
