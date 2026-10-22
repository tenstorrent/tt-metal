// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm compute (regimes R1 and R3, op_design.md -> Block schedule / API Mapping).
//
// Per block of R_b tile-rows of this core's W-slice (all kernel-lib helpers; the only raw calls are the
// caller-owned CB waits / pops that the strided / Col caller-managed chain policies hand back to the caller):
//   tilize_block           (RM)  tilize<C>(R_b * num_w_chunks): cb_x_rm_sticks -> cb_x_tiles
//   square_accumulate_block      eltwise_chain: x*x, DEST-accumulated per tile-row -> cb_sumsq_partial (raw)
//                                  interleaved: grid(R_b, WS), cumulative wait (pad tiles are zero)
//                                  sharded:     grid(R_b, V) over the resident shard, strided by WS — only the
//                                               slice's valid tiles (a ragged last shard's pad tiles are never
//                                               read); the masked core leaves its partial last tile out and
//                                               squares it separately -> cb_sumsq_last
//   collapse_block               reduce<SUM, REDUCE_ROW>(R_b x 1), scaler 1.0:
//                                  R1 / G == 1: + post-op *(1/W), +eps, rsqrt -> cb_inv_rms
//                                  R3:          no post-op -> cb_partial_col (column 0 = this slice's Σx²);
//                                               the reader sends it to the row-group root
//                                  masked core: the partial last tile is reduced with the PARTIAL scaler
//                                               (columns >= W%32 weighted 0), accumulated onto the full tiles'
//                                               collapse (cb_sumsq_acc) — the W-pad mask lives in the reduce
//   combine_block   (R3 owner)   reduce<SUM, REDUCE_COL, AccumulateViaAdd, Skip>(G x owned rows) over the gathered
//                                partials + post-op (column-0 finalize) -> cb_rms_root; the reader multicasts it
//                                (one owner = the root) or unicasts it to every member (reduce-scatter owners)
//   normalize_block              per chunk c (R_b x cw), SCALE_BLOCK tiles per DEST window:
//                                  [gamma] x * gamma (Row bcast)     -> cb_x_gamma
//                                          (runs before cb_inv_rms is waited: chunk 0's γ pass is the
//                                           stall shadow of the R3 combine)
//                                  (x*gamma | x) * inv_rms (Col bcast) -> cb_output (TILE) / cb_y_tiles (RM)
//                                          interleaved: pushed one tile-row = C at a time (the writer's quantum)
//                                          sharded:     packed strided into the output shard (reserved and
//                                                       pushed once per block)
//                                  [RM] untilize<C>(R_b)           -> cb_output
//   release_block                pop cb_x_tiles (R_b * WS), cb_inv_rms (R_b)

#include <stdint.h>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
// [bcast_srcb_reuse experiment] BSR_MODE bit flags (0 = the op's chains everywhere):
//   1 = γ pass: stock per-tile LLK (no chain)      2 = γ pass: HOLD (srcB resident per γ column, column-major)
//   4 = inv pass: stock per-tile LLK (no chain)    8 = inv pass: HOLD (inv_rms tile resident per tile-row)
//   16 = γ HOLD with per-tile unpacker contexts for every format (default: only block-float srcA)
//   32 / 64 = inv HOLD only for chunks >= 2 / >= 4 tiles wide (else the chain); 128 = γ HOLD needs 2 DEST windows of
//   rows
// RECOMMENDED: BSR_MODE = 74 (2 | 8 | 64). γ HOLD always needs rows_in_block >= one DEST window (else the chain).
// Raw passes re-run their init / reconfig only on the first call of a consecutive run of calls (do_init).
#ifndef BSR_MODE
#define BSR_MODE 0
#endif
#include "api/compute/bcast.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/reconfig_data_format.h"
#include "bsr_llk.hpp"
#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"
#include "sfpu/ckernel_sfpu_converter.h"
#endif

namespace ckl = compute_kernel_lib;

// Stage zones (permanent; MaybeDeviceZoneScope, kernel_lib/perf_instrumentation.hpp). Each is recorded on all three
// TRISCs; only unpack (TRISC0) blocks on cb_wait_front and only pack (TRISC2) on cb_reserve_back, so explicit
// waits are zoned apart (cmp_wait_*). The helper-internal waits stay inside their stage zone (occupancy):
//   cmp_tilize (RM) | cmp_wait_x (sharded) | cmp_square (interleaved: includes the cumulative x wait)
//   cmp_collapse                 local REDUCE_ROW collapse (+ finalize on R1 / G == 1)
//   cmp_gamma0                   reduce-scatter: chunk 0's γ pass ahead of the combine
//   cmp_combine                  R3 owner: sum of the G gathered partials + finalize (includes the gather wait)
//   cmp_wait_gamma / cmp_gamma   per chunk: γ tiles wait / x*γ pass
//   cmp_wait_inv                 cb_inv_rms of the block (R3: the group's combine result)
//   cmp_inv / cmp_untilize       per chunk: (x*γ | x) * inv_rms pass / RM untilize

namespace {

// finalize_col0 (perf lamp L7, catalog sfpu_tile_scope "c_skip"): the finalize post-op inv_rms = rsqrt(Σx²·(1/W) + ε)
// fused into ONE SFPU pass over column 0 only. Every finalized tile is a REDUCE_ROW result (column 0 valid) and
// every consumer reads column 0 only (BroadcastDim::Col), so the stock three full-tile passes (mul_unary_tile,
// add_unary_tile, rsqrt_tile: 3 x 32 vectors, ~1.8 k cycles per tile-row on Blackhole) are 92 % waste.
// VectorMode::C walks faces 0 and 2 (columns 0-15); inside a face the SFPU vectors alternate column parity
// [rg0-even, rg0-odd, rg1-even, ...] and column 0 lives only in the even ones, so the body visits 4 even
// vectors per face (dst_reg += 2; net +8 = the stock ITERATIONS, so the face stepping composes unchanged):
// 8 vectors per tile instead of 96. The x·(1/W) + ε product stays in an fp32 SFPU register (the stock path
// stored it to DEST between passes). Raw sfpi: no compute API exposes a column-scoped or fused SFPU op.
// The other columns of the tile are left unfinalized; nothing reads them.
#ifdef TRISC_MATH
sfpi_inline void finalize_col0_body(const uint32_t inv_w_bits, const uint32_t eps_bits) {
    // The scalars are (re)loaded per vector: the sqrt body needs every other LREG.
#pragma GCC unroll 0
    for (int rg = 0; rg < 4; rg++) {
        const sfpi::vFloat mean_sq = sfpi::dst_reg[0] * ckernel::sfpu::Converter::as_float(inv_w_bits) +
                                     ckernel::sfpu::Converter::as_float(eps_bits);
        sfpi::vFloat t = ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, false>(mean_sq);
        if constexpr (!DST_ACCUM_MODE) {
            t = sfpi::convert<sfpi::vFloat16b>(t, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = t;
        sfpi::dst_reg += 2;  // skip the odd-parity vector (columns 1, 3, ..., 15 — never column 0)
    }
}
#endif

// ---- bcast_srcb_reuse: raw broadcast-multiply passes (BSR_MODE > 0) ----
// out(r, j) = A(r, j) * bcast, r < rows, j < cw. A tile = a_base + r * a_rs + j; bcast tile = b_base + r (Col) or
// b_base + j (Row); out tile = o_base + r * o_rs + j. Caller owns every CB wait / pop / reserve / push.
// Stock per-tile LLK (the exact unpack / math pair the chain's BinaryFpu emits), DW tiles per DEST window.
template <bool COL, uint32_t cbA, uint32_t cbB, uint32_t cbO, uint32_t DW>
ALWI void bsr_stock(
    uint32_t rows,
    uint32_t cw,
    uint32_t a_base,
    uint32_t a_rs,
    uint32_t b_base,
    uint32_t o_base,
    uint32_t o_rs,
    bool do_init = true) {
    constexpr auto BT = COL ? BroadcastType::COL : BroadcastType::ROW;
    if (do_init) {
        reconfig_data_format(cbA, cbB);
        pack_reconfig_data_format(cbO);
        MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BT, MATH_FIDELITY>(cbA, cbB, 0)));
        UNPACK((llk_unpack_AB_init<BT>(cbA, cbB)));
    }
    const uint32_t n = rows * cw;
    uint32_t r = 0, j = 0;    // pack position
    uint32_t ru = 0, ju = 0;  // unpack / math position
    for (uint32_t t0 = 0; t0 < n; t0 += DW) {
        const uint32_t m = (n - t0) < DW ? n - t0 : DW;
        tile_regs_acquire();
        for (uint32_t i = 0; i < m; ++i) {
            MATH((llk_math_eltwise_binary<
                  EltwiseBinaryType::ELWMUL,
                  BT,
                  DST_ACCUM_MODE,
                  MATH_FIDELITY,
                  EltwiseBinaryReuseDestType::NONE>(cbA, cbB, i, true)));
            UNPACK((llk_unpack_AB<BT>(cbA, cbB, a_base + ru * a_rs + ju, COL ? b_base + ru : b_base + ju)));
            if (++ju == cw) {
                ju = 0;
                ++ru;
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < m; ++i) {
            pack_tile<true>(i, cbO, o_base + r * o_rs + j);
            if (++j == cw) {
                j = 0;
                ++r;
            }
        }
        tile_regs_release();
    }
}

// HOLD, Row bcast (γ): column-major — γ tile j is unpacked into srcB once and held for the column's `rows` tiles.
template <uint32_t cbA, uint32_t cbB, uint32_t cbO, uint32_t DW>
ALWI void bsr_hold_row(
    uint32_t rows,
    uint32_t cw,
    uint32_t a_base,
    uint32_t a_rs,
    uint32_t b_base,
    uint32_t o_base,
    uint32_t o_rs,
    bool do_init = true) {
    if (do_init) {
        reconfig_data_format(cbA, cbB);
        pack_reconfig_data_format(cbO);
        MATH((bsr::math_hold_init<false>()));
        UNPACK((bsr::unpack_hold_init()));
    }
    // srcB <- γ tile j once, then the column's `rows` x tiles stream through srcA
    auto unpack_column = [&](uint32_t j) {
#ifdef TRISC_UNPACK
        // The W-counter advance steps by the 1024-datum tile body and does not cover a block-float tile's exponent
        // section (measured: bfp8_b x -> wrong tiles), so block-float srcA programs each tile's address in its own
        // unpacker context (BSR_MODE bit 16 forces that path for every format).
        if ((BSR_MODE & 16) || IS_BFP_FORMAT(unpack_src_format[get_operand_id(cbA)])) {
            bsr::unpack_hold_run(bsr::unpack_addr(cbA, a_base + j), bsr::unpack_addr(cbB, b_base + j), 1, 1);
            for (uint32_t r = 1; r < rows; ++r) {
                // (srcB address unused here — load_b = false — but must be a valid L1 address for the LLK assert)
                bsr::unpack_hold_run_pf(
                    bsr::unpack_addr(cbA, a_base + r * a_rs + j), bsr::unpack_addr(cbB, b_base + j), 1, 1, false);
            }
        } else {
            bsr::unpack_hold_run(bsr::unpack_addr(cbA, a_base + j), bsr::unpack_addr(cbB, b_base + j), rows, a_rs);
        }
#endif
    };
    // DEST windows inside one column (callers only take this path when rows >= DW; a ragged last window is short)
    for (uint32_t j = 0; j < cw; ++j) {
        unpack_column(j);
        for (uint32_t r0 = 0; r0 < rows; r0 += DW) {
            const uint32_t m = (rows - r0) < DW ? rows - r0 : DW;
            tile_regs_acquire();
#ifdef TRISC_MATH
            for (uint32_t i = 0; i < m; ++i) {
                bsr::math_hold_tile<false>(i);
            }
#endif
            tile_regs_commit();
            tile_regs_wait();
            uint32_t o = o_base + r0 * o_rs + j;
            for (uint32_t i = 0; i < m; ++i, o += o_rs) {
                pack_tile<true>(i, cbO, o);
            }
            tile_regs_release();
        }
        MATH((bsr::math_hold_release_b()));
    }
}

// HOLD, Col bcast (inv_rms): row-major — inv tile r held in srcB for the row's cw tiles. DEST windows are whole
// tile-rows (DW / cw rows per window) when cw <= DW, else DW-tile pieces of one row. Fixed-trip loops on purpose: a
// per-tile "slot" state machine with windows spanning rows measured ~10 % slower on the same work (bench i_hold vs
// i_hold_fl).
template <uint32_t cbA, uint32_t cbB, uint32_t cbO, uint32_t DW>
ALWI void bsr_hold_col(
    uint32_t rows,
    uint32_t cw,
    uint32_t a_base,
    uint32_t a_rs,
    uint32_t b_base,
    uint32_t o_base,
    uint32_t o_rs,
    bool do_init = true) {
    if (do_init) {
        reconfig_data_format(cbA, cbB);
        pack_reconfig_data_format(cbO);
        MATH((bsr::math_hold_init<true>()));
        UNPACK((bsr::unpack_hold_init()));
    }
    auto unpack_row = [&](uint32_t r) {
#ifdef TRISC_UNPACK
        const uint32_t a0 = a_base + r * a_rs;
        if (IS_BFP_FORMAT(unpack_src_format[get_operand_id(cbA)])) {
            // block-float srcA: explicit per-tile addresses (the W-counter advance skips the exponent section)
            bsr::unpack_hold_run(bsr::unpack_addr(cbA, a0), bsr::unpack_addr(cbB, b_base + r), 1, 1);
            for (uint32_t j = 1; j < cw; ++j) {
                bsr::unpack_hold_run_pf(bsr::unpack_addr(cbA, a0 + j), bsr::unpack_addr(cbB, b_base + r), 1, 1, false);
            }
        } else {
            bsr::unpack_hold_run(bsr::unpack_addr(cbA, a0), bsr::unpack_addr(cbB, b_base + r), cw, 1);
        }
#endif
    };
    if (cw <= DW) {
        const uint32_t rpw = DW / cw;
        for (uint32_t r0 = 0; r0 < rows; r0 += rpw) {
            const uint32_t nr = (rows - r0) < rpw ? rows - r0 : rpw;
            tile_regs_acquire();
            for (uint32_t rr = 0; rr < nr; ++rr) {
                unpack_row(r0 + rr);
#ifdef TRISC_MATH
                for (uint32_t j = 0; j < cw; ++j) {
                    bsr::math_hold_tile<true>(rr * cw + j);
                }
                bsr::math_hold_release_b();
#endif
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t rr = 0; rr < nr; ++rr) {
                const uint32_t o = o_base + (r0 + rr) * o_rs;
                for (uint32_t j = 0; j < cw; ++j) {
                    pack_tile<true>(rr * cw + j, cbO, o + j);
                }
            }
            tile_regs_release();
        }
    } else {
        for (uint32_t r = 0; r < rows; ++r) {
            unpack_row(r);
            for (uint32_t j0 = 0; j0 < cw; j0 += DW) {
                const uint32_t m = (cw - j0) < DW ? cw - j0 : DW;
                tile_regs_acquire();
#ifdef TRISC_MATH
                for (uint32_t i = 0; i < m; ++i) {
                    bsr::math_hold_tile<true>(i);
                }
#endif
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < m; ++i) {
                    pack_tile<true>(i, cbO, o_base + r * o_rs + j0 + i);
                }
                tile_regs_release();
            }
            MATH((bsr::math_hold_release_b()));
        }
    }
}

template <bool COL, uint32_t cbA, uint32_t cbB, uint32_t cbO, uint32_t DW>
ALWI void bsr_pass(
    uint32_t rows,
    uint32_t cw,
    uint32_t a_base,
    uint32_t a_rs,
    uint32_t b_base,
    uint32_t o_base,
    uint32_t o_rs,
    bool do_init = true) {
    if constexpr (!COL && (BSR_MODE & 2)) {
        bsr_hold_row<cbA, cbB, cbO, DW>(rows, cw, a_base, a_rs, b_base, o_base, o_rs, do_init);
    } else if constexpr (COL && (BSR_MODE & 8)) {
        bsr_hold_col<cbA, cbB, cbO, DW>(rows, cw, a_base, a_rs, b_base, o_base, o_rs, do_init);
    } else {
        bsr_stock<COL, cbA, cbB, cbO, DW>(rows, cw, a_base, a_rs, b_base, o_base, o_rs, do_init);
    }
}

constexpr uint32_t cb_x_rm_sticks = 0;
constexpr uint32_t cb_x_tiles = 1;
constexpr uint32_t cb_reduce_scaler = 2;
constexpr uint32_t cb_sumsq_partial = 3;
constexpr uint32_t cb_inv_rms = 4;
constexpr uint32_t cb_gamma_tiles = 6;
constexpr uint32_t cb_x_gamma = 7;
constexpr uint32_t cb_y_tiles = 8;
constexpr uint32_t cb_partial_col = 9;
constexpr uint32_t cb_gather = 10;
constexpr uint32_t cb_rms_root = 11;
constexpr uint32_t cb_sumsq_last = 12;
constexpr uint32_t cb_sumsq_acc = 13;
constexpr uint32_t cb_output = 16;

template <
    bool IS_RM,
    bool HAS_GAMMA,
    bool GAMMA_RESIDENT,
    bool SHARDED,
    bool COMBINE,
    bool MASKED_CFG,
    uint32_t BLOCK_ROWS,
    uint32_t C,
    uint32_t NUM_W_CHUNKS,
    uint32_t WS,
    uint32_t G,
    uint32_t SCALE_BLOCK,
    bool FINALIZE_COL0,
    uint32_t OWNERS,
    uint32_t WAVE_ROWS,
    uint32_t WAVE_Q,
    uint32_t GAMMA0_POS,
    uint32_t INV_GROUP>
FORCE_INLINE void run(
    uint32_t rows_this_core,
    uint32_t inv_w_bits,
    uint32_t eps_bits,
    uint32_t slice_valid,
    uint32_t group_index,
    bool masked_core) {
    constexpr uint32_t Wp = NUM_W_CHUNKS * C;  // γ window (padded slice)
    constexpr uint32_t BSR_DEST_CAP = DST_ACCUM_MODE ? 4 : 8;
    constexpr uint32_t BSR_DW = SCALE_BLOCK < BSR_DEST_CAP ? SCALE_BLOCK : BSR_DEST_CAP;
    // inv pass raw (stock / HOLD). Guards (measured, see the experiment's report): BSR_MODE bit 32 keeps the chain
    // for 1-tile-wide chunks, bit 64 for chunks narrower than 4 tiles (inv HOLD); bit 128 makes the γ HOLD need 2 DEST
    // windows of tile-rows instead of 1 (measured: costs b99991x64 its -4 %, so not recommended).
    constexpr bool BSR_INV_RAW = (BSR_MODE & 12) != 0;
    constexpr uint32_t BSR_INV_MIN_CW = (BSR_MODE & 64) ? 4 : ((BSR_MODE & 32) ? 2 : 1);
    constexpr uint32_t BSR_G_MIN_ROWS = (BSR_MODE & 128) ? 2 * BSR_DW : BSR_DW;
    // Final scaling packs straight into the output CB (TILE) or into cb_y_tiles ahead of untilize (RM).
    constexpr uint32_t cb_scaled = IS_RM ? cb_y_tiles : cb_output;
    // The local collapse finalizes (R1 / single-core group) or emits the raw partial for the root (R3).
    constexpr uint32_t cb_collapsed = COMBINE ? cb_partial_col : cb_inv_rms;

    // Finalize post-op: inv_rms = rsqrt(Σx² · (1/W) + ε), on column 0 (FINALIZE_COL0) or the whole tile.
    auto finalize = [inv_w_bits, eps_bits](uint32_t dst) {
        if constexpr (FINALIZE_COL0) {
            rsqrt_tile_init();  // the sqrt body's programmable constants
            MATH((_llk_math_eltwise_unary_sfpu_params_(
                finalize_col0_body, dst, ckernel::VectorMode::C, inv_w_bits, eps_bits)));
        } else {
            binop_with_scalar_tile_init();
            mul_unary_tile(dst, inv_w_bits);
            add_unary_tile(dst, eps_bits);
            rsqrt_tile_init();
            rsqrt_tile(dst);
        }
    };

    using SquareAccumulate = ckl::BinaryFpu<
        ckl::BinaryFpuOp::Mul,
        ckl::input(cb_x_tiles, ckl::WaitPolicy::Cumulative, ckl::PopPolicy::None, ckl::InputTileMapping::Block),
        ckl::input(cb_x_tiles, ckl::WaitPolicy::Cumulative, ckl::PopPolicy::None, ckl::InputTileMapping::Block),
        ckl::Dst::D0,
        ckl::DestAccumulation::PerRow>;
    using SumsqPack = ckl::PackTile<ckl::output(
        cb_sumsq_partial,
        ckl::ReservePolicy::PerOuter,
        ckl::PushPolicy::PerOuter,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Direct,
        ckl::DestAccumulation::PerRow)>;

    // x tiles: caller-managed (waited by square_accumulate / the shard wait, popped by release_block),
    // strided {base, row_stride = WS}.
    constexpr auto x_strided_input = ckl::input(
        cb_x_tiles,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Block,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Strided);
    // Sharded square stage: the resident shard, strided, accumulated per tile-row.
    using SquareAccumulateStrided = ckl::
        BinaryFpu<ckl::BinaryFpuOp::Mul, x_strided_input, x_strided_input, ckl::Dst::D0, ckl::DestAccumulation::PerRow>;
    // inv_rms: column-broadcast, indexed by tile-row; caller waits once per block, pops at release.
    constexpr auto inv_rms_input = ckl::input(
        cb_inv_rms, ckl::BroadcastDim::Col, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::InputTileMapping::Col);
    // combine_pipeline: the same column-broadcast inv_rms, addressed from a wave's first row (base row0, stride 1).
    constexpr auto inv_rms_strided_input = ckl::input(
        cb_inv_rms,
        ckl::BroadcastDim::Col,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Col,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Strided);
    // Scaling stages run SCALE_BLOCK tiles per DEST window (the chain clamps it to DEST capacity), so the
    // per-window dst-sync + CB handshake is paid once per block, not per tile. Every operand is block-
    // compatible: x / γ / inv_rms are caller-managed or held, cb_x_gamma is fully packed by scale_by_gamma
    // before scale_by_inv_rms starts (same thread) so it is waited upfront and popped at the end, and the
    // interleaved outputs are reserved/pushed once per tile-row (C pages: the writer's / untilize's quantum).
    // Sharded outputs are packed strided into the output shard (caller reserves / pushes the block).
    constexpr auto scaled_row_output = [](uint32_t cb) {
        return ckl::output(cb, ckl::ReservePolicy::PerOuter, ckl::PushPolicy::PerOuter);
    };
    constexpr auto scaled_shard_output = ckl::output(
        cb_output,
        ckl::ReservePolicy::None,
        ckl::PushPolicy::None,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Strided);
    constexpr auto x_gamma_input =
        ckl::input(cb_x_gamma, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block);

    // Local collapse of rows_in_block raw partials of cb_in into cb_collapsed; FINAL = apply the finalize
    // post-op (R1 / single-core group).
    auto collapse = [&](uint32_t rows_in_block,
                        auto partial_scaler,
                        auto accumulate,
                        auto post_op,
                        auto cb_in_tag,
                        auto cb_out_tag) {
        constexpr uint32_t cb_in = decltype(cb_in_tag)::value;
        constexpr uint32_t cb_out = decltype(cb_out_tag)::value;
        ckl::reduce<
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW,
            cb_in,
            cb_reduce_scaler,
            cb_out,
            ckl::ReduceInputPolicy::BulkWaitBulkPop>(
            ckl::ReduceInputBlockShape::of(rows_in_block, 1),
            ckl::ReduceInputMemoryLayout::contiguous(),
            accumulate,
            post_op,
            partial_scaler);
    };
    using CbSumsqPartial = std::integral_constant<uint32_t, cb_sumsq_partial>;
    using CbSumsqLast = std::integral_constant<uint32_t, cb_sumsq_last>;
    using CbSumsqAcc = std::integral_constant<uint32_t, cb_sumsq_acc>;
    using CbCollapsed = std::integral_constant<uint32_t, cb_collapsed>;
    auto collapse_post = [&](uint32_t dst) {
        if constexpr (!COMBINE) {
            finalize(dst);
        }
    };

    // scale_by_gamma of chunk c (x * γ, γ row 0 broadcast down the tile, re-used by all R_b rows) -> cb_x_gamma.
    auto scale_by_gamma = [&](uint32_t c, uint32_t rows_in_block) {
        // Chunk width: C, except a shard-backed slice's narrower last chunk (WS = S there).
        const uint32_t cw = (c + 1 == NUM_W_CHUNKS) ? WS - c * C : C;
        const ckl::StridedTileRange x_chunk{c * C, WS};
        if constexpr (HAS_GAMMA && GAMMA_RESIDENT) {
            // scale_by_gamma, resident γ: chunk c's γ tiles are [c*C, c*C + C) of the resident window.
            // Cumulative wait: the writer pushes the window chunk by chunk; after block 0 it is a no-op.
            {
                MaybeDeviceZoneScope("cmp_wait_gamma");
                cb_wait_front(cb_gamma_tiles, (c + 1) * C);
            }
            MaybeDeviceZoneScope("cmp_gamma");
#if (BSR_MODE & 3)
            // HOLD needs a full DEST window of tile-rows per γ column (measured: fewer rows -> short windows, slower)
            if ((BSR_MODE & 1) || rows_in_block >= BSR_G_MIN_ROWS) {
                cb_reserve_back(cb_x_gamma, rows_in_block * cw);
                bsr_pass<false, cb_x_tiles, cb_gamma_tiles, cb_x_gamma, BSR_DW>(
                    rows_in_block, cw, c * C, WS, c * C, 0, cw);
                cb_push_back(cb_x_gamma, rows_in_block * cw);
                return;
            }
#endif
            ckl::eltwise_chain(
                ckl::IterationShape::grid(rows_in_block, cw).block_size(SCALE_BLOCK),
                ckl::BinaryFpu<
                    ckl::BinaryFpuOp::Mul,
                    x_strided_input,
                    ckl::input(
                        cb_gamma_tiles,
                        ckl::BroadcastDim::Row,
                        ckl::WaitPolicy::None,
                        ckl::PopPolicy::None,
                        ckl::InputTileMapping::Row,
                        ckl::DataFormatReconfig::Enabled,
                        ckl::TileAddressing::Offset)>{x_chunk, c * C},
                ckl::PackTile<scaled_row_output(cb_x_gamma)>{});
        } else if constexpr (HAS_GAMMA) {
            // scale_by_gamma: x * gamma (gamma row 0 broadcast down the tile, re-used by all R_b rows).
            // The streamed chunk is always C pages; a narrower shard chunk reads its first cw.
            {
                MaybeDeviceZoneScope("cmp_wait_gamma");
                cb_wait_front(cb_gamma_tiles, C);
            }
            MaybeDeviceZoneScope("cmp_gamma");
#if (BSR_MODE & 3)
            if ((BSR_MODE & 1) || rows_in_block >= BSR_G_MIN_ROWS) {
                cb_reserve_back(cb_x_gamma, rows_in_block * cw);
                bsr_pass<false, cb_x_tiles, cb_gamma_tiles, cb_x_gamma, BSR_DW>(rows_in_block, cw, c * C, WS, 0, 0, cw);
                cb_push_back(cb_x_gamma, rows_in_block * cw);
                cb_pop_front(cb_gamma_tiles, C);
                return;
            }
#endif
            ckl::eltwise_chain(
                ckl::IterationShape::grid(rows_in_block, cw).block_size(SCALE_BLOCK),
                ckl::BinaryFpu<
                    ckl::BinaryFpuOp::Mul,
                    x_strided_input,
                    ckl::input(
                        cb_gamma_tiles,
                        ckl::BroadcastDim::Row,
                        ckl::WaitPolicy::None,
                        ckl::PopPolicy::None,
                        ckl::InputTileMapping::Row)>{x_chunk},
                ckl::PackTile<scaled_row_output(cb_x_gamma)>{});
            cb_pop_front(cb_gamma_tiles, C);
        }
    };

    const uint32_t num_blocks = (rows_this_core + BLOCK_ROWS - 1) / BLOCK_ROWS;
    if constexpr (WAVE_ROWS > 0) {
        // ================= combine_pipeline: software-pipelined R3 reduce-scatter (TILE shard, OWNERS > 1) ======
        // The block's tile-rows are processed in WAVES of WAVE_ROWS rows (the last wave of a block may be short).
        // Each wave is split over the owners by WAVE_Q rows (owner i of wave k: rows [k*Rw + i*q_w, +q_w)), so
        // every wave's exchange is spread across the whole row-group (balanced NoC load, like the one-shot split).
        // Schedule per block (vs the one-shot square -> collapse -> γ0 -> combine -> wait inv -> inv):
        //   1. square + collapse of wave 0, 1, ..., W-1: the reader sends wave k's partials while this thread
        //      squares wave k+1 (front-side overlap: the exchange no longer waits for the whole shard's square)
        //   2. combine of wave 0..W-1 (owner of the wave), each waiting only its own wave's gather; γ chunk 0 runs
        //      right before combine(GAMMA0_POS) (>= W: after them) — it shadows the gather waits of the later
        //      waves, and the all-gather of the early waves runs under it
        //   3. per chunk c, per wave k: (c == 0) wait cb_inv_rms cumulatively through wave k, scale wave k —
        //      back-side overlap: wave k scales while later waves' inv_rms are still landing
        // Math is identical: the same chains / reduce over the same tiles; only their order and grouping change.
        // Full blocks use the host's (WAVE_ROWS, WAVE_Q); a short last block re-splits its rows over the same H
        // holders (q_w <= WAVE_Q, waves <= WF: its regions fit the CBs and the semaphores) so every owner keeps work.
        constexpr uint32_t WF = (BLOCK_ROWS + WAVE_ROWS - 1) / WAVE_ROWS;  // waves per full block
        constexpr uint32_t H = (WAVE_ROWS + WAVE_Q - 1) / WAVE_Q;  // gather-slot holders (push / pop G*q_w per wave)
        const bool holder = group_index < H;
        static_assert(SHARDED && COMBINE && OWNERS > 1, "combine_pipeline waves: TILE-sharded reduce-scatter only");
        for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
            const uint32_t remaining = rows_this_core - block_idx * BLOCK_ROWS;
            const uint32_t rows_in_block = remaining < BLOCK_ROWS ? remaining : BLOCK_ROWS;  // R_b
            uint32_t qw = WAVE_Q;
            if (rows_in_block != BLOCK_ROWS) {
                const uint32_t q = ((rows_in_block + H - 1) / H + WF - 1) / WF;
                qw = q < WAVE_Q ? q : WAVE_Q;
            }
            const uint32_t Rw = qw * H;
            const uint32_t waves = (rows_in_block + Rw - 1) / Rw;
            {
                MaybeDeviceZoneScope("cmp_wait_x");
                cb_wait_front(cb_x_tiles, rows_in_block * WS);
            }
            const uint32_t full_tiles = masked_core ? slice_valid - 1 : slice_valid;
            // ---- 1. square + collapse per wave ----
            for (uint32_t k = 0; k < waves; ++k) {
                const uint32_t row0 = k * Rw;
                const uint32_t n = (rows_in_block - row0) < Rw ? rows_in_block - row0 : Rw;
                const uint32_t xbase = row0 * WS;
                if (full_tiles > 0) {
                    MaybeDeviceZoneScope("cmp_square");
                    ckl::eltwise_chain(
                        ckl::IterationShape::grid(n, full_tiles),
                        SquareAccumulateStrided{ckl::StridedTileRange{xbase, WS}, ckl::StridedTileRange{xbase, WS}},
                        SumsqPack{});
                }
                MaybeDeviceZoneScope("cmp_collapse");
                if constexpr (MASKED_CFG) {
                    if (masked_core) {
                        const ckl::StridedTileRange last_tile{xbase + slice_valid - 1, WS};
                        ckl::eltwise_chain(
                            ckl::IterationShape::grid(n, 1),
                            SquareAccumulateStrided{last_tile, last_tile},
                            ckl::PackTile<ckl::output(
                                cb_sumsq_last,
                                ckl::ReservePolicy::PerOuter,
                                ckl::PushPolicy::PerOuter,
                                ckl::DataFormatReconfig::Enabled,
                                ckl::TileAddressing::Direct,
                                ckl::DestAccumulation::PerRow)>{});
                        if (full_tiles > 0) {
                            collapse(
                                n,
                                ckl::ReducePartialScaler::none(),
                                ckl::NoAccumulation{},
                                ckl::NoOp{},
                                CbSumsqPartial{},
                                CbSumsqAcc{});
                            collapse(
                                n,
                                ckl::ReducePartialScaler::with_partial(),
                                ckl::Accumulate::at(cb_sumsq_acc, 1),
                                collapse_post,
                                CbSumsqLast{},
                                CbCollapsed{});
                        } else {
                            collapse(
                                n,
                                ckl::ReducePartialScaler::with_partial(),
                                ckl::NoAccumulation{},
                                collapse_post,
                                CbSumsqLast{},
                                CbCollapsed{});
                        }
                    }
                }
                if (!(MASKED_CFG && masked_core)) {
                    collapse(
                        n,
                        ckl::ReducePartialScaler::none(),
                        ckl::NoAccumulation{},
                        collapse_post,
                        CbSumsqPartial{},
                        CbCollapsed{});
                }
            }
            // ---- 2. combine per wave (holder: own rows reduced, or an alignment-only quantum popped) ----
            auto combine_wave = [&](uint32_t k) {
                if (!holder) {
                    return;
                }
                const uint32_t row0 = k * Rw;
                const uint32_t n = (rows_in_block - row0) < Rw ? rows_in_block - row0 : Rw;
                const uint32_t first = group_index * qw;
                MaybeDeviceZoneScope("cmp_combine");
                if (first < n) {
                    const uint32_t owned = (n - first) < qw ? n - first : qw;
                    ckl::reduce<
                        ckernel::PoolType::SUM,
                        ckernel::ReduceDim::REDUCE_COL,
                        cb_gather,
                        cb_reduce_scaler,
                        cb_rms_root,
                        ckl::ReduceInputPolicy::BulkWaitBulkPop,
                        ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                        ReduceFp32Mode::Fast,
                        ckl::ReduceAlgorithm::AccumulateViaAdd,
                        ckl::ReduceWithinTile::Skip>(
                        ckl::ReduceInputBlockShape::of(G, owned),
                        ckl::ReduceInputMemoryLayout::with_row_stride(qw),
                        ckl::NoAccumulation{},
                        finalize);
                } else {
                    // This holder owns no row of this (short) wave: release the alignment-only quantum.
                    cb_wait_front(cb_gather, G * qw);
                    cb_pop_front(cb_gather, G * qw);
                }
            };
            // γ chunk 0 runs right before combine(GAMMA0_POS) (>= waves: after every combine). It is independent of
            // inv_rms, so it fills this core's wait for the later waves' gathers.
            // GAMMA0_POS == 255: half-way (before combine(waves / 2)).
            const uint32_t gamma0_at = GAMMA0_POS == 255 ? waves / 2 : (GAMMA0_POS < waves ? GAMMA0_POS : waves);
            for (uint32_t k = 0; k <= waves; ++k) {
                if constexpr (HAS_GAMMA) {
                    if (k == gamma0_at) {
                        MaybeDeviceZoneScope("cmp_gamma0");
                        scale_by_gamma(0, rows_in_block);
                    }
                }
                if (k < waves) {
                    combine_wave(k);
                }
            }
            // ---- 3. normalize per chunk, per wave ----
            cb_reserve_back(cb_output, rows_in_block * WS);
            for (uint32_t c = 0; c < NUM_W_CHUNKS; ++c) {
                const uint32_t cw = (c + 1 == NUM_W_CHUNKS) ? WS - c * C : C;
                if constexpr (HAS_GAMMA) {
                    if (c > 0) {
                        scale_by_gamma(c, rows_in_block);
                    }
                }
                // INV_GROUP waves per scaling call (fewer chain setups; the group waits for its last wave). Chunks
                // c > 0 run after chunk 0 waited every wave's inv_rms: one call over the whole block.
                // inv_rms addressing: with one W-chunk each wave's rows are popped after use, so the wave's rows are
                // always at the CB front and the plain Col mapping applies (measured: the Strided Col operand costs
                // ~8 cycles / tile on the unpacker). With several chunks the rows stay resident for the later chunks,
                // so chunk 0's per-wave calls address them Strided from the wave's first row, and c > 0 (one call
                // from row 0) is plain Col again.
                const uint32_t group = c == 0 ? INV_GROUP : waves;
                for (uint32_t k = 0; k < waves; k += group) {
                    const uint32_t row0 = k * Rw;
                    const uint32_t span = group * Rw;
                    const uint32_t n = (rows_in_block - row0) < span ? rows_in_block - row0 : span;
                    if (c == 0) {
                        MaybeDeviceZoneScope("cmp_wait_inv");
                        // waves 0..k+INV_GROUP-1 have landed (front-relative once earlier waves are popped)
                        cb_wait_front(cb_inv_rms, (NUM_W_CHUNKS == 1) ? n : row0 + n);
                    }
                    MaybeDeviceZoneScope("cmp_inv");
                    const ckl::StridedTileRange out_range{row0 * WS + c * C, WS};
                    const bool direct = (NUM_W_CHUNKS == 1) || c > 0;
                    if (BSR_INV_RAW && cw >= BSR_INV_MIN_CW) {
                        (void)out_range;
                        if constexpr (HAS_GAMMA) {
                            cb_wait_front(cb_x_gamma, n * cw);
                            bsr_pass<true, cb_x_gamma, cb_inv_rms, cb_output, BSR_DW>(
                                n, cw, 0, cw, direct ? 0 : row0, row0 * WS + c * C, WS, k == 0);
                            cb_pop_front(cb_x_gamma, n * cw);
                        } else {
                            bsr_pass<true, cb_x_tiles, cb_inv_rms, cb_output, BSR_DW>(
                                n, cw, row0 * WS + c * C, WS, direct ? 0 : row0, row0 * WS + c * C, WS, k == 0);
                        }
                    } else {
                        if (direct) {
                            if constexpr (HAS_GAMMA) {
                                ckl::eltwise_chain(
                                    ckl::IterationShape::grid(n, cw).block_size(SCALE_BLOCK),
                                    ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_gamma_input, inv_rms_input>{},
                                    ckl::PackTile<scaled_shard_output>{out_range});
                            } else {
                                ckl::eltwise_chain(
                                    ckl::IterationShape::grid(n, cw).block_size(SCALE_BLOCK),
                                    ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_strided_input, inv_rms_input>{out_range},
                                    ckl::PackTile<scaled_shard_output>{out_range});
                            }
                        } else {
                            const ckl::StridedTileRange inv_range{row0, 1};
                            if constexpr (HAS_GAMMA) {
                                ckl::eltwise_chain(
                                    ckl::IterationShape::grid(n, cw).block_size(SCALE_BLOCK),
                                    ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_gamma_input, inv_rms_strided_input>{
                                        0u, inv_range},
                                    ckl::PackTile<scaled_shard_output>{out_range});
                            } else {
                                ckl::eltwise_chain(
                                    ckl::IterationShape::grid(n, cw).block_size(SCALE_BLOCK),
                                    ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_strided_input, inv_rms_strided_input>{
                                        out_range, inv_range},
                                    ckl::PackTile<scaled_shard_output>{out_range});
                            }
                        }
                    }
                    if constexpr (NUM_W_CHUNKS == 1) {
                        cb_pop_front(cb_inv_rms, n);
                    }
                }
            }
            cb_push_back(cb_output, rows_in_block * WS);
            cb_pop_front(cb_x_tiles, rows_in_block * WS);
            if constexpr (NUM_W_CHUNKS > 1) {
                cb_pop_front(cb_inv_rms, rows_in_block);
            }
        }
    } else {
        for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
            const uint32_t remaining = rows_this_core - block_idx * BLOCK_ROWS;
            const uint32_t rows_in_block = remaining < BLOCK_ROWS ? remaining : BLOCK_ROWS;  // R_b

            // tilize_block (RM): R_b * num_w_chunks sub-blocks of 1 x C tiles, one init/uninit per block.
            if constexpr (IS_RM) {
                MaybeDeviceZoneScope("cmp_tilize");
                ckl::tilize<C, cb_x_rm_sticks, cb_x_tiles>(rows_in_block * NUM_W_CHUNKS);
            }

            // ---- square_accumulate_block + collapse_block ----
            if constexpr (!SHARDED) {
                // One chain per block; cumulative wait so math starts on the first landed tile.
                {
                    MaybeDeviceZoneScope("cmp_square");
                    ckl::eltwise_chain(ckl::IterationShape::grid(rows_in_block, WS), SquareAccumulate{}, SumsqPack{});
                }
                MaybeDeviceZoneScope("cmp_collapse");
                collapse(
                    rows_in_block,
                    ckl::ReducePartialScaler::none(),
                    ckl::NoAccumulation{},
                    collapse_post,
                    CbSumsqPartial{},
                    CbCollapsed{});
            } else {
                // The whole block of the resident shard is present (the reader pushed the shard).
                {
                    MaybeDeviceZoneScope("cmp_wait_x");
                    cb_wait_front(cb_x_tiles, rows_in_block * WS);
                }
                const uint32_t full_tiles = masked_core ? slice_valid - 1 : slice_valid;
                if (full_tiles > 0) {
                    MaybeDeviceZoneScope("cmp_square");
                    ckl::eltwise_chain(
                        ckl::IterationShape::grid(rows_in_block, full_tiles),
                        SquareAccumulateStrided{ckl::StridedTileRange{0, WS}, ckl::StridedTileRange{0, WS}},
                        SumsqPack{});
                }
                MaybeDeviceZoneScope("cmp_collapse");
                if constexpr (MASKED_CFG) {
                    if (masked_core) {
                        // The partial last tile, squared on its own: its pad columns are dropped by the reduce's
                        // partial scaler below (the caller's shard is read-only, so they cannot be zeroed).
                        const ckl::StridedTileRange last_tile{slice_valid - 1, WS};
                        ckl::eltwise_chain(
                            ckl::IterationShape::grid(rows_in_block, 1),
                            SquareAccumulateStrided{last_tile, last_tile},
                            ckl::PackTile<ckl::output(
                                cb_sumsq_last,
                                ckl::ReservePolicy::PerOuter,
                                ckl::PushPolicy::PerOuter,
                                ckl::DataFormatReconfig::Enabled,
                                ckl::TileAddressing::Direct,
                                ckl::DestAccumulation::PerRow)>{});
                        if (full_tiles > 0) {
                            collapse(
                                rows_in_block,
                                ckl::ReducePartialScaler::none(),
                                ckl::NoAccumulation{},
                                ckl::NoOp{},
                                CbSumsqPartial{},
                                CbSumsqAcc{});
                            collapse(
                                rows_in_block,
                                ckl::ReducePartialScaler::with_partial(),
                                ckl::Accumulate::at(cb_sumsq_acc, 1),
                                collapse_post,
                                CbSumsqLast{},
                                CbCollapsed{});
                        } else {
                            collapse(
                                rows_in_block,
                                ckl::ReducePartialScaler::with_partial(),
                                ckl::NoAccumulation{},
                                collapse_post,
                                CbSumsqLast{},
                                CbCollapsed{});
                        }
                    }
                }
                if (!(MASKED_CFG && masked_core)) {
                    collapse(
                        rows_in_block,
                        ckl::ReducePartialScaler::none(),
                        ckl::NoAccumulation{},
                        collapse_post,
                        CbSumsqPartial{},
                        CbCollapsed{});
                }
            }

            // Reduce-scatter (OWNERS > 1): chunk 0's γ pass runs BEFORE the combine — it is independent of inv_rms,
            // so it fills this owner's wait for the other members' partials. Not with a single root: there the root's
            // combine gates every member's multicast wait, so the root must reduce first (γ(0) then shadows the
            // multicast wait instead, as before).
            constexpr bool gamma0_before_combine = COMBINE && HAS_GAMMA && OWNERS > 1;
            if constexpr (gamma0_before_combine) {
                MaybeDeviceZoneScope("cmp_gamma0");
                scale_by_gamma(0, rows_in_block);
            }

            // ---- combine_block (R3): this core's owned tile-rows [index*q, +owned) of the block — sum their G
            // gathered partials, finalize. q = ceil(R_b / OWNERS) is the uniform per-owner quantum. cb_gather is laid
            // out contributor-major — contributor j's partials of the owned rows sit contiguously at slots [j*q, j*q +
            // owned) so each contributor delivers them in ONE NoC write — i.e. a (G x q) row-major block whose column r
            // holds row r's G partials. REDUCE_COL with AccumulateViaAdd + Skip sums each column's G tiles elementwise
            // (the partials are already collapsed: column 0 valid); the row pitch q skips a ragged last owner's unused
            // columns, and BulkWaitBulkPop waits / pops the whole G*q quantum the reader pushes on every owner (so each
            // owner's write pointer returns to the shared base). OWNERS == 1: the root (index 0) owns every row.
            if constexpr (COMBINE) {
                const uint32_t q = (rows_in_block + OWNERS - 1) / OWNERS;
                const uint32_t first = group_index * q;
                if (first < rows_in_block) {
                    MaybeDeviceZoneScope("cmp_combine");
                    const uint32_t owned = (rows_in_block - first) < q ? rows_in_block - first : q;
                    ckl::reduce<
                        ckernel::PoolType::SUM,
                        ckernel::ReduceDim::REDUCE_COL,
                        cb_gather,
                        cb_reduce_scaler,
                        cb_rms_root,
                        ckl::ReduceInputPolicy::BulkWaitBulkPop,
                        ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                        ReduceFp32Mode::Fast,
                        ckl::ReduceAlgorithm::AccumulateViaAdd,
                        ckl::ReduceWithinTile::Skip>(
                        ckl::ReduceInputBlockShape::of(G, owned),
                        ckl::ReduceInputMemoryLayout::with_row_stride(q),
                        ckl::NoAccumulation{},
                        finalize);
                }
            }

            // ---- normalize_block ----
            if constexpr (SHARDED) {
                cb_reserve_back(cb_output, rows_in_block * WS);
            }
            for (uint32_t c = 0; c < NUM_W_CHUNKS; ++c) {
                // Chunk width: C, except a shard-backed slice's narrower last chunk (WS = S there).
                const uint32_t cw = (c + 1 == NUM_W_CHUNKS) ? WS - c * C : C;
                const ckl::StridedTileRange x_chunk{c * C, WS};
                if constexpr (HAS_GAMMA) {
                    // Reduce-scatter: chunk 0 already ran ahead of the combine.
                    if (!(gamma0_before_combine && c == 0)) {
                        scale_by_gamma(c, rows_in_block);
                    }
                }
                if (c == 0) {
                    // inv_rms of the block (R1: produced above on this thread; R3: the multicast from the root).
                    MaybeDeviceZoneScope("cmp_wait_inv");
                    cb_wait_front(cb_inv_rms, rows_in_block);
                }
                {
                    MaybeDeviceZoneScope("cmp_inv");
                    if (SHARDED && BSR_INV_RAW && cw >= BSR_INV_MIN_CW) {
                        if constexpr (HAS_GAMMA) {
                            cb_wait_front(cb_x_gamma, rows_in_block * cw);
                            bsr_pass<true, cb_x_gamma, cb_inv_rms, cb_output, BSR_DW>(
                                rows_in_block, cw, 0, cw, 0, c * C, WS);
                            cb_pop_front(cb_x_gamma, rows_in_block * cw);
                        } else {
                            bsr_pass<true, cb_x_tiles, cb_inv_rms, cb_output, BSR_DW>(
                                rows_in_block, cw, c * C, WS, 0, c * C, WS);
                        }
                    } else if constexpr (SHARDED) {
                        if constexpr (HAS_GAMMA) {
                            ckl::eltwise_chain(
                                ckl::IterationShape::grid(rows_in_block, cw).block_size(SCALE_BLOCK),
                                ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_gamma_input, inv_rms_input>{},
                                ckl::PackTile<scaled_shard_output>{x_chunk});
                        } else {
                            ckl::eltwise_chain(
                                ckl::IterationShape::grid(rows_in_block, cw).block_size(SCALE_BLOCK),
                                ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_strided_input, inv_rms_input>{x_chunk},
                                ckl::PackTile<scaled_shard_output>{x_chunk});
                        }
                    } else if constexpr (HAS_GAMMA) {
                        // scale_by_inv_rms: (x*gamma) * inv_rms (cb_x_gamma fully packed by the γ chain above).
                        ckl::eltwise_chain(
                            ckl::IterationShape::grid(rows_in_block, C).block_size(SCALE_BLOCK),
                            ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_gamma_input, inv_rms_input>{},
                            ckl::PackTile<scaled_row_output(cb_scaled)>{});
                    } else {
                        // scale_by_inv_rms: x * inv_rms.
                        ckl::eltwise_chain(
                            ckl::IterationShape::grid(rows_in_block, C).block_size(SCALE_BLOCK),
                            ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_strided_input, inv_rms_input>{x_chunk},
                            ckl::PackTile<scaled_row_output(cb_scaled)>{});
                    }
                }  // cmp_inv
                if constexpr (IS_RM) {
                    // untilize_chunk: R_b tile-rows x C -> row-major pages for the writer.
                    MaybeDeviceZoneScope("cmp_untilize");
                    ckl::untilize<C, cb_y_tiles, cb_output>(rows_in_block);
                }
            }
            if constexpr (SHARDED) {
                cb_push_back(cb_output, rows_in_block * WS);
            }

            // release_block
            cb_pop_front(cb_x_tiles, rows_in_block * WS);
            cb_pop_front(cb_inv_rms, rows_in_block);
        }
    }

    // reduce() waits the scaler but never pops it; the caller owns that page (or pair).
    cb_pop_front(cb_reduce_scaler, MASKED_CFG ? 2 : 1);
    if constexpr (HAS_GAMMA && GAMMA_RESIDENT) {
        if (num_blocks > 0) {
            cb_pop_front(cb_gamma_tiles, Wp);
        }
    }
}

}  // namespace

void kernel_main() {
    constexpr bool is_rm = get_compile_time_arg_val(0) != 0;
    constexpr bool has_gamma = get_compile_time_arg_val(1) != 0;
    constexpr uint32_t block_rows = get_compile_time_arg_val(2);
    constexpr uint32_t w_chunk_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t num_w_chunks = get_compile_time_arg_val(4);
    constexpr bool gamma_resident = get_compile_time_arg_val(11) != 0;
    constexpr bool sharded = get_compile_time_arg_val(14) != 0;
    constexpr bool combine = get_compile_time_arg_val(15) != 0;
    constexpr uint32_t x_row_stride = get_compile_time_arg_val(16);  // WS
    constexpr uint32_t group_size = get_compile_time_arg_val(17);    // G
    constexpr bool masked_cfg = get_compile_time_arg_val(18) != 0;
    // Compute-only CT arg (after the 22-entry shared prefix): tiles per DEST window of the scaling stages.
    constexpr uint32_t scale_block_tiles = get_compile_time_arg_val(22);
    // Compute-only CT arg: finalize on column 0 only (one fused SFPU pass) instead of three full-tile passes.
    constexpr bool finalize_col0 = get_compile_time_arg_val(23) != 0;
    // R3 combine owners per row-group (1 = root reduce; > 1 = reduce-scatter over the first OWNERS members).
    constexpr uint32_t combine_owners = get_compile_time_arg_val(24);
    // combine_pipeline CT args: wave rows (0 = the one-shot baseline schedule), per-owner rows per wave, γ0 slot.
    constexpr uint32_t wave_rows = get_compile_time_arg_val(25);
    constexpr uint32_t wave_q = get_compile_time_arg_val(26);
    constexpr uint32_t gamma0_pos = get_compile_time_arg_val(27);
    constexpr uint32_t inv_group = get_compile_time_arg_val(28);

    const uint32_t rows_this_core = get_arg_val<uint32_t>(0);
    const uint32_t inv_w_bits = get_arg_val<uint32_t>(1);
    const uint32_t eps_bits = get_arg_val<uint32_t>(2);
    const uint32_t slice_valid = get_arg_val<uint32_t>(3);  // V: valid tiles of this core's W-slice
    const uint32_t group_index = get_arg_val<uint32_t>(4);  // position in the row-group (0 = root)
    const bool masked_core = get_arg_val<uint32_t>(5) != 0;

    if constexpr (is_rm) {
        compute_kernel_hw_startup(cb_x_rm_sticks, cb_x_rm_sticks, cb_x_tiles);
    } else {
        compute_kernel_hw_startup(cb_x_tiles, cb_x_tiles, cb_sumsq_partial);
    }

    run<is_rm,
        has_gamma,
        gamma_resident,
        sharded,
        combine,
        masked_cfg,
        block_rows,
        w_chunk_tiles,
        num_w_chunks,
        x_row_stride,
        group_size,
        scale_block_tiles,
        finalize_col0,
        combine_owners,
        wave_rows,
        wave_q,
        gamma0_pos,
        inv_group>(rows_this_core, inv_w_bits, eps_bits, slice_valid, group_index, masked_core);
}
