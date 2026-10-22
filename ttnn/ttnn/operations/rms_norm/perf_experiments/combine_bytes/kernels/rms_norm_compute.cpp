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
    uint32_t OWNERS>
FORCE_INLINE void run(
    uint32_t rows_this_core,
    uint32_t inv_w_bits,
    uint32_t eps_bits,
    uint32_t slice_valid,
    uint32_t group_index,
    bool masked_core) {
    constexpr uint32_t Wp = NUM_W_CHUNKS * C;  // γ window (padded slice)
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

        // ---- combine_block (R3): this core's owned tile-rows [index*q, +owned) of the block — sum their G gathered
        // partials, finalize. q = ceil(R_b / OWNERS) is the uniform per-owner quantum. cb_gather is laid out
        // contributor-major — contributor j's partials of the owned rows sit contiguously at slots [j*q, j*q + owned)
        // so each contributor delivers them in ONE NoC write — i.e. a (G x q) row-major block whose column r holds
        // row r's G partials. REDUCE_COL with AccumulateViaAdd + Skip sums each column's G tiles elementwise (the
        // partials are already collapsed: column 0 valid); the row pitch q skips a ragged last owner's unused
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
                if constexpr (SHARDED) {
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
        combine_owners>(rows_this_core, inv_w_bits, eps_bits, slice_valid, group_index, masked_core);
}
