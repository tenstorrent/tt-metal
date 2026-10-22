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

// [scale_fusion experiment copy] SF_VARIANT (compute define) selects the normalize stage's implementation:
//   0 (default) the op's current two-pass chain: x*γ (Row bcast) -> pack cb_x_gamma -> unpack -> *inv_rms (Col bcast)
//   1 fused, γ first: per DEST window, FPU x*γ (Row bcast) into DEST, then a DEST-reuse ELWMUL (DEST -> srcA via
//     MOVD2A) against inv_rms unpacked with Col broadcast into srcB. Same ops in the same order as 0: x*γ is never
//     packed to L1. Raw LLK: the kernel_lib chain's DestReuseBinary has no broadcast operand (op_design.md lamp L6).
//   2 fused, inv first: FPU x*inv_rms (Col bcast) into DEST, then DEST-reuse ELWMUL against γ (Row bcast in srcB).
//     Different rounding order ((x*inv)*γ), not bit-exact vs 0.
//   5 fused, γ first, custom raw LLK phase 2 (sf_reuse_col_mul_tile / sf_unpack_col_reuse_tile): one STALLWAIT +
//     16 MOVD2A per TILE (whole tile into the 64-row srcA bank), explicit ELWMULs, MOP-free so both phase-1 MOPs
//     are programmed once per call.
//   6 = 5 + γ srcB unpack of row 0 only (x_end = 16 datums) in phase 1.
//   Diagnostics (numerics intentionally wrong, attribution only): 3 phase 1 only; 4 phase 1 + phase-2 inits;
//   7 phase 2 = MOVD2A + ZEROACC only; 8 phase 2 = ELWMUL only (no MOVD2A); 9 = MOVD2A + 8 ELWMUL (fidelity
//   phase 0 only); 10 = 8 ELWMUL, no MOVD2A.
// Measured (Blackhole p150b, 8192x1024 BLOCK [1024,128] 8x8, bf16, HiFi2, 16-bit DEST, device kernel ns):
//   0 15,699 | 1 21,250 | 2 21,368 | 5 20,430 | 6 20,497 | diag 3 13,403, 4 13,873, 7 17,233, 8 18,411
//   -> REGRESSION: the normalize stage is FPU-bound (HiFi2 ELWMUL ~48 cycles / tile); the cb_x_gamma pack + unpack
//   run on TRISC2 / TRISC0 in parallel and cost nothing at the wall, while DEST reuse adds MOVD2A + srcA handshakes
//   to the binding math thread and removes the γ(0) pass that shadowed the combine's gather waits.
#ifndef SF_VARIANT
#define SF_VARIANT 0
#endif
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

// ---- scale_fusion raw LLK (SF_VARIANT 5/6) ----
// Phase 2 of the fused scaling: DEST[i] (x*γ) * inv_rms (Col bcast) -> DEST[i], at HiFi2 (MATH_FIDELITY), with the
// stock DEST-reuse mechanics re-scheduled per TILE instead of per FACE:
//   stock (llk_math_eltwise_binary<..., DEST_TO_SRCA>): per face STALLWAIT(MATH|SRCA_VLD) + 4 MOVD2A + ZEROACC +
//     MOP; the unpacker publishes one dummy srcA DVALID per face (4 per tile) -> 4 pipeline drains per tile.
//   here: ONE STALLWAIT per tile, 16 MOVD2A (all 64 DEST rows of the tile into the 64-row srcA bank), 4 ZEROACC,
//     then 16 explicit ELWMULs walking srcA rows 0..63 (no MOP, so the phase-1 MOP stays programmed);
//     the unpacker publishes ONE dummy srcA DVALID per tile + the 2 inv_rms srcB faces (0, 2), issued directly
//     (no MOP, so the phase-1 unpack MOP stays programmed too).
//   ELWMUL address modes (ADDR_MOD_4..6; the eltwise-binary init owns 0..3):
//     X: srcA/srcB/dest +0, fidelity +1      (phase 0 of an 8-row block)
//     Y: srcA/srcB/dest +8, fidelity clear   (phase 1 of the block, advance)
//     Z: srcA/dest +8, srcB clear, fid clear (phase 1 of a face's second block: the next face re-reads srcB rows 0..15)
//   per face: X Y X Z — the same per-row phase order (p0 then p1 accumulated into the zeroed DEST) as the stock MOP.
#ifdef TRISC_MATH
inline void sf_reuse_col_addrmods() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}, .fidelity = {.incr = 1, .clr = 0}}.set(
        ADDR_MOD_4);
    addr_mod_t{.srca = {.incr = 8}, .srcb = {.incr = 8}, .dest = {.incr = 8}, .fidelity = {.incr = 0, .clr = 1}}.set(
        ADDR_MOD_5);
    addr_mod_t{
        .srca = {.incr = 8}, .srcb = {.incr = 0, .clr = 1}, .dest = {.incr = 8}, .fidelity = {.incr = 0, .clr = 1}}
        .set(ADDR_MOD_6);
}
template <bool FP32_DEST, int DIAG = 0>
inline void sf_reuse_col_mul_tile(const uint32_t dst_index) {
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
    if constexpr (DIAG == 2 || DIAG == 4) {
        // diagnostic: no MOVD2A (srcA holds the unpacker's zeroed dummy bank) — ELWMUL cost alone
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD);
    } else {
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH | p_stall::SRCA_VLD);
        TTI_MOVD2A(0, 0, ADDR_MOD_1, p_movd2a::MOV_4_ROWS, 0);
        TTI_MOVD2A(0, 4, ADDR_MOD_1, p_movd2a::MOV_4_ROWS, 4);
        TTI_MOVD2A(0, 8, ADDR_MOD_1, p_movd2a::MOV_4_ROWS, 8);
        TTI_MOVD2A(0, 12, ADDR_MOD_1, p_movd2a::MOV_4_ROWS, 12);
        TTI_MOVD2A(0, 16, ADDR_MOD_1, p_movd2a::MOV_4_ROWS, 16);
        TTI_MOVD2A(0, 20, ADDR_MOD_1, p_movd2a::MOV_4_ROWS, 20);
        TTI_MOVD2A(0, 24, ADDR_MOD_1, p_movd2a::MOV_4_ROWS, 24);
        TTI_MOVD2A(0, 28, ADDR_MOD_1, p_movd2a::MOV_4_ROWS, 28);
        TTI_MOVD2A(0, 32, ADDR_MOD_1, p_movd2a::MOV_4_ROWS, 32);
        TTI_MOVD2A(0, 36, ADDR_MOD_1, p_movd2a::MOV_4_ROWS, 36);
        TTI_MOVD2A(0, 40, ADDR_MOD_1, p_movd2a::MOV_4_ROWS, 40);
        TTI_MOVD2A(0, 44, ADDR_MOD_1, p_movd2a::MOV_4_ROWS, 44);
        TTI_MOVD2A(0, 48, ADDR_MOD_1, p_movd2a::MOV_4_ROWS, 48);
        TTI_MOVD2A(0, 52, ADDR_MOD_1, p_movd2a::MOV_4_ROWS, 52);
        TTI_MOVD2A(0, 56, ADDR_MOD_1, p_movd2a::MOV_4_ROWS, 56);
        TTI_MOVD2A(0, 60, ADDR_MOD_1, p_movd2a::MOV_4_ROWS, 60);
    }
    // ELWMUL accumulates into DEST: clear the tile's 4 faces (absolute 16-row block index within the bank; with the
    // RWC at the tile start the bank-select sum stays < 512, see tt-metal#53693 in llk_math_eltwise_binary.h).
    constexpr uint32_t tiles_per_bank = FP32_DEST ? 4 : 8;
    const uint32_t f0 = (dst_index & (tiles_per_bank - 1)) << 2;
    TT_ZEROACC(p_zeroacc::CLR_16, FP32_DEST, 0, ADDR_MOD_1, f0 + 0);
    TT_ZEROACC(p_zeroacc::CLR_16, FP32_DEST, 0, ADDR_MOD_1, f0 + 1);
    TT_ZEROACC(p_zeroacc::CLR_16, FP32_DEST, 0, ADDR_MOD_1, f0 + 2);
    TT_ZEROACC(p_zeroacc::CLR_16, FP32_DEST, 0, ADDR_MOD_1, f0 + 3);
    constexpr uint32_t BC = p_elwise::SRCB_BCAST_COL;
    if constexpr (DIAG == 1) {
        // diagnostic: MOVD2A + ZEROACC only — release the srcB faces and the srcA bank without any ELWMUL
        TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, 0);
        TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
        return;
    }
    if constexpr (DIAG == 3 || DIAG == 4) {
        // diagnostic (precision NOT kept): fidelity phase 0 only — 8 ELWMULs per tile instead of 16
        TTI_ELWMUL(0, 0, BC, ADDR_MOD_5, 0);
        TTI_ELWMUL(0, 0, BC, ADDR_MOD_6, 0);
        TTI_ELWMUL(0, 0, BC, ADDR_MOD_5, 0);
        TTI_ELWMUL(p_setrwc::CLR_B, 0, BC, ADDR_MOD_6, 0);
        TTI_ELWMUL(0, 0, BC, ADDR_MOD_5, 0);
        TTI_ELWMUL(0, 0, BC, ADDR_MOD_6, 0);
        TTI_ELWMUL(0, 0, BC, ADDR_MOD_5, 0);
        TTI_ELWMUL(p_setrwc::CLR_AB, 0, BC, ADDR_MOD_6, 0);
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
        return;
    }
    // face 0 (srcB = inv face 0)
    TTI_ELWMUL(0, 0, BC, ADDR_MOD_4, 0);
    TTI_ELWMUL(0, 0, BC, ADDR_MOD_5, 0);
    TTI_ELWMUL(0, 0, BC, ADDR_MOD_4, 0);
    TTI_ELWMUL(0, 0, BC, ADDR_MOD_6, 0);
    // face 1 (same srcB face), then release srcB
    TTI_ELWMUL(0, 0, BC, ADDR_MOD_4, 0);
    TTI_ELWMUL(0, 0, BC, ADDR_MOD_5, 0);
    TTI_ELWMUL(0, 0, BC, ADDR_MOD_4, 0);
    TTI_ELWMUL(p_setrwc::CLR_B, 0, BC, ADDR_MOD_6, 0);
    // face 2 (srcB = inv face 2)
    TTI_ELWMUL(0, 0, BC, ADDR_MOD_4, 0);
    TTI_ELWMUL(0, 0, BC, ADDR_MOD_5, 0);
    TTI_ELWMUL(0, 0, BC, ADDR_MOD_4, 0);
    TTI_ELWMUL(0, 0, BC, ADDR_MOD_6, 0);
    // face 3, then release srcA (the tile's bank) and srcB
    TTI_ELWMUL(0, 0, BC, ADDR_MOD_4, 0);
    TTI_ELWMUL(0, 0, BC, ADDR_MOD_5, 0);
    TTI_ELWMUL(0, 0, BC, ADDR_MOD_4, 0);
    TTI_ELWMUL(p_setrwc::CLR_AB, 0, BC, ADDR_MOD_6, 0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
}
#endif
#ifdef TRISC_UNPACK
// One dummy srcA DVALID (DEST is moved into srcA by the math thread) + inv_rms faces 0 and 2 into srcB (Col bcast).
inline void sf_unpack_col_reuse_tile(const uint32_t cb, const uint32_t tile_index) {
    const uint32_t operand_id = get_operand_id(cb);
    const uint32_t address = get_local_cb_interface(operand_id).fifo_rd_ptr - 1 +
                             get_local_cb_interface(operand_id).fifo_page_size * tile_index;
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    volatile uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    wait_for_next_context(2);
    cfg[(unp_cfg_context == 0) ? THCON_SEC1_REG3_Base_address_ADDR32 : THCON_SEC1_REG3_Base_cntx1_address_ADDR32] =
        address;
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    TTI_UNPACR_NOP(
        SrcA, 0, 0, p_unpacr_nop::SET_DVALID, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
    TTI_UNPACR(SrcB, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 2, 0b0001);
    TTI_UNPACR(SrcB, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
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

    // ---- scale_fusion: γ and inv_rms scalings in ONE DEST window per block of tiles ----
    constexpr bool FUSED = HAS_GAMMA && (SF_VARIANT != 0);
    auto gamma_wait = [&](uint32_t c) {
        if constexpr (HAS_GAMMA) {
            MaybeDeviceZoneScope("cmp_wait_gamma");
            cb_wait_front(cb_gamma_tiles, GAMMA_RESIDENT ? (c + 1) * C : C);
        }
    };
    auto gamma_release = [&]() {
        if constexpr (HAS_GAMMA && !GAMMA_RESIDENT) {
            cb_pop_front(cb_gamma_tiles, C);
        }
    };
    // Tiles (r, j) of rows [row0, row0 + n) x chunk c's columns [0, cw), flattened row-major, B per DEST window.
    //   x      cb_x_tiles[(row0 + r) * WS + c * C + j]     (caller-managed, strided)
    //   γ      cb_gamma_tiles[gbase + j]                    (row-0 broadcast tiles; resident: gbase = c * C)
    //   inv    cb_inv_rms[inv0 + r]                         (column-0 broadcast)
    //   out    SHARDED: cb_output[(row0 + r) * WS + c * C + j] (out-of-order into the reserved shard)
    //          else:    cb_scaled, one tile-row (C pages) reserved / pushed at a time (the writer's quantum)
    auto fused_scale = [&](uint32_t c, uint32_t row0, uint32_t n, uint32_t cw, uint32_t inv0) {
        constexpr uint32_t B = SCALE_BLOCK < ckl::DEST_AUTO_LIMIT ? SCALE_BLOCK : ckl::DEST_AUTO_LIMIT;
        const uint32_t gbase = GAMMA_RESIDENT ? c * C : 0;
        const uint32_t total = n * cw;
        pack_reconfig_data_format(cb_scaled);
        uint32_t pr = 0, pj = 0;  // pack cursor (row, col)
        uint32_t ur = 0, uj = 0;  // unpack/math cursor
        if constexpr (SF_VARIANT >= 5 && SF_VARIANT <= 10) {
            // phase-1 MOPs programmed ONCE per call (phase 2 is MOP-free on both threads)
            reconfig_data_format(cb_x_tiles, cb_gamma_tiles);
            MATH((llk_math_eltwise_binary_init<
                  ckernel::EltwiseBinaryType::ELWMUL,
                  ckernel::BroadcastType::ROW,
                  MATH_FIDELITY>(cb_x_tiles, cb_gamma_tiles, 0)));
            MATH((sf_reuse_col_addrmods()));
            UNPACK((llk_unpack_AB_init<ckernel::BroadcastType::ROW>(cb_x_tiles, cb_gamma_tiles)));
            for (uint32_t t0 = 0; t0 < total; t0 += B) {
                const uint32_t nb = (total - t0) < B ? total - t0 : B;
                tile_regs_acquire();
                if (t0 > 0) {
                    reconfig_data_format_srcb(cb_inv_rms, cb_gamma_tiles);
                }
                if constexpr (SF_VARIANT == 6) {
                    // Row broadcast reads srcB row 0 only: unpack 1 row (16 datums) of each γ face, not 256.
                    UNPACK((TTI_SETADCXX(p_setadc::UNP_B, FACE_C_DIM - 1, 0x0)));
                }
                uint32_t r = ur, j = uj;
                for (uint32_t i = 0; i < nb; ++i) {
                    mul_tiles_bcast<ckernel::BroadcastType::ROW>(
                        cb_x_tiles, cb_gamma_tiles, (row0 + r) * WS + c * C + j, gbase + j, i);
                    if (++j == cw) {
                        j = 0;
                        ++r;
                    }
                }
                reconfig_data_format_srcb(cb_gamma_tiles, cb_inv_rms);
                if constexpr (SF_VARIANT == 6) {
                    UNPACK((TTI_SETADCXX(p_setadc::UNP_B, FACE_R_DIM * FACE_C_DIM - 1, 0x0)));
                }
                r = ur;
                j = uj;
                for (uint32_t i = 0; i < nb; ++i) {
                    UNPACK((sf_unpack_col_reuse_tile(cb_inv_rms, inv0 + r)));
                    MATH((sf_reuse_col_mul_tile<DST_ACCUM_MODE, (SF_VARIANT >= 7 ? SF_VARIANT - 6 : 0)>(i)));
                    if (++j == cw) {
                        j = 0;
                        ++r;
                    }
                }
                ur = r;
                uj = j;
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < nb; ++i) {
                    if constexpr (SHARDED) {
                        pack_tile<true>(i, cb_output, (row0 + pr) * WS + c * C + pj);
                    } else {
                        if (pj == 0) {
                            cb_reserve_back(cb_scaled, C);
                        }
                        pack_tile<true>(i, cb_scaled, pj);
                        if (pj + 1 == cw) {
                            cb_push_back(cb_scaled, C);
                        }
                    }
                    if (++pj == cw) {
                        pj = 0;
                        ++pr;
                    }
                }
                tile_regs_release();
            }
            return;
        }
        if constexpr (SF_VARIANT == 1) {
            reconfig_data_format(cb_x_tiles, cb_gamma_tiles);
        } else {
            reconfig_data_format(cb_x_tiles, cb_inv_rms);
        }
        for (uint32_t t0 = 0; t0 < total; t0 += B) {
            const uint32_t nb = (total - t0) < B ? total - t0 : B;
            tile_regs_acquire();
            if constexpr (SF_VARIANT == 1 || SF_VARIANT == 3 || SF_VARIANT == 4) {
                // phase 1: x * γ (Row bcast) -> DEST[i]
                if (t0 > 0 && SF_VARIANT != 3) {
                    reconfig_data_format_srcb(cb_inv_rms, cb_gamma_tiles);
                }
                MATH((llk_math_eltwise_binary_init<
                      ckernel::EltwiseBinaryType::ELWMUL,
                      ckernel::BroadcastType::ROW,
                      MATH_FIDELITY>(cb_x_tiles, cb_gamma_tiles, 0)));
                UNPACK((llk_unpack_AB_init<ckernel::BroadcastType::ROW>(cb_x_tiles, cb_gamma_tiles)));
                uint32_t r = ur, j = uj;
                for (uint32_t i = 0; i < nb; ++i) {
                    mul_tiles_bcast<ckernel::BroadcastType::ROW>(
                        cb_x_tiles, cb_gamma_tiles, (row0 + r) * WS + c * C + j, gbase + j, i);
                    if (++j == cw) {
                        j = 0;
                        ++r;
                    }
                }
                // phase 2: DEST[i] (-> srcA, MOVD2A) * inv_rms (Col bcast, srcB) -> DEST[i]
                // (diagnostics, wrong numerics: SF_VARIANT 3 = phase 1 only; 4 = phase 1 + phase-2 inits, no exec)
                if constexpr (SF_VARIANT != 3) {
                    reconfig_data_format_srcb(cb_gamma_tiles, cb_inv_rms);
                    MATH((llk_math_eltwise_binary_init<
                          ckernel::EltwiseBinaryType::ELWMUL,
                          ckernel::BroadcastType::COL,
                          MATH_FIDELITY,
                          ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_inv_rms, cb_inv_rms, 0)));
                    UNPACK((llk_unpack_A_init<
                            ckernel::BroadcastType::COL,
                            true,
                            ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCA>(0, 0, cb_inv_rms)));
                }
                r = ur;
                j = uj;
                for (uint32_t i = 0; i < nb && SF_VARIANT == 1; ++i) {
                    UNPACK((llk_unpack_A<
                            ckernel::BroadcastType::COL,
                            true,
                            ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_inv_rms, inv0 + r)));
                    MATH((llk_math_eltwise_binary<
                          ckernel::EltwiseBinaryType::ELWMUL,
                          ckernel::BroadcastType::COL,
                          DST_ACCUM_MODE,
                          MATH_FIDELITY,
                          ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_inv_rms, cb_inv_rms, i, true)));
                    if (++j == cw) {
                        j = 0;
                        ++r;
                    }
                }
                if constexpr (SF_VARIANT != 1) {
                    for (uint32_t i = 0; i < nb; ++i) {
                        if (++j == cw) {
                            j = 0;
                            ++r;
                        }
                    }
                }
                ur = r;
                uj = j;
            } else {
                // phase 1: x * inv_rms (Col bcast) -> DEST[i]
                if (t0 > 0) {
                    reconfig_data_format_srcb(cb_gamma_tiles, cb_inv_rms);
                }
                MATH((llk_math_eltwise_binary_init<
                      ckernel::EltwiseBinaryType::ELWMUL,
                      ckernel::BroadcastType::COL,
                      MATH_FIDELITY>(cb_x_tiles, cb_inv_rms, 0)));
                UNPACK((llk_unpack_AB_init<ckernel::BroadcastType::COL>(cb_x_tiles, cb_inv_rms)));
                uint32_t r = ur, j = uj;
                for (uint32_t i = 0; i < nb; ++i) {
                    mul_tiles_bcast<ckernel::BroadcastType::COL>(
                        cb_x_tiles, cb_inv_rms, (row0 + r) * WS + c * C + j, inv0 + r, i);
                    if (++j == cw) {
                        j = 0;
                        ++r;
                    }
                }
                // phase 2: DEST[i] (-> srcA) * γ (Row bcast, srcB) -> DEST[i]
                reconfig_data_format_srcb(cb_inv_rms, cb_gamma_tiles);
                MATH((llk_math_eltwise_binary_init<
                      ckernel::EltwiseBinaryType::ELWMUL,
                      ckernel::BroadcastType::ROW,
                      MATH_FIDELITY,
                      ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_gamma_tiles, cb_gamma_tiles, 0)));
                UNPACK((llk_unpack_A_init<
                        ckernel::BroadcastType::ROW,
                        true,
                        ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCA>(0, 0, cb_gamma_tiles)));
                r = ur;
                j = uj;
                for (uint32_t i = 0; i < nb; ++i) {
                    UNPACK((llk_unpack_A<
                            ckernel::BroadcastType::ROW,
                            true,
                            ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_gamma_tiles, gbase + j)));
                    MATH((llk_math_eltwise_binary<
                          ckernel::EltwiseBinaryType::ELWMUL,
                          ckernel::BroadcastType::ROW,
                          DST_ACCUM_MODE,
                          MATH_FIDELITY,
                          ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_gamma_tiles, cb_gamma_tiles, i, true)));
                    if (++j == cw) {
                        j = 0;
                        ++r;
                    }
                }
                ur = r;
                uj = j;
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < nb; ++i) {
                if constexpr (SHARDED) {
                    pack_tile<true>(i, cb_output, (row0 + pr) * WS + c * C + pj);
                } else {
                    if (pj == 0) {
                        cb_reserve_back(cb_scaled, C);
                    }
                    pack_tile<true>(i, cb_scaled, pj);
                    if (pj + 1 == cw) {
                        cb_push_back(cb_scaled, C);
                    }
                }
                if (++pj == cw) {
                    pj = 0;
                    ++pr;
                }
            }
            tile_regs_release();
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
                if constexpr (HAS_GAMMA && !FUSED) {
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
                if constexpr (HAS_GAMMA && !FUSED) {
                    if (c > 0) {
                        scale_by_gamma(c, rows_in_block);
                    }
                }
                if constexpr (FUSED) {
                    gamma_wait(c);
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
                    if constexpr (FUSED) {
                        MaybeDeviceZoneScope("cmp_scale");
                        fused_scale(c, row0, n, cw, (NUM_W_CHUNKS == 1) ? 0 : row0);
                        if constexpr (NUM_W_CHUNKS == 1) {
                            cb_pop_front(cb_inv_rms, n);
                        }
                        continue;
                    }
                    MaybeDeviceZoneScope("cmp_inv");
                    const ckl::StridedTileRange out_range{row0 * WS + c * C, WS};
                    const bool direct = (NUM_W_CHUNKS == 1) || c > 0;
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
                    if constexpr (NUM_W_CHUNKS == 1) {
                        cb_pop_front(cb_inv_rms, n);
                    }
                }
                if constexpr (FUSED) {
                    gamma_release();
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
            constexpr bool gamma0_before_combine = COMBINE && HAS_GAMMA && OWNERS > 1 && !FUSED;
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
                if constexpr (HAS_GAMMA && !FUSED) {
                    // Reduce-scatter: chunk 0 already ran ahead of the combine.
                    if (!(gamma0_before_combine && c == 0)) {
                        scale_by_gamma(c, rows_in_block);
                    }
                }
                if constexpr (FUSED) {
                    gamma_wait(c);
                }
                if (c == 0) {
                    // inv_rms of the block (R1: produced above on this thread; R3: the multicast from the root).
                    MaybeDeviceZoneScope("cmp_wait_inv");
                    cb_wait_front(cb_inv_rms, rows_in_block);
                }
                if constexpr (FUSED) {
                    {
                        MaybeDeviceZoneScope("cmp_scale");
                        fused_scale(c, 0, rows_in_block, cw, 0);
                    }
                    gamma_release();
                } else {
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
