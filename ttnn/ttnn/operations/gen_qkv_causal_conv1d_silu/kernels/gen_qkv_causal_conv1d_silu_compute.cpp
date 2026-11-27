// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// gen_qkv_causal_conv1d_silu compute.
//
// Per block (op_design.md -> Block schedule):
//   untilize_rows_block   (TILE regime) compute_kernel_lib::untilize: input tiles -> row-major
//                         tile-rows (the reader copies them into the cb_rows slot).
//   windowed_tilize_block compute_kernel_lib::tilize with stick pages: per tile-row 4 windows,
//                         window p = sticks [32m + p, 32m + p + 32) = x[t - 3 + p]. The shift is a
//                         slide-by-pop on the slot (pops 1,1,1,29; 1,1,1,32 on the block's last
//                         tile-row) — exact data movement, fidelity-independent.
//   conv_silu_block       y = silu(sum_p window_p * rowbcast(tap tile, row 2p)) in DEST, one pack;
//                         the MACs run on the math thread, SiLU on the packer thread. Output tiles
//                         are acquired, packed and pushed in groups of out_group_tiles (the writer
//                         waits the same quantum): the writer stores group g while the packer runs
//                         SiLU on group g+1, and the math thread's MACs for group g+1 overlap the
//                         SiLU of group g in the other DEST half.
//
// RAW-LLK DEVIATION (conv_silu_block): eltwise_chain cannot express it.
//   (1) chain.inl:3397 — in a DEST-accumulating chain every DEST-only (SFPU) element runs on a
//       transient lane at slot >= 1, never on the sticky accumulated D0, so "accumulate 4 products,
//       then SiLU the sum" is not one chain;
//   (2) BinaryFpu::exec calls mul_tiles_bcast<bt>(...) without bcast_row_idx (chain.inl:1469), so the
//       packed tap tile (tap p in row 2p) is not addressable.
//   Two chains would need a pack + unpack of the pre-activation through an extra CB (rejected:
//   fold into the accumulator). Gap closer: a chain option to apply post-accumulation DEST-only
//   elements on the sticky slot, plus bcast_row_idx on BinaryFpu.
//   The block op below relies on ELWMUL accumulating into DEST (llk_math_eltwise_binary.h: the
//   standard ELWMUL path issues no ZEROACC; fidelity phases accumulate) and on DEST being zero at
//   acquire (cleared by the packer on release) — the same precondition every mul_tiles has.
//
// RAW-LLK DEVIATION (SiLU body): compute_kernel_lib::apply_activation_from_pack /
//   ActivationApplyHelper<SILU> dispatch only the library's fixed calculate_silu body
//   (ckernel_sfpu_silu.h: exp_21f + clamp + 3-op float->int + reciprocal with a masked Newton step,
//   26 SFPU instructions per row; one pass over conv1d-004's 96 tiles per core = 78.3 us of SFPU).
//   The helper has no slot for another SFPU body, so silu_from_pack below mirrors
//   apply_activation_from_pack (math->pack semaphore wait, packer DEST-offset flip, per-tile
//   _llk_math_eltwise_unary_sfpu_params_ sequence, SFPU-done stall) and dispatches either the library
//   body (byte-for-byte the helper's sequence) or, for a 16-bit DEST when the plan selects it
//   (Plan.silu_body), silu_bf16::rows: a hand-scheduled raw-TTI body that also bypasses the sfpi
//   compiler's schedule, because three Blackhole mechanisms sfpi does not use carry the speedup:
//   (1) SFPLOADMACRO multi-issue: per row, one macro loads x and schedules y = x * -log2(e) + 639 on the
//       MAD sub-unit; a second re-loads x and schedules x * r (MAD), the bf16 round into LReg16 (round
//       sub-unit) and the store (store sub-unit), so 5 of the row's ops ride on 2 issue slots;
//   (2) both clamps compare against constant LRegs (SFPSWAP never writes a constant LReg back; mod 9 =
//       VD gets the max), so 512 / 766 are not re-materialized before every swap; LCONST_neg1 (L11) is
//       borrowed for 766 and restored to -1 after each block, -1 becomes NEGATE_VC of LCONST_1;
//   (3) the quartic correction is refit with fp16 G3, G2, G1 (one SFPLOADI each; minimax relative error
//       1.140e-4 vs 1.112e-4 all-fp32) and G4 held in L7 for a whole DEST group.
//   36 cycles per row pair: 42.1 us per pass vs 61.5 us for the sfpi-compiled quartic, correctly rounded
//   for ~99.5% of inputs vs ~90.6% for the library body, never more than 1 bf16 ULP off, clamp extremes
//   exact. A 32-bit DEST always runs the library body. Gap closer: an apply_activation_from_pack overload
//   taking a caller SFPU functor, plus a load-macro bf16 SiLU body in the LLK SFPU library.

// Stage zones (MaybeDeviceZoneScope, PERMANENT — perf tournaments rank stages from them; free when the
// profiler is off). Per block (= one tile-row at the default block_rows): compute_rows_wait (unpack
// starved on the reader), compute_tilize (4 windows), compute_windows_wait, compute_conv_silu (MACs on
// math, SiLU + pack on the packer, incl. the packer's cb_output_tiles reservations); TILE regime:
// compute_untilize; once: compute_taps_wait.
// Measurement-only ablations (compiled out unless defined; wrong results by construction):
// GQKV_ABLATE_SILU, GQKV_SILU_REPEAT, GQKV_ABLATE_TILIZE (windows: CB sync only),
// GQKV_ABLATE_MAC (no products; DEST stays zero).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_activation_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

#ifdef TRISC_PACK
// ---- SiLU on the packer thread, 16-bit DEST: silu(x) = x / (1 + exp(-x)) ----
namespace silu_bf16 {
// exp(-x) = 2^y, y = -x * log2(e). ybias = y + 639 is clamped to [512, 766]; in that binade its fp32
// encoding has exponent 9 and mantissa (y + 127) * 2^14, so shifting the encoding left by 9 drops the
// exponent and leaves (y + 127) * 2^23 — the bit pattern z = 2^floor(y) * (1 + frac(y)) (Schraudolph).
// The correction g(m) = 2^(m - 1) / m, m = 1 + frac = setexp(z, 127), is a quartic minimax in relative
// error on [1, 2) with G3, G2, G1 fp16 (1.140e-4). Then 1 / (1 + z * g) by approx_recip + one Newton step
// r -= r * (d * r - 1) (d <= 2^127 is finite, so no NaN mask: approx_recip(d >= 2^126) = 0 -> r = 0),
// times x, rounded to bf16 (round-to-nearest-even; SFPSTORE into a 16-bit DEST truncates).
constexpr float NEG_LOG2E = -1.44269504088896341f;  // L12
constexpr float Y_LO = 512.0f;                      // L13: y + 127 >= 0 (exp(-x) -> 0 for x > 88.03)
constexpr uint32_t Y_HI_FP16 = 0x61FC;              // L11 = 766.0: z <= 2^127 stays finite
constexpr uint32_t Y_BIAS_FP16 = 0x60FE;            // 639.0 = 127 + 512 (pins ybias's binade: exponent 9)
constexpr float G0 = 2.1355555057525635f;           // L14
constexpr uint32_t G4_LO = 0x29c0, G4_HI = 0x3da8;  // 0.08211088180541992, L7 for a whole DEST group
constexpr uint32_t G3_FP16 = 0xb8c4;                // -0.595703125
constexpr uint32_t G2_FP16 = 0x3f1f;                // 1.7802734375
constexpr uint32_t G1_FP16 = 0xc0ce;                // -2.40234375

constexpr uint32_t L0 = 0, L1 = 1, L2 = 2, L3 = 3, L4 = 4, L5 = 5, L6 = 6, L7 = 7;
constexpr uint32_t LZERO = 9, LONE = 10, LYHI = 11, LNEGLOG2E = 12, LYLO = 13, LG0 = 14;
constexpr uint32_t AM = 7;       // ADDR_MOD_7: dest += 0 (SFPU init)
constexpr uint32_t AM_INC4 = 6;  // ADDR_MOD_6: dest += 4 (one row pair), set in init
constexpr uint32_t NEG_VA = 1, NEG_VC = 2;
constexpr uint32_t SWAP_MIN = 1, SWAP_MAX = 9;  // VD gets min / max; VC = a constant LReg is never written

// Load-macro sequence byte: [2:0] template (4..7) / 3 store, [5:3] delay (executes delay + 1 cycles after the
// macro; all delays count cycles), 0x40 result / store operand = LReg16, 0x80 VB <- the macro's VD.
constexpr uint32_t seq_byte(uint32_t sel, uint32_t delay, uint32_t flags) { return flags | (delay << 3) | sel; }
// BEG (macro 0): load x; MAD (T0) y = L12 * x + L4 at +1.
constexpr uint32_t SEQ_BEG = seq_byte(4, 0, 0x80) << 8;
// FIN a / b (macros 1, 2): load x; MUL (T1: r_a in L4 / T2: r_b in L5) at +1, RND (T3) -> LReg16 at +3 / +4,
// STORE LReg16 at +4 / +5 (b one cycle later, so the two never share LReg16).
constexpr uint32_t SEQ_FIN_A =
    (seq_byte(3, 3, 0x40) << 24) | (seq_byte(7, 2, 0x40) << 16) | (seq_byte(5, 0, 0x80) << 8);
constexpr uint32_t SEQ_FIN_B =
    (seq_byte(3, 4, 0x40) << 24) | (seq_byte(7, 3, 0x40) << 16) | (seq_byte(6, 0, 0x80) << 8);

inline void load_seq(uint32_t macro, uint32_t seq) {
    TTI_SFPLOADI(L0, sfpi::SFPLOADI_MOD0_LOWER, seq & 0xffff);
    TTI_SFPLOADI(L0, sfpi::SFPLOADI_MOD0_UPPER, seq >> 16);
    TTI_SFPCONFIG(0, 4 + macro, 0);
    TTI_SFPNOP;
}

inline void init() {
    ckernel::math::reset_counters(p_setrwc::SET_ABD_F);
    sfpi::vConstFloatPrgm0 = NEG_LOG2E;  // L12
    sfpi::vConstFloatPrgm1 = Y_LO;       // L13
    sfpi::vConstFloatPrgm2 = G0;         // L14
    TTI_SFPLOADI(L0, sfpi::SFPLOADI_MOD0_FLOATA, Y_HI_FP16);
    TTI_SFPCONFIG(0, LYHI, 0);  // L11 = 766 (restore_neg1 puts -1 back)
    TTI_SFPNOP;
    // Instruction templates via the backdoor (VD = 12 + i; DISABLE_BACKDOOR_LOAD is clear after the SFPU init).
    TTI_SFPMAD(LNEGLOG2E, 0, L4, 12, 0);  // T0: VD = L12 * VD + L4 (Y_BIAS)
    TTI_SFPMUL(L4, 0, LZERO, 13, 0);      // T1: VD = L4 (r_a) * VD
    TTI_SFPMUL(L5, 0, LZERO, 14, 0);      // T2: VD = L5 (r_b) * VD
    TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_NEAREST, 0, LZERO, 0, 15, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);  // T3
    TTI_SFPNOP;
    load_seq(0, SEQ_BEG);
    load_seq(1, SEQ_FIN_A);
    load_seq(2, SEQ_FIN_B);
    TTI_SFPCONFIG(0x060, 8, 1);  // Misc: FIN macros store with the load's Mod0; every delay counts cycles
    TTI_SFPNOP;
    ckernel::addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(ADDR_MOD_6);
}

inline void restore_neg1() {
    TTI_SFPLOADI(L0, sfpi::SFPLOADI_MOD0_FLOATB, 0xBF80);
    TTI_SFPCONFIG(0, LYHI, 0);
    TTI_SFPNOP;
}

// Once per DEST group, before the first tile: G4 into L7 (read by nothing else).
inline void group_begin() {
    TTI_SFPLOADI(L7, sfpi::SFPLOADI_MOD0_USHORT, G4_LO);
    TTI_SFPLOADI(L7, sfpi::SFPLOADI_MOD0_UPPER, G4_HI);
}

// Once per DEST group, after the last tile: let the last FIN's scheduled round / store drain.
inline void group_end() {
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// One face: ITERATIONS rows, two per step (a = dst_reg[0], b = dst_reg[1]). Cycle-exact schedule, no auto
// stalls: every consumer of a MAD result issues >= 2 cycles after it (a load-macro MAD has no auto stall, an
// SFPSWAP compares a MAD result only after a 1-cycle gap), and no regular MAD issues on a cycle a macro's
// MAD executes (BEG / FIN are each followed by a load-unit instruction). 32 instructions + 4 SFPSWAP
// stall cycles = 36 cycles per row pair.
template <int ITERATIONS>
inline void rows() {
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS; d += 2) {
        TTI_SFPLOADI(L4, sfpi::SFPLOADI_MOD0_FLOATA, Y_BIAS_FP16);  // FIN a's MUL read r_a from L4 a cycle ago
        TTI_SFPLOADMACRO((0 << 2) | L0, 0, AM, 0);                  // BEG a: L0 = x_a, ya at +1
        TTI_SFPLOADMACRO((0 << 2) | L1, 0, AM, 2);                  // BEG b: L1 = x_b, yb at +1
        TTI_SFPLOADI(L6, sfpi::SFPLOADI_MOD0_FLOATA, G3_FP16);      // the gap ya needs before its swap
        TTI_SFPSWAP(0, LYLO, L0, SWAP_MAX);                         // ya = max(ya, 512)
        TTI_SFPSWAP(0, LYLO, L1, SWAP_MAX);
        TTI_SFPSWAP(0, LYHI, L0, SWAP_MIN);  // ya = min(ya, 766)
        TTI_SFPSWAP(0, LYHI, L1, SWAP_MIN);
        TTI_SFPSHFT(9, L0, L0, 5);  // za = ya << 9
        TTI_SFPSHFT(9, L1, L1, 5);
        TTI_SFPSETEXP(127, L0, L2, 1);  // ma = setexp(za, 127)
        TTI_SFPSETEXP(127, L1, L3, 1);
        TTI_SFPMAD(L2, L7, L6, L4, 0);  // ga = ma * G4 + G3
        TTI_SFPMAD(L3, L7, L6, L5, 0);
        TTI_SFPLOADI(L6, sfpi::SFPLOADI_MOD0_FLOATA, G2_FP16);
        TTI_SFPMAD(L4, L2, L6, L4, 0);  // ga = ga * ma + G2
        TTI_SFPMAD(L5, L3, L6, L5, 0);
        TTI_SFPLOADI(L6, sfpi::SFPLOADI_MOD0_FLOATA, G1_FP16);
        TTI_SFPMAD(L4, L2, L6, L4, 0);  // ga = ga * ma + G1
        TTI_SFPMAD(L5, L3, L6, L5, 0);
        TTI_SFPMAD(L4, L2, LG0, L4, 0);  // ga = ga * ma + G0
        TTI_SFPMAD(L5, L3, LG0, L5, 0);
        TTI_SFPMAD(L0, L4, LONE, L0, 0);  // da = za * ga + 1
        TTI_SFPMAD(L1, L5, LONE, L1, 0);
        TTI_SFPARECIP(0, L0, L4, sfpi::SFPARECIP_MOD1_RECIP);  // ra = approx_recip(da)
        TTI_SFPARECIP(0, L1, L5, sfpi::SFPARECIP_MOD1_RECIP);
        TTI_SFPMAD(L0, L4, LONE, L0, NEG_VC);  // ea = da * ra - 1
        TTI_SFPMAD(L1, L5, LONE, L1, NEG_VC);
        TTI_SFPMAD(L4, L0, L4, L4, NEG_VA);  // ra = ra - ra * ea
        TTI_SFPMAD(L5, L1, L5, L5, NEG_VA);
        TTI_SFPLOADMACRO((1 << 2) | L2, 0, AM, 0);       // FIN a: x_a * ra -> bf16 -> dst_reg[0]
        TTI_SFPLOADMACRO((2 << 2) | L3, 0, AM_INC4, 2);  // FIN b: x_b * rb -> bf16 -> dst_reg[1]; dest += 4
    }
}
}  // namespace silu_bf16
#endif

namespace {

constexpr uint32_t cb_rows = 0;
constexpr uint32_t cb_tap_tiles = 1;
constexpr uint32_t cb_window_tiles = 2;
constexpr uint32_t cb_output_tiles = 3;
constexpr uint32_t cb_input_tiles = 4;
constexpr uint32_t cb_untilized_rows = 5;

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t TAP_COUNT = 4;
constexpr uint32_t HALO_ROWS = TAP_COUNT - 1;
constexpr uint32_t WINDOW_STICKS = TILE_DIM + HALO_ROWS;  // every window of a tile-row lies in these
constexpr uint32_t TAP_ROW_STRIDE = 2;                    // tap p lives in row 2p of the packed tap tile

// SiLU body selector (compute CT arg 4, Plan.silu_body in the program descriptor): 0 = the library
// calculate_silu, 1 = silu_bf16::rows. A 32-bit DEST always runs the library body.
constexpr uint32_t SILU_LIBRARY = 0;

template <uint32_t silu_variant>
constexpr bool use_bf16_silu() {
    return silu_variant != SILU_LIBRARY && !DST_ACCUM_MODE;
}

template <uint32_t silu_variant>
ALWI void silu_init() {
    if constexpr (use_bf16_silu<silu_variant>()) {
        PACK((::ckernel::llk_math_eltwise_unary_sfpu_init<::SfpuType::silu>(silu_bf16::init)));
    } else {
        compute_kernel_lib::ActivationInitHelper<KernelActivation::SILU>::init();
    }
}

// The bf16 body borrows LCONST_neg1 (L11) for 766: put -1 back once the block's SiLU work is done.
template <uint32_t silu_variant>
ALWI void silu_uninit() {
    if constexpr (use_bf16_silu<silu_variant>()) {
        PACK((silu_bf16::restore_neg1()));
    }
}

#ifdef TRISC_PACK
// SiLU on one DEST tile from the packer thread: _llk_math_eltwise_unary_sfpu_params_(body, tile, RC)
// with its per-tile STALLWAIT(STALL_SFPU, MATH) optional (see silu_from_pack).
template <bool bf16_body, bool tile_stall>
inline void silu_tile_from_pack(uint32_t tile) {
    ckernel::math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(tile);
    if constexpr (tile_stall) {
        TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
    }
    if constexpr (bf16_body) {
        _llk_math_eltwise_sfpu_apply_vector_mode_(silu_bf16::rows<8>, VectorMode::RC);
    } else {
        _llk_math_eltwise_sfpu_apply_vector_mode_(ckernel::sfpu::calculate_silu<DST_ACCUM_MODE, 8>, VectorMode::RC);
    }
    ckernel::math::clear_dst_reg_addr();
}
#endif

// apply_activation_from_pack<SILU> with a selectable body and per-tile stall (see RAW-LLK DEVIATION
// above): replaces tile_regs_wait() — math->pack semaphore wait, packer DEST-offset flip, SiLU on
// each of `tiles` DEST tiles, SFPU-done stall before pack.
// tile_stall: the library start (_llk_math_eltwise_sfpu_start_) stalls the SFPU until the FPU pipe
// is idle before EVERY tile. From the packer that wait is redundant — math posts MATH_PACK only after
// STALLWAIT(MATH | WAIT_SFPU) (set_math_semaphores), so the whole DEST half is written once the
// SEMWAIT returns — and it is costly: the FPU is then busy with the NEXT group's MACs in the other
// DEST half, so each tile's SiLU waits for a gap in that stream while math's own commit waits for
// the SFPU (SILU_TILE_STALL in the program descriptor).
template <uint32_t silu_variant, bool tile_stall>
ALWI void silu_from_pack(uint32_t tiles) {
#ifdef GQKV_ABLATE_SILU
    // Perf ablation only (test_perf_r3.py; wrong results): the handshake without any SFPU work.
    tile_regs_wait();
    return;
#endif
#ifdef GQKV_SILU_REPEAT
    // Perf measurement only (test_perf_r3.py; wrong results): the body runs GQKV_SILU_REPEAT times
    // per tile, so the SFPU cost of one pass is a device-time difference.
    constexpr uint32_t passes = GQKV_SILU_REPEAT;
#else
    constexpr uint32_t passes = 1;
#endif
    PACK(TTI_SEMWAIT(
        p_stall::STALL_TDMA | p_stall::STALL_CFG, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO));
    PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
    if constexpr (use_bf16_silu<silu_variant>()) {
        PACK((silu_bf16::group_begin()));
    }
    for (uint32_t pass = 0; pass < passes; ++pass) {
        for (uint32_t i = 0; i < tiles; ++i) {
            PACK((silu_tile_from_pack<use_bf16_silu<silu_variant>(), tile_stall>(i)));
        }
    }
    if constexpr (use_bf16_silu<silu_variant>()) {
        PACK((silu_bf16::group_end()));
    }
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
}

using compute_kernel_lib::tilize_config::InitUninitMode;
using compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode;
using compute_kernel_lib::tilize_config::WaitMode;

// One window: tilize 32 sticks from the slot's read pointer, then pop `pages` sticks.
template <uint32_t block_ch_tiles, InitUninitMode init_mode, ReconfigureRegisterDatatypeMode reconfig>
ALWI void tilize_window(DataflowBuffer& rows, DataflowBuffer& windows, uint32_t pages) {
#ifdef GQKV_ABLATE_TILIZE
    // Perf ablation only: the helper's CB sequence (reserve, push, pop) without the tilize.
    windows.reserve_back(block_ch_tiles);
    windows.push_back(block_ch_tiles);
    rows.pop_front(pages);
#else
    compute_kernel_lib::tilize<block_ch_tiles, cb_rows, cb_window_tiles, init_mode, WaitMode::NoWait, reconfig>(
        rows, windows, 1, pages);
#endif
}

// windowed_tilize_block: rows_this_block tile-rows x 4 windows x block_ch_tiles tiles, layout
// [m][p][c] in cb_window_tiles. One tilize init / uninit per block; one wait per tile-row.
template <uint32_t block_ch_tiles>
void windowed_tilize_block(DataflowBuffer& rows, DataflowBuffer& windows, uint32_t rows_this_block) {
    constexpr auto Keep = ReconfigureRegisterDatatypeMode::NoReconfigure;
    for (uint32_t m = 0; m < rows_this_block; ++m) {
        {
            MaybeDeviceZoneScope("compute_rows_wait");
            rows.wait_front(WINDOW_STICKS);
        }
        MaybeDeviceZoneScope("compute_tilize");
        // p = 0
        if (m == 0) {
            tilize_window<
                block_ch_tiles,
                InitUninitMode::InitOnly,
                ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(rows, windows, 1);
        } else {
            tilize_window<block_ch_tiles, InitUninitMode::Neither, Keep>(rows, windows, 1);
        }
        // p = 1, 2
        for (uint32_t p = 1; p + 1 < TAP_COUNT; ++p) {
            tilize_window<block_ch_tiles, InitUninitMode::Neither, Keep>(rows, windows, 1);
        }
        // p = 3: release the rest of the tile-row; the block's last tile-row also releases the
        // 3 trailing sticks, so the block pops exactly its slot (32 * rows + 3).
        if (m + 1 == rows_this_block) {
            tilize_window<block_ch_tiles, InitUninitMode::UninitOnly, Keep>(rows, windows, TILE_DIM);
        } else {
            tilize_window<block_ch_tiles, InitUninitMode::Neither, Keep>(rows, windows, TILE_DIM - HALO_ROWS);
        }
    }
}

// conv_silu_block: rows_this_block x block_ch_tiles output tiles. Output tile (m, c) =
// silu(sum_{p=0..3} window[m*4w + p*w + c] * rowbcast(tap_tile[c], row 2p)), the 4 products
// accumulated in one DEST slot in fixed order p = 0..3, SiLU applied in DEST, packed once.
// Each tile-row's tiles go out in groups of out_group_tiles (the last group of a row may be short);
// a row's groups sum to block_ch_tiles, so no reservation straddles the end of cb_output_tiles.
template <uint32_t block_ch_tiles, uint32_t out_group_tiles, uint32_t silu_variant, bool silu_tile_stall>
void conv_silu_block(DataflowBuffer& windows, DataflowBuffer& outputs, uint32_t rows_this_block) {
    static_assert(
        out_group_tiles >= 1 && out_group_tiles <= compute_kernel_lib::DEST_AUTO_LIMIT,
        "an output group is one DEST acquire");
    constexpr uint32_t dest_group = out_group_tiles;
    const uint32_t window_tiles = TAP_COUNT * block_ch_tiles * rows_this_block;
    {
        MaybeDeviceZoneScope("compute_windows_wait");
        windows.wait_front(window_tiles);
    }
    MaybeDeviceZoneScope("compute_conv_silu");

    reconfig_data_format(cb_window_tiles, cb_tap_tiles);
    pack_reconfig_data_format(cb_output_tiles);
    mul_bcast_rows_init(cb_window_tiles, cb_tap_tiles);
    // SiLU runs on the packer thread (TRISC2), overlapping the math thread's next group of MACs
    // (op_design.md Perf lamps: measured 197.7 -> 185.9 us on conv1d-004 vs SiLU on the math thread).
    silu_init<silu_variant>();

    for (uint32_t m = 0; m < rows_this_block; ++m) {
        const uint32_t row_base = m * TAP_COUNT * block_ch_tiles;
        for (uint32_t c0 = 0; c0 < block_ch_tiles; c0 += dest_group) {
            const uint32_t group = (block_ch_tiles - c0 < dest_group) ? (block_ch_tiles - c0) : dest_group;
            outputs.reserve_back(group);
            tile_regs_acquire();
            for (uint32_t i = 0; i < group; ++i) {
                const uint32_t c = c0 + i;
#ifndef GQKV_ABLATE_MAC
                for (uint32_t p = 0; p < TAP_COUNT; ++p) {
                    mul_tiles_bcast_rows(
                        cb_window_tiles, cb_tap_tiles, row_base + p * block_ch_tiles + c, c, i, TAP_ROW_STRIDE * p);
                }
#endif
            }
            tile_regs_commit();
            // Replaces tile_regs_wait(): math->pack handshake, SiLU on each tile in DEST from the
            // packer thread, SFPU-done stall before pack.
            silu_from_pack<silu_variant, silu_tile_stall>(group);
            for (uint32_t i = 0; i < group; ++i) {
                pack_tile(i, cb_output_tiles);
            }
            tile_regs_release();
            outputs.push_back(group);
        }
    }
    silu_uninit<silu_variant>();
    windows.pop_front(window_tiles);
}

// untilize_rows_block (TILE regime): (rows + extra) tile-rows of input tiles -> row-major.
template <uint32_t block_ch_tiles>
void untilize_rows_block(uint32_t tile_rows) {
    DataflowBuffer input_tiles(cb_input_tiles);
    DataflowBuffer untilized_rows(cb_untilized_rows);
    MaybeDeviceZoneScope("compute_untilize");
    compute_kernel_lib::untilize<block_ch_tiles, cb_input_tiles, cb_untilized_rows>(
        input_tiles, untilized_rows, tile_rows);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t block_ch_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_rows = get_compile_time_arg_val(1);
    constexpr bool tile_input = get_compile_time_arg_val(2) == 1;
    constexpr uint32_t out_group_tiles = get_compile_time_arg_val(3);  // output tiles per push
    constexpr uint32_t silu_variant = get_compile_time_arg_val(4);     // SiLU body (silu_from_pack)
    constexpr bool silu_tile_stall = get_compile_time_arg_val(5) == 1;  // per-tile SFPU-after-FPU stall

    const uint32_t core_seq_start = get_arg_val<uint32_t>(0);
    const uint32_t core_seq_tiles = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks_this_core = (core_seq_tiles + block_rows - 1) / block_rows;
    const uint32_t last_block_rows = core_seq_tiles - (num_blocks_this_core - 1) * block_rows;

    if constexpr (tile_input) {
        compute_kernel_hw_startup(cb_input_tiles, cb_input_tiles, cb_untilized_rows);
    } else {
        compute_kernel_hw_startup(cb_rows, cb_rows, cb_window_tiles);
    }

    DataflowBuffer rows(cb_rows);
    DataflowBuffer taps(cb_tap_tiles);
    DataflowBuffer windows(cb_window_tiles);
    DataflowBuffer outputs(cb_output_tiles);

    for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
        const uint32_t rows_this_block = (block_idx + 1 == num_blocks_this_core) ? last_block_rows : block_rows;
        if constexpr (tile_input) {
            // A core whose range starts at t0 > 0 untilizes one extra tile-row on its first block:
            // its rows 29..31 are that block's halo.
            const uint32_t extra_rows = (block_idx == 0 && core_seq_start > 0) ? 1 : 0;
            untilize_rows_block<block_ch_tiles>(rows_this_block + extra_rows);
        }
        windowed_tilize_block<block_ch_tiles>(rows, windows, rows_this_block);
        if (block_idx == 0) {
            MaybeDeviceZoneScope("compute_taps_wait");
            taps.wait_front(block_ch_tiles);  // first use: the tap fetch hid behind the first tilize
        }
        conv_silu_block<block_ch_tiles, out_group_tiles, silu_variant, silu_tile_stall>(
            windows, outputs, rows_this_block);
    }
    taps.pop_front(block_ch_tiles);
}
