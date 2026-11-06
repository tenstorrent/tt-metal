// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// nomic_sdpa — compute. Per block of block_rows (<= Q) query tile-rows of one (b, h) pair, K and V of the pair
// resident (op_design.md § Block Operation Realization):
//   qk_block         scores = Q K^T                    matmul_block (transpose in1)   cb_query, cb_key_t -> cb_scores
//   row_max_block    m = rowmax(scores)                reduce_block_max_row (raw LLK)  cb_scores (kept) -> cb_row_stats
//   probs_block      P = exp(scale (scores - m))       eltwise_chain, IN PLACE         cb_scores -> cb_scores
//   pv_norm_row      [O | l] = P_row [V | 1] in DEST,  raw matmul_block LLK          cb_scores | cb_probs,
//                    then O / l in DEST, O packed      + pack-thread SFPU (normalize)  cb_value_aug -> cb_out
//                    (one row a DEST window; l never leaves DEST — see "normalize in DEST" below)
//
// Ring discipline (op_design.md § Circular Buffers): every block-consumed CB is advanced by its FULL nominal
// capacity each block (each segment for K / V). A ragged block (block_rows < Q) narrows only the helper shapes and
// closes each ring with pad pushes / pad pops of the remainder, so rd/wr pointers return to the CB base every block
// and no block straddles a ring wrap (helpers index a block linearly from rd_ptr). cb_scores holds the scores, then
// P in the same slots: after the qk pad push it is full, and the in-place chain pops tile i before it reserves tile i.
//
// Row-group schedule (exp_overlap, Refinement 3 / op_design.md perf lamp L2) — the default under the approximate exp.
// The block's rows run as groups of G = row_group_rows rows (a schedule inside the block: cb_query and cb_out stay
// whole-block). The exp runs on the PACK thread; the MATH thread only issues each probs window's s - m and moves on,
// and with half-sync DEST the math thread runs one window ahead of the pack thread, so each window's pack overlaps
// the NEXT window's math. The step's window stream (Perf 2, perf_experiments/rowmax_interleave):
//   prologue    qk(0), row_max(0)
//   step t      for each window k: qk_window(t+1, k) ; probs_window(t, k)
//               then pv_window(t-1) (one [O | l] row a window, normalized in DEST), then row_max(t+1)
// (three groups in flight: t+1's scores, t's P, t-1's [O | l]). Why this order: each probs window's exp + packs
// overlap the next Q K^T window's math, and the last one the long P V matmul; row_max(t+1) follows P V, so its
// unpack reads scores packed windows ago (no wait on the last Q K^T pack) and its math overlaps P V's normalize +
// pack; and the next step opens with a Q K^T window, which needs no m, so its math covers the m(t+1) pack. The
// previous order (probs first, P V then Q K^T as partners, row_max after the step's last Q K^T) put the row max on
// the critical path between steps: its unpack waited for the last Q K^T pack, and the next step's first probs
// window waited for m (measured 8x512 L1 compute floor 53.3 -> 48.8 us, whole op 70.1 -> 65.4 us; unpack row-max
// zone 5.8 -> 1.3 us a core; bit-identical).
// Scores and P are 2-group rings, the row stats a 1-group ring; a ragged group (fewer than G rows) pads each back to
// its nominal G-row extent, exactly as a ragged block pads the block-sized rings. cb_row_stats holds only m: it drains
// in the probs windows before m(t+1) lands. (reduce and the windows read their block from the ring front.)
//
// probs_window is a custom block operation (raw LLK), not an eltwise_chain: the chain runs every SFPU element on the
// MATH thread inside its compute phase, and has no element that runs between the pack thread's tile_regs_wait and
// its pack (kernel_lib gap: a pack-thread SFPU element / pre-pack hook on PackTile). matmul_block's packer-thread
// Activation slot is the only kernel_lib hook of that kind, and the sub is not a matmul. The window is the
// incumbent's sub_exp_block_bcast_cols (compute_streaming.hpp:581-668) on the op's CBs: s - m on the FPU, the
// unclamped fast exp (scale folded into its init) on the pack thread's SFPU, P packed with the ReLU that zeroes the
// arguments below the fast exp's range. Every SFPU op of this schedule (the exp, the normalize) runs on the pack
// thread, in its program order: no math-thread SFPU op, no fence. The exp init is issued once a kernel.
//
// Normalize in DEST (perf lamp L5, Perf 1 normalize_in_dest): the ones tile of [V | 1] is all-ones (reader), so the
// P V window's third DEST tile holds the row sum l replicated across all 32 columns and O / l is a lane-wise op
// between DEST tiles. After the window's math commit the PACK thread (the same thread as the exp, so no SFPU race
// and no fence in the row-group schedule) runs normalize_row_sfpu — one 1/l per SFPU row-quad (SFPARECIP seed + two
// Newton terms, fp32 in DEST) shared by the 8 O vectors of those rows — then packs O0, O1 straight to cb_out. It
// uses LREG0..5, the fixed constants and ADDR_MOD_7 only, so the fast exp's LREG12..14 / macro / replay state
// survives (no exp re-init). Measured (8x512 L1, compute floor) 75.2 -> 72.0 us; real rel-RMS 0.0255 -> 0.0216.
// Raw LLK there: ckl::matmul_block packs every DEST tile of its subblock to one CB (no hook to pack a subset — the
// helper + a custom packer Activation packing [O0 O1 l] measured 73.1 us, the extra l pack on the binding pack
// thread), and kernel_lib has no fused DEST-tile recip-and-multiply element.
//
// The ones column at its live fidelity phases (Perf 2, perf_experiments/ones_column_l): a third of the P V window's
// MVMULs multiply the ones tile, but at fidelity phase f an MVMUL multiplies only a slice of each operand's mantissa —
// odd phases SrcA's low bits, phases >= 2 SrcB's low bits — and the ones tile is in SrcA (matmul maps in1 -> SrcA) as
// bf16 1.0, whose mantissa has no low bits. So its odd phases add exact zeros: the ones tile runs phases {0} under
// HiFi2 and {0, 2} under HiFi3 / HiFi4 (phase 2 is 1.0 x P's low bits: measured not droppable, phase 0 alone changes
// l), the V tiles every phase. Bit-identical to the full MOP in every config (HiFi2/3/4, full-sync, bf16 DEST + exact
// exp). Measured 8x512 L1 compute floor HiFi2 48.8 -> 46.2 us (whole op 66.2 -> 63.8 us), HiFi3 59.9 -> 57.9 us,
// HiFi4 73.0 -> 68.5 us. Past the trimmed phases the k-step is unpack-bound (the ones tile's unpack), so an even
// shorter ones product (8 MVMULs into half the l tile) measured no faster.
// Raw LLK there (pv_ones::k_step): matmul_block / ckl::matmul_block run every in1 tile of the block through the one
// MOP at the configured fidelity — neither has a per-tile fidelity (or a per-tile phase subset), so "this operand is
// exact at phase 0" is inexpressible through them (a capability gap, not ergonomics). The unpack half is the stock
// llk_unpack_AB_matmul; the math half replays the MOP's own recorded tile body.
//
// One pack MOP a window (Perf 2, perf_experiments/compute_floor_r2): the pack thread binds the Q K^T / probs window
// pairs (slack probe: a 200-cycle delay a probs window costs ~85% of it on the pack thread, ~6% on the math thread),
// and pack_tile, per tile, programs the packer's L1 address (STALLWAIT on THCON), runs a one-tile MOP whose last PACR
// closes the tile and drains the packer, then resets the ADC Z counters. A window's tiles sit contiguous in DEST and
// in the CB ring (windows never straddle the ring end), so nomic_pack::pack_window packs them with ONE pack MOP run of
// up to 4 tiles (the pack MOP's num_tiles; 4 is the fp32-DEST maximum): one address program, one close. The MOP is
// re-programmed only when the tile count changes — qk_w for the Q K^T / probs windows, 1 for the m and O packs (two
// re-programs a row), so it runs only when a score row has >= 2 windows (pack_runs): with one window a row the two
// re-programs cost more than the 6 tile packs they save (measured 32x128 L1 floor 18.5 -> 19.0 us). Bit-identical.
// Measured compute floor (L1, 176 KB reserved): 8x512 46.6 -> 45.6 us, 4x512 26.1 -> 25.5, 8x512 full-sync 61.4 ->
// 59.4; flat at 16x256 / 8x384 (2 / 3 windows a row).
// Raw LLK there: pack_tile / pack_block issue one MOP a tile (pack_block is llk_matmul_pack, a pack_tile loop on
// Blackhole) and ckl::PackTile has no multi-tile pack — no API packs a DEST run with one MOP (a capability gap).
// The pack-thread exp init is issued once a kernel (it was once a group): no pack-thread op clobbers its replay buffer,
// macro registers or LREG12..14 (the normalize uses LREG0..5 and ADDR_MOD_7 only; packs use no replay).
//
// Every page is Float16_b and DEST is bf16, so every helper runs with data-format reconfig off.
// Raw CB calls below are only the ring pads, the K/V release and the raw P V window — no helper covers those.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/scalar.hpp"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/pack.h"
#include "api/compute/reduce_custom.h"
#include "nomic_sdpa_schedule.hpp"
// Permanent per-stage device zones (Perf 1; free when the profiler is off). Waits have their own zones
// (compute_wait_qk, compute_wait_v: read-only waits on the unpack thread) so the payload zones are not starvation.
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace ckl = compute_kernel_lib;

#if defined(TRISC_PACK)
namespace nomic_exp {
// exp_window<w>: the pack-thread fast exp of a w-tile probs window with ONE SFPU start / done (Perf 1,
// perf_experiments/packthread_exp_impl). Raw LLK: exp_packthread_tile has no multi-tile form, so it pays a start /
// done (+ drain) per tile. Same instruction body as exp_packthread_tile<true, false, InputClamping::None, 32>: the
// 32-instruction LM / SHFT2 body that exp_packthread_tile_init records in the replay buffer walks DEST by +2 per
// SFPLOADMACRO, and the window's tiles are contiguous in DEST, so 2 replays a tile chain tile to tile; one drain at
// the end. Bit-identical outputs. Measured micro-bench 83.2 -> 76.0 ns a tile; 8x512 compute floor 75.2 -> 73.9 us
// on the pre-Perf-1 kernel. Requires exp_packthread_tile_init<true, scale, InputClamping::None> since the last SFPU
// init that touched the replay buffer / macro registers (the normalize in DEST does not).
template <uint32_t w>
ALWI void exp_window() {
    _llk_math_eltwise_sfpu_start_(0);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_7);
#pragma GCC unroll 8
    for (uint32_t j = 0; j < 2 * w; ++j) {
        lltt::replay(0, 32);
    }
    TTI_SFPNOP;  // drain: SHFT2 of the last two vectors (as calculate_exponential<approx, 32 iterations>)
    TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG14, p_sfpu::LREG4, 5);
    TTI_SFPNOP;
    TTI_SFPSHFT2(p_sfpu::LREG3, p_sfpu::LREG14, p_sfpu::LREG4, 5);
    TTI_SFPNOP;
    TTI_SFPNOP;
    _llk_math_eltwise_sfpu_done_();
}
}  // namespace nomic_exp
#endif

#if defined(TRISC_MATH)
// pv_ones::k_step<ct_v, ct>: one k-step of the P V window on the MATH thread — matmul_block's math half
// (matmul_block_math_dynamic_throttle(in0, in1, 0, false, ct, 1), its dynamic-throttle branches included), except that
// the ones tile (the k-step's last in1 tile, DEST tile ct_v) runs only its live fidelity phases (file header "The ones
// column at its live fidelity phases").
namespace pv_ones {
constexpr uint32_t kPhases = static_cast<uint32_t>(MATH_FIDELITY);  // MOP phases a tile: HiFi2 2, HiFi3 3, HiFi4 4
static_assert(kPhases >= 2, "LoFi is refused by the host");
static_assert(matmul_throttle_drives_phases<MATH_FIDELITY, MM_THROTTLE_MAX>(), "throttled MOP: one call a phase");
// The full 32x32 tile body the matmul init records in the replay buffer (high fidelity): 16 MVMULs, the last on
// ADDR_MOD_5 (A / B / D counters reset, phase + 1); the MOP's end op (SETRWC CLR_A, SET_ABD_F) follows the phases.
constexpr uint32_t kBody = 16;

// ADDR_MOD_3 (the Blackhole matmul programs 0, 1, 2, 4, 5, 6): ADDR_MOD_5 with phase + 2 (skips odd phase 1).
// Re-set after every matmul init: other ops program ADDR_MOD_3.
inline void addrmod_init() {
    addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 1},
        .srcb = {.incr = 0, .clr = 1, .cr = 1},
        .dest = {.incr = 0, .clr = 1, .cr = 1},
        .fidelity = {.incr = 2, .clr = 0},
    }
        .set(ADDR_MOD_3);
}

template <uint32_t ct_v, uint32_t ct>
inline void k_step(uint32_t in0_cb, uint32_t in1_cb) {
    const bool throttled = (*(throttle_ptr) % 2) != 0;  // firmware-controlled (matmul_block_math_dynamic_throttle)
    if (throttled && throttled_mop_status != 1) {
        llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE_MAX>(in0_cb, in1_cb, false, ct, 1);
        throttled_mop_status = 1;
        addrmod_init();
    } else if (!throttled && throttled_mop_status != 0) {
        llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE>(in0_cb, in1_cb, false, ct, 1);
        throttled_mop_status = 0;
        addrmod_init();
    }
    for (uint32_t t = 0; t < ct_v; ++t) {  // V tiles: every phase
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(t);
        if (throttled) {
            for (uint32_t phase = 0; phase < kPhases; ++phase) {
                ckernel_template::run();
            }
            TTI_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
        } else {
            ckernel_template::run();
        }
    }
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(ct_v);  // the ones tile
    if (throttled) {
        // the throttled MOP runs one phase a call: phases 0 .. 2 under HiFi3 / HiFi4 (odd phase 1 adds zeros), 0 else
        for (uint32_t phase = 0; phase < (kPhases >= 3 ? 3 : 1); ++phase) {
            ckernel_template::run();
        }
    } else {
        if constexpr (kPhases >= 3) {
            lltt::replay(ckernel::math::replay_buf_offset, kBody - 1);  // phase 0 but its last MVMUL,
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_3, 0);            // which steps to phase 2
        }
        lltt::replay(ckernel::math::replay_buf_offset, kBody);  // phase 0 (HiFi2) / phase 2 (HiFi3, HiFi4)
    }
    TTI_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_ABD_F);  // the MOP end op: release SrcA, phase 0
    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD_F);  // the k-step's last tile: release SrcB
}
}  // namespace pv_ones
#endif

#if defined(TRISC_PACK)
// nomic_pack: one pack MOP run for a window's w tiles (file header "One pack MOP a window"). cfg_tiles mirrors the
// pack MOP's num_tiles (compute_kernel_hw_startup programs 1; kernel_main resets the mirror).
namespace nomic_pack {
uint32_t cfg_tiles = 1;
template <uint32_t cb>
inline void ensure(uint32_t n) {
    if (cfg_tiles != n) {
        llk_pack_init<PackMode::Default, false, true, true>(cb, n);  // the MOP only: ADDR_MODs and strides kept
        cfg_tiles = n;
    }
}
// = pack_tile<true>(j, cb, j) for j < w: DEST tiles 0..w-1 to the reserved pages 0..w-1, chunks of <= 4 tiles.
template <uint32_t w, uint32_t cb, bool pack_runs>
inline void pack_window() {
    if constexpr (!pack_runs) {
        for (uint32_t j = 0; j < w; ++j) {
            llk_pack<DST_ACCUM_MODE, true, PackMode::Default>(j, cb, j);
        }
        return;
    }
    constexpr uint32_t chunk = (w % 4 == 0) ? 4 : ((w % 2 == 0) ? 2 : 1);
    ensure<cb>(chunk);
#pragma GCC unroll 2
    for (uint32_t j = 0; j < w; j += chunk) {
        llk_pack<DST_ACCUM_MODE, true, PackMode::Default>(j, cb, j);
    }
}
}  // namespace nomic_pack
#endif

namespace {

constexpr uint32_t largest_divisor_at_most(uint32_t n, uint32_t cap) {
    for (uint32_t d = n < cap ? n : cap; d > 1; --d) {
        if (n % d == 0) {
            return d;
        }
    }
    return 1;
}

// Ring pads: pages without data that close a ragged block's ring (see the file header).
template <typename Buf>
ALWI void pad_push(Buf& buf, uint32_t pages) {
    if (pages) {
        buf.reserve_back(pages);
        buf.push_back(pages);
    }
}
template <typename Buf>
ALWI void pad_pop(Buf& buf, uint32_t pages) {
    if (pages) {
        buf.wait_front(pages);
        buf.pop_front(pages);
    }
}

// probs_block exp, incumbent form (op_design.md perf lamp L3, compute_streaming.hpp:617-625): the fast approximate
// exp WITHOUT the input clamp. Arguments are scale * (s - m) <= 0; ones below ~-88.5 come out negative and the
// pack ReLU (PackRelu::Zero on PackP) zeroes them. ckl::Exp<Approx::Fast> hardwires InputClamping::ClampToNegative,
// which is the helper limitation this element closes (kernel_lib gap: an InputClamping parameter on ckl::Exp).
template <ckl::Dst Slot = ckl::Dst::D0>
struct ExpFastNoClamp : ckl::UnaryOp<ExpFastNoClamp<Slot>, Slot> {
    static ALWI void init() { exp_tile_init<true, 0x3F800000, InputClamping::None>(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        exp_tile<true, false, InputClamping::None>(ckl::to_u32(Slot) + slot_offset);
    }
};

// probs_window (see the file header): P[row, w tiles] = exp(scale * (s - m[row])) with the exp on the pack thread.
// Consumes w tiles from the scores front, produces w tiles of P. The caller holds cb_row_stats (row = its index from
// the front) and has issued the pack-thread exp init.
template <uint32_t cb_scores, uint32_t cb_row_stats, uint32_t cb_probs, uint32_t w, bool pack_runs, typename Buf>
ALWI void probs_window(Buf& scores_buf, Buf& probs_buf, uint32_t row) {
    sub_bcast_cols_init(cb_scores, cb_row_stats);
    scores_buf.wait_front(w);
    tile_regs_acquire();
    for (uint32_t j = 0; j < w; ++j) {
        sub_tiles_bcast_cols(cb_scores, cb_row_stats, j, row, j);
    }
    tile_regs_commit();
    scores_buf.pop_front(w);

    probs_buf.reserve_back(w);
    tile_regs_wait();
#ifndef ABLATE_EXP                     // PROFILING ONLY: P = s - m (no exp)
    PACK(nomic_exp::exp_window<w>());  // = exp_packthread_tile<true, false, InputClamping::None, 32>(j) for j < w
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
#endif
#ifndef ABLATE_RELU
    pack_relu_config(ReluConfig::zero());
#endif
    PACK((nomic_pack::pack_window<w, cb_probs, pack_runs>()));  // = pack_tile<true>(j, cb_probs, j) for j < w
#ifndef ABLATE_RELU
    pack_relu_config(ReluConfig::none());
#endif
    tile_regs_release();
    probs_buf.push_back(w);
}

// pack_sfpu_fence: hold the MATH thread's next SFPU instruction until the pack thread has released every DEST window
// committed so far — so a pack-thread SFPU op still running on the shared SFPU finishes first. The SFPU init of a math-
// thread op (exp_tile_init) is not gated by any operand, so without this it can reprogram the SFPU mid-op (measured:
// a hang at 2x512 when normalize's recip ran on the math thread). Two empty windows: the second acquire waits
// (STALL_MATH | STALL_SFPU) until only the first is outstanding, i.e. every earlier window is released, and a window
// with pack-thread SFPU work releases only after its STALLWAIT on the SFPU. Used by the block-sequential schedule,
// whose exp runs on the MATH thread, after the block's pack-thread normalizes.
ALWI void pack_sfpu_fence() {
    for (uint32_t i = 0; i < 2; ++i) {
        tile_regs_acquire();
        tile_regs_commit();
        tile_regs_wait();
        tile_regs_release();
    }
}

// mm_window: one DEST window of a Q K^T chunk / P V column block, raw LLK (bypasses ckl::matmul_block):
//   out[0, ct) = sum_k in0[k] * in1[col + k * stride + c],  k < kt  (in0 = the in0 CB front row).
// Why not the helper: ckl::matmul_block issues PACK(pack_reconfig_data_format(out)) for every output subblock
// whenever fp32 DEST is on, even with DataFormatReconfig::NONE (every page here is Float16_b, so it is a no-op
// reconfig that drains the packer): measured 8x512 compute floor 61.7 -> 56.9 us without it (with the block row max).
// It also re-waits in0 / in1 and re-issues the short init every window. Here: `init` only when the previous
// FPU op was not a matmul of this shape (the caller tracks it); the caller holds in0 / in1 (waits once a row).
template <
    uint32_t cb_in0,
    uint32_t cb_in1,
    uint32_t cb_out,
    bool transpose,
    uint32_t ct,
    uint32_t kt,
    uint32_t stride,
    bool pack_runs,
    typename Buf>
ALWI void mm_window(Buf& out_buf, uint32_t col, bool init) {
    if (init) {
        matmul_block_init(cb_in0, cb_in1, transpose, ct, 1, kt);
    }
    tile_regs_acquire();
    for (uint32_t k = 0; k < kt; ++k) {
        matmul_block(cb_in0, cb_in1, k, col + k * stride, 0, transpose, ct, 1, kt);
    }
    tile_regs_commit();
    out_buf.reserve_back(ct);
    tile_regs_wait();
    PACK((nomic_pack::pack_window<ct, cb_out, pack_runs>()));  // = pack_tile<true>(j, cb_out, j) for j < ct
    tile_regs_release();
    out_buf.push_back(ct);
}

#if defined(TRISC_PACK)
// normalize_row_sfpu body, one row-quad (see the file header). DEST tiles relative to the call's base: O0 = 0, O1 = 1,
// l = 2 (SFPLOAD address units: 64 a tile, 16 a face). l is constant along a row and an SFPU load at offset a covers
// rows 4 * (a / 4 % 4) .. + 3 of its face (even columns at a % 4 == 0, odd at 2), so the loads at R, R + 2 and at
// R + 16, R + 18 (the right-hand face, same rows) all see the same l lane for lane: one 1/l serves 4 positions x
// {O0, O1}, and its dependency chain is covered by the four independent O loads behind it.
// LREG0 x -> e -> t, LREG1 y, LREG2..5 the O vectors.
template <int R>
inline void normalize_quad() {
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 128 + R);  // x = l
    TTI_SFPARECIP(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPARECIP_MOD1_RECIP);   // y ~ 1/x
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::DEFAULT, ADDR_MOD_7, R);
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 64 + R);
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::DEFAULT, ADDR_MOD_7, R + 2);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 64 + R + 2);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG0, 1);  // e = 1 - x*y
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LREG0, 0);     // t = e*e + e
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG1, p_sfpu::LREG1, 0);     // y = t*y + y (error ~e^3)
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG4, 0);
    TTI_SFPMAD(p_sfpu::LREG5, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG5, 0);
    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::DEFAULT, ADDR_MOD_7, R);
    TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 64 + R);
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::DEFAULT, ADDR_MOD_7, R + 2);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 64 + R + 2);
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::DEFAULT, ADDR_MOD_7, R + 16);
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 64 + R + 16);
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::DEFAULT, ADDR_MOD_7, R + 18);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 64 + R + 18);
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG4, 0);
    TTI_SFPMAD(p_sfpu::LREG5, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG5, 0);
    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::DEFAULT, ADDR_MOD_7, R + 16);
    TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 64 + R + 16);
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::DEFAULT, ADDR_MOD_7, R + 18);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 64 + R + 18);
}
inline void normalize_row_quads() {
    normalize_quad<0>();  // faces 0 | 1
    normalize_quad<4>();
    normalize_quad<8>();
    normalize_quad<12>();
    normalize_quad<32>();  // faces 2 | 3
    normalize_quad<36>();
    normalize_quad<40>();
    normalize_quad<44>();
}
inline void normalize_row_llk() {
    // The pack-thread fast exp leaves ADDR_MOD_7 at dest += 2 (ADDR_MODs are per-thread state): no increment here.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    _llk_math_eltwise_unary_sfpu_params_(normalize_row_quads, 0, VectorMode::None);
}
#endif

// normalize_row_sfpu: O0, O1 (DEST 0, 1) *= 1/l (DEST 2) on the PACK thread, between its tile_regs_wait and its packs.
ALWI void normalize_row_sfpu() {
    PACK((normalize_row_llk()));
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_query = get_named_compile_time_arg_val("cb_query");
    constexpr uint32_t cb_key_t = get_named_compile_time_arg_val("cb_key_t");
    constexpr uint32_t cb_value_aug = get_named_compile_time_arg_val("cb_value_aug");
    constexpr uint32_t cb_scores = get_named_compile_time_arg_val("cb_scores");
    constexpr uint32_t cb_row_stats = get_named_compile_time_arg_val("cb_row_stats");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t cb_reduce_scaler = get_named_compile_time_arg_val("cb_reduce_scaler");
    constexpr uint32_t cb_probs = get_named_compile_time_arg_val("cb_probs");
    constexpr uint32_t Q = get_named_compile_time_arg_val("q_block_rows");
    constexpr uint32_t N = get_named_compile_time_arg_val("key_block_tiles");
    constexpr uint32_t HD = get_named_compile_time_arg_val("head_dim_tiles");
    constexpr uint32_t PV = get_named_compile_time_arg_val("pv_out_tiles");
    constexpr uint32_t seq_tiles = get_named_compile_time_arg_val("seq_tiles");
    constexpr uint32_t grid_x = get_named_compile_time_arg_val("grid_x");
    constexpr uint32_t kv_depth = get_named_compile_time_arg_val("kv_depth");
    constexpr bool apply_scale = get_named_compile_time_arg_val("apply_scale") == 1;
    constexpr uint32_t scale_bits = get_named_compile_time_arg_val("scale_bits");
    constexpr ckl::Approx exp_approx =
        get_named_compile_time_arg_val("exp_approx") == 1 ? ckl::Approx::Fast : ckl::Approx::Exact;
    // exp_form (host knob EXP_FORM): 0 = ckl::Exp<exp_approx>; 1 = ExpFastNoClamp + pack ReLU (approx mode only).
    constexpr bool exp_no_clamp = get_named_compile_time_arg_val("exp_form") == 1 && exp_approx == ckl::Approx::Fast;
    using ExpElem = std::conditional_t<exp_no_clamp, ExpFastNoClamp<>, ckl::Exp<exp_approx>>;
    // Row-group schedule (file header). Host gate: EXP_OVERLAP, the approximate exp, EXP_FORM 1.
    constexpr bool exp_overlap = get_named_compile_time_arg_val("exp_overlap") == 1;
    constexpr uint32_t G = get_named_compile_time_arg_val("row_group_rows");
    static_assert(!exp_overlap || exp_no_clamp, "the row-group schedule runs the unclamped fast exp");

    // Mechanism caps (op_design.md): subblocks fill DEST. qk: w = largest divisor of N <= DEST; pv: w = PV.
    constexpr uint32_t dest_tiles = ckl::DEST_AUTO_LIMIT;
    constexpr uint32_t qk_w = largest_divisor_at_most(N, dest_tiles);
    static_assert(PV <= dest_tiles, "the [O | l] row of P V must fit DEST");
    constexpr uint32_t scores_pages = Q * N;  // cb_scores capacity
    // one pack MOP a window when a score row has >= 2 windows (file header "One pack MOP a window"); the row-group
    // schedule only (the block-sequential one packs through the helpers)
    constexpr bool pack_runs = exp_overlap && N / qk_w >= 2;

    const uint32_t core = nomic_sdpa::core_index(grid_x);
    constexpr uint32_t kTableArg = get_named_compile_time_arg_val("table_arg");  // core_row_start offset
    const uint32_t row_begin = get_common_arg_val<uint32_t>(kTableArg + core);
    const uint32_t row_end = get_common_arg_val<uint32_t>(kTableArg + core + 1);

    DataflowBuffer query_buf(cb_query);
    DataflowBuffer key_buf(cb_key_t);
    DataflowBuffer value_buf(cb_value_aug);
    DataflowBuffer scores_buf(cb_scores);
    DataflowBuffer stats_buf(cb_row_stats);
    DataflowBuffer out_buf(cb_out);
    DataflowBuffer scaler_buf(cb_reduce_scaler);
    DataflowBuffer probs_buf(cb_probs);

    // boot: matmul maps in0 -> SrcB, in1 -> SrcA (Reverse); later helpers issue their own short inits.
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_query, cb_key_t, cb_scores);

    constexpr auto kNoReconfig = ckl::matmul_config::DataFormatReconfig::NONE;
    PACK((nomic_pack::cfg_tiles = 1));  // the pack MOP hw_startup programmed
    if constexpr (exp_overlap) {
        exp_packthread_tile_init<true, scale_bits, InputClamping::None>();  // once a kernel (file header)
    }

    // K/V phantom slots (reader file header): the rings start (first pair % kv_depth) data-less slots in, so pair p
    // sits in ring slot p % kv_depth on every core — the address the pair's multicast lands at.
    const uint32_t kv_phantoms = (row_begin / seq_tiles) % kv_depth;
    pad_pop(key_buf, kv_phantoms * N * HD);
    pad_pop(value_buf, kv_phantoms * N * PV);

    // pv_norm_row: [O | l] = P_row [V | 1] (the in0 front row: N tiles of P, popped) in one DEST window, then O / l
    // in DEST on the pack thread and O0, O1 packed to cb_out (file header "Normalize in DEST"). V is retained.
    const auto pv_norm_row = [&](DataflowBuffer& in0_buf, uint32_t in0_cb) {
        ckernel::matmul_block_init(in0_cb, cb_value_aug, false, PV, 1, N);
        MATH((pv_ones::addrmod_init()));
        in0_buf.wait_front(N);
        value_buf.wait_front(N * PV);
        tile_regs_acquire();
        for (uint32_t k = 0; k < N; ++k) {  // = matmul_block(in0, cb_value_aug, k, k * PV, 0, false, PV, 1, N)
            UNPACK((llk_unpack_AB_matmul(in0_cb, cb_value_aug, k, k * PV, PV, 1, N)));
            MATH((pv_ones::k_step<HD, PV>(in0_cb, cb_value_aug)));  // the ones tile at its live phases only
        }
        tile_regs_commit();
        in0_buf.pop_front(N);
        out_buf.reserve_back(HD);
        tile_regs_wait();
        normalize_row_sfpu();
        if constexpr (pack_runs) {
            PACK((nomic_pack::ensure<cb_out>(1)));
        }
        for (uint32_t j = 0; j < HD; ++j) {
            pack_tile<true>(j, cb_out, j);
        }
        tile_regs_release();
        out_buf.push_back(HD);
    };

    // ---------------- row-group schedule (exp_overlap; file header) ----------------
    constexpr uint32_t row_windows = N / qk_w;  // DEST windows of one score row
    constexpr uint32_t group_scores = G * N;    // nominal pages of a group in cb_scores / cb_probs
    const auto group_rows = [](uint32_t r, uint32_t g) { return r - g * G < G ? r - g * G : G; };

    // qk_window: scores row chunk [c * qk_w, (c + 1) * qk_w) of the cb_query front row (K retained, Q retained).
    // The FPU op last issued (mm_window skips the matmul short init between two windows of one shape).
    enum : uint32_t { kOpOther = 0, kOpQk = 1 };
    uint32_t last_op = kOpOther;
    const auto qk_window = [&](uint32_t c) {
        if (c == 0) {
            query_buf.wait_front(HD);  // the row's Q (K is held for the block)
        }
        mm_window<cb_query, cb_key_t, cb_scores, true, qk_w, HD, N, pack_runs>(scores_buf, c * qk_w, last_op != kOpQk);
        last_op = kOpQk;
        if (c == row_windows - 1) {
            query_buf.pop_front(HD);  // the row's Q is done
        }
    };
    // block_row_max: m = rowmax(scores) for `rows` rows at the scores front (kept), one m tile a row pushed to
    // cb_row_stats. reduce_block_max_row (reduce_custom.h; the stock SDPA's row max) pools the whole N-tile row and
    // transposes ONCE a row; ckl::reduce<MAX, REDUCE_ROW> pools + transposes every tile (measured 8x512 compute floor
    // 75.3 -> 63.2 us). Max is exact: m is bit-identical. Preconditions: bf16 operands, the scaler tile bf16 1.0 in row
    // 0 of each face (the writer's MAX / REDUCE_ROW scaler). Kernel_lib gap: no block row-max path in ckl::reduce.
    const auto block_row_max = [&](uint32_t rows) {
        scores_buf.wait_front(rows * N);
        scaler_buf.wait_front(1);
        reduce_block_max_row_init<N>(cb_row_stats);
        for (uint32_t r = 0; r < rows; ++r) {
            stats_buf.reserve_back(1);
            tile_regs_acquire();
            reduce_block_max_row<N>(cb_scores, cb_reduce_scaler, r * N, 0);
            tile_regs_commit();
            tile_regs_wait();
            if constexpr (pack_runs) {
                PACK((nomic_pack::ensure<cb_row_stats>(1)));
            }
            pack_tile(0, cb_row_stats);
            tile_regs_release();
            stats_buf.push_back(1);
        }
        reduce_block_max_row_uninit(cb_scores);
    };
    // row_max(g): m = rowmax over the group's scores (kept for the probs windows); pads cb_row_stats to G.
    const auto row_max = [&](uint32_t rows) {
        block_row_max(rows);
        pad_push(stats_buf, G - rows);
        last_op = kOpOther;
    };

    // pv_window: the whole [O | l] row of the cb_probs front row, normalized in DEST (O and l share the window, so a
    // row is exactly one P V window). Its matmul init differs from Q K^T's: last_op = other.
    constexpr uint32_t pv_windows = 1;
    const auto pv_window = [&](uint32_t) {
        pv_norm_row(probs_buf, cb_probs);
        last_op = kOpOther;
    };

    const auto row_group_block = [&](const nomic_sdpa::Block& blk) {
        const uint32_t r = blk.rows;
        const uint32_t groups = (r + G - 1) / G;

        // prologue: group 0's scores and row max
        const uint32_t rows0 = group_rows(r, 0);
        {
            MaybeDeviceZoneScope("compute_prologue");
            for (uint32_t w = 0; w < rows0 * row_windows; ++w) {
                qk_window(w % row_windows);
            }
            pad_push(scores_buf, group_scores - rows0 * N);
            if (groups == 1 && blk.last_of_segment) {
                key_buf.pop_front(N * HD);  // release_kv (K half)
            }
            row_max(rows0);
        }

        // step t: Q K^T of group t+1 and probs of group t in alternating windows (Q K^T first), then P V of group
        // t-1 (one row a window, normalized in DEST), then row_max(t+1) (file header: the step's window stream).
        for (uint32_t t = 0; t <= groups; ++t) {
            const uint32_t probs_rows = t < groups ? group_rows(r, t) : 0;
            const uint32_t pv_rows = t > 0 ? group_rows(r, t - 1) : 0;
            const uint32_t qk_rows = t + 1 < groups ? group_rows(r, t + 1) : 0;
            const uint32_t probs_n = probs_rows * row_windows;
            const uint32_t pv_n = pv_rows * pv_windows;
            const uint32_t qk_n = qk_rows * row_windows;

            if (t ==
                1) {  // V for the block's first P V (read-only wait, no pop; functional: mm_window does not wait V)
                MaybeDeviceZoneScope("compute_wait_v");
                value_buf.wait_front(N * PV);
            }
            {
                // Q K^T / probs window pairs (s - m on the FPU, exp on the pack thread), then P V
                MaybeDeviceZoneScope("compute_exp_pv_qk");
                if (probs_n) {
                    stats_buf.wait_front(G);  // m(t)
                }
                for (uint32_t k = 0; k < probs_n || k < qk_n; ++k) {
                    if (k < qk_n) {
                        qk_window(k % row_windows);
                    }
                    if (k < probs_n) {
                        probs_window<cb_scores, cb_row_stats, cb_probs, qk_w, pack_runs>(
                            scores_buf, probs_buf, k / row_windows);
                        last_op = kOpOther;
                    }
                }
                for (uint32_t k = 0; k < pv_n; ++k) {
                    pv_window(k % pv_windows);
                }
            }

            // ring pads back to the nominal group extent (group t-1's P pads sit ahead of group t's P)
            if (probs_n) {
                pad_pop(scores_buf, group_scores - probs_rows * N);
                stats_buf.pop_front(G);  // m(t)
            }
            if (qk_rows) {
                pad_push(scores_buf, group_scores - qk_rows * N);
                if (t + 2 == groups && blk.last_of_segment) {
                    key_buf.pop_front(N * HD);  // release_kv (K half): the block's last Q K^T is done
                }
            }
            if (pv_rows) {
                pad_pop(probs_buf, group_scores - pv_rows * N);
                if (t == groups && blk.last_of_segment) {
                    value_buf.pop_front(N * PV);  // release_kv (V half)
                }
            }
            if (probs_n) {
                pad_push(probs_buf, group_scores - probs_rows * N);
            }

            if (qk_rows) {
                MaybeDeviceZoneScope("compute_row_max");
                row_max(qk_rows);
            }
        }
        pad_pop(query_buf, (Q - r) * HD);
        pad_push(out_buf, (Q - r) * HD);
    };

    nomic_sdpa::for_each_block<seq_tiles, Q>(row_begin, row_end, [&](const nomic_sdpa::Block& blk) {
        const uint32_t r = blk.rows;
        const uint32_t live_scores = r * N;
#ifdef ABLATE_COMPUTE
        // ABLATION ONLY: the block's CB protocol with no unpack / math / pack.
        query_buf.wait_front(Q * HD);
        query_buf.pop_front(Q * HD);
        key_buf.wait_front(N * HD);
        value_buf.wait_front(N * PV);
        if (blk.last_of_segment) {
            key_buf.pop_front(N * HD);
            value_buf.pop_front(N * PV);
        }
        out_buf.reserve_back(Q * HD);
        out_buf.push_back(Q * HD);
        return;
#endif

        {  // the block's input waits (read-only, no pops). FUNCTIONAL since Perf 1: the raw Q K^T windows (mm_window)
           // rely on this K wait, and on compute_wait_v below for V; the zone only times them.
            MaybeDeviceZoneScope("compute_wait_qk");
            query_buf.wait_front(HD);  // the first Q row: later rows may follow V (reader q_lead_rows)
            key_buf.wait_front(N * HD);
        }
        if constexpr (exp_overlap) {
            row_group_block(blk);
            return;
        }

        // ---- qk_block: scores = Q K^T (K retained across the segment's blocks) ----
        const uint32_t qk_h = largest_divisor_at_most(r, dest_tiles / qk_w);
        {
            MaybeDeviceZoneScope("compute_qk");
            ckl::matmul_block<
                true,
                false,
                ckl::LastBlockTarget::Out,
                ckl::OutputCBLayout::TileRowMajor,
                ckl::matmul_config::InitMode::Short,
                ckl::InputPolicy::WaitAndPopPerKBlock,
                ckl::InputPolicy::WaitAndRetainOnLastBlock,
                ckl::NoPostCompute,
                ckl::NoPreKBlock,
                ckl::NoPostKBlock,
                0,
                ckl::NoKBlockInnerDimFn,
                ckl::NoIn0Source,
                ckl::NoIn1BaseOffset,
                false,
                ckl::NoneActivation,
                kNoReconfig>(
                query_buf,
                key_buf,
                scores_buf,
                scores_buf,
                ckl::MatmulBlockShape::of(r / qk_h, N / qk_w, qk_h, qk_w, HD, 1));
        }
        pad_pop(query_buf, (Q - r) * HD);
        pad_push(scores_buf, scores_pages - live_scores);
        if (blk.last_of_segment) {
            key_buf.pop_front(N * HD);  // release_kv (K half)
        }

        // ---- row_max_block: m = rowmax(scores); scores stay resident ----
        {
            MaybeDeviceZoneScope("compute_row_max");
            block_row_max(r);
        }
        pad_push(stats_buf, Q - r);

        // ---- probs_block: P = exp(scale * (scores - m)), in place in cb_scores ----
        // PerBlockSize on both sides: each DEST window pops its qk_w score tiles (compute phase) before it reserves
        // the same qk_w slots (pack phase), so P lands exactly where the scores were.
        const auto probs_shape = ckl::IterationShape::grid(r, N).block_size(qk_w);
        using SubMax = ckl::BinaryFpu<
            ckl::BinaryFpuOp::Sub,
            ckl::input(
                cb_scores,
                ckl::WaitPolicy::PerBlockSize,
                ckl::PopPolicy::PerBlockSize,
                ckl::InputTileMapping::Block,
                ckl::DataFormatReconfig::Disabled),
            ckl::input(
                cb_row_stats,
                ckl::BroadcastDim::Col,
                ckl::WaitPolicy::Upfront,
                ckl::PopPolicy::AtEnd,
                ckl::InputTileMapping::Col,
                ckl::DataFormatReconfig::Disabled)>;
        using PackP = ckl::PackTile<ckl::output(
            cb_scores,
            ckl::ReservePolicy::PerBlockSize,
            ckl::PushPolicy::PerBlockSize,
            ckl::DataFormatReconfig::Disabled,
            ckl::TileAddressing::Direct,
            ckl::DestAccumulation::Disabled,
            ckl::L1Accumulation::Disabled,
            exp_no_clamp ? ckl::PackRelu::Zero : ckl::PackRelu::Disabled)>;
        if constexpr (apply_scale) {
            MaybeDeviceZoneScope("compute_probs");
            ckl::eltwise_chain(
                probs_shape, SubMax{scores_buf, stats_buf}, ckl::MulUnary<>{scale_bits}, ExpElem{}, PackP{scores_buf});
        } else {
            MaybeDeviceZoneScope("compute_probs");
            ckl::eltwise_chain(probs_shape, SubMax{scores_buf, stats_buf}, ExpElem{}, PackP{scores_buf});
        }
        pad_pop(stats_buf, Q - r);
        pad_pop(scores_buf, scores_pages - live_scores);
        pad_push(scores_buf, scores_pages - live_scores);

        {
            MaybeDeviceZoneScope("compute_wait_v");
            value_buf.wait_front(N * PV);
        }
        // ---- pv_block + normalize_block: [O | l] = P [V | 1] a row a DEST window, O / l in DEST -> cb_out ----
        {
            MaybeDeviceZoneScope("compute_pv");
            for (uint32_t row = 0; row < r; ++row) {
                pv_norm_row(scores_buf, cb_scores);
            }
        }
        pad_pop(scores_buf, scores_pages - live_scores);
        if (blk.last_of_segment) {
            value_buf.pop_front(N * PV);  // release_kv (V half)
        }
        pack_sfpu_fence();  // the pack-thread normalizes off the SFPU before the next block's math-thread exp
        pad_push(out_buf, (Q - r) * HD);
    });
}
