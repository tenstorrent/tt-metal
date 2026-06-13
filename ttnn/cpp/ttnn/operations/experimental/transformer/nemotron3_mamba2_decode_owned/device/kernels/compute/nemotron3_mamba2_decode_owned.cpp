// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0
//
// Mamba2 SSD decode-step owned kernel — TRISC compute side.
//
// Forked from
//   ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_gdn_decode_owned/
//     device/kernels/compute/qwen36_gdn_decode_owned.cpp
// per the Path-B G1 plan (research/mm7_g1_mamba2_kernel_design.md).
//
// Math implemented here (per
// research/nemotron3_nano_architecture_brief.md §4.3 and
// wiki/65_mamba_state_space_models.md §3):
//
//   dt_eff[b, h] = clamp(softplus(dt[b, h] + dt_bias[h]),
//                        time_step_floor, time_step_max)
//   A[h]         = -exp(A_log[h])
//   decay[b, h]  = exp(dt_eff[b, h] * A[h])
//   dt_B[b, h, s] = dt_eff[b, h] * B[b, g, s]   (broadcast over heads of a group)
//
//   ssm_state[b, h, d, s] = decay[b, h] * ssm_state[b, h, d, s]
//                         + dt_B[b, h, s] * x[b, h, d]    (outer over d × s)
//
//   y[b, h, d] = sum_s(C[b, g, s] * ssm_state[b, h, d, s])
//              + D[h] * x[b, h, d]
//
// Where (per Nemotron-3 Nano shapes):
//   num_heads = 64, head_dim = 64 (= 2 tiles), ssm_state = 128 (= 4 tiles),
//   n_groups = 8 (heads_per_group = 8).
//
// All math runs in fp32 inside the dest accumulator (fp32_dest_acc_en=true).
// State CBs are bf16 on the L1 side and fp32 in the dest register file —
// the same trick the GDN owned kernel uses; see decision D4 in
// research/mm7_g1_dataflow_decisions.md.
//
// SPMD: each Tensix handles ONE (batch, head) block per kernel invocation
// (decision D1). Per-block loop covers the head_dim tile dimension
// (head_dim = 64 = 2 tiles).
//
// Build-up via runtime arg `debug_mode` (decision D7):
//   0 = production: full SSD recursion (state correct + y correct)
//   1 = fill_one smoke (no compute; output y filled with 1.0)
//   2 = decay × state only (no input contribution; output y = 1.0 sentinel)
//   3 = decay × state + input contribution (state correct; y = 1.0 sentinel)
//   4 = state correct + y = D·x (output ignores C·state)
//   5 = production equivalent (mode 0)
//
// Each subsequent commit lands one more debug_mode step. As of G1 day-4,
// modes 1, 2, and 3 are wired (1 = fill_one smoke; 2 = decay×state only;
// 3 = full state update with input contribution). Modes 4, 5 land at
// day-4.5 (D·x skip and full math).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_wh.h"  // G1 day-4: transpose_wh_tile for x → col-vec

namespace {

constexpr uint32_t ONE_TILE = 1;

// ─────────────────────────────────────────────────────────────────────────────
// FORCE_INLINE helpers — Mamba2 SSD–specific tile ops.
//
// We keep each helper small and named after the math step (decision D7's
// debug_mode pattern: each mode wires up the next helper). The GDN kernel's
// helpers (mul_alpha_*, matmul_reduce, mul_beta_*, etc.) live in the fork-base
// file; this file uses Mamba2-specific names instead so a future reader
// doesn't have to mentally remap.

// fill_one(cb_out): write a single tile of 1.0 to cb_out. Used by debug_mode=1
// to validate the scaffolding (program build, kernel dispatch, output channel
// non-NaN) BEFORE any Mamba2 math runs.
FORCE_INLINE void fill_one(uint32_t cb_out) {
    pack_reconfig_data_format(cb_out);
    cb_reserve_back(cb_out, ONE_TILE);

    tile_regs_acquire();
    fill_tile_init();
    fill_tile(0, 1.0f);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_push_back(cb_out, ONE_TILE);
}

// compute_decay(cb_A_log, cb_decay)
//
// Stage A of the decay computation. Produces A = -exp(A_log) into cb_decay.
// Caller follows up with finalize_decay_with_dt_eff() which multiplies by
// the recomputed dt_eff and exps the product to get the final decay.
//
// This helper consumes ONLY cb_A_log. cb_dt and cb_dt_bias are consumed
// by finalize_decay (decision D11 v2: keep the two stages separate, one
// tile_regs cycle each, simpler to debug + matches GDN's single-purpose
// helper convention).
//
// Decision D8 RESOLVED (LLK API survey at G1 day-3): `exp_tile` and
// `negative_tile` are first-class SFPU primitives at
// /home/aditya/tenstorrent/tt-metal/tt_metal/hw/inc/api/compute/
// eltwise_unary/{exp,negative}.h. No decomposition.
FORCE_INLINE void compute_decay(uint32_t cb_A_log, uint32_t cb_decay) {
    cb_wait_front(cb_A_log, ONE_TILE);

    pack_reconfig_data_format(cb_decay);
    reconfig_data_format(cb_A_log, cb_A_log);
    cb_reserve_back(cb_decay, ONE_TILE);

    tile_regs_acquire();
    // A_log → exp → negate → A
    copy_tile_init(cb_A_log);
    copy_tile(cb_A_log, 0, 0);
    exp_tile_init();
    exp_tile(0);
    negative_tile_init();
    negative_tile(0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(0, cb_decay);
    tile_regs_release();

    cb_push_back(cb_decay, ONE_TILE);
    cb_pop_front(cb_A_log, ONE_TILE);
}

// compute_dt_eff(cb_dt, cb_dt_bias, cb_dt_eff_dst, softplus/clamp constants)
//
// Stage 1 of the decay computation, split from the old monolithic
// `finalize_decay_with_dt_eff` at G1 day-4 so that dt_eff stays alive in
// cb_dt_eff_dst for the downstream `compute_dt_B` (mode=3 needs dt_eff for
// both the decay-finalize multiply AND the dt_eff*B outer-product
// precompute).
//
// Pushes 1 tile to cb_dt_eff_dst (dt_eff = clamp(softplus(dt+dt_bias))).
// Pops cb_dt and cb_dt_bias (consumes both). Does NOT pop cb_dt_eff_dst —
// the caller (multiply_decay_by_dt_eff, then compute_dt_B at mode=3)
// reuses the dt_eff tile.
FORCE_INLINE void compute_dt_eff(
    uint32_t cb_dt,
    uint32_t cb_dt_bias,
    uint32_t cb_dt_eff_dst,
    uint32_t softplus_beta_bits,
    uint32_t softplus_beta_recip_bits,
    uint32_t softplus_threshold_bits,
    uint32_t time_step_floor_bits,
    uint32_t time_step_max_bits) {
    cb_wait_front(cb_dt, ONE_TILE);
    cb_wait_front(cb_dt_bias, ONE_TILE);

    pack_reconfig_data_format(cb_dt_eff_dst);
    reconfig_data_format(cb_dt, cb_dt_bias);
    add_tiles_init(cb_dt, cb_dt_bias);
    cb_reserve_back(cb_dt_eff_dst, ONE_TILE);

    tile_regs_acquire();
    add_tiles(cb_dt, cb_dt_bias, 0, 0, 0);
    softplus_tile_init();
    softplus_tile(0, softplus_beta_bits, softplus_beta_recip_bits, softplus_threshold_bits);
    clamp_tile_init();
    clamp_tile(0, time_step_floor_bits, time_step_max_bits);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_dt_eff_dst);
    tile_regs_release();
    cb_push_back(cb_dt_eff_dst, ONE_TILE);

    cb_pop_front(cb_dt, ONE_TILE);
    cb_pop_front(cb_dt_bias, ONE_TILE);
}

// multiply_decay_by_dt_eff(cb_decay_inout, cb_dt_eff)
//
// Stage 2+3 of the decay computation, split from the old
// `finalize_decay_with_dt_eff`. Reads A from cb_decay_inout (placed there by
// compute_decay) and dt_eff from cb_dt_eff (placed by compute_dt_eff).
// Computes decay = exp(A * dt_eff) and overwrites cb_decay_inout.
//
// Does NOT pop cb_dt_eff. The caller is responsible: at debug_mode=2 the
// kernel-main pops cb_dt_eff after this returns; at debug_mode=3
// compute_dt_B consumes it via the queue-pop-after-push pattern.
FORCE_INLINE void multiply_decay_by_dt_eff(uint32_t cb_decay_inout, uint32_t cb_dt_eff) {
    // CB queue semantics: cb_decay_inout currently has [A]. We compute
    // [A * dt_eff → exp] in dest, push as a new tile, then pop the OLD
    // A from the front, leaving [decay] for the downstream
    // mul_decay_state_to.
    cb_wait_front(cb_dt_eff, ONE_TILE);
    cb_wait_front(cb_decay_inout, ONE_TILE);

    pack_reconfig_data_format(cb_decay_inout);
    reconfig_data_format(cb_decay_inout, cb_dt_eff);
    mul_tiles_init(cb_decay_inout, cb_dt_eff);
    cb_reserve_back(cb_decay_inout, ONE_TILE);

    tile_regs_acquire();
    mul_tiles(cb_decay_inout, cb_dt_eff, 0, 0, 0);  // A * dt_eff
    exp_tile_init();
    exp_tile(0);  // exp(...)
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_decay_inout);  // push new decay
    tile_regs_release();
    cb_push_back(cb_decay_inout, ONE_TILE);
    // Queue is now [OLD_A, decay]. Pop the OLD_A so the front exposes decay.
    cb_pop_front(cb_decay_inout, ONE_TILE);
}

// compute_dt_B(cb_dt_eff_inout, cb_B, ssm_state_tiles)
//
// Implements (for the per-block dt_eff scalar and B vector):
//   dt_B[s] = dt_eff * B[s]   (broadcast scalar across each ssm_state tile)
//
// In the kernel, the same physical CB (cb_dt_B) holds dt_eff transiently at
// the front, then gets overwritten with ssm_state_tiles dt_B tiles. The
// queue-pop-after-push pattern is the same one used by
// multiply_decay_by_dt_eff: push all dt_B tiles before popping the dt_eff
// tile, so dt_eff stays addressable at index 0 during each mul_tiles_bcast_scalar
// call. Capacity check (program_factory):
//   CB_DT_B size = ssm_state_tiles * 2 = 8 slots; transient peak = 1 + 4 = 5. OK.
//
// Pops cb_dt_eff_inout's dt_eff tile (1) and cb_B's vector tiles (ssm_state_tiles).
// Net: cb_dt_eff_inout front = [dt_B[0], dt_B[1], …, dt_B[ssm_state_tiles-1]].
FORCE_INLINE void compute_dt_B(uint32_t cb_dt_eff_inout, uint32_t cb_B, uint32_t ssm_state_tiles) {
    cb_wait_front(cb_dt_eff_inout, ONE_TILE);
    cb_wait_front(cb_B, ssm_state_tiles);

    reconfig_data_format(cb_B, cb_dt_eff_inout);
    pack_reconfig_data_format(cb_dt_eff_inout);
    mul_tiles_bcast_scalar_init_short(cb_B, cb_dt_eff_inout);

    for (uint32_t s = 0; s < ssm_state_tiles; ++s) {
        cb_reserve_back(cb_dt_eff_inout, ONE_TILE);

        tile_regs_acquire();
        mul_tiles_bcast_scalar(cb_B, cb_dt_eff_inout, s, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_dt_eff_inout);
        tile_regs_release();

        cb_push_back(cb_dt_eff_inout, ONE_TILE);
    }

    cb_pop_front(cb_dt_eff_inout, ONE_TILE);  // pop dt_eff
    cb_pop_front(cb_B, ssm_state_tiles);
}

// mul_decay_state_to(cb_state, cb_decay, head_dim_tile, ssm_state_tile,
//                    cb_state_scaled)
//
// Implements (for one tile of state):
//   state_scaled[head_dim_tile, ssm_state_tile, :, :] =
//       state[head_dim_tile, ssm_state_tile, :, :] * decay   (broadcast scalar)
//
// state is laid out as [head_dim_tiles=2 * ssm_state_tiles=4 = 8 tiles per
// (batch, head)]. The tile_index is `head_dim_tile * ssm_state_tiles +
// ssm_state_tile`. Caller's loop iterates over both dims.
//
// REUSE: this is a direct fork of
//   qwen36_gdn_decode_owned/device/kernels/compute/qwen36_gdn_decode_owned.cpp
//   line 57: `mul_alpha_scalar_tile_indexed`
// renamed for Mamba2 semantics. Same LLK call (`mul_tiles_bcast_scalar`),
// same tile_regs pattern.
FORCE_INLINE void mul_decay_state_to(
    uint32_t cb_state, uint32_t cb_decay, uint32_t tile_index, uint32_t cb_state_scaled) {
    reconfig_data_format(cb_state, cb_decay);
    pack_reconfig_data_format(cb_state_scaled);
    mul_tiles_bcast_scalar_init_short(cb_state, cb_decay);
    cb_wait_front(cb_state, tile_index + 1);
    cb_wait_front(cb_decay, ONE_TILE);
    cb_reserve_back(cb_state_scaled, ONE_TILE);

    tile_regs_acquire();
    mul_tiles_bcast_scalar(cb_state, cb_decay, tile_index, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_state_scaled);
    tile_regs_release();

    cb_push_back(cb_state_scaled, ONE_TILE);
}

// transpose_x_to_col(cb_x, tile_index, cb_x_col)
//
// Reads cb_x at `tile_index` (the d-th head_dim tile, a row-vector tile with
// 32 head_dim values along row 0) and writes the transposed tile to cb_x_col
// (col-vector tile with 32 head_dim values along col 0). Used to feed
// matmul_outer_x_dt_B, where matmul_tiles(col_vec, row_vec) computes a
// rank-1 outer product directly.
//
// REUSE: direct fork of
//   qwen36_gdn_decode_owned/device/kernels/compute/qwen36_gdn_decode_owned.cpp
//   line 312: `transpose_k_indexed`
// renamed for clarity (cb_k → cb_x). Same LLK pattern.
FORCE_INLINE void transpose_x_to_col(uint32_t cb_x, uint32_t tile_index, uint32_t cb_x_col) {
    // GDN canonical pattern: unary_op_init_common is required to fully
    // re-init the LLK pipeline so subsequent matmul calls don't inherit
    // sticky unpacker state. Verified empirically — without this, multiple
    // matmuls in the inner loop hang (step-10). With it (+ transpose
    // inside the inner loop per GDN), the kernel runs.
    unary_op_init_common(cb_x, cb_x_col);
    transpose_wh_init_short(cb_x);
    pack_reconfig_data_format(cb_x_col);
    cb_wait_front(cb_x, tile_index + 1);
    cb_reserve_back(cb_x_col, ONE_TILE);

    tile_regs_acquire();
    transpose_wh_tile(cb_x, tile_index, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_x_col);
    tile_regs_release();

    cb_push_back(cb_x_col, ONE_TILE);
}

// matmul_outer_x_dt_B(cb_x_col, cb_dt_B, s_tile_index, cb_outer)
//
// Computes the outer-product tile:
//   outer[i, j] = x_col[i, 0] * dt_B[s][0, j]
//              = x[d][i] * (dt_eff * B[s])[j]
//
// for the s-th ssm_state tile. cb_x_col holds the transposed head_dim
// vector (col-vector); cb_dt_B holds the (dt_eff * B) vector across
// ssm_state_tiles row-vector tiles. matmul_tiles on a col-vector × row-vector
// is exactly a rank-1 outer product (other rows/cols of both inputs are
// zero-padded by the tilizer).
//
// Reads cb_x_col at index 0 (caller pushes 1 tile per d-iter) and cb_dt_B
// at index `s_tile_index`. Pushes 1 outer tile to cb_outer. Does NOT pop
// either input — caller drains cb_x_col after the inner s-loop and
// cb_dt_B after the entire mode-3 block.
//
// REUSE: forked from
//   qwen36_gdn_decode_owned/device/kernels/compute/qwen36_gdn_decode_owned.cpp
//   line 346: `matmul_outer`
// extended with a tile_index arg so we can select dt_B[s] within the
// inner loop.
FORCE_INLINE void matmul_outer_x_dt_B(
    uint32_t cb_x_col,
    uint32_t cb_dt_B,
    uint32_t head_dim_tile,  // day-4.2: read cb_x_col at this index (was hardcoded 0)
    uint32_t s_tile_index,
    uint32_t cb_outer) {
    // day-5: revert to full mm_init now that the transpose is in a
    // pre-loop phase (no transpose interleaved with matmul). mm_init_short
    // (lighter) was the day-4 fix when transpose+matmul shared the inner
    // loop, but with separation the iter cap doesn't bite. Full mm_init
    // re-runs llk_pack_hw_configure per call, which is REQUIRED for the
    // fp32 cb_outer pack to work — otherwise pack hw config drifts from
    // Phase 2's bf16 cb_y_partial setup and outer values come out wrong
    // (multi-step step 0 with state_in=0 exposed: state cos ~0).
    mm_init(cb_x_col, cb_dt_B, cb_outer);
    cb_wait_front(cb_x_col, head_dim_tile + 1);
    cb_wait_front(cb_dt_B, s_tile_index + 1);
    cb_reserve_back(cb_outer, ONE_TILE);

    tile_regs_acquire();
    matmul_tiles(cb_x_col, cb_dt_B, head_dim_tile, s_tile_index, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_outer);
    tile_regs_release();

    cb_push_back(cb_outer, ONE_TILE);
}

// add_state_scaled_outer(cb_state_scaled, cb_outer, tile_index, cb_state_out)
//
// Implements (for one (d, s) tile of the state update):
//   state_out[d, s] = state_scaled[d, s] + outer[d, s]
//                   = decay * state_in[d, s] + dt_eff * x[d] * B[s]
//
// Reads cb_state_scaled at `tile_index` and cb_outer at index 0. Pushes
// to cb_state_out. Does NOT pop either input — caller drains
// cb_state_scaled after the entire mode-3 block and pops cb_outer at the
// bottom of each inner-loop iteration.
//
// REUSE: direct fork of
//   qwen36_gdn_decode_owned/device/kernels/compute/qwen36_gdn_decode_owned.cpp
//   line 389: `add_state_to_out` (renamed for Mamba2 readability).
FORCE_INLINE void add_state_scaled_outer(
    uint32_t cb_state_scaled, uint32_t cb_outer, uint32_t tile_index, uint32_t cb_state_out) {
    reconfig_data_format(cb_state_scaled, cb_outer);
    pack_reconfig_data_format(cb_state_out);
    add_tiles_init(cb_state_scaled, cb_outer);
    cb_wait_front(cb_state_scaled, tile_index + 1);
    cb_wait_front(cb_outer, ONE_TILE);
    cb_reserve_back(cb_state_out, ONE_TILE);

    tile_regs_acquire();
    add_tiles(cb_state_scaled, cb_outer, tile_index, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_state_out);
    tile_regs_release();

    cb_push_back(cb_state_out, ONE_TILE);
}

// add_state_scaled_outer_two(cb_state_scaled, cb_outer, tile_index,
//                            cb_state_out, cb_state_post_update)
//
// Mode=5 variant of add_state_scaled_outer that packs the SAME computed
// state_out value to TWO destination CBs:
//   - cb_state_out: the writer-bound output (drained by the dataflow kernel)
//   - cb_state_post_update: an internal copy that compute reads back in
//     Phase 4 to compute the corrected y = C · state_out^T + D·x.
//
// REUSE: direct fork of
//   qwen36_gdn_decode_owned/device/kernels/compute/qwen36_gdn_decode_owned.cpp
//   line 389: `add_state_to_two`. GDN uses the exact same dual-pack pattern
//   for the same reason (one output for writer, one for downstream compute).
FORCE_INLINE void add_state_scaled_outer_two(
    uint32_t cb_state_scaled,
    uint32_t cb_outer,
    uint32_t tile_index,
    uint32_t cb_state_out,
    uint32_t cb_state_post_update) {
    reconfig_data_format(cb_state_scaled, cb_outer);
    pack_reconfig_data_format(cb_state_out);
    add_tiles_init(cb_state_scaled, cb_outer);
    cb_wait_front(cb_state_scaled, tile_index + 1);
    cb_wait_front(cb_outer, ONE_TILE);
    cb_reserve_back(cb_state_out, ONE_TILE);
    cb_reserve_back(cb_state_post_update, ONE_TILE);

    tile_regs_acquire();
    add_tiles(cb_state_scaled, cb_outer, tile_index, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    // GDN pack order (qwen36_gdn_decode_owned.cpp:389): internal CB FIRST,
    // writer-bound CB LAST. Reversed order initially gave correct y (which
    // reads the internal CB via matmul_reduce) but WRONG cb_state_out
    // writer readback (cos ~0 with state_in=zeros).
    pack_tile(0, cb_state_post_update);
    pack_tile(0, cb_state_out);
    tile_regs_release();

    cb_push_back(cb_state_post_update, ONE_TILE);
    cb_push_back(cb_state_out, ONE_TILE);
}

// matmul_reduce_C_state(cb_C, cb_state_in, head_dim_tile, ssm_state_tiles,
//                       cb_y_partial)
//
// Computes (for one head_dim tile):
//   y_partial[d][j] = sum_{global s} state_in[d, s][j, k] * C[s][k]
//                   = (C · state_in^T)[d, j]
//
// where state_in[d, s] is fp32 [head_dim × ssm_state] within the tile,
// j ranges over head_dim_within_tile, and C[s] is bf16 with values in
// row 0 (the W axis carries the ssm_state index).
//
// LAYOUT — operand orientation chosen so y_partial lands in ROW-vector
// form (values in row 0), matching x[d]'s layout. This lets mode=4's
// `y[d] = y_partial[d] + D * x[d]` use ordinary `add_tiles` on
// equivalently-shaped tiles, no extra transpose.
//
// Math derivation: with mm_init transpose=1, matmul computes A @ B^T:
//   result[i, j] = sum_k(A[i, k] * B[j, k])
// Setting A = C (in0), B = state_in (in1, transposed by the flag):
//   A[i, k] = C[s, k] if i=0 else 0      (C has values only in row 0)
//   B[j, k] = state_in[d, s][j, k]
//   result[0, j] = sum_k(C[s, k] * state_in[d, s][j, k])
// Accumulating across all `ssm_state_tiles` tiles gives the full reduce
// in row 0 of dst.
//
// REUSE: direct fork of
//   qwen36_gdn_decode_owned/device/kernels/compute/qwen36_gdn_decode_owned.cpp
//   line 215: `matmul_reduce`. Structurally identical to GDN's
//   `pred = k · s_prev` reduce — that early matmul implicitly primes
//   the LLK matmul engine for the downstream transpose+matmul_outer
//   loop, matching the canonical GDN pipeline shape and dropping the
//   bare-mm_init prime workaround. See
//   [[feedback-gdn-vs-mamba2-kernel-delta]] and
//   [[feedback-mm-init-prime-required]].
//
// SDPA-style escape hatch (memory: [[feedback-sdpa-transpose-b-flag-escape-hatch]]):
// passing `transpose=1` to mm_init folds the row-to-col flip into the
// matmul itself instead of calling `transpose_wh_tile` on the data —
// no extra CB allocation, no sticky-unpacker-bit risk.
FORCE_INLINE void matmul_reduce_C_state(
    uint32_t cb_C,         // in0 (bf16, row-vector tile)
    uint32_t cb_state_in,  // in1 (fp32, full tile) — will be transposed
    uint32_t head_dim_tile,
    uint32_t ssm_state_tiles,
    uint32_t cb_y_partial) {
    constexpr uint32_t TRANSPOSE_B = 1;
    mm_init(cb_C, cb_state_in, cb_y_partial, TRANSPOSE_B);

    cb_wait_front(cb_C, ssm_state_tiles);
    cb_wait_front(cb_state_in, (head_dim_tile + 1) * ssm_state_tiles);
    cb_reserve_back(cb_y_partial, ONE_TILE);

    tile_regs_acquire();
    for (uint32_t s = 0; s < ssm_state_tiles; ++s) {
        const uint32_t state_idx = head_dim_tile * ssm_state_tiles + s;
        matmul_tiles(cb_C, cb_state_in, s, state_idx, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_y_partial);
    tile_regs_release();

    cb_push_back(cb_y_partial, ONE_TILE);
}

// mul_D_x_to(cb_x, cb_D, head_dim_tile, cb_D_x_scratch)
//
// Computes:
//   D_x[d] = D * x[d]    (D is scalar broadcast across x[d]'s row-vec)
//
// x[d] is a bf16 row-vector tile (values in row 0). D is a bf16 scalar
// tile (broadcast across all 1024 positions). Result is a row-vector
// tile (values in row 0 = D * x[d] values).
//
// REUSE: same `mul_tiles_bcast_scalar_init_short` pattern as
// mul_decay_state_to. cb_D_x_scratch will be cb_outer in mode=4
// (re-used after state-update loop drains it).
FORCE_INLINE void mul_D_x_to(uint32_t cb_x, uint32_t cb_D, uint32_t head_dim_tile, uint32_t cb_D_x_scratch) {
    reconfig_data_format(cb_x, cb_D);
    pack_reconfig_data_format(cb_D_x_scratch);
    mul_tiles_bcast_scalar_init_short(cb_x, cb_D);
    cb_wait_front(cb_x, head_dim_tile + 1);
    cb_wait_front(cb_D, ONE_TILE);
    cb_reserve_back(cb_D_x_scratch, ONE_TILE);

    tile_regs_acquire();
    mul_tiles_bcast_scalar(cb_x, cb_D, head_dim_tile, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_D_x_scratch);
    tile_regs_release();

    cb_push_back(cb_D_x_scratch, ONE_TILE);
}

// add_y_partial_D_x(cb_y_partial, cb_D_x_scratch, head_dim_tile, cb_y)
//
// Computes:
//   y[d] = y_partial[d] + D_x[d]
//
// Both inputs are bf16 row-vector tiles (values in row 0). add_tiles
// element-wise sum → row 0 = y[d] values.
//
// REUSE: same `add_tiles_init` pattern as add_state_scaled_outer
// (forked from GDN add_state_to_out).
FORCE_INLINE void add_y_partial_D_x(
    uint32_t cb_y_partial, uint32_t cb_D_x_scratch, uint32_t head_dim_tile, uint32_t cb_y) {
    reconfig_data_format(cb_y_partial, cb_D_x_scratch);
    pack_reconfig_data_format(cb_y);
    add_tiles_init(cb_y_partial, cb_D_x_scratch);
    cb_wait_front(cb_y_partial, head_dim_tile + 1);
    cb_wait_front(cb_D_x_scratch, ONE_TILE);
    cb_reserve_back(cb_y, ONE_TILE);

    tile_regs_acquire();
    add_tiles(cb_y_partial, cb_D_x_scratch, head_dim_tile, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_y);
    tile_regs_release();

    cb_push_back(cb_y, ONE_TILE);
}

// ─────────────────────────────────────────────────────────────────────────────
// kernel_main — orchestration.
//
// Compile-time args = CB index assignments (must match program_factory).
// Runtime args = per-block work + debug_mode switch.
//
// The CB index numbering keeps the GDN layout for the producer/consumer
// pipeline pattern (decision D7 — debug_mode flexibility), but the CB roles
// are Mamba2-specific:
//   cb_x          : input x  (head_dim vector)
//   cb_z          : gate (pass-through; not consumed here — decision D10)
//   cb_dt         : scalar dt
//   cb_dt_bias    : scalar dt_bias (weight)
//   cb_A_log      : scalar A_log (weight)
//   cb_D          : scalar D (weight)
//   cb_B          : per-group B (ssm_state vector)
//   cb_C          : per-group C (ssm_state vector)
//   cb_state_in   : ssm_state read-side ([head_dim, ssm_state] = 2×4 tiles)
//   cb_decay      : intermediate scalar
//   cb_dt_B       : intermediate [ssm_state] vector
//   cb_state_scaled : intermediate [head_dim, ssm_state] (decay × state)
//   cb_y_partial  : intermediate [head_dim] (C·state reduce result)
//   cb_state_out  : ssm_state write-side
//   cb_y          : output y ([head_dim])

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_z = get_compile_time_arg_val(1);
    constexpr uint32_t cb_dt = get_compile_time_arg_val(2);
    constexpr uint32_t cb_dt_bias = get_compile_time_arg_val(3);
    constexpr uint32_t cb_A_log = get_compile_time_arg_val(4);
    constexpr uint32_t cb_D = get_compile_time_arg_val(5);
    constexpr uint32_t cb_B = get_compile_time_arg_val(6);
    constexpr uint32_t cb_C = get_compile_time_arg_val(7);
    constexpr uint32_t cb_state_in = get_compile_time_arg_val(8);
    constexpr uint32_t cb_decay = get_compile_time_arg_val(9);
    constexpr uint32_t cb_dt_B = get_compile_time_arg_val(10);
    constexpr uint32_t cb_state_scaled = get_compile_time_arg_val(11);
    constexpr uint32_t cb_y_partial = get_compile_time_arg_val(12);
    constexpr uint32_t cb_state_out = get_compile_time_arg_val(13);
    constexpr uint32_t cb_y = get_compile_time_arg_val(14);
    // G1 day-4 (debug_mode=3) scratch CBs:
    constexpr uint32_t cb_x_col = get_compile_time_arg_val(15);
    constexpr uint32_t cb_outer = get_compile_time_arg_val(16);
    // G1 day-4.6 (debug_mode=5):
    constexpr uint32_t cb_state_post_update = get_compile_time_arg_val(17);

    // Runtime args:
    //   block_count       : how many (batch, head) blocks this core owns
    //   head_dim_tiles    : 2 for Nemotron-3 (head_dim=64 / TILE_W=32)
    //   ssm_state_tiles   : 4 for Nemotron-3 (ssm_state=128 / TILE_W=32)
    //   debug_mode        : 0 (production) .. 5 (incremental); see file header
    const uint32_t block_count = get_arg_val<uint32_t>(0);
    const uint32_t head_dim_tiles = get_arg_val<uint32_t>(1);
    const uint32_t ssm_state_tiles = get_arg_val<uint32_t>(2);
    const uint32_t debug_mode = get_arg_val<uint32_t>(3);
    // Softplus + clamp config (passed as float32 bits per LLK calling convention).
    //
    // CORRECTED 2026-06-05 (v0.1.2.d root-cause fix): HF Nemotron uses
    // `self.time_step_limit = (0.0, inf)` for the clamp — NOT the
    // `time_step_min`/`time_step_max` config fields (which exist but are
    // not what HF clamps against). The wrong (0.0001, 0.1) constants
    // gave dt_eff ~2.77x smaller than HF, dropping y cos to 0.943.
    // Numpy oracle verified at cos=0.999999 vs HF y_pre_norm after the
    // fix. See `feedback_nemotron3_time_step_clamp_bug.md`.
    constexpr uint32_t SOFTPLUS_BETA_BITS = 0x3f800000u;        // 1.0f
    constexpr uint32_t SOFTPLUS_BETA_RECIP_BITS = 0x3f800000u;  // 1.0f
    constexpr uint32_t SOFTPLUS_THRESHOLD_BITS = 0x41a00000u;   // 20.0f
    constexpr uint32_t TIME_STEP_FLOOR_BITS = 0x00000000u;      // 0.0f (HF: time_step_limit[0])
    constexpr uint32_t TIME_STEP_MAX_BITS = 0x7f800000u;        // +inf (HF: time_step_limit[1])
    (void)cb_z;  // not consumed in the kernel; pass-through to caller (decision D10)

    binary_op_init_common(cb_state_in, cb_decay, cb_state_scaled);

    for (uint32_t block = 0; block < block_count; ++block) {
        if (debug_mode == 1) {
            // ── Mode 1: scaffolding smoke ───────────────────────────────────
            // No compute. Drain every input CB so the reader doesn't stall,
            // then write `1.0` tiles to the output CBs (cb_y and cb_state_out).
            //
            // Purpose: validate that the program builds, the kernel dispatches,
            // the CB plumbing is correct (reader/compute/writer pipelining
            // works), and the output channel is connected to the writer.
            //
            // Gate (G0a harness): `--kernel-callable …` returns output of
            // shape `[B, num_heads=64, head_dim=64]` filled with 1.0 — easy
            // to assert. Compare cosine of all-ones vs oracle expected to be
            // not-NaN (cos will be near-zero because oracle is non-trivial).
            // Pass criterion: output exists, no NaN, shape matches.
            cb_wait_front(cb_dt, ONE_TILE);
            cb_wait_front(cb_dt_bias, ONE_TILE);
            cb_wait_front(cb_A_log, ONE_TILE);
            cb_wait_front(cb_D, ONE_TILE);
            cb_wait_front(cb_x, head_dim_tiles);
            cb_wait_front(cb_B, ssm_state_tiles);
            cb_wait_front(cb_C, ssm_state_tiles);
            cb_wait_front(cb_state_in, head_dim_tiles * ssm_state_tiles);

            // Fill output state with 1.0 (one tile per [head_dim_tile, s_tile]).
            for (uint32_t i = 0; i < head_dim_tiles * ssm_state_tiles; ++i) {
                fill_one(cb_state_out);
            }
            // Fill output y with 1.0 (one tile per head_dim_tile).
            for (uint32_t d = 0; d < head_dim_tiles; ++d) {
                fill_one(cb_y);
            }

            cb_pop_front(cb_dt, ONE_TILE);
            cb_pop_front(cb_dt_bias, ONE_TILE);
            cb_pop_front(cb_A_log, ONE_TILE);
            cb_pop_front(cb_D, ONE_TILE);
            cb_pop_front(cb_x, head_dim_tiles);
            cb_pop_front(cb_B, ssm_state_tiles);
            cb_pop_front(cb_C, ssm_state_tiles);
            cb_pop_front(cb_state_in, head_dim_tiles * ssm_state_tiles);
        } else if (debug_mode == 2) {
            // ── Mode 2: state_out = decay * state_in, no input contribution ───
            // Pipeline:
            //   compute_decay(cb_A_log → cb_decay)
            //   compute_dt_eff(cb_dt, cb_dt_bias → cb_dt_B [scratch slot])
            //   multiply_decay_by_dt_eff(cb_decay, cb_dt_B)
            //   loop: mul_decay_state_to over (head_dim, ssm_state) tiles
            //
            // cb_dt_B is reused as the dt_eff scratch (decision D11 double-duty).
            // Mode 2 doesn't need dt_eff for anything past the decay-finalize,
            // so we pop the dt_eff tile from cb_dt_B at end-of-block.
            // (At day-4 mode=3, compute_dt_B consumes the dt_eff tile and
            // pushes ssm_state_tiles dt_B tiles in its place.)
            compute_decay(cb_A_log, cb_decay);
            compute_dt_eff(
                cb_dt,
                cb_dt_bias,
                cb_dt_B,
                SOFTPLUS_BETA_BITS,
                SOFTPLUS_BETA_RECIP_BITS,
                SOFTPLUS_THRESHOLD_BITS,
                TIME_STEP_FLOOR_BITS,
                TIME_STEP_MAX_BITS);
            multiply_decay_by_dt_eff(cb_decay, cb_dt_B);

            // state has shape [head_dim_tiles, ssm_state_tiles] tiles.
            // Tile index = head_dim_tile * ssm_state_tiles + ssm_state_tile.
            // Loop order: outer over head_dim (decision D5 — decay reused).
            for (uint32_t d = 0; d < head_dim_tiles; ++d) {
                for (uint32_t s = 0; s < ssm_state_tiles; ++s) {
                    const uint32_t tile_idx = d * ssm_state_tiles + s;
                    mul_decay_state_to(cb_state_in, cb_decay, tile_idx, cb_state_out);
                }
            }

            // No y math in mode 2; sentinel fill so the writer drains its CB.
            for (uint32_t d = 0; d < head_dim_tiles; ++d) {
                fill_one(cb_y);
            }

            cb_pop_front(cb_state_in, head_dim_tiles * ssm_state_tiles);
            cb_pop_front(cb_decay, ONE_TILE);
            cb_pop_front(cb_dt_B, ONE_TILE);  // dt_eff scratch (mode=2 doesn't need it)
            // cb_dt, cb_dt_bias, cb_A_log all popped by helpers above.
        } else if (debug_mode == 3) {
            // ── Mode 3: state_out = decay * state_in + dt_eff * x ⊗ B ────────
            // y still sentinel (output reduce + D·x skip wire at mode=4+).
            //
            // Pipeline (G1 day-4):
            //   compute_decay(cb_A_log → cb_decay)                 [A scalar]
            //   compute_dt_eff(cb_dt, cb_dt_bias → cb_dt_B)        [dt_eff @ front]
            //   multiply_decay_by_dt_eff(cb_decay, cb_dt_B)        [decay scalar]
            //   compute_dt_B(cb_dt_B, cb_B, ssm_state_tiles)       [dt_B[0..3]]
            //   for d in head_dim_tiles:
            //     transpose_x_to_col(cb_x, d → cb_x_col)
            //     for s in ssm_state_tiles:
            //       mul_decay_state_to → cb_state_scaled
            //       matmul_outer_x_dt_B → cb_outer
            //       add_state_scaled_outer → cb_state_out
            //       pop cb_outer
            //     pop cb_x_col
            //   fill_one(cb_y) × head_dim_tiles
            //
            // Per-tile math:
            //   state_out[d, s][i, j] = decay * state_in[d, s][i, j]
            //                         + x[d][i] * (dt_eff * B[s])[j]
            //
            // The outer product is realized as matmul(x_col_vec, dt_B_row_vec)
            // → full 32×32 tile. Pattern forked from GDN's mul_outer /
            // matmul_outer (see helper comments).
            compute_decay(cb_A_log, cb_decay);
            compute_dt_eff(
                cb_dt,
                cb_dt_bias,
                cb_dt_B,
                SOFTPLUS_BETA_BITS,
                SOFTPLUS_BETA_RECIP_BITS,
                SOFTPLUS_THRESHOLD_BITS,
                TIME_STEP_FLOOR_BITS,
                TIME_STEP_MAX_BITS);
            multiply_decay_by_dt_eff(cb_decay, cb_dt_B);
            compute_dt_B(cb_dt_B, cb_B, ssm_state_tiles);

            // Day-4.2 restructure: split transpose and matmul into
            // separate phases to bypass the iter-count cap on
            // transpose-interleaved-with-matmul on Blackhole.
            //
            // Phase 1: pre-transpose x ONCE per d. cb_x_col ends with
            // head_dim_tiles column-vector tiles queued.
            // Phase 2: matmul_reduce_C_state — real matmul, primes
            // engine + produces y_partial for mode=4/5.
            // Phase 3: 8-iter state-update loop. NO transpose inside;
            // matmul_outer reads pre-transposed cb_x_col[d] via the
            // updated head_dim_tile arg.
            //
            // Rationale: bisect-D confirmed that 8 transpose+matmul
            // iters in succession hang regardless of mm_init prime. By
            // moving transposes into a 2-call pre-phase (well under
            // the ~4-iter cap) and using only matmul+binary ops inside
            // the loop, we sidestep the sticky-bit accumulator.
            for (uint32_t d = 0; d < head_dim_tiles; ++d) {
                transpose_x_to_col(cb_x, d, cb_x_col);
            }

            for (uint32_t d = 0; d < head_dim_tiles; ++d) {
                matmul_reduce_C_state(cb_C, cb_state_in, d, ssm_state_tiles, cb_y_partial);
            }

            for (uint32_t d = 0; d < head_dim_tiles; ++d) {
                for (uint32_t s = 0; s < ssm_state_tiles; ++s) {
                    const uint32_t tile_idx = d * ssm_state_tiles + s;
                    matmul_outer_x_dt_B(cb_x_col, cb_dt_B, d, s, cb_outer);
                    mul_decay_state_to(cb_state_in, cb_decay, tile_idx, cb_state_scaled);
                    add_state_scaled_outer(cb_state_scaled, cb_outer, tile_idx, cb_state_out);
                    cb_pop_front(cb_outer, ONE_TILE);
                }
            }

            for (uint32_t d = 0; d < head_dim_tiles; ++d) {
                fill_one(cb_y);
            }

            cb_pop_front(cb_x, head_dim_tiles);
            cb_pop_front(cb_x_col, head_dim_tiles);  // day-4.2: pre-transposed, drained here
            cb_pop_front(cb_state_in, head_dim_tiles * ssm_state_tiles);
            cb_pop_front(cb_state_scaled, head_dim_tiles * ssm_state_tiles);
            cb_pop_front(cb_decay, ONE_TILE);
            cb_pop_front(cb_dt_B, ssm_state_tiles);
            cb_pop_front(cb_C, ssm_state_tiles);
            cb_pop_front(cb_D, ONE_TILE);
            cb_pop_front(cb_z, head_dim_tiles);
            cb_pop_front(cb_y_partial, head_dim_tiles);  // mode=3 doesn't consume
        } else if (debug_mode == 4) {
            // ── Mode 4: state correct + y_partial = C·state_in^T + D·x ──
            // Same state-update pipeline as mode=3 (proven), but now consume
            // cb_y_partial (already produced by matmul_reduce_C_state) and
            // add D·x to produce the real y. mode=4 differs from production
            // mode=5 only in that y_partial uses state_in instead of state_out
            // (a fixup C·outer add at mode=5 corrects this).
            //
            // Math:
            //   state_out = decay * state_in + dt_eff * x ⊗ B   (full)
            //   y_partial = C · state_in^T                       (from matmul_reduce_C_state)
            //   y[d]      = y_partial[d] + D · x[d]              (NEW for mode=4)
            compute_decay(cb_A_log, cb_decay);
            compute_dt_eff(
                cb_dt,
                cb_dt_bias,
                cb_dt_B,
                SOFTPLUS_BETA_BITS,
                SOFTPLUS_BETA_RECIP_BITS,
                SOFTPLUS_THRESHOLD_BITS,
                TIME_STEP_FLOOR_BITS,
                TIME_STEP_MAX_BITS);
            multiply_decay_by_dt_eff(cb_decay, cb_dt_B);
            compute_dt_B(cb_dt_B, cb_B, ssm_state_tiles);

            // Phase 1: pre-transpose x (same as mode=3).
            for (uint32_t d = 0; d < head_dim_tiles; ++d) {
                transpose_x_to_col(cb_x, d, cb_x_col);
            }

            // Phase 2: matmul_reduce_C_state — primes the matmul engine
            // AND produces cb_y_partial for the mode=4 y add (consumed below,
            // NOT drained like mode=3).
            for (uint32_t d = 0; d < head_dim_tiles; ++d) {
                matmul_reduce_C_state(cb_C, cb_state_in, d, ssm_state_tiles, cb_y_partial);
            }

            // Phase 3: state-update loop (same as mode=3).
            for (uint32_t d = 0; d < head_dim_tiles; ++d) {
                for (uint32_t s = 0; s < ssm_state_tiles; ++s) {
                    const uint32_t tile_idx = d * ssm_state_tiles + s;
                    matmul_outer_x_dt_B(cb_x_col, cb_dt_B, d, s, cb_outer);
                    mul_decay_state_to(cb_state_in, cb_decay, tile_idx, cb_state_scaled);
                    add_state_scaled_outer(cb_state_scaled, cb_outer, tile_idx, cb_state_out);
                    cb_pop_front(cb_outer, ONE_TILE);
                }
            }

            // Phase 4: y[d] = y_partial[d] + D · x[d] (NEW at mode=4).
            // Reuse cb_outer as the D·x scratch (it's empty after the
            // state-update loop's per-iter pop).
            for (uint32_t d = 0; d < head_dim_tiles; ++d) {
                mul_D_x_to(cb_x, cb_D, d, cb_outer);
                add_y_partial_D_x(cb_y_partial, cb_outer, d, cb_y);
                cb_pop_front(cb_outer, ONE_TILE);
            }

            // Drain. cb_y_partial fully consumed by add_y_partial_D_x.
            cb_pop_front(cb_x, head_dim_tiles);
            cb_pop_front(cb_x_col, head_dim_tiles);
            cb_pop_front(cb_state_in, head_dim_tiles * ssm_state_tiles);
            cb_pop_front(cb_state_scaled, head_dim_tiles * ssm_state_tiles);
            cb_pop_front(cb_decay, ONE_TILE);
            cb_pop_front(cb_dt_B, ssm_state_tiles);
            cb_pop_front(cb_C, ssm_state_tiles);
            cb_pop_front(cb_D, ONE_TILE);
            cb_pop_front(cb_z, head_dim_tiles);
            cb_pop_front(cb_y_partial, head_dim_tiles);
        } else if (debug_mode == 5) {
            // ── Mode 5: PRODUCTION-EQUIVALENT y = C · state_out^T + D·x ──
            // Differs from mode=4 only in that y reflects POST-update state.
            //
            // Math: state_out = decay*state_in + outer = state_scaled + outer.
            // So C · state_out^T = C · (state_scaled + outer)^T. We capture
            // state_out per (d, s) tile in cb_state_post_update via
            // add_state_scaled_outer_two (forked from GDN's add_state_to_two),
            // then run matmul_reduce_C_state against it in Phase 4.
            //
            // Phase 2's matmul_reduce_C_state on cb_state_in is kept for
            // its prime-engine role; its 2 tiles in cb_y_partial are
            // drained at the start of Phase 4 before the correct reduce
            // overwrites them.
            compute_decay(cb_A_log, cb_decay);
            compute_dt_eff(
                cb_dt,
                cb_dt_bias,
                cb_dt_B,
                SOFTPLUS_BETA_BITS,
                SOFTPLUS_BETA_RECIP_BITS,
                SOFTPLUS_THRESHOLD_BITS,
                TIME_STEP_FLOOR_BITS,
                TIME_STEP_MAX_BITS);
            multiply_decay_by_dt_eff(cb_decay, cb_dt_B);
            compute_dt_B(cb_dt_B, cb_B, ssm_state_tiles);

            // Phase 1: pre-transpose x.
            for (uint32_t d = 0; d < head_dim_tiles; ++d) {
                transpose_x_to_col(cb_x, d, cb_x_col);
            }

            // Phase 2: matmul_reduce_C_state PRIME (on state_in, value
            // discarded later).
            for (uint32_t d = 0; d < head_dim_tiles; ++d) {
                matmul_reduce_C_state(cb_C, cb_state_in, d, ssm_state_tiles, cb_y_partial);
            }

            // Phase 3: state-update loop. add_state_scaled_outer_two
            // packs state_out to BOTH cb_state_out (writer) AND
            // cb_state_post_update (compute reads back in Phase 4).
            for (uint32_t d = 0; d < head_dim_tiles; ++d) {
                for (uint32_t s = 0; s < ssm_state_tiles; ++s) {
                    const uint32_t tile_idx = d * ssm_state_tiles + s;
                    matmul_outer_x_dt_B(cb_x_col, cb_dt_B, d, s, cb_outer);
                    mul_decay_state_to(cb_state_in, cb_decay, tile_idx, cb_state_scaled);
                    add_state_scaled_outer_two(cb_state_scaled, cb_outer, tile_idx, cb_state_out, cb_state_post_update);
                    cb_pop_front(cb_outer, ONE_TILE);
                }
            }

            // Phase 4a: drain Phase 2's naive y_partial so Phase 4b can
            // push fresh correct values without CB overflow.
            cb_pop_front(cb_y_partial, head_dim_tiles);

            // Phase 4b: correct y_partial = C · state_out^T. Use
            // cb_state_post_update (which has the full 8 post-update tiles).
            for (uint32_t d = 0; d < head_dim_tiles; ++d) {
                matmul_reduce_C_state(cb_C, cb_state_post_update, d, ssm_state_tiles, cb_y_partial);
            }

            // Phase 5: y[d] = y_partial[d] + D · x[d] (same as mode=4).
            for (uint32_t d = 0; d < head_dim_tiles; ++d) {
                mul_D_x_to(cb_x, cb_D, d, cb_outer);
                add_y_partial_D_x(cb_y_partial, cb_outer, d, cb_y);
                cb_pop_front(cb_outer, ONE_TILE);
            }

            // Drain.
            cb_pop_front(cb_x, head_dim_tiles);
            cb_pop_front(cb_x_col, head_dim_tiles);
            cb_pop_front(cb_state_in, head_dim_tiles * ssm_state_tiles);
            cb_pop_front(cb_state_scaled, head_dim_tiles * ssm_state_tiles);
            cb_pop_front(cb_state_post_update, head_dim_tiles * ssm_state_tiles);
            cb_pop_front(cb_decay, ONE_TILE);
            cb_pop_front(cb_dt_B, ssm_state_tiles);
            cb_pop_front(cb_C, ssm_state_tiles);
            cb_pop_front(cb_D, ONE_TILE);
            cb_pop_front(cb_z, head_dim_tiles);
            cb_pop_front(cb_y_partial, head_dim_tiles);
        }
    }
}
