// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// =============================================================================
//  Top-K XL — SFPU bitonic sort / merge / rebuild for K = 512, 1024, 2048
// =============================================================================
//
// This header implements the SFPU primitives that the higher-level top-k LLK
// API (`llk_math_eltwise_unary_sfpu_topk_xl_*`) dispatches to. There are three
// public entry points that the user-facing API exposes:
//
//   _topk_xl_local_sort_  : sort K elements in place.
//   _topk_xl_merge_       : bitonic-merge two adjacent K-element sorted
//                           runs into a single K-element top-K run.
//   _topk_xl_rebuild_     : restore the bitonic property of a K-element
//                           run after a merge so the next merge can run.
//
// Everything else in this file is a helper for those three.
//
// Data layout
// -----------
// Every operated-on word is 32 bits in DST. In the "fused" mode used by the
// distributed top-k op, each word carries two pieces of data:
//
//      [ bf16 value (high 16) | u16 index (low 16) ]
//
// Because the bf16 value sits in the high bits of the 32-bit DST word, the
// fused compare acts as a value compare with the index as a deterministic
// tie-breaker. Since that fused word is an opaque sort key, fused load/store
// paths move it with INT32 modes so FP32 store denormal flushing cannot erase
// low index bits. (`fused=false` keeps values and indices in two distinct DST
// regions.)
//
// Bitonic sort, briefly
// ---------------------
// A bitonic sequence is one that first ascends then descends (or vice
// versa). Sorting two such sequences of length N takes log2(N) "merge"
// passes of compare-exchange at stride N/2, N/4, ... 1. We build a fully
// sorted sequence of length 2*N from two sorted halves by a single bitonic
// merge — that's exactly the loop structure here.
//
// The functions `bitonic_sort_len_{2,4,8,16,32,k}` each perform a single
// pass of "step N" compare-exchanges, where step N means a stride of 2^(N-1).
// They build (or merge) a bitonic sequence of length 2^N. The numeric step
// labels in the source map directly to those stride exponents.
//
// SFPU execution model
// --------------------
// The SFPU has 8 lanes operating in parallel. Each `LREG` holds one element
// per lane (4 rows × 8 lanes), and `SFPSWAP` is a 4-lane parallel
// compare-exchange. `SFPTRANSP` rotates between two views of the data (rows
// ↔ columns of the 4-element lane block), and `MOVD2B / MOVB2A / MOVA2D /
// MOVB2D` shuttle 4-row strips between Dst, SrcA, and SrcB so that we can
// transpose larger faces (the `transpose_dest_face_32b` / `transpose_8_faces`
// path used for strides > 256).
//
// Replay buffer + MOP expander
// ----------------------------
// The math thread can record a window of up to 32 instructions into the
// "replay buffer" and re-issue any contiguous sub-range with `lltt::replay`.
// On top of that, the MOP expander can fire a 5-slot template per
// `TTI_MOP` issue, where each slot can itself be a REPLAY(start, len).
// We use:
//
//   * Plain replay buffers for the load/sort/store bodies of small loops
//     (`canonical_big_block_with_replay`, `_topk_xl_local_sort_` body, etc).
//   * The MOP expander for the merge inner loop, so each "column" of the
//     merge fires as a single TTI_MOP issue (`topk_mop_config`).
//   * A 5-slot MOP template for the stride-2048 rebuild whose body
//     (34 instructions) doesn't fit in one replay window, so we split it
//     across two REPLAY ranges plus two inline SFPTRANSPs in the template
//     itself (`topk_rebuild_build2048_mop_config`).
//
// Address-mod recipe (ADDR_MOD_0..7)
// ----------------------------------
// The kernel programs several ADDR_MODs at init time so that SFPSTORE can
// advance Dst by configurable strides without a separate INCRWC issue:
//
//   ADDR_MOD_7 : zero advance (used by SFPLOAD/SFPSTORE that don't move Dst).
//                Left at the kernel-startup default of all-zero increments —
//                no `topk_xl` init reprograms it.
//   ADDR_MOD_6 : +32 (one face row)
//   ADDR_MOD_5 : +16 (half face row)
//   ADDR_MOD_4 : +8     (unfused) / +16 (add_lsb_indices reuses this slot)
//   ADDR_MOD_3 : +40 = 32 + 8     (folds trailing +8 into the last store)
//   ADDR_MOD_2 : +24 = 16 + 8     (same idea at 16-stride)
//   ADDR_MOD_1 : +48 = 32 + 16    (folds trailing +16 into the last store)
//   ADDR_MOD_0 : +2  (hi16 / lo16 stride for the post-reduction phases —
//                     `remove_msb_values` and `separate_indices`. Not
//                     touched by `_topk_xl_init_` so it never collides with
//                     the +24 / +40 / +48 folds above.)
//
// "Folding" means the trailing INCRWC(0, X, ...) advance that would normally
// follow a store gets absorbed into the store's own ADDR_MOD increment,
// saving one math-thread issue per inner loop iter.
//
// Post-reduction phases (`add_lsb_indices`, `remove_msb_values`,
// `separate_indices`) reprogram a subset of slots for their own use — but
// they all run AFTER the last merge/rebuild has consumed the bitonic
// ADDR_MODs above, so the reprogramming is invisible to the hot loops.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_template.h"
#include "lltt.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_load_config.h"

namespace ckernel
{
namespace sfpu
{

// =============================================================================
//  MOP Expander programs
// =============================================================================
//
// The math-thread MOP Expander packs up to 5 instruction slots (A0..A3 + B)
// behind a single `TTI_MOP` issue, with an optional iteration count. We
// program it once at init so the merge's hot loop collapses to one MOP
// issue per "column".

// Program the MOP Expander to issue a single `REPLAY(0, body_len)` per slot.
// `_topk_xl_merge_` uses this to fire its inner loop body — recorded into
// replay slots [0, body_len) — with one MOP issue per column.
//
// Recording window sizes (must match the body recorded by the merge):
//   fused:   8 (load16) + 4 (sort_k) + 4 (store4_top_only) = 16 instructions
//   unfused: 8 (load8)  + 2 (sort_k) + 8 (store8)          = 18 instructions
//
// The fused body drops the 4 SFPSTOREs for LREG4..7 (the per-pair min
// half), which is dead by construction — see `store4_rows_top_only`.
//
// Nothing else on the math thread touches the MOP Expander, so this
// programming survives across every merge call.
template <bool fused>
inline void topk_mop_config()
{
    constexpr int body_len              = fused ? 16 : 18;
    constexpr std::uint32_t replay_body = lltt::replay_insn(0, body_len);
    ckernel_unpack_template tmpl        = ckernel_unpack_template::lA(replay_body, TT_OP_NOP);
    tmpl.program();
}

// Program the MOP Expander for the stride-2 length-2048 build phase of
// `_topk_xl_rebuild_<fused=true>`. The loop body is
//
//     load16_rows_x2<2> + bitonic_sort_len_16_alt(dir) + store16_rows_x2<2, 16>
//     = 8 + 18 + 8 = 34 instructions
//
// which is too large to fit in the 32-slot replay buffer as one contiguous
// window. The two `SFPTRANSP` instructions inside the sort are natural
// structural break points, so we record only the 32 "data" instructions
// and emit the two SFPTRANSPs from the MOP template itself (slots A1 / A3).
//
//   Replay buffer layout (recorded once at the start of the loop):
//     slots [0..15] : load16_rows_x2<2>           (8)
//                   + first 8 SFPSWAPs of sort    (8)
//     slots [16..23]: second 8 SFPSWAPs of sort   (8)
//     slots [24..31]: store16_rows_x2<2, 16>      (8)
//
//   MOP template (5 slots, unpackB + unpack_halo path):
//     A0 = REPLAY(0, 16)   ← load + first half of sort
//     A1 = SFPTRANSP       ← first transpose of the sort
//     A2 = REPLAY(16, 8)   ← second half of sort
//     A3 = SFPTRANSP       ← second transpose of the sort
//     B  = REPLAY(24, 8)   ← store
//
// One `TTI_MOP` issue per iter replaces 2 REPLAY + 18 inline sort = 20
// math-thread issues. Clobbers the merge MOP setup, so callers must run
// `topk_mop_config<true>()` again before the next `_topk_xl_merge_`.
inline void topk_rebuild_build2048_mop_config()
{
    constexpr std::uint32_t replay_load_and_first_swaps = lltt::replay_insn(0, 16);
    constexpr std::uint32_t replay_second_swaps         = lltt::replay_insn(16, 8);
    constexpr std::uint32_t replay_store                = lltt::replay_insn(24, 8);
    constexpr std::uint32_t sfptransp_op                = TT_OP_SFPTRANSP(0, 0, 0, 0);

    // Build the 5-slot template directly (unpackB + unpack_halo both enabled),
    // bypassing the `lA` / `lBhA` factories whose default A0..A3 / B values
    // are UNPACR instructions — we need arbitrary instruction words here.
    ckernel_unpack_template tmpl(
        /*unpackB=*/true,
        /*unpack_halo=*/true,
        /*A0=*/replay_load_and_first_swaps,
        /*A1=*/sfptransp_op,
        /*A2=*/replay_second_swaps,
        /*A3=*/sfptransp_op,
        /*skipA=*/TT_OP_NOP,
        /*B=*/replay_store,
        /*skipB=*/TT_OP_NOP);
    tmpl.program();
}

// =============================================================================
//  Init
// =============================================================================
//
// Programs the ADDR_MODs (see "Address-mod recipe" at the top of the file)
// and the merge MOP Expander. Must be called once per query before any of
// the three public entry points. The same set of ADDR_MODs / MOP
// programming is used by every supported K — the generic K=512/1024
// bodies have been ported onto the same replay / MOP / ADDR_MOD recipe
// the K=2048 fast path uses, so they no longer need a separate, lighter
// init.
//
// In `fused` mode values and indices share a single 32-bit DST word, so
// no separate index-tracking path needs to be programmed. The two
// stride-folding ADDR_MODs (`ADDR_MOD_2`, `ADDR_MOD_3`) that only the
// unfused path consumes — plus `ADDR_MOD_4`'s +8 step also unused fused
// — are skipped in that case, saving ~9 CFG writes per init.
template <std::uint32_t K, bool fused>
inline void _topk_xl_init_()
{
    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");

    // ADDR_MOD_6 — +32 (one whole face row).
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 32},
    }
        .set(ADDR_MOD_6);

    // ADDR_MOD_5 — +16 (half a face row).
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 16},
    }
        .set(ADDR_MOD_5);

    // ADDR_MOD_2/3/4 are only consumed by the unfused path (the unfused
    // store helper plus the unfused rebuild's stride-32 / stride-16 inline
    // LREG7 tails). Fused merge / rebuild / local_sort never reference
    // them, so we skip the writes when fused — saves about 9 CFG writes.
    if constexpr (!fused)
    {
        // ADDR_MOD_4 — +8 (used by the unfused store helper).
        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 8},
        }
            .set(ADDR_MOD_4);

        // ADDR_MOD_3 — +40 = 32 + 8. Folds a trailing INCRWC(+8) into the
        // previous SFPSTORE's address increment, saving one math-thread
        // issue per outer iter of the unfused rebuild's stride-32 loop.
        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 40},
        }
            .set(ADDR_MOD_3);

        // ADDR_MOD_2 — +24 = 16 + 8. Same idea as ADDR_MOD_3 but for the
        // unfused rebuild's stride-16 loop.
        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 24},
        }
            .set(ADDR_MOD_2);
    }

    // ADDR_MOD_1 — +48 = 32 + 16. Folds the trailing +16 advance into the
    // last SFPSTORE of sub-block B's inner pair. Pays off in every spot
    // where sub-block B or the len=64 phase runs (both K=2048 and the
    // optimised K=1024 generic body).
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 48},
    }
        .set(ADDR_MOD_1);

    if constexpr (fused == false)
    {
        // Enable SFPU index-tracking mode (bit [2] of SFPU_CONTROL_REG)
        // so the unfused path can keep indices in a parallel DST region.
        _sfpu_load_config32_(0xF, 0x0, 0x4);
    }

    // Program the MOP Expander so the merge's inner loop can fire with a
    // single MOP issue per column — works the same way for every K, only
    // the per-column `n_iters` differs.
    topk_mop_config<fused>();
}

// =============================================================================
//  Load / store helpers
// =============================================================================
//
// All load/store helpers fill or drain 8 LREGs (LREG0..LREG7) per call.
// The "x2" variants split the 8 LREGs into two groups of 4 separated by
// `group_2_offset` so a single call covers two interleaved 4-row strips
// (the natural unit at every level of the bitonic sort).
//
// The fused / unfused split:
//   * `load16_rows_x2 / store16_rows_x2` — fused path. All 8 LREGs hold the
//     packed [value | index] INT32 payload.
//   * `load8_rows_x2_unfused / store8_rows_x2_unfused` — unfused path. The
//     first 4 LREGs hold FP32 values; the last 4 hold the matching INT32
//     indices read from a parallel DST region at `indices_offset`.
//
// `inc_dst_addr` selects which ADDR_MOD the final SFPSTORE uses, so the
// caller can fold a trailing INCRWC into the last store. See "Address-mod
// recipe" at the top of the file.

// Rebase the Dst write pointer for subsequent SFPSTOREs. Used by the
// top-level functions to switch between the even and odd columns of the
// two-tile DST region (offsets +0 and +2 from the tile base).
inline void set_dst_write_addr_offset(std::uint32_t addr)
{
    std::uint32_t dst_index = addr + get_dest_buffer_base();
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, dst_index);
}

// Load 16 rows × 2 strips into LREG0..LREG7 (fused path).
//   group 1: LREG0..3 at base+{0,4,8,12}
//   group 2: LREG4..7 at base+group_2_offset+{0,4,8,12}
template <int group_2_offset = 16>
inline void load16_rows_x2()
{
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, 4);
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::INT32, ADDR_MOD_7, 8);
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::INT32, ADDR_MOD_7, 12);
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, group_2_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_7, group_2_offset + 4);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_7, group_2_offset + 8);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_7, group_2_offset + 12);
}

// Store LREG0..LREG7 back to Dst (fused path), mirror of `load16_rows_x2`.
//
// `inc_dst_addr` picks the ADDR_MOD used by the LREG7 store, which is what
// folds a trailing Dst advance into the store itself:
//   0  → no advance
//   16 → +16 (one half face row)
//   32 → +32 (one face row)
//   48 → +48 = 32 + 16 (folds the trailing +16 INCRWC into the store)
template <int group_2_offset = 16, int inc_dst_addr = 0>
inline void store16_rows_x2()
{
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, 4);
    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::INT32, ADDR_MOD_7, 8);
    TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::INT32, ADDR_MOD_7, 12);
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, group_2_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_7, group_2_offset + 4);
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_7, group_2_offset + 8);
    if constexpr (inc_dst_addr == 48)
    {
        TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_1, group_2_offset + 12);
    }
    else if constexpr (inc_dst_addr == 32)
    {
        TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_6, group_2_offset + 12);
    }
    else if constexpr (inc_dst_addr == 16)
    {
        TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_5, group_2_offset + 12);
    }
    else if constexpr (inc_dst_addr == 0)
    {
        TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_7, group_2_offset + 12);
    }
    else
    {
        static_assert(false, "Invalid inc_dst_addr, must be 0, 16, 32, or 48");
    }
}

// Store only the "top half" of an 8-LREG block — LREG0..LREG3 at offsets
// {0, 4, 8, 12}. Used by `_topk_xl_merge_` (fused) where the bitonic
// compare-exchange has placed the per-pair max into LREG0..LREG3 and the
// per-pair min into LREG4..LREG7. The min half is dead by construction
// — `_topk_xl_rebuild_<fused=true>` only transposes / reads the value
// faces at offsets 0..112 (i.e., DST tiles 0..1), and the next merge
// stage overwrites tiles 2..3 via `topk_xl_copy_tile(recv, 2, ...)` —
// so the 4 SFPSTOREs that would have written the min half to offsets
// 128..140 can be skipped. Saves 4 SFPU instructions per merge body
// iter (20 → 16 instructions), or ~320 SFPU instructions across the
// 5-stage merge tree at N = 65536.
//
// `inc_dst_addr` selects the ADDR_MOD on the LREG3 store so the caller
// can fold the trailing +16 INCRWC into the last SFPSTORE — same idea
// as `store16_rows_x2`'s last-store ADDR_MOD select, just shifted from
// LREG7 to LREG3 because LREG7's store has been dropped.
template <int inc_dst_addr = 0>
inline void store4_rows_top_only()
{
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, 4);
    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::INT32, ADDR_MOD_7, 8);
    if constexpr (inc_dst_addr == 16)
    {
        TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::INT32, ADDR_MOD_5, 12);
    }
    else if constexpr (inc_dst_addr == 0)
    {
        TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::INT32, ADDR_MOD_7, 12);
    }
    else
    {
        static_assert(false, "store4_rows_top_only: inc_dst_addr must be 0 or 16");
    }
}

// The first 7 stores of `store16_rows_x2`. The 8th store differs only in
// which ADDR_MOD it carries, so the caller emits it inline as a one-issue
// tail after replaying the recorded 7 stores. Used by the rebuild's
// split-record stride-32 / stride-16 loops where each outer iter needs a
// different last-store ADDR_MOD.
template <int group_2_offset = 16>
inline void store_first_7_rows_x2()
{
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, 4);
    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::INT32, ADDR_MOD_7, 8);
    TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::INT32, ADDR_MOD_7, 12);
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, group_2_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_7, group_2_offset + 4);
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_7, group_2_offset + 8);
}

// Unfused-path load: LREG0..3 carry 4 rows of FP32 values, LREG4..7 carry
// the corresponding 4 rows of INT32 indices stored at `indices_offset` in
// the parallel DST region.
template <int indices_offset = 256, int group_2_offset = 16>
inline void load8_rows_x2_unfused()
{
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_7, 0);
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_7, 4);
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::FP32, ADDR_MOD_7, group_2_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::FP32, ADDR_MOD_7, group_2_offset + 4);
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, indices_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_7, indices_offset + 4);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_7, indices_offset + group_2_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_7, indices_offset + group_2_offset + 4);
}

// Split-tail counterpart of `store_first_7_rows_x2` for the unfused path.
// Same rationale: callers that want a different last-store ADDR_MOD per
// outer iter record these 7 stores and emit the LREG7 store inline.
template <int indices_offset = 256, int group_2_offset = 16>
inline void store_first_7_rows_x2_unfused()
{
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_7, 0);
    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_7, 4);
    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::FP32, ADDR_MOD_7, group_2_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::FP32, ADDR_MOD_7, group_2_offset + 4);
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, indices_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_7, indices_offset + 4);
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_7, indices_offset + group_2_offset + 0);
}

// Unfused-path store, mirror of `load8_rows_x2_unfused`. The LREG7 store
// (the index half) is the one that picks an ADDR_MOD based on
// `inc_dst_addr`, exactly like the fused `store16_rows_x2` does for values.
//   0 / 8 / 16 / 24 / 32 / 40 → no advance / +8 / +16 / +24 / +32 / +40
template <int indices_offset = 256, int group_2_offset = 16, int inc_dst_addr = 0>
inline void store8_rows_x2_unfused()
{
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_7, 0);
    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_7, 4);
    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::FP32, ADDR_MOD_7, group_2_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::FP32, ADDR_MOD_7, group_2_offset + 4);
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, indices_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_7, indices_offset + 4);
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_7, indices_offset + group_2_offset + 0);
    if constexpr (inc_dst_addr == 40)
    {
        TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_3, indices_offset + group_2_offset + 4);
    }
    else if constexpr (inc_dst_addr == 32)
    {
        TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_6, indices_offset + group_2_offset + 4);
    }
    else if constexpr (inc_dst_addr == 24)
    {
        TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_2, indices_offset + group_2_offset + 4);
    }
    else if constexpr (inc_dst_addr == 16)
    {
        TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_5, indices_offset + group_2_offset + 4);
    }
    else if constexpr (inc_dst_addr == 8)
    {
        TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_4, indices_offset + group_2_offset + 4);
    }
    else if constexpr (inc_dst_addr == 0)
    {
        TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_7, indices_offset + group_2_offset + 4);
    }
    else
    {
        static_assert(false, "Invalid inc_dst_addr");
    }
}

// =============================================================================
//  Bitonic sort primitives
// =============================================================================
//
// Each `bitonic_sort_len_X` performs the compare-exchange passes that turn a
// pair of sorted length-X/2 sub-sequences into a sorted length-X sequence
// (or, when called as part of a build, the bitonic-merge step at distance
// X/2, X/4, ..., 1). The `Step N` comments inside each function refer to
// the bitonic step exponent: Step N means "compare-exchange at stride
// 2^(N-1)". Steps are run from largest to smallest.
//
// All swap directions are encoded by the choice of operand order to
// `SFPSWAP` — `(a, b, MAX)` puts max(a, b) into `b` and min(a, b) into `a`.
// Flipping the operand order flips the sort direction.
//
// Special variants:
//   * `bitonic_sort_len_16_alt`        — the "alt" variant used by the
//     stride-2048 build inside rebuild. Splits the work in half so each
//     half fits in a 32-slot replay buffer (see
//     `topk_rebuild_build2048_mop_config`).
//   * `bitonic_sort_len_k(<fused>)`    — the smallest-step merge body. In
//     fused mode all 8 LREGs participate; in unfused mode only the first 4
//     (values) participate, indices ride along through their parallel DST
//     region.

// Sort length-2: pairwise compare-exchange of 4 lane pairs.
//
// Direction is fixed (`MAX` into the second operand) — direction handling
// is folded into the `bitonic_sort_len_4` call that always follows.
//
// Skips the trailing `SFPTRANSP` because it fuses with `bitonic_sort_len_4`,
// which expects the same SrcA layout.
inline void bitonic_sort_len_2()
{
    TTI_SFPTRANSP(0, 0, 0, 0);

    // Step 1 — stride 1.
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG6, p_sfpswap::ALL_ROWS_MAX);

    // No trailing SFPTRANSP — `bitonic_sort_len_4` will reuse the layout.
}

// Sort length-4: Step 2 (stride 2) + Step 1 (stride 1), in that order. The
// leading SFPTRANSP is omitted because this routine is only called right
// after `bitonic_sort_len_2`, which left the data in the right layout.
inline void bitonic_sort_len_4(bool ascending)
{
    if (ascending)
    {
        // Step 2 — stride 2.
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG0, p_sfpswap::ROWS_02_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG1, p_sfpswap::ROWS_02_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG4, p_sfpswap::ROWS_02_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG5, p_sfpswap::ROWS_02_MAX);

        // Step 1 — stride 1.
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, p_sfpswap::ROWS_02_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ROWS_02_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG4, p_sfpswap::ROWS_02_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG6, p_sfpswap::ROWS_02_MAX);
    }
    else
    {
        // Step 2 — stride 2.
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ROWS_02_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG6, p_sfpswap::ROWS_02_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG7, p_sfpswap::ROWS_02_MAX);

        // Step 1 — stride 1.
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ROWS_02_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG5, p_sfpswap::ROWS_02_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG7, p_sfpswap::ROWS_02_MAX);
    }

    TTI_SFPTRANSP(0, 0, 0, 0);
}

// Sort length-8: Step 3 (stride 4) + transpose + Steps 2 & 1.
// The transpose between Step 3 and Step 2 is needed because Step 2 / Step 1
// operate on a different axis of the lane block than Step 3.
inline void bitonic_sort_len_8(bool ascending)
{
    if (ascending)
    {
        // Step 3 — stride 4.
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG4, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG7, p_sfpswap::ALL_ROWS_MAX);

        TTI_SFPTRANSP(0, 0, 0, 0);

        // Step 2 — stride 2.
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG0, p_sfpswap::ROWS_01_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG1, p_sfpswap::ROWS_01_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG4, p_sfpswap::ROWS_01_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG5, p_sfpswap::ROWS_01_MAX);

        // Step 1 — stride 1.
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, p_sfpswap::ROWS_01_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ROWS_01_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG4, p_sfpswap::ROWS_01_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG6, p_sfpswap::ROWS_01_MAX);

        TTI_SFPTRANSP(0, 0, 0, 0);
    }
    else
    {
        // Step 3 — stride 4.
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG6, p_sfpswap::ALL_ROWS_MAX);

        TTI_SFPTRANSP(0, 0, 0, 0);

        // Step 2 — stride 2.
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ROWS_01_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ROWS_01_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG6, p_sfpswap::ROWS_01_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG7, p_sfpswap::ROWS_01_MAX);

        // Step 1 — stride 1.
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ROWS_01_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ROWS_01_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG5, p_sfpswap::ROWS_01_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG7, p_sfpswap::ROWS_01_MAX);

        TTI_SFPTRANSP(0, 0, 0, 0);
    }
}

// Sort length-16: two passes of (Step 4 + Step 3) separated by a transpose.
// Direction is fixed (used only by the local sort's build phase where
// direction comes from the surrounding loop's ascending/descending toggle).
inline void bitonic_sort_len_16()
{
    // Step 4 — stride 8.
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG4, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);

    // Step 3 — stride 4.
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG4, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG6, p_sfpswap::ALL_ROWS_MAX);

    TTI_SFPTRANSP(0, 0, 0, 0);

    // Step 4 — stride 8 (second half, after the layout flip).
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG4, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);

    // Step 3 — stride 4 (second half).
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG4, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG6, p_sfpswap::ALL_ROWS_MAX);

    TTI_SFPTRANSP(0, 0, 0, 0);
}

// Half of `bitonic_sort_len_16_alt` — performs Step 4 + Step 3 with the
// operand order flipped relative to `bitonic_sort_len_16` (the "alt"
// variant). Emits 8 SFPSWAPs when fused (all 8 LREGs participate) or 4
// when unfused (only the value LREGs participate; indices ride through
// the parallel DST region).
//
// Extracted so the rebuild's stride-2 build can record just the swap
// halves into the replay buffer and emit the two SFPTRANSPs from the MOP
// body itself (see `topk_rebuild_build2048_mop_config`).
template <bool fused = true>
inline void bitonic_sort_len_16_alt_swaps(bool ascending)
{
    if (ascending)
    {
        // Step 4 — stride 8.
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
        if constexpr (fused)
        {
            TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG4, p_sfpswap::ALL_ROWS_MAX);
            TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);
        }

        // Step 3 — stride 4.
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
        if constexpr (fused)
        {
            TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG4, p_sfpswap::ALL_ROWS_MAX);
            TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG6, p_sfpswap::ALL_ROWS_MAX);
        }
    }
    else
    {
        // Step 4 — stride 8.
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
        if constexpr (fused)
        {
            TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG6, p_sfpswap::ALL_ROWS_MAX);
            TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG7, p_sfpswap::ALL_ROWS_MAX);
        }

        // Step 3 — stride 4.
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
        if constexpr (fused)
        {
            TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);
            TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG7, p_sfpswap::ALL_ROWS_MAX);
        }
    }
}

// Sort length-16 (alt variant): two halves of (Step 4 + Step 3) separated
// by a transpose, ending with another transpose. See
// `bitonic_sort_len_16_alt_swaps` for why this is split.
template <bool fused = true>
inline void bitonic_sort_len_16_alt(bool ascending)
{
    bitonic_sort_len_16_alt_swaps<fused>(ascending);
    TTI_SFPTRANSP(0, 0, 0, 0);
    bitonic_sort_len_16_alt_swaps<fused>(ascending);
    TTI_SFPTRANSP(0, 0, 0, 0);
}

// Sort length-32: Step 5 + Steps 4 & 3 + transpose + Steps 4 & 3 again.
// Used to build len=32 bitonic runs from sorted len=16 sub-runs.
inline void bitonic_sort_len_32(bool ascending)
{
    if (ascending)
    {
        // Step 5 — stride 16.
        TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);

        // Step 4 — stride 8.
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG4, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);

        // Step 3 — stride 4.
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG4, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG6, p_sfpswap::ALL_ROWS_MAX);

        TTI_SFPTRANSP(0, 0, 0, 0);

        // Step 4 — stride 8 (second pass).
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG4, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);

        // Step 3 — stride 4 (second pass).
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG4, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG6, p_sfpswap::ALL_ROWS_MAX);

        TTI_SFPTRANSP(0, 0, 0, 0);
    }
    else
    {
        // Step 5 — stride 16.
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG4, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG6, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG7, p_sfpswap::ALL_ROWS_MAX);

        // Step 4 — stride 8.
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG6, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG7, p_sfpswap::ALL_ROWS_MAX);

        // Step 3 — stride 4.
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG7, p_sfpswap::ALL_ROWS_MAX);

        TTI_SFPTRANSP(0, 0, 0, 0);

        // Step 4 — stride 8 (second pass).
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG6, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG7, p_sfpswap::ALL_ROWS_MAX);

        // Step 3 — stride 4 (second pass).
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG7, p_sfpswap::ALL_ROWS_MAX);

        TTI_SFPTRANSP(0, 0, 0, 0);
    }
}

// Smallest-stride merge step (Step 6 — stride 32). Pairs the lower 4 LREGs
// with the upper 4 LREGs across the full lane block.
//
// In fused mode all 8 LREGs participate (the packed value+index moves
// together). In unfused mode only the value LREGs (0..3) are swapped here;
// the matching index swaps are issued by the surrounding code through the
// parallel DST region at `indices_offset`.
template <bool fused = true>
inline void bitonic_sort_len_k(bool ascending)
{
    if constexpr (fused)
    {
        if (ascending)
        {
            // Step 6 — stride 32.
            TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX);
            TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
            TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
            TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
        }
        else
        {
            // Step 6 — stride 32.
            TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG4, p_sfpswap::ALL_ROWS_MAX);
            TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);
            TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG6, p_sfpswap::ALL_ROWS_MAX);
            TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG7, p_sfpswap::ALL_ROWS_MAX);
        }
    }
    else
    {
        if (ascending)
        {
            // Step 6 — stride 32 (values only; indices handled separately).
            TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX);
            TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
        }
        else
        {
            // Step 6 — stride 32 (values only; indices handled separately).
            TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
            TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
        }
    }
}

// =============================================================================
//  Dst transpose helpers
// =============================================================================
//
// The bitonic build steps for lengths > 256 need to view the same 2048-word
// region with rows and columns swapped (otherwise the stride-2 / stride-8
// compare-exchange would have to address Dst words non-sequentially).
// These helpers transpose a 16×16 face of 32-bit Dst data by shuttling
// half-words through SrcB and back, using TRNSPSRCB to rotate the SrcB
// view between passes.
//
// The 32-bit transpose is done in two passes through SrcB[16:31]:
//   Pass 1 (lo16) : copy the low 16 bits of each Dst word into SrcA via
//                   SrcB+TRNSPSRCB+MOVB2A, then write them back to Dst.
//   Pass 2 (hi16) : same shuffle for the high 16 bits, then write back to
//                   Dst with Fp32_enabled=0 so the writeback preserves the
//                   already-stored hi16 half.
//
// Entry/exit invariant: Fp32_enabled = 1 (needed for 32-bit Dst reads).

// Transpose one 16x16 face of 32-bit Dst data at `dst_offset`.
template <int dst_offset>
inline void transpose_dest_face_32b()
{
    // Transpose low 16 bits and backup it in SrcA
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(to_underlying(DataFormat::Float16_b));
    TTI_MOVD2B(1, 16, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 0);
    TTI_MOVD2B(1, 20, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 4);
    TTI_MOVD2B(1, 24, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 8);
    TTI_MOVD2B(1, 28, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 12);

    TTI_TRNSPSRCB;

    TTI_MOVB2A(0, ADDR_MOD_7, p_movb2a::MOV_4_ROWS, 16);
    TTI_MOVB2A(4, ADDR_MOD_7, p_movb2a::MOV_4_ROWS, 20);
    TTI_MOVB2A(8, ADDR_MOD_7, p_movb2a::MOV_4_ROWS, 24);
    TTI_MOVB2A(12, ADDR_MOD_7, p_movb2a::MOV_4_ROWS, 28);

    // --- Pass 1: hi16 ---
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(to_underlying(DataFormat::Tf32));

    TTI_MOVD2B(0, 16, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 0);
    TTI_MOVD2B(0, 20, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 4);
    TTI_MOVD2B(0, 24, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 8);
    TTI_MOVD2B(0, 28, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 12);

    TTI_TRNSPSRCB;

    TTI_MOVB2D(0, 16, ADDR_MOD_7, p_movb2d::MOV_4_ROWS, dst_offset + 0);
    TTI_MOVB2D(0, 20, ADDR_MOD_7, p_movb2d::MOV_4_ROWS, dst_offset + 4);
    TTI_MOVB2D(0, 24, ADDR_MOD_7, p_movb2d::MOV_4_ROWS, dst_offset + 8);
    TTI_MOVB2D(0, 28, ADDR_MOD_7, p_movb2d::MOV_4_ROWS, dst_offset + 12);

    // lo16 writeback: Fp32_enabled=0 + SrcA=Float32 -> writes lo16, preserves hi16
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(0);
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(to_underlying(DataFormat::Float32));

    TTI_MOVA2D(1, 0, ADDR_MOD_7, p_mova2d::MOV_8_ROWS, dst_offset + 0);
    TTI_MOVA2D(1, 8, ADDR_MOD_7, p_mova2d::MOV_8_ROWS, dst_offset + 8);

    // Restore Fp32_enabled=1 for subsequent 32-bit Dst access
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(1);
}

// `enter_transpose_cfg_block` / `leave_transpose_cfg_block` wrap a group of
// `transpose_8_faces` calls in a single CFG mode switch. Both `transpose_*`
// helpers need:
//   * `DISABLE_IMPLIED_SRCA_FMT_Base_ADDR32 = 1` so the MOVD2B-style ops
//     read the explicit SrcA format rather than the implied one.
//   * `ALU_ACC_CTRL_Zero_Flag_disabled_src = 1` so the SrcB → SrcA copy
//     preserves the upper half-word during the two-pass transpose.
//
// The bitonic work that runs between transposes does not touch either bit,
// so wrapping the whole batch saves 4 CFG ops per transpose call.
inline void enter_transpose_cfg_block()
{
    TTI_SETC16(DISABLE_IMPLIED_SRCA_FMT_Base_ADDR32, 1);
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(1);
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_dst_RMW>(1);
}

inline void leave_transpose_cfg_block()
{
    TTI_SETC16(DISABLE_IMPLIED_SRCA_FMT_Base_ADDR32, 0);
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(0);
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_dst_RMW>(0);
}

// Transpose all 8 value faces (and, for the unfused path, all 8 index
// faces). When the caller has already opened a transpose CFG block via
// `enter_transpose_cfg_block`, pass `manage_outer_cfg=false` to skip the
// (redundant) CFG ops at the entry and exit of this helper.
template <bool fused = true, int indices_offset = 256, bool manage_outer_cfg = true>
inline void transpose_8_faces()
{
    if constexpr (manage_outer_cfg)
    {
        enter_transpose_cfg_block();
    }

    // 8 value faces at +0, +16, ..., +112.
    transpose_dest_face_32b<0>();
    transpose_dest_face_32b<16>();
    transpose_dest_face_32b<32>();
    transpose_dest_face_32b<48>();
    transpose_dest_face_32b<64>();
    transpose_dest_face_32b<80>();
    transpose_dest_face_32b<96>();
    transpose_dest_face_32b<112>();
    if constexpr (!fused)
    {
        // Unfused mode keeps the indices in a parallel DST region. Transpose
        // the 8 index faces so they match the (now-transposed) values.
        transpose_dest_face_32b<indices_offset + 0>();
        transpose_dest_face_32b<indices_offset + 16>();
        transpose_dest_face_32b<indices_offset + 32>();
        transpose_dest_face_32b<indices_offset + 48>();
        transpose_dest_face_32b<indices_offset + 64>();
        transpose_dest_face_32b<indices_offset + 80>();
        transpose_dest_face_32b<indices_offset + 96>();
        transpose_dest_face_32b<indices_offset + 112>();
    }

    if constexpr (manage_outer_cfg)
    {
        leave_transpose_cfg_block();
    }
}

// Generalised face-transpose for the K=512 / K=1024 / K=2048 paths. `N`
// is the number of value faces to transpose (2 / 4 / 8 — corresponds to
// `row_scale_factor * 2` in the local_sort / rebuild bodies). With
// `manage_outer_cfg=true` (the default) this helper opens and closes its
// own transpose CFG block; pass `false` when the caller is wrapping
// several `transpose_N_faces` calls in one shared
// `enter_transpose_cfg_block` / `leave_transpose_cfg_block` pair.
//
// `transpose_8_faces<...>` above is a thin alias for the `N=8` instance
// (kept for the K=2048 fast path's existing call sites); the `N != 8`
// instances are reached from the generic paths.
template <int N, bool fused = true, int indices_offset = 256, bool manage_outer_cfg = true>
inline void transpose_N_faces()
{
    if constexpr (manage_outer_cfg)
    {
        enter_transpose_cfg_block();
    }

    transpose_dest_face_32b<0>();
    transpose_dest_face_32b<16>();
    if constexpr (N > 2)
    {
        transpose_dest_face_32b<32>();
        transpose_dest_face_32b<48>();
    }
    if constexpr (N > 4)
    {
        transpose_dest_face_32b<64>();
        transpose_dest_face_32b<80>();
        transpose_dest_face_32b<96>();
        transpose_dest_face_32b<112>();
    }
    if constexpr (!fused)
    {
        transpose_dest_face_32b<indices_offset + 0>();
        transpose_dest_face_32b<indices_offset + 16>();
        if constexpr (N > 2)
        {
            transpose_dest_face_32b<indices_offset + 32>();
            transpose_dest_face_32b<indices_offset + 48>();
        }
        if constexpr (N > 4)
        {
            transpose_dest_face_32b<indices_offset + 64>();
            transpose_dest_face_32b<indices_offset + 80>();
            transpose_dest_face_32b<indices_offset + 96>();
            transpose_dest_face_32b<indices_offset + 112>();
        }
    }

    if constexpr (manage_outer_cfg)
    {
        leave_transpose_cfg_block();
    }
}

// =============================================================================
//  Canonical "big block" (A + B + C)
// =============================================================================
//
// Every bitonic build phase whose per-column "tail" runs through the
// stride-64 / stride-32 / stride-16 cascade ends up calling this helper.
// It is shared between the K=2048 fast path, the K=1024 generic path, and
// the K=512 generic path; which sub-blocks are emitted is controlled by
// `row_scale_factor`:
//
//   row_scale_factor == 4 (K=2048): sub-blocks A + B + C
//   row_scale_factor == 2 (K=1024): sub-blocks B + C
//   row_scale_factor == 1 (K=512):  sub-block C only (single iter)
//
// Sub-block A only kicks in when the per-column run is deep enough to need
// a stride-64 compare-exchange — K=2048 is the only such case. For K=1024
// the column run is half as deep, so stride-32 is the first scale; for
// K=512 only the stride-16 tail runs.
//
// Replay-buffer layout (re-recorded fresh in sub-blocks A and C because
// the load/store immediates differ; sub-block B is left inline since
// recording it doesn't pay back at this size). Recording is `<Exec>` and
// IS iter 0 of each inner loop:
//   sub-block A: slots [0..7]  = load16_rows_x2<64>
//                slots [8..15] = store16_rows_x2<64, 16>
//   sub-block C: slots [0..7]  = load16_rows_x2<16>
//                slots [8..15] = store16_rows_x2<16, 32>
//
// `dir` is runtime so the sort calls stay inline (the conditional inside
// them would not fold under replay anyway).
//
// Codegen note: for `row_scale_factor == 4` the body below is bit-for-bit
// identical to the previous non-templated `canonical_big_block_with_replay`
// — the `if constexpr` branches collapse to the same instruction stream.
template <int row_scale_factor>
inline void canonical_big_block_with_replay(bool dir)
{
    constexpr int consecutive_32_offset = 16;

    if constexpr (row_scale_factor >= 4)
    {
        // ── Sub-block A: `row_scale_factor` × (load<64> + sort_k + store<64, 16>) ──
        load_replay_buf<Exec>(0, 8, [] { load16_rows_x2<64>(); });
        bitonic_sort_len_k(dir);
        load_replay_buf<Exec>(8, 8, [] { store16_rows_x2<64, 16>(); });
        for (int i = 1; i < row_scale_factor; i++)
        {
            lltt::replay(0, 8);
            bitonic_sort_len_k(dir);
            lltt::replay(8, 8);
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    }

    if constexpr (row_scale_factor >= 2)
    {
        // ── Sub-block B: `row_scale_factor / 2` × paired (load<32> + sort_k + store<32, 16/48>) ──
        // The +48 store on the second of each pair uses ADDR_MOD_1 to fold
        // the trailing +16 advance into the last SFPSTORE — saves one
        // math-thread issue per inner iter compared to the original
        // K!=2048 generic body which emitted `store<32, 32> + 2×INCRWC(+8)`.
        for (int i = 0; i < (row_scale_factor >> 1); i++)
        {
            load16_rows_x2<32>();
            bitonic_sort_len_k(dir);
            store16_rows_x2<32, 16>();
            load16_rows_x2<32>();
            bitonic_sort_len_k(dir);
            store16_rows_x2<32, 48>();
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

        // ── Sub-block C: `row_scale_factor` × (load<16> + sort_32 + store<16, 32>) ──
        load_replay_buf<Exec>(0, 8, [] { load16_rows_x2<consecutive_32_offset>(); });
        bitonic_sort_len_32(dir);
        load_replay_buf<Exec>(8, 8, [] { store16_rows_x2<consecutive_32_offset, 32>(); });
        for (int i = 1; i < row_scale_factor; i++)
        {
            lltt::replay(0, 8);
            bitonic_sort_len_32(dir);
            lltt::replay(8, 8);
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    }
    else
    {
        // K=512 (row_scale_factor=1): single stride-16 sort_32 inline.
        // No replay buffer here because there's nothing to amortise the
        // recording across.
        load16_rows_x2<consecutive_32_offset>();
        bitonic_sort_len_32(dir);
        store16_rows_x2<consecutive_32_offset, 32>();
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    }
}

// =============================================================================
//  Public entry: local sort
// =============================================================================
//
// Sort 2048 elements (two consecutive 1024-element DST tiles starting at
// `dst_index`) in place. Each element is the packed u32
// `[ bf16 value | u16 index ]`.
//
// The sort proceeds bottom-up, doubling the run length at each phase:
//
//   len  =  32 : build inside each face        (sort_8 + sort_16 + sort_32)
//   len  =  64 : merge two len-32 runs         (sort_k at stride 64)
//   len = 128 : merge two len-64 runs          (canonical A+B+C big block)
//   len = 256 : merge two len-128 runs         (uses SFPCONFIG dir alternation)
//   len = 512 / 1024 / 2048 : need transposes  (data straddles whole faces)
//
// The "two columns" loop variable `col` (= 0, 1) reflects the fact that
// each DST tile has two 16-wide column groups that have to be sorted
// independently before being merged across columns.
//
// `set_dst_write_addr_offset(tile_offset + (col ? 0 : 2))` flips the Dst
// pointer between the even and odd columns of the current pair of DST tiles.
// Forward declaration — defined below the K=2048 optimized body.
template <std::uint32_t K, bool APPROXIMATION_MODE>
inline void _topk_xl_local_sort_generic_(std::uint32_t dst_index, bool ascending);

template <std::uint32_t K, bool APPROXIMATION_MODE>
inline void _topk_xl_local_sort_(const std::uint32_t dst_index, const bool ascending)
{
    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");
    if constexpr (K != 2048)
    {
        _topk_xl_local_sort_generic_<K, APPROXIMATION_MODE>(dst_index, ascending);
        return;
    }

    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    bool dir                            = ascending;
    const std::uint32_t tile_offset     = dst_index << DstTileSizeLog2[DstTileShape::Tile32x32];
    constexpr int consecutive_32_offset = 16;
    // Kept as a local so the inner `if constexpr (row_scale_factor > 1)`
    // checks below match the historical K=2048 codegen byte-for-byte.
    constexpr int row_scale_factor = 4;

    for (int col = 0; col < 2; col++)
    {
        // ── Phase 1 — build bitonic sequences of length 32 ─────────────────
        //
        // Inner iter i=0 records `sort_8 + sort_16` (14 + 18 = 32 instr)
        // into replay slots [0..31] — an exact fit for the 32-slot replay
        // buffer. Iters 1..3 fire that body from the replay buffer.
        //
        // `sort_4` is left inline because adding it to the recording would
        // overflow the 32-slot buffer. `sort_32` is left inline because
        // `dir` alternates per iter (replaying it would freeze the
        // direction).
        for (int i = 0; i < 4; i++)
        {
            load16_rows_x2<consecutive_32_offset>();
            bitonic_sort_len_2();
            bitonic_sort_len_4(ascending);
            if (i == 0)
            {
                load_replay_buf<Exec>(
                    0,
                    32,
                    [ascending]
                    {
                        bitonic_sort_len_8(ascending);
                        bitonic_sort_len_16();
                    });
            }
            else
            {
                lltt::replay(0, 32);
            }
            bitonic_sort_len_32(dir);
            store16_rows_x2<consecutive_32_offset, 32>();
            if constexpr (row_scale_factor > 1)
            {
                dir = !dir;
            }
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

        // ── Phase 2 — build bitonic sequences of length 64 ─────────────────
        //
        // Inline — recording the body doesn't pay back at this size. The
        // second store in each pair uses `store16_rows_x2<32, 48>` so
        // ADDR_MOD_1 (+48) folds the trailing +16 advance into the last
        // SFPSTORE, saving one math-thread issue per inner iter.
        for (int i = 0; i < 2; i++)
        {
            load16_rows_x2<32>();
            bitonic_sort_len_k(dir);
            store16_rows_x2<32, 16>();
            load16_rows_x2<32>();
            bitonic_sort_len_k(dir);
            store16_rows_x2<32, 48>();
            dir = !dir;
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

        // ── Phase 3 — stride-16 sort_32 (4 effective iters) ────────────────
        // `dir` flips once per outer pass below.
        load_replay_buf<Exec>(0, 8, [] { load16_rows_x2<consecutive_32_offset>(); });
        bitonic_sort_len_32(dir);
        load_replay_buf<Exec>(8, 8, [] { store16_rows_x2<consecutive_32_offset, 32>(); });
        lltt::replay(0, 8);
        bitonic_sort_len_32(dir);
        lltt::replay(8, 8);
        dir = !dir;
        lltt::replay(0, 8);
        bitonic_sort_len_32(dir);
        lltt::replay(8, 8);
        lltt::replay(0, 8);
        bitonic_sort_len_32(dir);
        lltt::replay(8, 8);
        dir = !dir;
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

        // ── Phase 4 — build bitonic sequences of length 128 ────────────────
        // Canonical A + B + C big block (see above).
        canonical_big_block_with_replay<4>(dir);

        // Switch from the even column group to the odd one (and back on
        // iter 2). Flipping `dir` here gives the next len-128 build pass an
        // already-bitonic input across the two columns.
        set_dst_write_addr_offset(tile_offset + (col ? 0 : 2));
        dir = !dir;
    }

    // ── Phase 5 — build bitonic sequences of length 256 ───────────────────
    //
    // Now the runs span across LREGs. We use SFPCONFIG to flip every SFPU
    // instance's swap direction, so a single load+sort_k+store body covers
    // both halves of the merge.
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0x0100);
    TTI_SFPCONFIG(0x4444, 0xF, 8);

    // Stride-2 (load + sort_k + store<2, 16>), N = 8. Recording IS iter 0.
    load_replay_buf<Exec>(0, 8, [] { load16_rows_x2<2>(); });
    bitonic_sort_len_k(dir);
    load_replay_buf<Exec>(8, 8, [] { store16_rows_x2<2, 16>(); });
    for (int i = 1; i < 8; i++)
    {
        lltt::replay(0, 8);
        bitonic_sort_len_k(dir);
        lltt::replay(8, 8);
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    for (int col = 0; col < 2; col++)
    {
        canonical_big_block_with_replay<4>(dir);
        set_dst_write_addr_offset(tile_offset + (col ? 0 : 2));
    }
    TTI_SFPCONFIG(0x0000, 0xF, 1); // restore default SFPU config

    // ── Phases 6–8 — build bitonic sequences of length 512 / 1024 / 2048 ──
    //
    // Lengths beyond 256 require transposing whole faces of Dst so that
    // adjacent rows in a face represent adjacent elements in the sequence.
    // We open one big transpose CFG block here that covers all 6 transpose
    // calls below — the bitonic work between them doesn't touch the
    // affected CFG bits, so this saves 20 redundant CFG ops vs setting
    // them per call.
    enter_transpose_cfg_block();

    // ── Phase 6 — length 512 ───────────────────────────────────────────────
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU | p_stall::SRCA_VLD | p_stall::SRCB_VLD);
    transpose_8_faces<true, 256, /*manage_outer_cfg=*/false>();

    // Stride-2 (load + TRANSP + sort_4 + store<2, 16>), N = 8.
    load_replay_buf<Exec>(0, 8, [] { load16_rows_x2<2>(); });
    TTI_SFPTRANSP(0, 0, 0, 0);
    bitonic_sort_len_4(dir);
    load_replay_buf<Exec>(8, 8, [] { store16_rows_x2<2, 16>(); });
    for (int i = 1; i < 8; i++)
    {
        lltt::replay(0, 8);
        TTI_SFPTRANSP(0, 0, 0, 0);
        bitonic_sort_len_4(dir);
        lltt::replay(8, 8);
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    transpose_8_faces<true, 256, /*manage_outer_cfg=*/false>();

    // Flip SFPU swap direction for every 2 lanes.
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0x0100);
    TTI_SFPCONFIG(0x5050, 0xF, 8);
    for (int col = 0; col < 2; col++)
    {
        canonical_big_block_with_replay<4>(dir);
        set_dst_write_addr_offset(tile_offset + (col ? 0 : 2));
    }
    TTI_SFPCONFIG(0x0000, 0xF, 1); // restore default SFPU config

    // ── Phase 7 — length 1024 ──────────────────────────────────────────────
    transpose_8_faces<true, 256, /*manage_outer_cfg=*/false>();

    // Stride-2 (load + sort_8 + store<2, 16>), N = 8.
    load_replay_buf<Exec>(0, 8, [] { load16_rows_x2<2>(); });
    bitonic_sort_len_8(dir);
    load_replay_buf<Exec>(8, 8, [] { store16_rows_x2<2, 16>(); });
    for (int i = 1; i < 8; i++)
    {
        lltt::replay(0, 8);
        bitonic_sort_len_8(dir);
        lltt::replay(8, 8);
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    transpose_8_faces<true, 256, /*manage_outer_cfg=*/false>();

    // Flip SFPU swap direction for every 4 lanes.
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0x0100);
    TTI_SFPCONFIG(0x5500, 0xF, 8);
    for (int col = 0; col < 2; col++)
    {
        canonical_big_block_with_replay<4>(dir);
        set_dst_write_addr_offset(tile_offset + (col ? 0 : 2));
    }
    TTI_SFPCONFIG(0x0000, 0xF, 1); // restore default SFPU config

    // ── Phase 8 — length 2048 ──────────────────────────────────────────────
    transpose_8_faces<true, 256, /*manage_outer_cfg=*/false>();

    // Stride-2 (load + sort_16_alt + store<2, 16>), N = 8.
    load_replay_buf<Exec>(0, 8, [] { load16_rows_x2<2>(); });
    bitonic_sort_len_16_alt(dir);
    load_replay_buf<Exec>(8, 8, [] { store16_rows_x2<2, 16>(); });
    for (int i = 1; i < 8; i++)
    {
        lltt::replay(0, 8);
        bitonic_sort_len_16_alt(dir);
        lltt::replay(8, 8);
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    transpose_8_faces<true, 256, /*manage_outer_cfg=*/false>();

    leave_transpose_cfg_block();

    // Final A + B + C pass per column to finish the sort at the new length.
    for (int col = 0; col < 2; col++)
    {
        canonical_big_block_with_replay<4>(dir);
        set_dst_write_addr_offset(tile_offset + (col ? 0 : 2));
    }

    // Clear SrcA and SrcB valids — the next caller might issue a DST
    // transpose, which requires invalid SrcA/SrcB to be safe.
    TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD);
}

// =============================================================================
//  Generic local_sort for K=512 / K=1024 (optimised in place)
// =============================================================================
//
// Same algorithmic structure as PR #567 (bitonic build phases at lengths
// 32, 64, K/8, K/4, K/2, K) with `row_scale_factor`-driven loop counts,
// but with the same replay-buffer / ADDR_MOD-fold / CFG-block-widening
// optimisations the K=2048 fast path uses:
//
//   * `<Exec>` replay-record on iter 0 of each post-per-col stride-2 loop;
//     iters 1..n_iters-1 reduce to a 3-issue body (replay + sort + replay).
//   * sub-B-style ADDR_MOD_1 fold (`store<32, 48>` in place of
//     `store<32, 32>` + two `INCRWC(+8)`) wherever the K=2048 path uses
//     it. Saves 2 math-thread issues per outer iter.
//   * `canonical_big_block_with_replay<row_scale_factor>(dir)` for the
//     per-col post-transpose stride-32 / stride-16 work, which for
//     rsf=1 and rsf=2 collapses to exactly the same byte-stream as the
//     hand-rolled per-col tail above (sub-A is `if constexpr`'d away
//     unless rsf>=4) but ships the replay-buf / ADDR_MOD-fold versions.
//   * One `enter_transpose_cfg_block` / `leave_transpose_cfg_block` pair
//     wrapping the three transposed phases at the end (K/4, K/2, K).
//
// The K=2048 case continues to be routed to the K=2048 fast path by
// `_topk_xl_local_sort_`'s `if constexpr (K != 2048)` guard, so the
// codegen here is exercised only by K=512 and K=1024.
template <std::uint32_t K, bool APPROXIMATION_MODE>
inline void _topk_xl_local_sort_generic_(const std::uint32_t dst_index, const bool ascending)
{
    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    bool dir                            = ascending;
    const std::uint32_t tile_offset     = dst_index << DstTileSizeLog2[DstTileShape::Tile32x32];
    constexpr int consecutive_32_offset = 16;
    constexpr int row_scale_factor      = K == 512 ? 1 : K == 1024 ? 2 : 4;
    constexpr int n_iters_stride2       = row_scale_factor * 2;

    for (int col = 0; col < 2; col++)
    {
        // ── build bitonic sequences of len=32 ──────────────────────────────
        //
        // Inner iter i=0 records `sort_8 + sort_16` (14 + 18 = 32 instr)
        // into replay slots [0..31] — an exact fit for the 32-slot
        // replay buffer. Iters 1..rsf-1 fire the recorded body via
        // `lltt::replay`. `sort_4` is left inline because adding it to
        // the recording would overflow the buffer; `sort_32` is left
        // inline because `dir` alternates per iter (replaying it would
        // freeze the direction).
        //
        // For K=512 (rsf=1) only iter 0 runs, so the record is wasted
        // (~1 REPLAY-config issue). At rsf=2 it pays for itself once;
        // at rsf=4 (K=2048 — handled by the fast path) the saving is
        // larger but that branch isn't reached here.
        for (int i = 0; i < row_scale_factor; i++)
        {
            load16_rows_x2<consecutive_32_offset>();
            bitonic_sort_len_2();
            bitonic_sort_len_4(ascending);
            if (i == 0)
            {
                load_replay_buf<Exec>(
                    0,
                    32,
                    [ascending]
                    {
                        bitonic_sort_len_8(ascending);
                        bitonic_sort_len_16();
                    });
            }
            else
            {
                lltt::replay(0, 32);
            }
            bitonic_sort_len_32(dir);
            store16_rows_x2<consecutive_32_offset, 32>();
            if constexpr (row_scale_factor > 1)
            {
                dir = !dir;
            }
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

        if constexpr (K >= 1024)
        {
            // ── build bitonic sequences of len=64 ──────────────────────────
            //
            // Pair stores using ADDR_MOD_1 (+48) fold to drop the two
            // trailing `TTI_INCRWC(+8)` issues PR #567's version had.
            for (int i = 0; i < (row_scale_factor >> 1); i++)
            {
                load16_rows_x2<32>();
                bitonic_sort_len_k(dir);
                store16_rows_x2<32, 16>();
                load16_rows_x2<32>();
                bitonic_sort_len_k(dir);
                store16_rows_x2<32, 48>();
                dir = !dir;
            }
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

            // ── stride-16 sort_32 × rsf, dir alternates per pair ───────────
            //
            // Same recording pattern as the K=2048 fast path's Phase 3:
            // iter 0 is recorded into slots [0..7] / [8..15] in Exec mode;
            // remaining iters replay. `dir` flips after iters 1, 3, ...
            // so pairs of iters share a direction.
            load_replay_buf<Exec>(0, 8, [] { load16_rows_x2<consecutive_32_offset>(); });
            bitonic_sort_len_32(dir);
            load_replay_buf<Exec>(8, 8, [] { store16_rows_x2<consecutive_32_offset, 32>(); });
            for (int i = 1; i < row_scale_factor; i++)
            {
                lltt::replay(0, 8);
                bitonic_sort_len_32(dir);
                lltt::replay(8, 8);
                if ((i & 1) == 1)
                {
                    dir = !dir;
                }
            }
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        }

        set_dst_write_addr_offset(tile_offset + (col ? 0 : 2));
        dir = !dir;
    }

    // ── build bitonic sequences of len=(K/8) ──────────────────────────────
    //
    // Cross-column stride-2 sort_k via SFPCONFIG flip. Recording is
    // `<Exec>` and IS iter 0; iters 1..n_iters-1 fire from the replay
    // buffer with the bare-minimum 3-issue body shape.
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0x0100);
    TTI_SFPCONFIG(0x4444, 0xF, 8);
    load_replay_buf<Exec>(0, 8, [] { load16_rows_x2<2>(); });
    bitonic_sort_len_k(dir);
    load_replay_buf<Exec>(8, 8, [] { store16_rows_x2<2, 16>(); });
    for (int i = 1; i < n_iters_stride2; i++)
    {
        lltt::replay(0, 8);
        bitonic_sort_len_k(dir);
        lltt::replay(8, 8);
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    for (int col = 0; col < 2; col++)
    {
        // Per-col post-stride-2 work = sub-B (stride-32 pair, K>=1024)
        // + sub-C (stride-16 sort_32 × rsf). `canonical_big_block_with_replay`
        // emits exactly that for rsf=1 and rsf=2, with the ADDR_MOD_1
        // fold on sub-B's last store and a replay buffer on sub-C.
        // Sub-A (stride-64) is `if constexpr`'d away for rsf<4.
        canonical_big_block_with_replay<row_scale_factor>(dir);

        set_dst_write_addr_offset(tile_offset + (col ? 0 : 2));
    }
    TTI_SFPCONFIG(0x0000, 0xF, 1); // restore default SFPU config

    // ── Transpose-phase block ─────────────────────────────────────────────
    //
    // The three transposed phases (K/4, K/2, K) all need the same two
    // transpose CFG bits (`DISABLE_IMPLIED_SRCA_FMT_Base_ADDR32` and
    // `ALU_ACC_CTRL_Zero_Flag_disabled_src`). The bitonic / canonical
    // work in between doesn't touch either bit, so we open ONE
    // transpose CFG block and pass `manage_outer_cfg=false` to each
    // `transpose_N_faces` call. Saves a handful of CFG writes per
    // transposed phase.
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU | p_stall::SRCA_VLD | p_stall::SRCB_VLD);
    enter_transpose_cfg_block();

    // ── build bitonic sequences of len=(K/4) ──────────────────────────────
    transpose_N_faces</*N*/ row_scale_factor * 2, /*fused=*/true, /*indices_offset=*/256, /*manage_outer_cfg=*/false>();
    load_replay_buf<Exec>(0, 8, [] { load16_rows_x2<2>(); });
    TTI_SFPTRANSP(0, 0, 0, 0);
    bitonic_sort_len_4(dir);
    load_replay_buf<Exec>(8, 8, [] { store16_rows_x2<2, 16>(); });
    for (int i = 1; i < n_iters_stride2; i++)
    {
        lltt::replay(0, 8);
        TTI_SFPTRANSP(0, 0, 0, 0);
        bitonic_sort_len_4(dir);
        lltt::replay(8, 8);
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    transpose_N_faces</*N*/ row_scale_factor * 2, /*fused=*/true, /*indices_offset=*/256, /*manage_outer_cfg=*/false>();
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0x0100);
    TTI_SFPCONFIG(0x5050, 0xF, 8);
    for (int col = 0; col < 2; col++)
    {
        canonical_big_block_with_replay<row_scale_factor>(dir);
        set_dst_write_addr_offset(tile_offset + (col ? 0 : 2));
    }
    TTI_SFPCONFIG(0x0000, 0xF, 1); // restore default SFPU config

    // ── build bitonic sequences of len=(K/2) ──────────────────────────────
    transpose_N_faces</*N*/ row_scale_factor * 2, /*fused=*/true, /*indices_offset=*/256, /*manage_outer_cfg=*/false>();
    load_replay_buf<Exec>(0, 8, [] { load16_rows_x2<2>(); });
    bitonic_sort_len_8(dir);
    load_replay_buf<Exec>(8, 8, [] { store16_rows_x2<2, 16>(); });
    for (int i = 1; i < n_iters_stride2; i++)
    {
        lltt::replay(0, 8);
        bitonic_sort_len_8(dir);
        lltt::replay(8, 8);
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    transpose_N_faces</*N*/ row_scale_factor * 2, /*fused=*/true, /*indices_offset=*/256, /*manage_outer_cfg=*/false>();
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0x0100);
    TTI_SFPCONFIG(0x5500, 0xF, 8);
    for (int col = 0; col < 2; col++)
    {
        canonical_big_block_with_replay<row_scale_factor>(dir);
        set_dst_write_addr_offset(tile_offset + (col ? 0 : 2));
    }
    TTI_SFPCONFIG(0x0000, 0xF, 1); // restore default SFPU config

    // ── build bitonic sequences of len=K ──────────────────────────────────
    transpose_N_faces</*N*/ row_scale_factor * 2, /*fused=*/true, /*indices_offset=*/256, /*manage_outer_cfg=*/false>();
    load_replay_buf<Exec>(0, 8, [] { load16_rows_x2<2>(); });
    bitonic_sort_len_16_alt(dir);
    load_replay_buf<Exec>(8, 8, [] { store16_rows_x2<2, 16>(); });
    for (int i = 1; i < n_iters_stride2; i++)
    {
        lltt::replay(0, 8);
        bitonic_sort_len_16_alt(dir);
        lltt::replay(8, 8);
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    transpose_N_faces</*N*/ row_scale_factor * 2, /*fused=*/true, /*indices_offset=*/256, /*manage_outer_cfg=*/false>();

    leave_transpose_cfg_block();

    // Final per-column canonical pass to finish the sort at len=K.
    for (int col = 0; col < 2; col++)
    {
        canonical_big_block_with_replay<row_scale_factor>(dir);
        set_dst_write_addr_offset(tile_offset + (col ? 0 : 2));
    }

    // Clear srcA and srcB valids — needed for DST transpose.
    TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD);
}

// =============================================================================
//  Public entry: merge
// =============================================================================
//
// Bitonic-merge two adjacent sorted runs in DST into a single top-K run.
// For K=2048 each run spans two DST tiles; for K=1024 / K=512 a run spans
// one DST tile (or half a tile for K=512). On entry, the first run sits
// at `dst_index` and the second run sits one sorted-sequence further
// along; on exit the merged top-K replaces the first run and the second
// run's slots are left dead (the next merge stage overwrites them via
// `topk_xl_copy_tile(recv, ...)`).
//
// The same body shape is used for every K — only the load/store distance,
// the parallel-index offset (unfused only), and the per-column iteration
// count change with K:
//
//   distance       = (fused ? 64 : 128) * num_tiles_per_sequence
//   indices_offset = 64 * num_tiles_per_sequence
//   n_iters        = fused ? row_scale_factor * 2 : row_scale_factor * 4
//
// where `num_tiles_per_sequence = (K == 2048) ? 2 : 1` and
// `row_scale_factor = K / 512`. For K=2048 this evaluates to the legacy
// values (`distance=128/256`, `n_iters=8/16`) and the body below is
// bit-for-bit identical to the previous K=2048 fast path.
//
// Body lengths fit in a single replay window:
//
//   fused (16 slots):    8 (load16) + 4 (sort_k) + 4 (store4_top_only)
//   unfused (18 slots):  8 (load8)  + 2 (sort_k) + 8 (store8)
//
// Fused path: only the "top half" (LREG0..3, the per-pair max) is stored
// back to DST. LREG4..7 hold the per-pair min, which is dead by
// construction — the next-stage `topk_xl_copy_tile(recv, ...)` overwrites
// the slot it lives in before any reader sees it (in K=2048 that's the
// +128-tile region; in K=1024 it's the +64 region within the same tile).
// See `store4_rows_top_only` for the full justification.
//
// We record the body once (recording IS iter 0 of col=0), then fire it
// through the MOP Expander programmed at init: each column collapses to
// a single TTI_MOP issue that re-runs the recorded body N_ITERS times.
template <std::uint32_t K, bool APPROXIMATION_MODE, bool fused>
inline void _topk_xl_merge_(const std::uint32_t dst_index)
{
    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");

    constexpr bool descending                     = false; // direction the merge writes
    constexpr int num_tiles_per_sequence          = (K == 2048) ? 2 : 1;
    constexpr int row_scale_factor                = (K == 512) ? 1 : (K == 1024) ? 2 : 4;
    constexpr int distance                        = fused ? (64 * num_tiles_per_sequence) : (128 * num_tiles_per_sequence);
    [[maybe_unused]] constexpr int indices_offset = 64 * num_tiles_per_sequence;
    const std::uint32_t tile_offset               = dst_index << DstTileSizeLog2[DstTileShape::Tile32x32];

    constexpr int body_len = fused ? 16 : 18;
    constexpr int n_iters  = fused ? (row_scale_factor * 2) : (row_scale_factor * 4);
    static_assert(n_iters >= 2, "n_iters < 2 would skip the MOP firing of col=0");
    static_assert(n_iters <= 255, "ckernel_unpack_template::run takes a uint8_t count");

    // Record the loop body once into replay slots [0, body_len). The Exec
    // mode means this recording also counts as iter 0 of col=0's work, so
    // col=0 only needs (n_iters - 1) replays below.
    if constexpr (fused)
    {
        load_replay_buf<Exec>(
            0,
            body_len,
            []
            {
                load16_rows_x2<distance>();
                bitonic_sort_len_k(descending);
                store4_rows_top_only<16>();
            });
    }
    else
    {
        load_replay_buf<Exec>(
            0,
            body_len,
            []
            {
                load8_rows_x2_unfused<indices_offset, distance>();
                bitonic_sort_len_k<fused>(descending);
                store8_rows_x2_unfused<indices_offset, distance, 8 /* inc_dst_addr */>();
            });
    }

    // col=0: fire the remaining (n_iters - 1) iters via one MOP issue.
    ckernel_unpack_template::run(n_iters - 1);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

    // Switch the Dst write pointer to the odd column group.
    set_dst_write_addr_offset(tile_offset + 2);

    // col=1: full n_iters worth of work, again as one MOP issue.
    ckernel_unpack_template::run(n_iters);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    set_dst_write_addr_offset(tile_offset + 0);
}

// =============================================================================
//  Public entry: rebuild
// =============================================================================
//
// After `_topk_xl_merge_` the top-2048 result lives in DST tiles
// dst_index / dst_index+1 but its bitonic structure is broken (the merge
// only guarantees ordering, not the bitonic property the next merge would
// need). Rebuild restores the bitonic property by repeating the length-2048
// build phase of the local sort.
//
// The fused and unfused paths diverge: fused mode uses a 5-slot MOP
// template (the body is 34 instructions and doesn't fit in one 32-slot
// replay window), unfused mode uses straightforward split-record loops.
// Forward declaration — defined below the K=2048 optimized body.
template <std::uint32_t K, bool APPROXIMATION_MODE, bool fused>
inline void _topk_xl_rebuild_generic_(std::uint32_t dst_index, bool ascending);

template <std::uint32_t K, bool APPROXIMATION_MODE, bool fused>
inline void _topk_xl_rebuild_(const std::uint32_t dst_index, const bool ascending)
{
    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");
    if constexpr (K != 2048)
    {
        _topk_xl_rebuild_generic_<K, APPROXIMATION_MODE, fused>(dst_index, ascending);
        return;
    }

    bool dir                                             = ascending;
    [[maybe_unused]] constexpr int consecutive_32_offset = 16;
    const std::uint32_t tile_offset                      = dst_index << DstTileSizeLog2[DstTileShape::Tile32x32];

    if constexpr (fused)
    {
        // ── Fused path ─────────────────────────────────────────────────────
        // One CFG block wraps both transposes since the bitonic work between
        // them doesn't touch the transpose CFG bits.
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU | p_stall::SRCA_VLD | p_stall::SRCB_VLD);
        enter_transpose_cfg_block();
        transpose_8_faces<true, 256, /*manage_outer_cfg=*/false>();

        // Length-2048 build phase via the 5-slot MOP template.
        //
        // The body would be 8 (load) + 18 (sort_16_alt) + 8 (store) = 34
        // instructions, which is too large for a contiguous 32-slot replay
        // window. We split on the two SFPTRANSPs inside the sort: record
        // the 32 "data" instructions across three replay ranges and let
        // the MOP template emit the two SFPTRANSPs from its A1 / A3 slots.
        // See `topk_rebuild_build2048_mop_config` for the template layout.
        //
        // Iter 0 is the recording itself (Exec mode); iters 1..7 fire from
        // a single TTI_MOP issue.
        topk_rebuild_build2048_mop_config();
        load_replay_buf<Exec>(
            0,
            16,
            [dir]
            {
                load16_rows_x2<2>();
                bitonic_sort_len_16_alt_swaps<true>(dir);
            });
        TTI_SFPTRANSP(0, 0, 0, 0);
        load_replay_buf<Exec>(16, 8, [dir] { bitonic_sort_len_16_alt_swaps<true>(dir); });
        TTI_SFPTRANSP(0, 0, 0, 0);
        load_replay_buf<Exec>(24, 8, [] { store16_rows_x2<2, 16>(); });
        ckernel_unpack_template::run(7);

        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        transpose_8_faces<true, 256, /*manage_outer_cfg=*/false>();
        leave_transpose_cfg_block();

        // Final A + B + C pass per column to finish the rebuild.
        for (int col = 0; col < 2; col++)
        {
            canonical_big_block_with_replay<4>(dir);
            set_dst_write_addr_offset(tile_offset + (col ? 0 : 2));
        }

        // The build phase clobbered the MOP Expander config. Restore the
        // merge MOP setup (REPLAY(0, 16)) so the next `_topk_xl_merge_`
        // finds the programming `_topk_xl_init_` left at init time.
        topk_mop_config<true>();
    }
    else
    {
        // ── Unfused path (extended 256K path on main) ──────────────────────
        // Same overall shape as the fused path, but the body is small
        // enough to fit in a single 32-slot replay window so we don't need
        // the 5-slot MOP template. The unfused tile layout has indices
        // sitting at offset +128 inside each value region (vs +256 in the
        // older non-distributed-topk unfused path).
        constexpr int indices_offset = 2 * 64; // 128
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU | p_stall::SRCA_VLD | p_stall::SRCB_VLD);
        enter_transpose_cfg_block();
        transpose_8_faces<fused, indices_offset, /*manage_outer_cfg=*/false>();

        // Stride-8 (load + sort_16_alt<false> + store<8, 16>), N = 8 per
        // col × 2 cols. Recording IS col=0 iter=0; the rest are replays.
        load_replay_buf<Exec>(0, 8, [] { load8_rows_x2_unfused<indices_offset, 8>(); });
        bitonic_sort_len_16_alt<fused>(dir);
        load_replay_buf<Exec>(8, 8, [] { store8_rows_x2_unfused<indices_offset, 8, 16>(); });
        for (int i = 1; i < 8; i++)
        {
            lltt::replay(0, 8);
            bitonic_sort_len_16_alt<fused>(dir);
            lltt::replay(8, 8);
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        set_dst_write_addr_offset(tile_offset + 2);
        for (int i = 0; i < 8; i++)
        {
            lltt::replay(0, 8);
            bitonic_sort_len_16_alt<fused>(dir);
            lltt::replay(8, 8);
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        set_dst_write_addr_offset(tile_offset + 0);
        transpose_8_faces<fused, indices_offset, /*manage_outer_cfg=*/false>();
        leave_transpose_cfg_block();
        for (int col = 0; col < 2; col++)
        {
            // ── Stride-64 — clean N = 8 replay, recording IS iter 0 ────────
            load_replay_buf<Exec>(0, 8, [] { load8_rows_x2_unfused<indices_offset, 64>(); });
            bitonic_sort_len_k<fused>(dir);
            load_replay_buf<Exec>(8, 8, [] { store8_rows_x2_unfused<indices_offset, 64, 8>(); });
            for (int i = 1; i < 8; i++)
            {
                lltt::replay(0, 8);
                bitonic_sort_len_k<fused>(dir);
                lltt::replay(8, 8);
            }
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

            // ── Stride-32 — split-record with inline LREG7 tail ────────────
            // The trailing LREG7 store needs a different ADDR_MOD on the
            // last inner iter of each outer pair so we can fold the +8
            // INCRWC into it:
            //   inner < 3       → ADDR_MOD_4  (incr = 8)
            //   inner == 3 (j=3 → ADDR_MOD_3  (incr = 40 = 32 + 8)
            // So we record the load (slots [0..7]) plus the first 7 stores
            // (slots [8..14]) and emit only the LREG7 store inline.
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    if (i == 0 && j == 0)
                    {
                        load_replay_buf<Exec>(0, 8, [] { load8_rows_x2_unfused<indices_offset, 32>(); });
                        bitonic_sort_len_k<fused>(dir);
                        load_replay_buf<Exec>(8, 7, [] { store_first_7_rows_x2_unfused<indices_offset, 32>(); });
                    }
                    else
                    {
                        lltt::replay(0, 8);
                        bitonic_sort_len_k<fused>(dir);
                        lltt::replay(8, 7);
                    }
                    if (j < 3)
                    {
                        TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_4, indices_offset + 32 + 4);
                    }
                    else
                    {
                        TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_3, indices_offset + 32 + 4);
                    }
                }
            }
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

            // ── Stride-16 — same split-record pattern as stride-32 ─────────
            // The replay slots are re-recorded fresh because
            // `group_2_offset` is baked into the SFPLOAD/SFPSTORE
            // immediates. LREG7 tail picks ADDR_MOD per inner iter:
            //   inner = 0 → ADDR_MOD_4  (incr = 8)
            //   inner = 1 → ADDR_MOD_2  (incr = 24 = 16 + 8)
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    if (i == 0 && j == 0)
                    {
                        load_replay_buf<Exec>(0, 8, [] { load8_rows_x2_unfused<indices_offset, 16>(); });
                        bitonic_sort_len_k<fused>(dir);
                        load_replay_buf<Exec>(8, 7, [] { store_first_7_rows_x2_unfused<indices_offset, 16>(); });
                    }
                    else
                    {
                        lltt::replay(0, 8);
                        bitonic_sort_len_k<fused>(dir);
                        lltt::replay(8, 7);
                    }
                    if (j == 0)
                    {
                        TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_4, indices_offset + 16 + 4);
                    }
                    else
                    {
                        TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_2, indices_offset + 16 + 4);
                    }
                }
            }
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

            // ── Stride-8 — clean N = 8 replay, recording IS iter 0 ────────
            load_replay_buf<Exec>(0, 8, [] { load8_rows_x2_unfused<indices_offset, 8>(); });
            bitonic_sort_len_16_alt<fused>(dir);
            load_replay_buf<Exec>(8, 8, [] { store8_rows_x2_unfused<indices_offset, 8, 16>(); });
            for (int i = 1; i < 8; i++)
            {
                lltt::replay(0, 8);
                bitonic_sort_len_16_alt<fused>(dir);
                lltt::replay(8, 8);
            }
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

            set_dst_write_addr_offset(tile_offset + (col ? 0 : 2));
        }
    }

    // Clear SrcA / SrcB valids — the next caller might issue a DST
    // transpose, which requires both to be invalid to be safe.
    TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD);
}

// =============================================================================
//  Generic rebuild for K=512 / K=1024 (optimised in place)
// =============================================================================
//
// Same algorithmic shape as PR #567's parametric rebuild (len=K build via
// transpose + stride-2 + per-col tails), but with the K=2048 fast path's
// optimisations applied where the generic body's structure allows it:
//
//   Fused path:
//     * 5-slot MOP template for the stride-2 + sort_16_alt build (the
//       same `topk_rebuild_build2048_mop_config` the K=2048 fast path
//       uses — the body is 34 instructions, too large for one 32-slot
//       replay window, so the two `SFPTRANSP`s inside `sort_16_alt`
//       are emitted from the MOP template's A1 / A3 slots).
//     * One `enter_transpose_cfg_block` / `leave_transpose_cfg_block`
//       pair wrapping both `transpose_N_faces` calls.
//     * `canonical_big_block_with_replay<row_scale_factor>(dir)` for
//       the per-col post-transpose work (sub-B + sub-C; sub-A is
//       `if constexpr`'d away for rsf<4).
//     * `topk_mop_config<true>()` at the end to restore the merge MOP
//       config (the rebuild template clobbered it).
//
//   Unfused path:
//     * `<Exec>`-record on iter 0 of every stride-N inner loop; iters
//       1..n-1 fire from the replay buffer.
//     * ADDR_MOD_3 fold (`store<X, 40>` in place of `store<X, 32>` +
//       trailing `INCRWC(+8)`) on the last stride-32 / stride-16 sort_k
//       iter of each outer pair. Saves one math-thread issue per outer
//       pair compared to the original PR #567 shape.
//     * Same shared `enter/leave_transpose_cfg_block` pair as above.
//
// The K=2048 case continues to be routed to the K=2048 fast path by
// `_topk_xl_rebuild_`'s `if constexpr (K != 2048)` guard.
template <std::uint32_t K, bool APPROXIMATION_MODE, bool fused>
inline void _topk_xl_rebuild_generic_(const std::uint32_t dst_index, const bool ascending)
{
    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");
    bool dir                                             = ascending;
    constexpr int num_tiles_per_sequence                 = K == 512 ? 1 : K == 1024 ? 1 : 2;
    [[maybe_unused]] constexpr int consecutive_32_offset = 16;
    const std::uint32_t tile_offset                      = dst_index << DstTileSizeLog2[DstTileShape::Tile32x32];
    constexpr int row_scale_factor                       = K == 512 ? 1 : K == 1024 ? 2 : 4;
    constexpr int n_iters_stride2                        = row_scale_factor * 2;

    if constexpr (fused)
    {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU | p_stall::SRCA_VLD | p_stall::SRCB_VLD);
        enter_transpose_cfg_block();
        transpose_N_faces</*N*/ row_scale_factor * 2, /*fused=*/true, /*indices_offset=*/256, /*manage_outer_cfg=*/false>();

        // ── stride-2 + sort_16_alt build phase ─────────────────────────
        //
        // Body is `load(8) + sort_16_alt(18) + store(8)` = 34 instr.
        // Doesn't fit in one 32-slot replay window; we split on the two
        // SFPTRANSPs inside sort_16_alt and let the 5-slot MOP template
        // emit them from A1 / A3. Iter 0 is the recording itself (Exec
        // mode); iters 1..n_iters-1 collapse to a single TTI_MOP issue.
        topk_rebuild_build2048_mop_config();
        load_replay_buf<Exec>(
            0,
            16,
            [dir]
            {
                load16_rows_x2<2>();
                bitonic_sort_len_16_alt_swaps<true>(dir);
            });
        TTI_SFPTRANSP(0, 0, 0, 0);
        load_replay_buf<Exec>(16, 8, [dir] { bitonic_sort_len_16_alt_swaps<true>(dir); });
        TTI_SFPTRANSP(0, 0, 0, 0);
        load_replay_buf<Exec>(24, 8, [] { store16_rows_x2<2, 16>(); });
        ckernel_unpack_template::run(n_iters_stride2 - 1);

        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        transpose_N_faces</*N*/ row_scale_factor * 2, /*fused=*/true, /*indices_offset=*/256, /*manage_outer_cfg=*/false>();
        leave_transpose_cfg_block();

        // Per-col post-transpose work = sub-B (stride-32 pair, K>=1024) +
        // sub-C (stride-16 sort_32 × rsf), folded into `canonical_big_block_with_replay`
        // exactly as the K=2048 fast path does it (with sub-A disabled
        // for rsf<4 via `if constexpr`).
        for (int col = 0; col < 2; col++)
        {
            canonical_big_block_with_replay<row_scale_factor>(dir);
            set_dst_write_addr_offset(tile_offset + (col ? 0 : 2));
        }

        // The rebuild MOP template clobbered the merge MOP setup; put it
        // back so the next `_topk_xl_merge_` finds the programming
        // `_topk_xl_init_` left.
        topk_mop_config<true>();
    }
    else
    {
        constexpr int indices_offset = num_tiles_per_sequence * 64;
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU | p_stall::SRCA_VLD | p_stall::SRCB_VLD);
        enter_transpose_cfg_block();
        transpose_N_faces</*N*/ row_scale_factor * 2, fused, indices_offset, /*manage_outer_cfg=*/false>();

        // ── stride-8 + sort_16_alt build phase (both columns) ──────────
        //
        // Body is `load8_unfused(8) + sort_16_alt<false>(10) + store8_unfused(8)`
        // = 26 instructions. `sort_16_alt<false>` emits only 8 SFPSWAPs
        // (values only — indices ride along via the parallel DST region
        // at `indices_offset`) plus its two SFPTRANSPs, so the body fits
        // in the 32-slot replay window. We still keep the K=2048 fast
        // path's load / store split into [0..7] / [8..15] with the
        // direction-dependent sort_16_alt inline, so the per-col stride
        // phases below can re-record [0..15] for their own load / store
        // bodies. Both cols fire from the same recording.
        load_replay_buf<Exec>(0, 8, [] { load8_rows_x2_unfused<indices_offset, 8>(); });
        bitonic_sort_len_16_alt<fused>(dir);
        load_replay_buf<Exec>(8, 8, [] { store8_rows_x2_unfused<indices_offset, 8, 16>(); });
        for (int i = 1; i < n_iters_stride2; i++)
        {
            lltt::replay(0, 8);
            bitonic_sort_len_16_alt<fused>(dir);
            lltt::replay(8, 8);
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        set_dst_write_addr_offset(tile_offset + 2);
        for (int i = 0; i < n_iters_stride2; i++)
        {
            lltt::replay(0, 8);
            bitonic_sort_len_16_alt<fused>(dir);
            lltt::replay(8, 8);
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        set_dst_write_addr_offset(tile_offset + 0);

        transpose_N_faces</*N*/ row_scale_factor * 2, fused, indices_offset, /*manage_outer_cfg=*/false>();
        leave_transpose_cfg_block();

        for (int col = 0; col < 2; col++)
        {
            // The inner loop counts here mirror `row_scale_factor` exactly
            // like the fused branch above. PR #567 ported these from the
            // K=2048 optimized unfused rebuild verbatim (hardcoded 2 / 4 / 8)
            // and never K-scaled them. That was harmless until the kernel
            // actually started taking the unfused path for K=1024 (the
            // (K=1024, cores=64) regression), at which point the rebuild
            // walked off the end of the 1-tile sequence with garbage
            // load/store pointers and the final output came back all zeros.
            //
            // Sanity check: for K=2048 (where this branch is not used today
            // because `_topk_xl_rebuild_` short-circuits to the optimized
            // body) the scaled counts evaluate to 2 / 4 / 8 — identical to
            // the previous hardcoded values, so dispatch behaviour is
            // unchanged in that regime.
            //
            // Each stride-N pair re-records its load / store body fresh
            // because `group_2_offset` is baked into the SFPLOAD/SFPSTORE
            // immediates. Recording is `<Exec>` so iter 0 doubles as the
            // first inline iteration.
            if constexpr (K >= 1024)
            {
                // ── stride-32 ─────────────────────────────────────────
                // Four sort_k pairs per outer iter. The last pair folds
                // its trailing `INCRWC(+8)` into the LREG7 store via
                // ADDR_MOD_3 (+40 = +32 + +8); the first three pairs
                // keep ADDR_MOD_4 (+8). Saves one math-thread issue per
                // outer iter compared to PR #567's `store<32,32> + INCRWC`.
                load_replay_buf<Exec>(0, 8, [] { load8_rows_x2_unfused<indices_offset, 32>(); });
                bitonic_sort_len_k<fused>(dir);
                load_replay_buf<Exec>(8, 8, [] { store8_rows_x2_unfused<indices_offset, 32, 8>(); });
                for (int i = 0; i < (row_scale_factor >> 1); i++)
                {
                    if (!(i == 0))
                    {
                        lltt::replay(0, 8);
                        bitonic_sort_len_k<fused>(dir);
                        lltt::replay(8, 8);
                    }
                    lltt::replay(0, 8);
                    bitonic_sort_len_k<fused>(dir);
                    lltt::replay(8, 8);
                    lltt::replay(0, 8);
                    bitonic_sort_len_k<fused>(dir);
                    lltt::replay(8, 8);
                    // Last sort_k of the outer iter: fold +8 into the LREG7 store.
                    load8_rows_x2_unfused<indices_offset, 32>();
                    bitonic_sort_len_k<fused>(dir);
                    store8_rows_x2_unfused<indices_offset, 32, 40 /* inc_dst_addr */>();
                }
                TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
            }

            // ── stride-16 ─────────────────────────────────────────────
            // Two sort_k iters per outer iter. The first uses ADDR_MOD_4
            // (+8); the second folds its trailing +8 INCRWC into the
            // LREG7 store via ADDR_MOD_2 (+24 = +16 + +8). Replay buffer
            // records the load / store bodies fresh because the offset
            // differs from the stride-32 phase.
            load_replay_buf<Exec>(0, 8, [] { load8_rows_x2_unfused<indices_offset, 16>(); });
            bitonic_sort_len_k<fused>(dir);
            load_replay_buf<Exec>(8, 8, [] { store8_rows_x2_unfused<indices_offset, 16, 8>(); });
            // Iter 0 already executed via the recording. Trailing pair (with the
            // ADDR_MOD_2 fold) for iter 0:
            load8_rows_x2_unfused<indices_offset, 16>();
            bitonic_sort_len_k<fused>(dir);
            store8_rows_x2_unfused<indices_offset, 16, 24 /* inc_dst_addr */>();
            for (int i = 1; i < row_scale_factor; i++)
            {
                lltt::replay(0, 8);
                bitonic_sort_len_k<fused>(dir);
                lltt::replay(8, 8);
                load8_rows_x2_unfused<indices_offset, 16>();
                bitonic_sort_len_k<fused>(dir);
                store8_rows_x2_unfused<indices_offset, 16, 24 /* inc_dst_addr */>();
            }
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

            // ── stride-8 + sort_16_alt ────────────────────────────────
            // Clean N-iter replay (no LREG7 fold needed; the store's
            // ADDR_MOD_5 (+16) already lands on the right boundary).
            load_replay_buf<Exec>(0, 8, [] { load8_rows_x2_unfused<indices_offset, 8>(); });
            bitonic_sort_len_16_alt<fused>(dir);
            load_replay_buf<Exec>(8, 8, [] { store8_rows_x2_unfused<indices_offset, 8, 16>(); });
            for (int i = 1; i < n_iters_stride2; i++)
            {
                lltt::replay(0, 8);
                bitonic_sort_len_16_alt<fused>(dir);
                lltt::replay(8, 8);
            }
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

            set_dst_write_addr_offset(tile_offset + (col ? 0 : 2));
        }

        // No MOP-Expander reprogramming happened on the unfused path — the
        // stride-8 build phase above just uses a plain replay window, so
        // the merge MOP config from `_topk_xl_init_` is still live and we
        // don't need to call `topk_mop_config<fused>()` here.
    }

    // Clear srcA and srcB valids — needed for DST transpose.
    TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD);
}

// =============================================================================
//  Index injection / extraction
// =============================================================================
//
// These two pairs are what make the "fused" representation work:
//   * `add_lsb_indices`   : write each element's global index into the low
//                           16 bits of its DST word, just after the input
//                           tiles have been copied in.
//   * `remove_msb_values` : strip the high 16 bits (the value bits) at the
//                           end, leaving only the index in each DST word so
//                           the caller can pack the indices to output.

// Programs the ADDR_MODs used by `_topk_xl_add_lsb_indices_`. Reuses
// ADDR_MOD_4 for a +16 advance — the other ADDR_MOD_4 user is the unfused
// topk path, but `distributed_topk` runs fused, so the reuse is safe.
// ADDR_MOD_7 (zero advance) is left at its kernel-startup default and
// not touched here.
inline void _topk_xl_add_lsb_indices_init_()
{
    // ADDR_MOD_6 — +4 (one element).
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 4},
    }
        .set(ADDR_MOD_6);

    // ADDR_MOD_4 — +16. Lets the "skip a face pair" no-op SFPLOAD between
    // outer iters collapse from two issues (each +8) into a single +16
    // issue, saving 4 SFPU cycles per query on the final core.
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 16},
    }
        .set(ADDR_MOD_4);
}

// Builds the per-element global index in LREG0..LREG3 and ORs it into the
// low 16 bits of each fused payload in the first two DST tiles.
//
// The 16-bit index layout is:
//
//   bits [15:11] — core_id (5 bits, up to 32 cores)
//   bits [10: 5] — row    (6 bits, derived from the tile id)
//   bits [ 4: 0] — within-row column (5 bits, lane id)
//
// After this routine each DST word reads as `[ bf16 value | u16 index ]`.
template <std::uint32_t K, bool APPROXIMATION_MODE, std::uint32_t core_id>
inline void _topk_xl_add_lsb_indices_()
{
    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

    // Tile ID into LREG0 (across the 32 lanes: 0, 2, 4, ..., 60, 62).
    TTI_SFPMOV(0, p_sfpu::LTILEID, p_sfpu::LREG0, 0);

    // LREG1 = LREG0 + 1, LREG2 = LREG0 + 16, LREG3 = LREG0 + 17 — give the
    // four index variants we need across the four-row LREG block.
    TTI_SFPIADD(1, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(16, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(17, p_sfpu::LREG0, p_sfpu::LREG3, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);

    TTI_SFPTRANSP(0, 0, 0, 0);

    // Multiply by 64 (one row per 64 lanes), so per-column start values are
    // (0, 64, 128, 192, ..., 1920, 1984).
    // NOTE: SFPSHFT_MOD1_SHIFT_IMM in sfpi is buggy — passing 1 as the imm
    // arg is the documented workaround.
    TTI_SFPSHFT(6, 0, p_sfpu::LREG0, 1);

    // Load core_id and place it in bits [15:11] of every lane.
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, core_id);
    TTI_SFPSHFT(11, 0, p_sfpu::LREG1, 1);

    // Merge core_id into LREG0, then propagate to LREG1..3 with the same
    // +1 / +16 / +17 offsets so all four LREGs hold their final 16-bit
    // index values.
    TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(1, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(2, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(3, p_sfpu::LREG0, p_sfpu::LREG3, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);

    TTI_SFPTRANSP(0, 0, 0, 0);

    // ── OR the precomputed indices into the low 16 bits of every DST word.
    // The body (12 instructions: 4 loads + 4 ORs + 4 stores + 4 IADDs) is
    // recorded into replay slots [0..15] once and replayed for the
    // remaining iters.
    lltt::record<lltt::Exec>(0, 16);
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_7, 2);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_7, 16 + 0);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_7, 16 + 2);

    TTI_SFPOR(0, p_sfpu::LREG0, p_sfpu::LREG4, 0);
    TTI_SFPOR(0, p_sfpu::LREG1, p_sfpu::LREG5, 0);
    TTI_SFPOR(0, p_sfpu::LREG2, p_sfpu::LREG6, 0);
    TTI_SFPOR(0, p_sfpu::LREG3, p_sfpu::LREG7, 0);

    TTI_SFPIADD(4, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(4, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(4, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(4, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);

    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_7, 2);
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_7, 16 + 0);
    TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_6, 16 + 2);

    for (int i = 1; i < 4; i++)
    {
        lltt::replay(0, 16);
    }

    // Outer loop over the remaining face-pairs. We hoist the
    // "skip-a-face-pair" SFPLOAD to the prologue of each subsequent outer
    // iter so the trailing iter doesn't pay for it: the next caller
    // (`_topk_xl_local_sort_`) starts with TTI_SETRWC(SET_D), which would
    // clobber RWC anyway.
    //
    // Using ADDR_MOD_4 (+16) collapses what would otherwise be two +8
    // SFPLOADs into a single issue.
    //
    // Iteration count is K-derived:
    //   K=512  → 1 face-pair total (just the initial recording above); no
    //            extra outer iters needed.
    //   K=1024 → 2 face-pairs total → 1 extra outer iter here.
    //   K=2048 → 4 face-pairs total → 3 extra outer iters here.
    constexpr int row_scale_factor = K == 512 ? 1 : K == 1024 ? 2 : 4;
    for (int j = 1; j < row_scale_factor; j++)
    {
        TTI_SFPLOAD(p_sfpu::LREG4, 10, ADDR_MOD_4, 0);

        for (int i = 0; i < 4; i++)
        {
            lltt::replay(0, 16);
        }
    }
}

// =============================================================================
//  Post-reduction PACK-thread phases (`remove_msb_values`, `separate_indices`)
// =============================================================================
//
// These two helpers run AFTER all merge/rebuild work has finished, so they
// are free to reprogram ADDR_MODs without disturbing the bitonic hot path.
// To keep that contract explicit they touch ADDR_MOD_0 (which the bitonic
// init never programs) rather than reusing a slot that the +24 / +40 / +48
// folds depend on.
//
// `remove_msb_values` runs on the PACK thread (TRISC2): the >64K (extended
// 256K) flow needs to pack values out first, then overwrite the value half
// of each DST word with zero, then pack out indices. Owning that overwrite
// on PACK lets the value pack and the zero-overwrite overlap with MATH's
// final merge tail, see the `op.hpp` wiring.

// Program ADDR_MOD_0 with a +2 increment — the stride between the lo16
// and hi16 halves of an FP32 DST word. Used by both `remove_msb_values`
// (overwrite hi16 with 0) and `separate_indices` (write index into the
// neighbouring DST word's hi16).
inline void _topk_xl_remove_msb_values_init_()
{
    // ADDR_MOD_0 — +2 (hi16 vs lo16 stride). Not programmed by
    // `_topk_xl_init_`, so this never collides with the unfused +24
    // fold living in ADDR_MOD_2.
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(ADDR_MOD_0);
}

// Store zero into the hi16 half of every DST word in the K-sized region.
// Runs on PACK (TRISC2): on the extended 256K path the value half has
// already been packed out by the time we get here, so blanking the hi16
// half leaves clean indices in DST ready for the second pack.
//
// K=2048 → 64 iters; K=1024 → 32 iters; K=512 → 16 iters. The body is one
// instruction so there's nothing to replay or MOP across — TRISC2's issue
// rate is fine at this volume.
template <std::uint32_t K>
inline void _topk_xl_remove_msb_values_()
{
    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");
    constexpr int row_scale_factor = K == 512 ? 1 : K == 1024 ? 2 : 4;
    for (int i = 0; i < row_scale_factor * 16; i++)
    {
        // store 0 to the hi-16 bits of DST
        TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::FP16B, ADDR_MOD_0, 0);
    }
}

// Same +2-stride ADDR_MOD setup as `remove_msb_values_init`. Kept as a
// separate function to keep the LLK API surface 1:1 with the compute
// kernel API; the bodies are identical but the calling site differs.
//
// In addition to programming ADDR_MOD_0, this also stashes the runtime
// `group_id_bit_shift` into LREG12 so that `_topk_xl_separate_indices_`
// can SFPSHFT the (template-static) `group_id` by it at body issue time.
inline void _topk_xl_separate_indices_init_(std::uint32_t group_id_bit_shift)
{
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(ADDR_MOD_0);

    // Save the runtime group_id bit shift into a programmable SFPU constant
    // register (LREG12), so _topk_xl_separate_indices_ can SFPSHFT by it.
    _sfpu_load_config32_(p_sfpu::LREG12, /*upper16=*/0, /*lower16=*/group_id_bit_shift);
}

// ============================================================================
// TOPK_LARGE_INDICES ADDITION: row-major UINT32 index split support
// ============================================================================
//
// The base `_topk_xl_separate_indices_` helper keeps the low-16 index field as a
// tile/SFPU coordinate. This TTNN op needs public row-major UINT32 indices, so
// the helpers in this block decode that coordinate to a within-row position,
// track the input chunk base, and write the result into the existing unfused
// INT32 index region consumed by merge/rebuild.
//
inline void _topk_xl_separate_indices_row_major_init_(std::uint32_t chunk_base)
{
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(ADDR_MOD_0);

    _sfpu_load_config32_(p_sfpu::LREG12, /*upper16=*/(chunk_base >> 16) & 0xFFFF, /*lower16=*/chunk_base & 0xFFFF);
    _sfpu_load_config32_(p_sfpu::LREG13, /*upper16=*/0, /*lower16=*/0x000F);
    _sfpu_load_config32_(p_sfpu::LREG14, /*upper16=*/0, /*lower16=*/0x0001);
}

template <std::uint32_t chunk_base_upper16>
inline void _topk_xl_separate_indices_row_major_init_upper_(std::uint32_t chunk_base_low16)
{
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(ADDR_MOD_0);

    _sfpu_load_config32_(p_sfpu::LREG12, /*upper16=*/chunk_base_upper16, /*lower16=*/chunk_base_low16 & 0xFFFF);
    _sfpu_load_config32_(p_sfpu::LREG13, /*upper16=*/0, /*lower16=*/0x000F);
    _sfpu_load_config32_(p_sfpu::LREG14, /*upper16=*/0, /*lower16=*/0x0001);
}

template <std::uint32_t chunk_base_upper16, std::uint32_t chunk_base_lower16>
inline void _topk_xl_separate_indices_row_major_init_static_()
{
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(ADDR_MOD_0);

    _sfpu_load_config32_(p_sfpu::LREG12, /*upper16=*/chunk_base_upper16, /*lower16=*/chunk_base_lower16);
    _sfpu_load_config32_(p_sfpu::LREG13, /*upper16=*/0, /*lower16=*/0x000F);
    _sfpu_load_config32_(p_sfpu::LREG14, /*upper16=*/0, /*lower16=*/0x0001);
}

inline void _topk_xl_separate_indices_row_major_reinit_()
{
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(ADDR_MOD_0);
}

template <std::uint32_t K>
inline void _topk_xl_separate_indices_row_major_advance_chunk_base_()
{
    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");
    TTI_SFPMOV(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);
    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_USHORT, K);
    TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG0, 4);
    TTI_SFPCONFIG(0, p_sfpu::LREG12, 0);
}

// In:
//   LREG0 raw low bits: [col bits at 10:6 | tile bit at 5 | row bits at 4:0]
// Out:
//   LREG0: row-major within-chunk index.
//
// K=1024 ignores raw bit 5. K=2048 maps raw bit 5 to row-major bit 10.
template <std::uint32_t K>
inline void _topk_xl_decode_row_major_index_()
{
    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");

    // part0 = col_low4: ((raw >> 6) & 0xf)
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
    TTI_SFPSHFT((-6) & 0xFFF, p_sfpu::LREG2, p_sfpu::LREG2, 1);
    TTI_SFPAND(0, p_sfpu::LREG13, p_sfpu::LREG2, 0);

    // part1 = row_low4 << 4
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG3, 0);
    TTI_SFPAND(0, p_sfpu::LREG13, p_sfpu::LREG3, 0);
    TTI_SFPSHFT(4, p_sfpu::LREG3, p_sfpu::LREG3, 1);
    TTI_SFPOR(0, p_sfpu::LREG3, p_sfpu::LREG2, 0);

    // part2 = col_hi << 8
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG3, 0);
    TTI_SFPSHFT((-10) & 0xFFF, p_sfpu::LREG3, p_sfpu::LREG3, 1);
    TTI_SFPAND(0, p_sfpu::LREG14, p_sfpu::LREG3, 0);
    TTI_SFPSHFT(8, p_sfpu::LREG3, p_sfpu::LREG3, 1);
    TTI_SFPOR(0, p_sfpu::LREG3, p_sfpu::LREG2, 0);

    // part3 = row_hi << 9
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG3, 0);
    TTI_SFPSHFT((-4) & 0xFFF, p_sfpu::LREG3, p_sfpu::LREG3, 1);
    TTI_SFPAND(0, p_sfpu::LREG14, p_sfpu::LREG3, 0);
    TTI_SFPSHFT(9, p_sfpu::LREG3, p_sfpu::LREG3, 1);
    TTI_SFPOR(0, p_sfpu::LREG3, p_sfpu::LREG2, 0);

    if constexpr (K == 2048)
    {
        // part4 = tile_in_sequence << 10
        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG3, 0);
        TTI_SFPSHFT((-5) & 0xFFF, p_sfpu::LREG3, p_sfpu::LREG3, 1);
        TTI_SFPAND(0, p_sfpu::LREG14, p_sfpu::LREG3, 0);
        TTI_SFPSHFT(10, p_sfpu::LREG3, p_sfpu::LREG3, 1);
        TTI_SFPOR(0, p_sfpu::LREG3, p_sfpu::LREG2, 0);
    }

    TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG0, 0);
}

template <std::uint32_t K, bool APPROXIMATION_MODE>
inline void _topk_xl_separate_indices_row_major_()
{
    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");
    constexpr int row_scale_factor       = K == 512 ? 1 : K == 1024 ? 2 : 4;
    constexpr int num_tiles_per_sequence = K == 512 ? 1 : K == 1024 ? 1 : 2;
    constexpr int indices_offset         = num_tiles_per_sequence * 64;

    for (int i = 0; i < row_scale_factor * 16; i++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);

        // Value region: keep BF16 value high half and clear low half.
        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_LOWER, 0);

        // Index region: clear high half, decode low tile coordinate, OR chunk base.
        TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0);
        _topk_xl_decode_row_major_index_<K>();
        TTI_SFPOR(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);

        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_0, indices_offset);
    }
}

// ============================================================================
// END TOPK_LARGE_INDICES ADDITION: row-major UINT32 index split support
// ============================================================================

// Split the packed `[ bf16 value | u16 index ]` word into a separate
// values-only region and an index-only region with the "extended"
// index `[group_id << group_id_bit_shift | u16 index]`. Used at stage 5
// of the >64K reduction tree to switch the remaining stages into unfused
// mode. The bit position where `group_id` is placed is set at init time
// via `_topk_xl_separate_indices_init_(group_id_bit_shift)`, which stores
// that shift into LREG12 so this body can SFPSHFT by it.
//
// Body per face-pair (9 instructions):
//   SFPLOAD  LREG0 ← DST[+0]                       (value+index word)
//   SFPMOV   LREG1 ← LREG0                         (copy)
//   SFPLOADI LREG1 imm-low  0                      (zero lo16 of LREG1 → keeps hi16 = value)
//   SFPLOADI LREG0 imm-high 0                      (clear LREG0 hi16; keeps lo16 = u16 index)
//   SFPLOADI LREG2 imm-ushort group_id             (LREG2 = zero-extended group_id, lo16 = group_id)
//   SFPSHFT  LREG2 ← LREG2 << LREG12               (shift group_id by runtime bit position)
//   SFPOR    LREG0 |= LREG2                        (LREG0 = [group_id << shift | u16 index])
//   SFPSTORE LREG1 → DST[+0]                       (values)
//   SFPSTORE LREG0 → DST[+indices_offset]          (extended indices, advances Dst by +2)
//
// `indices_offset` is K-derived: the index region sits one sorted-sequence
// worth of tiles away from the values region. For K=2048 (2 tiles per
// sequence) that's +128; for K=512/1024 (1 tile per sequence) it's +64.
//
// Recording-then-replaying this 9-instruction body saves 8 × (n-1) TRISC1
// issues vs an inline n-iter loop, at zero correctness risk because the
// body is fully template-static (`group_id` is a template parameter) and
// the per-call shift amount has already been latched into LREG12.
template <std::uint32_t K, bool APPROXIMATION_MODE, std::uint32_t group_id>
inline void _topk_xl_separate_indices_()
{
    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");
    constexpr int row_scale_factor       = K == 512 ? 1 : K == 1024 ? 2 : 4;
    constexpr int num_tiles_per_sequence = K == 512 ? 1 : K == 1024 ? 1 : 2;
    constexpr int indices_offset         = num_tiles_per_sequence * 64;

    lltt::record<lltt::Exec>(0, 9);
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_LOWER, 0);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0);
    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_USHORT, group_id);
    TTI_SFPSHFT(0, p_sfpu::LREG12, p_sfpu::LREG2, 0);
    TTI_SFPOR(0, p_sfpu::LREG2, p_sfpu::LREG0, 0);
    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_0, indices_offset);

    for (int i = 1; i < row_scale_factor * 16; i++)
    {
        lltt::replay(0, 9);
    }
}

} // namespace sfpu
} // namespace ckernel
