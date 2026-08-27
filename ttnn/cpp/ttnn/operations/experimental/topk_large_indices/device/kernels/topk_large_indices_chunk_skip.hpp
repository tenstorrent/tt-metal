// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Data-dependent chunk-skip early-out for the ROW-PARALLEL topk_large_indices
// compute kernel (compute.cpp) ONLY. The
// column-parallel merge-tree kernels must not include this header: their
// per-slice streams are 1-5 chunks long and the skip probability there is
// ~0 on any realistic distribution (measured in the tileskip forecast), so
// the test would be pure regression.
//
// MECHANISM
// ---------
// The row-parallel leaf loop streams all N/K chunks of a row through one
// core: per chunk it pays copy + lsb-stamp + fused bitonic sort + index
// split + merge + rebuild (~2 merge units). For chunk c >= kFirstTestedChunk
// we test, right after the (unavoidable) copy of the chunk into DST, whether
// the chunk can possibly contribute to the final top-USER_K:
//
//     skip  iff  max(chunk) < T,   T = running USER_K-th largest survivor
//
// where T is read from the resident sorted-descending window in DST (the
// unfused values region of slot0, refreshed by every merge+rebuild) at the
// DST location holding rank USER_K-1.
//
// SOUNDNESS (exact top-k value multiset is preserved)
// ---------------------------------------------------
// Invariant: at test time, T equals the USER_K-th largest value of ALL row
// elements seen so far (processed or skipped). Proof sketch, by induction:
//   * The resident window holds the top-llk_K of all *merged* elements, and
//     USER_K <= llk_K, so rank USER_K-1 of the window is the USER_K-th
//     largest merged element.
//   * Every skipped element was < T_at_its_skip_time <= current T (T is
//     monotone non-decreasing: merges only improve the window), so skipped
//     elements never displace the top-USER_K of the merged set; the USER_K-th
//     largest of "merged" equals that of "all seen".
// Now let x be any element of an exact top-USER_K set of the full row, i.e.
// x >= v_k, the final USER_K-th largest value. Since T <= v_k at all times
// (at most USER_K-1 elements ever exceed v_k), a chunk containing x has
// chunk_max >= x >= v_k >= T, so the STRICT test chunk_max < T never fires
// on it: every top-k candidate enters the window, survives every subsequent
// top-llk_K merge (it stays within the top-USER_K <= top-llk_K), and reaches
// the output. Conversely a skipped element is < T <= v_k and belongs to no
// exact top-k set. Boundary ties (chunk_max == T) are NOT skipped, so tie
// candidates at v_k always enter the merge; tie *membership* is resolved by
// the selected deterministic tie policy (stable mode uses ascending global
// index, non-stable mode uses the established merge order). Skip decisions
// are pure functions of the input data, so results remain deterministic for
// a fixed input.
//
// DECISION MACHINERY (silicon-validated components)
// -------------------------------------------------
// * MATH computes max(chunk) on the SFPU: 32 SFPLOADs per 32-bit DST tile
//   (auto-advance-2 walk covers all 1024 datums; validated by the cgtceq
//   bench walk model), lane-wise SFPSWAP/ALL_ROWS_MAX accumulate, then a
//   full cross-lane max fold (SFPTRANSP + 3 SWAPs + 7x(SFPSHFT2-ROR1 +
//   SFPNOP + SWAP) -- the ROR1 needs a trailing SFPNOP, and rotations pull
//   from the RUNNING max because SFPSWAP clobbers both operands).
// * The folded max is SFPSTOREd (raw bits, InstrModLoadStore::INT32) into
//   the first INDICES tile of the chunk's own DST sequence -- dead space at
//   test time: the index split only writes it later, and only when the chunk
//   is NOT skipped.
// * MATH RISC does tensix_sync() then reads both the folded max and the
//   threshold word through the memory-mapped DST window @0xFFBD8000
//   (RISC_DEST_ACCESS_CTRL SEC1 configured once per kernel for Float32 +
//   swizzle: on Blackhole this returns raw fp32 words at word index
//   row*16+col -- the in-tree dprint_tensix_dest_reg float32 path). The
//   tensix_sync/fold/single-word-read rendezvous is the 81-cycle S0/R0
//   arrangement validated in tt_metal/tt-llk/tests/sources/cgtceq_perf.cpp;
//   the same sync also orders the preceding rebuild's stores, so the
//   threshold read needs no extra sync.
// * The compare runs on the MATH RISC in sign-magnitude (== IEEE float)
//   order via a monotone bit transform. Value payloads occupy the high bf16
//   bits end to end; stable mode may use the low 16 bits for its rank stamp,
//   which is masked from the threshold before comparison. The bf16 datapath
//   admits no NaNs (canonicalized to inf on ingest), so sign-magnitude order
//   is total and correct.
// * Cross-TRISC propagation: MATH -> UNPACK through the T1->T0 hardware
//   mailbox (ckernel::mailbox_write / blocking mailbox_read). NOTE: this op's
//   copy path ALREADY uses the same T1->T0 FIFO -- topk_xl_copy's
//   unpack-to-dest handoff mails dst_index per tile (cmath_common.h
//   set_dst_write_addr<.., UnpackDestination::DestReg>). That stays safe
//   because the FIFO is order-preserving, single-writer/single-reader, and
//   both threads issue their reads/writes in identical per-chunk program
//   order with identical branch predicates: [dst_index x tiles] then
//   [skip decision, tested chunks only]. Exactly one skip write pairs with
//   exactly one skip read; worst-case FIFO occupancy is 3 (skip + 2 dst_index
//   at K=2048), within the depth-4 hardware FIFO, and an overflow would only
//   stall the writer, not reorder. PACK has no per-chunk work in the leaf
//   loop and needs no decision.
// * UNPACK must branch in tandem with MATH: its per-chunk conditional work is
//   the two llk_unpack_set_srcb_dummy_valid() calls (local_sort + rebuild).
//   Issuing them for a skipped chunk would leave SrcA/SrcB banks valid with
//   no consumer (the skipped sort never runs its CLEARDVALID), wedging the
//   next chunk's real unpack -- hence the mailbox, not an unconditional path.
//
// WHY THE CHUNK IS STILL UNPACKED: the copy into DST is the only way to
// inspect the data; skipping saves the fused sort + index split + merge +
// rebuild (the ~2 "merge units" that dominate the leaf loop).

#pragma once

#include <cstdint>
#include "api/compute/compute_kernel_api.h"

#ifdef TRISC_MATH
#include "ckernel_dest.h"  // configure_dest_access, RISCV_DEST_START_ADDR
#endif

// Bring-up tracing of every decision (MATH). Never enable with perf runs.
// #define CHUNK_SKIP_DEBUG 1
#if defined(CHUNK_SKIP_DEBUG) && defined(TRISC_MATH)
#include "api/debug/dprint.h"
#endif

// Skip-rate telemetry (paper gap G4): per-row skipped-chunk count plus a
// per-position skip bitmask, emitted by MATH over DPRINT once per row,
// OUTSIDE the chunk loop. Observation only -- outputs must stay bit-exact.
// DPRINT serializes: never enable together with perf runs or the device
// profiler, and always run with the DPRINT server up (TT_METAL_DPRINT_CORES
// set), otherwise the kernel-side writer stalls. When left undefined (the
// default) every telemetry token preprocesses away and the compiled kernels
// are byte-identical to the pre-telemetry binaries (same guarantee class as
// CHUNK_SKIP_DEBUG).
// #define CHUNK_SKIP_TELEMETRY 1
#if defined(CHUNK_SKIP_TELEMETRY) && defined(TRISC_MATH)
#include "api/debug/dprint.h"
#endif

namespace topk_large_indices_chunk_skip {

// First chunk index eligible for the skip test.
//
// Floor of 2: chunk 0 seeds the window and chunk 1 tests against a window
// that has only been LOCALLY sorted (its DST layout is the post-local-sort
// one, not the post-rebuild rank layout the threshold address below assumes).
//
// Amortization gate USER_K/4: for iid data the skip probability at stream
// position c is P = C(c*K, USER_K)/C((c+1)*K, USER_K) ~= e^(-USER_K/(c+1)),
// negligible (< e^-4 ~= 1.8%) below c = USER_K/4 -- there the test is pure
// overhead (~150-350 ns/chunk, measured ungated: +6.8% at user_k=512@128
// chunks, +1.8% at user_k=1536@25 chunks, while forfeiting < 1 expected skip
// per row). The gate is a compile-time function of USER_K only, so all
// TRISCs derive the identical tested-chunk set.
//
// Gate-divisor A/B knob: default 4 == the shipping USER_K/4 constant (the
// default compiles to identical code). Swept in-source by
// tests/.../_topk_large_indices_gate_ab.sh, which sed-edits the value per
// arm (JIT rehash, no host rebuild). Do NOT change the default without the
// full A/B evidence + guard battery (see the harness header + RUN_PLAN).
#ifndef CHUNK_SKIP_GATE_DIVISOR
#define CHUNK_SKIP_GATE_DIVISOR 4
#endif
template <uint32_t USER_K>
constexpr uint32_t first_tested_chunk() {
    constexpr uint32_t layout_floor = 2;
    constexpr uint32_t amortization_floor = USER_K / CHUNK_SKIP_GATE_DIVISOR;
    return amortization_floor > layout_floor ? amortization_floor : layout_floor;
}

// ---------------------------------------------------------------------------
// rank -> DST MMIO word address of the resident window's values region
// (slot0, MMIO word = physical_row*16 + col).
//
// EMPIRICALLY CALIBRATED ON SILICON (p150a, 2026-08-16): the CHUNK_SKIP_DIAG
// dump in compute.cpp was run per K with a distinct-monotone bf16 input
// (bits 0x3800+j) and every dumped word matched against torch rank order.
// Result, exact for all ranks of all three windows (512/1024/2048 of 512/
// 1024/2048 words):
//
//     word(r) = (r % rows) * 16 + r / rows,   rows = K/16
//
// i.e. the post-rebuild descending window is COLUMN-MAJOR over the values
// region's populated physical rows (32 rows for K=512 -- the top half-tile --
// 64 for K=1024, 128 for K=2048), 16 words per row. Calibrated on silicon
// by exhaustive DST dumps against torch rank order at all three windows.
//
// Soundness note: an address error here would be UNSOUND only if it pointed
// at a HIGHER-ranked (larger) element; the exhaustive per-rank calibration
// rules that out.
// ---------------------------------------------------------------------------
template <uint32_t K>
constexpr uint32_t rank_to_values_word(uint32_t r) {
    constexpr uint32_t rows = K / 16;
    return (r % rows) * 16 + (r / rows);
}

#ifdef TRISC_MATH

// Monotone map from IEEE-754 bit pattern (sign-magnitude) to unsigned order.
inline uint32_t sm_key(uint32_t x) { return (x & 0x80000000u) ? ~x : (x | 0x80000000u); }

namespace sfpu_detail {

// SFPU max-fold over the values tiles of the chunk sequence based at the
// wrapper's dst_index (slot1). Leaves the 32-lane global max SFPSTOREd (raw
// bits) at the first row of the sequence's indices region (offset 64*tiles
// from the base -- where the auto-advance walk counter lands). Clobbers
// LREG0..LREG7 (all scratch at this point in the leaf loop; the LLK config
// constants live in LREG12+ and are untouched). ADDR_MOD_0/ADDR_MOD_6 are
// reprogrammed here and re-initialized by every downstream topk_xl_*_init
// before their next use (the established per-phase re-init pattern of this
// kernel).
template <uint32_t K>
inline void chunk_maxfold() {
    using namespace ckernel;
    constexpr uint32_t tiles = (K + 1023) / 1024;
    constexpr uint32_t num_loads = tiles * 32;

    // The preceding classic top-k phase enables LaneConfig index tracking.
    // This custom SFPU call bypasses the normal unary init, so clear that
    // state before using LREG4/LREG5 as ordinary max-fold scratch registers.
    sfpu::_init_sfpu_config_reg();

    // ADDR_MOD_6: the load walk (advance 2 u10 units per SFPLOAD, the
    // 4-row/32-datum coverage stride). ADDR_MOD_0: no movement (store).
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(ADDR_MOD_6);
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_0);

    // Accumulator LREG0 and its transpose partners LREG1..3 = bf16 -inf
    // (0xFF800000): identity for max, and matches the copy path's padding.
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0x0000);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0xFF80);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_LOWER, 0x0000);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_UPPER, 0xFF80);
    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, 0x0000);
    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, 0xFF80);
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_LOWER, 0x0000);
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_UPPER, 0xFF80);

    // Lane-wise max accumulate. Two alternating load temps so each SFPLOAD
    // issues under the previous SFPSWAP's bubble. SFPSWAP(0, VC, VD,
    // ALL_ROWS_MAX) puts max into VC and min into VD (SFPSWAP.md functional
    // model, silicon-confirmed via the CHUNK_SKIP_DEBUG trace), so the
    // accumulator rides in the VC slot.
    for (uint32_t i = 0; i < num_loads / 2; ++i) {
        TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_6, 0);
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG4, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_6, 0);
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);
    }

    // Cross-lane fold: 4 subvectors -> 1 (TRANSP + 3 SWAPs against the -inf
    // partners), then 8 lanes -> 1 (7x rotate-running-max). SFPSHFT2-ROR1
    // requires the trailing SFPNOP; rotations source LREG0 (the running max)
    // each step because SFPSWAP overwrites both operands.
    TTI_SFPNOP;
    TTI_SFPTRANSP(0, 0, 0, 0);
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    for (uint32_t i = 0; i < 7; ++i) {
        TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG1, 3 /*SUBVEC_SHFLROR1*/);
        TTI_SFPNOP;
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    }

    // The walk counter now sits at 64*tiles past the base = row 0 of the
    // sequence's indices region. Store the folded max there (raw bits); lane
    // 0 lands at MMIO word 0 of that 4-row window (cgtceq R0_WORD=0).
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_0, 0);
}

}  // namespace sfpu_detail

#endif  // TRISC_MATH

// One-time per-kernel setup (call once before the row loop, MATH only):
// program MATH's RISC_DEST_ACCESS_CTRL section for raw fp32 MMIO DST reads.
// Only affects the RISC MMIO window (dprint recipe: no restore needed).
inline void chunk_skip_configure() {
#ifdef TRISC_MATH
    ckernel::configure_dest_access<ckernel::MathThreadId>(DataFormat::Float32, /*enable_swizzle=*/true);
    ckernel::tensix_sync();
    // Match the dprint DEST-MMIO recipe's conservative posted-config-write settle delay.
    ckernel::wait(1000 /* cycles */);
#endif
}

// Per-tested-chunk decision, callable from all three TRISC builds.
//  * MATH: computes the decision and mails it to UNPACK.
//  * UNPACK: blocks on the mailbox for MATH's decision.
//  * PACK: constant false (no per-chunk work exists on PACK in this loop).
template <uint32_t K, uint32_t USER_K>
inline bool chunk_skip_decide(uint32_t slot1) {
    static_assert(USER_K >= 1 && USER_K <= K, "USER_K must be in [1, K]");
    (void)slot1;
#if defined(TRISC_MATH)
    constexpr uint32_t tiles = (K + 1023) / 1024;
    constexpr uint32_t threshold_word = rank_to_values_word<K>(USER_K - 1);
    _llk_math_eltwise_unary_sfpu_params_(sfpu_detail::chunk_maxfold<K>, slot1, VectorMode::RC_custom);
    ckernel::tensix_sync();
    volatile uint32_t* mm = reinterpret_cast<volatile uint32_t*>(RISCV_DEST_START_ADDR);
    const uint32_t max_bits = mm[(slot1 + tiles) * 64 * 16];
    // Stable-tie mode stamps the sequential rank in the low 16 bits of the
    // resident value word. The skip predicate compares values only.
    const uint32_t thr_bits = mm[threshold_word] & 0xFFFF0000u;
    const bool skip = sm_key(max_bits) < sm_key(thr_bits);
#ifdef CHUNK_SKIP_DEBUG
    DPRINT("CSD max {} thr {} skip {}\n", max_bits, thr_bits, skip ? 1u : 0u);
#endif
    ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, skip ? 1u : 0u);
    return skip;
#elif defined(TRISC_UNPACK)
    return ckernel::mailbox_read(ckernel::ThreadId::MathThreadId) != 0;
#else
    return false;
#endif
}

// ---------------------------------------------------------------------------
// Telemetry recorder (CHUNK_SKIP_TELEMETRY builds only). MATH records the
// decisions it already owns; UNPACK/PACK get no-op stubs so the shared call
// sites in compute.cpp compile on all three TRISC builds. The recorder adds NO mailbox traffic and keeps the tested-chunk
// predicate identical on all TRISCs, so the T1->T0 FIFO occupancy analysis
// above (worst case 3 <= depth 4) is unchanged. The tested-chunk count is
// compile-time deterministic (num_chunks - first_tested_chunk), so only the
// data-dependent SKIPS are recorded. Emission, once per row (parsed by
// tests/.../_topk_large_indices_skip_telemetry_parse.py):
//   CSTL r <row> n <num_chunks> f <first_tested> s <skipped>
//   CSTLM <word_idx> <mask_word>     (ceil(num_chunks/32) lines per row)
// ---------------------------------------------------------------------------
#ifdef CHUNK_SKIP_TELEMETRY
#ifdef TRISC_MATH

// Route width ceiling 2^19 / min llk window 512 = 1024 chunks max -> 32
// words. File-scope inline arrays land in TRISC1 local-memory .bss, not the
// stack.
inline constexpr uint32_t kTelemetryMaxChunkWords = 32;
inline uint32_t g_telemetry_skip_mask[kTelemetryMaxChunkWords];
inline uint32_t g_telemetry_row_skipped;

inline void telemetry_row_begin(uint32_t num_chunks) {
    const uint32_t words = (num_chunks + 31) / 32;
    for (uint32_t w = 0; w < words && w < kTelemetryMaxChunkWords; ++w) {
        g_telemetry_skip_mask[w] = 0;
    }
    g_telemetry_row_skipped = 0;
}

inline void telemetry_record(uint32_t chunk, bool skipped) {
    if (skipped) {
        ++g_telemetry_row_skipped;
        const uint32_t w = chunk / 32;
        if (w < kTelemetryMaxChunkWords) {
            g_telemetry_skip_mask[w] |= (1u << (chunk % 32));
        }
    }
}

// Runs between FPU phases on the MATH RISC, after all of the row's skip
// decisions have completed -- it cannot interleave with the decision mailbox.
template <uint32_t USER_K>
inline void telemetry_row_end(uint32_t row, uint32_t num_chunks) {
    DPRINT("CSTL r {} n {} f {} s {}\n", row, num_chunks, first_tested_chunk<USER_K>(), g_telemetry_row_skipped);
    const uint32_t words = (num_chunks + 31) / 32;
    for (uint32_t w = 0; w < words && w < kTelemetryMaxChunkWords; ++w) {
        DPRINT("CSTLM {} {}\n", w, g_telemetry_skip_mask[w]);
    }
}

#else   // CHUNK_SKIP_TELEMETRY on UNPACK/PACK: observe nothing.
inline void telemetry_row_begin(uint32_t) {}
inline void telemetry_record(uint32_t, bool) {}
template <uint32_t USER_K>
inline void telemetry_row_end(uint32_t, uint32_t) {}
#endif  // TRISC_MATH
#endif  // CHUNK_SKIP_TELEMETRY

}  // namespace topk_large_indices_chunk_skip
