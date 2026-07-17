// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_chain.hpp
 * @brief Element-wise compute helper — one chain surface for all eltwise patterns.
 *
 * Every element-wise compute pattern (FPU binary, SFPU unary/binary/ternary, dest-reuse, copy,
 * pack, fill, rand, unary broadcast) is expressed as a sequence of chain elements passed to
 * `eltwise_chain(shape, elem0, elem1, ...)` (shape is an `EltwiseShape`, e.g.
 * `EltwiseShape::tiles(num_tiles)`).
 *
 * The chain owns, per call:
 *   - the dst-sync window (`tile_regs_acquire/commit/wait/release`);
 *   - per-element init and exec dispatch;
 *   - CB lifecycle (input wait/pop, output reserve/push), selected by each element's policy enums;
 *   - input- and pack-side dtype reconfig, compile-time-elided when the previous CB on that side
 *     already carries the right format;
 *   - compile-time invariant checks (illegal lifecycle/index combos, duplicate upfront CBs,
 *     pack-output collisions, hoist-safety).
 *
 * Caller-init contract
 * --------------------
 * The chain never issues engine-wide ("BIG") init. The caller owns `compute_kernel_hw_startup`
 * (plus `binary_op_init_common` / `mm_init` / `reduce_init` when the kernel mixes those
 * primitives). The chain owns only per-element init — `*_tile_init`, `init_bcast`,
 * `copy_tile_init` / `copy_tile_to_dst_init_short`, the `reconfig_data_format_*` fold, and the
 * dst-sync lifecycle. Do not add a `*_with_init` wrapper that folds `compute_kernel_hw_startup`
 * into the chain: it is only correct for single-stage kernels and breaks multi-stage / mid-loop
 * ones.
 *
 * compute_kernel_hw_startup placement
 * -----------------------------------
 * Call it as the first statement of `MAIN()` for any chain that both reads and writes a CB.
 * Multi-stage kernels (a different pack-output CB per stage) issue one boot per stage: stage 1
 * at the top of `MAIN()`, later stages immediately before their chain call. It is an MMIO write,
 * so it is undefined mid-`MAIN()`; when an outer `binary_op_init_common` already covers the chain
 * (the moreh inner-loop pattern), omit it.
 *
 * FP32 DEST accumulation
 * ----------------------
 * Determined kernel-wide by the build flag `FP32_DEST_ACC_EN`; `DEST_AUTO_LIMIT` (dest_helpers.hpp)
 * already halves the usable slot count when it is on. There is no per-element opt-in and no
 * mid-kernel `enable_fp32_dest_acc()` / `disable_fp32_dest_acc()` toggle.
 *
 * Examples
 * --------
 *   // Streaming unary — Exp(x) -> out (dfb_* are dataflow-buffer ids, i.e. buffer indices)
 *   eltwise_chain(EltwiseShape::tiles(num_tiles),
 *       CopyTile<dfb_in, Dst::D0, InputLifecycle::Streaming>{},
 *       Exp<>{},
 *       PackTile<dfb_out, OutputLifecycle::Streaming>{});
 *
 *   // Streaming binary — A + B -> out (BinaryFpu writes DEST; the output buffer lives on PackTile)
 *   eltwise_chain(EltwiseShape::tiles(num_tiles),
 *       BinaryFpu<dfb_a, dfb_b, BinaryFpuOp::Add>{},
 *       PackTile<dfb_out, OutputLifecycle::Streaming, PackTileReconfig::Output>{});
 *
 * Not supported: per-iteration (mid-loop) dtype swaps — each element's dtype reconfig point is
 * resolved per element at compile time (fold-driven, emitted once at element entry), so there is
 * no per-loop-iteration reconfig path; pack-relu; and the legacy `acquire_dst/release_dst` macros
 * (modern dst-sync only).
 */

#include <cstdint>
#include <type_traits>
#include <utility>

#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "api/compute/common_globals.h"  // ALWI (used by the public eltwise_chain() declaration)
// The heavier LLK / compute-API includes + <tuple> are impl-only and live in eltwise_chain.inl.

namespace compute_kernel_lib {

// Buffer-identity values throughout the chain (the `dfb`-named NTTPs, accessors, ElemDesc
// fields and the INVALID_DFB / NO_PREV_DFB sentinels) are dataflow-buffer ids: today the
// integer buffer index (a `tt::CBIndex` value, 0..31) passed as an NTTP.

// (The marker-tag hierarchy — CbReaderTag/CbWriterTag/DestOnlyTag + the per-element
//  leaf tags — and the is_*_op_v classification predicates are internal pipeline
//  scaffolding, defined in eltwise_chain.inl. Concrete elements declared below inherit
//  the leaf tags from there.)

// =============================================================================
// 1b. 2D shape — (Ht, Wt) tile grid for the 2D chain overload
// =============================================================================

/// Iteration shape for `eltwise_chain`. Carries both the tile grid (Ht × Wt, both in
/// tiles) and the per-outer-iter `block_size`. Ht=1 expresses the 1D case (no row
/// axis, plain linear walk); the `Row`/`Col` indexing modes degenerate for
/// 1D usage but remain well-defined.
///
/// Factories cover the common construction paths:
///   - `EltwiseShape::tiles(n)`           — 1D, block_size = 1
///   - `EltwiseShape::tiles(n, blk)`      — 1D + block
///   - `EltwiseShape::grid(H, W)`         — 2D, block_size = 1
///   - `EltwiseShape::grid(H, W, blk)`    — 2D + block
///
/// Construction from a tile count is `explicit`: a bare number is NOT accepted as a
/// shape — call sites must spell the iteration shape out as `EltwiseShape::tiles(n)`
/// (or `EltwiseShape::single()` for one tile). This keeps `eltwise_chain(...)` and the
/// convenience wrappers from silently treating a stray integer as a tile count.
///
/// `of/row/col/single` aliases mirror `binary_op_helpers`' `BinaryInputBlockShape`.
struct EltwiseShape {
    uint32_t Ht;
    uint32_t Wt;
    uint32_t block_size;

    constexpr EltwiseShape(uint32_t H, uint32_t W, uint32_t blk = 1) : Ht(H), Wt(W), block_size(blk) {}

    // Explicit: bare numbers are forbidden at call sites. Use EltwiseShape::tiles(n) or
    // EltwiseShape::single() so the iteration shape is always written out.
    explicit constexpr EltwiseShape(uint32_t n_tiles) : Ht(1), Wt(n_tiles), block_size(1) {}

    static constexpr EltwiseShape tiles(uint32_t n, uint32_t blk = 1) { return {1, n, blk}; }
    static constexpr EltwiseShape grid(uint32_t H, uint32_t W, uint32_t blk = 1) { return {H, W, blk}; }

    static constexpr EltwiseShape of(uint32_t r, uint32_t c) { return {r, c, 1}; }
    static constexpr EltwiseShape row(uint32_t c) { return {1, c, 1}; }
    static constexpr EltwiseShape col(uint32_t r) { return {r, 1, 1}; }
    static constexpr EltwiseShape single() { return {1, 1, 1}; }
};

/// Who performs the chain's one-time setup — init + reconfig — the leading template arg to
/// `eltwise_chain`. This is about *ownership*, NOT about whether inits are hoistable: which inits
/// are hoistable is deduced from the chain's uniformity and is never a manual choice.
///
///   eltwise_chain(shape, elts...);                       // default: SetupOwner::Chain
///   // To hoist the setup out of your own loop: emit it ONCE before the loop yourself (e.g. the
///   // original raw *_init call), then hand ownership to the caller so the chain skips it:
///   <emit the chain's one-time setup once, before the loop>
///   for (...) eltwise_chain<SetupOwner::Caller>(EltwiseShape::single(), elts...);
///
/// SetupOwner::Caller is only valid when the chain's entire setup is boot-hoistable (uniform math
/// MOP + SFPU init AND homogeneous pack CBs) — i.e. there's a single "once, before the loop" the
/// caller can own. eltwise_chain static_asserts this; a chain that must re-emit setup per tile
/// (so the caller can't pre-do it once) is a compile error pointing you back to SetupOwner::Chain.
enum class SetupOwner {
    Chain,   // this eltwise_chain call emits the one-time setup (init + reconfig)
    Caller,  // the caller emitted it once, outside the loop — the chain emits none of it here
};

// -----------------------------------------------------------------------------
// Skip-compute — a performance-debugging BUILD knob, NOT part of the eltwise_chain API.
//
// With CKL_ELTWISE_CHAIN_SKIP_COMPUTE=1, every eltwise_chain in the translation unit emits only the
// CB lifecycle (wait/pop/reserve/push) + the tile_regs window, skipping all init, reconfig, and
// compute. CB counts are unchanged, so the reader/writer handshake holds — no hang, just fast
// garbage output.
//
// USE IT to profile a kernel skip-off vs skip-on: the delta is the compute+init cost, and the
// skip-on time is the CB/data-movement floor. That split tells you whether an eltwise kernel is
// compute-bound or dataflow-bound before you spend effort optimizing the wrong half.
//
// DON'T ship it: output is garbage, so it is only ever a local profiling build — never production,
// and never a correctness run.
//
// Opt in per kernel, before the include (call sites never change):
//   #define CKL_ELTWISE_CHAIN_SKIP_COMPUTE 1
//   #include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
// See tests/axes/skip_compute_exp.cpp for the pattern.
//
// Scope: the skip covers EVERY chain walk — ordinary, L1-accumulation, and DEST-accumulation — so no
// chain silently ignores the knob. In every case the CB lifecycle stays intact and only init +
// reconfig + compute are elided.
#ifndef CKL_ELTWISE_CHAIN_SKIP_COMPUTE
#define CKL_ELTWISE_CHAIN_SKIP_COMPUTE 0
#endif

// =============================================================================
// 1c. Taxonomy: Lifecycle as reserve/push axes plus a purpose tag
// =============================================================================
//
// Each input's lifecycle is a `(WaitPolicy, PopPolicy)` pair and each output's core behavior is a
// `(ReservePolicy, PushPolicy)` pair. OutputLifecycle also carries a purpose tag so the no-op
// L1-accumulation lifecycle remains distinguishable from general CallerManaged synchronization.
// The named constants below are the legal set:
// `is_legal_input_lifecycle` / `is_legal_output_lifecycle` are whitelists of exactly
// those constants. A custom struct literal is accepted only when it equals one of the
// named constants (e.g. `InputLifecycle{WaitPolicy::Upfront, PopPolicy::PerTile}` is just
// another spelling of `InputLifecycle::BulkDrain`); arbitrary `{wait, pop}` combinations
// outside the named set are rejected.

enum class WaitPolicy : uint8_t {
    None,        // chain emits no wait_front
    PerTile,     // wait 1 per iter
    PerChunk,    // wait K per K-iter chunk (K = EltwiseShape::block_size, the per-outer-iter tile count)
    PerOuter,    // wait 1 at each OUTER (ht/row) iteration entry — one tile per row
    Upfront,     // wait M once at entry (M = kind's tile count)
    Cumulative,  // wait (i+1)*block_size per iteration (number is clamped as not to wait for more than Wt)
};

enum class PopPolicy : uint8_t {
    None,      // chain emits no pop_front
    PerTile,   // pop 1 per iter
    PerChunk,  // pop K per K-iter chunk
    PerOuter,  // pop 1 at each OUTER (ht/row) iteration exit — one tile per row
    AtEnd,     // pop M once at exit
};

struct InputLifecycle {
    WaitPolicy wait;
    PopPolicy pop;

    constexpr bool operator==(InputLifecycle other) const noexcept { return wait == other.wait && pop == other.pop; }
    constexpr bool operator!=(InputLifecycle other) const noexcept { return !(*this == other); }

    // Named cells — written type-qualified (e.g. `InputLifecycle::Bulk`). Declared here,
    // defined out-of-line below (a static member of the class's own type needs the class
    // complete at the point of definition).
    static const InputLifecycle Streaming, Chunked, Bulk, Pipelined, CallerManaged, BulkDrain, HeldBulk, HeldCumulative,
        HeldStream, DeferredPop, NoWaitPop, OuterStream;
};

// Default: wait for and pop 1 tile each iteration. Use for a normal input read once, tile by tile.
inline constexpr InputLifecycle InputLifecycle::Streaming = {WaitPolicy::PerTile, PopPolicy::PerTile};
// Wait for and pop block_size tiles each iteration (block_size is sized to fit DEST).
inline constexpr InputLifecycle InputLifecycle::Chunked = {WaitPolicy::PerChunk, PopPolicy::PerChunk};
// Wait for the whole input upfront, pop it all at the end. Tile count = the operand's size (Scalar 1, Row Wt, Col Ht,
// Block Ht*Wt). Use when all its tiles must stay available for the whole chain and you do not reuse it afterward.
inline constexpr InputLifecycle InputLifecycle::Bulk = {WaitPolicy::Upfront, PopPolicy::AtEnd};
// Wait for a growing amount each iter ((i+1)*block_size tiles) and pop it all at the end, so compute can start before
// the producer has filled the whole buffer. Use when the producer fills the buffer gradually and you do not reuse it
// afterward.
inline constexpr InputLifecycle InputLifecycle::Pipelined = {WaitPolicy::Cumulative, PopPolicy::AtEnd};
// Chain does not wait or pop. Use when you already waited for the input before the chain and still need it after (chain
// leaves it alone). Something else must have waited for it.
inline constexpr InputLifecycle InputLifecycle::CallerManaged = {WaitPolicy::None, PopPolicy::None};

// Wait for the whole input upfront, then pop 1 tile per iteration. Use when the data is already produced and the
// input buffer is also the output buffer (in-place: same buffer read and written).
inline constexpr InputLifecycle InputLifecycle::BulkDrain = {WaitPolicy::Upfront, PopPolicy::PerTile};

// The next lifecycles do only half the work — the chain does the wait OR the pop, and you do the other half:
//   - Held* : the chain waits but never pops, because you reuse the input after the chain.
//   - *Pop  : the chain does not wait (you already did before the chain), it only pops the input.

// Wait for the whole input upfront, but never pop. Use for an input you wait for once and reuse after the chain (e.g.
// gamma/beta).
inline constexpr InputLifecycle InputLifecycle::HeldBulk = {WaitPolicy::Upfront, PopPolicy::None};
// Chain does not wait, pops everything at the end. Use when you already waited for the input before the chain
// and will not reuse it after.
inline constexpr InputLifecycle InputLifecycle::DeferredPop = {WaitPolicy::None, PopPolicy::AtEnd};
// Wait for a growing amount each iter ((i+1)*block_size tiles), but never pop. Use when the producer fills gradually
// AND you reuse the input after the chain (e.g. layernorm square, where the mean reduce reads it again).
inline constexpr InputLifecycle InputLifecycle::HeldCumulative = {WaitPolicy::Cumulative, PopPolicy::None};
// Wait for 1 tile per iter, never pop. Use when a later iter in the SAME chain reads the same tile again
// (e.g. copy in0, copy in0, then combine — this is the FIRST copy).
inline constexpr InputLifecycle InputLifecycle::HeldStream = {WaitPolicy::PerTile, PopPolicy::None};
// Chain does not wait, pops 1 tile per step. Use for the second reader of a tile whose wait a previous step already did
// (e.g. copy in0, copy in0 — this is the SECOND copy: the wait was already done, so it just pops).
inline constexpr InputLifecycle InputLifecycle::NoWaitPop = {WaitPolicy::None, PopPolicy::PerTile};

// Wait for and pop 1 tile per row (one per outer/ht step). Use for an input with one tile per row, reused across that
// row's columns (Scalar only).
inline constexpr InputLifecycle InputLifecycle::OuterStream = {WaitPolicy::PerOuter, PopPolicy::PerOuter};

/// Validates a caller-constructed `InputLifecycle` against the legal set; every input
/// element static_asserts on it. The wait-only / pop-only cells above are legal;
/// other wait/pop combinations are rejected.
constexpr bool is_legal_input_lifecycle(InputLifecycle lc) noexcept {
    return lc == InputLifecycle::Streaming || lc == InputLifecycle::Chunked || lc == InputLifecycle::Bulk ||
           lc == InputLifecycle::Pipelined || lc == InputLifecycle::CallerManaged || lc == InputLifecycle::BulkDrain ||
           lc == InputLifecycle::HeldBulk || lc == InputLifecycle::HeldCumulative || lc == InputLifecycle::HeldStream ||
           lc == InputLifecycle::DeferredPop || lc == InputLifecycle::NoWaitPop || lc == InputLifecycle::OuterStream;
}

enum class ReservePolicy : uint8_t {
    None,
    PerTile,
    PerChunk,
    PerOuter,  // reserve one accumulated output at each outer-row entry
    Upfront,
    OneUpfront,  // reserve one accumulator tile once at chain entry
};

enum class PushPolicy : uint8_t {
    None,
    PerTile,
    PerChunk,
    PerOuter,  // push one accumulated output at each outer-row exit
    AtEnd,
    OneAtEnd,  // push one accumulator tile once at chain exit
};

enum class OutputLifecyclePurpose : uint8_t {
    General,
    L1Accumulation,
    DestAccumulation,
};

struct OutputLifecycle {
    ReservePolicy reserve;
    PushPolicy push;
    OutputLifecyclePurpose purpose = OutputLifecyclePurpose::General;

    constexpr bool operator==(OutputLifecycle other) const noexcept {
        return reserve == other.reserve && push == other.push && purpose == other.purpose;
    }
    constexpr bool operator!=(OutputLifecycle other) const noexcept { return !(*this == other); }

    // Named cells — written type-qualified (e.g. `OutputLifecycle::Bulk`). Defined out-of-line below.
    // Naming: single-word names (Streaming/Chunked/Bulk/CallerManaged) are the symmetric cells where
    // reserve and push move together; the asymmetric cells spell BOTH axes literally as
    // Reserve<rate>Push<rate> (e.g. ReserveAllPushPerTile = reserve all upfront, push one per tile).
    static const OutputLifecycle Streaming, Chunked, Bulk, ReserveAllPushPerTile, ReserveAllPushPerChunk, CallerManaged,
        ReserveNonePushEnd, L1Accumulation, L1AccumulationCallerManaged, DestAccumulation,
        DestAccumulationCallerManaged;
};

// Default: reserve and push 1 output tile each step.
inline constexpr OutputLifecycle OutputLifecycle::Streaming = {ReservePolicy::PerTile, PushPolicy::PerTile};
// Reserve and push block_size output tiles each step (block_size is sized to fit DEST).
inline constexpr OutputLifecycle OutputLifecycle::Chunked = {ReservePolicy::PerChunk, PushPolicy::PerChunk};
// Reserve all output tiles upfront, push them all at the end.
inline constexpr OutputLifecycle OutputLifecycle::Bulk = {ReservePolicy::Upfront, PushPolicy::AtEnd};
// Reserve all output tiles upfront, but push 1 per step so a downstream consumer can start reading before the chain
// finishes.
inline constexpr OutputLifecycle OutputLifecycle::ReserveAllPushPerTile = {ReservePolicy::Upfront, PushPolicy::PerTile};
// Reserve all output tiles upfront, push block_size tiles at a time.
inline constexpr OutputLifecycle OutputLifecycle::ReserveAllPushPerChunk = {
    ReservePolicy::Upfront, PushPolicy::PerChunk};
// Chain does not reserve or push (it only writes the tile). Use when you wrap the chain in your own reserve/push.
inline constexpr OutputLifecycle OutputLifecycle::CallerManaged = {ReservePolicy::None, PushPolicy::None};
// Do not reserve (you reserved upfront yourself), push all at the end.
inline constexpr OutputLifecycle OutputLifecycle::ReserveNonePushEnd = {ReservePolicy::None, PushPolicy::AtEnd};
// L1 accumulation owns one persistent output tile for the whole chain: reserve it once, repeatedly
// accumulate into it, then publish exactly that one tile at chain exit.
inline constexpr OutputLifecycle OutputLifecycle::L1Accumulation = {
    ReservePolicy::OneUpfront, PushPolicy::OneAtEnd, OutputLifecyclePurpose::L1Accumulation};
// L1 accumulation with both synchronization edges owned by the caller. The chain only accumulates
// into the caller's already-reserved tile; the caller decides when to publish it.
inline constexpr OutputLifecycle OutputLifecycle::L1AccumulationCallerManaged = {
    ReservePolicy::None, PushPolicy::None, OutputLifecyclePurpose::L1Accumulation};
// DEST accumulation keeps one sticky DEST tile live across each outer row, then packs and
// publishes one result per row. A 1D shape has one row and therefore produces one result.
inline constexpr OutputLifecycle OutputLifecycle::DestAccumulation = {
    ReservePolicy::PerOuter, PushPolicy::PerOuter, OutputLifecyclePurpose::DestAccumulation};
// DEST accumulation with reserve/push owned by the caller. The caller provides one output
// slot per outer row.
inline constexpr OutputLifecycle OutputLifecycle::DestAccumulationCallerManaged = {
    ReservePolicy::None, PushPolicy::None, OutputLifecyclePurpose::DestAccumulation};

constexpr bool is_legal_output_lifecycle(OutputLifecycle lc) noexcept {
    return lc == OutputLifecycle::Streaming || lc == OutputLifecycle::Chunked || lc == OutputLifecycle::Bulk ||
           lc == OutputLifecycle::ReserveAllPushPerTile || lc == OutputLifecycle::ReserveAllPushPerChunk ||
           lc == OutputLifecycle::CallerManaged || lc == OutputLifecycle::ReserveNonePushEnd ||
           lc == OutputLifecycle::L1Accumulation || lc == OutputLifecycle::L1AccumulationCallerManaged ||
           lc == OutputLifecycle::DestAccumulation || lc == OutputLifecycle::DestAccumulationCallerManaged;
}

constexpr bool is_l1_accumulation_output_lifecycle(OutputLifecycle lc) noexcept {
    return lc == OutputLifecycle::L1Accumulation || lc == OutputLifecycle::L1AccumulationCallerManaged;
}

constexpr bool is_dest_accumulation_output_lifecycle(OutputLifecycle lc) noexcept {
    return lc == OutputLifecycle::DestAccumulation || lc == OutputLifecycle::DestAccumulationCallerManaged;
}

/// Which tile of an input operand to read at each step of the (Ht x Wt) walk.
/// Pick the one that matches how your input maps onto the output:
///   - Block  — a distinct tile every step; the index advances with the walk (full Ht x Wt input).
///   - Row    — indexed by column only: the same tile-row is re-read for every output row ([1, Wt] input).
///   - Col    — indexed by row only: the same tile-column is re-read for every output column ([Ht, 1] input).
///   - Scalar — always the same single tile, every step.
/// The size aspect only matters with a Bulk-style (upfront-wait) lifecycle, where the kind also sets
/// how many tiles are waited/popped upfront: Scalar 1, Row Wt, Col Ht, Block Ht x Wt.
/// The 1D tiles(n) shape allows only Block and Scalar; Row and Col need the 2D grid(H, W) shape.
/// The output is always Block, so there is no output kind.
enum class OperandKind : uint8_t {
    Block,
    Row,
    Col,
    Scalar,
};

/// Kind × InputLifecycle compatibility.
///
/// Block walks the absolute CB-front index `base_tile + i`, so it rejects PerTile-pop
/// (the front shifts each iter → absolute indexing reads the wrong tile) and PerTile-wait-
/// of-1 (never tracks a walking reader's per-iter need). Scalar/Row/Col are caller-sized
/// and reject any lifecycle whose wait/pop count grows per iter.
constexpr bool is_legal_kind_lifecycle(OperandKind kind, InputLifecycle lc) noexcept {
    if (!is_legal_input_lifecycle(lc)) {
        return false;
    }
    if (kind == OperandKind::Block) {
        // M = Ht·Wt = iter count, so growing (Cumulative) and chunked counts are safe;
        // only PerTile-pop and PerTile-wait-of-1 are excluded.
        return lc == InputLifecycle::Bulk || lc == InputLifecycle::Pipelined || lc == InputLifecycle::HeldBulk ||
               lc == InputLifecycle::HeldCumulative || lc == InputLifecycle::Chunked ||
               lc == InputLifecycle::CallerManaged || lc == InputLifecycle::DeferredPop;
    }
    // Non-Block: M < iter count, so growing (Pipelined / HeldCumulative) or chunk-scaled
    // (Chunked) counts would exceed M and deadlock.
    if (lc == InputLifecycle::Pipelined || lc == InputLifecycle::HeldCumulative || lc == InputLifecycle::Chunked) {
        return false;
    }
    if (kind == OperandKind::Scalar) {
        return true;  // M=1; every remaining lifecycle is caller-sized for one tile
    }
    // Row / Col (2D only): the window is re-read across the whole Ht·Wt walk, so only
    // persistent (non-draining) lifecycles work — PerTile-pop and HeldStream drain too early.
    return lc == InputLifecycle::Bulk || lc == InputLifecycle::HeldBulk || lc == InputLifecycle::CallerManaged ||
           lc == InputLifecycle::DeferredPop;
}

// =============================================================================
// 1d. TileOffset — orthogonal tile-index offset (present / absent)
// =============================================================================
//
// Composes with `OperandKind`: `tile_id = base + derived_from_kind(r, c)`, where
// TileOffset supplies `base` and OperandKind supplies the kind-derived term.
//   - `Unset` (default): no offset, zero overhead — the `+base` term and stored value
//     are compile-time-elided.
//   - `Set`: offset present; its value comes from the element's constructor (runtime, or
//     a compile-time constant that constant-propagates into the address add).
//
// `Set` is restricted to Bulk-family / CallerManaged lifecycles (single upfront wait,
// single end pop or none). Iter-dependent counts (Streaming / Chunked / Cumulative /
// Held{Stream,Cumulative} / NoWaitPop) can't compose with a runtime base. Caller must
// size the CB for `base + window`; the chain inflates its wait/reserve/pop/push counts
// by `base` at runtime.

enum class TileOffset : bool { Unset = false, Set = true };

// (tile_base_value(stored) — the runtime offset extractor keyed on TileOffset — is an
//  impl helper used only by element exec bodies; it lives in eltwise_chain.inl.)

/// Lifecycle compatibility check for `TileBase != None` on input elements.
/// Only InputLifecycle::Bulk-family (single upfront wait, single end pop or no pop) and
/// InputLifecycle::CallerManaged are legal — iter-dependent counts
/// (InputLifecycle::Streaming/InputLifecycle::Chunked/Cumulative) can't be expressed as `base + window`.
constexpr bool is_legal_input_lifecycle_with_base(InputLifecycle lc) noexcept {
    return lc == InputLifecycle::Bulk || lc == InputLifecycle::HeldBulk || lc == InputLifecycle::DeferredPop ||
           lc == InputLifecycle::BulkDrain || lc == InputLifecycle::CallerManaged;
}

/// Lifecycle compatibility check for `TileBase != None` on output elements.
constexpr bool is_legal_output_lifecycle_with_base(OutputLifecycle lc) noexcept {
    return lc == OutputLifecycle::Bulk || lc == OutputLifecycle::ReserveNonePushEnd ||
           lc == OutputLifecycle::CallerManaged || lc == OutputLifecycle::L1AccumulationCallerManaged ||
           lc == OutputLifecycle::DestAccumulationCallerManaged;
}

// =============================================================================
// 2. DEST slot enum — capped at compile-time DEST capacity
// =============================================================================

/// Compile-time DEST slot identifier. Cap depends on sync mode + fp32_dest_acc (DEST_AUTO_LIMIT).
/// Names D0..D15 are nominal — `static_assert` on each slot's use checks
/// `(uint32_t)Slot < DEST_AUTO_LIMIT`. Never use a literal `8` to bound DEST slots.
enum class Dst : uint32_t {
    D0 = 0,
    D1 = 1,
    D2 = 2,
    D3 = 3,
    D4 = 4,
    D5 = 5,
    D6 = 6,
    D7 = 7,
    D8 = 8,
    D9 = 9,
    D10 = 10,
    D11 = 11,
    D12 = 12,
    D13 = 13,
    D14 = 14,
    D15 = 15,
};

constexpr uint32_t to_u32(Dst s) noexcept { return static_cast<uint32_t>(s); }

// =============================================================================
// 3. Block size — `EltwiseShape::block_size` semantics
// =============================================================================
//
// Op-struct template-param enums (Approx / Legacy) live in eltwise_op_params.hpp — they
// are an op-helper concern, not part of the chain mechanics, so they are not defined here.

/// Block size. Carried by `EltwiseShape` (the `blk` arg of `EltwiseShape::tiles(n, blk)` /
/// `grid(H, W, blk)`), passed as the shape to `eltwise_chain(shape, ...)`. Each outer iter
/// processes `block_size` tiles across `block_size` DEST lanes (lane j at slot
/// dst_slot + j * chain_lane_width); `block_size == 1` is the per-tile shape.
///
/// The chain clamps `block_size` at runtime so `block_size * chain_lane_width` always fits DEST
/// (`DEST_AUTO_LIMIT`): an oversized value can't overflow DEST, it only costs extra outer
/// iterations. Streaming CB-reader chains consume one tile per iter, so block_size is clamped to 1
/// for them.

// =============================================================================
// 4. Policy enums — CB lifecycle, indexing, reconfig, broadcast
// =============================================================================

/// CB-input tile indexing — `OperandKind` as the index-mode template parameter on
/// CopyTile / BinaryFpu / PackTile.
///
/// 2D-walk tile index per kind (Scalar→0, Block→ht*Wt+wt, Row→wt, Col→ht; upfront window
/// 1 / Ht*Wt / Wt / Ht respectively). Row/Col are meaningful only with a 2D
/// `EltwiseShape::grid(Ht, Wt)` shape; there is no separate 1D overload.
/// Tile offsets are layered on via `TileBase`, not separate index modes.
///
/// Row/Col require a non-streaming policy (caller stages the whole Row/Col window — Wt or Ht
/// tiles — upfront, since it is re-read across the full walk) — same constraint as
/// `binary_op_helpers`' ROW/COL static_assert.

/// CopyTile dtype-reconfig.
///
/// `None` is load-bearing, not just a perf knob. The fold elides a reconfig only when
/// prev_cb == cur_cb in-chain; the FIRST CB-reader has no in-chain predecessor, so `Input`
/// emits an unconditional single-arg reconfig on entry — and the single-arg form does NOT
/// short-circuit on format equality (unlike the two-arg `_with_dt`). `None` asserts the
/// boot init already programmed this exact format, skipping that redundant entry reprogram
/// (canonical: a copy/identity/typecast kernel whose CBs are set once at boot). Keep it.
enum class CopyTileReconfig : uint8_t {
    None,   // no reconfig (boot init already programmed this CB's format)
    Input,  // reconfig_data_format_srca on entry (single-arg first-emit; two-arg _with_dt with prev)
};

/// FPU binary op selector.
enum class BinaryFpuOp : uint8_t { Add, Sub, Mul };

/// Whether an FPU binary result overwrites its lane-relative DEST tile or accumulates every
/// logical input into the chain's single sticky DEST tile. The enabled mode is a type property:
/// it selects a compile-time-specialized chain schedule with no per-tile mode branch.
enum class DestAccumulation : uint8_t {
    Disabled,
    Enabled,
};

/// FPU binary dtype-reconfig. Input-side only — pack-side reconfig is owned by
/// the downstream `PackTile` element (`PackTileReconfig::Output`). BinaryFpu writes
/// to DEST, never to a CB, so it has no pack-side responsibility.
///
/// `Input` is the safe default (both sides folded). `SrcA` / `SrcB` opt into a
/// single-side fold when the caller knows the *other* side is already programmed
/// (e.g. previous chain element bound that CB on the same side, or the side is
/// programmed via external init outside the chain).
enum class BinaryDataFormatReconfig : uint8_t {
    None,
    Input,  // srca and srcb on entry (default — safest, no skip)
    SrcA,   // srca only — caller asserts srcb is already programmed
    SrcB,   // srcb only — caller asserts srca is already programmed
};

/// FPU broadcast dimension. Caller MUST pass explicitly — no inference. Mirrors
/// `ckernel::BroadcastType` values (NONE=0, COL=1, ROW=2, SCALAR=3).
///
/// The dim names which axis is BROADCAST, not which was reduced. A REDUCE_ROW result is
/// column-shaped (N,1) and broadcasts back across columns via `BroadcastDim::Col`; a
/// REDUCE_COL result (1,M) uses `BroadcastDim::Row`. (E.g. softmax's per-row-max subtract
/// is `sub_tiles_bcast<BroadcastDim::Col>`, not Row.)
enum class BroadcastDim : uint8_t {
    None = 0,
    Col = 1,
    Row = 2,
    Scalar = 3,
};

/// DestReuseBinary side selector.
enum class DestReuseType : uint8_t {
    DEST_TO_SRCA,  // CB → srcb, DEST → srca
    DEST_TO_SRCB,  // CB → srca, DEST → srcb
};

/// DestReuseBinary reconfig (an enum, not a bool — the SrcA/SrcB cases need distinct values).
///
/// `Input` folds the side the CB is loaded into (driven by `ReuseType`: DEST_TO_SRCA
/// reconfigs srcb, DEST_TO_SRCB reconfigs srca). `SrcA` / `SrcB` explicitly pick a
/// side, decoupled from `ReuseType` — useful when the caller wants to assert which
/// unpack lane needs reprogramming irrespective of which lane DEST is feeding into.
enum class DestReuseReconfig : uint8_t {
    None,
    Input,  // srca-or-srcb reconfig per ReuseType
    SrcA,   // srca only — explicit, independent of ReuseType
    SrcB,   // srcb only — explicit, independent of ReuseType
};

/// UnaryBcast reconfig.
enum class UnaryBcastReconfig : uint8_t {
    None,
    Input,  // reconfig srca + srcb for the bcast input CB (no pack side — UnaryBcast never packs)
};

/// Pack-side dtype-reconfig.
///
/// The fold emits `pack_reconfig_data_format(prev_p, curr_p)` (two-arg `_with_dt`)
/// when a prior chain element established the pack target, and falls back to the
/// single-arg form on first emit. The LLK's runtime format-equality check means a
/// single `Output` cell suffices — the two-arg form is a no-op at the hardware
/// level when the formats already match.
enum class PackTileReconfig : uint8_t {
    None,
    Output,  // fold emits pack_reconfig_data_format(prev_p, curr_p) when prev_p known, else (curr_p)
};

/// How this PackTile accumulates DEST into its pinned L1 output tile.
/// `Enabled` expects that tile to be preloaded and accumulates every logical input tile.
/// `SeedFirst` overwrites it with the first logical input tile, then enables accumulation for
/// every remaining tile. In both modes the chain restores the packer to overwrite mode at exit.
enum class PackTileL1Accumulation : uint8_t {
    Disabled,
    Enabled,
    SeedFirst,
};

/// Packer-side ReLU activation applied as tiles are packed DEST -> L1 (a free clamp riding the
/// existing pack; no SFPU/MATH pass). `Zero` is plain ReLU (max(x, 0)). The packer ReLU register
/// (STACC_RELU) is a *latched* mode, so the chain programs it once before the loop and restores it
/// to pass-through at exit — exactly like L1 accumulation — so unrelated pack work after this chain
/// cannot inherit the activation. All PackTile elements in one chain must agree on the mode (mixed
/// ReLU across pack sites would need per-stage toggling and is not supported yet); ReLU also does not
/// yet compose with L1 or DEST accumulation.
///
/// Threshold modes (ReluConfig::min_threshold / max_threshold) are intentionally omitted for now —
/// no current op needs them, and their threshold must be pre-encoded in the packer output format.
enum class PackRelu : uint8_t {
    None,  // pass-through (default) — no packer ReLU
    Zero,  // plain ReLU: clamp negatives to 0 (ckernel::ReluConfig::zero())
};

// (The CRTP op bases — UnaryOp / BinaryOp / TernaryOp, from which SFPU/FPU op-helper
//  headers derive their concrete ops — are defined in eltwise_chain.inl. They are not
//  part of the kernel-author surface; only op-helper headers (eltwise_math.hpp, …)
//  inherit them, and those are parsed after eltwise_chain.hpp pulls in the .inl.)

// =============================================================================
// 7. Chain element types — declarations
//
//    Implementation lives in eltwise_chain.inl and the per-family headers.
//    Every element exposes:
//
//      init()          // per-op hw init (instance method; may read ctor args, e.g. seed/scalar)
//      exec(...)       // body per tile — (i, slot_offset) for DEST-only ops; (i_flat, ht, wt,
//                      //   slot_offset) for CB elements. slot_offset shifts the DEST lane for block>1.
//      wait_per_tile / wait_per_block / wait_upfront / wait_per_row   // CB wait  (readers, per policy)
//      pop_per_tile / pop_per_block / pop_upfront_end / pop_per_row   // CB pop   (readers, per policy)
//      reserve_per_tile / reserve_per_block / reserve_upfront         // CB reserve (writers, per policy)
//      push_per_tile / push_per_block / push_at_end                  // CB push    (writers, per policy)
//
//    Plus static-constexpr traits used by chain-shape predicates:
//      is_upfront, lane_width, etc.
// =============================================================================

template <
    uint32_t Cb,
    Dst DstSlot = Dst::D0,
    InputLifecycle Policy = InputLifecycle::Streaming,
    CopyTileReconfig Reconfig = CopyTileReconfig::Input,
    OperandKind IndexMode = OperandKind::Scalar,
    TileOffset Offset = TileOffset::Unset>
struct CopyTile;

template <
    uint32_t CbA,
    uint32_t CbB,
    BinaryFpuOp Op = BinaryFpuOp::Add,
    BroadcastDim Bcast = BroadcastDim::None,
    InputLifecycle APolicy = InputLifecycle::Streaming,
    InputLifecycle BPolicy = InputLifecycle::Streaming,
    BinaryDataFormatReconfig Reconfig = BinaryDataFormatReconfig::Input,
    Dst DstSlot = Dst::D0,
    OperandKind AIndex = OperandKind::Scalar,
    OperandKind BIndex = AIndex,
    TileOffset OffsetA = TileOffset::Unset,
    TileOffset OffsetB = TileOffset::Unset,
    DestAccumulation Accumulation = DestAccumulation::Disabled>
struct BinaryFpu;

template <
    uint32_t Cb,
    BinaryFpuOp Op,
    DestReuseType ReuseType,
    InputLifecycle Policy = InputLifecycle::Streaming,
    DestReuseReconfig Reconfig = DestReuseReconfig::Input,
    Dst DstIn = Dst::D0,
    Dst DstOut = Dst::D0,
    OperandKind IndexMode = OperandKind::Scalar,
    TileOffset Offset = TileOffset::Unset>
struct DestReuseBinary;

template <
    BroadcastDim Dim,
    uint32_t Cb,
    InputLifecycle Policy = InputLifecycle::Streaming,
    UnaryBcastReconfig Reconfig = UnaryBcastReconfig::Input,
    Dst DstSlot = Dst::D0>
struct UnaryBcast;

template <
    uint32_t Cb,
    OutputLifecycle Policy = OutputLifecycle::Streaming,
    PackTileReconfig Reconfig = PackTileReconfig::Output,
    Dst DstSlot = Dst::D0,
    TileOffset Offset = TileOffset::Unset,
    PackTileL1Accumulation L1Accumulation = PackTileL1Accumulation::Disabled,
    PackRelu Relu = PackRelu::None>
struct PackTile;

// Fill / Rand forward declarations — implementations live in eltwise_fill.hpp / eltwise_rand.hpp.
template <Dst DstSlot = Dst::D0>
struct FillScalar;
template <DataFormat DF, Dst DstSlot>
struct FillInt;
template <Dst DstSlot = Dst::D0>
struct FillBitcast;
template <Dst DstSlot = Dst::D0>
struct RandTile;

// (Chain-shape trait predicates, the EltwiseChain type-list wrapper, and the INVALID_DFB sentinel
//  are implementation detail — declared in eltwise_chain.inl, not on this public surface.)

// =============================================================================
// 9. Public API — eltwise_chain
// =============================================================================
//
// Caller-init contract (see the @file block): the caller owns engine-wide init
// (compute_kernel_hw_startup as the first statement of MAIN() for any read+write chain,
// one boot per stage for multi-stage kernels); the chain owns only per-element init.

/// Run the chain over an (Ht, Wt) tile grid with optional per-outer-iter block size.
/// `EltwiseShape` covers both walks: `tiles(n[, blk])` (1D, Ht=1) or `grid(H, W[, blk])`
/// (2D). A bare number is not accepted — write `EltwiseShape::tiles(n)` (or
/// `EltwiseShape::single()` for one tile) so the iteration shape is always explicit.
///
/// Compile-time validation static_asserts on: illegal (Policy × IndexMode) cells,
/// duplicate upfront CBs across CB-readers, colliding pack writes, and hoist requested on
/// a non-hoist-safe chain.
///
/// Index-mode (OperandKind) and block-mode behavior match the enum docs above: Block /
/// Row / Col / Scalar pick the per-iter tile index; any `is_upfront` element takes the
/// upfront-block path; Streaming chains clamp block_size to 1. Row/Col need a non-streaming
/// policy.
template <SetupOwner SO = SetupOwner::Chain, class... Es>
ALWI void eltwise_chain(EltwiseShape shape, Es... elts);

}  // namespace compute_kernel_lib

// Bring the implementation in.
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.inl"
