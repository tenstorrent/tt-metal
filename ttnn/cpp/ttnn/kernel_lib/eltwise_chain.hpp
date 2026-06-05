// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_chain.hpp
 * @brief Element-wise compute helper — one chain surface for all eltwise patterns.
 *
 * Every element-wise compute pattern (FPU binary, SFPU unary/binary/ternary, dest-reuse, copy,
 * pack, fill, rand, unary broadcast) is expressed as a sequence of chain elements passed to
 * `eltwise_chain(num_tiles, elem0, elem1, ...)`.
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
 *   eltwise_chain(num_tiles,
 *       CopyTile<dfb_in, Dst::D0, InputLifecycle::Streaming>{},
 *       Exp<>{},
 *       PackTile<dfb_out, Dst::D0, OutputLifecycle::Streaming>{});
 *
 *   // Streaming binary — A + B -> out (BinaryFpu writes DEST; the output buffer lives on PackTile)
 *   eltwise_chain(num_tiles,
 *       BinaryFpu<dfb_a, dfb_b, BinaryFpuOp::Add>{},
 *       PackTile<dfb_out, Dst::D0, OutputLifecycle::Streaming, OperandKind::Scalar,
 *                PackTileReconfig::Output>{});
 *
 * Not supported: per-iteration (mid-loop) dtype swaps — each element's dtype reconfig point is
 * resolved per element at compile time (fold-driven, emitted once at element entry), so there is
 * no per-loop-iteration reconfig path; L1 pack accumulation / pack-relu; and the legacy
 * `acquire_dst/release_dst` macros (modern dst-sync only).
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
/// axis, plain linear walk); broadcast indexing modes (`Row`/`Col`) degenerate for
/// 1D usage but remain well-defined.
///
/// Factories cover the common construction paths:
///   - `EltwiseShape::tiles(n)`           — 1D, block_size = 1
///   - `EltwiseShape::tiles(n, blk)`      — 1D + block
///   - `EltwiseShape::grid(H, W)`         — 2D, block_size = 1
///   - `EltwiseShape::grid(H, W, blk)`    — 2D + block
///
/// Implicit conversion from `uint32_t` produces `tiles(n_tiles)` so bare callers
/// (`eltwise_chain(n_tiles, ...)`) resolve without an explicit shape.
///
/// `of/row/col/single` aliases mirror `binary_op_helpers`' `BinaryInputBlockShape`.
struct EltwiseShape {
    uint32_t Ht;
    uint32_t Wt;
    uint32_t block_size;

    constexpr EltwiseShape(uint32_t H, uint32_t W, uint32_t blk = 1) : Ht(H), Wt(W), block_size(blk) {}

    // Implicit so `eltwise_chain(n_tiles, ...)` resolves via uint32_t -> EltwiseShape.
    constexpr EltwiseShape(uint32_t n_tiles) : Ht(1), Wt(n_tiles), block_size(1) {}

    static constexpr EltwiseShape tiles(uint32_t n, uint32_t blk = 1) { return {1, n, blk}; }
    static constexpr EltwiseShape grid(uint32_t H, uint32_t W, uint32_t blk = 1) { return {H, W, blk}; }

    static constexpr EltwiseShape of(uint32_t r, uint32_t c) { return {r, c, 1}; }
    static constexpr EltwiseShape row(uint32_t c) { return {1, c, 1}; }
    static constexpr EltwiseShape col(uint32_t r) { return {r, 1, 1}; }
    static constexpr EltwiseShape single() { return {1, 1, 1}; }
};

// =============================================================================
// 1c. Taxonomy: Lifecycle as a two-axis struct
// =============================================================================
//
// Each input's lifecycle is a `(WaitPolicy, PopPolicy)` pair and each output's a
// `(ReservePolicy, PushPolicy)` pair. The named constants below are the legal set:
// `is_legal_input_lifecycle` / `is_legal_output_lifecycle` are whitelists of exactly
// those constants. A custom struct literal is accepted only when it equals one of the
// named constants (e.g. `InputLifecycle{WaitPolicy::Upfront, PopPolicy::PerTile}` is just
// another spelling of `InputLifecycle::BulkDrain`); arbitrary `{wait, pop}` combinations
// outside the named set are rejected.

enum class WaitPolicy : uint8_t {
    None,        // chain emits no wait_front
    PerTile,     // wait 1 per iter
    PerChunk,    // wait K per K-iter chunk
    Upfront,     // wait M once at entry (M = kind's tile count)
    Cumulative,  // wait (i+1) per iter / chunk
};

enum class PopPolicy : uint8_t {
    None,      // chain emits no pop_front
    PerTile,   // pop 1 per iter
    PerChunk,  // pop K per K-iter chunk
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
        HeldStream, DeferredPop, NoWaitPop;
};

inline constexpr InputLifecycle InputLifecycle::Streaming = {WaitPolicy::PerTile, PopPolicy::PerTile};
inline constexpr InputLifecycle InputLifecycle::Chunked = {WaitPolicy::PerChunk, PopPolicy::PerChunk};
inline constexpr InputLifecycle InputLifecycle::Bulk = {WaitPolicy::Upfront, PopPolicy::AtEnd};
inline constexpr InputLifecycle InputLifecycle::Pipelined = {WaitPolicy::Cumulative, PopPolicy::AtEnd};
inline constexpr InputLifecycle InputLifecycle::CallerManaged = {WaitPolicy::None, PopPolicy::None};

// Bulk wait + per-tile pop: caller bulk-waits M upfront, chain drains one per iter.
inline constexpr InputLifecycle InputLifecycle::BulkDrain = {WaitPolicy::Upfront, PopPolicy::PerTile};

// Half-edge lifecycles — chain owns wait OR pop, not both; caller owns the other edge.
// Load-bearing for persistent broadcast operands (gamma, beta, mean, recip_std) that
// outlive the chain call.
inline constexpr InputLifecycle InputLifecycle::HeldBulk = {
    WaitPolicy::Upfront, PopPolicy::None};  // wait M upfront, no pop
inline constexpr InputLifecycle InputLifecycle::HeldCumulative = {
    WaitPolicy::Cumulative, PopPolicy::None};  // wait i+1 per iter, no pop
inline constexpr InputLifecycle InputLifecycle::HeldStream = {
    WaitPolicy::PerTile, PopPolicy::None};  // wait 1 per iter, no pop
inline constexpr InputLifecycle InputLifecycle::DeferredPop = {
    WaitPolicy::None, PopPolicy::AtEnd};  // caller waited, chain bulk-pops M
inline constexpr InputLifecycle InputLifecycle::NoWaitPop = {
    WaitPolicy::None, PopPolicy::PerTile};  // caller waited, chain pops per-tile

/// Validates a caller-constructed `InputLifecycle` against the legal set; every input
/// element static_asserts on it. The half-edge cells above are legal (audit-confirmed as
/// load-bearing); other half-edge combinations are rejected.
constexpr bool is_legal_input_lifecycle(InputLifecycle lc) noexcept {
    return lc == InputLifecycle::Streaming || lc == InputLifecycle::Chunked || lc == InputLifecycle::Bulk ||
           lc == InputLifecycle::Pipelined || lc == InputLifecycle::CallerManaged || lc == InputLifecycle::BulkDrain ||
           lc == InputLifecycle::HeldBulk || lc == InputLifecycle::HeldCumulative || lc == InputLifecycle::HeldStream ||
           lc == InputLifecycle::DeferredPop || lc == InputLifecycle::NoWaitPop;
}

enum class ReservePolicy : uint8_t {
    None,
    PerTile,
    PerChunk,
    Upfront,
};

enum class PushPolicy : uint8_t {
    None,
    PerTile,
    PerChunk,
    AtEnd,
};

struct OutputLifecycle {
    ReservePolicy reserve;
    PushPolicy push;

    constexpr bool operator==(OutputLifecycle other) const noexcept {
        return reserve == other.reserve && push == other.push;
    }
    constexpr bool operator!=(OutputLifecycle other) const noexcept { return !(*this == other); }

    // Named cells — written type-qualified (e.g. `OutputLifecycle::Bulk`). Defined out-of-line below.
    static const OutputLifecycle Streaming, Chunked, Bulk, BulkReservePerTile, BulkReservePerChunk, CallerManaged,
        HeldReserve, DeferredReserve;
};

inline constexpr OutputLifecycle OutputLifecycle::Streaming = {ReservePolicy::PerTile, PushPolicy::PerTile};
inline constexpr OutputLifecycle OutputLifecycle::Chunked = {ReservePolicy::PerChunk, PushPolicy::PerChunk};
inline constexpr OutputLifecycle OutputLifecycle::Bulk = {ReservePolicy::Upfront, PushPolicy::AtEnd};
// SDPA reduce_c family: bulk reserve + incremental push for downstream pipelining.
inline constexpr OutputLifecycle OutputLifecycle::BulkReservePerTile = {ReservePolicy::Upfront, PushPolicy::PerTile};
inline constexpr OutputLifecycle OutputLifecycle::BulkReservePerChunk = {ReservePolicy::Upfront, PushPolicy::PerChunk};
// L1-accumulator pack helper (tt-train compute_utils): chain emits pack_tile only,
// caller wraps the chain with its own reserve+push window. 4 catalog sites.
inline constexpr OutputLifecycle OutputLifecycle::CallerManaged = {ReservePolicy::None, PushPolicy::None};
// Chain reserves per-tile, caller pushes (rare deferred-push pattern).
inline constexpr OutputLifecycle OutputLifecycle::HeldReserve = {ReservePolicy::PerTile, PushPolicy::None};
// Caller bulk-reserved externally, chain bulk-pushes at end.
inline constexpr OutputLifecycle OutputLifecycle::DeferredReserve = {ReservePolicy::None, PushPolicy::AtEnd};

constexpr bool is_legal_output_lifecycle(OutputLifecycle lc) noexcept {
    return lc == OutputLifecycle::Streaming || lc == OutputLifecycle::Chunked || lc == OutputLifecycle::Bulk ||
           lc == OutputLifecycle::BulkReservePerTile || lc == OutputLifecycle::BulkReservePerChunk ||
           lc == OutputLifecycle::CallerManaged || lc == OutputLifecycle::HeldReserve ||
           lc == OutputLifecycle::DeferredReserve;
}

/// Per-input operand kind. The output kind is always `Block` (single column
/// in the output matrix), so no enum is defined for the output side.
///
/// Runtime/compile-time tile-index offsets are expressed by composing one of these
/// four canonical kinds with a `TileBase` (see `TileBase` types below). The kind
/// carries the iteration shape; `TileBase` carries the offset.
enum class OperandKind : uint8_t {
    Block,   // Ht × Wt — walks the full iteration domain
    Row,     // 1  × Wt — broadcast down rows
    Col,     // Ht × 1  — broadcast across cols
    Scalar,  // 1  × 1  — broadcast everywhere
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
    return lc == OutputLifecycle::Bulk || lc == OutputLifecycle::DeferredReserve ||
           lc == OutputLifecycle::HeldReserve || lc == OutputLifecycle::CallerManaged;
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
// 3. Self-documenting enums for op-struct template params
// =============================================================================

enum class Approx : bool { Exact = false, Fast = true };
enum class Legacy : bool { Off = false, On = true };

/// Block size. Runtime arg on `eltwise_chain(n_tiles, block_size, ...)`. Each outer iter
/// processes `block_size` tiles across `block_size` DEST lanes (lane j at slot
/// dst_slot + j * chain_lane_width); `block_size == 1` is the per-tile shape.
///
/// DEST footprint `block_size * chain_lane_width <= DEST_AUTO_LIMIT` is the caller's
/// responsibility — query `chain_max_block_v<Chain>` and static_assert at the call site
/// for a build-time check. Streaming CB-reader chains consume one tile per iter, so the
/// chain silently clamps block_size to 1 for them (`!chain_supports_block_v<Chain>`).

// =============================================================================
// 4. Policy enums — CB lifecycle, indexing, reconfig, broadcast
// =============================================================================

/// CB-input tile indexing — `OperandKind` as the index-mode template parameter on
/// CopyTile / BinaryFpu / PackTile.
///
/// 2D-walk tile index per kind (Scalar→0, Block→ht*Wt+wt, Row→wt, Col→ht; upfront window
/// 1 / Ht*Wt / Wt / Ht respectively). Row/Col are meaningful only in the 2D
/// `EltwiseShape{Ht, Wt}` overload — the 1D `n_tiles` overload static_asserts them out.
/// Tile offsets are layered on via `TileBase`, not separate index modes.
///
/// Row/Col require a non-streaming policy (caller stages all broadcast tiles upfront) —
/// same constraint as `binary_op_helpers`' ROW/COL static_assert.

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
    Input,  // copy_tile_to_dst_init_short_with_dt(old_cb, new_cb)
};

/// FPU binary op selector.
enum class BinaryFpuOp : uint8_t { Add, Sub, Mul };

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
    Input,  // reconfigure_unary_bcast(old_icb, new_icb, old_ocb, new_ocb)
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

// (The CRTP op bases — UnaryOp / BinaryOp / TernaryOp, from which SFPU/FPU op-helper
//  headers derive their concrete ops — are defined in eltwise_chain.inl. They are not
//  part of the kernel-author surface; only op-helper headers (eltwise_math.hpp, …)
//  inherit them, and those are parsed after eltwise_chain.hpp pulls in the .inl.)

// =============================================================================
// 7. Chain element types — declarations
//
//    Implementation lives in eltwise_chain.inl and the per-family headers.
//    Every element has the following surface:
//
//      static void init();                   // hardware init (per chain entry, or per tile)
//      static void wait_inputs(uint32_t i);  // CB wait phase (CB readers only)
//      static void exec(uint32_t i);         // body (always runs per tile)
//      static void pop_inputs(uint32_t i);   // CB pop phase  (CB readers only)
//      static void reserve_outputs(uint32_t i); // CB reserve  (CB writers only)
//      static void push_outputs(uint32_t i);    // CB push     (CB writers only)
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
    TileOffset OffsetB = TileOffset::Unset>
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
    TileOffset Offset = TileOffset::Unset>
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
/// `EltwiseShape` covers both walks: `tiles(n[, blk])` (1D, Ht=1), `grid(H, W[, blk])`
/// (2D), or a bare `uint32_t n_tiles` (implicitly `tiles(n)`).
///
/// Compile-time validation static_asserts on: illegal (Policy × IndexMode) cells,
/// duplicate upfront CBs across CB-readers, colliding pack writes, and hoist requested on
/// a non-hoist-safe chain.
///
/// Index-mode (OperandKind) and block-mode behavior match the enum docs above: Block /
/// Row / Col / Scalar pick the per-iter tile index; any `is_upfront` element takes the
/// upfront-block path; Streaming chains clamp block_size to 1. Row/Col need a non-streaming
/// policy. Query `chain_supports_block_v` / `chain_max_block_v` for a build-time block check.
template <class... Es>
ALWI void eltwise_chain(EltwiseShape shape, Es... elts);

}  // namespace compute_kernel_lib

// Bring the implementation in.
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.inl"
