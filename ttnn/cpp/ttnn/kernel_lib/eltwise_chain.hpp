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
 * so it is undefined mid-`MAIN()`; when an outer `binary_op_init_common` already covers the chain,
 * omit it.
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
 *       CopyTile<dfb_in>{},
 *       Exp<>{},
 *       PackTile<dfb_out>{});
 *
 *   // Streaming binary — A + B -> out (BinaryFpu writes DEST; the output buffer lives on PackTile)
 *   eltwise_chain(EltwiseShape::tiles(num_tiles),
 *       BinaryFpu<dfb_a, dfb_b, BinaryFpuOp::Add>{},
 *       PackTile<dfb_out>{});
 *
 * Not supported: per-iteration (mid-loop) dtype swaps — each element's dtype reconfig point is
 * resolved per element at compile time (fold-driven, emitted once at element entry), so there is
 * no per-loop-iteration reconfig path; and the legacy `acquire_dst/release_dst` macros
 * (modern dst-sync only).
 */

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "api/compute/common_globals.h"  // ALWI (used by the public eltwise_chain() declaration)
// The heavier LLK / compute-API includes are impl-only and live in eltwise_chain.inl.

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

    constexpr EltwiseShape(uint32_t H, uint32_t W, uint32_t blk = 1);

    // Explicit: bare numbers are forbidden at call sites. Use EltwiseShape::tiles(n) or
    // EltwiseShape::single() so the iteration shape is always written out.
    explicit constexpr EltwiseShape(uint32_t n_tiles);

    static constexpr EltwiseShape tiles(uint32_t n, uint32_t blk = 1);
    static constexpr EltwiseShape grid(uint32_t H, uint32_t W, uint32_t blk = 1);

    static constexpr EltwiseShape of(uint32_t r, uint32_t c);
    static constexpr EltwiseShape row(uint32_t c);
    static constexpr EltwiseShape col(uint32_t r);
    static constexpr EltwiseShape single();
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

// =============================================================================
// 1c. Input and output lifecycle
// =============================================================================

enum class WaitPolicy : uint8_t;
enum class PopPolicy : uint8_t;
enum class ReservePolicy : uint8_t;
enum class PushPolicy : uint8_t;

struct InputLifecycle {
    WaitPolicy wait_policy;
    PopPolicy pop_policy;

    constexpr bool operator==(InputLifecycle other) const noexcept;
    constexpr bool operator!=(InputLifecycle other) const noexcept;

    // Wait and pop one tile per iteration.
    static const InputLifecycle Streaming;
    // Wait and pop one block_size-tile chunk per iteration.
    static const InputLifecycle Chunked;
    // Wait for the full operand window upfront and pop it at chain exit.
    static const InputLifecycle Bulk;
    // Wait for a growing window per iteration and pop the full window at chain exit.
    static const InputLifecycle Pipelined;
    // The caller owns both wait and pop.
    static const InputLifecycle CallerManaged;
    // Wait for the full walk upfront and pop one tile per iteration.
    static const InputLifecycle BulkDrain;
    // Wait for the full operand window upfront and do not pop it.
    static const InputLifecycle HeldBulk;
    // Wait for a growing window per iteration and do not pop it.
    static const InputLifecycle HeldCumulative;
    // Wait for one tile per iteration and do not pop it.
    static const InputLifecycle HeldStream;
    // Do not wait; pop the full operand window at chain exit.
    static const InputLifecycle DeferredPop;
    // Do not wait; pop one tile per iteration.
    static const InputLifecycle NoWaitPop;
    // Wait and pop one tile per outer row.
    static const InputLifecycle OuterStream;
};

struct OutputLifecycle {
    ReservePolicy reserve_policy;
    PushPolicy push_policy;

    constexpr bool operator==(OutputLifecycle other) const noexcept;
    constexpr bool operator!=(OutputLifecycle other) const noexcept;

    // Reserve and push one tile per iteration.
    static const OutputLifecycle Streaming;
    // Reserve and push one block_size-tile chunk per iteration.
    static const OutputLifecycle Chunked;
    // Reserve the full output window upfront and push it at chain exit.
    static const OutputLifecycle Bulk;
    // Reserve the full output window upfront and push one tile per iteration.
    static const OutputLifecycle ReserveAllPushPerTile;
    // Reserve the full output window upfront and push one chunk per iteration.
    static const OutputLifecycle ReserveAllPushPerChunk;
    // The caller owns both reserve and push.
    static const OutputLifecycle CallerManaged;
    // Do not reserve; push the full output window at chain exit.
    static const OutputLifecycle ReserveNonePushEnd;
    // Reserve one persistent accumulator tile at entry and push it at exit.
    static const OutputLifecycle L1Accumulation;
    // Reserve and push one reduced tile per outer row.
    static const OutputLifecycle DestAccumulation;
};

/// Which tile of an input operand to read at each step of the (Ht x Wt) walk.
/// Pick the one that matches how your input maps onto the output:
///   - Block  — a distinct tile every step; the index advances with the walk (full Ht x Wt input).
///   - Row    — indexed by column only: the same tile-row is re-read for every output row ([1, Wt] input).
///   - Col    — indexed by row only: the same tile-column is re-read for every output column ([Ht, 1] input).
///   - Scalar — contribute index 0 every step, pinning the read to the operand's base tile
///              (`TileOffset::Set` may make that base nonzero). This is inter-tile indexing, not a
///              hardware scalar broadcast; it is independent of `BroadcastDim`.
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

/// Whether the chain updates the data format for an operand.
enum class DataFormatReconfig : bool { Disabled = false, Enabled = true };

/// Whether a pack adds DEST to the output tile already in L1.
enum class L1Accumulation : uint8_t {
    Disabled,
    Enabled,
    SeedFirst,
};

/// Whether an FPU binary accumulates into a persistent DEST tile.
enum class DestAccumulation : bool { Disabled = false, Enabled = true };

/// ReLU applied by the packer before writing an output tile.
enum class PackRelu : bool { Disabled = false, Zero = true };

// =============================================================================
// 1e. Grouped operand configuration
// =============================================================================
//
// `input(...)` and `output(...)` group the compile-time properties of one operand.

struct InputSpec {
    InputLifecycle lifecycle;
    OperandKind index;
    DataFormatReconfig reconfig;
    TileOffset offset;
};

struct OutputSpec {
    OutputLifecycle lifecycle;
    DataFormatReconfig reconfig;
    PackRelu relu;
    L1Accumulation l1_accumulation;
    DestAccumulation dest_accumulation;
    TileOffset offset;
};

/// Group one input operand's configuration.
/// Defaults: Streaming lifecycle, Scalar indexing, reconfig enabled, and no tile offset.
constexpr InputSpec input(
    InputLifecycle lifecycle = InputLifecycle::Streaming,
    OperandKind index = OperandKind::Scalar,
    DataFormatReconfig reconfig = DataFormatReconfig::Enabled,
    TileOffset offset = TileOffset::Unset) noexcept;
constexpr InputSpec input(InputLifecycle lifecycle, DataFormatReconfig reconfig) noexcept;

/// Group one output operand's configuration.
/// Defaults: Streaming lifecycle, reconfig enabled, no accumulation, no pack ReLU,
/// and no tile offset.
constexpr OutputSpec output(
    OutputLifecycle lifecycle = OutputLifecycle::Streaming,
    DataFormatReconfig reconfig = DataFormatReconfig::Enabled,
    PackRelu relu = PackRelu::Disabled,
    L1Accumulation l1_accumulation = L1Accumulation::Disabled,
    DestAccumulation dest_accumulation = DestAccumulation::Disabled,
    TileOffset offset = TileOffset::Unset) noexcept;

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

constexpr uint32_t to_u32(Dst s) noexcept;

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
// 4. Operation selectors
// =============================================================================

/// FPU binary op selector.
enum class BinaryFpuOp : uint8_t { Add, Sub, Mul };

/// FPU broadcast dimension. Caller MUST pass explicitly — no inference. Mirrors
/// `ckernel::BroadcastType` values (NONE=0, COL=1, ROW=2, SCALAR=3).
///
/// `BinaryFpu<CbA, CbB, ...>` always applies this intra-tile broadcast to operand B (`CbB`);
/// operand A is never the broadcast source. This is independent of each operand's `OperandKind`,
/// which only selects the tile index read during the (Ht x Wt) walk.
///
/// The dim names which axis is BROADCAST, not which was reduced. A REDUCE_ROW result is
/// column-shaped (N,1) and broadcasts back across columns via `BroadcastDim::Col`; a
/// REDUCE_COL result (1,M) uses `BroadcastDim::Row`.
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

// =============================================================================
// 5. Chain element declarations
// =============================================================================

namespace detail {

constexpr uint32_t copy_tile_config_bits(Dst dst, InputSpec input_spec) noexcept;

constexpr uint32_t pack_tile_config_bits(OutputSpec output_spec, Dst dst) noexcept;

constexpr uint32_t binary_fpu_config_bits(
    BinaryFpuOp op, BroadcastDim bcast, InputSpec a, InputSpec b, Dst dst, DestAccumulation accumulation) noexcept;

constexpr uint32_t dest_reuse_binary_config_bits(
    BinaryFpuOp op, DestReuseType reuse, InputSpec input_spec, Dst dst_in, Dst dst_out) noexcept;

template <uint32_t Cb, uint32_t ConfigBits>
struct CopyTileImpl;
template <uint32_t Cb, uint32_t ConfigBits>
struct PackTileImpl;
template <uint32_t CbA, uint32_t CbB, uint32_t ConfigBits>
struct BinaryFpuImpl;
template <uint32_t Cb, uint32_t ConfigBits>
struct DestReuseBinaryImpl;

}  // namespace detail

template <uint32_t Cb, Dst DstSlot = Dst::D0, InputSpec Input = input()>
using CopyTile = detail::CopyTileImpl<Cb, detail::copy_tile_config_bits(DstSlot, Input)>;

template <
    uint32_t CbA,
    uint32_t CbB,
    BinaryFpuOp Op = BinaryFpuOp::Add,
    BroadcastDim Bcast = BroadcastDim::None,
    InputSpec AInput = input(),
    InputSpec BInput = input(),
    Dst DstSlot = Dst::D0,
    DestAccumulation Accumulation = DestAccumulation::Disabled>
using BinaryFpu =
    detail::BinaryFpuImpl<CbA, CbB, detail::binary_fpu_config_bits(Op, Bcast, AInput, BInput, DstSlot, Accumulation)>;

template <
    uint32_t Cb,
    BinaryFpuOp Op,
    DestReuseType ReuseType,
    InputSpec Input = input(),
    Dst DstIn = Dst::D0,
    Dst DstOut = Dst::D0>
using DestReuseBinary =
    detail::DestReuseBinaryImpl<Cb, detail::dest_reuse_binary_config_bits(Op, ReuseType, Input, DstIn, DstOut)>;

template <uint32_t Cb, OutputSpec Output = output(), Dst DstSlot = Dst::D0>
using PackTile = detail::PackTileImpl<Cb, detail::pack_tile_config_bits(Output, DstSlot)>;

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
