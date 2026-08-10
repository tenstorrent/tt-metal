// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file chain.hpp
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
 * (plus `compute_kernel_hw_startup` / `mm_init` / `reduce_init` when the kernel mixes those
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
 * so it is undefined mid-`MAIN()`; when an outer `compute_kernel_hw_startup` already covers the chain,
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
 *       CopyTile<input(dfb_in)>{},
 *       Exp<>{},
 *       PackTile<output(dfb_out)>{});
 *
 *   // Streaming binary — A + B -> out (BinaryFpu writes DEST; the output buffer lives on PackTile)
 *   eltwise_chain(EltwiseShape::tiles(num_tiles),
 *       BinaryFpu<input(dfb_a), input(dfb_b), BinaryFpuOp::Add>{},
 *       PackTile<output(dfb_out)>{});
 *
 * Not supported: per-iteration (mid-loop) dtype swaps — each element's dtype reconfig point is
 * resolved per element at compile time (fold-driven, emitted once at element entry), so there is
 * no per-loop-iteration reconfig path; and the legacy `acquire_dst/release_dst` macros
 * (modern dst-sync only).
 */

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "api/compute/common_globals.h"  // ALWI (used by the public eltwise_chain() declaration)
// The heavier LLK / compute-API includes are impl-only and live in chain.inl.

namespace compute_kernel_lib {

// Buffer identities throughout the chain (the `cb_id` fields on InputSpec / OutputSpec,
// `dfb`-named implementation accessors and ElemDesc fields, and the INVALID_DFB /
// NO_PREV_DFB sentinels) are dataflow-buffer ids: today an integer `tt::CBIndex` value.
// The public element aliases accept complete input(...) / output(...) specs and forward each
// id as a separate implementation NTTP so it does not consume packed configuration bits.

// (The marker-tag hierarchy — CbReaderTag/CbWriterTag/DestOnlyTag + the per-element
//  leaf tags — and the is_*_op_v classification predicates are internal pipeline
//  scaffolding, defined in chain.inl. Concrete elements declared below inherit
//  the leaf tags from there.)

// =============================================================================
// 1a. 2D shape — (Ht, Wt) tile grid for the 2D chain overload
// =============================================================================

/// How a `PerBlockSize` lifecycle synchronizes the partial tail of a blocked walk.
///
/// Math and pack always execute only valid tiles. This policy controls only the CB lifecycle
/// count (wait/pop/reserve/push) of the partial final block.
enum class BlockTailSync : uint8_t {
    ValidTiles,  // synchronize min(block_size, tiles remaining)
    FullBlock,   // synchronize block_size even when fewer tiles are mathematically valid
};

/// Compile-time dimensional intent of an elementwise walk.
///
/// The tag is carried by the factory return type rather than stored in the runtime shape, so
/// APIs can reject dimensionally meaningless combinations without adding a device-side field.
enum class EltwiseShapeKind : uint8_t {
    Tiles,  // one contiguous 1D tile sequence
    Grid,   // a 2D row/column walk, including grid(1, W)
};

template <EltwiseShapeKind Kind>
struct TypedEltwiseShape;

/// Iteration shape for `eltwise_chain`. Carries the tile grid (Ht × Wt, both in tiles), the
/// per-outer-iter `block_size`, and the tail synchronization policy. Ht=1 expresses the 1D case
/// (no row axis, plain linear walk); the `Row`/`Col` indexing modes degenerate for 1D usage but
/// remain well-defined.
///
/// Factories cover the common construction paths:
///   - `EltwiseShape::tiles(n)`           — 1D, block_size = 1
///   - `EltwiseShape::tiles(n, blk)`      — 1D + block
///   - `EltwiseShape::tiles(n, blk, BlockTailSync::FullBlock)`
///                                       — 1D fixed-size physical blocks
///   - `EltwiseShape::grid(H, W)`         — 2D, block_size = 1
///   - `EltwiseShape::grid(H, W, blk)`    — 2D + block
///   - `EltwiseShape::grid(H, W, blk, BlockTailSync::FullBlock)`
///                                       — 2D row-blocked fixed-size physical blocks
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
    BlockTailSync tail_sync;

    constexpr EltwiseShape(
        uint32_t H, uint32_t W, uint32_t blk = 1, BlockTailSync tail_sync = BlockTailSync::ValidTiles);

    // Explicit: bare numbers are forbidden at call sites. Use EltwiseShape::tiles(n) or
    // EltwiseShape::single() so the iteration shape is always written out.
    explicit constexpr EltwiseShape(uint32_t n_tiles);

    static constexpr TypedEltwiseShape<EltwiseShapeKind::Tiles> tiles(
        uint32_t n, uint32_t blk = 1, BlockTailSync tail_sync = BlockTailSync::ValidTiles);
    static constexpr TypedEltwiseShape<EltwiseShapeKind::Grid> grid(
        uint32_t H, uint32_t W, uint32_t blk = 1, BlockTailSync tail_sync = BlockTailSync::ValidTiles);

    static constexpr TypedEltwiseShape<EltwiseShapeKind::Grid> of(uint32_t r, uint32_t c);
    static constexpr TypedEltwiseShape<EltwiseShapeKind::Grid> row(uint32_t c);
    static constexpr TypedEltwiseShape<EltwiseShapeKind::Grid> col(uint32_t r);
    static constexpr TypedEltwiseShape<EltwiseShapeKind::Tiles> single();
};

/// Zero-overhead factory tag around the common runtime shape payload.
///
/// `eltwise_chain_impl` still accepts the untyped base, so Kind affects only the thin public
/// validation wrapper and does not multiply the core walk implementation.
template <EltwiseShapeKind Kind>
struct TypedEltwiseShape : EltwiseShape {
    static constexpr EltwiseShapeKind kind = Kind;

    constexpr TypedEltwiseShape(
        uint32_t H, uint32_t W, uint32_t blk = 1, BlockTailSync tail_sync = BlockTailSync::ValidTiles) :
        EltwiseShape(H, W, blk, tail_sync) {}
};

/// Who performs the chain's one-time setup — init + reconfig — the leading template arg to
/// `eltwise_chain`. This is about *ownership*, NOT about whether inits are hoistable: which inits
/// are hoistable is deduced from the chain's uniformity and is never a manual choice.
///
///   eltwise_chain(shape, elts...);                       // default: InitReconfigOwner::Chain
///   // To hoist the setup out of your own loop: emit it ONCE before the loop yourself (e.g. the
///   // original raw *_init call), then hand ownership to the caller so the chain skips it:
///   <emit the chain's one-time setup once, before the loop>
///   for (...) eltwise_chain<InitReconfigOwner::Caller>(EltwiseShape::single(), elts...);
///
/// InitReconfigOwner::Caller is only valid when the chain's entire setup is boot-hoistable (uniform math
/// MOP + SFPU init AND homogeneous pack CBs) — i.e. there's a single "once, before the loop" the
/// caller can own. eltwise_chain static_asserts this; a chain that must re-emit setup per tile
/// (so the caller can't pre-do it once) is a compile error pointing you back to InitReconfigOwner::Chain.
enum class InitReconfigOwner {
    Chain,   // this eltwise_chain call emits the one-time setup (init + reconfig)
    Caller,  // the caller emitted it once, outside the loop — the chain emits none of it here
};

// -----------------------------------------------------------------------------
// Skip-compute — a performance-debugging BUILD knob, NOT part of the eltwise_chain API.
//
// With CKL_ELTWISE_CHAIN_SKIP_COMPUTE=1, every eltwise_chain in the translation unit emits only
// its CB lifecycle (wait/pop/reserve/push) and tile_regs synchronization. All helper-owned init,
// reconfiguration, compute, and pack execution is compile-time-elided. CB counts are unchanged, so
// the dataflow handshake remains intact, but the published output is intentionally garbage.
//
// Use this only for local run-versus-skip profiling to separate compute cost from the
// CB/data-movement floor. Do not use it in production or correctness runs.
//
// Opt in either before including this header or through the kernel's compiler defines:
//
//   #define CKL_ELTWISE_CHAIN_SKIP_COMPUTE 1
//   #include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
//
// The knob covers ordinary, L1-accumulation, and DEST-accumulation walks. It does not suppress
// caller-owned work outside eltwise_chain, including compute_kernel_hw_startup or setup emitted for
// InitReconfigOwner::Caller.
#ifndef CKL_ELTWISE_CHAIN_SKIP_COMPUTE
#define CKL_ELTWISE_CHAIN_SKIP_COMPUTE 0
#endif

// =============================================================================
// 1b. Input and output CB synchronization policies
// =============================================================================

/// `PerBlockSize` synchronizes a CB once per `EltwiseShape::block_size` group. It is unrelated to
/// `OperandKind::Block`, which controls how an input tile index maps onto the output shape. On a
/// partial final group, `BlockTailSync` selects whether the synchronized count is the valid
/// remainder or the full `block_size`.

/// When the chain waits for an input CB.
enum class WaitPolicy : uint8_t { None, PerTile, PerBlockSize, PerOuter, Upfront, Cumulative };

/// When the chain pops an input CB.
enum class PopPolicy : uint8_t { None, PerTile, PerBlockSize, PerOuter, AtEnd };

/// When the chain reserves an output CB.
enum class ReservePolicy : uint8_t { None, PerTile, PerBlockSize, Upfront, PerOuter, OneUpfront };

/// When the chain pushes an output CB.
enum class PushPolicy : uint8_t { None, PerTile, PerBlockSize, AtEnd, PerOuter, OneAtEnd };

/// Which tile of an input operand to read at each step of the (Ht x Wt) walk.
/// Pick the one that matches how your input maps onto the output:
///   - Block  — a distinct tile every step; the index advances with the walk (full Ht x Wt input).
///   - Row    — indexed by column only: the same tile-row is re-read for every output row ([1, Wt] input).
///   - Col    — indexed by row only: the same tile-column is re-read for every output column ([Ht, 1] input).
///   - Scalar — contribute index 0 every step, pinning the read to the operand's base tile
///              (`TileOffset::Set` may make that base nonzero). This is inter-tile indexing, not a
///              hardware scalar broadcast; it is independent of `BroadcastDim`.
/// The size aspect only matters with an upfront-wait policy, where the kind also sets
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
// 1c. TileOffset — orthogonal tile-index addressing
// =============================================================================
//
// Composes with `OperandKind`: `tile_id = base + derived_from_kind(r, c)`, where
// TileOffset supplies `base` and OperandKind supplies the kind-derived term.
//   - `Unset` (default): no offset, zero overhead — the `+base` term and stored value
//     are compile-time-elided.
//   - `Set`: offset present; its value comes from the element's constructor (runtime, or
//     a compile-time constant that constant-propagates into the address add).
//   - `Strided`: base and row stride come from a `StridedTileRange`. Block maps to
//     `base + r * row_stride + c`, Col to `base + r * row_stride`, while Row and
//     Scalar retain their ordinary column/pinned behavior.
//
// `Set` is restricted to upfront/deferred-pop or caller-managed wait/pop pairs.
// Iter-dependent wait/pop counts can't compose with a runtime base. Caller must
// size the CB for `base + window`; the chain inflates its wait/reserve/pop/push counts
// by `base` at runtime.
//
// `Strided` is restricted to caller-managed `(None, None)` policies: a gapped window cannot be
// represented by a single wait/pop/reserve/push count, so the enclosing kernel owns it.

enum class TileOffset : uint8_t { Unset, Set, Strided };

struct StridedTileRange {
    uint32_t base;
    uint32_t row_stride;
};

/// Whether the chain updates the data format for an operand.
enum class DataFormatReconfig : bool { Disabled = false, Enabled = true };

/// Whether a pack accumulates into one output tile in L1.
enum class L1Accumulation : uint8_t {
    Disabled,
    Enabled,        // the first pack seeds the output; subsequent packs accumulate
    AddToExisting,  // every pack, including the first, adds to the existing output
};

/// Scope of an FPU binary's persistent DEST accumulation.
///
/// PerRow acquires, packs, and clears DEST once per row of a 2D grid. WholeShape keeps DEST
/// acquired across the complete shape and emits one tile. `tiles(...)` is intrinsically one
/// contiguous shape, so PerRow is rejected at compile time; spell WholeShape for a 1D reduction.
enum class DestAccumulation : uint8_t {
    Disabled,
    PerRow,
    WholeShape,
};

/// ReLU applied by the packer before writing an output tile.
enum class PackRelu : bool { Disabled = false, Zero = true };

// =============================================================================
// 1d. Grouped operand configuration
// =============================================================================
//
// `input(...)` and `output(...)` bind a buffer id to the compile-time properties of one operand.

struct InputSpec {
    uint32_t cb_id;
    WaitPolicy wait;
    PopPolicy pop;
    OperandKind index;
    DataFormatReconfig reconfig;
    TileOffset offset;
};

struct OutputSpec {
    uint32_t cb_id;
    ReservePolicy reserve;
    PushPolicy push;
    DataFormatReconfig reconfig;
    PackRelu relu;
    L1Accumulation l1_accumulation;
    DestAccumulation dest_accumulation;
    TileOffset offset;
};

/// Bind one input buffer id to its configuration.
/// Defaults: wait/pop per tile, Scalar indexing, reconfig enabled, and no tile offset.
constexpr InputSpec input(
    uint32_t cb_id,
    WaitPolicy wait = WaitPolicy::PerTile,
    PopPolicy pop = PopPolicy::PerTile,
    OperandKind index = OperandKind::Scalar,
    DataFormatReconfig reconfig = DataFormatReconfig::Enabled,
    TileOffset offset = TileOffset::Unset) noexcept;
constexpr InputSpec input(uint32_t cb_id, WaitPolicy wait, PopPolicy pop, DataFormatReconfig reconfig) noexcept;
constexpr InputSpec input(
    uint32_t cb_id, WaitPolicy wait, PopPolicy pop, OperandKind index, TileOffset offset) noexcept;

/// Bind one output buffer id to its configuration.
/// Defaults: reserve/push per tile, reconfig enabled, no accumulation, no pack ReLU,
/// and no tile offset.
constexpr OutputSpec output(
    uint32_t cb_id,
    ReservePolicy reserve = ReservePolicy::PerTile,
    PushPolicy push = PushPolicy::PerTile,
    DataFormatReconfig reconfig = DataFormatReconfig::Enabled,
    PackRelu relu = PackRelu::Disabled,
    L1Accumulation l1_accumulation = L1Accumulation::Disabled,
    DestAccumulation dest_accumulation = DestAccumulation::Disabled,
    TileOffset offset = TileOffset::Unset) noexcept;
constexpr OutputSpec output(uint32_t cb_id, ReservePolicy reserve, PushPolicy push, TileOffset offset) noexcept;

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
// Op-struct template-param enums (Approx / Legacy) live in op_params.hpp — they
// are an op-helper concern, not part of the chain mechanics, so they are not defined here.

/// Block size. Carried by `EltwiseShape` (the `blk` arg of `EltwiseShape::tiles(n, blk)` /
/// `grid(H, W, blk)`), passed as the shape to `eltwise_chain(shape, ...)`. Each full outer iter
/// processes `block_size` tiles across `block_size` DEST lanes (lane j at slot
/// dst_slot + j * chain_lane_width); `block_size == 1` is the per-tile shape. A partial final
/// iter always executes only its valid remainder. `BlockTailSync` selects whether `PerBlockSize`
/// lifecycles synchronize that valid remainder or the full `block_size`.
///
/// For numeric shapes, the chain clamps `block_size` at runtime so
/// `block_size * chain_lane_width` always fits DEST (`DEST_AUTO_LIMIT`): an oversized value can't
/// overflow DEST, it only costs extra outer iterations. Streaming CB-reader chains consume one
/// tile per iter, so block_size is clamped to 1 for them. A shape using `FullBlock` mode instead
/// describes a physical CB contract and must already fit; the chain
/// asserts rather than changing it.

// =============================================================================
// 4. Operation selectors
// =============================================================================

/// FPU binary op selector.
enum class BinaryFpuOp : uint8_t { Add, Sub, Mul };

/// FPU broadcast dimension. Caller MUST pass explicitly — no inference. Mirrors
/// `ckernel::BroadcastType` values (NONE=0, COL=1, ROW=2, SCALAR=3).
///
/// `BinaryFpu<input(CbA), input(CbB), ...>` always applies this intra-tile broadcast to operand B (`CbB`);
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
    BinaryFpuOp op, DestReuseType reuse, InputSpec input_spec, Dst dst) noexcept;

template <uint32_t Cb, uint32_t ConfigBits>
struct CopyTileImpl;
template <uint32_t Cb, uint32_t ConfigBits>
struct PackTileImpl;
template <uint32_t CbA, uint32_t CbB, uint32_t ConfigBits>
struct BinaryFpuImpl;
template <uint32_t Cb, uint32_t ConfigBits>
struct DestReuseBinaryImpl;

}  // namespace detail

template <InputSpec Input, Dst DstSlot = Dst::D0>
using CopyTile = detail::CopyTileImpl<Input.cb_id, detail::copy_tile_config_bits(DstSlot, Input)>;

template <
    InputSpec AInput,
    InputSpec BInput,
    BinaryFpuOp Op = BinaryFpuOp::Add,
    BroadcastDim Bcast = BroadcastDim::None,
    Dst DstSlot = Dst::D0,
    DestAccumulation Accumulation = DestAccumulation::Disabled>
using BinaryFpu = detail::BinaryFpuImpl<
    AInput.cb_id,
    BInput.cb_id,
    detail::binary_fpu_config_bits(Op, Bcast, AInput, BInput, DstSlot, Accumulation)>;

/// Apply an FPU binary operation between one CB input and `DstSlot`.
/// The LLK operation is in-place in DEST: it reads and overwrites the same slot.
template <InputSpec Input, BinaryFpuOp Op, DestReuseType ReuseType, Dst DstSlot = Dst::D0>
using DestReuseBinary =
    detail::DestReuseBinaryImpl<Input.cb_id, detail::dest_reuse_binary_config_bits(Op, ReuseType, Input, DstSlot)>;

template <OutputSpec Output, Dst DstSlot = Dst::D0>
using PackTile = detail::PackTileImpl<Output.cb_id, detail::pack_tile_config_bits(Output, DstSlot)>;

// (Chain-shape trait predicates, the EltwiseChain type-list wrapper, and the INVALID_DFB sentinel
//  are implementation detail — declared in chain.inl, not on this public surface.)

// =============================================================================
// 6. Public API — eltwise_chain
// =============================================================================
//
// Caller-init contract (see the @file block): the caller owns engine-wide init
// (compute_kernel_hw_startup as the first statement of MAIN() for any read+write chain,
// one boot per stage for multi-stage kernels); the chain owns only per-element init.

/// Run the chain over an (Ht, Wt) tile grid with optional per-outer-iter block size.
/// `EltwiseShape` covers both walks: `tiles(n[, blk])` (1D, Ht=1),
/// `tiles(n, blk, BlockTailSync::FullBlock)` for a fixed-block 1D CB contract, or
/// `grid(H, W, blk, BlockTailSync::FullBlock)` for a row-blocked 2D contract.
/// A bare number is not accepted — write `EltwiseShape::tiles(n)` (or
/// `EltwiseShape::single()` for one tile) so the iteration shape is always explicit.
///
/// Compile-time validation static_asserts on: illegal (Policy × IndexMode) cells,
/// duplicate upfront CBs across CB-readers, colliding pack writes, and hoist requested on
/// a non-hoist-safe chain.
///
/// Index-mode (OperandKind) and block-mode behavior match the enum docs above: Block /
/// Row / Col / Scalar pick the per-iter tile index; input policies that own a staged CB
/// window take the upfront-block path; Streaming chains clamp block_size to 1.
/// `BlockTailSync` affects only per-block-size synchronization counts. Row/Col need a non-streaming policy.
template <InitReconfigOwner Owner = InitReconfigOwner::Chain, EltwiseShapeKind Kind, class... Es>
ALWI void eltwise_chain(TypedEltwiseShape<Kind> shape, Es... elts);

}  // namespace compute_kernel_lib

// Bring the implementation in.
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.inl"
