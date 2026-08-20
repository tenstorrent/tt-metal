// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file chain.hpp
 * @brief Element-wise compute helper — one chain surface for all eltwise patterns.
 *
 * Every element-wise compute pattern (FPU binary, SFPU unary/binary/ternary, dest-reuse, copy,
 * pack, fill, rand, unary broadcast) is expressed as a sequence of chain elements passed to
 * `eltwise_chain(shape, elem0, elem1, ...)` (shape is an `IterationShape`, e.g.
 * `IterationShape::tiles(num_tiles)`).
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
 *   eltwise_chain(IterationShape::tiles(num_tiles),
 *       CopyTile<input(dfb_in)>{},
 *       Exp<>{},
 *       PackTile<output(dfb_out)>{});
 *
 *   // Streaming binary — A + B -> out (BinaryFpu writes DEST; the output buffer lives on PackTile)
 *   eltwise_chain(IterationShape::tiles(num_tiles),
 *       BinaryFpu<BinaryFpuOp::Add, input(dfb_a), input(dfb_b)>{},
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

/// Looping shape for `eltwise_chain`. `Ht`, `Wt`, and `block_size` describe how the chain driver
/// iterates and groups its work; they do not necessarily equal the number of tiles present on any
/// input or output. Operand indexing and lifecycle policies may pin, reuse, accumulate, or defer
/// tiles independently of this iteration space. Ht=1 expresses the 1D case (no row axis, plain
/// linear walk); the `Row`/`Col` indexing modes degenerate for 1D usage but remain well-defined.
///
/// Factories establish the iteration extent. Blocking is configured fluently when needed:
///   - `IterationShape::tiles(n)` — 1D, block size 1
///   - `IterationShape::tiles(n).block_size(blk)` — 1D + block
///   - `IterationShape::tiles(n).block_size(blk, BlockTailSync::FullBlock)`
///                                                   — 1D fixed-size physical blocks
///   - `IterationShape::grid(H, W)` — 2D, block size 1
///   - `IterationShape::grid(H, W).block_size(blk)` — 2D + block
///   - `IterationShape::grid(H, W).block_size(blk, BlockTailSync::FullBlock)`
///                                                   — 2D row-blocked fixed-size physical blocks
///
/// Construction from a tile count is `explicit`: a bare number is NOT accepted as a
/// shape — call sites must spell the iteration shape out as `IterationShape::tiles(n)`
/// (or `IterationShape::one_tile()` for one tile). This keeps `eltwise_chain(...)` and the
/// convenience wrappers from silently treating a stray integer as a tile count.
///
/// `row/col/one_tile` aliases mirror `binary_op_helpers`' `BinaryInputBlockShape`.
struct IterationShape {
    uint32_t Ht;
    uint32_t Wt;
    uint32_t block_tiles;  // tiles per inner block iteration; named apart from the fluent method
    BlockTailSync tail_sync;

    constexpr IterationShape(uint32_t H, uint32_t W) :
        Ht(H), Wt(W), block_tiles(1), tail_sync(BlockTailSync::ValidTiles) {}

    // Explicit: bare numbers are forbidden at call sites. Use IterationShape::tiles(n) or
    // IterationShape::one_tile() so the iteration shape is always written out.
    explicit constexpr IterationShape(uint32_t n_tiles) : IterationShape(1, n_tiles) {}

    static constexpr IterationShape tiles(uint32_t n) { return {1, n}; }
    static constexpr IterationShape grid(uint32_t H, uint32_t W) { return {H, W}; }

    static constexpr IterationShape row(uint32_t c) { return {1, c}; }
    static constexpr IterationShape col(uint32_t r) { return {r, 1}; }
    static constexpr IterationShape one_tile() { return {1, 1}; }

    /// Configures blocking and returns the shape, so it chains on factory temporaries:
    /// `IterationShape::tiles(n).block_size(blk)`.
    constexpr IterationShape block_size(uint32_t value, BlockTailSync tail = BlockTailSync::ValidTiles) {
        block_tiles = value;
        tail_sync = tail;
        return *this;
    }
};

/// Who performs the chain's one-time setup — init + reconfig — the leading template arg to
/// `eltwise_chain`. This is about *ownership*, NOT about whether inits are hoistable: which inits
/// are hoistable is deduced from the chain's uniformity and is never a manual choice.
///
///   eltwise_chain(shape, elts...);                       // default: InitReconfigOwner::Chain
///   // To hoist the setup out of your own loop: emit it ONCE before the loop yourself (e.g. the
///   // original raw *_init call), then hand ownership to the caller so the chain skips it:
///   <emit the chain's one-time setup once, before the loop>
///   for (...) eltwise_chain<InitReconfigOwner::Caller>(IterationShape::one_tile(), elts...);
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
//   #include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
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

/// Lifecycle policies control FIFO synchronization together with `InputTileMapping`, which controls tile
/// indexing. `PerBlockSize` synchronizes once per `IterationShape::block_size` group. For an input,
/// `PerTile + Col` denotes a streamed column: the chain waits for one tile at each grid-row boundary,
/// reuses the current front tile across that row, then pops it. `Upfront + Col` instead stages an
/// Ht-tile window and indexes it by row. Output `PerTile` remains literal: one output tile is reserved
/// and pushed per grid cell. `PerOuter` exists only for output DEST-row accumulation.
/// On a partial final block, `BlockTailSync` selects whether synchronization covers the valid
/// remainder or the full `block_size`.

/// When the chain waits for an input CB.
enum class WaitPolicy : uint8_t { None, PerTile, PerBlockSize, Upfront, Cumulative };

/// When the chain pops an input CB.
enum class PopPolicy : uint8_t { None, PerTile, PerBlockSize, AtEnd };

/// When the chain reserves an output CB.
enum class ReservePolicy : uint8_t { None, PerTile, PerBlockSize, Upfront, PerOuter, OneUpfront };

/// When the chain pushes an output CB.
enum class PushPolicy : uint8_t { None, PerTile, PerBlockSize, AtEnd, PerOuter, OneAtEnd };

/// Which tile of an input operand to read at each step of the (Ht x Wt) walk.
/// Pick the one that matches how your input maps onto the output:
///   - Block  — a distinct tile every step; the index advances with the walk (full Ht x Wt input).
///   - Row    — indexed by column only: the same tile-row is re-read for every output row ([1, Wt] input).
///   - Col    — with a staged lifecycle, indexed by row so the same tile-column is re-read for every
///              output column ([Ht, 1] input); with PerTile/PerTile, streamed one tile per row.
///   - Scalar — contribute index 0 every step, pinning the read to the operand's base tile
///              (`TileAddressing::Offset` may make that base nonzero). This is inter-tile indexing, not a
///              hardware scalar broadcast; it is independent of `BroadcastDim`.
/// The size aspect only matters with an upfront-wait policy, where the mapping also sets
/// how many tiles are waited/popped upfront: Scalar 1, Row Wt, Col Ht, Block Ht x Wt.
/// The 1D tiles(n) shape allows only Block and Scalar; Row and Col need the 2D grid(H, W) shape.
/// The output is always Block, so there is no output mapping.
enum class InputTileMapping : uint8_t {
    Block,
    Row,
    Col,
    Scalar,
};

// =============================================================================
// 1c. TileAddressing — orthogonal tile-index addressing
// =============================================================================
//
// Composes with `InputTileMapping`: `tile_id = base + derived_from_mapping(r, c)`, where
// TileAddressing supplies `base` and InputTileMapping supplies the mapping-derived term.
//   - `Direct` (default): no base offset, zero overhead — the `+base` term and stored value
//     are compile-time-elided.
//   - `Offset`: base offset present; its value comes from the element's constructor (runtime,
//     or a compile-time constant that constant-propagates into the address add).
//   - `Strided`: base and row stride come from a `StridedTileRange`. Block maps to
//     `base + r * row_stride + c`, Col to `base + r * row_stride`, while Row and
//     Scalar retain their ordinary column/pinned behavior.
//
// `Offset` is restricted to upfront/deferred-pop or caller-managed wait/pop pairs.
// Iter-dependent wait/pop counts can't compose with a runtime base. Caller must
// size the CB for `base + window`; the chain inflates its wait/reserve/pop/push counts
// by `base` at runtime.
//
// `Strided` is restricted to caller-managed `(None, None)` policies: a gapped window cannot be
// represented by a single wait/pop/reserve/push count, so the enclosing kernel owns it.

enum class TileAddressing : uint8_t { Direct, Offset, Strided };

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
/// PerRow acquires, packs, and clears DEST once per row of the walk. WholeShape keeps DEST
/// acquired across the complete shape and emits one tile. With a single-row shape (`tiles(n)`)
/// the two coincide; spell WholeShape for a 1D reduction.
enum class DestAccumulation : uint8_t {
    Disabled,
    PerRow,
    WholeShape,
};

/// ReLU applied by the packer before writing an output tile.
enum class PackRelu : bool { Disabled = false, Zero = true };

/// Intra-tile broadcast applied to an FPU binary's srcB operand. Mirrors
/// `ckernel::BroadcastType` values (NONE=0, COL=1, ROW=2, SCALAR=3).
///
/// The dim names the axis being broadcast, not the axis that was reduced. A REDUCE_ROW result is
/// column-shaped (N,1) and broadcasts back across columns via `BroadcastDim::Col`; a REDUCE_COL
/// result (1,M) uses `BroadcastDim::Row`.
enum class BroadcastDim : uint8_t {
    None = 0,
    Col = 1,
    Row = 2,
    Scalar = 3,
};

// =============================================================================
// 1d. Grouped operand configuration
// =============================================================================
//
// `input(...)` and `output(...)` bind a buffer id to the compile-time properties of one operand.

struct InputSpec {
    uint32_t cb_id;
    WaitPolicy wait;
    PopPolicy pop;
    InputTileMapping mapping;
    DataFormatReconfig reconfig;
    TileAddressing addressing;
};

/// The second (srcB) input of a `BinaryFpu`. srcB is the only operand the FPU broadcast path
/// supports, so it carries a richer spec than the plain `InputSpec` used for srcA. Broadcast is
/// kept out of `InputSpec` because ordinary inputs (CopyTile, SFPU operands, unary broadcast,
/// and srcA) cannot consume this FPU-specific state.
struct BroadcastInputSpec {
    InputSpec input_spec;
    BroadcastDim broadcast;

    constexpr BroadcastInputSpec(InputSpec spec, BroadcastDim dim) noexcept : input_spec(spec), broadcast(dim) {}

    /// Implicit: a plain input participates as srcB without broadcasting.
    constexpr BroadcastInputSpec(InputSpec spec) noexcept : input_spec(spec), broadcast(BroadcastDim::None) {}
};

struct OutputSpec {
    uint32_t cb_id;
    ReservePolicy reserve;
    PushPolicy push;
    DataFormatReconfig reconfig;
    TileAddressing addressing;
    DestAccumulation dest_accumulation;
    L1Accumulation l1_accumulation;
    PackRelu relu;
};

/// Bind one input buffer id to its configuration.
/// Defaults: wait/pop per tile, Scalar mapping, reconfig enabled, and Direct addressing.
/// `PerTile/PerTile + Col` streams one front tile per grid row; other `Col` lifecycles retain
/// row-indexed window semantics.
constexpr InputSpec input(
    uint32_t cb_id,
    WaitPolicy wait = WaitPolicy::PerTile,
    PopPolicy pop = PopPolicy::PerTile,
    InputTileMapping mapping = InputTileMapping::Scalar,
    DataFormatReconfig reconfig = DataFormatReconfig::Enabled,
    TileAddressing addressing = TileAddressing::Direct) noexcept;
constexpr InputSpec input(uint32_t cb_id, WaitPolicy wait, PopPolicy pop, DataFormatReconfig reconfig) noexcept;

/// Bind srcB of a BinaryFpu and optionally select its intra-tile broadcast. The overload taking
/// an existing InputSpec makes compile-time helper-produced input configurations composable.
constexpr BroadcastInputSpec input(InputSpec input_spec, BroadcastDim broadcast) noexcept;
constexpr BroadcastInputSpec input(
    uint32_t cb_id,
    BroadcastDim broadcast,
    WaitPolicy wait = WaitPolicy::PerTile,
    PopPolicy pop = PopPolicy::PerTile,
    InputTileMapping mapping = InputTileMapping::Scalar,
    DataFormatReconfig reconfig = DataFormatReconfig::Enabled,
    TileAddressing addressing = TileAddressing::Direct) noexcept;
constexpr BroadcastInputSpec input(
    uint32_t cb_id, BroadcastDim broadcast, WaitPolicy wait, PopPolicy pop, DataFormatReconfig reconfig) noexcept;

/// Bind one output buffer id to its configuration.
/// Defaults: reserve/push per tile, reconfig enabled, Direct addressing, no accumulation,
/// and no pack ReLU. Parameters are ordered by how often call sites set them, so the rare
/// knobs pay the trailing-default spelling cost instead of the common ones.
constexpr OutputSpec output(
    uint32_t cb_id,
    ReservePolicy reserve = ReservePolicy::PerTile,
    PushPolicy push = PushPolicy::PerTile,
    DataFormatReconfig reconfig = DataFormatReconfig::Enabled,
    TileAddressing addressing = TileAddressing::Direct,
    DestAccumulation dest_accumulation = DestAccumulation::Disabled,
    L1Accumulation l1_accumulation = L1Accumulation::Disabled,
    PackRelu relu = PackRelu::Disabled) noexcept;

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
// 3. Block size — `IterationShape::block_size` semantics
// =============================================================================
//
// Op-struct template-param enums (Approx / Legacy) live in op_params.hpp — they
// are an op-helper concern, not part of the chain mechanics, so they are not defined here.

/// Block size. Configured with `IterationShape::tiles(n).block_size(blk)` or
/// `grid(H, W).block_size(blk)`, then passed as the shape to `eltwise_chain(shape, ...)`.
/// Each full outer iter processes `block_size` tiles across `block_size` DEST lanes (lane j at
/// slot dst_slot + j * chain_lane_width); `block_size == 1` is the per-tile shape. A partial
/// final iter always executes only its valid remainder. `BlockTailSync` selects whether
/// `PerBlockSize` lifecycles synchronize that valid remainder or the full `block_size`.
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
    BinaryFpuOp op, InputSpec a, BroadcastInputSpec b, Dst dst, DestAccumulation accumulation) noexcept;

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
    BinaryFpuOp Op,
    InputSpec AInput,
    BroadcastInputSpec BInput,
    Dst DstSlot = Dst::D0,
    DestAccumulation Accumulation = DestAccumulation::Disabled>
using BinaryFpu = detail::BinaryFpuImpl<
    AInput.cb_id,
    BInput.input_spec.cb_id,
    detail::binary_fpu_config_bits(Op, AInput, BInput, DstSlot, Accumulation)>;

/// Apply an FPU binary operation between one CB input and `DstSlot`.
/// The LLK operation is in-place in DEST: it reads and overwrites the same slot.
/// `Op` is the first template argument, matching `BinaryFpu`.
template <BinaryFpuOp Op, InputSpec Input, DestReuseType ReuseType, Dst DstSlot = Dst::D0>
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
/// `IterationShape` covers both walks: `tiles(n)` (1D, Ht=1),
/// `tiles(n).block_size(blk, BlockTailSync::FullBlock)` for a fixed-block 1D CB contract, or
/// `grid(H, W).block_size(blk, BlockTailSync::FullBlock)` for a row-blocked 2D contract.
/// A bare number is not accepted — write `IterationShape::tiles(n)` (or
/// `IterationShape::one_tile()` for one tile) so the iteration shape is always explicit.
///
/// Compile-time validation static_asserts on: illegal (Policy × Mapping) cells,
/// duplicate upfront CBs across CB-readers, colliding pack writes, and hoist requested on
/// a non-hoist-safe chain.
///
/// Index-mode (InputTileMapping) and block-mode behavior match the enum docs above: Block /
/// Row / Col / Scalar pick the per-iter tile index; input policies that own a staged CB
/// window take the upfront-block path; chains with per-tile input or output lifecycles clamp block_size to 1.
/// `BlockTailSync` affects only per-block-size synchronization counts. Row/Col need a non-streaming policy.
template <InitReconfigOwner Owner = InitReconfigOwner::Chain, class... Es>
ALWI void eltwise_chain(IterationShape shape, Es... elts);

}  // namespace compute_kernel_lib

// Bring the implementation in.
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.inl"
