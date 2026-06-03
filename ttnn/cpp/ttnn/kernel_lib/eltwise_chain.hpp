// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_chain.hpp
 * @brief Element-wise compute helper — single chain surface for all eltwise patterns.
 *
 * One helper, one dispatch path. All element-wise compute (FPU binary, SFPU unary/binary/ternary,
 * dest reuse, copy, pack, fill, rand, unary broadcast) is expressed as chain elements composed
 * via `eltwise_chain(elem0, elem1, ...)`.
 *
 * The chain owns:
 *   - the modern dst-sync window (`tile_regs_acquire/commit/wait/release`),
 *   - per-chain-element init / exec dispatch,
 *   - CB lifecycle (wait/pop on inputs, reserve/push on outputs) via per-element policy enums,
 *   - input-side and pack-side dtype reconfig via per-element policy enums (compile-time-elided
 *     via prev-CB fold),
 *   - compile-time invariant checks (illegal lifecycle/index combos, duplicate upfront CBs,
 *     pack collisions, hoist-safety).
 *
 * The chain does NOT emit any deprecated dst-sync (`acquire_dst`/`release_dst`) — modern only.
 *
 * @section caller_init_contract Caller-init contract (D8)
 *
 * The chain helper does **not** wrap any "BIG init". Engine-wide setup is the caller's
 * responsibility. The chain owns ONLY per-element setup.
 *
 * | Init kind | Owner | When to call | Notes |
 * |---|---|---|---|
 * | `compute_kernel_hw_startup(cb_a, cb_b, cb_out)` | **caller** | First statement of `MAIN()` (D5). | Engine boot.
 * MMIO-unsafe mid-kernel. Required for chains that read/write CBs. | | `binary_op_init_common(cb_a, cb_b, cb_out)` |
 * **caller** (when applicable) | Once per `MAIN()`, before any chain or raw binary call. | Required when the kernel
 * mixes raw binary primitives with chain calls; not required for chain-only kernels. | | `mm_init(...)` | **caller** |
 * N/A for eltwise chain (chain is eltwise-only). | If a kernel mixes matmul and chain, kernel author owns `mm_init`
 * placement. | | `reduce_init<...>(...)` | **caller** | N/A for eltwise chain. | Same as `mm_init`. | |
 * `add_tiles_init` / `sub_tiles_init` / `mul_tiles_init` / `init_bcast<...>(...)` | **chain** | Per-element, before
 * each binary element's `exec()`. | Chain owns the per-element programming. Caller does NOT call these. | |
 * `copy_tile_init(cb)` / `copy_tile_to_dst_init_short(cb)` | **chain** | Per-CopyTile / per-BlockCopyTile, fold-driven
 * prev-CB. | The fold emits the equivalent `_with_dt` form by combining `reconfig_data_format_srca(curr) +
 * copy_tile_init(curr)`. | | `reconfig_data_format_srca/srcb(cb)` / `pack_reconfig_data_format(cb)` | **chain** |
 * Per-element, fold-driven (D2 + D7). | Compile-time-elided when prev_cb == cur_cb. | | `tile_regs_acquire / commit /
 * wait / release` | **chain** | Per-iteration. | Chain owns the lifecycle. |
 *
 * Note on FP32 DEST accumulation: the chain inherits the kernel's build-time DST_ACCUM_MODE
 * (from `FP32_DEST_ACC_EN`). All DEST slot indexing already accounts for this via
 * `DEST_AUTO_LIMIT`. No per-element opt-in or mid-kernel `enable_fp32_dest_acc()` /
 * `disable_fp32_dest_acc()` toggles — DEST mode is kernel-wide.
 *
 * @section hw_startup_placement compute_kernel_hw_startup placement (D5)
 *
 * `compute_kernel_hw_startup` is the first statement of `MAIN()` if the chain shape requires
 * it (chains with at least one CB-reader and one CB-writer). Multi-stage kernels (different PACK
 * output CB per stage) emit one boot per stage — stage 1 at top of `MAIN()`, stages 2+
 * immediately before that stage's chain call. Mid-`MAIN()` placement is undefined per
 * `compute_kernel_hw_startup.h:26-30` (MMIO writes unsafe to call mid-kernel).
 *
 * | Chain shape | Caller pre-chain init | Placement |
 * |---|---|---|
 * | Unary `CopyTile<cbA, …>{} … PackTile<cbOut, …>{}` | `compute_kernel_hw_startup(cbA, cbA, cbOut);` | First statement
 * of `MAIN()`. | | Binary `BinaryFpu<cbA, cbB, …>{} … PackTile<cbOut>{}` | `compute_kernel_hw_startup(cbA, cbB,
 * cbOut);` | First statement of `MAIN()`. | | `DestReuseBinary<cb, …>{}` only | `compute_kernel_hw_startup(cb, cb,
 * cbOut);` | First statement of `MAIN()`. | | Multi-stage (e.g. `logit_kernel.cpp`) | One `compute_kernel_hw_startup`
 * per stage. | Stage 1 at top of `MAIN()`; stages 2+ immediately before that stage's chain call. | | Mid-loop chain
 * (moreh inner-loop pattern) | Caller's outer `binary_op_init_common(...)` already covers the chain — no extra boot
 * needed. | Omit; calling `compute_kernel_hw_startup` mid-`MAIN()` here is **undefined per D5**. | | Copy-only chain
 * whose CB formats already match defaults | Omit. | N/A. |
 *
 * Rejected pattern: a `*_with_init` convenience that folds `compute_kernel_hw_startup` into the
 * chain call is an antipattern — it reintroduces BIG init into the chain (violating the D8
 * caller-init contract), is only correct for single-stage kernels, and produces undefined behaviour
 * the moment it is used in a multi-stage or mid-loop chain. The caller owns BIG init; do not add one.
 *
 * @section fp32_dest_acc FP32 DEST accumulation — build-flag-driven (no per-element opt-in)
 *
 * The kernel's build flag `FP32_DEST_ACC_EN` determines DEST accumulation mode for the whole
 * kernel. `DEST_AUTO_LIMIT` (in `dest_helpers.hpp`) already accounts for the halved slot count
 * when fp32 is on. Chain elements operate transparently under whatever mode the build picked —
 * no per-element template parameter, no SFINAE fold, no mid-kernel `enable_fp32_dest_acc()` /
 * `disable_fp32_dest_acc()` transitions.
 *
 * @section block_path_fold Block-path fold (D7)
 *
 * Block elements (`BlockCopyTile`, `BlockBinaryFpu`, `BlockPackTile`) participate in the same
 * compile-time prev-CB / prev-fp32 fold as streaming elements via the uniform
 * `reconfig_srca_cb` / `reconfig_srcb_cb` / `reconfig_pack_cb` static accessors. `init()` bodies
 * no longer emit reconfig — that's fold-driven. The `_with_dt` two-arg LLK forms (formerly at
 * `eltwise_block.hpp:72,236`) are now decomposed into the chain's
 * `reconfig_data_format_srca(curr) + copy_tile_init(curr)` sequence, compile-time-elided when
 * prev_cb == curr_cb.
 *
 * @section caller_init_wrong_way Anti-examples (D8)
 *
 * Three failure modes the contract above prevents:
 * 1. **Mid-`MAIN()` `compute_kernel_hw_startup`** — undefined behaviour per
 *    `compute_kernel_hw_startup.h:26-30` (MMIO write mid-kernel; race conditions under load).
 * 2. **Chain-only kernel forgetting `compute_kernel_hw_startup`** — silent miscompile; default-
 *    format reconfigs may match by accident; first mismatched-dtype kernel produces garbage.
 *
 * @section big_init_grep_gate D8 grep gate
 *
 * Manual one-liner the reviewer / future contributor runs ad-hoc to verify the chain helper
 * has no BIG-init call sites:
 *
 * @code
 *   grep -nE 'init_common|compute_kernel_hw_startup|mm_init|reduce_init' \
 *        ttnn/cpp/ttnn/kernel_lib/eltwise_{chain.hpp,chain.inl,block.hpp}
 * @endcode
 *
 * Expected: only the `#include "compute_kernel_hw_startup.h"` line in this header (and any
 * doxygen comment matches such as this one). Zero call sites in helper bodies.
 *
 * Worked examples
 * ---------------
 *
 *   // InputLifecycle::Streaming unary — Exp(x) → out
 *   eltwise_chain(num_tiles,
 *       CopyTile<cb_in,  Dst::D0, InputLifecycle::Streaming>{},
 *       Exp<>{},
 *       PackTile<cb_out, Dst::D0, OutputLifecycle::Streaming>{}
 *   );
 *
 *   // InputLifecycle::Streaming binary — A + B → out
 *   //   BinaryFpu writes to DEST; the output CB lives on the PackTile element.
 *   eltwise_chain(num_tiles,
 *       BinaryFpu<cb_a, cb_b, BinaryFpuOp::Add>{},
 *       PackTile<cb_out, Dst::D0, OutputLifecycle::Streaming,
 *                OperandKind::Scalar, PackTileReconfig::Output>{}
 *   );
 *
 *   // Fan-out — same input, two outputs
 *   eltwise_chain(num_tiles,
 *       CopyTile<cb_in, Dst::D0, InputLifecycle::HeldStream>{},
 *       CopyTile<cb_in, Dst::D1, InputLifecycle::NoWaitPop>{},
 *       Exp<Approx::Exact, Approx::Fast, Dst::D0>{},
 *       Tanh<Dst::D1>{},
 *       PackTile<cb_out_a, Dst::D0, OutputLifecycle::Streaming>{},
 *       PackTile<cb_out_b, Dst::D1, OutputLifecycle::Streaming>{}
 *   );
 *
 *   // Block reduction with upfront reserve / pop-at-end (auto-detected via `Es::is_upfront`)
 *   eltwise_chain(num_tiles,
 *       CopyTile<cb_in, Dst::D0, InputLifecycle::Bulk, OperandKind::Block>{},
 *       Exp<>{},
 *       PackTile<cb_out, Dst::D0, OutputLifecycle::Bulk>{}
 *   );
 *
 *   // Asymmetric bcast walk — A streams the tile range, B pinned at tile 0
 *   //   (softmax-style: out[t] = exp(in[t] - max), max pinned at tile 0)
 *   //   BinaryFpu's 8th template arg is AIndex; 10th (trailing) is BIndex (defaults to AIndex).
 *   eltwise_chain(num_tiles,
 *       BinaryFpu<cb_in, cb_max, BinaryFpuOp::Sub, BroadcastDim::Col,
 *                 BinaryDataFormatReconfig::None,
 *                 InputLifecycle::Bulk,                    // A: wait N upfront, pop at end
 *                 InputLifecycle::HeldStream,              // B: wait 1, never pop
 *                 OperandKind::Block,      // AIndex — A walks 0..num_tiles-1
 *                 Dst::D0,
 *                 OperandKind::Scalar>{},  // BIndex — B pinned at tile 0
 *       Exp<>{},
 *       PackTile<cb_out, Dst::D0, OutputLifecycle::Streaming>{}
 *   );
 *
 * Non-goals
 * ---------
 *  - Cumulative wait policy (`cb_wait_front(base + i)`). Out of scope; raw LLK only.
 *  - Mid-loop dtype swaps. Reconfig is entry-time per chain element.
 *  - L1 accumulation (`pack_reconfig_l1_acc`), pack-relu, pack-rows. Future `OutputLifecycle` extensions.
 *  - Held-DEST patterns. Out of scope (zero TSV evidence).
 *  - `acquire_dst/release_dst` and `ACQ()/REL()` macros — modern dst-sync only. Kernels migrate
 *    their dst-sync as part of adopting the chain.
 *
 * Reconfig (`with_dt_tree`-style) — fold-driven post commits 2-3
 * ----------------------------------------------------------------
 *  - CopyTileReconfig::Input         → fold emits single-side reconfig on srca (compile-time-elided when prev == curr).
 *  - BinaryDataFormatReconfig::Input → fold emits per-side reconfig on srca + srcb (compile-time-elided per side).
 *    Pack-side reconfig is owned by the downstream `PackTile` (`PackTileReconfig::Output`); BinaryFpu writes to DEST,
 *    never to a CB.
 *  - BinaryDataFormatReconfig::SrcA  → fold emits srca reconfig only (caller asserts srcb is already programmed).
 *  - BinaryDataFormatReconfig::SrcB  → fold emits srcb reconfig only (caller asserts srca is already programmed).
 *  - DestReuseReconfig::Input        → fold emits per-side reconfig (srca OR srcb depending on ReuseType).
 *  - DestReuseReconfig::SrcA         → fold emits srca reconfig only, decoupled from ReuseType.
 *  - DestReuseReconfig::SrcB         → fold emits srcb reconfig only, decoupled from ReuseType.
 *  - PackTileReconfig::Output        → fold emits pack reconfig — two-arg `_with_dt` form when prev_pack_cb is known,
 *    single-arg on first emit.
 *  - UnaryBcastReconfig::Input       → currently bundled into `unary_bcast_init`.
 *
 * Emission shapes the fold chooses between (see `emit_pre_element_transitions`):
 *
 *   srca + srcb both reconfig, both have prev   → reconfig_data_format(prev_a, curr_a, prev_b, curr_b)  (4-arg
 * _with_dt) srca + srcb both reconfig, both first-emit  → reconfig_data_format(curr_a, curr_b)                  (2-arg
 * combined) srca + srcb both reconfig, mixed prev-state → reconfig_data_format_src{a,b}(prev, curr) or (curr) per side
 *   one side only                               → reconfig_data_format_src{a,b}(prev, curr) or (curr)
 *   pack-side                                   → pack_reconfig_data_format(prev_p, curr_p) or (curr_p)
 *
 * The LLK's `_with_dt` overloads include a runtime format-equality check against the CB metadata tables
 * (`unpack_src_format[]` / `unpack_dst_format[]`) and short-circuit the unpack-side and math-side reprograms
 * independently when formats match — so emitted reconfigs are no-ops at the hardware level when the involved CBs
 * happen to carry the same dtype.
 */

#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

// LLK / compute-API includes consumed by the inline implementation.
#include "api/compute/bcast.h"
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"

namespace compute_kernel_lib {

// =============================================================================
// 1. Marker tag hierarchy (data direction → kind)
// =============================================================================

/// Element reads ≥1 CB. Pure marker — concrete elements declare `cb_a_id()` and
/// (if binary) `cb_b_id()`. No stub defaults (§1.7 footgun avoidance).
struct CbReaderTag {};
/// Element writes to a CB. Pure marker — concrete elements declare `pack_cb_id()`.
/// No stub defaults.
struct CbWriterTag {};
/// Element neither reads nor writes a CB (DEST-internal). Carries only the
/// behavioural trait defaults (`is_upfront`, `clashes_with_fpu`) — no CB-id stubs.
/// The chain pipeline SFINAE-detects `cb_a_id()` / `cb_b_id()` / `pack_cb_id()` on
/// the element directly and never reaches a DestOnlyTag default.
struct DestOnlyTag {
    static constexpr bool is_upfront = false;
    static constexpr bool clashes_with_fpu = false;
};

/// Pure CB → DEST move (no compute).
struct CopyTileTag : CbReaderTag {};
/// 2 CBs → DEST FPU compute (add/sub/mul + bcast variants).
struct BinaryFpuTag : CbReaderTag {};
/// 1 CB + DEST → DEST FPU compute (binary_dest_reuse_tiles).
struct DestReuseBinaryTag : CbReaderTag {};
/// 1 CB → DEST row/col/scalar broadcast (unary_bcast).
struct UnaryBcastTag : CbReaderTag {};

/// DEST → CB store (pack_tile / pack_tile_block).
struct PackTileTag : CbWriterTag {};

/// Constant → DEST (no CB read).
struct FillTileTag : DestOnlyTag {};
/// RNG → DEST (no CB read).
struct RandTileTag : DestOnlyTag {};

// Trait predicates (cheat-sheet in proposal §2.1):
//
//  Sweep / decision                                            | predicate
//  ------------------------------------------------------------|---------------------------
//  Duplicate upfront-CB check across all CB-consumers          | is_cb_reader_op_v
//  Output-CB collision / fan-out across all writers            | is_cb_writer_op_v
//  Hoist-safety "chain shape is CopyTile + 1 SFPU op"          | is_copy_tile_op_v
//  FPU-clash reinit                                            | is_binary_fpu_op_v ‖
//                                                              | is_dest_reuse_binary_op_v ‖
//                                                              | is_unary_bcast_op_v
//  Hoist exclusion: element issues a pack inside the loop      | is_pack_tile_op_v
//  No CB lifecycle to check — pure DEST internal               | is_dest_only_op_v
template <class T>
inline constexpr bool is_cb_reader_op_v = std::is_base_of_v<CbReaderTag, T>;
template <class T>
inline constexpr bool is_cb_writer_op_v = std::is_base_of_v<CbWriterTag, T>;
template <class T>
inline constexpr bool is_dest_only_op_v = std::is_base_of_v<DestOnlyTag, T>;
template <class T>
inline constexpr bool is_copy_tile_op_v = std::is_base_of_v<CopyTileTag, T>;
template <class T>
inline constexpr bool is_binary_fpu_op_v = std::is_base_of_v<BinaryFpuTag, T>;
template <class T>
inline constexpr bool is_dest_reuse_binary_op_v = std::is_base_of_v<DestReuseBinaryTag, T>;
template <class T>
inline constexpr bool is_unary_bcast_op_v = std::is_base_of_v<UnaryBcastTag, T>;
template <class T>
inline constexpr bool is_pack_tile_op_v = std::is_base_of_v<PackTileTag, T>;
template <class T>
inline constexpr bool is_fill_tile_op_v = std::is_base_of_v<FillTileTag, T>;
template <class T>
inline constexpr bool is_rand_tile_op_v = std::is_base_of_v<RandTileTag, T>;

/// SFPU (DEST-internal, non-RNG, non-fill) element predicate. SFPU ops inherit
/// from `DestOnlyTag` via `UnaryOp` / `BinaryOp` / `TernaryOp` / `QuaternaryOp`;
/// Fill / Rand share the `DestOnlyTag` lineage but their init programs PRNG /
/// fill state, not the SFPU MOP / ADDR_MOD_7 lane. The hoist gate counts distinct
/// SFPU init types — `is_sfpu_op_v` is the predicate.
template <class T>
inline constexpr bool is_sfpu_op_v = is_dest_only_op_v<T> && !is_fill_tile_op_v<T> && !is_rand_tile_op_v<T>;

/// FPU-kind (non-CopyTile, FPU-MOP-touching) element predicate. Groups
/// `BinaryFpu`, `DestReuseBinary`, `UnaryBcast` — each programs the FPU MOP /
/// ADDR_MOD_0..3 lane on init via the binary-op init path.
template <class T>
inline constexpr bool is_fpu_kind_op_v =
    is_binary_fpu_op_v<T> || is_dest_reuse_binary_op_v<T> || is_unary_bcast_op_v<T>;

/// MATH-MOP-touching element predicate. Groups every element whose init
/// programs the MATH MOP / ADDR_MOD_0..3 lane: `CopyTile` (via
/// `copy_tile_to_dst_init_short`) and the FPU-kind ops (`BinaryFpu`,
/// `DestReuseBinary`, `UnaryBcast`). The hoist gate (doc G3 + the
/// CopyTile-versus-FPU clash from the old `chain_has_non_copy_tile_fpu_clash`
/// predicate) requires all such elements in a chain to be the same
/// instantiated type — otherwise the boot-time fold leaves only the last
/// init's MOP programmed and earlier elements run with the wrong MOP.
template <class T>
inline constexpr bool is_math_mop_op_v = is_copy_tile_op_v<T> || is_fpu_kind_op_v<T>;

// =============================================================================
// 1a-bis. CB id sentinel
// =============================================================================

/// "Not a CB" sentinel for compile-time CB-id queries (`cb_a_of`, `cb_b_of`,
/// `pack_cb_of`). CB indices are `tt::CBIndex` (uint8_t, 0..31 in practice), so
/// `uint32_t` max is guaranteed not to collide with any real CB id. Used by the
/// collision-detection predicates to skip elements that don't carry a CB on the
/// queried slot, without confusing "absent" with "cb_id == 0" (which is a valid
/// `tt::CBIndex::c_0`).
inline constexpr uint32_t kNoCb = 0xFFFFFFFFu;

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
/// (`eltwise_chain(n_tiles, ...)`) keep working without churn.
///
/// `of/row/col/single` aliases mirror `binary_op_helpers`' `BinaryInputBlockShape`
/// and are preserved for callers that already use them.
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
// Per `eltwise_taxonomy.md`, each input's lifecycle is a `(WaitPolicy, PopPolicy)`
// pair, each output's lifecycle is a `(ReservePolicy, PushPolicy)` pair. Named
// constants compose the legal pairs; custom struct literals are validated by
// `is_legal_input_lifecycle` / `is_legal_output_lifecycle`.
//
// Custom struct literals (e.g. `InputLifecycle{WaitPolicy::Upfront, PopPolicy::PerTile}`)
// are accepted at the template-parameter site and validated against the 2-axis legal
// set; this gives callers fine-grained `{wait, pop}` control beyond the named cells.

enum class WaitPolicy : uint8_t {
    None,        // chain emits no cb_wait_front
    PerTile,     // wait 1 per iter
    PerChunk,    // wait K per K-iter chunk
    Upfront,     // wait M once at entry (M = kind's tile count)
    Cumulative,  // wait (i+1) per iter / chunk
};

enum class PopPolicy : uint8_t {
    None,      // chain emits no cb_pop_front
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

// InputLifecycle::Bulk wait + per-tile pop. Caller (or upstream stage) bulk-waits M tiles upfront,
// chain drains one per iter. Used by SDPA in-place block helpers and groupnorm
// sharded in-place gamma/beta (~5 sites).
inline constexpr InputLifecycle InputLifecycle::BulkDrain = {WaitPolicy::Upfront, PopPolicy::PerTile};

// Half-edge lifecycles — chain owns wait OR pop, not both. The chain emits its
// edge; the caller is responsible for the other. Load-bearing for persistent
// broadcast operands (gamma, beta, mean, recip_std, etc.) that outlive the
// chain call.

// Chain waits M upfront, never pops. Caller owns the final pop. Used by
// reduction-result tiles consumed by many bcast pack calls (~52 sites).
inline constexpr InputLifecycle InputLifecycle::HeldBulk = {WaitPolicy::Upfront, PopPolicy::None};

// Chain waits cumulatively (i+1 per iter), never pops. Caller owns the final
// pop. Persistent gamma/beta operands in normalization (~33 sites).
inline constexpr InputLifecycle InputLifecycle::HeldCumulative = {WaitPolicy::Cumulative, PopPolicy::None};

// Chain waits 1 per iter (idempotent), never pops. Caller owns the final pop.
// Moreh helper `pop=0` caller param convention (~14 sites).
inline constexpr InputLifecycle InputLifecycle::HeldStream = {WaitPolicy::PerTile, PopPolicy::None};

// Caller bulk-waited externally, chain bulk-pops M at exit. Multi-phase
// consumer chains in softmax cb_exps (~7 sites).
inline constexpr InputLifecycle InputLifecycle::DeferredPop = {WaitPolicy::None, PopPolicy::AtEnd};

// Caller waited externally, chain pops per-tile. Used in some pre-staged
// (sharded) operand patterns where the chain doesn't re-wait but does drain
// per-tile.
inline constexpr InputLifecycle InputLifecycle::NoWaitPop = {WaitPolicy::None, PopPolicy::PerTile};

/// Validates a caller-constructed `InputLifecycle` against the legal set.
/// Used by every input element's `static_assert` at chain composition.
/// Half-edge cells (InputLifecycle::HeldBulk, InputLifecycle::HeldCumulative, InputLifecycle::HeldStream,
/// InputLifecycle::DeferredPop) are legal because the catalog audit found them load-bearing for persistent broadcast
/// operands. Other half-edge combinations are static_assert rejected — see audit-confirmed cells in
/// eltwise_taxonomy.md.
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

    // Named cells — written type-qualified (e.g. `OutputLifecycle::Bulk`). The historical
    // `Out` prefix is dropped now that the type qualifies them. Defined out-of-line below.
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
/// Runtime/compile-time tile-index offsets that previously lived as separate
/// kinds (`Pinned`/`Absolute`/`BlockIterOffset`) are now expressed by composing
/// one of these four canonical kinds with a `TileBase` (see `TileBase` types
/// below). The kind carries the iteration shape; `TileBase` carries the offset.
enum class OperandKind : uint8_t {
    Block,   // Ht × Wt — walks the full iteration domain
    Row,     // 1  × Wt — broadcast down rows
    Col,     // Ht × 1  — broadcast across cols
    Scalar,  // 1  × 1  — broadcast everywhere
};

/// Kind × InputLifecycle compatibility.
///
/// Only Block carries structural restrictions; non-Block (Scalar / Row / Col)
/// is caller-sized and works with any lifecycle as long as the caller's
/// `n_tiles` matches the lifecycle's consumption pattern.
///
/// Block walks absolute CB-front index `base_tile + i` per iter (chain
/// dispatcher passes the absolute flat index; InputLifecycle::Chunked is the one exception —
/// it uses a chunk-local index). Two structural footguns follow:
///
///   (a) PerTile pop (InputLifecycle::Streaming / InputLifecycle::BulkDrain / InputLifecycle::NoWaitPop) shifts the CB
///   front
///       each iter; combined with absolute indexing the chain reads the wrong
///       tile after iter 0 (idx (base+i) into the now-shifted front yields
///       original tile (base + 2i)). Caller sizing cannot rescue this.
///   (b) PerTile wait of 1 (InputLifecycle::HeldStream) is either redundant (caller pre-pushed
///       all n — use InputLifecycle::HeldBulk) or under-waiting (caller streams — chain reads
///       tile i before producer pushed it). Never tracks the per-iter
///       requirement for a walking Block reader.
///
/// Non-Block kinds dodge both footguns: index is constant (Scalar) or driven
/// by ht/wt alone (Row/Col), so the CB-front shift from PerTile pop is benign
/// and PerTile wait of 1 can be satisfied by the producer pushing the single
/// broadcast tile once. Whether the chain actually drains the right number of
/// tiles is the caller's responsibility (depends on their `n_tiles`).
///
/// Block — legal lifecycles (7):
///   InputLifecycle::Bulk / InputLifecycle::Pipelined / InputLifecycle::HeldBulk / InputLifecycle::HeldCumulative /
///   InputLifecycle::Chunked / InputLifecycle::CallerManaged / InputLifecycle::DeferredPop
///
/// Scalar / Row / Col — every legal InputLifecycle (caller-sized).
constexpr bool is_legal_kind_lifecycle(OperandKind kind, InputLifecycle lc) noexcept {
    if (!is_legal_input_lifecycle(lc)) {
        return false;
    }
    if (kind == OperandKind::Block) {
        // Block walks absolute idx with M = Ht·Wt = iter count. Exclude PerTile-pop
        // (InputLifecycle::Streaming / InputLifecycle::BulkDrain / InputLifecycle::NoWaitPop — front-shift +
        // absolute-idx footgun) and PerTile-wait of 1 (InputLifecycle::HeldStream — never tracks per-iter requirement).
        // Growing (Cumulative) and chunked (PerChunk) counts ARE legal here because
        // M = iter count, so the counts never exceed operand size.
        return lc == InputLifecycle::Bulk || lc == InputLifecycle::Pipelined || lc == InputLifecycle::HeldBulk ||
               lc == InputLifecycle::HeldCumulative || lc == InputLifecycle::Chunked ||
               lc == InputLifecycle::CallerManaged || lc == InputLifecycle::DeferredPop;
    }
    // Non-Block (Scalar / Row / Col): M < iter count. Reject lifecycles whose
    // wait/pop count grows with iter index (InputLifecycle::Pipelined, InputLifecycle::HeldCumulative) or scales
    // with chunk size (InputLifecycle::Chunked) — these emit counts that exceed M (deadlock past
    // iter M). Only Block, where M = iter count, can absorb these counts safely.
    if (lc == InputLifecycle::Pipelined || lc == InputLifecycle::HeldCumulative || lc == InputLifecycle::Chunked) {
        return false;
    }
    if (kind == OperandKind::Scalar) {
        // Scalar (M=1, single broadcast tile): accepts the remaining 8 lifecycles —
        // PerTile-pop ones (InputLifecycle::Streaming / InputLifecycle::BulkDrain / InputLifecycle::NoWaitPop) and
        // InputLifecycle::HeldStream are caller-sized for n_tiles=1, InputLifecycle::Bulk / InputLifecycle::HeldBulk /
        // InputLifecycle::CallerManaged / InputLifecycle::DeferredPop are unconditional.
        return true;
    }
    // Row / Col (2D only — 1D rejects these at entry): the operand window is
    // re-read across the full Ht·Wt iteration (Row's Wt tiles get read Ht times,
    // Col's Ht tiles get read Wt times). PerTile-pop lifecycles (InputLifecycle::Streaming /
    // InputLifecycle::BulkDrain / InputLifecycle::NoWaitPop) drain the operand before re-iteration completes;
    // InputLifecycle::HeldStream's PerTile wait of 1 says nothing about which tile arrived.
    // Only "operand persists across all iters" lifecycles work.
    return lc == InputLifecycle::Bulk || lc == InputLifecycle::HeldBulk || lc == InputLifecycle::CallerManaged ||
           lc == InputLifecycle::DeferredPop;
}

// =============================================================================
// 1d. TileOffset — orthogonal tile-index offset (present / absent)
// =============================================================================
//
// Composes with `OperandKind` to express compound CB tile addressing:
//
//     tile_id = base + derived_from_kind(r, c)
//              ^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^
//              TileOffset              OperandKind (Block / Row / Col / Scalar)
//
// Binary: an element either has an offset or it doesn't.
//   - `TileOffset::Unset` : default — no offset, zero overhead (the `+base` term and the
//                           stored value are compile-time-elided).
//   - `TileOffset::Set`   : an offset is present; its (runtime) value is supplied via the
//                           element's constructor (a compile-time constant works too — it
//                           constant-propagates into the address add).
//
// Lifecycle restriction: `TileOffset::Set` requires InputLifecycle::Bulk-family or InputLifecycle::CallerManaged
// lifecycles (input: InputLifecycle::Bulk / InputLifecycle::HeldBulk / InputLifecycle::DeferredPop /
// InputLifecycle::BulkDrain / InputLifecycle::CallerManaged; output: OutputLifecycle::Bulk /
// OutputLifecycle::DeferredReserve / OutputLifecycle::HeldReserve / OutputLifecycle::CallerManaged).
// InputLifecycle::Streaming / InputLifecycle::Chunked / Cumulative / Held{Stream,Cumulative} /
// InputLifecycle::NoWaitPop are forbidden because their wait/pop counts are iter-dependent and don't compose with
// runtime base offsets cleanly. Caller must size the CB to hold `base + window` tiles before the chain reads them. The
// chain's emitted wait/reserve/pop/push counts inflate by `base` at runtime.

enum class TileOffset : bool { Unset = false, Set = true };

/// Extract the offset value stored on an element. Returns 0 (compile-time-folded) when the
/// element's `Offset` is `Unset`, so the `+base` term and the stored field vanish.
template <TileOffset Offset>
ALWI uint32_t tile_base_value(uint32_t stored) noexcept {
    if constexpr (Offset == TileOffset::Unset) {
        (void)stored;
        return 0u;
    } else {
        return stored;
    }
}

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

/// Block size. Runtime arg on `eltwise_chain(n_tiles, block_size, ...)`. Each outer
/// iter processes `block_size` tiles in `block_size` DEST lanes (lane j at slot
/// dst_slot + j * chain_lane_width). `block_size == 1` reproduces the per-tile shape;
/// an implicit-block overload defaults to 1 for callers that don't pass it.
///
/// Caller responsibilities (no runtime check fires inside the chain):
///   - DEST footprint: `block_size * chain_lane_width <= DEST_AUTO_LIMIT`. Query
///     `chain_max_block_v<Chain>` for the maximum legal value and static_assert at
///     the call site if a build-time check is desired.
///   - Policy compat: streaming CB-reader chains (WaitAndPop / WaitNoPop / InputLifecycle::NoWaitPop)
///     consume one tile per iter — incompatible with `block_size > 1`. The chain
///     silently clamps `block_size` to 1 via `if constexpr (!chain_supports_block_v<Chain>)`
///     when the chain type doesn't support block-mode, so an explicit value > 1 is
///     just ignored for those chains.

// =============================================================================
// 4. Policy enums — CB lifecycle, indexing, reconfig, broadcast
// =============================================================================

/// CB-input tile indexing — `OperandKind` values used as the index-mode
/// template parameter on CopyTile / BinaryFpu / PackTile.
///
/// 2D-mode semantics (only meaningful in the `EltwiseShape{Ht, Wt}` chain overload —
/// in the 1D `n_tiles` overload, Row/Col are static_assert-rejected since there is
/// no Ht axis):
///
///   | Kind     | Tile index in 2D walk    | Upfront window |
///   |----------|--------------------------|----------------|
///   | Scalar   | 0                        | 1              |
///   | Block    | ht * Wt + wt   (flat)    | Ht * Wt        |
///   | Row      | wt                       | Wt             |
///   | Col      | ht                       | Ht             |
///
/// Runtime/compile-time tile offsets are expressed via `TileBase` (composed with
/// any of the four kinds), not as separate index modes.
///
/// Row / Col require non-streaming CB policy (`InputLifecycle::Bulk`, `InputLifecycle::HeldStream`,
/// `InputLifecycle::NoWaitPop`, `InputLifecycle::CallerManaged`, `InputLifecycle::Pipelined`) — caller stages all
/// broadcast operand tiles before the chain starts. Same constraint as `binary_op_helpers`' ROW/COL static_assert.

/// CopyTile dtype-reconfig.
///
/// Why `None` is load-bearing (not just a perf opt-out): the fold compile-time-elides a
/// reconfig only when prev_cb == cur_cb *within the chain*. The FIRST CB-reader has no
/// in-chain predecessor, so with `Input` it emits an unconditional single-arg reconfig on
/// entry — and the single-arg form does NOT short-circuit on format equality the way the
/// two-arg `_with_dt` form does. `None` is the caller asserting "the boot init
/// (compute_kernel_hw_startup / init_sfpu / binary_op_init_common) already programmed this
/// exact format — skip the redundant entry reprogram." Canonical case: a single-input,
/// single-output kernel (copy / identity / typecast) whose CBs are set once at boot. Do not
/// default-on or remove this knob.
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

/// FPU broadcast dimension. Caller MUST pass explicitly — no inference.
/// Mirrors `ckernel::BroadcastType` values (NONE=0, COL=1, ROW=2, SCALAR=3).
///
/// Reduce↔Broadcast mapping — the "reduce-row produces column-shaped output"
/// surprise that lives where it is needed:
///
///   | Reduce direction | Output shape | Broadcast direction downstream     |
///   |------------------|--------------|------------------------------------|
///   | REDUCE_ROW       | (N, 1)       | BroadcastDim::Col (bcast cols)    |
///   | REDUCE_COL       | (1, M)       | BroadcastDim::Row (bcast rows)    |
///   | REDUCE_SCALAR    | (1, 1)       | BroadcastDim::Scalar              |
///   | REDUCE_W (alias) | (N, 1)       | BroadcastDim::Col                 |
///   | REDUCE_H (alias) | (1, M)       | BroadcastDim::Row                 |
///
/// Example: softmax computes a per-row max (REDUCE_ROW → (N,1)) then needs to
/// subtract that vector across columns of the original — that subtract is a
/// `sub_tiles_bcast<BroadcastDim::Col>`, NOT `BroadcastDim::Row`. The dim names
/// describe which axis is BROADCAST, not which axis was reduced.
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

/// DestReuseBinary reconfig (NEVER a bool — see proposal §2.5).
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
/// single-arg form on first emit. The LLK's runtime format-equality check makes
/// the legacy `OutputConditional` distinction redundant; only `Output` remains.
enum class PackTileReconfig : uint8_t {
    None,
    Output,  // fold emits pack_reconfig_data_format(prev_p, curr_p) when prev_p known, else (curr_p)
};

// =============================================================================
// 5. CRTP bases — UnaryOp / BinaryOp / TernaryOp / QuaternaryOp
// =============================================================================
//
// Single dispatch contract (§4.5): every chain element exposes `void exec(uint32_t)
// const`. The CRTP bases provide a default that forwards to a static `exec_impl()`
// supplied by the derived op. Runtime-param ops (Power, Hardtanh, Threshold, …)
// override `exec(uint32_t)` directly to capture their instance state. Forgetting
// both is a compile error — no silent fallthrough.
//
// Example (static SFPU):
//
//   template <Approx A = Approx::Exact, Approx F = Approx::Fast, Dst Slot = Dst::D0>
//   struct Exp : UnaryOp<Exp<A, F, Slot>, Slot> {
//       static void init()       { exp_tile_init<A == Approx::Fast, F == Approx::Fast>(); }
//       static void exec_impl()  { exp_tile<A == Approx::Fast, F == Approx::Fast>(to_u32(Slot)); }
//   };
//
// Example (runtime-param SFPU — overrides exec(uint32_t) directly):
//
//   template <Dst Slot = Dst::D0>
//   struct Power : UnaryOp<Power<Slot>, Slot> {
//       uint32_t exponent;
//       constexpr explicit Power(uint32_t e) noexcept : exponent(e) {}
//       static void init() { power_tile_init(); }
//       void exec(uint32_t /*i*/) const { power_tile(to_u32(Slot), exponent); }
//   };

template <class Derived, Dst Slot>
struct UnaryOp : DestOnlyTag {
    static_assert(
        to_u32(Slot) < DEST_AUTO_LIMIT, "UnaryOp: DEST slot exceeds compile-time DEST capacity (DEST_AUTO_LIMIT)");

    static constexpr Dst dst_idx = Slot;
    static constexpr uint32_t max_dst() { return to_u32(Slot); }
    /// Per-lane DEST footprint (item 2). Used by chain to pick auto BlockSize.
    /// Default = `to_u32(Slot) + 1` (op writes only Slot). Override per-op when the
    /// op references more slots (e.g. Mask uses DataSlot AND DataSlot+1).
    static constexpr uint32_t lane_width = to_u32(Slot) + 1;

    /// Pipeline dispatch — forwards to `Derived::exec_impl(slot_offset)`. Override
    /// in derived to consume runtime payload (per-instance fields). `slot_offset`
    /// is added by the chain to shift DEST writes into lane `j` when `BlockSize > 1`;
    /// `BlockSize == 1` passes 0 (per-tile shape).
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const { Derived::exec_impl(slot_offset); }
};

template <class Derived, Dst In0, Dst In1, Dst Out>
struct BinaryOp : DestOnlyTag {
    static_assert(
        to_u32(In0) < DEST_AUTO_LIMIT && to_u32(In1) < DEST_AUTO_LIMIT && to_u32(Out) < DEST_AUTO_LIMIT,
        "BinaryOp: DEST slot exceeds compile-time DEST capacity (DEST_AUTO_LIMIT)");
    // NOTE: slot-distinctness is *not* enforced here. SFPU binary ops (AddBinary /
    // SubBinary / MulBinary / DivBinary) routinely operate in-place (Out == In0 or
    // Out == In1) and even `In0 == In1` is legal (e.g. squaring). The FPU binary
    // chain element (`BinaryFpu`) reads its inputs from CBs, not DEST slots, so it
    // also doesn't need In0 != In1. If a future op truly requires distinct DEST
    // slots, it should enforce that locally rather than at the CRTP base.

    static constexpr Dst in0 = In0;
    static constexpr Dst in1 = In1;
    static constexpr Dst out = Out;
    static constexpr uint32_t max_dst() {
        uint32_t a = to_u32(In0), b = to_u32(In1), c = to_u32(Out);
        return a > b ? (a > c ? a : c) : (b > c ? b : c);
    }
    static constexpr uint32_t lane_width = max_dst() + 1;

    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const { Derived::exec_impl(slot_offset); }
};

template <class Derived, Dst In0, Dst In1, Dst In2, Dst Out>
struct TernaryOp : DestOnlyTag {
    static_assert(
        to_u32(In0) < DEST_AUTO_LIMIT && to_u32(In1) < DEST_AUTO_LIMIT && to_u32(In2) < DEST_AUTO_LIMIT &&
            to_u32(Out) < DEST_AUTO_LIMIT,
        "TernaryOp: DEST slot exceeds compile-time DEST capacity (DEST_AUTO_LIMIT)");
    // NOTE: slot-distinctness is *not* enforced here (mirrors BinaryOp). SFPU ternary
    // ops (where / lerp / addcmul / addcdiv) routinely write Out into one of the input
    // slots in-place — the kernel reads all three inputs before overwriting. If a
    // future op truly requires distinct DEST slots, enforce it locally rather than at
    // the CRTP base.

    static constexpr Dst in0 = In0;
    static constexpr Dst in1 = In1;
    static constexpr Dst in2 = In2;
    static constexpr Dst out = Out;
    static constexpr uint32_t lane_width = []() {
        uint32_t m = to_u32(In0);
        if (to_u32(In1) > m) {
            m = to_u32(In1);
        }
        if (to_u32(In2) > m) {
            m = to_u32(In2);
        }
        if (to_u32(Out) > m) {
            m = to_u32(Out);
        }
        return m + 1;
    }();

    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const { Derived::exec_impl(slot_offset); }
};

template <class Derived, Dst In0, Dst In1, Dst In2, Dst In3, Dst Out>
struct QuaternaryOp : DestOnlyTag {
    static_assert(
        to_u32(In0) < DEST_AUTO_LIMIT && to_u32(In1) < DEST_AUTO_LIMIT && to_u32(In2) < DEST_AUTO_LIMIT &&
            to_u32(In3) < DEST_AUTO_LIMIT && to_u32(Out) < DEST_AUTO_LIMIT,
        "QuaternaryOp: DEST slot exceeds compile-time DEST capacity");
    // NOTE: slot-distinctness is *not* enforced here (mirrors BinaryOp/TernaryOp).
    // SFPU ops typically read all inputs before writing Out, so Out may alias an
    // input slot. Enforce stricter constraints locally if a future op needs them.

    static constexpr Dst in0 = In0;
    static constexpr Dst in1 = In1;
    static constexpr Dst in2 = In2;
    static constexpr Dst in3 = In3;
    static constexpr Dst out = Out;
    static constexpr uint32_t lane_width = []() {
        uint32_t m = to_u32(In0);
        if (to_u32(In1) > m) {
            m = to_u32(In1);
        }
        if (to_u32(In2) > m) {
            m = to_u32(In2);
        }
        if (to_u32(In3) > m) {
            m = to_u32(In3);
        }
        if (to_u32(Out) > m) {
            m = to_u32(Out);
        }
        return m + 1;
    }();

    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const { Derived::exec_impl(slot_offset); }
};

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
//      is_upfront, clashes_with_fpu, hoist_safe, etc.
// =============================================================================

template <
    uint32_t Cb,
    Dst DstSlot = Dst::D0,
    InputLifecycle Policy = InputLifecycle::Streaming,
    OperandKind IndexMode = OperandKind::Scalar,
    CopyTileReconfig Reconfig = CopyTileReconfig::Input,
    TileOffset Offset = TileOffset::Unset>
struct CopyTile;

template <
    uint32_t CbA,
    uint32_t CbB,
    BinaryFpuOp Op = BinaryFpuOp::Add,
    BroadcastDim Bcast = BroadcastDim::None,
    BinaryDataFormatReconfig DfReconfig = BinaryDataFormatReconfig::Input,
    InputLifecycle APolicy = InputLifecycle::Streaming,
    InputLifecycle BPolicy = InputLifecycle::Streaming,
    OperandKind AIndex = OperandKind::Scalar,
    Dst DstSlot = Dst::D0,
    OperandKind BIndex = AIndex,
    TileOffset OffsetA = TileOffset::Unset,
    TileOffset OffsetB = TileOffset::Unset>
struct BinaryFpu;

template <
    uint32_t Cb,
    BinaryFpuOp Op,
    DestReuseType ReuseType,
    Dst DstIn = Dst::D0,
    Dst DstOut = Dst::D0,
    DestReuseReconfig Reconfig = DestReuseReconfig::Input,
    InputLifecycle Policy = InputLifecycle::Streaming,
    OperandKind IndexMode = OperandKind::Scalar,
    TileOffset Offset = TileOffset::Unset>
struct DestReuseBinary;

template <
    BroadcastDim Dim,
    uint32_t Cb,
    Dst DstSlot = Dst::D0,
    InputLifecycle Policy = InputLifecycle::Streaming,
    UnaryBcastReconfig Reconfig = UnaryBcastReconfig::Input>
struct UnaryBcast;

template <
    uint32_t Cb,
    Dst DstSlot = Dst::D0,
    OutputLifecycle Policy = OutputLifecycle::Streaming,
    PackTileReconfig Reconfig = PackTileReconfig::Output,
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

// =============================================================================
// 8. Chain-shape trait predicates (forward declarations — defined in .inl)
// =============================================================================

template <class... Es>
struct EltwiseChain;  // typed list of elements

template <class Chain>
struct chain_has_duplicate_upfront_cbs;
template <class Chain>
struct chain_pack_writes_collide;
template <class Chain>
struct chain_per_side_cbs_consistent;
template <class Chain>
struct chain_math_mop_uniform;
template <class Chain>
struct chain_sfpu_inits_uniform;
template <class Chain>
struct chain_hoist_math_mop;
template <class Chain>
struct chain_hoist_sfpu;

template <class Chain>
inline constexpr bool chain_has_duplicate_upfront_cbs_v = chain_has_duplicate_upfront_cbs<Chain>::value;
template <class Chain>
inline constexpr bool chain_pack_writes_collide_v = chain_pack_writes_collide<Chain>::value;
template <class Chain>
inline constexpr bool chain_per_side_cbs_consistent_v = chain_per_side_cbs_consistent<Chain>::value;
template <class Chain>
inline constexpr bool chain_math_mop_uniform_v = chain_math_mop_uniform<Chain>::value;
template <class Chain>
inline constexpr bool chain_sfpu_inits_uniform_v = chain_sfpu_inits_uniform<Chain>::value;
template <class Chain>
inline constexpr bool chain_hoist_math_mop_v = chain_hoist_math_mop<Chain>::value;
template <class Chain>
inline constexpr bool chain_hoist_sfpu_v = chain_hoist_sfpu<Chain>::value;

// =============================================================================
// 9. Public API — eltwise_chain
// =============================================================================
//
// **Caller-init contract (D8).** The chain helper does NOT wrap any "BIG init"
// (`compute_kernel_hw_startup`, `binary_op_init_common`, `mm_init`, `reduce_init`).
// Engine-wide setup is the caller's responsibility. The chain owns ONLY per-element
// init (`add_tiles_init`, `*_tile_init`, `init_bcast`, `copy_tile_to_dst_init_short`,
// `reconfig_data_format_*`, `tile_regs_*` lifecycle).
//
// **D5 placement.** Caller calls `compute_kernel_hw_startup(cb_a, cb_b, cb_out)` as the
// FIRST statement of `MAIN()` for chains that require it (chains with at least one
// CB-reader and one CB-writer). Multi-stage kernels emit one boot per stage. Mid-`MAIN()`
// placement is undefined per `compute_kernel_hw_startup.h:26-30` (MMIO writes unsafe to
// call mid-kernel).
//
// **D8 grep gate (manual one-liner; full doxygen lives in U6):**
//   grep -nE 'init_common|compute_kernel_hw_startup|mm_init|reduce_init' eltwise_{chain.hpp,chain.inl,block.hpp}
// Result: header `#include` only; zero call sites in helper bodies.

/// Run the chain over an (Ht, Wt) tile grid with optional per-outer-iter block size.
///
/// One entry point covers both 1D and 2D walks via `EltwiseShape`:
///   - `EltwiseShape::tiles(n)`        — 1D, Ht=1, block_size=1
///   - `EltwiseShape::tiles(n, blk)`   — 1D, Ht=1, block_size=blk
///   - `EltwiseShape::grid(H, W)`      — 2D, block_size=1
///   - `EltwiseShape::grid(H, W, blk)` — 2D, block_size=blk
///   - bare `uint32_t n_tiles`         — implicit conversion to `EltwiseShape::tiles(n)`
///
/// Compile-time validation:
///   - illegal `(Policy × IndexMode)` cells static_assert.
///   - duplicate upfront CBs across CB-readers static_assert.
///   - colliding pack writes static_assert.
///   - hoist requested on non-hoist-safe chain static_assert.
///
/// Block-mode auto-detection: if any element in `Es...` exposes `is_upfront == true`,
/// the helper takes the upfront-block path (wait N upfront, loop, pop N at end).
/// InputLifecycle::Streaming CB-reader chains silently clamp `block_size` to 1 via `if constexpr` —
/// query `chain_supports_block_v<Chain>` and `chain_max_block_v<Chain>` at the call
/// site if you want a build-time check on the block choice.
///
/// Index-mode semantics:
///   - `BlockIter` → `ht * Wt + wt`     (window = Ht*Wt)
///   - `RowBcast`  → `wt`                (window = Wt)
///   - `ColBcast`  → `ht`                (window = Ht)
///   - `FirstTile` → 0                   (window = 1)
///
/// `RowBcast`/`ColBcast` require non-streaming CB policy (Upfront, Cumulative,
/// NoWait* / WaitNoPop / InputLifecycle::NoWaitPop) — caller stages broadcast operand tiles before
/// the chain starts.
template <class... Es>
ALWI void eltwise_chain(EltwiseShape shape, Es... elts);

}  // namespace compute_kernel_lib

// Bring the implementation in.
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.inl"
