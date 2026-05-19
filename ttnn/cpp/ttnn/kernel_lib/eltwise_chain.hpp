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
 * @section deduced_wrapper eltwise_chain_with_init — single-stage convenience (U4)
 *
 * `eltwise_chain_with_init(num_tiles, elts...)` deduces `(cb_a, cb_b, cb_out)` from the chain
 * element pack at compile time and emits `compute_kernel_hw_startup` before the chain. **Use only
 * for single-stage kernels.** Multi-stage (different PACK output CB per stage) MUST keep the
 * explicit per-stage `compute_kernel_hw_startup` pattern.
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
 *   // Streaming unary — Exp(x) → out
 *   eltwise_chain(num_tiles,
 *       CopyTile<cb_in,  Dst::D0, CopyTilePolicy::WaitAndPop>{},
 *       Exp<>{},
 *       PackTile<cb_out, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{}
 *   );
 *
 *   // Streaming binary — A + B → out
 *   eltwise_chain(num_tiles,
 *       BinaryFpu<cb_a, cb_b, BinaryFpuOp::Add>{},
 *       PackTile<cb_out, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{}
 *   );
 *
 *   // Single-stage with deduced wrapper — U4
 *   eltwise_chain_with_init(num_tiles,
 *       CopyTile<cb_in, Dst::D0, CopyTilePolicy::WaitAndPop>{},
 *       Exp<>{},
 *       PackTile<cb_out, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{}
 *   );
 *
 *   // Fan-out — same input, two outputs
 *   eltwise_chain(num_tiles,
 *       CopyTile<cb_in, Dst::D0, CopyTilePolicy::WaitNoPop>{},
 *       CopyTile<cb_in, Dst::D1, CopyTilePolicy::NoWaitPop>{},
 *       Exp<Approx::Exact, Approx::Fast, Dst::D0>{},
 *       Tanh<Dst::D1>{},
 *       PackTile<cb_out_a, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{},
 *       PackTile<cb_out_b, Dst::D1, PackTilePolicy::PerTileReserveAndPush>{}
 *   );
 *
 *   // Block reduction with upfront reserve / pop-at-end (auto-detected via `Es::is_upfront`)
 *   eltwise_chain(num_tiles,
 *       CopyTile<cb_in, Dst::D0, CopyTilePolicy::WaitUpfrontPopAtEnd, CbIndexMode::BlockIter>{},
 *       Exp<>{},
 *       PackTile<cb_out, Dst::D0, PackTilePolicy::UpfrontReservePushAtEnd, PackTileIndexMode::BlockIter>{}
 *   );
 *
 *   // Asymmetric bcast walk — A streams the tile range, B pinned at tile 0
 *   //   (softmax-style: out[t] = exp(in[t] - max), max pinned at tile 0)
 *   //   BinaryFpu's 8th template arg is AIndex; 10th (trailing) is BIndex (defaults to AIndex).
 *   eltwise_chain(num_tiles,
 *       BinaryFpu<cb_in, cb_max, BinaryFpuOp::Sub, BroadcastDim::COL,
 *                 BinaryDataFormatReconfig::None,
 *                 CopyTilePolicy::WaitUpfrontPopAtEnd,   // A: wait N upfront, pop at end
 *                 CopyTilePolicy::WaitNoPop,             // B: wait 1, never pop
 *                 CbIndexMode::BlockIter,                // AIndex — A walks 0..num_tiles-1
 *                 Dst::D0,
 *                 CbIndexMode::FirstTile>{},             // BIndex — B pinned at tile 0
 *       Exp<>{},
 *       PackTile<cb_out, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{}
 *   );
 *
 * Non-goals
 * ---------
 *  - Cumulative wait policy (`cb_wait_front(base + i)`). Out of scope; raw LLK only.
 *  - Mid-loop dtype swaps. Reconfig is entry-time per chain element.
 *  - L1 accumulation (`pack_reconfig_l1_acc`), pack-relu, pack-rows. Future PackTilePolicy extensions.
 *  - Held-DEST patterns. Out of scope (zero TSV evidence).
 *  - `acquire_dst/release_dst` and `ACQ()/REL()` macros — modern dst-sync only. Kernels migrate
 *    their dst-sync as part of adopting the chain.
 *
 * Reconfig (`with_dt_tree`-style) — fold-driven post commits 2-3
 * ----------------------------------------------------------------
 *  - CopyTileReconfig::Input         → fold emits `reconfig_data_format_srca(curr)` (compile-time-elided when prev ==
 * curr).
 *  - BinaryDataFormatReconfig::Input → fold emits `reconfig_data_format_srca / _srcb` per side (compile-time-elided per
 * side). Pack reconfig for binary chains is owned by the downstream PackTile element.
 *  - DestReuseReconfig::Input        → fold emits per-side reconfig (srca OR srcb depending on ReuseType).
 *  - PackTileReconfig::Output        → fold emits `pack_reconfig_data_format(new_cb)`.
 *  - PackTileReconfig::OutputConditional → currently emits same as ::Output; future extension may
 *    select two-arg `pack_reconfig_data_format(prev, curr)` form when prev_pack is known (D7 note).
 *  - UnaryBcastReconfig::Input       → currently bundled into `unary_bcast_init`.
 *
 * The combined `reconfig_data_format(srca, srcb)` overloads expand to the same two MOPs that
 * `reconfig_data_format_srca` + `reconfig_data_format_srcb` issue independently, so the per-side
 * elision in the fold yields the same MOP count as the combined form when both sides change.
 */

#include <cstdint>
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

/// Auto-block toggle (item 2 of eltwise_helper_proposal.md). Chain-wide template
/// parameter on `eltwise_chain<AutoBlock, ...>`. When On, chain computes
/// `BlockSize = DEST_AUTO_LIMIT / chain_lane_width` and runs that many lanes per
/// outer iter (each lane offsets DEST slot by `j * chain_lane_width`). When Off,
/// `BlockSize = 1` — every outer iter processes one tile (today's per-tile shape).
enum class AutoBlock : bool { Off = false, On = true };

// =============================================================================
// 4. Policy enums — CB lifecycle, indexing, reconfig, broadcast
// =============================================================================

/// CB-input lifecycle (CopyTile, BinaryFpu A/B operands, DestReuseBinary, UnaryBcast).
enum class CopyTilePolicy : uint8_t {
    WaitAndPop,              // per-tile wait + per-tile pop  (default — streaming)
    WaitNoPop,               // per-tile wait + no pop        (fan-out first / persistent)
    NoWaitPop,               // no wait     + per-tile pop    (fan-out last / pre-waited single)
    NoWaitNoPop,             // no wait     + no pop          (caller owns lifecycle / sharded)
    WaitUpfrontPopAtEnd,     // upfront wait + upfront pop    (block access — BlockIter / Absolute legal)
    CumulativeWaitPopAtEnd,  // per-iter cumulative wait (cb_wait_front(cb, i+1)) + bulk pop at end
                             // (block access with producer streaming: consumer iter i starts as
                             // soon as producer has pushed i+1 tiles, vs WaitUpfrontPopAtEnd which
                             // blocks iter 0 on the full N. BlockIter / Absolute / Pinned all
                             // legal — cumulative wait guarantees tile i present at iter i.)
};

/// CB-input tile indexing.
enum class CbIndexMode : uint8_t {
    FirstTile,  // always tile 0 of the CB
    BlockIter,  // tile i (loop var). Requires WaitUpfrontPopAtEnd or NoWaitNoPop.
    Pinned,     // fixed runtime k. Under single-tile-window policies, k must be 0.
    Absolute,   // runtime idx ∈ caller's window. Requires WaitUpfrontPopAtEnd or NoWaitNoPop.
};

/// CopyTile dtype-reconfig.
enum class CopyTileReconfig : uint8_t {
    None,   // no reconfig
    Input,  // copy_tile_to_dst_init_short_with_dt(old_cb, new_cb)
};

/// FPU binary op selector.
enum class BinaryFpuOp : uint8_t { Add, Sub, Mul };

/// FPU binary dtype-reconfig.
enum class BinaryDataFormatReconfig : uint8_t {
    None,
    Input,  // srca and/or srcb on entry (default). Pack reconfig is owned by PackTile.
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
enum class DestReuseReconfig : uint8_t {
    None,
    Input,  // srca-or-srcb reconfig per ReuseType
};

/// UnaryBcast reconfig.
enum class UnaryBcastReconfig : uint8_t {
    None,
    Input,  // reconfigure_unary_bcast(old_icb, new_icb, old_ocb, new_ocb)
};

/// Pack-side lifecycle. Five values cover all observed pack patterns from the TSV survey.
enum class PackTilePolicy : uint8_t {
    PerTileReserveAndPush,    // cb_reserve_back(1); pack; cb_push_back(1)              (default)
    PerTileReserveNoPush,     // reserve happens; push deferred to caller
    NoReservePushAtEnd,       // pack into pre-reserved CB; push N at end
    NoReserveNoPush,          // caller owns reserve+push
    UpfrontReservePushAtEnd,  // reserve N upfront; pack sequentially; push N at end
};

/// PackTile output-tile-index mode (mirrors CbIndexMode).
enum class PackTileIndexMode : uint8_t {
    FirstTile,  // always output index 0
    BlockIter,  // i (loop var). Requires UpfrontReservePushAtEnd / NoReserve* with caller-managed window.
    Pinned,     // fixed runtime k.
    Absolute,   // runtime idx.
};

/// Pack-side dtype-reconfig.
enum class PackTileReconfig : uint8_t {
    None,
    Output,             // pack_reconfig_data_format(new_cb)
    OutputConditional,  // pack_reconfig_data_format(old_cb, new_cb)  (FP32_DEST_ACC-gated)
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
    /// is added by the chain to shift DEST writes into lane `j` when auto-block is
    /// on; AutoBlock::Off passes 0 (today's shape).
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
    CopyTilePolicy Policy = CopyTilePolicy::WaitAndPop,
    CbIndexMode IndexMode = CbIndexMode::FirstTile,
    CopyTileReconfig Reconfig = CopyTileReconfig::Input>
struct CopyTile;

template <
    uint32_t CbA,
    uint32_t CbB,
    BinaryFpuOp Op = BinaryFpuOp::Add,
    BroadcastDim Bcast = BroadcastDim::None,
    BinaryDataFormatReconfig DfReconfig = BinaryDataFormatReconfig::Input,
    CopyTilePolicy APolicy = CopyTilePolicy::WaitAndPop,
    CopyTilePolicy BPolicy = CopyTilePolicy::WaitAndPop,
    CbIndexMode AIndex = CbIndexMode::FirstTile,
    Dst DstSlot = Dst::D0,
    CbIndexMode BIndex = AIndex>
struct BinaryFpu;

template <
    uint32_t Cb,
    BinaryFpuOp Op,
    DestReuseType ReuseType,
    Dst DstIn = Dst::D0,
    Dst DstOut = Dst::D0,
    DestReuseReconfig Reconfig = DestReuseReconfig::Input,
    CopyTilePolicy Policy = CopyTilePolicy::WaitAndPop,
    CbIndexMode IndexMode = CbIndexMode::FirstTile>
struct DestReuseBinary;

template <
    BroadcastDim Dim,
    uint32_t Cb,
    uint32_t CbOut = 0,
    Dst DstSlot = Dst::D0,
    CopyTilePolicy Policy = CopyTilePolicy::WaitAndPop,
    UnaryBcastReconfig Reconfig = UnaryBcastReconfig::Input>
struct UnaryBcast;

template <
    uint32_t Cb,
    Dst DstSlot = Dst::D0,
    PackTilePolicy Policy = PackTilePolicy::PerTileReserveAndPush,
    PackTileIndexMode IndexMode = PackTileIndexMode::FirstTile,
    PackTileReconfig Reconfig = PackTileReconfig::Output>
struct PackTile;

template <
    uint32_t Cb,
    Dst FirstSlot,
    uint32_t NTiles,
    PackTilePolicy Policy = PackTilePolicy::PerTileReserveAndPush,
    PackTileReconfig Reconfig = PackTileReconfig::Output>
struct PackTileBlock;

// Fill / Rand forward declarations — implementations live in eltwise_fill.hpp / eltwise_rand.hpp.
template <Dst DstSlot = Dst::D0>
struct FillScalar;
template <DataFormat DF, Dst DstSlot>
struct FillInt;
template <Dst DstSlot = Dst::D0>
struct FillBitcast;
template <Dst DstSlot = Dst::D0, uint32_t Seed = 0>
struct RandTile;

// =============================================================================
// 8. Chain-shape trait predicates (forward declarations — defined in .inl)
// =============================================================================

template <class... Es>
struct EltwiseChain;  // typed list of elements

template <class Chain>
struct chain_has_any_copy_tile;
template <class Chain>
struct chain_has_any_pack_tile;
template <class Chain>
struct chain_has_any_cb_reader;
template <class Chain>
struct chain_has_any_cb_writer;
template <class Chain>
struct chain_has_non_copy_tile_fpu_clash;
template <class Chain>
struct chain_loads_share_cb;
template <class Chain>
struct chain_has_duplicate_upfront_cbs;
template <class Chain>
struct chain_pack_writes_collide;
template <class Chain>
struct chain_is_hoist_safe;

template <class Chain>
inline constexpr bool chain_has_any_copy_tile_v = chain_has_any_copy_tile<Chain>::value;
template <class Chain>
inline constexpr bool chain_has_any_pack_tile_v = chain_has_any_pack_tile<Chain>::value;
template <class Chain>
inline constexpr bool chain_has_any_cb_reader_v = chain_has_any_cb_reader<Chain>::value;
template <class Chain>
inline constexpr bool chain_has_any_cb_writer_v = chain_has_any_cb_writer<Chain>::value;
template <class Chain>
inline constexpr bool chain_has_non_copy_tile_fpu_clash_v = chain_has_non_copy_tile_fpu_clash<Chain>::value;
template <class Chain>
inline constexpr bool chain_loads_share_cb_v = chain_loads_share_cb<Chain>::value;
template <class Chain>
inline constexpr bool chain_has_duplicate_upfront_cbs_v = chain_has_duplicate_upfront_cbs<Chain>::value;
template <class Chain>
inline constexpr bool chain_pack_writes_collide_v = chain_pack_writes_collide<Chain>::value;
template <class Chain>
inline constexpr bool chain_is_hoist_safe_v = chain_is_hoist_safe<Chain>::value;

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

/// Run the chain over `n_tiles` iterations.
///
/// Compile-time validation:
///   - illegal `(Policy × IndexMode)` cells static_assert.
///   - duplicate upfront CBs across CB-readers static_assert.
///   - colliding pack writes static_assert.
///   - hoist requested on non-hoist-safe chain static_assert.
///
/// Block-mode auto-detection: if any element in `Es...` exposes `is_upfront == true`,
/// the helper takes the upfront-block path (wait N upfront, loop, pop N at end).
template <AutoBlock Block = AutoBlock::Off, class... Es>
ALWI void eltwise_chain(uint32_t n_tiles, Es... elts);

/// Run the chain over `n_tiles` iterations, plus emit `compute_kernel_hw_startup`
/// at chain entry with CBs deduced from the chain element pack.
///
/// Single-stage convenience: deduces (cb_a, cb_b, cb_out) by walking `Es...`:
///   - `cb_a` ← first element with `is_cb_reader_op_v` and `cb_a_id() != 0`
///   - `cb_b` ← first element with `is_binary_fpu_op_v` and `cb_b_id() != 0`,
///              else `cb_a` (unary chains)
///   - `cb_out` ← first element with `is_pack_tile_op_v` and `pack_cb_id() != 0`
/// then calls `compute_kernel_hw_startup(cb_a, cb_b, cb_out)` and `eltwise_chain(...)`.
///
/// **Multi-stage caveat (D5).** Use this only for single-stage kernels. Multi-stage
/// kernels (different PACK output CB per stage) MUST keep explicit per-stage
/// `compute_kernel_hw_startup` calls — `eltwise_chain_with_init` would emit it once
/// with stage-1 CBs and stage 2's PACK would target the wrong CB.
template <AutoBlock Block = AutoBlock::Off, class... Es>
ALWI void eltwise_chain_with_init(uint32_t n_tiles, Es... elts);

}  // namespace compute_kernel_lib

// Bring the implementation in.
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.inl"
