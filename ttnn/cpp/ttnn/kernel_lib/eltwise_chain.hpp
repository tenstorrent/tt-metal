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
 *   - input-side and pack-side dtype reconfig via per-element policy enums,
 *   - compile-time invariant checks (illegal lifecycle/index combos, duplicate upfront CBs,
 *     pack collisions, hoist-safety).
 *
 * The chain does NOT emit any deprecated dst-sync (`acquire_dst`/`release_dst`) — modern only.
 *
 * Worked examples
 * ---------------
 *
 *   // Streaming unary — Exp(x) → out
 *   eltwise_chain(
 *       CopyTile<cb_in,  Dst::D0, CopyTilePolicy::WaitAndPop>{},
 *       Exp<>{},
 *       PackTile<cb_out, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{}
 *   );
 *
 *   // Streaming binary — A + B → out
 *   eltwise_chain(
 *       CopyTile<cb_a, Dst::D0, CopyTilePolicy::WaitAndPop>{},
 *       CopyTile<cb_b, Dst::D1, CopyTilePolicy::WaitAndPop>{},
 *       BinaryFpu<cb_a, cb_b, BinaryFpuOp::Add>{},
 *       PackTile<cb_out, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{}
 *   );
 *
 *   // Fan-out — same input, two outputs
 *   eltwise_chain(
 *       CopyTile<cb_in, Dst::D0, CopyTilePolicy::WaitNoPop>{},
 *       CopyTile<cb_in, Dst::D1, CopyTilePolicy::NoWaitPop>{},
 *       Exp<Approx::Exact, Approx::Fast, Dst::D0>{},
 *       Tanh<Dst::D1>{},
 *       PackTile<cb_out_a, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{},
 *       PackTile<cb_out_b, Dst::D1, PackTilePolicy::PerTileReserveAndPush>{}
 *   );
 *
 *   // Block reduction with upfront reserve / pop-at-end
 *   eltwise_chain<EltwiseChainOptions{ .upfront_block_size = 64 }>(
 *       CopyTile<cb_in, Dst::D0, CopyTilePolicy::WaitUpfrontPopAtEnd, CbIndexMode::BlockIter>{},
 *       Exp<>{},
 *       PackTile<cb_out, Dst::D0, PackTilePolicy::UpfrontReservePushAtEnd, PackTileIndexMode::BlockIter>{}
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
 * Reconfig (`with_dt_tree`-style)
 * --------------------------------
 *  - CopyTileReconfig::Input         → reconfig_data_format_srca(old, new) + copy_tile init.
 *  - BinaryDataFormatReconfig::INPUT → reconfig_data_format_srca / _srcb (per side).
 *  - BinaryDataFormatReconfig::OUTPUT → pack_reconfig_data_format(old, new).
 *  - DestReuseReconfig::Input        → srca OR srcb reconfig (per ReuseType).
 *  - PackTileReconfig::Output        → pack_reconfig_data_format(new_cb).
 *  - PackTileReconfig::OutputConditional → pack_reconfig_data_format(old_cb, new_cb).
 *  - UnaryBcastReconfig::Input       → reconfigure_unary_bcast(old, new, old_ocb, new_ocb).
 *
 * The combined `reconfig_data_format(srca, srcb)` overloads expand to the same two MOPs that
 * `reconfig_data_format_srca` + `reconfig_data_format_srcb` issue independently, so the helper
 * picks the per-side variant when only one operand changes dtype.
 */

#include <cstdint>
#include <type_traits>
#include <utility>

#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

// LLK / compute-API includes consumed by the inline implementation.
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

/// Element reads ≥1 CB. Provides default trait values; concrete elements override.
struct CbReaderTag {
    static constexpr uint32_t pack_cb_id() { return 0; }
};
/// Element writes to a CB. Provides default trait values; concrete elements override.
struct CbWriterTag {
    static constexpr uint32_t cb_a_id() { return 0; }
    static constexpr uint32_t cb_b_id() { return 0; }
};
/// Element neither reads nor writes a CB (DEST-internal). Provides default trait values
/// so SFPU/Fill/Rand op-structs don't have to declare them per derived type.
struct DestOnlyTag {
    static constexpr bool is_upfront = false;
    static constexpr bool clashes_with_fpu = false;
    static constexpr uint32_t cb_a_id() { return 0; }
    static constexpr uint32_t cb_b_id() { return 0; }
    static constexpr uint32_t pack_cb_id() { return 0; }
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
//  No CB lifecycle to validate (pure DEST internal)            | is_dest_only_op_v
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

// =============================================================================
// 4. Policy enums — CB lifecycle, indexing, reconfig, broadcast
// =============================================================================

/// CB-input lifecycle (CopyTile, BinaryFpu A/B operands, DestReuseBinary, UnaryBcast).
enum class CopyTilePolicy : uint8_t {
    WaitAndPop,           // per-tile wait + per-tile pop  (default — streaming)
    WaitNoPop,            // per-tile wait + no pop        (fan-out first / persistent)
    NoWaitPop,            // no wait     + per-tile pop    (fan-out last / pre-waited single)
    NoWaitNoPop,          // no wait     + no pop          (caller owns lifecycle / sharded)
    WaitUpfrontPopAtEnd,  // upfront wait + upfront pop    (block access — BlockIter / Absolute legal)
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

/// FPU binary output sync policy.
enum class BinaryFpuOutputPolicy : uint8_t {
    PerTile,              // default — release/acquire each tile
    HoistAcquireRelease,  // single acquire/release wraps the whole loop
};

/// FPU binary dtype-reconfig.
enum class BinaryDataFormatReconfig : uint8_t {
    None,
    Input,           // srca and/or srcb on entry
    Output,          // pack reconfig on entry
    InputAndOutput,  // both — default (safest, no skip)
};

/// FPU broadcast dimension. Caller MUST pass explicitly — no inference.
/// Mirrors `ckernel::BroadcastType` values (NONE=0, COL=1, ROW=2, SCALAR=3).
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
// 5. EltwiseChainOptions — compile-time NTTP carrying chain-wide knobs
// =============================================================================

struct EltwiseChainOptions {
    bool enable_fp32_dest_acc = false;
    uint32_t upfront_block_size = 0;  // > 0 enables UpfrontReservePushAtEnd / WaitUpfrontPopAtEnd policies.
    // dst sync is always modern — no field, no enum.
};

// =============================================================================
// 6. CRTP bases — UnaryOp / BinaryOp / TernaryOp / QuaternaryOp
// =============================================================================
//
// Derived structs supply `init()` and `call()`. The base provides slot fields,
// `static_assert` guards on slot ranges + distinctness, `apply() = init() + exec()`,
// and the DestOnlyTag inheritance.
//
// Example:
//
//   template <Approx A = Approx::Exact, Approx F = Approx::Fast, Dst Slot = Dst::D0>
//   struct Exp : UnaryOp<Exp<A, F, Slot>, Slot> {
//       static void init()              { exp_tile_init<A == Approx::Fast, F == Approx::Fast>(); }
//       static void call(uint32_t idst) { exp_tile<A == Approx::Fast, F == Approx::Fast>(idst); }
//   };

template <class Derived, Dst Slot>
struct UnaryOp : DestOnlyTag {
    static_assert(
        to_u32(Slot) < DEST_AUTO_LIMIT, "UnaryOp: DEST slot exceeds compile-time DEST capacity (DEST_AUTO_LIMIT)");

    static constexpr Dst dst_idx = Slot;
    static constexpr uint32_t max_dst() { return to_u32(Slot); }

    /// init() + exec()
    static ALWI void apply() {
        Derived::init();
        exec();
    }

    /// exec() runs Derived::call() on the bound DEST slot.
    static ALWI void exec() { Derived::call(to_u32(Slot)); }
};

template <class Derived, Dst In0, Dst In1, Dst Out>
struct BinaryOp : DestOnlyTag {
    static_assert(
        to_u32(In0) < DEST_AUTO_LIMIT && to_u32(In1) < DEST_AUTO_LIMIT && to_u32(Out) < DEST_AUTO_LIMIT,
        "BinaryOp: DEST slot exceeds compile-time DEST capacity (DEST_AUTO_LIMIT)");
    static_assert(In0 != In1 && In0 != Out && In1 != Out, "BinaryOp slots must be distinct");

    static constexpr Dst in0 = In0;
    static constexpr Dst in1 = In1;
    static constexpr Dst out = Out;
    static constexpr uint32_t max_dst() {
        uint32_t a = to_u32(In0), b = to_u32(In1), c = to_u32(Out);
        return a > b ? (a > c ? a : c) : (b > c ? b : c);
    }

    static ALWI void apply() {
        Derived::init();
        exec();
    }
    static ALWI void exec() { Derived::call(to_u32(In0), to_u32(In1), to_u32(Out)); }
};

template <class Derived, Dst In0, Dst In1, Dst In2, Dst Out>
struct TernaryOp : DestOnlyTag {
    static_assert(
        to_u32(In0) < DEST_AUTO_LIMIT && to_u32(In1) < DEST_AUTO_LIMIT && to_u32(In2) < DEST_AUTO_LIMIT &&
            to_u32(Out) < DEST_AUTO_LIMIT,
        "TernaryOp: DEST slot exceeds compile-time DEST capacity (DEST_AUTO_LIMIT)");
    static_assert(
        In0 != In1 && In0 != In2 && In0 != Out && In1 != In2 && In1 != Out && In2 != Out,
        "TernaryOp input slots must be distinct");

    static constexpr Dst in0 = In0;
    static constexpr Dst in1 = In1;
    static constexpr Dst in2 = In2;
    static constexpr Dst out = Out;

    static ALWI void apply() {
        Derived::init();
        exec();
    }
    static ALWI void exec() { Derived::call(to_u32(In0), to_u32(In1), to_u32(In2), to_u32(Out)); }
};

template <class Derived, Dst In0, Dst In1, Dst In2, Dst In3, Dst Out>
struct QuaternaryOp : DestOnlyTag {
    static_assert(
        to_u32(In0) < DEST_AUTO_LIMIT && to_u32(In1) < DEST_AUTO_LIMIT && to_u32(In2) < DEST_AUTO_LIMIT &&
            to_u32(In3) < DEST_AUTO_LIMIT && to_u32(Out) < DEST_AUTO_LIMIT,
        "QuaternaryOp: DEST slot exceeds compile-time DEST capacity");
    static_assert(
        In0 != In1 && In0 != In2 && In0 != In3 && In0 != Out && In1 != In2 && In1 != In3 && In1 != Out && In2 != In3 &&
            In2 != Out && In3 != Out,
        "QuaternaryOp input slots must be distinct");

    static constexpr Dst in0 = In0;
    static constexpr Dst in1 = In1;
    static constexpr Dst in2 = In2;
    static constexpr Dst in3 = In3;
    static constexpr Dst out = Out;

    static ALWI void apply() {
        Derived::init();
        exec();
    }
    static ALWI void exec() { Derived::call(to_u32(In0), to_u32(In1), to_u32(In2), to_u32(In3), to_u32(Out)); }
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
    CopyTileReconfig Reconfig = CopyTileReconfig::None,
    uint32_t OldCb = 0>
struct CopyTile;

template <
    uint32_t CbA,
    uint32_t CbB,
    BinaryFpuOp Op,
    BroadcastDim Bcast = BroadcastDim::None,
    BinaryFpuOutputPolicy OutPolicy = BinaryFpuOutputPolicy::PerTile,
    BinaryDataFormatReconfig DfReconfig = BinaryDataFormatReconfig::InputAndOutput,
    CopyTilePolicy APolicy = CopyTilePolicy::WaitAndPop,
    CopyTilePolicy BPolicy = CopyTilePolicy::WaitAndPop,
    CbIndexMode AIndex = CbIndexMode::FirstTile,
    CbIndexMode BIndex = CbIndexMode::FirstTile,
    Dst DstSlot = Dst::D0,
    uint32_t OldCbA = 0,
    uint32_t OldCbB = 0,
    uint32_t OldCbOut = 0,
    uint32_t CbOut = 0>
struct BinaryFpu;

template <
    uint32_t Cb,
    BinaryFpuOp Op,
    DestReuseType ReuseType,
    Dst DstIn = Dst::D0,
    Dst DstOut = Dst::D0,
    DestReuseReconfig Reconfig = DestReuseReconfig::None,
    CopyTilePolicy Policy = CopyTilePolicy::WaitAndPop,
    CbIndexMode IndexMode = CbIndexMode::FirstTile,
    uint32_t OldCb = 0>
struct DestReuseBinary;

template <
    BroadcastDim Dim,
    uint32_t Cb,
    uint32_t CbOut = 0,
    Dst DstSlot = Dst::D0,
    CopyTilePolicy Policy = CopyTilePolicy::WaitAndPop,
    UnaryBcastReconfig Reconfig = UnaryBcastReconfig::None,
    uint32_t OldCb = 0,
    uint32_t OldCbOut = 0>
struct UnaryBcast;

template <
    uint32_t Cb,
    Dst DstSlot = Dst::D0,
    PackTilePolicy Policy = PackTilePolicy::PerTileReserveAndPush,
    PackTileIndexMode IndexMode = PackTileIndexMode::FirstTile,
    PackTileReconfig Reconfig = PackTileReconfig::None,
    uint32_t OldCb = 0>
struct PackTile;

template <
    uint32_t Cb,
    Dst FirstSlot,
    uint32_t NTiles,
    PackTilePolicy Policy = PackTilePolicy::PerTileReserveAndPush,
    PackTileReconfig Reconfig = PackTileReconfig::None,
    uint32_t OldCb = 0>
struct PackTileBlock;

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
// 9. Public API — eltwise_pipeline_init + eltwise_chain
// =============================================================================

/// One-time hardware boot per chain shape. Standardises on `compute_kernel_hw_startup` per HQ rule.
/// `Chain` is `EltwiseChain<E0, E1, ...>` — usually deduced from a sample chain instance via
/// the `eltwise_pipeline_init_for(chain)` helper.
template <class Chain>
ALWI void eltwise_pipeline_init();

/// Convenience: deduce the chain type from a sample chain instance.
template <class... Es>
ALWI void eltwise_pipeline_init_for(const EltwiseChain<Es...>&) {
    eltwise_pipeline_init<EltwiseChain<Es...>>();
}

/// Run the chain over `n_tiles` iterations.
///
/// Compile-time validation:
///   - illegal `(Policy × IndexMode)` cells static_assert.
///   - duplicate upfront CBs across CB-readers static_assert.
///   - colliding pack writes static_assert.
///   - hoist requested on non-hoist-safe chain static_assert.
template <EltwiseChainOptions Opts = EltwiseChainOptions{}, class... Es>
ALWI void eltwise_chain(uint32_t n_tiles, Es... elts);

}  // namespace compute_kernel_lib

// Bring the implementation in.
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.inl"
