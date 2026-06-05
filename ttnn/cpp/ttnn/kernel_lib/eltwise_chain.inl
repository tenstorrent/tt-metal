// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_chain.inl
 * @brief Implementation of the eltwise chain pipeline + chain element types + traits.
 *
 * Included from `eltwise_chain.hpp`. Do NOT include directly.
 */

// Impl-only includes (the public eltwise_chain.hpp surface — element decls + enums — needs
// none of these; they live here, with the implementation that uses them).
#include <tuple>
#include "api/compute/bcast.h"
#include "api/compute/cb_api.h"
#include "api/dataflow/dataflow_buffer.h"  // DataflowBuffer — chain routes CB sync (wait/pop/reserve/push) through it
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"

namespace compute_kernel_lib {

// Internal sentinel + type-list wrapper + chain-shape trait declarations. These are
// implementation detail of the chain pipeline — no chain caller references them, so they
// live here rather than on the public eltwise_chain.hpp surface.
inline constexpr uint32_t INVALID_DFB = 0xFFFFFFFFu;  // "no CB on this slot" (== NO_PREV_DFB); a real
                                                // tt::CBIndex is 0..31, so 0xFFFFFFFF never aliases one.

template <class... Es>
struct EltwiseChain;  // typed list of elements (defined below)

template <class Chain> struct chain_has_duplicate_upfront_dfbs;
template <class Chain> struct chain_pack_writes_collide;
template <class Chain> struct chain_per_side_dfbs_consistent;
template <class Chain> struct chain_math_mop_uniform;
template <class Chain> struct chain_sfpu_inits_uniform;
template <class Chain> struct chain_hoist_math_mop;
template <class Chain> struct chain_hoist_sfpu;

template <class Chain>
inline constexpr bool chain_has_duplicate_upfront_cbs_v = chain_has_duplicate_upfront_dfbs<Chain>::value;
template <class Chain>
inline constexpr bool chain_pack_writes_collide_v = chain_pack_writes_collide<Chain>::value;
template <class Chain>
inline constexpr bool chain_per_side_cbs_consistent_v = chain_per_side_dfbs_consistent<Chain>::value;
template <class Chain>
inline constexpr bool chain_math_mop_uniform_v = chain_math_mop_uniform<Chain>::value;
template <class Chain>
inline constexpr bool chain_sfpu_inits_uniform_v = chain_sfpu_inits_uniform<Chain>::value;
template <class Chain>
inline constexpr bool chain_hoist_math_mop_v = chain_hoist_math_mop<Chain>::value;
template <class Chain>
inline constexpr bool chain_hoist_sfpu_v = chain_hoist_sfpu<Chain>::value;

// =============================================================================
// Marker tag hierarchy (data direction → kind)
// =============================================================================
// Internal classification scaffolding — chain callers never name these. Concrete
// element types (declared on the .hpp surface) inherit the leaf tags; the chain
// pipeline + SFPU/FPU op-helper headers classify elements via the is_*_op_v
// predicates below.

/// Element reads ≥1 CB. Pure marker — concrete elements declare `dfb_a_id()` and
/// (if binary) `dfb_b_id()`. No stub defaults, so a missing accessor is a compile
/// error rather than a silently-wrong default CB id.
struct CbReaderTag {};
/// Element writes to a CB. Pure marker — concrete elements declare `pack_dfb_id()`.
/// No stub defaults.
struct CbWriterTag {};
/// Element neither reads nor writes a CB (DEST-internal). Carries only the
/// `is_upfront` default — no CB-id stubs. The chain pipeline SFINAE-detects
/// `dfb_a_id()` / `dfb_b_id()` / `pack_dfb_id()` on the element directly and never
/// reaches a DestOnlyTag default.
struct DestOnlyTag {
    static constexpr bool is_upfront = false;
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

// Trait predicates — which predicate drives each chain decision:
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
/// from `DestOnlyTag` via `UnaryOp` / `BinaryOp` / `TernaryOp`;
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
/// `DestReuseBinary`, `UnaryBcast`). The hoist gate requires all such
/// elements in a chain (including the CopyTile-versus-FPU init clash) to be
/// the same instantiated type — otherwise the boot-time fold leaves only the
/// last init's MOP programmed and earlier elements run with the wrong MOP.
template <class T>
inline constexpr bool is_math_mop_op_v = is_copy_tile_op_v<T> || is_fpu_kind_op_v<T>;

// =============================================================================
// tile_base_value — orthogonal tile-index offset extractor (impl helper)
// =============================================================================
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

// =============================================================================
// CRTP bases — UnaryOp / BinaryOp / TernaryOp
// =============================================================================
//
// Single dispatch contract: every element exposes `void exec(uint32_t) const`. The bases
// default-forward to a static `exec_impl()` supplied by the derived op; runtime-param ops
// (Power, Hardtanh, …) override `exec` directly to capture instance state. Defining
// neither is a compile error (no silent fallthrough).
//
//   template <Approx A = Approx::Exact, Approx F = Approx::Fast, Dst Slot = Dst::D0>
//   struct Exp : UnaryOp<Exp<A, F, Slot>, Slot> {
//       static void init()       { exp_tile_init<A == Approx::Fast, F == Approx::Fast>(); }
//       static void exec_impl()  { exp_tile<A == Approx::Fast, F == Approx::Fast>(to_u32(Slot)); }
//   };

template <class Derived, Dst Slot>
struct UnaryOp : DestOnlyTag {
    static_assert(
        to_u32(Slot) < DEST_AUTO_LIMIT, "UnaryOp: DEST slot exceeds compile-time DEST capacity (DEST_AUTO_LIMIT)");

    static constexpr Dst dst_idx = Slot;
    static constexpr uint32_t max_dst() { return to_u32(Slot); }
    /// Per-lane DEST footprint. Used by chain to pick auto BlockSize.
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

// =============================================================================
// Compile-time prev-CB tracking — drives the reconfig fold
// =============================================================================
//
// Each element exposes uniform static accessors for the CB it routes to each Side
// (SrcA / SrcB / Pack), letting the pipeline compute the most recent CB on each side
// before a given element. NO_PREV_DFB is the "doesn't touch this side" sentinel.

inline constexpr uint32_t NO_PREV_DFB = 0xFFFFFFFFu;

enum class Side : uint8_t { SrcA, SrcB, Pack };

namespace detail {

// =============================================================================
// A0. 2D index-mode helpers (OperandKind → tile index / upfront window)
//
// Compile-time-elided `idx` / `window` (defined below), inlined by every CB-reader's
// `exec` / `wait_upfront` — `if constexpr` collapses to one arithmetic op at run time.
// `TileBase` layers a runtime offset on top. `is_bcast_mode_v<M>` drives the (Policy ×
// Mode) static_asserts (Row/Col reject streaming policies, as in `binary_op_helpers`).
// =============================================================================

template <OperandKind M>
inline constexpr bool is_bcast_mode_v =
    (M == OperandKind::Row) || (M == OperandKind::Col);

template <OperandKind M>
ALWI constexpr uint32_t idx(uint32_t i_flat, uint32_t ht, uint32_t wt) noexcept {
    if constexpr (M == OperandKind::Scalar) { (void)i_flat; (void)ht; (void)wt; return 0; }
    else if constexpr (M == OperandKind::Block) { (void)ht; (void)wt; return i_flat; }
    else if constexpr (M == OperandKind::Row)  { (void)i_flat; (void)ht; return wt; }
    else                                       { (void)i_flat; (void)wt; return ht; }  // Col
}

template <OperandKind M>
ALWI constexpr uint32_t window(uint32_t Ht, uint32_t Wt) noexcept {
    if constexpr (M == OperandKind::Block) return Ht * Wt;
    else if constexpr (M == OperandKind::Row) { (void)Ht; return Wt; }
    else if constexpr (M == OperandKind::Col) { (void)Wt; return Ht; }
    else                                      { (void)Ht; (void)Wt; return 1u; }  // Scalar
}

// Allowed (Policy × Mode) combinations. Row/Col cannot stream per-tile —
// the producer must stage the full row/col upfront. Matches the
// `binary_op_helpers` static_assert (ROW/SCALAR require InputLifecycle::Bulk-family or NoWait*).
template <InputLifecycle P, OperandKind M>
inline constexpr bool valid_policy_mode_v =
    !(is_bcast_mode_v<M> && (P == InputLifecycle::Streaming || P == InputLifecycle::Chunked));

// =============================================================================
// A. Chain typed-list machinery
// =============================================================================

template <class... Es>
struct ElementList {
    static constexpr size_t size = sizeof...(Es);
};

// Build an `EltwiseChain<...>` from a parameter pack. (Defined in the public namespace.)
template <class... Es>
ALWI constexpr auto make_chain(Es...) -> EltwiseChain<Es...> { return {}; }

// Fold helper: `if constexpr` over each element with a callable.
template <class F, class... Es>
ALWI constexpr void for_each_element(F&& f) {
    (F::template apply_one<Es>(f), ...);
}

// Disjunction over a pack with a predicate template.
template <template <class> class Pred, class... Es>
struct any_v_helper : std::bool_constant<(Pred<Es>::value || ...)> {};

template <template <class> class Pred, class... Es>
struct count_v_helper : std::integral_constant<size_t, (size_t{Pred<Es>::value} + ...)> {};

// =============================================================================
// B. Static cb-id / dst-slot extraction predicates per element
//
// Every CB-reader element must expose:
//   static constexpr uint32_t dfb_a_id();             // primary CB
//   static constexpr uint32_t dfb_b_id();             // secondary CB or 0 if N/A
//   static constexpr InputLifecycle a_policy();
//   static constexpr InputLifecycle b_policy();
// (default impls below cover non-CB-reader elements.)
//
// Every CB-writer element must expose:
//   static constexpr uint32_t pack_dfb_id();
//   static constexpr Dst pack_dst_slot();
//   static constexpr uint32_t pack_output_index();   // runtime fallback OK; used only for index mode FirstTile/Pinned k
// =============================================================================

template <class T, class = void> struct has_dfb_a    : std::false_type {};
template <class T> struct has_dfb_a<T, std::void_t<decltype(T::dfb_a_id())>> : std::true_type {};

template <class T, class = void> struct has_dfb_b    : std::false_type {};
template <class T> struct has_dfb_b<T, std::void_t<decltype(T::dfb_b_id())>> : std::true_type {};

template <class T, class = void> struct has_pack_dfb : std::false_type {};
template <class T> struct has_pack_dfb<T, std::void_t<decltype(T::pack_dfb_id())>> : std::true_type {};

// Forward declarations — defined below (used by reader_pair_collide / writer_pair_collide
// earlier in the file's flow).
template <class E> constexpr uint32_t dfb_a_of();
template <class E> constexpr uint32_t dfb_b_of();
template <class E> constexpr uint32_t pack_dfb_of();

// ChainTraits<Es...> — the one value-reflection aggregate for the whole chain (defined
// once below, after the per-element accessors it reads). Forward-declared here so the
// trait wrappers (chain_lane_width etc.) can name it.
template <class... Es> struct ChainTraits;

// =============================================================================
// Per-Side prev-CB SFINAE probe
//
// `dfb_for_side<Side, E>` reads `E::reconfig_srca_dfb` / `_srcb_cb` / `_pack_cb`
// when present, returns `NO_PREV_DFB` otherwise. Elements that don't declare the
// accessors still participate in the fold transparently (treated as no-prev).
// =============================================================================

template <class E, class = void>
struct has_reconfig_srca : std::false_type {};
template <class E>
struct has_reconfig_srca<E, std::void_t<decltype(E::reconfig_srca_dfb)>> : std::true_type {};

template <class E, class = void>
struct has_reconfig_srcb : std::false_type {};
template <class E>
struct has_reconfig_srcb<E, std::void_t<decltype(E::reconfig_srcb_dfb)>> : std::true_type {};

template <class E, class = void>
struct has_reconfig_pack : std::false_type {};
template <class E>
struct has_reconfig_pack<E, std::void_t<decltype(E::reconfig_pack_dfb)>> : std::true_type {};

template <Side S, class E>
constexpr uint32_t dfb_for_side() {
    if constexpr (S == Side::SrcA) {
        if constexpr (has_reconfig_srca<E>::value) return E::reconfig_srca_dfb;
        else                                       return NO_PREV_DFB;
    } else if constexpr (S == Side::SrcB) {
        if constexpr (has_reconfig_srcb<E>::value) return E::reconfig_srcb_dfb;
        else                                       return NO_PREV_DFB;
    } else {  // Pack
        if constexpr (has_reconfig_pack<E>::value) return E::reconfig_pack_dfb;
        else                                       return NO_PREV_DFB;
    }
}

// Per-side prev-CB history, last opt-in pack CB, and heterogeneous-pack detection are
// single-sweep fields on `ChainTraits` (prev / last_pack_cb / pack_hetero), computed
// once from the reflected ElemDesc array.

}  // namespace detail

// =============================================================================
// 1. CopyTile chain element
// =============================================================================

template <uint32_t Cb,
          Dst DstSlot,
          InputLifecycle Policy,
          CopyTileReconfig Reconfig,
          OperandKind IndexMode,
          TileOffset Offset>
struct CopyTile : CopyTileTag {
    // ---- compile-time validation ----
    static_assert(to_u32(DstSlot) < DEST_AUTO_LIMIT,
                  "CopyTile: DEST slot exceeds DEST_AUTO_LIMIT");
    // Comprehensive (IndexMode, Policy) legality. Block rejects PerTile-pop
    // (InputLifecycle::Streaming/InputLifecycle::BulkDrain/InputLifecycle::NoWaitPop — absolute-index pitfall) and PerTile-wait-of-1
    // (InputLifecycle::HeldStream — never tracks per-iter requirement). Scalar/Row/Col accept every
    // legal lifecycle — caller-sized.
    static_assert(is_legal_kind_lifecycle(IndexMode, Policy),
                  "CopyTile: (IndexMode, Policy) is illegal for Block — exclude "
                  "InputLifecycle::Streaming / InputLifecycle::HeldStream / InputLifecycle::BulkDrain / InputLifecycle::NoWaitPop on Block walkers.");
    // 2D: RowBcast / ColBcast require non-streaming policy (matches binary_op_helpers ROW/SCALAR rule).
    static_assert(detail::valid_policy_mode_v<Policy, IndexMode>,
                  "CopyTile: RowBcast / ColBcast index require non-streaming policy "
                  "(WaitUpfrontPopAtEnd, WaitNoPop, InputLifecycle::NoWaitPop, NoWaitNoPop, CumulativeWaitPopAtEnd)");
    // TileOffset::Set requires InputLifecycle::Bulk-family / InputLifecycle::CallerManaged lifecycle — iter-dependent
    // counts (InputLifecycle::Streaming/InputLifecycle::Chunked/Cumulative/Held{Stream,Cumulative}/InputLifecycle::NoWaitPop) can't
    // compose with runtime base offsets. Caller must size CB to base+window.
    static_assert(Offset == TileOffset::Unset || is_legal_input_lifecycle_with_base(Policy),
                  "CopyTile: TileOffset::Set requires InputLifecycle::Bulk-family or InputLifecycle::CallerManaged lifecycle "
                  "(InputLifecycle::Bulk / InputLifecycle::HeldBulk / InputLifecycle::DeferredPop / InputLifecycle::BulkDrain / InputLifecycle::CallerManaged)");

    static constexpr uint32_t dfb             = Cb;
    static constexpr uint32_t       dfb_a_id()       { return Cb; }
    // CopyTile reads one CB front (srcA via dfb_a_id); dfb_b / b_policy absent -> defaults apply.
    static constexpr InputLifecycle a_policy()      { return Policy; }
    static constexpr bool           is_upfront      = (Policy == InputLifecycle::Bulk) ||
                                                      (Policy == InputLifecycle::HeldBulk) ||
                                                      (Policy == InputLifecycle::Pipelined);

    // Prev-CB fold: CopyTile loads CbA only. srcb/pack sides are absent -> dfb_for_side
    // defaults them to NO_PREV_DFB.
    static constexpr uint32_t       reconfig_srca_dfb = (Reconfig == CopyTileReconfig::Input) ? Cb : NO_PREV_DFB;

    uint32_t tile_base = 0;

    constexpr CopyTile() noexcept = default;
    constexpr explicit CopyTile(uint32_t base) noexcept : tile_base(base) {}

    // ---- chain pipeline hooks ----
    static ALWI void init() {
        copy_tile_init(Cb);
    }

    /// Per-iter wait. Element fires at its own granularity — streaming policies wait 1
    /// per iter, Cumulative grows wait count with i (i+1), Upfront fires once via
    /// wait_upfront with full n_tiles. None scale by chain block_size — block_size
    /// only drives the inner DEST-lane loop and slot_offset.
    ALWI void wait_per_tile(uint32_t cumulative_count) const {
        if constexpr (Policy == InputLifecycle::Streaming || Policy == InputLifecycle::HeldStream) {
            DataflowBuffer(Cb).wait_front(1);
        } else if constexpr (Policy == InputLifecycle::Pipelined ||
                             Policy == InputLifecycle::HeldCumulative) {
            DataflowBuffer(Cb).wait_front(cumulative_count);
        }
    }

    /// Per-outer-iter wait of `inner_count` tiles (chunked streaming).
    /// inner_count == BlockSize for steady iters, == tail size for the last iter.
    ALWI void wait_per_block(uint32_t inner_count) const {
        if constexpr (Policy == InputLifecycle::Chunked) {
            DataflowBuffer(Cb).wait_front(inner_count);
        }
    }



    // 2D variants — Ht/Wt-aware. Routes through `idx` and `window`; TileBase
    // adds the runtime offset on top. InputLifecycle::Streaming policies handled by the same
    // `wait_per_tile` / `pop_per_tile` as 1D.
    ALWI void wait_upfront(uint32_t Ht, uint32_t Wt) const {
        if constexpr (Policy == InputLifecycle::Bulk ||
                      Policy == InputLifecycle::HeldBulk ||
                      Policy == InputLifecycle::BulkDrain) {
            DataflowBuffer(Cb).wait_front(detail::window<IndexMode>(Ht, Wt) + tile_base_value<Offset>(tile_base));
        }
    }

    ALWI void exec(uint32_t i_flat, uint32_t ht, uint32_t wt, uint32_t slot_offset) const {
        const uint32_t in_idx = tile_base_value<Offset>(tile_base) + detail::idx<IndexMode>(i_flat, ht, wt);
        copy_tile(Cb, in_idx, to_u32(DstSlot) + slot_offset);
    }

    ALWI void pop_upfront_end(uint32_t Ht, uint32_t Wt) const {
        if constexpr (Policy == InputLifecycle::Bulk ||
                      Policy == InputLifecycle::Pipelined ||
                      Policy == InputLifecycle::DeferredPop) {
            DataflowBuffer(Cb).pop_front(detail::window<IndexMode>(Ht, Wt) + tile_base_value<Offset>(tile_base));
        }
    }

    static constexpr uint32_t lane_width = to_u32(DstSlot) + 1;

    ALWI void pop_per_tile(uint32_t /*i*/) const {
        if constexpr (Policy == InputLifecycle::Streaming ||
                      Policy == InputLifecycle::NoWaitPop ||
                      Policy == InputLifecycle::BulkDrain) {
            DataflowBuffer(Cb).pop_front(1);
        }
    }

    /// Per-outer-iter pop of `inner_count` tiles (chunked streaming).
    ALWI void pop_per_block(uint32_t inner_count) const {
        if constexpr (Policy == InputLifecycle::Chunked) {
            DataflowBuffer(Cb).pop_front(inner_count);
        }
    }

};

// =============================================================================
// 2. PackTile chain element
// =============================================================================

template <uint32_t Cb,
          OutputLifecycle Policy,
          PackTileReconfig Reconfig,
          Dst DstSlot,
          TileOffset Offset>
struct PackTile : PackTileTag {
    static_assert(to_u32(DstSlot) < DEST_AUTO_LIMIT,
                  "PackTile: DEST slot exceeds DEST_AUTO_LIMIT");
    // TileBase != None on pack side requires caller-managed-style lifecycle on the
    // output CB (caller pre-reserved a window large enough for base + kind window).
    // InputLifecycle::Streaming / InputLifecycle::Chunked reserve+push counts can't be inflated by a runtime base
    // without per-iter bookkeeping the chain doesn't own.
    static_assert(Offset == TileOffset::Unset || is_legal_output_lifecycle_with_base(Policy),
                  "PackTile: TileOffset::Set requires InputLifecycle::Bulk-family or OutputLifecycle::CallerManaged lifecycle "
                  "(OutputLifecycle::Bulk / OutputLifecycle::DeferredReserve / OutputLifecycle::HeldReserve / OutputLifecycle::CallerManaged)");

    static constexpr uint32_t  dfb                 = Cb;
    static constexpr uint32_t          pack_dfb_id()        { return Cb; }
    static constexpr Dst               pack_dst_slot       = DstSlot;
    static constexpr bool              is_upfront          = (Policy == OutputLifecycle::Bulk);
    static constexpr bool              uses_per_block_pack = (Policy == OutputLifecycle::Chunked);
    // Walk vs pinned output addressing is DERIVED from the OutputLifecycle (no caller knob):
    // the upfront-reserve policies (Bulk, BulkReservePerTile, BulkReservePerChunk) reserve the
    // whole window once and write distinct tiles into it (walk); every per-tile/per-chunk-reserve
    // policy advances the CB front itself, so the write index stays pinned at base. (For a 1-tile
    // output, walk and pinned are identical: base + 0 == base.)
    static constexpr bool              walk                = (Policy == OutputLifecycle::Bulk) ||
                                                             (Policy == OutputLifecycle::BulkReservePerTile) ||
                                                             (Policy == OutputLifecycle::BulkReservePerChunk);

    // Prev-CB fold: PackTile writes pack-side; mark Cb under reconfig only when
    // the user opted into pack reconfig (Output). Otherwise no pack reconfig is
    // emitted — fold keeps prior pack target.
    // srca/srcb absent -> dfb_for_side defaults them to NO_PREV_DFB; PackTile programs pack only.
    static constexpr uint32_t          reconfig_pack_dfb    =
        (Reconfig == PackTileReconfig::Output) ? Cb : NO_PREV_DFB;

    uint32_t tile_base = 0;

    constexpr PackTile() noexcept = default;
    constexpr explicit PackTile(uint32_t base) noexcept : tile_base(base) {}

    static ALWI void init() {
        // Pack reconfig is fold-driven (compile-time-elided when prev_pack_cb == Cb).
        // The chain emits the reconfig in `emit_pre_element_transitions()` before this
        // element runs; init() here is a no-op for reconfig.
        // Retained empty so trait-dispatch stays uniform.
    }

    ALWI void reserve_per_tile(uint32_t /*i*/) const {
        if constexpr (Policy == OutputLifecycle::Streaming ||
                      Policy == OutputLifecycle::HeldReserve) {
            DataflowBuffer(Cb).reserve_back(1);
        }
    }

    /// Per-outer-iter reserve of `inner_count` tiles (chunked streaming).
    ALWI void reserve_per_block(uint32_t inner_count) const {
        if constexpr (Policy == OutputLifecycle::Chunked) {
            DataflowBuffer(Cb).reserve_back(inner_count);
        }
    }



    // Pack exec — walk the reserved output window (base + i_flat) for OutputLifecycle::Bulk, or stay pinned
    // at base for per-tile/chunk policies whose CB front already advanced. TileOffset adds base.
    //
    // OOO gating: the LLK's sequential pack path (out_of_order_output=false) derives its write
    // address from an internal running `fifo_wr_tile_ptr` and IGNORES `out_idx` entirely. That is
    // correct only when the intended index coincides with the sequential counter — i.e. when there
    // is no base offset (walk: 0,1,2,…; pinned: 0). The moment `Offset == Set`, `out_idx` carries a
    // non-coincident base that the sequential path would silently drop (data lands at index 0, not
    // base). So we switch to `pack_tile<true>` for `TileOffset::Set`, which honors `out_idx`
    // (addr = fifo_wr_ptr + page_size*out_idx - 1) without advancing the internal counter — exactly
    // matching the explicit `base + i_flat` we pass each iteration. Unset keeps the proven
    // sequential path with zero behavior change.
    ALWI void exec(uint32_t i_flat, uint32_t /*ht*/, uint32_t /*wt*/, uint32_t slot_offset) const {
        const uint32_t base = tile_base_value<Offset>(tile_base);
        const uint32_t out_idx = walk ? (base + i_flat) : base;
        pack_tile</*out_of_order_output=*/Offset == TileOffset::Set>(to_u32(DstSlot) + slot_offset, Cb, out_idx);
    }

    // Upfront reserve/push — OutputLifecycle::Bulk walks the full window (Ht*Wt); OutputLifecycle::DeferredReserve is pinned (1).
    // Reserve the full output window once (Ht*Wt tiles). Shared by every upfront-reserve
    // policy: Bulk (push the whole window at end), BulkReservePerTile (push 1 per tile) and
    // BulkReservePerChunk (push inner_count per chunk). Called once before the outer loop.
    ALWI void reserve_upfront(uint32_t Ht, uint32_t Wt) const {
        if constexpr (Policy == OutputLifecycle::Bulk ||
                      Policy == OutputLifecycle::BulkReservePerTile ||
                      Policy == OutputLifecycle::BulkReservePerChunk) {
            DataflowBuffer(Cb).reserve_back((Ht * Wt) + tile_base_value<Offset>(tile_base));
        }
    }
    ALWI void push_at_end(uint32_t Ht, uint32_t Wt) const {
        if constexpr (Policy == OutputLifecycle::DeferredReserve ||
                      Policy == OutputLifecycle::Bulk) {
            DataflowBuffer(Cb).push_back((walk ? (Ht * Wt) : 1u) + tile_base_value<Offset>(tile_base));
        }
    }

    static constexpr uint32_t lane_width = to_u32(DstSlot) + 1;

    ALWI void push_per_tile(uint32_t /*i*/) const {
        if constexpr (Policy == OutputLifecycle::Streaming ||
                      Policy == OutputLifecycle::BulkReservePerTile) {
            DataflowBuffer(Cb).push_back(1);
        }
    }

    /// Per-outer-iter push of `inner_count` tiles. Used by Chunked (reserve+push per chunk)
    /// and BulkReservePerChunk (reserve the whole window upfront, push it out one chunk at a time).
    ALWI void push_per_block(uint32_t inner_count) const {
        if constexpr (Policy == OutputLifecycle::Chunked ||
                      Policy == OutputLifecycle::BulkReservePerChunk) {
            DataflowBuffer(Cb).push_back(inner_count);
        }
    }

};

// =============================================================================
// 3. BinaryFpu chain element
// =============================================================================

template <uint32_t CbA,
          uint32_t CbB,
          BinaryFpuOp Op,
          BroadcastDim Bcast,
          InputLifecycle APolicy,
          InputLifecycle BPolicy,
          BinaryDataFormatReconfig Reconfig,
          Dst DstSlot,
          OperandKind AIndex,
          OperandKind BIndex,
          TileOffset OffsetA,
          TileOffset OffsetB>
struct BinaryFpu : BinaryFpuTag {
    static_assert(to_u32(DstSlot) < DEST_AUTO_LIMIT,
                  "BinaryFpu: DEST slot exceeds DEST_AUTO_LIMIT");
    // Comprehensive per-side (IndexMode, Policy) legality. Block rejects PerTile-pop
    // (InputLifecycle::Streaming/InputLifecycle::BulkDrain/InputLifecycle::NoWaitPop — absolute-index pitfall) and PerTile-wait-of-1
    // (InputLifecycle::HeldStream — never tracks per-iter requirement). Scalar/Row/Col accept every
    // legal lifecycle — caller-sized.
    static_assert(is_legal_kind_lifecycle(AIndex, APolicy),
                  "BinaryFpu: (AIndex, APolicy) is illegal for Block — exclude "
                  "InputLifecycle::Streaming / InputLifecycle::HeldStream / InputLifecycle::BulkDrain / InputLifecycle::NoWaitPop on Block walkers.");
    static_assert(is_legal_kind_lifecycle(BIndex, BPolicy),
                  "BinaryFpu: (BIndex, BPolicy) is illegal for Block — exclude "
                  "InputLifecycle::Streaming / InputLifecycle::HeldStream / InputLifecycle::BulkDrain / InputLifecycle::NoWaitPop on Block walkers.");
    // same_dfb dedup safety: when CbA == CbB the B-side wait/pop is skipped, so the
    // helper would under-wait if A and B walked different ranges of the shared CB.
    static_assert((CbA != CbB) || AIndex == BIndex,
                  "BinaryFpu: when CbA == CbB, AIndex and BIndex must match "
                  "(B-side wait/pop is deduped — asymmetric indices would under-wait).");
    // 2D: RowBcast / ColBcast on either side require non-streaming policy.
    static_assert(detail::valid_policy_mode_v<APolicy, AIndex>,
                  "BinaryFpu: A-side RowBcast / ColBcast index require non-streaming APolicy");
    static_assert(detail::valid_policy_mode_v<BPolicy, BIndex>,
                  "BinaryFpu: B-side RowBcast / ColBcast index require non-streaming BPolicy");
    // Per-operand TileBase lifecycle compatibility — InputLifecycle::Streaming/InputLifecycle::Chunked/Cumulative
    // can't compose with runtime base offsets (iter-dependent wait/pop counts).
    static_assert(OffsetA == TileOffset::Unset || is_legal_input_lifecycle_with_base(APolicy),
                  "BinaryFpu: OffsetA Set requires APolicy to be InputLifecycle::Bulk-family or InputLifecycle::CallerManaged");
    static_assert(OffsetB == TileOffset::Unset || is_legal_input_lifecycle_with_base(BPolicy),
                  "BinaryFpu: OffsetB Set requires BPolicy to be InputLifecycle::Bulk-family or InputLifecycle::CallerManaged");
    // Per-block streaming uses chunk-local CB front. When the two sides use
    // DIFFERENT regimes (one per-block → chunk-local index `j`; the other upfront /
    // caller-managed → absolute index `base_tile + j`), the chain dispatcher
    // resolves them separately via the 3-arg exec / exec overloads gated by
    // `needs_per_side_idx`. Same-regime hits the 2-arg fast path.

    static constexpr uint32_t      dfb_a_id()  { return CbA; }
    static constexpr uint32_t      dfb_b_id()  { return CbB; }
    static constexpr InputLifecycle a_policy(){ return APolicy; }
    static constexpr InputLifecycle b_policy(){ return BPolicy; }
    static constexpr bool          is_upfront = (APolicy == InputLifecycle::Bulk) ||
                                                (APolicy == InputLifecycle::HeldBulk) ||
                                                (APolicy == InputLifecycle::Pipelined) ||
                                                (BPolicy == InputLifecycle::Bulk) ||
                                                (BPolicy == InputLifecycle::HeldBulk) ||
                                                (BPolicy == InputLifecycle::Pipelined);
    static constexpr bool          same_dfb    = (CbA == CbB);

    // Per-side local-vs-absolute index resolution. When the two operands declare
    // DIFFERENT regimes (A=PerBlock + B=Upfront, or vice versa), the chain calls
    // the 3-arg exec / exec overload and passes both indices; each side picks.
    // Same-regime falls through to the 2-arg forwarder.
    static constexpr bool a_uses_local_idx = (APolicy == InputLifecycle::Chunked);
    static constexpr bool b_uses_local_idx = (BPolicy == InputLifecycle::Chunked);
    static constexpr bool needs_per_side_idx = (a_uses_local_idx != b_uses_local_idx);

    // Prev-CB fold: BinaryFpu touches srca (CbA) and srcb (CbB) only. Pack-side
    // reconfig is owned by the downstream PackTile element (`PackTileReconfig::Output`)
    // — BinaryFpu writes to DEST, not to a CB, so it has no pack-side responsibility.
    //
    // Per-side selection (Input / SrcA / SrcB) lets the caller opt into a single-side
    // fold when the other side is already programmed (by a previous chain element on
    // the same side, or by external init).
    static constexpr uint32_t      reconfig_srca_dfb =
        (Reconfig == BinaryDataFormatReconfig::Input ||
         Reconfig == BinaryDataFormatReconfig::SrcA) ? CbA : NO_PREV_DFB;
    static constexpr uint32_t      reconfig_srcb_dfb =
        (Reconfig == BinaryDataFormatReconfig::Input ||
         Reconfig == BinaryDataFormatReconfig::SrcB) ? CbB : NO_PREV_DFB;
    // pack side absent -> dfb_for_side defaults to NO_PREV_DFB (downstream PackTile owns pack).

    uint32_t tile_base_a = 0;
    uint32_t tile_base_b = 0;

    constexpr BinaryFpu() noexcept = default;
    constexpr BinaryFpu(uint32_t a, uint32_t b) noexcept : tile_base_a(a), tile_base_b(b) {}
    constexpr explicit BinaryFpu(uint32_t a) noexcept : tile_base_a(a) {}

    // Helper: when same_dfb, both bases live in the single shared wait window.
    // Wait/pop count uses max(base_a, base_b) — caller must stage that many tiles
    // in front of both reads.
    ALWI uint32_t same_dfb_base_max() const noexcept {
        const uint32_t bA = tile_base_value<OffsetA>(tile_base_a);
        const uint32_t bB = tile_base_value<OffsetB>(tile_base_b);
        return bA > bB ? bA : bB;
    }

    // ---- init / reconfig ----
    // srca / srcb / pack reconfig are fold-driven (compile-time-elided
    // when prev_*_cb == cur_*_cb). init() programs only the per-op LLK shape.
    static ALWI void init() {
        // Op-specific init.
        if constexpr (Bcast == BroadcastDim::None) {
            if constexpr      (Op == BinaryFpuOp::Add) add_tiles_init(CbA, CbB);
            else if constexpr (Op == BinaryFpuOp::Sub) sub_tiles_init(CbA, CbB);
            else                                       mul_tiles_init(CbA, CbB);
        } else {
            // Use the *_init_short form from bcast.h:352-446 (math init + unpack init only,
            // no hw_configure / pack_dest_init / sync_init — the full init is undefined
            // mid-MAIN). The operand form reads the actual tensor shape from CB metadata via
            // get_operand_tensor_shape, matching `add_bcast_rows_init_short` /
            // `sub_bcast_cols_init_short` etc. exactly.
            constexpr auto bt = static_cast<ckernel::BroadcastType>(static_cast<uint8_t>(Bcast));
            constexpr auto et = (Op == BinaryFpuOp::Add) ? ckernel::EltwiseBinaryType::ELWADD :
                                (Op == BinaryFpuOp::Sub) ? ckernel::EltwiseBinaryType::ELWSUB :
                                                           ckernel::EltwiseBinaryType::ELWMUL;
            if constexpr (Op == BinaryFpuOp::Mul) {
                MATH((llk_math_eltwise_binary_init<et, bt, MATH_FIDELITY>(CbA, CbB)));
            } else {
                MATH((llk_math_eltwise_binary_init<et, bt, MathFidelity::LoFi>(CbA, CbB)));
            }
            UNPACK((llk_unpack_AB_init<bt>(CbA, CbB)));
        }
    }

    // ---- CB lifecycle (per-tile) ----
    // InputLifecycle::Streaming policies (WaitAndPop / WaitNoPop) always wait 1 — they are incompatible
    // with BlockSize > 1 per-iter consumption. Cumulative scales `(i+1) * block_size`
    // — caller passes `cumulative_count = (i_outer + 1) * block_size`.
    ALWI void wait_per_tile(uint32_t cumulative_count) const {
        if constexpr (APolicy == InputLifecycle::Streaming || APolicy == InputLifecycle::HeldStream) {
            DataflowBuffer(CbA).wait_front(1);
        } else if constexpr (APolicy == InputLifecycle::Pipelined ||
                             APolicy == InputLifecycle::HeldCumulative) {
            DataflowBuffer(CbA).wait_front(cumulative_count);
        }
        if constexpr (!same_dfb) {
            if constexpr (BPolicy == InputLifecycle::Streaming || BPolicy == InputLifecycle::HeldStream) {
                DataflowBuffer(CbB).wait_front(1);
            } else if constexpr (BPolicy == InputLifecycle::Pipelined ||
                                 BPolicy == InputLifecycle::HeldCumulative) {
                DataflowBuffer(CbB).wait_front(cumulative_count);
            }
        }
    }

    /// Per-outer-iter chunked wait. Per-side: A waits `inner_count` if APolicy is
    /// per-block; same for B (same_dfb dedup).
    ALWI void wait_per_block(uint32_t inner_count) const {
        if constexpr (APolicy == InputLifecycle::Chunked) {
            DataflowBuffer(CbA).wait_front(inner_count);
        }
        if constexpr (!same_dfb && BPolicy == InputLifecycle::Chunked) {
            DataflowBuffer(CbB).wait_front(inner_count);
        }
    }


    // 2D: per-side upfront wait — A uses AIndex's window, B uses BIndex's window.
    // Same `same_dfb` dedup as 1D (skip B side when CbA == CbB).
    ALWI void wait_upfront(uint32_t Ht, uint32_t Wt) const {
        if constexpr (APolicy == InputLifecycle::Bulk ||
                      APolicy == InputLifecycle::HeldBulk ||
                      APolicy == InputLifecycle::BulkDrain) {
            const uint32_t a_base = same_dfb ? same_dfb_base_max() : tile_base_value<OffsetA>(tile_base_a);
            DataflowBuffer(CbA).wait_front(detail::window<AIndex>(Ht, Wt) + a_base);
        }
        if constexpr (!same_dfb && (BPolicy == InputLifecycle::Bulk ||
                                   BPolicy == InputLifecycle::HeldBulk ||
                                   BPolicy == InputLifecycle::BulkDrain)) {
            DataflowBuffer(CbB).wait_front(detail::window<BIndex>(Ht, Wt) + tile_base_value<OffsetB>(tile_base_b));
        }
    }

    // Per-side index mode. AIndex drives a_idx, BIndex drives b_idx. The canonical
    // bcast walk is A=BlockIter (walks the tile range) + B=FirstTile (pins the
    // scaler/vector operand at tile 0). OffsetA / OffsetB add a runtime or
    // compile-time base offset to the per-iter index. The 3-arg overload accepts a
    // chunk-local index (`i_local`) and an absolute index (`i_abs`); each side
    // picks via `a_uses_local_idx` / `b_uses_local_idx`. The 2-arg overload is
    // the same-regime fast path and forwards with i_local == i_abs.


    static constexpr uint32_t lane_width = to_u32(DstSlot) + 1;

    ALWI void pop_per_tile(uint32_t /*i*/) const {
        if constexpr (APolicy == InputLifecycle::Streaming ||
                      APolicy == InputLifecycle::NoWaitPop ||
                      APolicy == InputLifecycle::BulkDrain) {
            DataflowBuffer(CbA).pop_front(1);
        }
        if constexpr (!same_dfb && (BPolicy == InputLifecycle::Streaming ||
                                   BPolicy == InputLifecycle::NoWaitPop ||
                                   BPolicy == InputLifecycle::BulkDrain)) {
            DataflowBuffer(CbB).pop_front(1);
        }
    }

    ALWI void pop_per_block(uint32_t inner_count) const {
        if constexpr (APolicy == InputLifecycle::Chunked) {
            DataflowBuffer(CbA).pop_front(inner_count);
        }
        if constexpr (!same_dfb && BPolicy == InputLifecycle::Chunked) {
            DataflowBuffer(CbB).pop_front(inner_count);
        }
    }


    // 2D variants — per-side index + window. 3-arg form takes both chunk-local
    // (`i_local`) and absolute (`i_abs`) flat indices; each side picks via the
    // per-side traits. `ht` is unchanged (always absolute row); `wt_local` /
    // `wt_abs` cover the per-side column index when needed. Same-regime fast
    // path forwards through the 2-arg overload.
    ALWI void exec(uint32_t i_flat_local,
                      uint32_t i_flat_abs,
                      uint32_t ht,
                      uint32_t wt_local,
                      uint32_t wt_abs,
                      uint32_t slot_offset) const {
        const uint32_t a_flat = a_uses_local_idx ? i_flat_local : i_flat_abs;
        const uint32_t b_flat = b_uses_local_idx ? i_flat_local : i_flat_abs;
        const uint32_t a_wt   = a_uses_local_idx ? wt_local     : wt_abs;
        const uint32_t b_wt   = b_uses_local_idx ? wt_local     : wt_abs;
        const uint32_t a_idx  = tile_base_value<OffsetA>(tile_base_a) + detail::idx<AIndex>(a_flat, ht, a_wt);
        const uint32_t b_idx  = tile_base_value<OffsetB>(tile_base_b) + detail::idx<BIndex>(b_flat, ht, b_wt);
        const uint32_t dst    = to_u32(DstSlot) + slot_offset;
        if constexpr (Bcast == BroadcastDim::None) {
            if constexpr      (Op == BinaryFpuOp::Add) add_tiles(CbA, CbB, a_idx, b_idx, dst);
            else if constexpr (Op == BinaryFpuOp::Sub) sub_tiles(CbA, CbB, a_idx, b_idx, dst);
            else                                       mul_tiles(CbA, CbB, a_idx, b_idx, dst);
        } else {
            constexpr auto bt = static_cast<ckernel::BroadcastType>(static_cast<uint8_t>(Bcast));
            if constexpr      (Op == BinaryFpuOp::Add) add_tiles_bcast<bt>(CbA, CbB, a_idx, b_idx, dst);
            else if constexpr (Op == BinaryFpuOp::Sub) sub_tiles_bcast<bt>(CbA, CbB, a_idx, b_idx, dst);
            else                                       mul_tiles_bcast<bt>(CbA, CbB, a_idx, b_idx, dst);
        }
    }

    ALWI void exec(uint32_t i_flat, uint32_t ht, uint32_t wt, uint32_t slot_offset) const {
        exec(i_flat, i_flat, ht, wt, wt, slot_offset);
    }

    ALWI void pop_upfront_end(uint32_t Ht, uint32_t Wt) const {
        if constexpr (APolicy == InputLifecycle::Bulk ||
                      APolicy == InputLifecycle::Pipelined ||
                      APolicy == InputLifecycle::DeferredPop) {
            const uint32_t a_base = same_dfb ? same_dfb_base_max() : tile_base_value<OffsetA>(tile_base_a);
            DataflowBuffer(CbA).pop_front(detail::window<AIndex>(Ht, Wt) + a_base);
        }
        if constexpr (!same_dfb && (BPolicy == InputLifecycle::Bulk ||
                                   BPolicy == InputLifecycle::Pipelined ||
                                   BPolicy == InputLifecycle::DeferredPop)) {
            DataflowBuffer(CbB).pop_front(detail::window<BIndex>(Ht, Wt) + tile_base_value<OffsetB>(tile_base_b));
        }
    }
};

// =============================================================================
// 5. DestReuseBinary chain element
// =============================================================================

template <uint32_t Cb,
          BinaryFpuOp Op,
          DestReuseType ReuseType,
          InputLifecycle Policy,
          DestReuseReconfig Reconfig,
          Dst DstIn,
          Dst DstOut,
          OperandKind IndexMode,
          TileOffset Offset>
struct DestReuseBinary : DestReuseBinaryTag {
    static_assert(to_u32(DstIn) < DEST_AUTO_LIMIT && to_u32(DstOut) < DEST_AUTO_LIMIT,
                  "DestReuseBinary: DEST slot exceeds DEST_AUTO_LIMIT");
    static_assert(is_legal_kind_lifecycle(IndexMode, Policy),
                  "DestReuseBinary: (IndexMode, Policy) is illegal for Block — exclude "
                  "InputLifecycle::Streaming / InputLifecycle::HeldStream / InputLifecycle::BulkDrain / InputLifecycle::NoWaitPop on Block walkers.");
    static_assert(detail::valid_policy_mode_v<Policy, IndexMode>,
                  "DestReuseBinary: RowBcast / ColBcast index require non-streaming policy");
    static_assert(Offset == TileOffset::Unset || is_legal_input_lifecycle_with_base(Policy),
                  "DestReuseBinary: TileOffset::Set requires InputLifecycle::Bulk-family or InputLifecycle::CallerManaged lifecycle");

    // The one CB feeds the src that DEST is NOT routed to: DEST_TO_SRCB -> CB on srcA (dfb_a),
    // DEST_TO_SRCA -> CB on srcB (dfb_b). The other side is the DEST register, not a CB (INVALID_DFB).
    static constexpr uint32_t       dfb_a_id()         { return (ReuseType == DestReuseType::DEST_TO_SRCB) ? Cb : INVALID_DFB; }
    static constexpr uint32_t       dfb_b_id()         { return (ReuseType == DestReuseType::DEST_TO_SRCA) ? Cb : INVALID_DFB; }
    static constexpr InputLifecycle a_policy()        { return Policy; }
    static constexpr bool           is_upfront        = (Policy == InputLifecycle::Bulk) ||
                                                        (Policy == InputLifecycle::HeldBulk) ||
                                                        (Policy == InputLifecycle::Pipelined);

    // Prev-CB fold: DestReuseBinary loads CB into srca (when DEST → srcb) or srcb
    // (when DEST → srca). Reconfig only fires when opted in.
    //
    // `Input` follows ReuseType (programs the side the CB actually unpacks into).
    // `SrcA` / `SrcB` explicitly pick a side, decoupled from ReuseType — used when
    // the caller wants to program a specific unpack lane regardless of which lane
    // DEST is feeding into.
    static constexpr uint32_t       reconfig_srca_dfb  =
        ((Reconfig == DestReuseReconfig::Input && ReuseType == DestReuseType::DEST_TO_SRCB) ||
         Reconfig == DestReuseReconfig::SrcA) ? Cb : NO_PREV_DFB;
    static constexpr uint32_t       reconfig_srcb_dfb  =
        ((Reconfig == DestReuseReconfig::Input && ReuseType == DestReuseType::DEST_TO_SRCA) ||
         Reconfig == DestReuseReconfig::SrcB) ? Cb : NO_PREV_DFB;
    // pack side absent -> dfb_for_side defaults to NO_PREV_DFB.

    uint32_t tile_base = 0;

    constexpr DestReuseBinary() noexcept = default;
    constexpr explicit DestReuseBinary(uint32_t base) noexcept : tile_base(base) {}

    // srca / srcb reconfig is fold-driven; init() programs only the per-op
    // LLK shape.
    static ALWI void init() {
        constexpr auto et = (Op == BinaryFpuOp::Add) ? ckernel::EltwiseBinaryType::ELWADD :
                            (Op == BinaryFpuOp::Sub) ? ckernel::EltwiseBinaryType::ELWSUB :
                                                       ckernel::EltwiseBinaryType::ELWMUL;
        constexpr auto reuse = (ReuseType == DestReuseType::DEST_TO_SRCA)
                                   ? ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCA
                                   : ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCB;
        binary_dest_reuse_tiles_init<et, reuse>(Cb);
    }

    ALWI void wait_per_tile(uint32_t cumulative_count) const {
        if constexpr (Policy == InputLifecycle::Streaming || Policy == InputLifecycle::HeldStream) {
            DataflowBuffer(Cb).wait_front(1);
        } else if constexpr (Policy == InputLifecycle::Pipelined ||
                             Policy == InputLifecycle::HeldCumulative) {
            DataflowBuffer(Cb).wait_front(cumulative_count);
        }
    }
    ALWI void wait_per_block(uint32_t inner_count) const {
        if constexpr (Policy == InputLifecycle::Chunked) {
            DataflowBuffer(Cb).wait_front(inner_count);
        }
    }

    // 2D variants
    ALWI void wait_upfront(uint32_t Ht, uint32_t Wt) const {
        if constexpr (Policy == InputLifecycle::Bulk ||
                      Policy == InputLifecycle::HeldBulk ||
                      Policy == InputLifecycle::BulkDrain) {
            DataflowBuffer(Cb).wait_front(detail::window<IndexMode>(Ht, Wt) + tile_base_value<Offset>(tile_base));
        }
    }
    ALWI void exec(uint32_t i_flat, uint32_t ht, uint32_t wt, uint32_t slot_offset) const {
        constexpr auto et = (Op == BinaryFpuOp::Add) ? ckernel::EltwiseBinaryType::ELWADD :
                            (Op == BinaryFpuOp::Sub) ? ckernel::EltwiseBinaryType::ELWSUB :
                                                       ckernel::EltwiseBinaryType::ELWMUL;
        constexpr auto reuse = (ReuseType == DestReuseType::DEST_TO_SRCA)
                                   ? ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCA
                                   : ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCB;
        const uint32_t in_idx = tile_base_value<Offset>(tile_base) + detail::idx<IndexMode>(i_flat, ht, wt);
        binary_dest_reuse_tiles<et, reuse>(Cb, in_idx, to_u32(DstIn) + slot_offset);
    }
    ALWI void pop_upfront_end(uint32_t Ht, uint32_t Wt) const {
        if constexpr (Policy == InputLifecycle::Bulk ||
                      Policy == InputLifecycle::Pipelined ||
                      Policy == InputLifecycle::DeferredPop) {
            DataflowBuffer(Cb).pop_front(detail::window<IndexMode>(Ht, Wt) + tile_base_value<Offset>(tile_base));
        }
    }

    static constexpr uint32_t lane_width =
        (to_u32(DstIn) > to_u32(DstOut)) ? (to_u32(DstIn) + 1) : (to_u32(DstOut) + 1);
    ALWI void pop_per_tile(uint32_t /*i*/) const {
        if constexpr (Policy == InputLifecycle::Streaming ||
                      Policy == InputLifecycle::NoWaitPop ||
                      Policy == InputLifecycle::BulkDrain) {
            DataflowBuffer(Cb).pop_front(1);
        }
    }
    ALWI void pop_per_block(uint32_t inner_count) const {
        if constexpr (Policy == InputLifecycle::Chunked) {
            DataflowBuffer(Cb).pop_front(inner_count);
        }
    }
};

// =============================================================================
// 6. UnaryBcast chain element
// =============================================================================

template <BroadcastDim Dim,
          uint32_t Cb,
          InputLifecycle Policy,
          UnaryBcastReconfig Reconfig,
          Dst DstSlot>
struct UnaryBcast : UnaryBcastTag {
    static_assert(to_u32(DstSlot) < DEST_AUTO_LIMIT,
                  "UnaryBcast: DEST slot exceeds DEST_AUTO_LIMIT");

    // UnaryBcast reads one CB front (dfb_a); dfb_b is absent -> dfb_b_of defaults to INVALID_DFB.
    static constexpr uint32_t       dfb_a_id()         { return Cb; }
    static constexpr InputLifecycle a_policy()        { return Policy; }
    static constexpr bool           is_upfront        = (Policy == InputLifecycle::Bulk) ||
                                                        (Policy == InputLifecycle::HeldBulk) ||
                                                        (Policy == InputLifecycle::Pipelined);

    // Prev-CB fold: UnaryBcast binds BOTH srca and srcb to Cb. The broadcast datacopy MOP
    // drives the FPU SrcB lane (ELWADD + SRCB_BCAST_*), so srcb must be reprogrammed too — a
    // srca-only reconfig leaves ALU_FORMAT_SPEC_REG1_SrcB stale from a preceding two-operand op
    // (e.g. layernorm's BinaryFpu(cb_ex2, cb_eps) leaves SrcB = cb_eps), which corrupts the bcast.
    // Declaring both CBs lets the chain's reconfig fold (emit_pre_element_transitions) emit the
    // reconfig before init() AND record Cb as the post-element srca/srcb state for the next
    // element — so a subsequent srca/srcb reader sees the correct prev-CB and won't wrongly elide.
    // Pack-side reconfig is owned by the downstream PackTile (PackTileReconfig::Output), exactly
    // like BinaryFpu — UnaryBcast never configures pack.
    static constexpr uint32_t       reconfig_srca_dfb  = (Reconfig == UnaryBcastReconfig::Input) ? Cb : NO_PREV_DFB;
    static constexpr uint32_t       reconfig_srcb_dfb  = (Reconfig == UnaryBcastReconfig::Input) ? Cb : NO_PREV_DFB;
    // pack side absent -> dfb_for_side defaults to NO_PREV_DFB.

    static ALWI void init() {
        constexpr auto bt = static_cast<ckernel::BroadcastType>(static_cast<uint8_t>(Dim));
        // Small per-element init only — the caller owns BIG init (compute_kernel_hw_startup /
        // a boot unary_bcast_init). This does NOT re-run any hw_configure or pack init, and it
        // does NOT do the srca/srcb format reconfig: that is fold-driven (see reconfig_srca_dfb /
        // reconfig_srcb_dfb above), emitted by emit_pre_element_transitions() before this init().
        // init() emits only the bcast datacopy MOP (unpack-A + math datacopy), icb-only — mirrors
        // the MOP-init portion of `unary_bcast_init`, minus every BIG hw_configure / pack line.
#if defined(TRISC_UNPACK) || defined(TRISC_MATH)
        const std::uint32_t dst_format = get_operand_dst_format(Cb);
#ifndef ARCH_QUASAR
        const bool enable_unpack_to_dest = (dst_format == (std::uint32_t)DataFormat::Float32) ||
                                           (dst_format == (std::uint32_t)DataFormat::UInt32) ||
                                           (dst_format == (std::uint32_t)DataFormat::Int32);
        if (enable_unpack_to_dest) {
            UNPACK((llk_unpack_A_init<bt, false, ckernel::EltwiseBinaryReuseDestType::NONE, true>(false, false, Cb)));
            MATH((llk_math_eltwise_unary_datacopy_init<ckernel::DataCopyType::A2D, DST_ACCUM_MODE, bt>(Cb)));
        } else {
            UNPACK((llk_unpack_A_init<bt, false, ckernel::EltwiseBinaryReuseDestType::NONE, false>(false, false, Cb)));
            MATH((llk_math_eltwise_unary_datacopy_init<ckernel::DataCopyType::B2D, DST_ACCUM_MODE, bt>(Cb)));
        }
#else
        const bool enable_unpack_to_dest =
            (dst_format == (std::uint32_t)DataFormat::Float32) || (dst_format == (std::uint32_t)DataFormat::Int32);
        if (enable_unpack_to_dest) {
            ASSERT(false);  // Quasar unpack_to_dest unary bcast not implemented yet
            UNPACK((llk_unpack_A_init<false /*TRANSPOSE_EN*/, true /*IS_32b_DEST_EN*/>(Cb)));
            MATH((llk_math_eltwise_unary_datacopy_init<ckernel::DataCopyType::A2D, true>(Cb)));
        } else {
            UNPACK((llk_unpack_A_init<false, false>(Cb)));
            MATH((llk_math_eltwise_unary_datacopy_init<ckernel::DataCopyType::B2D, false>(Cb)));
        }
#endif
#endif
    }

    ALWI void wait_per_tile(uint32_t cumulative_count) const {
        if constexpr (Policy == InputLifecycle::Streaming || Policy == InputLifecycle::HeldStream) {
            DataflowBuffer(Cb).wait_front(1);
        } else if constexpr (Policy == InputLifecycle::Pipelined ||
                             Policy == InputLifecycle::HeldCumulative) {
            DataflowBuffer(Cb).wait_front(cumulative_count);
        }
    }
    ALWI void wait_per_block(uint32_t inner_count) const {
        if constexpr (Policy == InputLifecycle::Chunked) {
            DataflowBuffer(Cb).wait_front(inner_count);
        }
    }

    // 2D variants — UnaryBcast always reads tile 0 (intra-tile bcast LLK), no per-iter
    // tile index. Upfront window in 2D = Ht * Wt (every (ht, wt) iter consumes one tile).
    ALWI void wait_upfront(uint32_t Ht, uint32_t Wt) const {
        if constexpr (Policy == InputLifecycle::Bulk ||
                      Policy == InputLifecycle::HeldBulk ||
                      Policy == InputLifecycle::BulkDrain) {
            DataflowBuffer(Cb).wait_front(Ht * Wt);
        }
    }
    ALWI void exec(uint32_t /*i_flat*/, uint32_t /*ht*/, uint32_t /*wt*/, uint32_t slot_offset) const {
        constexpr auto bt = static_cast<ckernel::BroadcastType>(static_cast<uint8_t>(Dim));
        unary_bcast<bt>(Cb, /*in_tile_index=*/0, to_u32(DstSlot) + slot_offset);
    }
    ALWI void pop_upfront_end(uint32_t Ht, uint32_t Wt) const {
        if constexpr (Policy == InputLifecycle::Bulk ||
                      Policy == InputLifecycle::Pipelined ||
                      Policy == InputLifecycle::DeferredPop) {
            DataflowBuffer(Cb).pop_front(Ht * Wt);
        }
    }

    static constexpr uint32_t lane_width = to_u32(DstSlot) + 1;
    ALWI void pop_per_tile(uint32_t /*i*/) const {
        if constexpr (Policy == InputLifecycle::Streaming ||
                      Policy == InputLifecycle::NoWaitPop ||
                      Policy == InputLifecycle::BulkDrain) {
            DataflowBuffer(Cb).pop_front(1);
        }
    }
    ALWI void pop_per_block(uint32_t inner_count) const {
        if constexpr (Policy == InputLifecycle::Chunked) {
            DataflowBuffer(Cb).pop_front(inner_count);
        }
    }
};

// =============================================================================
// 7. Fill / Rand chain elements — declared in eltwise_chain.hpp; full bodies
//    live in eltwise_fill.hpp / eltwise_rand.hpp. Keeping the bodies out of this
//    header keeps the fill/rand LLK headers out of every chain-using kernel's
//    include cone, while the declarations still satisfy the trait predicates.
// =============================================================================
//
// (FillScalar / FillInt / FillBitcast / RandTile are declared in eltwise_chain.hpp;
//  eltwise_fill.hpp specializes those templates with full bodies.)

// =============================================================================
// 8. EltwiseChain typed list — defined here, declared in .hpp
// =============================================================================

template <class... Es>
struct EltwiseChain {
    static constexpr size_t size = sizeof...(Es);
};

// =============================================================================
// 9. Chain-shape trait predicates
// =============================================================================

// chain_lane_width — N-element fold. Max of per-element `lane_width`. Bounds
// the legal BlockSize at the chain call site via the static_assert
// `BlockSize * chain_lane_width <= DEST_AUTO_LIMIT`. Each element writes to
// DEST[dst_slot + j * chain_lane_width] for lane j in [0, BlockSize).
//
// SFINAE fallback: elements that don't expose a `lane_width` member (caller-defined
// chain elements that inherit directly from `CopyTileTag` / `PackTileTag` / `DestOnlyTag`
// without inheriting from `UnaryOp`/`BinaryOp`/`TernaryOp` bases) default
// to 1. Multiple-inheritance ambiguity (e.g. `OptionalChainElement<true, FillScalar>`)
// is sidestepped by reading via this detector instead of `Es::lane_width` directly.
namespace detail {
template <class, class = void>
struct elem_lane_width : std::integral_constant<uint32_t, 1> {};

template <class E>
struct elem_lane_width<E, std::void_t<decltype(E::lane_width)>>
    : std::integral_constant<uint32_t, E::lane_width> {};

template <class E>
constexpr uint32_t elem_lane_width_v = elem_lane_width<E>::value;
}  // namespace detail

template <class Chain>
struct chain_lane_width;

template <class... Es>
struct chain_lane_width<EltwiseChain<Es...>>
    : std::integral_constant<uint32_t, detail::ChainTraits<Es...>::lane_width> {};

template <class Chain>
inline constexpr uint32_t chain_lane_width_v = chain_lane_width<Chain>::value;

// chain_max_block_v — largest block_size that fits in DEST for this chain, given its
// lane-width fold. Caller-facing compile-time constant: pass any value <= this to
// the runtime `block_size` arg on `eltwise_chain`. Caller can `static_assert` their
// chosen block against this value to recover the build-time DEST overflow signal.
template <class Chain>
inline constexpr uint32_t chain_max_block_v = DEST_AUTO_LIMIT / chain_lane_width_v<Chain>;

// chain_supports_block — N-element fold. True when every CB-reader element uses a
// policy that stages a multi-tile DEST window (Upfront / Cumulative / NoWaitNoPop).
// InputLifecycle::Streaming policies (WaitAndPop / WaitNoPop / InputLifecycle::NoWaitPop) consume ONE tile per iter
// and are incompatible with chain BlockSize > 1 (chain consumes BlockSize tiles per
// outer iter). The chain `static_assert`s on this predicate when `BlockSize > 1`.
namespace detail {
template <class E> constexpr InputLifecycle b_policy_of();  // defined below (defaults to CallerManaged)

constexpr bool policy_supports_block(InputLifecycle p) {
    return p == InputLifecycle::Bulk ||
           p == InputLifecycle::HeldBulk ||
           p == InputLifecycle::Pipelined ||
           p == InputLifecycle::HeldCumulative ||
           p == InputLifecycle::CallerManaged ||
           p == InputLifecycle::Chunked;
}

template <class E>
constexpr bool element_supports_block() {
    if constexpr (is_cb_reader_op_v<E>) {
        return policy_supports_block(E::a_policy()) && policy_supports_block(b_policy_of<E>());
    } else {
        return true;  // non-CB-reader elements don't constrain block_size
    }
}

}  // namespace detail

// =============================================================================
// ChainTraits — reflect each element once into an ElemDesc record, then derive every
// value-based chain property as a field. Type-uniformity (chain_*_uniform) stays
// separate — it reads element types, not values. Emission stays separate too.
// All compile-time: the whole struct folds to constants (zero runtime cost).
// =============================================================================
namespace detail {

// SFINAE accessors for the two members the collision derivations read but not every
// element declares (CB readers carry is_upfront; PackTile carries pack_dst_slot).
template <class E, class = void> struct has_is_upfront_m : std::false_type {};
template <class E> struct has_is_upfront_m<E, std::void_t<decltype(E::is_upfront)>> : std::true_type {};
template <class E> constexpr bool is_upfront_of() {
    if constexpr (has_is_upfront_m<E>::value) return E::is_upfront; else return false;
}
template <class E, class = void> struct has_pack_dst_slot_m : std::false_type {};
template <class E> struct has_pack_dst_slot_m<E, std::void_t<decltype(E::pack_dst_slot)>> : std::true_type {};
template <class E> constexpr Dst pack_dst_slot_of() {
    if constexpr (has_pack_dst_slot_m<E>::value) return E::pack_dst_slot; else return Dst::D0;
}
// b_policy defaults to CallerManaged (the unary-reader / no-srcB-CB case), so only elements
// with a genuine srcB operand (BinaryFpu) declare it.
template <class E, class = void> struct has_b_policy_m : std::false_type {};
template <class E> struct has_b_policy_m<E, std::void_t<decltype(E::b_policy())>> : std::true_type {};
template <class E> constexpr InputLifecycle b_policy_of() {
    if constexpr (has_b_policy_m<E>::value) return E::b_policy(); else return InputLifecycle::CallerManaged;
}

// One plain-data descriptor per element — reflected once via the existing accessors.
struct ElemDesc {
    bool is_cb_reader;
    bool is_pack;
    uint32_t srca_cb;      // dfb_for_side<SrcA> (NO_PREV_DFB when not programmed) — per-side consistency + prev input
    uint32_t srcb_cb;      // dfb_for_side<SrcB>
    uint32_t pack_side_dfb; // dfb_for_side<Pack> (reconfig_pack_dfb) — prev / last_pack / hetero input
    uint32_t dfb_a;         // dfb_a_of (INVALID_DFB when n/a) — reader-collision input
    uint32_t dfb_b;         // dfb_b_of (INVALID_DFB when n/a)
    uint32_t pack_dfb;      // pack_dfb_of (INVALID_DFB when n/a) — writer-collision input
    Dst pack_dst_slot;
    bool is_upfront;
    uint32_t lane_width;
    bool supports_block;
};

template <class E>
constexpr ElemDesc describe() {
    return ElemDesc{
        is_cb_reader_op_v<E>,
        is_pack_tile_op_v<E>,
        dfb_for_side<Side::SrcA, E>(),
        dfb_for_side<Side::SrcB, E>(),
        dfb_for_side<Side::Pack, E>(),
        dfb_a_of<E>(),
        dfb_b_of<E>(),
        pack_dfb_of<E>(),
        pack_dst_slot_of<E>(),
        is_upfront_of<E>(),
        elem_lane_width_v<E>,
        element_supports_block<E>(),
    };
}

// Derivations over the descriptor array — flat loops bounded by the real element count
// `n` (the array is sized [N?N:1], so an empty chain must NOT read the lone default slot).
constexpr uint32_t ct_lane_width(const ElemDesc* d, int n) {
    uint32_t w = 1;
    for (int i = 0; i < n; ++i)
        if (d[i].lane_width > w) w = d[i].lane_width;
    return w;
}
constexpr bool ct_supports_block(const ElemDesc* d, int n) {
    bool r = true;
    for (int i = 0; i < n; ++i) r = r && d[i].supports_block;
    return r;
}
constexpr bool ct_side_consistent(const ElemDesc* d, int n, uint32_t ElemDesc::*side) {
    uint32_t seen = NO_PREV_DFB;
    for (int i = 0; i < n; ++i) {
        uint32_t dfb = d[i].*side;
        if (dfb == NO_PREV_DFB) continue;
        if (seen == NO_PREV_DFB) seen = dfb;
        else if (seen != dfb) return false;
    }
    return true;
}
constexpr bool ct_reader_collide(const ElemDesc* d, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            if (!(d[i].is_cb_reader && d[j].is_cb_reader)) continue;
            if (!(d[i].is_upfront && d[j].is_upfront)) continue;
            uint32_t a0 = d[i].dfb_a, a1 = d[i].dfb_b, b0 = d[j].dfb_a, b1 = d[j].dfb_b;
            if ((a0 != INVALID_DFB && (a0 == b0 || a0 == b1)) || (a1 != INVALID_DFB && (a1 == b0 || a1 == b1)))
                return true;
        }
    return false;
}
constexpr bool ct_writer_collide(const ElemDesc* d, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (d[i].is_pack && d[j].is_pack && d[i].pack_dfb == d[j].pack_dfb &&
                d[i].pack_dst_slot == d[j].pack_dst_slot)
                return true;
    return false;
}

// Per-side "previous programmed CB at each index" tables, built in ONE forward sweep:
// carry a running prev per side, record it BEFORE each element, update it when the
// element programs that side. prev.srca[I] is the most recent CB programmed on SrcA by
// any element before index I (NO_PREV_DFB if none).
template <int M>
struct PrevTable {
    uint32_t srca[M];
    uint32_t srcb[M];
    uint32_t pack[M];
};
template <int M>
constexpr PrevTable<M> ct_build_prev(const ElemDesc* d, int n) {
    PrevTable<M> t{};
    uint32_t pa = NO_PREV_DFB, pb = NO_PREV_DFB, pp = NO_PREV_DFB;
    for (int i = 0; i < n; ++i) {
        t.srca[i] = pa;
        t.srcb[i] = pb;
        t.pack[i] = pp;
        if (d[i].srca_cb != NO_PREV_DFB) pa = d[i].srca_cb;
        if (d[i].srcb_cb != NO_PREV_DFB) pb = d[i].srcb_cb;
        if (d[i].pack_side_dfb != NO_PREV_DFB) pp = d[i].pack_side_dfb;
    }
    return t;
}
// Last opt-in pack CB in chain order (iter-to-iter wraparound prev for pack site 0).
constexpr uint32_t ct_last_pack_cb(const ElemDesc* d, int n) {
    uint32_t last = NO_PREV_DFB;
    for (int i = 0; i < n; ++i)
        if (d[i].pack_side_dfb != NO_PREV_DFB) last = d[i].pack_side_dfb;
    return last;
}
// True iff ≥2 opt-in pack sites declare different CBs (boot can't program all).
constexpr bool ct_pack_hetero(const ElemDesc* d, int n) {
    uint32_t first = NO_PREV_DFB;
    for (int i = 0; i < n; ++i) {
        if (d[i].pack_side_dfb == NO_PREV_DFB) continue;
        if (first == NO_PREV_DFB) first = d[i].pack_side_dfb;
        else if (first != d[i].pack_side_dfb) return true;
    }
    return false;
}

template <class... Es>
struct ChainTraits {
    static constexpr int N = int(sizeof...(Es));
    static constexpr ElemDesc d[N ? N : 1] = {describe<Es>()...};  // the one walk

    static constexpr uint32_t lane_width = ct_lane_width(d, N);
    static constexpr bool supports_block = ct_supports_block(d, N);
    static constexpr bool srca_consistent = ct_side_consistent(d, N, &ElemDesc::srca_cb);
    static constexpr bool srcb_consistent = ct_side_consistent(d, N, &ElemDesc::srcb_cb);
    static constexpr bool reader_collide = ct_reader_collide(d, N);
    static constexpr bool writer_collide = ct_writer_collide(d, N);

    // Per-side prev-CB history (one sweep), + pack-side metadata.
    static constexpr PrevTable<N ? N : 1> prev = ct_build_prev<N ? N : 1>(d, N);
    static constexpr uint32_t last_pack_cb = ct_last_pack_cb(d, N);
    static constexpr bool pack_hetero = ct_pack_hetero(d, N);
};

}  // namespace detail

template <class Chain>
struct chain_supports_block;

template <class... Es>
struct chain_supports_block<EltwiseChain<Es...>>
    : std::bool_constant<detail::ChainTraits<Es...>::supports_block> {};

template <class Chain>
inline constexpr bool chain_supports_block_v = chain_supports_block<Chain>::value;

// element_uses_per_block_index_v — true when the element's CB front is chunk-
// local (per-block streaming reader or pack), so the pipeline passes `j`
// instead of `base_tile + j` as the tile index seen by `exec`. SFINAE on each
// element kind. False for elements that don't read/write CBs (DEST-only ops).
namespace detail {

template <class E, class = void>
struct elem_per_block_reader : std::false_type {};

template <class E>
struct elem_per_block_reader<E, std::enable_if_t<is_cb_reader_op_v<E>>>
    : std::bool_constant<(E::a_policy() == InputLifecycle::Chunked) ||
                         (b_policy_of<E>() == InputLifecycle::Chunked)> {};

template <class E, class = void>
struct elem_per_block_pack : std::false_type {};

template <class E>
struct elem_per_block_pack<E, std::void_t<decltype(E::uses_per_block_pack)>>
    : std::bool_constant<E::uses_per_block_pack> {};

template <class E>
inline constexpr bool element_uses_per_block_index_v =
    elem_per_block_reader<E>::value || elem_per_block_pack<E>::value;

// elem_needs_per_side_idx_v — true when the element wants per-side
// local-vs-absolute index resolution (BinaryFpu when A/B policies disagree on
// the per-block regime). SFINAE-tolerant: absence of `needs_per_side_idx`
// static member defaults to false.
template <class E, class = void>
struct elem_needs_per_side_idx : std::false_type {};

template <class E>
struct elem_needs_per_side_idx<E, std::void_t<decltype(E::needs_per_side_idx)>>
    : std::bool_constant<E::needs_per_side_idx> {};

template <class E>
inline constexpr bool elem_needs_per_side_idx_v = elem_needs_per_side_idx<E>::value;

}  // namespace detail

// chain_has_duplicate_upfront_dfbs / chain_pack_writes_collide — pairwise collision
// checks, implemented as flat nested loops in ChainTraits (reader_collide / writer_collide).
template <class... Es> struct chain_has_duplicate_upfront_dfbs<EltwiseChain<Es...>>
    : std::bool_constant<detail::ChainTraits<Es...>::reader_collide> {};

template <class... Es> struct chain_pack_writes_collide<EltwiseChain<Es...>>
    : std::bool_constant<detail::ChainTraits<Es...>::writer_collide> {};

// `chain_hoist_math_mop` / `chain_hoist_sfpu` — per-cohort decisions on whether each
// element's init() can run once at boot instead of per tile. The two cohorts (math-MOP =
// ADDR_MOD_0..3 + MATH MOP; SFPU = ADDR_MOD_7 + SFPU CSR) are hardware-disjoint, so the
// decisions are independent in principle; we constrain hoist_sfpu to imply hoist_math_mop
// to keep the dispatcher simple. Eltwise-only scope (no matmul/reduce hazards).
//
// All three conditions guard the same failure mode — boot programs each lane once, so a
// non-uniform chain leaves only the LAST init's state programmed and earlier elements run
// with the wrong MOP/format:
//   - Per-side CB consistency: every element programming a side (srcA/srcB) uses the same CB.
//   - MATH-MOP uniformity: all `is_math_mop_op_v` elements (CopyTile / BinaryFpu /
//     DestReuseBinary / UnaryBcast) are the same instantiated type.
//   - SFPU-init uniqueness: all `is_sfpu_op_v` elements are the same type. (Caught
//     mish FP32 Exp/Log1p/Tanh → tanh saturation PCC 0.988; logit stage-2
//     Rsub/DivBinary/Log → ATOL 9-12.)
//
// "Identical" is `std::is_same_v` — rejects some safe chains (two `Exp<...>` differing only
// in slot), but false negatives only cost a per-tile init, never correctness.

namespace detail {

// Per-side CB consistency is ChainTraits::srca_consistent / srcb_consistent.

// Trait wrappers (Pred<E>::value form) — `is_sfpu_op_v` / `is_math_mop_op_v`
// are `inline constexpr bool` variable templates; wrap them so they fit the
// `Pred<E>::value` interface used by the uniformity fold.
template <class E>
struct is_sfpu_op_t : std::bool_constant<is_sfpu_op_v<E>> {};
template <class E>
struct is_math_mop_op_t : std::bool_constant<is_math_mop_op_v<E>> {};

// Uniformity helper (used for both the MATH-MOP and SFPU cohorts): across all
// `Es...` satisfying `Pred<E>`, require every instantiated type to be
// `std::is_same_v` with every other (≤ 1 distinct).
//
// Strategy: find the first `E` in `Es...` with `Pred<E>::value == true`;
// call it `Rep`. Then every other `E` with `Pred<E>::value == true` must
// satisfy `std::is_same_v<Rep, E>`. If no element matches, the chain is
// vacuously uniform (returns true).
template <template <class> class Pred, class... Es>
struct first_match { using type = void; };

template <template <class> class Pred, class First, class... Rest>
struct first_match<Pred, First, Rest...> {
    using type = std::conditional_t<Pred<First>::value, First,
                                    typename first_match<Pred, Rest...>::type>;
};

template <template <class> class Pred, class Rep, class... Es>
struct all_match_rep
    : std::bool_constant<(((!Pred<Es>::value) || std::is_same_v<Rep, Es>) && ...)> {};

// Empty `Rep == void` short-circuits to `true` (no element matched Pred at all,
// so the "must equal Rep" condition is vacuously satisfied).
template <template <class> class Pred, class... Es>
constexpr bool chain_all_pred_uniform_v = []() {
    using Rep = typename first_match<Pred, Es...>::type;
    if constexpr (std::is_same_v<Rep, void>) {
        return true;
    } else {
        return all_match_rep<Pred, Rep, Es...>::value;
    }
}();

}  // namespace detail

template <class Chain>
struct chain_per_side_dfbs_consistent : std::true_type {};

template <class... Es>
struct chain_per_side_dfbs_consistent<EltwiseChain<Es...>>
    : std::bool_constant<detail::ChainTraits<Es...>::srca_consistent &&
                         detail::ChainTraits<Es...>::srcb_consistent> {};

template <class Chain>
struct chain_math_mop_uniform : std::true_type {};

template <class... Es>
struct chain_math_mop_uniform<EltwiseChain<Es...>>
    : std::bool_constant<detail::chain_all_pred_uniform_v<detail::is_math_mop_op_t, Es...>> {};

template <class Chain>
struct chain_sfpu_inits_uniform : std::true_type {};

template <class... Es>
struct chain_sfpu_inits_uniform<EltwiseChain<Es...>>
    : std::bool_constant<detail::chain_all_pred_uniform_v<detail::is_sfpu_op_t, Es...>> {};

// Math-MOP cohort hoist: per-side CB consistency + math-MOP uniformity.
// True when CopyTile / BinaryFpu / DestReuseBinary / UnaryBcast inits can be
// emitted once at boot instead of per tile. Independent of SFPU uniformity.
template <class Chain>
struct chain_hoist_math_mop : std::false_type {};

template <class... Es>
struct chain_hoist_math_mop<EltwiseChain<Es...>>
    : std::bool_constant<chain_per_side_cbs_consistent_v<EltwiseChain<Es...>> &&
                         chain_math_mop_uniform_v<EltwiseChain<Es...>>> {};

// SFPU cohort hoist: requires math-MOP hoist AND SFPU init uniformity.
// True when every element's init can be emitted at boot (the fully-hoisted shape).
template <class Chain>
struct chain_hoist_sfpu : std::false_type {};

template <class... Es>
struct chain_hoist_sfpu<EltwiseChain<Es...>>
    : std::bool_constant<chain_hoist_math_mop_v<EltwiseChain<Es...>> &&
                         chain_sfpu_inits_uniform_v<EltwiseChain<Es...>>> {};

// =============================================================================
// 10. Chain pipeline — per-iteration emit
// =============================================================================

namespace detail {

// SFINAE wrappers for the per-block lifecycle hooks — user-defined chain elements
// (custom CbReaderTag / PackTileTag types declared in individual kernel sources)
// may not provide wait_per_block / pop_per_block / reserve_per_block / push_per_block.
// These no-op when the method is absent so the chain pipeline keeps working with
// elements that don't implement the per-block (chunked) lifecycle hooks.

template <class E, class = void>
struct has_wait_per_block : std::false_type {};
template <class E>
struct has_wait_per_block<E, std::void_t<decltype(std::declval<const E&>().wait_per_block(0u))>>
    : std::true_type {};

template <class E, class = void>
struct has_pop_per_block : std::false_type {};
template <class E>
struct has_pop_per_block<E, std::void_t<decltype(std::declval<const E&>().pop_per_block(0u))>>
    : std::true_type {};

template <class E, class = void>
struct has_reserve_per_block : std::false_type {};
template <class E>
struct has_reserve_per_block<E, std::void_t<decltype(std::declval<const E&>().reserve_per_block(0u))>>
    : std::true_type {};

template <class E, class = void>
struct has_push_per_block : std::false_type {};
template <class E>
struct has_push_per_block<E, std::void_t<decltype(std::declval<const E&>().push_per_block(0u))>>
    : std::true_type {};

template <class E>
ALWI void elem_wait_per_block(const E& e, uint32_t inner_count) {
    if constexpr (has_wait_per_block<E>::value) e.wait_per_block(inner_count);
    else (void)e, (void)inner_count;
}
template <class E>
ALWI void elem_pop_per_block(const E& e, uint32_t inner_count) {
    if constexpr (has_pop_per_block<E>::value) e.pop_per_block(inner_count);
    else (void)e, (void)inner_count;
}
template <class E>
ALWI void elem_reserve_per_block(const E& e, uint32_t inner_count) {
    if constexpr (has_reserve_per_block<E>::value) e.reserve_per_block(inner_count);
    else (void)e, (void)inner_count;
}
template <class E>
ALWI void elem_push_per_block(const E& e, uint32_t inner_count) {
    if constexpr (has_push_per_block<E>::value) e.push_per_block(inner_count);
    else (void)e, (void)inner_count;
}

// init() dispatch — CRTP bases (DestOnly) have static init()s that take no args.
// CB-bound elements have static init() that may emit reconfig. Both are static.
template <class E>
ALWI void elem_init() { E::init(); }

// =============================================================================
// emit_pre_element_transitions<E, I, Es...>()
//
// Emits srca / srcb / pack reconfig for element I, each compile-time-elided when
// prev_*_cb == curr_*_cb. A chain that shares a CB on a side thus reconfigs it once (at
// element 0, prev == NO_PREV_DFB) and never again. Zero run-time cost — `if constexpr`.
//
// srca+srcb coalesce: both-with-prev → 4-arg _with_dt; both-first-emit → 2-arg combined;
// mixed → independent per-side. Pack is always independent. The LLK _with_dt overloads
// fast-path-skip on format equality, so an emitted reconfig on matching dtypes is a
// hardware no-op (a few compares).
//
// Pack-side hoist: homogeneous chains (≤1 opt-in pack site, or all share a CB) emit at
// boot via `pack_init_for_each`; heterogeneous chains (≥2 sites, different CBs) emit only
// the first site at boot and defer the rest to per-stage `emit_per_stage_pack_reconfig`.
// DEST accumulation is build-flag-driven (no per-element fp32 fold here).
// =============================================================================

template <class E, std::size_t I, class... Es>
ALWI void emit_pre_element_transitions() {
    constexpr uint32_t curr_a = dfb_for_side<Side::SrcA, E>();
    constexpr uint32_t curr_b = dfb_for_side<Side::SrcB, E>();
    constexpr uint32_t curr_p = dfb_for_side<Side::Pack, E>();

    constexpr uint32_t prev_a =
        (curr_a != NO_PREV_DFB) ? ChainTraits<Es...>::prev.srca[I] : NO_PREV_DFB;
    constexpr uint32_t prev_b =
        (curr_b != NO_PREV_DFB) ? ChainTraits<Es...>::prev.srcb[I] : NO_PREV_DFB;
    constexpr uint32_t prev_p =
        (curr_p != NO_PREV_DFB) ? ChainTraits<Es...>::prev.pack[I] : NO_PREV_DFB;

    constexpr bool reconf_a = (curr_a != NO_PREV_DFB) && (curr_a != prev_a);
    constexpr bool reconf_b = (curr_b != NO_PREV_DFB) && (curr_b != prev_b);
    constexpr bool reconf_p = (curr_p != NO_PREV_DFB) && (curr_p != prev_p);

    // Pack-side deferral: in heterogeneous chains, only the first opt-in pack
    // site (prev_p == NO_PREV_DFB) emits at boot. Later sites defer to per-stage
    // via `emit_per_stage_pack_reconfig`, where the 2-arg LLK form's cache check
    // handles intra-stage transitions cheaply and the per-iter wraparound from
    // last-pack-cb to first-pack-cb is correctly programmed.
    constexpr bool defer_pack_to_per_stage =
        ChainTraits<Es...>::pack_hetero && (prev_p != NO_PREV_DFB);

    // ---- srca + srcb: coalesce when both sides share prev-state ----
    if constexpr (reconf_a && reconf_b) {
        if constexpr (prev_a != NO_PREV_DFB && prev_b != NO_PREV_DFB) {
            // both sides have prev → 4-arg _with_dt
            reconfig_data_format(prev_a, curr_a, prev_b, curr_b);
        } else if constexpr (prev_a == NO_PREV_DFB && prev_b == NO_PREV_DFB) {
            // first-emit on both sides → 2-arg combined (unconditional reprogram)
            reconfig_data_format(curr_a, curr_b);
        } else {
            // mixed prev-state → independent per-side
            if constexpr (prev_a != NO_PREV_DFB) {
                reconfig_data_format_srca(prev_a, curr_a);
            } else {
                reconfig_data_format_srca(curr_a);
            }
            if constexpr (prev_b != NO_PREV_DFB) {
                reconfig_data_format_srcb(prev_b, curr_b);
            } else {
                reconfig_data_format_srcb(curr_b);
            }
        }
    } else if constexpr (reconf_a) {
        if constexpr (prev_a != NO_PREV_DFB) {
            reconfig_data_format_srca(prev_a, curr_a);
        } else {
            reconfig_data_format_srca(curr_a);
        }
    } else if constexpr (reconf_b) {
        if constexpr (prev_b != NO_PREV_DFB) {
            reconfig_data_format_srcb(prev_b, curr_b);
        } else {
            reconfig_data_format_srcb(curr_b);
        }
    }

    // ---- pack: always independent; deferred to per-stage in heterogeneous chains ----
    if constexpr (reconf_p && !defer_pack_to_per_stage) {
        if constexpr (prev_p != NO_PREV_DFB) {
            pack_reconfig_data_format(prev_p, curr_p);
        } else {
            pack_reconfig_data_format(curr_p);
        }
    }
}

// emit_per_stage_pack_reconfig<E, I, Es...>()
//
// Per-stage pack reconfig used only when the chain has heterogeneous opt-in
// pack CBs (boot can't program all of them). For every opt-in pack site,
// emit the 2-arg `pack_reconfig_data_format(prev, curr)` form before that
// site's per-iter pack work, using wraparound `prev` for the first pack site
// (so iter k+1's site 0 sees iter k's site N-1 as the previous descriptor
// state). The LLK's compare-and-skip on matching formats keeps the cost to
// a few cycles when adjacent stages happen to share a dtype.
template <class E, std::size_t I, class... Es>
ALWI void emit_per_stage_pack_reconfig() {
    if constexpr (!ChainTraits<Es...>::pack_hetero) return;
    constexpr uint32_t curr_p = dfb_for_side<Side::Pack, E>();
    if constexpr (curr_p == NO_PREV_DFB) return;
    constexpr uint32_t prev_chain = ChainTraits<Es...>::prev.pack[I];
    // Wraparound: first opt-in pack site has no in-chain prev; on iter ≥ 1 the
    // packer ended the previous iter on `last_pack_cb`. The LLK 2-arg form does
    // the right thing on iter 0 too (cache check vs. boot-initialized state).
    constexpr uint32_t prev_p =
        (prev_chain != NO_PREV_DFB) ? prev_chain : ChainTraits<Es...>::last_pack_cb;
    if constexpr (prev_p != NO_PREV_DFB) {
        pack_reconfig_data_format(prev_p, curr_p);
    }
}

// Pack-phase init (Pack* only). Pack is its own cohort (disjoint from math-MOP / SFPU),
// excluded from `hoist_compute_init` and always boot-hoisted here via `pack_init_for_each`.
// Reconfig is fold-driven (see emit_pre_element_transitions): homogeneous chains program
// the packer once at boot; heterogeneous chains defer later sites to per-stage emission so
// the per-iter wraparound stays correct.
template <std::size_t I, class E, class... Es>
ALWI void elem_pack_init() {
    if constexpr (is_pack_tile_op_v<E>) {
        emit_pre_element_transitions<E, I, Es...>();
        E::init();
    }
}

// Hoisted pack-init dispatcher — visits each chain element by compile-time index
// and forwards (Is, Es, Es...) into the per-element pack init.
template <class... Es, std::size_t... Is>
ALWI void pack_init_for_each(std::index_sequence<Is...>) {
    (elem_pack_init<Is, Es, Es...>(), ...);
}

// =============================================================================
// Two-phase per-element apply: compute / pack
//
// Each element owns its full lifecycle slice of the outer iteration. Per outer iter:
//
//   tile_regs_acquire();
//   apply_compute_phase(...);   // per element: wait + init? + for(j) exec + pop
//   tile_regs_commit();
//   tile_regs_wait();
//   apply_pack_phase(...);      // per pack element: reserve + for(j) pack_exec + push
//   tile_regs_release();
//
// Upfront-policy lifecycle (elem_pop_upfront_end / elem_push_at_end) fires after the loop.
//
// pop_per_tile / push_per_tile live inside the apply body. cb_pop_front right after exec
// is safe — by then the unpack-read is queued and the framework manages in-flight reads
// vs producer L1 reuse; push right after pack_exec wakes the consumer while DEST is still
// held. Both are policy-guarded (no-op for upfront / no-pop / no-push policies).
// =============================================================================

// Boot-time hoist of compute-cohort init (math-MOP and/or SFPU), filtered
// per element by which cohort it belongs to. The chain dispatcher computes
// `HoistMath` from `chain_hoist_math_mop_v` and `HoistSfpu` from
// `chain_hoist_sfpu_v`, then this walk emits the element's transitions +
// init() only when the element's cohort is hoisted at this level.
//
// PackTile is intentionally excluded from this walk — pack-side reconfig is
// emitted unconditionally at boot via `pack_init_for_each` (PACK cohort is
// disjoint from compute cohorts and is always hoisted).
template <bool HoistMath, bool HoistSfpu, std::size_t... Is, class... Es>
ALWI void hoist_compute_init(std::index_sequence<Is...>, Es&... elts) {
    auto run_one = [&](auto idx, auto& elem) {
        constexpr std::size_t II = decltype(idx)::value;
        using ElemT = std::remove_reference_t<decltype(elem)>;
        constexpr bool emit =
            (is_math_mop_op_v<ElemT> && HoistMath) ||
            (is_dest_only_op_v<ElemT> && HoistSfpu);
        if constexpr (emit) {
            emit_pre_element_transitions<ElemT, II, Es...>();
            ElemT::init();
        }
        (void)elem;
    };
    (run_one(std::integral_constant<std::size_t, Is>{}, elts), ...);
}

}  // namespace detail

// =============================================================================
// 11. Public eltwise_chain()
//
// Caller-init contract:
//   - Caller writes `compute_kernel_hw_startup(dfb_a, dfb_b, cb_out)` (or its
//     unary/binary variant) as the FIRST statement of `MAIN()`.
//   - Helper does NOT wrap any "BIG init" (`compute_kernel_hw_startup`,
//     `binary_op_init_common`, `mm_init`, `reduce_init`).
//   - Helper owns per-element init only — `add_tiles_init`, `*_tile_init`,
//     `init_bcast`, `copy_tile_to_dst_init_short`, `reconfig_data_format_*`,
//     `tile_regs_*` lifecycle.
//   - Per `compute_kernel_hw_startup.h:26-30`, mid-`MAIN()` boot is undefined.
//     Multi-stage kernels are the only exception (one boot per stage,
//     immediately before that stage's chain call).
// =============================================================================

// =============================================================================
// 11b. eltwise_chain — unified (Ht, Wt) walk with per-element broadcast indexing
//
// Walks an (Ht, Wt) tile grid (Ht=1 expresses the 1D case). Inner loop blocks W
// (block_size tiles per inner iter). Per-element index mode picks the tile index
// for each CB-reader: BlockIter → flat (ht*Wt + wt), RowBcast → wt, ColBcast → ht,
// FirstTile → 0.
// =============================================================================

namespace detail {

template <bool EmitMathInit, bool EmitSfpuInit, std::size_t I, class ElemT, class... Es>
ALWI void elem_apply_compute(
    const ElemT& elem,
    uint32_t i_flat,
    uint32_t ht,
    uint32_t wt,
    uint32_t inner_count,
    uint32_t chain_lane_width,
    uint32_t Ht,
    uint32_t Wt) {
    // Per-block streaming: pass chunk-local index `j` to exec so BlockIter
    // returns the local CB-front offset (the just-waited window).
    constexpr bool use_local_idx = element_uses_per_block_index_v<ElemT>;
    if constexpr (is_pack_tile_op_v<ElemT>) {
        (void)elem; (void)i_flat; (void)ht; (void)wt; (void)inner_count;
        (void)chain_lane_width; (void)Ht; (void)Wt;
    } else if constexpr (is_cb_reader_op_v<ElemT>) {
        // InputLifecycle::Streaming wait fires per-tile (Block walks); upfront wait is idempotent.
        elem.wait_per_tile(i_flat + inner_count);
        elem_wait_per_block(elem, inner_count);
        elem.wait_upfront(Ht, Wt);
        if constexpr (EmitMathInit) {
            emit_pre_element_transitions<ElemT, I, Es...>();
            ElemT::init();
        }
        constexpr bool per_side = elem_needs_per_side_idx_v<ElemT>;
        for (uint32_t j = 0; j < inner_count; ++j) {
            if constexpr (per_side) {
                // Per-side path: chain hands both indices; element picks per operand.
                elem.exec(/*i_flat_local=*/j,
                             /*i_flat_abs=*/(i_flat + j),
                             ht,
                             /*wt_local=*/j,
                             /*wt_abs=*/(wt + j),
                             j * chain_lane_width);
            } else {
                const uint32_t i_arg = use_local_idx ? j : (i_flat + j);
                elem.exec(i_arg, ht, wt + j, j * chain_lane_width);
            }
        }
        elem.pop_per_tile(i_flat);
        elem_pop_per_block(elem, inner_count);
    } else if constexpr (is_dest_only_op_v<ElemT>) {
        if constexpr (EmitSfpuInit) {
            emit_pre_element_transitions<ElemT, I, Es...>();
            ElemT::init();
        }
        for (uint32_t j = 0; j < inner_count; ++j) {
            elem.exec(i_flat + j, j * chain_lane_width);
        }
    }
}

template <std::size_t I, class ElemT, class... Es>
ALWI void elem_apply_pack(
    const ElemT& elem,
    uint32_t i_flat,
    uint32_t ht,
    uint32_t wt,
    uint32_t inner_count,
    uint32_t chain_lane_width,
    uint32_t Ht,
    uint32_t Wt) {
    constexpr bool use_local_idx = element_uses_per_block_index_v<ElemT>;
    if constexpr (is_pack_tile_op_v<ElemT>) {
        (void)Ht; (void)Wt;  // upfront reserve is emitted once before the loop (see eltwise_chain_impl)
        emit_per_stage_pack_reconfig<ElemT, I, Es...>();
        elem.reserve_per_tile(i_flat);
        elem_reserve_per_block(elem, inner_count);
        for (uint32_t j = 0; j < inner_count; ++j) {
            const uint32_t i_arg = use_local_idx ? j : (i_flat + j);
            elem.exec(i_arg, ht, wt + j, j * chain_lane_width);
        }
        elem.push_per_tile(i_flat);
        elem_push_per_block(elem, inner_count);
    } else {
        (void)elem; (void)i_flat; (void)ht; (void)wt; (void)inner_count;
        (void)chain_lane_width; (void)Ht; (void)Wt;
    }
}

template <bool EmitMathInit, bool EmitSfpuInit, std::size_t... Is, class... Es>
ALWI void apply_compute_phase(
    std::index_sequence<Is...>,
    uint32_t i_flat,
    uint32_t ht,
    uint32_t wt,
    uint32_t inner_count,
    uint32_t chain_lane_width,
    uint32_t Ht,
    uint32_t Wt,
    Es&... elts) {
    auto run_one = [&](auto idx_const, auto& elem) {
        constexpr std::size_t II = decltype(idx_const)::value;
        using ElemT = std::remove_reference_t<decltype(elem)>;
        elem_apply_compute<EmitMathInit, EmitSfpuInit, II, ElemT, Es...>(
            elem, i_flat, ht, wt, inner_count, chain_lane_width, Ht, Wt);
    };
    (run_one(std::integral_constant<std::size_t, Is>{}, elts), ...);
}

template <std::size_t... Is, class... Es>
ALWI void apply_pack_phase(
    std::index_sequence<Is...>,
    uint32_t i_flat,
    uint32_t ht,
    uint32_t wt,
    uint32_t inner_count,
    uint32_t chain_lane_width,
    uint32_t Ht,
    uint32_t Wt,
    Es&... elts) {
    auto run_one = [&](auto idx_const, auto& elem) {
        constexpr std::size_t II = decltype(idx_const)::value;
        using ElemT = std::remove_reference_t<decltype(elem)>;
        elem_apply_pack<II, ElemT, Es...>(
            elem, i_flat, ht, wt, inner_count, chain_lane_width, Ht, Wt);
    };
    (run_one(std::integral_constant<std::size_t, Is>{}, elts), ...);
}

template <class E>
ALWI void elem_reserve_upfront(const E& e, uint32_t Ht, uint32_t Wt) {
    if constexpr (is_cb_writer_op_v<E>) e.reserve_upfront(Ht, Wt);
}
template <class E>
ALWI void elem_pop_upfront_end(const E& e, uint32_t Ht, uint32_t Wt) {
    if constexpr (is_cb_reader_op_v<E>) e.pop_upfront_end(Ht, Wt);
}
template <class E>
ALWI void elem_push_at_end(const E& e, uint32_t Ht, uint32_t Wt) {
    if constexpr (is_cb_writer_op_v<E>) e.push_at_end(Ht, Wt);
}

}  // namespace detail

template <class... Es>
ALWI void eltwise_chain_impl(EltwiseShape shape, Es... elts) {
    using Chain = EltwiseChain<Es...>;

    // ---- Compile-time invariant checks ----
    static_assert(!chain_has_duplicate_upfront_cbs_v<Chain>,
                  "eltwise_chain: two CB-reader elements share a CB on upfront-wait policy.");
    static_assert(!chain_pack_writes_collide_v<Chain>,
                  "eltwise_chain: two PackTile elements collide on (dfb, dst_slot).");

    // Per-cohort hoist decisions. The dispatcher picks the most aggressive safe
    // emission shape — math-MOP init can be hoisted at boot even when SFPU isn't
    // uniform; the SFPU side then re-inits per tile.
    constexpr bool hoist_math = chain_hoist_math_mop_v<Chain>;
    constexpr bool hoist_sfpu = chain_hoist_sfpu_v<Chain>;

    // Block size lives on the shape. The DEST footprint is block_size * chain_lane_width;
    // the chain clamps block_size so it can never overflow DEST (see below).
    constexpr uint32_t chain_lane_w = chain_lane_width_v<Chain>;
    uint32_t block_size = shape.block_size;
    // InputLifecycle::Streaming CB-reader chains can't multi-tile their DEST window (WaitAndPop /
    // WaitNoPop / InputLifecycle::NoWaitPop consume one tile per iter). Force block_size to 1 in that
    // case — compile-time gated, so the override path emits no code when the chain
    // supports block-mode.
    if constexpr (!chain_supports_block_v<Chain>) {
        block_size = 1;
    } else {
        // Clamp the runtime block_size to the chain's compile-time DEST capacity
        // (chain_max_block_v = DEST_AUTO_LIMIT / chain_lane_width). block_size is a
        // runtime field so a static_assert isn't possible; an oversized value would
        // otherwise silently overflow DEST. Clamping down is correctness-safe — it
        // only makes the outer loop take more iterations; total tile coverage is
        // unchanged.
        constexpr uint32_t max_block = chain_max_block_v<Chain>;
        if (block_size > max_block) {
            block_size = max_block;
        }
    }

    using IdxSeq = std::make_index_sequence<sizeof...(Es)>;

    // Pack-cohort boot init — always hoisted.
    detail::pack_init_for_each<Es...>(IdxSeq{});

    // Compute-cohort boot init — filtered per cohort.
    detail::hoist_compute_init<hoist_math, hoist_sfpu>(IdxSeq{}, elts...);

    const uint32_t Ht = shape.Ht;
    const uint32_t Wt = shape.Wt;

    // Upfront output reserve — fires once for the whole Ht*Wt window for the upfront-reserve
    // policies (Bulk, BulkReservePerTile, BulkReservePerChunk), mirroring the end-of-chain
    // push/pop folds below. Per-tile / per-chunk-reserve policies emit nothing here.
    (detail::elem_reserve_upfront(elts, Ht, Wt), ...);

    // Outer 2D loop. `flat_base = ht * Wt + wt_base` is computed once per (ht, wt_base)
    // pair — single MUL on the inner-W path. Block-mode elements consume `flat_base + j`
    // directly; bcast-mode elements read `ht` or `wt = wt_base + j` instead. No
    // per-tile multiplication inside the element's exec.
    for (uint32_t ht = 0; ht < Ht; ++ht) {
        const uint32_t row_base = ht * Wt;
        for (uint32_t wt_base = 0; wt_base < Wt; wt_base += block_size) {
            const uint32_t inner_count =
                (wt_base + block_size <= Wt) ? block_size : (Wt - wt_base);
            const uint32_t i_flat = row_base + wt_base;
            tile_regs_acquire();
            detail::apply_compute_phase</*EmitMathInit=*/!hoist_math,
                                            /*EmitSfpuInit=*/!hoist_sfpu>(
                IdxSeq{}, i_flat, ht, wt_base, inner_count, chain_lane_w, Ht, Wt, elts...);
            tile_regs_commit();
            tile_regs_wait();
            detail::apply_pack_phase(
                IdxSeq{}, i_flat, ht, wt_base, inner_count, chain_lane_w, Ht, Wt, elts...);
            tile_regs_release();
        }
    }

    // End-of-chain upfront-policy lifecycle.
    (detail::elem_pop_upfront_end(elts, Ht, Wt), ...);
    (detail::elem_push_at_end(elts, Ht, Wt), ...);
}

// =============================================================================
// 11c. Public eltwise_chain — strips compile-time-disabled optional elements (those
// carrying `is_disabled = true`, i.e. OptionalChainElement<false, _>) from the pack before
// any stage runs, so the impl and every stage (static_asserts, hoist, reconfig fold,
// per-tile loop) only ever see enabled elements. Detection is member-based, so the chain
// needs no knowledge of OptionalChainElement (the dependency runs one way).
//
// Each chain_keep yields a 0- or 1-tuple; we tuple-cat and expand via a direct std::get<I>
// call into eltwise_chain_impl. We deliberately do NOT use std::apply — its INVOKE
// indirection routes through a callable and can defeat `always_inline`, which on a Tensix
// MATH kernel pushes the compute body out of line and miscompiles it (nan). The std::get
// expansion calls eltwise_chain_impl directly, no closure.
// =============================================================================

template <class T, class = void>
struct chain_element_disabled : std::false_type {};
template <class T>
struct chain_element_disabled<T, std::void_t<decltype(T::is_disabled)>>
    : std::bool_constant<T::is_disabled> {};

template <class E>
ALWI auto chain_keep(E e) {
    if constexpr (chain_element_disabled<E>::value) {
        return std::tuple<>{};
    } else {
        return std::tuple<E>{e};
    }
}

template <class Tup, std::size_t... I>
ALWI void chain_dispatch(EltwiseShape shape, Tup tup, std::index_sequence<I...>) {
    eltwise_chain_impl(shape, std::get<I>(tup)...);
}

template <class... Es>
ALWI void eltwise_chain(EltwiseShape shape, Es... elts) {
    auto kept = std::tuple_cat(chain_keep(elts)...);
    chain_dispatch(shape, kept, std::make_index_sequence<std::tuple_size_v<decltype(kept)>>{});
}

// =============================================================================
// 12. dfb_a_of / dfb_b_of / pack_dfb_of — single-element CB id deducers used by the
// collision-detection static_asserts (chain_has_duplicate_upfront_cbs_v,
// chain_pack_writes_collide_v). Caller-side hw_startup is the caller's
// responsibility — there is no deduced wrapper.
// =============================================================================

namespace detail {

template <class E>
constexpr uint32_t dfb_a_of() {
    if constexpr (is_cb_reader_op_v<E>) {
        static_assert(has_dfb_a<E>::value,
                      "CbReader element must declare 'static constexpr uint32_t dfb_a_id()'");
        return E::dfb_a_id();
    } else {
        return INVALID_DFB;
    }
}

template <class E>
constexpr uint32_t dfb_b_of() {
    if constexpr (is_binary_fpu_op_v<E> || is_dest_reuse_binary_op_v<E>) {
        static_assert(has_dfb_b<E>::value,
                      "Binary CbReader element must declare 'static constexpr uint32_t dfb_b_id()'");
        return E::dfb_b_id();
    } else {
        return INVALID_DFB;
    }
}

template <class E>
constexpr uint32_t pack_dfb_of() {
    if constexpr (is_pack_tile_op_v<E>) {
        static_assert(has_pack_dfb<E>::value,
                      "CbWriter element must declare 'static constexpr uint32_t pack_dfb_id()'");
        return E::pack_dfb_id();
    } else {
        return INVALID_DFB;
    }
}

}  // namespace detail

}  // namespace compute_kernel_lib
