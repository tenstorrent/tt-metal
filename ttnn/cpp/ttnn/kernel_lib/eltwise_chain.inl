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
template <class Chain> struct chain_hoist_pack;

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
template <class Chain>
inline constexpr bool chain_hoist_pack_v = chain_hoist_pack<Chain>::value;

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
ALWI uint32_t tile_base_value([[maybe_unused]] uint32_t stored) noexcept {
    if constexpr (Offset == TileOffset::Unset) {
        return 0u;
    } else {
        return stored;
    }
}

// =============================================================================
// CRTP bases — UnaryOp / BinaryOp / TernaryOp
// =============================================================================
//
// Dispatch contract: the DEST-only op bases expose `void exec(uint32_t i, uint32_t slot_offset) const`
// and forward to a static `exec_impl(uint32_t slot_offset)` supplied by the derived op; runtime-param
// ops (Power, Hardtanh, …) override `exec` directly to capture instance state. (CB elements — CopyTile /
// BinaryFpu — define a wider exec: (i_flat, ht, wt, slot_offset) or the per-side form.) Defining
// neither is a compile error (no silent fallthrough).
//
//   template <Approx A = Approx::Exact, Approx F = Approx::Fast, Dst Slot = Dst::D0>
//   struct Exp : UnaryOp<Exp<A, F, Slot>, Slot> {
//       static void init()                        { exp_tile_init<A == Approx::Fast, F == Approx::Fast>(); }
//       static void exec_impl(uint32_t slot_off)  { exp_tile<A == Approx::Fast, F == Approx::Fast>(to_u32(Slot) +
//       slot_off); }
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

// Compile-time membership test: is `V` one of `Set...`? Empty set → false. Collapses the
// repeated `X == A || X == B || ...` equality ladders into `is_one_of_v<X, A, B, ...>`.
// `auto V` + `decltype(V)...` lets one definition serve every enum (InputLifecycle,
// OutputLifecycle, OperandKind, reconfig enums, …); the fold is a constant expression, usable in
// `if constexpr`, `static_assert`, and `static constexpr` members. Requires V and Set... to be
// constant expressions — it does NOT apply to runtime values or function parameters.
template <auto V, decltype(V)... Set>
inline constexpr bool is_one_of_v = ((V == Set) || ...);

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
inline constexpr bool is_bcast_mode_v = is_one_of_v<M, OperandKind::Row, OperandKind::Col>;

template <OperandKind M>
ALWI constexpr uint32_t idx(
    [[maybe_unused]] uint32_t i_flat, [[maybe_unused]] uint32_t ht, [[maybe_unused]] uint32_t wt) noexcept {
    if constexpr (M == OperandKind::Scalar) {
        return 0;
    } else if constexpr (M == OperandKind::Block) {
        return i_flat;
    } else if constexpr (M == OperandKind::Row) {
        return wt;
    } else {
        return ht;  // Col
    }
}

template <OperandKind M>
ALWI constexpr uint32_t window([[maybe_unused]] uint32_t Ht, [[maybe_unused]] uint32_t Wt) noexcept {
    if constexpr (M == OperandKind::Block) {
        return Ht * Wt;
    } else if constexpr (M == OperandKind::Row) {
        return Wt;
    } else if constexpr (M == OperandKind::Col) {
        return Ht;
    } else {
        return 1u;  // Scalar
    }
}

// Allowed (Policy × Mode) combinations. Row/Col cannot stream per-tile —
// the producer must stage the full row/col upfront. Matches the
// `binary_op_helpers` static_assert (ROW/SCALAR require InputLifecycle::Bulk-family or NoWait*).
template <InputLifecycle P, OperandKind M>
inline constexpr bool valid_policy_mode_v =
    !(is_bcast_mode_v<M> && is_one_of_v<P, InputLifecycle::Streaming, InputLifecycle::Chunked>);

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
// Every CB-reader element exposes dfb_a_id() (primary CB) + a_policy(). Binary readers also expose
// dfb_b_id() (secondary CB) + b_policy(); unary readers omit them — the defaults below supply
// dfb_b = INVALID_DFB (0xFFFFFFFF, "no CB"; not 0, which is a real CB) and b_policy = CallerManaged.
// (default impls below cover non-CB-reader elements.)
//
// Every CB-writer element must expose:
//   static constexpr uint32_t pack_dfb_id();
//   static constexpr Dst pack_dst_slot;
// (output addressing derives from the walk + TileOffset, not a pack_output_index accessor.)
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
        if constexpr (has_reconfig_srca<E>::value) {
            return E::reconfig_srca_dfb;
        } else {
            return NO_PREV_DFB;
        }
    } else if constexpr (S == Side::SrcB) {
        if constexpr (has_reconfig_srcb<E>::value) {
            return E::reconfig_srcb_dfb;
        } else {
            return NO_PREV_DFB;
        }
    } else {  // Pack
        if constexpr (has_reconfig_pack<E>::value) {
            return E::reconfig_pack_dfb;
        } else {
            return NO_PREV_DFB;
        }
    }
}

// Packer ReLU mode of an element — PackRelu::None for anything that isn't a PackTile (only pack
// sites carry the packer-global ReLU knob). Reflected into ElemDesc and used to detect a chain that
// mixes ReLU modes across pack sites, and to forbid an inert ReLU knob under SetupOwner::Caller.
template <class E>
constexpr PackRelu pack_relu_of() {
    if constexpr (is_pack_tile_op_v<E>) {
        return E::pack_relu;
    } else {
        return PackRelu::None;
    }
}

// True iff NO element in the pack requests any srca / srcb / pack reconfig — i.e. every element's
// reconfig knob is None (and no packer ReLU). Under SetupOwner::Caller the chain emits zero setup, so
// a non-None knob is inert and lies about what the helper does; eltwise_chain uses this to forbid
// that, forcing the caller to declare None — which honestly reflects "the chain does no reconfig, I
// own the format."
template <class... Es>
constexpr bool chain_requests_no_reconfig() {
    return ((dfb_for_side<Side::SrcA, Es>() == NO_PREV_DFB &&
             dfb_for_side<Side::SrcB, Es>() == NO_PREV_DFB &&
             dfb_for_side<Side::Pack, Es>() == NO_PREV_DFB &&
             pack_relu_of<Es>() == PackRelu::None) &&
            ...);
}

// Per-side prev-CB history, last opt-in pack CB, and heterogeneous-pack detection are
// single-sweep fields on `ChainTraits` (prev / last_pack_cb / pack_dtype_hetero /
// pack_relu_hetero / any_pack_relu), computed once from the reflected ElemDesc array.

}  // namespace detail

// =============================================================================
// 0. Operand streams — the shared CB-lifecycle hook providers
//
// Every CB-reader element's wait/pop hooks are a function of exactly one operand
// tuple (Cb, Policy, IndexMode, Offset, tile_base); every CB-writer's reserve/push
// hooks of one (Cb, Policy, Offset, tile_base). `InputStream` / `OutputStream`
// define each hook ONCE. Single-stream elements (CopyTile / DestReuseBinary /
// UnaryBcast) inherit `InputStream`; BinaryFpu HOLDS two and fans out with a
// `same_dfb` dedup; PackTile inherits `OutputStream`. The driver calls the same
// hook names either way (inherited member, or the element's own fan-out override).
//
// All policy gates below are a verbatim transcription of the per-element bodies —
// a wrong gate is a hang or a PCC failure, never benign.
// =============================================================================

template <uint32_t Cb,
          InputLifecycle Policy,
          OperandKind IndexMode = OperandKind::Block,
          TileOffset Offset = TileOffset::Unset>
struct InputStream {
    uint32_t tile_base = 0;

    constexpr InputStream() noexcept = default;
    constexpr explicit InputStream(uint32_t base) noexcept : tile_base(base) {}

    // Upfront (Bulk-family) window count: the index-mode window plus the runtime base.
    // (UnaryBcast uses IndexMode=Block + Offset=Unset, which yields exactly Ht*Wt.)
    ALWI uint32_t window_count(uint32_t Ht, uint32_t Wt) const {
        return detail::window<IndexMode>(Ht, Wt) + tile_base_value<Offset>(tile_base);
    }

    // Gate-free count primitives — reused by BinaryFpu's same_dfb branch, which
    // collapses both sides' bases into the single shared CB's window.
    ALWI void wait_n(uint32_t n) const { DataflowBuffer(Cb).wait_front(n); }
    ALWI void pop_n(uint32_t n) const { DataflowBuffer(Cb).pop_front(n); }

    ALWI void wait_per_tile(uint32_t cumulative_count) const {
        if constexpr (is_one_of_v<Policy, InputLifecycle::Streaming, InputLifecycle::HeldStream>) {
            DataflowBuffer(Cb).wait_front(1);
        } else if constexpr (is_one_of_v<Policy, InputLifecycle::Pipelined, InputLifecycle::HeldCumulative>) {
            DataflowBuffer(Cb).wait_front(cumulative_count);
        }
    }
    ALWI void wait_per_block(uint32_t inner_count) const {
        if constexpr (is_one_of_v<Policy, InputLifecycle::Chunked>) {
            DataflowBuffer(Cb).wait_front(inner_count);
        }
    }
    ALWI void wait_upfront(uint32_t Ht, uint32_t Wt) const {
        if constexpr (is_one_of_v<Policy, InputLifecycle::BulkDrain>) {
            // BulkDrain pops one tile per iter with no per-tile wait, so it must wait for the WHOLE
            // walk (Ht*Wt, + TileOffset base) upfront — NOT window<IndexMode>, which is 1 for the
            // Scalar index BulkDrain is restricted to. Waiting 1 would silently rely on the producer
            // having already staged every tile.
            DataflowBuffer(Cb).wait_front(Ht * Wt + tile_base_value<Offset>(tile_base));
        } else if constexpr (is_one_of_v<Policy, InputLifecycle::Bulk, InputLifecycle::HeldBulk>) {
            // Bulk stages its window once and pops at end; HeldBulk holds it — a held Scalar operand
            // legitimately waits window<Scalar>=1 (see the Bulk+Scalar held-operand contract).
            DataflowBuffer(Cb).wait_front(window_count(Ht, Wt));
        }
    }
    ALWI void pop_upfront_end(uint32_t Ht, uint32_t Wt) const {
        if constexpr (is_one_of_v<Policy, InputLifecycle::Bulk, InputLifecycle::Pipelined, InputLifecycle::DeferredPop>) {
            DataflowBuffer(Cb).pop_front(window_count(Ht, Wt));
        }
    }
    ALWI void pop_per_tile(uint32_t /*i*/) const {
        if constexpr (is_one_of_v<Policy, InputLifecycle::Streaming, InputLifecycle::NoWaitPop, InputLifecycle::BulkDrain>) {
            DataflowBuffer(Cb).pop_front(1);
        }
    }
    ALWI void pop_per_block(uint32_t inner_count) const {
        if constexpr (is_one_of_v<Policy, InputLifecycle::Chunked>) {
            DataflowBuffer(Cb).pop_front(inner_count);
        }
    }
    ALWI void wait_per_row() const {
        if constexpr (is_one_of_v<Policy, InputLifecycle::OuterStream>) {
            DataflowBuffer(Cb).wait_front(1);
        }
    }
    ALWI void pop_per_row() const {
        if constexpr (is_one_of_v<Policy, InputLifecycle::OuterStream>) {
            DataflowBuffer(Cb).pop_front(1);
        }
    }
};

template <uint32_t Cb,
          OutputLifecycle Policy,
          TileOffset Offset = TileOffset::Unset>
struct OutputStream {
    uint32_t tile_base = 0;

    constexpr OutputStream() noexcept = default;
    constexpr explicit OutputStream(uint32_t base) noexcept : tile_base(base) {}

    // Walk vs pinned output addressing is DERIVED from the OutputLifecycle (no caller knob):
    // upfront-reserve policies reserve the whole window once and write distinct tiles into it
    // (walk); per-tile/per-chunk-reserve policies advance the CB front, so the index stays pinned.
    static constexpr bool walk =
        is_one_of_v<Policy, OutputLifecycle::Bulk, OutputLifecycle::ReserveAllPushPerTile, OutputLifecycle::ReserveAllPushPerChunk>;

    ALWI void reserve_per_tile(uint32_t /*i*/) const {
        if constexpr (Policy == OutputLifecycle::Streaming) {
            DataflowBuffer(Cb).reserve_back(1);
        }
    }
    ALWI void reserve_per_block(uint32_t inner_count) const {
        if constexpr (is_one_of_v<Policy, OutputLifecycle::Chunked>) {
            DataflowBuffer(Cb).reserve_back(inner_count);
        }
    }
    ALWI void reserve_upfront(uint32_t Ht, uint32_t Wt) const {
        if constexpr (is_one_of_v<Policy, OutputLifecycle::Bulk, OutputLifecycle::ReserveAllPushPerTile, OutputLifecycle::ReserveAllPushPerChunk>) {
            DataflowBuffer(Cb).reserve_back((Ht * Wt) + tile_base_value<Offset>(tile_base));
        } else if constexpr (Policy == OutputLifecycle::L1Accumulation) {
            DataflowBuffer(Cb).reserve_back(1);
        }
    }
    ALWI void push_at_end(uint32_t Ht, uint32_t Wt) const {
        if constexpr (is_one_of_v<Policy, OutputLifecycle::ReserveNonePushEnd, OutputLifecycle::Bulk>) {
            DataflowBuffer(Cb).push_back((walk ? (Ht * Wt) : 1u) + tile_base_value<Offset>(tile_base));
        } else if constexpr (Policy == OutputLifecycle::L1Accumulation) {
            DataflowBuffer(Cb).push_back(1);
        }
    }
    ALWI void push_per_tile(uint32_t /*i*/) const {
        if constexpr (is_one_of_v<Policy, OutputLifecycle::Streaming, OutputLifecycle::ReserveAllPushPerTile>) {
            DataflowBuffer(Cb).push_back(1);
        }
    }
    ALWI void push_per_block(uint32_t inner_count) const {
        if constexpr (is_one_of_v<Policy, OutputLifecycle::Chunked, OutputLifecycle::ReserveAllPushPerChunk>) {
            DataflowBuffer(Cb).push_back(inner_count);
        }
    }
    ALWI void reserve_per_row() const {
        if constexpr (Policy == OutputLifecycle::DestAccumulation) {
            DataflowBuffer(Cb).reserve_back(1);
        }
    }
    ALWI void push_per_row() const {
        if constexpr (Policy == OutputLifecycle::DestAccumulation) {
            DataflowBuffer(Cb).push_back(1);
        }
    }
};

// =============================================================================
// 1. CopyTile chain element
// =============================================================================

template <uint32_t Cb,
          Dst DstSlot,
          InputLifecycle Policy,
          CopyTileReconfig Reconfig,
          OperandKind IndexMode,
          TileOffset Offset>
struct CopyTile : InputStream<Cb, Policy, IndexMode, Offset>, CopyTileTag {
    using Base = InputStream<Cb, Policy, IndexMode, Offset>;
    using Base::tile_base;

    // ---- compile-time validation ----
    static_assert(to_u32(DstSlot) < DEST_AUTO_LIMIT, "CopyTile: DEST slot exceeds DEST_AUTO_LIMIT");
    // Comprehensive (IndexMode, Policy) legality. Block rejects PerTile-pop
    // (InputLifecycle::Streaming/InputLifecycle::BulkDrain/InputLifecycle::NoWaitPop — absolute-index pitfall) and
    // PerTile-wait-of-1 (InputLifecycle::HeldStream — never tracks per-iter requirement). Scalar/Row/Col accept every
    // legal lifecycle — caller-sized.
    static_assert(
        is_legal_kind_lifecycle(IndexMode, Policy),
        "CopyTile: (IndexMode, Policy) is illegal for Block — exclude "
        "InputLifecycle::Streaming / InputLifecycle::HeldStream / InputLifecycle::BulkDrain / "
        "InputLifecycle::NoWaitPop on Block walkers.");
    // 2D: RowBcast / ColBcast require non-streaming policy (matches binary_op_helpers ROW/SCALAR rule).
    static_assert(
        detail::valid_policy_mode_v<Policy, IndexMode>,
        "CopyTile: RowBcast / ColBcast index require non-streaming policy "
        "(WaitUpfrontPopAtEnd, WaitNoPop, InputLifecycle::NoWaitPop, NoWaitNoPop, CumulativeWaitPopAtEnd)");
    // TileOffset::Set requires InputLifecycle::Bulk-family / InputLifecycle::CallerManaged lifecycle — iter-dependent
    // counts
    // (InputLifecycle::Streaming/InputLifecycle::Chunked/Cumulative/Held{Stream,Cumulative}/InputLifecycle::NoWaitPop)
    // can't compose with runtime base offsets. Caller must size CB to base+window.
    static_assert(
        Offset == TileOffset::Unset || is_legal_input_lifecycle_with_base(Policy),
        "CopyTile: TileOffset::Set requires InputLifecycle::Bulk-family or InputLifecycle::CallerManaged lifecycle "
        "(InputLifecycle::Bulk / InputLifecycle::HeldBulk / InputLifecycle::DeferredPop / InputLifecycle::BulkDrain / "
        "InputLifecycle::CallerManaged)");

    static constexpr uint32_t dfb = Cb;
    static constexpr uint32_t dfb_a_id() { return Cb; }
    // CopyTile reads one CB front (srcA via dfb_a_id); dfb_b / b_policy absent -> defaults apply.
    static constexpr InputLifecycle a_policy() { return Policy; }
    static constexpr bool is_upfront =
        is_one_of_v<Policy, InputLifecycle::Bulk, InputLifecycle::HeldBulk, InputLifecycle::Pipelined>;

    // Prev-CB fold: CopyTile loads CbA only. srcb/pack sides are absent -> dfb_for_side
    // defaults them to NO_PREV_DFB.
    static constexpr uint32_t       reconfig_srca_dfb = (Reconfig == CopyTileReconfig::Input) ? Cb : NO_PREV_DFB;

    constexpr CopyTile() noexcept = default;
    constexpr explicit CopyTile(uint32_t base) noexcept : Base(base) {}

    // ---- chain pipeline hooks ----
    static ALWI void init() {
        copy_tile_init(Cb);
    }

    ALWI void exec(uint32_t i_flat, uint32_t ht, uint32_t wt, uint32_t slot_offset) const {
        const uint32_t in_idx = tile_base_value<Offset>(tile_base) + detail::idx<IndexMode>(i_flat, ht, wt);
        copy_tile(Cb, in_idx, to_u32(DstSlot) + slot_offset);
    }

    static constexpr uint32_t lane_width = to_u32(DstSlot) + 1;

    // wait_per_tile / wait_per_block / wait_upfront / pop_upfront_end / pop_per_tile /
    // pop_per_block / wait_per_row / pop_per_row inherited from InputStream.
};

// =============================================================================
// 2. PackTile chain element
// =============================================================================

template <uint32_t Cb,
          OutputLifecycle Policy,
          PackTileReconfig Reconfig,
          Dst DstSlot,
          TileOffset Offset,
          PackTileL1Accumulation L1Accumulation,
          PackRelu Relu>
struct PackTile : OutputStream<Cb, Policy, Offset>, PackTileTag {
    using Base = OutputStream<Cb, Policy, Offset>;
    using Base::tile_base;
    using Base::walk;

    static_assert(to_u32(DstSlot) < DEST_AUTO_LIMIT,
                  "PackTile: DEST slot exceeds DEST_AUTO_LIMIT");
    static_assert(is_legal_output_lifecycle(Policy),
                  "PackTile: output lifecycle is not one of the named legal OutputLifecycle values");
    static_assert(
        (L1Accumulation != PackTileL1Accumulation::Disabled) == is_l1_accumulation_output_lifecycle(Policy),
        "PackTile: L1 accumulation requires OutputLifecycle::L1Accumulation or "
        "OutputLifecycle::L1AccumulationCallerManaged, and those lifecycles require L1 accumulation");
    // Packer ReLU and L1 accumulation both write the packer's output; whether ReLU clamps the
    // accumulated sum or the pre-accumulation DEST value is unverified, so forbid the combination
    // for now (a chain-level assert forbids ReLU + DEST accumulation).
    static_assert(Relu == PackRelu::None || L1Accumulation == PackTileL1Accumulation::Disabled,
                  "PackTile: packer ReLU combined with L1 accumulation is not supported yet");
    // TileBase != None on pack side requires caller-managed-style lifecycle on the
    // output CB (caller pre-reserved a window large enough for base + kind window).
    // InputLifecycle::Streaming / InputLifecycle::Chunked reserve+push counts can't be inflated by a runtime base
    // without per-iter bookkeeping the chain doesn't own.
    static_assert(
        Offset == TileOffset::Unset || is_legal_output_lifecycle_with_base(Policy),
        "PackTile: TileOffset::Set requires an upfront or caller-managed output lifecycle "
        "(OutputLifecycle::Bulk / OutputLifecycle::ReserveNonePushEnd / OutputLifecycle::CallerManaged / "
        "OutputLifecycle::L1AccumulationCallerManaged)");

    static constexpr uint32_t dfb = Cb;
    static constexpr uint32_t pack_dfb_id() { return Cb; }
    static constexpr Dst pack_dst_slot = DstSlot;
    static constexpr PackRelu pack_relu = Relu;  // packer ReLU mode — reflected for hetero detection
    static constexpr bool uses_l1_accumulation = (L1Accumulation != PackTileL1Accumulation::Disabled);
    static constexpr bool seeds_l1_accumulation = (L1Accumulation == PackTileL1Accumulation::SeedFirst);
    static constexpr bool manages_l1_accumulation_lifecycle = (Policy == OutputLifecycle::L1Accumulation);
    static constexpr bool uses_dest_accumulation_lifecycle = is_dest_accumulation_output_lifecycle(Policy);
    static constexpr bool manages_dest_accumulation_lifecycle = (Policy == OutputLifecycle::DestAccumulation);
    static constexpr bool is_upfront = (Policy == OutputLifecycle::Bulk);
    static constexpr bool uses_per_block_pack = (Policy == OutputLifecycle::Chunked);
    // `walk` (walk vs pinned output addressing) is derived from the OutputLifecycle and
    // inherited from OutputStream (see `using Base::walk;` above).

    // Prev-CB fold: PackTile writes pack-side; mark Cb under reconfig only when
    // the user opted into pack reconfig (Output). Otherwise no pack reconfig is
    // emitted — fold keeps prior pack target.
    // srca/srcb absent -> dfb_for_side defaults them to NO_PREV_DFB; PackTile programs pack only.
    static constexpr uint32_t          reconfig_pack_dfb    =
        (Reconfig == PackTileReconfig::Output) ? Cb : NO_PREV_DFB;

    constexpr PackTile() noexcept = default;
    constexpr explicit PackTile(uint32_t base) noexcept : Base(base) {}

    static ALWI void init() {
        // Pack reconfig is fold-driven (compile-time-elided when prev_pack_cb == Cb).
        // The chain emits the reconfig in `emit_pre_element_transitions()` before this
        // element runs; init() here is a no-op for reconfig.
        // Retained empty so trait-dispatch stays uniform.
    }

    // Pack exec — walk the reserved output window (base + i_flat) for the upfront-reserve outputs
    // (Bulk / ReserveAllPushPerTile / ReserveAllPushPerChunk), or stay pinned at base for the
    // front-advancing policies (Streaming / Chunked) whose CB front already advanced. TileOffset adds base.
    //
    // OOO gating: the LLK's sequential pack path (out_of_order_output=false) derives its write
    // address from an internal running `fifo_wr_tile_ptr` and IGNORES `out_idx` entirely. That is
    // correct only when the intended index coincides with the sequential counter — i.e. when there
    // is no base offset (walk: 0,1,2,…; pinned: 0). The moment `Offset == Set`, `out_idx` carries a
    // non-coincident base that the sequential path would silently drop (data lands at index 0, not
    // base). L1 accumulation likewise has to keep every pack pinned to the same output tile. So we
    // switch to `pack_tile<true>` for `TileOffset::Set` or L1 accumulation, which honors `out_idx`
    // (addr = fifo_wr_ptr + page_size*out_idx - 1) without advancing the internal counter — exactly
    // matching the explicit `base + i_flat` we pass each iteration. Unset keeps the proven
    // sequential path with zero behavior change.
    ALWI void exec(uint32_t i_flat, uint32_t /*ht*/, uint32_t /*wt*/, uint32_t slot_offset) const {
        const uint32_t base = tile_base_value<Offset>(tile_base);
        const uint32_t out_idx = walk ? (base + i_flat) : base;
        pack_tile</*out_of_order_output=*/Offset == TileOffset::Set ||
                  L1Accumulation != PackTileL1Accumulation::Disabled>(
            to_u32(DstSlot) + slot_offset, Cb, out_idx);
    }

    static constexpr uint32_t lane_width = to_u32(DstSlot) + 1;

    // reserve_per_tile / reserve_per_block / reserve_upfront / push_at_end / push_per_tile /
    // push_per_block inherited from OutputStream.
};

// =============================================================================
// 3. BinaryFpu chain element
// =============================================================================

template <
    uint32_t CbA,
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
    TileOffset OffsetB,
    DestAccumulation Accumulation>
struct BinaryFpu : BinaryFpuTag {
    static_assert(to_u32(DstSlot) < DEST_AUTO_LIMIT, "BinaryFpu: DEST slot exceeds DEST_AUTO_LIMIT");
    static_assert(
        Accumulation == DestAccumulation::Disabled || DstSlot == Dst::D0,
        "BinaryFpu: DEST accumulation currently requires the single sticky slot Dst::D0");
#ifdef ARCH_QUASAR
    // TODO: Re-enable once Quasar's broadcast LLK forwards acc_to_dest instead of ignoring it.
    // static_assert(
    //     Accumulation == DestAccumulation::Disabled || Bcast == BroadcastDim::None,
    //     "BinaryFpu: Quasar does not support acc_to_dest on the broadcast FPU path");
#endif
    // Comprehensive per-side (IndexMode, Policy) legality. Block rejects PerTile-pop
    // (InputLifecycle::Streaming/InputLifecycle::BulkDrain/InputLifecycle::NoWaitPop — absolute-index pitfall) and
    // PerTile-wait-of-1 (InputLifecycle::HeldStream — never tracks per-iter requirement). Scalar/Row/Col accept every
    // legal lifecycle — caller-sized.
    static_assert(
        is_legal_kind_lifecycle(AIndex, APolicy),
        "BinaryFpu: (AIndex, APolicy) is illegal for Block — exclude "
        "InputLifecycle::Streaming / InputLifecycle::HeldStream / InputLifecycle::BulkDrain / "
        "InputLifecycle::NoWaitPop on Block walkers.");
    static_assert(
        is_legal_kind_lifecycle(BIndex, BPolicy),
        "BinaryFpu: (BIndex, BPolicy) is illegal for Block — exclude "
        "InputLifecycle::Streaming / InputLifecycle::HeldStream / InputLifecycle::BulkDrain / "
        "InputLifecycle::NoWaitPop on Block walkers.");
    // same_dfb dedup safety: when CbA == CbB the B-side wait/pop is skipped, so the
    // helper would under-wait if A and B walked different ranges of the shared CB.
    static_assert(
        (CbA != CbB) || AIndex == BIndex,
        "BinaryFpu: when CbA == CbB, AIndex and BIndex must match "
        "(B-side wait/pop is deduped — asymmetric indices would under-wait).");
    // 2D: RowBcast / ColBcast on either side require non-streaming policy.
    static_assert(
        detail::valid_policy_mode_v<APolicy, AIndex>,
        "BinaryFpu: A-side RowBcast / ColBcast index require non-streaming APolicy");
    static_assert(
        detail::valid_policy_mode_v<BPolicy, BIndex>,
        "BinaryFpu: B-side RowBcast / ColBcast index require non-streaming BPolicy");
    // Per-operand TileBase lifecycle compatibility — InputLifecycle::Streaming/InputLifecycle::Chunked/Cumulative
    // can't compose with runtime base offsets (iter-dependent wait/pop counts).
    static_assert(
        OffsetA == TileOffset::Unset || is_legal_input_lifecycle_with_base(APolicy),
        "BinaryFpu: OffsetA Set requires APolicy to be InputLifecycle::Bulk-family or InputLifecycle::CallerManaged");
    static_assert(
        OffsetB == TileOffset::Unset || is_legal_input_lifecycle_with_base(BPolicy),
        "BinaryFpu: OffsetB Set requires BPolicy to be InputLifecycle::Bulk-family or InputLifecycle::CallerManaged");
    // Per-block streaming uses chunk-local CB front. When the two sides use
    // DIFFERENT regimes (one per-block → chunk-local index `j`; the other upfront /
    // caller-managed → absolute index `base_tile + j`), the chain dispatcher
    // resolves them separately via the 3-arg exec / exec overloads gated by
    // `needs_per_side_idx`. Same-regime hits the 2-arg fast path.

    static constexpr uint32_t dfb_a_id() { return CbA; }
    static constexpr uint32_t dfb_b_id() { return CbB; }
    static constexpr InputLifecycle a_policy() { return APolicy; }
    static constexpr InputLifecycle b_policy() { return BPolicy; }
    static constexpr bool is_upfront =
        is_one_of_v<APolicy, InputLifecycle::Bulk, InputLifecycle::HeldBulk, InputLifecycle::Pipelined> ||
        is_one_of_v<BPolicy, InputLifecycle::Bulk, InputLifecycle::HeldBulk, InputLifecycle::Pipelined>;
    static constexpr bool same_dfb = (CbA == CbB);
    static constexpr bool uses_dest_accumulation = (Accumulation == DestAccumulation::Enabled);
    static constexpr Dst accumulated_dst_slot = DstSlot;

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
        is_one_of_v<Reconfig, BinaryDataFormatReconfig::Input, BinaryDataFormatReconfig::SrcA> ? CbA : NO_PREV_DFB;
    static constexpr uint32_t      reconfig_srcb_dfb =
        is_one_of_v<Reconfig, BinaryDataFormatReconfig::Input, BinaryDataFormatReconfig::SrcB> ? CbB : NO_PREV_DFB;
    // pack side absent -> dfb_for_side defaults to NO_PREV_DFB (downstream PackTile owns pack).

    InputStream<CbA, APolicy, AIndex, OffsetA> a;
    InputStream<CbB, BPolicy, BIndex, OffsetB> b;

    constexpr BinaryFpu() noexcept = default;
    constexpr BinaryFpu(uint32_t base_a, uint32_t base_b) noexcept : a(base_a), b(base_b) {}
    constexpr explicit BinaryFpu(uint32_t base_a) noexcept : a(base_a) {}

    // Helper: when same_dfb, both bases live in the single shared wait window.
    // Wait/pop count uses max(base_a, base_b) — caller must stage that many tiles
    // in front of both reads.
    ALWI uint32_t same_dfb_base_max() const noexcept {
        const uint32_t bA = tile_base_value<OffsetA>(a.tile_base);
        const uint32_t bB = tile_base_value<OffsetB>(b.tile_base);
        return bA > bB ? bA : bB;
    }

    // ---- init / reconfig ----
    // srca / srcb / pack reconfig are fold-driven (compile-time-elided
    // when prev_*_cb == cur_*_cb). init() programs only the per-op LLK shape.
    static ALWI void init() {
        // Op-specific init.
        if constexpr (Bcast == BroadcastDim::None) {
            constexpr bool acc_to_dest = Accumulation == DestAccumulation::Enabled;
            if constexpr (Op == BinaryFpuOp::Add) {
                add_tiles_init(CbA, CbB, acc_to_dest);
            } else if constexpr (Op == BinaryFpuOp::Sub) {
                sub_tiles_init(CbA, CbB, acc_to_dest);
            } else {
                mul_tiles_init(CbA, CbB, static_cast<uint32_t>(acc_to_dest), __builtin_LINE());
            }
        } else {
            // Use the *_init_short form from bcast.h:352-446 (math init + unpack init only,
            // no hw_configure / pack_dest_init / sync_init — the full init is undefined
            // mid-MAIN). The operand form reads the actual tensor shape from CB metadata via
            // get_operand_tensor_shape, matching `add_bcast_rows_init_short` /
            // `sub_bcast_cols_init_short` etc. exactly.
            constexpr auto bt = static_cast<ckernel::BroadcastType>(static_cast<uint8_t>(Bcast));
            constexpr auto et = (Op == BinaryFpuOp::Add)   ? ckernel::EltwiseBinaryType::ELWADD
                                : (Op == BinaryFpuOp::Sub) ? ckernel::EltwiseBinaryType::ELWSUB
                                                           : ckernel::EltwiseBinaryType::ELWMUL;
            if constexpr (Op == BinaryFpuOp::Mul) {
                MATH((llk_math_eltwise_binary_init<et, bt, MATH_FIDELITY>(
                    CbA, CbB, Accumulation == DestAccumulation::Enabled)));
            } else {
                MATH((llk_math_eltwise_binary_init<et, bt, MathFidelity::LoFi>(
                    CbA, CbB, Accumulation == DestAccumulation::Enabled)));
            }
            UNPACK((llk_unpack_AB_init<bt>(CbA, CbB)));
        }
    }

    // ---- CB lifecycle ----
    // Each hook fans out per side: A always, B only when !same_dfb (when CbA == CbB the
    // B-side wait/pop is deduped). The per-side body lives in InputStream.
    ALWI void wait_per_tile(uint32_t cumulative_count) const {
        a.wait_per_tile(cumulative_count);
        if constexpr (!same_dfb) {
            b.wait_per_tile(cumulative_count);
        }
    }

    ALWI void wait_per_block(uint32_t inner_count) const {
        a.wait_per_block(inner_count);
        if constexpr (!same_dfb) {
            b.wait_per_block(inner_count);
        }
    }

    // 2D: per-side upfront wait — A uses AIndex's window, B uses BIndex's window.
    // When same_dfb, both bases collapse into A's single shared window (max base) and the
    // B side is skipped; otherwise each side delegates to its own InputStream.
    ALWI void wait_upfront(uint32_t Ht, uint32_t Wt) const {
        if constexpr (same_dfb) {
            if constexpr (is_one_of_v<APolicy, InputLifecycle::BulkDrain>) {
                // See InputStream::wait_upfront — BulkDrain waits the whole walk, not window<Scalar>.
                a.wait_n(Ht * Wt + same_dfb_base_max());
            } else if constexpr (is_one_of_v<APolicy, InputLifecycle::Bulk, InputLifecycle::HeldBulk>) {
                a.wait_n(detail::window<AIndex>(Ht, Wt) + same_dfb_base_max());
            }
        } else {
            a.wait_upfront(Ht, Wt);
            b.wait_upfront(Ht, Wt);
        }
    }

    // Per-side index mode. AIndex drives a_idx, BIndex drives b_idx. The canonical
    // bcast walk is A=Block (walks the tile range) + B=Scalar (pins the
    // scaler/vector operand at tile 0). OffsetA / OffsetB add a runtime or
    // compile-time base offset to the per-iter index. The 3-arg overload accepts a
    // chunk-local index (`i_local`) and an absolute index (`i_abs`); each side
    // picks via `a_uses_local_idx` / `b_uses_local_idx`. The 2-arg overload is
    // the same-regime fast path and forwards with i_local == i_abs.


    static constexpr uint32_t lane_width = to_u32(DstSlot) + 1;

    ALWI void pop_per_tile(uint32_t i) const {
        a.pop_per_tile(i);
        if constexpr (!same_dfb) {
            b.pop_per_tile(i);
        }
    }

    ALWI void pop_per_block(uint32_t inner_count) const {
        a.pop_per_block(inner_count);
        if constexpr (!same_dfb) {
            b.pop_per_block(inner_count);
        }
    }

    /// Per-outer-row wait/pop for streamed broadcasts (InputLifecycle::OuterStream) — per side,
    /// same_dfb-deduped. One operand tile per row, re-read at the front across the row's cols.
    /// OuterStream is restricted to OperandKind::Scalar, so exec reads the front (0) already.
    ALWI void wait_per_row() const {
        a.wait_per_row();
        if constexpr (!same_dfb) {
            b.wait_per_row();
        }
    }
    ALWI void pop_per_row() const {
        a.pop_per_row();
        if constexpr (!same_dfb) {
            b.pop_per_row();
        }
    }

    // 2D variants — per-side index + window. 3-arg form takes both chunk-local
    // (`i_local`) and absolute (`i_abs`) flat indices; each side picks via the
    // per-side traits. `ht` is unchanged (always absolute row); `wt_local` /
    // `wt_abs` cover the per-side column index when needed. Same-regime fast
    // path forwards through the 2-arg overload.
    ALWI void exec(
        uint32_t i_flat_local,
        uint32_t i_flat_abs,
        uint32_t ht,
        uint32_t wt_local,
        uint32_t wt_abs,
        uint32_t slot_offset) const {
        const uint32_t a_flat = a_uses_local_idx ? i_flat_local : i_flat_abs;
        const uint32_t b_flat = b_uses_local_idx ? i_flat_local : i_flat_abs;
        const uint32_t a_wt = a_uses_local_idx ? wt_local : wt_abs;
        const uint32_t b_wt = b_uses_local_idx ? wt_local : wt_abs;
        const uint32_t a_idx = tile_base_value<OffsetA>(a.tile_base) + detail::idx<AIndex>(a_flat, ht, a_wt);
        const uint32_t b_idx = tile_base_value<OffsetB>(b.tile_base) + detail::idx<BIndex>(b_flat, ht, b_wt);
        uint32_t dst;
        if constexpr (Accumulation == DestAccumulation::Enabled) {
            dst = to_u32(DstSlot);
        } else {
            dst = to_u32(DstSlot) + slot_offset;
        }
        if constexpr (Bcast == BroadcastDim::None) {
            if constexpr (Op == BinaryFpuOp::Add) {
                add_tiles(CbA, CbB, a_idx, b_idx, dst);
            } else if constexpr (Op == BinaryFpuOp::Sub) {
                sub_tiles(CbA, CbB, a_idx, b_idx, dst);
            } else {
                mul_tiles(CbA, CbB, a_idx, b_idx, dst);
            }
        } else {
            constexpr auto bt = static_cast<ckernel::BroadcastType>(static_cast<uint8_t>(Bcast));
            if constexpr (Op == BinaryFpuOp::Add) {
                add_tiles_bcast<bt>(CbA, CbB, a_idx, b_idx, dst);
            } else if constexpr (Op == BinaryFpuOp::Sub) {
                sub_tiles_bcast<bt>(CbA, CbB, a_idx, b_idx, dst);
            } else {
                mul_tiles_bcast<bt>(CbA, CbB, a_idx, b_idx, dst);
            }
        }
    }

    ALWI void exec(uint32_t i_flat, uint32_t ht, uint32_t wt, uint32_t slot_offset) const {
        exec(i_flat, i_flat, ht, wt, wt, slot_offset);
    }

    ALWI void pop_upfront_end(uint32_t Ht, uint32_t Wt) const {
        if constexpr (same_dfb) {
            if constexpr (is_one_of_v<APolicy, InputLifecycle::Bulk, InputLifecycle::Pipelined, InputLifecycle::DeferredPop>) {
                a.pop_n(detail::window<AIndex>(Ht, Wt) + same_dfb_base_max());
            }
        } else {
            a.pop_upfront_end(Ht, Wt);
            b.pop_upfront_end(Ht, Wt);
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
struct DestReuseBinary : InputStream<Cb, Policy, IndexMode, Offset>, DestReuseBinaryTag {
    using Base = InputStream<Cb, Policy, IndexMode, Offset>;
    using Base::tile_base;

    static_assert(
        to_u32(DstIn) < DEST_AUTO_LIMIT && to_u32(DstOut) < DEST_AUTO_LIMIT,
        "DestReuseBinary: DEST slot exceeds DEST_AUTO_LIMIT");
    static_assert(
        is_legal_kind_lifecycle(IndexMode, Policy),
        "DestReuseBinary: (IndexMode, Policy) is illegal for Block — exclude "
        "InputLifecycle::Streaming / InputLifecycle::HeldStream / InputLifecycle::BulkDrain / "
        "InputLifecycle::NoWaitPop on Block walkers.");
    static_assert(
        detail::valid_policy_mode_v<Policy, IndexMode>,
        "DestReuseBinary: RowBcast / ColBcast index require non-streaming policy");
    static_assert(
        Offset == TileOffset::Unset || is_legal_input_lifecycle_with_base(Policy),
        "DestReuseBinary: TileOffset::Set requires InputLifecycle::Bulk-family or InputLifecycle::CallerManaged "
        "lifecycle");

    // The one CB feeds the src that DEST is NOT routed to: DEST_TO_SRCB -> CB on srcA (dfb_a),
    // DEST_TO_SRCA -> CB on srcB (dfb_b). The other side is the DEST register, not a CB (INVALID_DFB).
    static constexpr uint32_t       dfb_a_id()         { return (ReuseType == DestReuseType::DEST_TO_SRCB) ? Cb : INVALID_DFB; }
    static constexpr uint32_t       dfb_b_id()         { return (ReuseType == DestReuseType::DEST_TO_SRCA) ? Cb : INVALID_DFB; }
    static constexpr InputLifecycle a_policy()        { return Policy; }
    static constexpr bool           is_upfront        =
        is_one_of_v<Policy, InputLifecycle::Bulk, InputLifecycle::HeldBulk, InputLifecycle::Pipelined>;

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

    constexpr DestReuseBinary() noexcept = default;
    constexpr explicit DestReuseBinary(uint32_t base) noexcept : Base(base) {}

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

    static constexpr uint32_t lane_width =
        (to_u32(DstIn) > to_u32(DstOut)) ? (to_u32(DstIn) + 1) : (to_u32(DstOut) + 1);

    // wait_per_tile / wait_per_block / wait_upfront / pop_upfront_end / pop_per_tile /
    // pop_per_block / wait_per_row / pop_per_row inherited from InputStream.
};

// =============================================================================
// 6. UnaryBcast chain element
// =============================================================================

template <BroadcastDim Dim,
          uint32_t Cb,
          InputLifecycle Policy,
          UnaryBcastReconfig Reconfig,
          Dst DstSlot>
struct UnaryBcast
    : InputStream<Cb, Policy, OperandKind::Block, TileOffset::Unset>,
      UnaryBcastTag {
    static_assert(to_u32(DstSlot) < DEST_AUTO_LIMIT,
                  "UnaryBcast: DEST slot exceeds DEST_AUTO_LIMIT");

    // UnaryBcast reads one CB front (dfb_a); dfb_b is absent -> dfb_b_of defaults to INVALID_DFB.
    static constexpr uint32_t       dfb_a_id()         { return Cb; }
    static constexpr InputLifecycle a_policy()        { return Policy; }
    static constexpr bool           is_upfront        =
        is_one_of_v<Policy, InputLifecycle::Bulk, InputLifecycle::HeldBulk, InputLifecycle::Pipelined>;

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

    ALWI void exec(uint32_t /*i_flat*/, uint32_t /*ht*/, uint32_t /*wt*/, uint32_t slot_offset) const {
        constexpr auto bt = static_cast<ckernel::BroadcastType>(static_cast<uint8_t>(Dim));
        unary_bcast<bt>(Cb, /*in_tile_index=*/0, to_u32(DstSlot) + slot_offset);
    }
    static constexpr uint32_t lane_width = to_u32(DstSlot) + 1;

    // wait_per_tile / wait_per_block / wait_upfront / pop_upfront_end / pop_per_tile /
    // pop_per_block inherited from InputStream (IndexMode=Block + Offset=Unset → Ht*Wt window).
    // UnaryBcast always reads tile 0, so it can't be a streamed outer-axis broadcast — the
    // per-row hooks are explicitly inert, overriding InputStream's OuterStream-gated versions.
    ALWI void wait_per_row() const {}
    ALWI void pop_per_row() const {}
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

// chain_lane_width — N-element fold. Max of per-element `lane_width`. The chain clamps block_size
// at runtime so `block_size * chain_lane_width <= DEST_AUTO_LIMIT` (block_size is a runtime
// EltwiseShape field, so this is a clamp, not a static_assert). Each element writes to
// DEST[dst_slot + j * chain_lane_width] for lane j in [0, block_size).
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

// DEST-accumulating chains reserve D0 as a non-replicated sticky slot. This width contains
// only elements replicated across block lanes; it is intentionally allowed to be zero.
template <class Chain>
struct chain_transient_lane_width;

template <class... Es>
struct chain_transient_lane_width<EltwiseChain<Es...>>
    : std::integral_constant<uint32_t, detail::ChainTraits<Es...>::transient_lane_width> {};

template <class Chain>
inline constexpr uint32_t chain_transient_lane_width_v = chain_transient_lane_width<Chain>::value;

// chain_max_block_v — largest block_size that fits in DEST for this chain, given its
// lane-width fold. Caller-facing compile-time constant: pass any value <= this to
// the runtime `block_size` arg on `eltwise_chain`. Caller can `static_assert` their
// chosen block against this value to recover the build-time DEST overflow signal.
template <class Chain>
struct chain_max_block;

namespace detail {
template <class... Es>
constexpr uint32_t chain_max_block_value() {
    using Traits = ChainTraits<Es...>;
    if constexpr (Traits::any_dest_accumulation) {
        if constexpr (Traits::transient_lane_width == 0) {
            // No transient lane consumes DEST capacity, so DEST imposes no block bound.
            // Keep this branch separate: the zero-width specialization contains no divide.
            return ~uint32_t{0};
        } else {
            return (DEST_AUTO_LIMIT - 1) / Traits::transient_lane_width;
        }
    } else {
        return DEST_AUTO_LIMIT / Traits::lane_width;
    }
}
}  // namespace detail

template <class... Es>
struct chain_max_block<EltwiseChain<Es...>> : std::integral_constant<uint32_t, detail::chain_max_block_value<Es...>()> {
};

template <class Chain>
inline constexpr uint32_t chain_max_block_v = chain_max_block<Chain>::value;

// chain_supports_block — N-element fold. True when every CB-reader element uses a policy that
// stages a multi-tile window per outer iter (an upfront Bulk-family policy or per-chunk Chunked).
// Per-tile policies (Streaming / HeldStream) consume ONE tile per iter and can't do block_size > 1;
// for such chains the chain clamps block_size to 1 at runtime (not a static_assert).
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
template <class E, class = void>
struct has_is_upfront_m : std::false_type {};
template <class E>
struct has_is_upfront_m<E, std::void_t<decltype(E::is_upfront)>> : std::true_type {};
template <class E>
constexpr bool is_upfront_of() {
    if constexpr (has_is_upfront_m<E>::value) {
        return E::is_upfront;
    } else {
        return false;
    }
}
template <class E, class = void>
struct has_pack_dst_slot_m : std::false_type {};
template <class E>
struct has_pack_dst_slot_m<E, std::void_t<decltype(E::pack_dst_slot)>> : std::true_type {};
template <class E>
constexpr Dst pack_dst_slot_of() {
    if constexpr (has_pack_dst_slot_m<E>::value) {
        return E::pack_dst_slot;
    } else {
        return Dst::D0;
    }
}
// b_policy defaults to CallerManaged (the unary-reader / no-srcB-CB case), so only elements
// with a genuine srcB operand (BinaryFpu) declare it.
template <class E, class = void>
struct has_b_policy_m : std::false_type {};
template <class E>
struct has_b_policy_m<E, std::void_t<decltype(E::b_policy())>> : std::true_type {};
template <class E>
constexpr InputLifecycle b_policy_of() {
    if constexpr (has_b_policy_m<E>::value) {
        return E::b_policy();
    } else {
        return InputLifecycle::CallerManaged;
    }
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
    bool uses_l1_accumulation;
    bool seeds_l1_accumulation;
    bool manages_l1_accumulation_lifecycle;
    bool uses_dest_accumulation;
    Dst accumulated_dst_slot;
    bool uses_dest_accumulation_lifecycle;
    bool manages_dest_accumulation_lifecycle;
    bool is_upfront;
    uint32_t lane_width;
    uint32_t transient_lane_width;
    bool supports_block;
    PackRelu pack_relu;  // packer ReLU mode (PackRelu::None for non-pack elements) — hetero/any-relu input
};

template <class E>
constexpr bool uses_l1_accumulation_of() {
    if constexpr (is_pack_tile_op_v<E>) {
        return E::uses_l1_accumulation;
    } else {
        return false;
    }
}

template <class E>
constexpr bool manages_l1_accumulation_lifecycle_of() {
    if constexpr (is_pack_tile_op_v<E>) {
        return E::manages_l1_accumulation_lifecycle;
    } else {
        return false;
    }
}

template <class E>
constexpr bool seeds_l1_accumulation_of() {
    if constexpr (is_pack_tile_op_v<E>) {
        return E::seeds_l1_accumulation;
    } else {
        return false;
    }
}

template <class E>
constexpr bool uses_dest_accumulation_of() {
    if constexpr (is_binary_fpu_op_v<E>) {
        return E::uses_dest_accumulation;
    } else {
        return false;
    }
}

template <class E>
constexpr Dst accumulated_dst_slot_of() {
    if constexpr (is_binary_fpu_op_v<E>) {
        return E::accumulated_dst_slot;
    } else {
        return Dst::D0;
    }
}

template <class E>
constexpr bool uses_dest_accumulation_lifecycle_of() {
    if constexpr (is_pack_tile_op_v<E>) {
        return E::uses_dest_accumulation_lifecycle;
    } else {
        return false;
    }
}

template <class E>
constexpr bool manages_dest_accumulation_lifecycle_of() {
    if constexpr (is_pack_tile_op_v<E>) {
        return E::manages_dest_accumulation_lifecycle;
    } else {
        return false;
    }
}

template <class E>
constexpr uint32_t transient_lane_width_of() {
    if constexpr (is_binary_fpu_op_v<E>) {
        return E::uses_dest_accumulation ? 0u : elem_lane_width_v<E>;
    } else if constexpr (is_pack_tile_op_v<E>) {
        return 0u;
    } else {
        return elem_lane_width_v<E>;
    }
}

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
        uses_l1_accumulation_of<E>(),
        seeds_l1_accumulation_of<E>(),
        manages_l1_accumulation_lifecycle_of<E>(),
        uses_dest_accumulation_of<E>(),
        accumulated_dst_slot_of<E>(),
        uses_dest_accumulation_lifecycle_of<E>(),
        manages_dest_accumulation_lifecycle_of<E>(),
        is_upfront_of<E>(),
        elem_lane_width_v<E>,
        transient_lane_width_of<E>(),
        element_supports_block<E>(),
        pack_relu_of<E>(),
    };
}

// Derivations over the descriptor array — flat loops bounded by the real element count
// `n` (the array is sized [N?N:1], so an empty chain must NOT read the lone default slot).
constexpr uint32_t ct_lane_width(const ElemDesc* d, int n) {
    uint32_t w = 1;
    for (int i = 0; i < n; ++i) {
        if (d[i].lane_width > w) {
            w = d[i].lane_width;
        }
    }
    return w;
}
constexpr uint32_t ct_transient_lane_width(const ElemDesc* d, int n) {
    uint32_t w = 0;
    for (int i = 0; i < n; ++i) {
        if (d[i].transient_lane_width > w) {
            w = d[i].transient_lane_width;
        }
    }
    return w;
}
constexpr bool ct_supports_block(const ElemDesc* d, int n) {
    bool r = true;
    for (int i = 0; i < n; ++i) {
        r = r && d[i].supports_block;
    }
    return r;
}
constexpr bool ct_side_consistent(const ElemDesc* d, int n, uint32_t ElemDesc::* side) {
    uint32_t seen = NO_PREV_DFB;
    for (int i = 0; i < n; ++i) {
        uint32_t dfb = d[i].*side;
        if (dfb == NO_PREV_DFB) {
            continue;
        }
        if (seen == NO_PREV_DFB) {
            seen = dfb;
        } else if (seen != dfb) {
            return false;
        }
    }
    return true;
}
constexpr bool ct_reader_collide(const ElemDesc* d, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (!(d[i].is_cb_reader && d[j].is_cb_reader)) {
                continue;
            }
            if (!(d[i].is_upfront && d[j].is_upfront)) {
                continue;
            }
            uint32_t a0 = d[i].dfb_a, a1 = d[i].dfb_b, b0 = d[j].dfb_a, b1 = d[j].dfb_b;
            if ((a0 != INVALID_DFB && (a0 == b0 || a0 == b1)) || (a1 != INVALID_DFB && (a1 == b0 || a1 == b1))) {
                return true;
            }
        }
    }
    return false;
}
constexpr bool ct_writer_collide(const ElemDesc* d, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (d[i].is_pack && d[j].is_pack && d[i].pack_dfb == d[j].pack_dfb &&
                d[i].pack_dst_slot == d[j].pack_dst_slot) {
                return true;
            }
        }
    }
    return false;
}
constexpr bool ct_any_l1_accumulation(const ElemDesc* d, int n) {
    for (int i = 0; i < n; ++i) {
        if (d[i].uses_l1_accumulation) {
            return true;
        }
    }
    return false;
}
constexpr bool ct_all_writers_l1_accumulation(const ElemDesc* d, int n) {
    for (int i = 0; i < n; ++i) {
        if (d[i].is_pack && !d[i].uses_l1_accumulation) {
            return false;
        }
    }
    return true;
}
constexpr bool ct_l1_accumulation_modes_consistent(const ElemDesc* d, int n) {
    bool found = false;
    bool seed_first = false;
    for (int i = 0; i < n; ++i) {
        if (!d[i].is_pack || !d[i].uses_l1_accumulation) {
            continue;
        }
        if (!found) {
            found = true;
            seed_first = d[i].seeds_l1_accumulation;
        } else if (seed_first != d[i].seeds_l1_accumulation) {
            return false;
        }
    }
    return true;
}
constexpr bool ct_any_seed_first_l1_accumulation(const ElemDesc* d, int n) {
    for (int i = 0; i < n; ++i) {
        if (d[i].seeds_l1_accumulation) {
            return true;
        }
    }
    return false;
}
constexpr bool ct_pack_dfbs_consistent(const ElemDesc* d, int n) {
    uint32_t seen = INVALID_DFB;
    for (int i = 0; i < n; ++i) {
        if (!d[i].is_pack) {
            continue;
        }
        if (seen == INVALID_DFB) {
            seen = d[i].pack_dfb;
        } else if (seen != d[i].pack_dfb) {
            return false;
        }
    }
    return true;
}
constexpr uint32_t ct_managed_l1_accumulation_lifecycles(const ElemDesc* d, int n) {
    uint32_t count = 0;
    for (int i = 0; i < n; ++i) {
        if (d[i].manages_l1_accumulation_lifecycle) {
            ++count;
        }
    }
    return count;
}
constexpr bool ct_any_dest_accumulation(const ElemDesc* d, int n) {
    for (int i = 0; i < n; ++i) {
        if (d[i].uses_dest_accumulation) {
            return true;
        }
    }
    return false;
}
constexpr uint32_t ct_dest_accumulation_slot_count(const ElemDesc* d, int n) {
    bool seen[DEST_AUTO_LIMIT] = {};
    uint32_t count = 0;
    for (int i = 0; i < n; ++i) {
        if (!d[i].uses_dest_accumulation) {
            continue;
        }
        const uint32_t slot = to_u32(d[i].accumulated_dst_slot);
        if (!seen[slot]) {
            seen[slot] = true;
            ++count;
        }
    }
    return count;
}
constexpr uint32_t ct_pack_writer_count(const ElemDesc* d, int n) {
    uint32_t count = 0;
    for (int i = 0; i < n; ++i) {
        if (d[i].is_pack) {
            ++count;
        }
    }
    return count;
}
constexpr bool ct_dest_accumulation_pack_matches(const ElemDesc* d, int n) {
    Dst sticky = Dst::D0;
    bool found = false;
    for (int i = 0; i < n; ++i) {
        if (!d[i].uses_dest_accumulation) {
            continue;
        }
        if (!found) {
            sticky = d[i].accumulated_dst_slot;
            found = true;
        }
    }
    for (int i = 0; i < n; ++i) {
        if (d[i].is_pack && d[i].pack_dst_slot != sticky) {
            return false;
        }
    }
    return true;
}
constexpr bool ct_all_writers_dest_accumulation_lifecycle(const ElemDesc* d, int n) {
    for (int i = 0; i < n; ++i) {
        if (d[i].is_pack && !d[i].uses_dest_accumulation_lifecycle) {
            return false;
        }
    }
    return true;
}
constexpr bool ct_any_dest_accumulation_lifecycle(const ElemDesc* d, int n) {
    for (int i = 0; i < n; ++i) {
        if (d[i].uses_dest_accumulation_lifecycle) {
            return true;
        }
    }
    return false;
}
constexpr uint32_t ct_managed_dest_accumulation_lifecycles(const ElemDesc* d, int n) {
    uint32_t count = 0;
    for (int i = 0; i < n; ++i) {
        if (d[i].manages_dest_accumulation_lifecycle) {
            ++count;
        }
    }
    return count;
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
        if (d[i].srca_cb != NO_PREV_DFB) {
            pa = d[i].srca_cb;
        }
        if (d[i].srcb_cb != NO_PREV_DFB) {
            pb = d[i].srcb_cb;
        }
        if (d[i].pack_side_dfb != NO_PREV_DFB) {
            pp = d[i].pack_side_dfb;
        }
    }
    return t;
}
// Last opt-in pack CB in chain order (iter-to-iter wraparound prev for pack site 0).
constexpr uint32_t ct_last_pack_cb(const ElemDesc* d, int n) {
    uint32_t last = NO_PREV_DFB;
    for (int i = 0; i < n; ++i) {
        if (d[i].pack_side_dfb != NO_PREV_DFB) {
            last = d[i].pack_side_dfb;
        }
    }
    return last;
}
// True iff ≥2 opt-in pack sites declare different CBs (boot can't program all). This is the
// pack DATA-FORMAT heterogeneity axis — it gates data-format reconfig deferral (see
// emit_per_stage_pack_reconfig) and is orthogonal to ReLU (ct_pack_relu_hetero below).
constexpr bool ct_pack_dtype_hetero(const ElemDesc* d, int n) {
    uint32_t first = NO_PREV_DFB;
    for (int i = 0; i < n; ++i) {
        if (d[i].pack_side_dfb == NO_PREV_DFB) {
            continue;
        }
        if (first == NO_PREV_DFB) {
            first = d[i].pack_side_dfb;
        } else if (first != d[i].pack_side_dfb) {
            return true;
        }
    }
    return false;
}
// True iff ≥2 PackTile sites declare different packer ReLU modes. STACC_RELU is a latched
// packer-global mode, so a mixed chain would need per-stage set/reset around each pack (not yet
// supported); this is the ReLU heterogeneity axis, orthogonal to data-format (ct_pack_dtype_hetero).
constexpr bool ct_pack_relu_hetero(const ElemDesc* d, int n) {
    bool have_first = false;
    PackRelu first = PackRelu::None;
    for (int i = 0; i < n; ++i) {
        if (!d[i].is_pack) {
            continue;
        }
        if (!have_first) {
            first = d[i].pack_relu;
            have_first = true;
        } else if (d[i].pack_relu != first) {
            return true;
        }
    }
    return false;
}
// True iff any PackTile site applies a packer ReLU — drives the once-before-loop set and the
// restore-before-publish reset in chain_run_loop.
constexpr bool ct_any_pack_relu(const ElemDesc* d, int n) {
    for (int i = 0; i < n; ++i) {
        if (d[i].is_pack && d[i].pack_relu != PackRelu::None) {
            return true;
        }
    }
    return false;
}

template <class... Es>
struct ChainTraits {
    static constexpr int N = int(sizeof...(Es));
    static constexpr ElemDesc d[N ? N : 1] = {describe<Es>()...};  // the one walk

    static constexpr uint32_t lane_width = ct_lane_width(d, N);
    static constexpr uint32_t transient_lane_width = ct_transient_lane_width(d, N);
    static constexpr bool supports_block = ct_supports_block(d, N);
    static constexpr bool srca_consistent = ct_side_consistent(d, N, &ElemDesc::srca_cb);
    static constexpr bool srcb_consistent = ct_side_consistent(d, N, &ElemDesc::srcb_cb);
    static constexpr bool reader_collide = ct_reader_collide(d, N);
    static constexpr bool writer_collide = ct_writer_collide(d, N);
    static constexpr bool any_l1_accumulation = ct_any_l1_accumulation(d, N);
    static constexpr bool any_seed_first_l1_accumulation = ct_any_seed_first_l1_accumulation(d, N);
    static constexpr bool all_writers_l1_accumulation = ct_all_writers_l1_accumulation(d, N);
    static constexpr bool l1_accumulation_modes_consistent = ct_l1_accumulation_modes_consistent(d, N);
    static constexpr bool pack_dfbs_consistent = ct_pack_dfbs_consistent(d, N);
    static constexpr uint32_t managed_l1_accumulation_lifecycles = ct_managed_l1_accumulation_lifecycles(d, N);
    static constexpr bool any_dest_accumulation = ct_any_dest_accumulation(d, N);
    static constexpr uint32_t dest_accumulation_slot_count = ct_dest_accumulation_slot_count(d, N);
    static constexpr uint32_t pack_writer_count = ct_pack_writer_count(d, N);
    static constexpr bool dest_accumulation_pack_matches = ct_dest_accumulation_pack_matches(d, N);
    static constexpr bool all_writers_dest_accumulation_lifecycle = ct_all_writers_dest_accumulation_lifecycle(d, N);
    static constexpr bool any_dest_accumulation_lifecycle = ct_any_dest_accumulation_lifecycle(d, N);
    static constexpr uint32_t managed_dest_accumulation_lifecycles = ct_managed_dest_accumulation_lifecycles(d, N);

    // Per-side prev-CB history (one sweep), + pack-side metadata.
    static constexpr PrevTable<N ? N : 1> prev = ct_build_prev<N ? N : 1>(d, N);
    static constexpr uint32_t last_pack_cb = ct_last_pack_cb(d, N);
    // Two orthogonal pack-heterogeneity axes: data-format (which CB/dtype each pack site targets)
    // drives data-format reconfig deferral; ReLU (which packer activation each site applies) drives
    // the packer-ReLU set/reset. `chain_hoist_pack` is boot-hoistable only when NEITHER is hetero.
    static constexpr bool pack_dtype_hetero = ct_pack_dtype_hetero(d, N);
    static constexpr bool pack_relu_hetero = ct_pack_relu_hetero(d, N);
    static constexpr bool any_pack_relu = ct_any_pack_relu(d, N);
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

// Pack cohort hoist: pack init + reconfig + packer ReLU are fully boot-emitted, with NO per-stage
// pack work in the loop. True iff NEITHER pack-heterogeneity axis is set — homogeneous opt-in pack
// CBs (data-format) AND a single packer ReLU mode across pack sites. The output leg of "all one-time
// setup is boot-hoistable" (with math-MOP + SFPU).
template <class Chain>
struct chain_hoist_pack : std::true_type {};

template <class... Es>
struct chain_hoist_pack<EltwiseChain<Es...>>
    : std::bool_constant<!detail::ChainTraits<Es...>::pack_dtype_hetero &&
                         !detail::ChainTraits<Es...>::pack_relu_hetero> {};

// True iff no element requests any reconfig (all reconfig knobs None). SetupOwner::Caller requires
// this so a set-but-inert reconfig knob (which the helper would silently ignore) is a compile error
// instead of a lie about what runs inside the chain.
template <class Chain>
struct chain_no_reconfig_requested : std::true_type {};

template <class... Es>
struct chain_no_reconfig_requested<EltwiseChain<Es...>>
    : std::bool_constant<detail::chain_requests_no_reconfig<Es...>()> {};

template <class Chain>
inline constexpr bool chain_no_reconfig_requested_v = chain_no_reconfig_requested<Chain>::value;

// =============================================================================
// 10. Chain pipeline — per-iteration emit
// =============================================================================

namespace detail {

// Per-block (chunked) lifecycle hooks — wait_per_block / pop_per_block / reserve_per_block /
// push_per_block — are dispatched directly on the element, same as wait_per_tile / wait_upfront.
// Every cb-reader / pack element defines them as policy-gated `if constexpr` bodies (a no-op that
// compiles to nothing when the element's lifecycle doesn't chunk), exactly like the per_tile /
// upfront hooks every element already defines for the policies it doesn't use. They're only ever
// called inside the `is_cb_reader_op_v` / `is_pack_tile_op_v` branches, so only element kinds that
// define them are instantiated. A new element kind must define them (loud compile error if not) —
// no SFINAE no-op that would let a needed chunked wait/pop silently vanish.

// init() dispatch convention — the compute-cohort init is emitted on the element
// *instance* (`elem.init()`), exactly like `exec`, so an init can read the struct's
// runtime members when it needs to (e.g. Dropout seeding the SFPU with its runtime
// `seed`). Most inits don't and are declared `static`; a static member is callable
// through an instance, so the call site is uniform either way. PackTile is the one
// exception — its init is dispatched by type in `pack_init_for_each` (no instance in
// scope there), which is fine because pack init never needs runtime state.

// =============================================================================
// emit_pre_element_transitions<E, PrevA, PrevB, PrevP, PackHetero>()
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

// De-templated from the full chain pack: takes only the compile-time facts it consumes — the
// element's prev-CB descriptors (srca/srcb/pack) and pack-heterogeneity — precomputed by the caller
// from `ChainTraits<Es...>` at the element's index. This keeps the per-element mangled name (and thus
// the -g debug info that records it) proportional to the ELEMENT, not the whole chain. The values
// passed in are exactly what this function used to read internally, so reconfig behavior is identical.
template <class E, uint32_t PrevA, uint32_t PrevB, uint32_t PrevP, bool PackHetero>
ALWI void emit_pre_element_transitions() {
    constexpr uint32_t curr_a = dfb_for_side<Side::SrcA, E>();
    constexpr uint32_t curr_b = dfb_for_side<Side::SrcB, E>();
    constexpr uint32_t curr_p = dfb_for_side<Side::Pack, E>();

    constexpr uint32_t prev_a = (curr_a != NO_PREV_DFB) ? PrevA : NO_PREV_DFB;
    constexpr uint32_t prev_b = (curr_b != NO_PREV_DFB) ? PrevB : NO_PREV_DFB;
    constexpr uint32_t prev_p = (curr_p != NO_PREV_DFB) ? PrevP : NO_PREV_DFB;

    constexpr bool reconf_a = (curr_a != NO_PREV_DFB) && (curr_a != prev_a);
    constexpr bool reconf_b = (curr_b != NO_PREV_DFB) && (curr_b != prev_b);
    constexpr bool reconf_p = (curr_p != NO_PREV_DFB) && (curr_p != prev_p);

    // Pack-side deferral: in heterogeneous chains, only the first opt-in pack
    // site (prev_p == NO_PREV_DFB) emits at boot. Later sites defer to per-stage
    // via `emit_per_stage_pack_reconfig`, where the 2-arg LLK form's cache check
    // handles intra-stage transitions cheaply and the per-iter wraparound from
    // last-pack-cb to first-pack-cb is correctly programmed.
    constexpr bool defer_pack_to_per_stage = PackHetero && (prev_p != NO_PREV_DFB);

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

// emit_per_stage_pack_reconfig<E, PrevPack, LastPackCb, PackHetero>()
//
// Per-stage pack reconfig used only when the chain has heterogeneous opt-in
// pack CBs (boot can't program all of them). For every opt-in pack site,
// emit the 2-arg `pack_reconfig_data_format(prev, curr)` form before that
// site's per-iter pack work, using wraparound `prev` for the first pack site
// (so iter k+1's site 0 sees iter k's site N-1 as the previous descriptor
// state). The LLK's compare-and-skip on matching formats keeps the cost to
// a few cycles when adjacent stages happen to share a dtype.
template <class E, uint32_t PrevPack, uint32_t LastPackCb, bool PackHetero>
ALWI void emit_per_stage_pack_reconfig() {
    if constexpr (!PackHetero) return;
    constexpr uint32_t curr_p = dfb_for_side<Side::Pack, E>();
    if constexpr (curr_p == NO_PREV_DFB) return;
    constexpr uint32_t prev_chain = PrevPack;
    // Wraparound: first opt-in pack site has no in-chain prev; on iter ≥ 1 the
    // packer ended the previous iter on `last_pack_cb`. The LLK 2-arg form does
    // the right thing on iter 0 too (cache check vs. boot-initialized state).
    constexpr uint32_t prev_p =
        (prev_chain != NO_PREV_DFB) ? prev_chain : LastPackCb;
    if constexpr (prev_p != NO_PREV_DFB) {
        pack_reconfig_data_format(prev_p, curr_p);
    }
}

// Pack-phase init (Pack* only). Pack is its own cohort (disjoint from math-MOP / SFPU),
// excluded from `hoist_compute_init` and always boot-hoisted here via `pack_init_for_each`.
// Reconfig is fold-driven (see emit_pre_element_transitions): homogeneous chains program
// the packer once at boot; heterogeneous chains defer later sites to per-stage emission so
// the per-iter wraparound stays correct.
template <uint32_t PrevA, uint32_t PrevB, uint32_t PrevP, bool PackHetero, class E>
ALWI void elem_pack_init() {
    if constexpr (is_pack_tile_op_v<E>) {
        emit_pre_element_transitions<E, PrevA, PrevB, PrevP, PackHetero>();
        E::init();
    }
}

// Hoisted pack-init dispatcher — visits each chain element by compile-time index and precomputes the
// element's reconfig facts from `ChainTraits<Es...>` so the per-element init carries only those ints
// (not the whole chain type) in its mangled name.
template <class... Es, std::size_t... Is>
ALWI void pack_init_for_each(std::index_sequence<Is...>) {
    (elem_pack_init<
         ChainTraits<Es...>::prev.srca[Is],
         ChainTraits<Es...>::prev.srcb[Is],
         ChainTraits<Es...>::prev.pack[Is],
         ChainTraits<Es...>::pack_hetero,
         Es>(),
     ...);
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

// Boot-time hoist of compute-cohort init (math-MOP and/or DEST-only ops). The chain dispatcher
// computes `HoistMath` from `chain_hoist_math_mop_v` and `HoistSfpu` from `chain_hoist_sfpu_v`,
// then this walk emits the element's transitions + init() only when its cohort is hoisted. The
// HoistSfpu leg gates on `is_dest_only_op_v`, so Fill/Rand init() ride along (chain_hoist_sfpu_v
// itself is decided from SFPU-op uniformity alone).
//
// PackTile is intentionally excluded from this walk — pack-side reconfig is
// emitted unconditionally at boot via `pack_init_for_each` (PACK cohort is
// disjoint from compute cohorts and is always hoisted).
template <bool HoistMath, bool HoistSfpu, uint32_t PrevA, uint32_t PrevB, uint32_t PrevP, bool PackHetero, class ElemT>
ALWI void hoist_compute_init_one([[maybe_unused]] ElemT& elem) {
    constexpr bool emit =
        (is_math_mop_op_v<ElemT> && HoistMath) ||
        (is_dest_only_op_v<ElemT> && HoistSfpu);
    if constexpr (emit) {
        emit_pre_element_transitions<ElemT, PrevA, PrevB, PrevP, PackHetero>();
        elem.init();  // instance dispatch (see convention note above): a runtime-stateful init reads its members here
    }
}

// Direct indexed fold (no generic-lambda closure): precompute each element's reconfig facts from
// `ChainTraits<Es...>` and hand them to the de-templated per-element init.
template <bool HoistMath, bool HoistSfpu, std::size_t... Is, class... Es>
ALWI void hoist_compute_init(std::index_sequence<Is...>, Es&... elts) {
    (hoist_compute_init_one<HoistMath, HoistSfpu,
         ChainTraits<Es...>::prev.srca[Is],
         ChainTraits<Es...>::prev.srcb[Is],
         ChainTraits<Es...>::prev.pack[Is],
         ChainTraits<Es...>::pack_hetero,
         std::remove_reference_t<Es>>(elts),
     ...);
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
// for each CB-reader: Block → flat (ht*Wt + wt), Row → wt, Col → ht,
// Scalar → 0.
// =============================================================================

namespace detail {

template <bool EmitMathInit, bool EmitSfpuInit, uint32_t PrevA, uint32_t PrevB, uint32_t PrevP, bool PackHetero, class ElemT>
ALWI void elem_apply_compute(
    const ElemT& elem,
    uint32_t i_flat,
    uint32_t ht,
    uint32_t wt,
    uint32_t inner_count,
    uint32_t chain_lane_width,
    uint32_t Ht,
    uint32_t Wt) {
    // Per-block streaming: pass chunk-local index `j` to exec so Block
    // returns the local CB-front offset (the just-waited window).
    constexpr bool use_local_idx = element_uses_per_block_index_v<ElemT>;
    if constexpr (is_pack_tile_op_v<ElemT>) {
        (void)elem; (void)i_flat; (void)ht; (void)wt; (void)inner_count;
        (void)chain_lane_width; (void)Ht; (void)Wt;
    } else if constexpr (is_cb_reader_op_v<ElemT>) {
        // Per-block-iteration input waits — mutually exclusive by policy (Streaming forces
        // block_size 1, so per_tile and per_block never both fire). The Bulk upfront wait is NOT
        // here: it's hoisted once to the chain boundary (elem_wait_upfront, pre-loop fold), so it's
        // placed exactly once rather than re-issued per block-iter relying on idempotency.
        elem.wait_per_tile(i_flat + inner_count);
        elem.wait_per_block(inner_count);
        if constexpr (EmitMathInit) {
            emit_pre_element_transitions<ElemT, PrevA, PrevB, PrevP, PackHetero>();
            elem.init();  // instance dispatch (see convention note above)
        }
        constexpr bool per_side = elem_needs_per_side_idx_v<ElemT>;
        for (uint32_t j = 0; j < inner_count; ++j) {
            if constexpr (per_side) {
                // Per-side path: chain hands both indices; element picks per operand.
                elem.exec(
                    /*i_flat_local=*/j,
                    /*i_flat_abs=*/(i_flat + j),
                    ht,
                    /*wt_local=*/j,
                    /*wt_abs=*/(wt + j),
                    SlotBase + j * chain_lane_width);
            } else {
                const uint32_t i_arg = use_local_idx ? j : (i_flat + j);
                elem.exec(i_arg, ht, wt + j, SlotBase + j * chain_lane_width);
            }
        }
        elem.pop_per_tile(i_flat);
        elem.pop_per_block(inner_count);
    } else if constexpr (is_dest_only_op_v<ElemT>) {
        if constexpr (EmitSfpuInit) {
            emit_pre_element_transitions<ElemT, PrevA, PrevB, PrevP, PackHetero>();
            elem.init();  // instance dispatch (see convention note above)
        }
        for (uint32_t j = 0; j < inner_count; ++j) {
            elem.exec(i_flat + j, SlotBase + j * chain_lane_width);
        }
    }
}

template <uint32_t PrevPack, uint32_t LastPackCb, bool PackHetero, bool PackReluHetero, class ElemT>
ALWI void elem_apply_pack(
    const ElemT& elem,
    uint32_t i_flat,
    uint32_t ht,
    uint32_t wt,
    uint32_t inner_count,
    uint32_t chain_lane_width,
    [[maybe_unused]] uint32_t Ht,
    [[maybe_unused]] uint32_t Wt) {
    constexpr bool use_local_idx = element_uses_per_block_index_v<ElemT>;
    if constexpr (is_pack_tile_op_v<ElemT>) {
        // upfront reserve is emitted once before the loop (see eltwise_chain_impl)
        emit_per_stage_pack_reconfig<ElemT, PrevPack, LastPackCb, PackHetero>();
        // Heterogeneous packer ReLU: this site sets its own mode right before its packs and restores
        // pass-through right after, so the next (differently-configured) pack site is unaffected and
        // the chain still exits with ReLU off. Homogeneous ReLU is bracketed once around the whole
        // loop in chain_run_loop, so this per-pack path is compiled out for it. Only sites that apply
        // ReLU act — a None site needs nothing because every ReLU site restores none() after itself.
        constexpr bool per_stage_relu = PackReluHetero && (ElemT::pack_relu != PackRelu::None);
        elem.reserve_per_tile(i_flat);
        elem.reserve_per_block(inner_count);
        if constexpr (per_stage_relu) {
            ckernel::pack_relu_config(ckernel::ReluConfig::zero());
        }
        for (uint32_t j = 0; j < inner_count; ++j) {
            const uint32_t i_arg = use_local_idx ? j : (i_flat + j);
            elem.exec(i_arg, ht, wt + j, j * chain_lane_width);
        }
        if constexpr (per_stage_relu) {
            ckernel::pack_relu_config(ckernel::ReluConfig::none());  // bring it back before publish
        }
        elem.push_per_tile(i_flat);
        elem.push_per_block(inner_count);
    } else {
        (void)elem; (void)i_flat; (void)ht; (void)wt; (void)inner_count;
        (void)chain_lane_width;
    }
}

template <bool EmitMathInit, bool EmitSfpuInit, uint32_t SlotBase, std::size_t... Is, class... Es>
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
    // Direct indexed fold: precompute each element's reconfig facts from `ChainTraits<Es...>` and call
    // the de-templated worker. No generic-lambda closure, and the worker's name no longer embeds Es....
    (elem_apply_compute<EmitMathInit, EmitSfpuInit,
         ChainTraits<Es...>::prev.srca[Is],
         ChainTraits<Es...>::prev.srcb[Is],
         ChainTraits<Es...>::prev.pack[Is],
         ChainTraits<Es...>::pack_hetero,
         std::remove_reference_t<Es>>(
         elts, i_flat, ht, wt, inner_count, chain_lane_width, Ht, Wt),
     ...);
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
    // Direct indexed fold (see apply_compute_phase): de-templated pack worker fed precomputed facts.
    (elem_apply_pack<
         ChainTraits<Es...>::prev.pack[Is],
         ChainTraits<Es...>::last_pack_cb,
         ChainTraits<Es...>::pack_hetero,
         ChainTraits<Es...>::pack_relu_hetero,
         std::remove_reference_t<Es>>(
         elts, i_flat, ht, wt, inner_count, chain_lane_width, Ht, Wt),
     ...);
}

// Seed-first L1 accumulation needs tile-major pack ordering. In particular, a chain may
// produce several accumulator tiles in the same output CB (for example x^2/x^4/x^6): every
// writer must overwrite its slot for logical tile zero before the packer's global accumulation
// mode is enabled. The L1-specific output lifecycles have no per-tile/per-block reserve or push,
// and the chain assertions require a single pack CB, so the normal element-major pack phase can
// be reduced to this tile-major exec fold.
template <std::size_t I, class ElemT, class... Es>
ALWI void elem_apply_seed_first_l1_pack_one(
    const ElemT& elem,
    uint32_t i_flat,
    uint32_t ht,
    uint32_t wt,
    uint32_t j,
    uint32_t chain_lane_width) {
    if constexpr (is_pack_tile_op_v<ElemT>) {
        constexpr bool use_local_idx = element_uses_per_block_index_v<ElemT>;
        const uint32_t i_arg = use_local_idx ? j : (i_flat + j);
        elem.exec(i_arg, ht, wt + j, j * chain_lane_width);
    } else {
        (void)elem;
        (void)i_flat;
        (void)ht;
        (void)wt;
        (void)j;
        (void)chain_lane_width;
    }
}

template <std::size_t... Is, class... Es>
ALWI void apply_seed_first_l1_pack_phase(
    std::index_sequence<Is...>,
    uint32_t i_flat,
    uint32_t ht,
    uint32_t wt,
    uint32_t inner_count,
    uint32_t chain_lane_width,
    Es&... elts) {
    for (uint32_t j = 0; j < inner_count; ++j) {
        auto run_one = [&](auto idx_const, auto& elem) {
            constexpr std::size_t II = decltype(idx_const)::value;
            using ElemT = std::remove_reference_t<decltype(elem)>;
            elem_apply_seed_first_l1_pack_one<II, ElemT, Es...>(
                elem, i_flat, ht, wt, j, chain_lane_width);
        };
        (run_one(std::integral_constant<std::size_t, Is>{}, elts), ...);

        // Mode changes only after every accumulator slot has received logical tile zero.
        if (i_flat == 0 && j == 0) {
            pack_reconfig_l1_acc(1);
        }
    }
}

template <class E>
ALWI void elem_wait_upfront(const E& e, uint32_t Ht, uint32_t Wt) {
    if constexpr (is_cb_reader_op_v<E>) {
        e.wait_upfront(Ht, Wt);
    }
}
template <class E>
ALWI void elem_reserve_upfront(const E& e, uint32_t Ht, uint32_t Wt) {
    if constexpr (is_cb_writer_op_v<E>) {
        e.reserve_upfront(Ht, Wt);
    }
}
template <class E>
ALWI void elem_pop_upfront_end(const E& e, uint32_t Ht, uint32_t Wt) {
    if constexpr (is_cb_reader_op_v<E>) {
        e.pop_upfront_end(Ht, Wt);
    }
}
template <class E>
ALWI void elem_push_at_end(const E& e, uint32_t Ht, uint32_t Wt) {
    if constexpr (is_cb_writer_op_v<E>) {
        e.push_at_end(Ht, Wt);
    }
}
// Per-outer-row hooks (InputLifecycle::OuterStream readers): wait at row entry, pop at row exit.
// Inert for every other policy.
template <class E>
ALWI void elem_wait_per_row(const E& e) {
    if constexpr (is_cb_reader_op_v<E>) {
        e.wait_per_row();
    }
}
template <class E>
ALWI void elem_pop_per_row(const E& e) {
    if constexpr (is_cb_reader_op_v<E>) {
        e.pop_per_row();
    }
}
template <class E>
ALWI void elem_reserve_per_row(const E& e) {
    if constexpr (is_cb_writer_op_v<E>) {
        e.reserve_per_row();
    }
}
template <class E>
ALWI void elem_push_per_row(const E& e) {
    if constexpr (is_cb_writer_op_v<E>) {
        e.push_per_row();
    }
}

}  // namespace detail

// Shared per-tile walk. `EmitMathInit`/`EmitSfpuInit` control whether the per-element compute init
// is emitted inside the loop. eltwise_chain_impl always passes `!hoist_*`: a hoistable cohort emits
// nothing here (it was hoisted to boot by this call under SetupOwner::Chain, or by the caller under
// SetupOwner::Caller), and a non-hoistable cohort re-inits per tile regardless.
template <bool EmitMathInit, bool EmitSfpuInit, class... Es>
ALWI void chain_run_loop(EltwiseShape shape, Es... elts) {
    using Chain = EltwiseChain<Es...>;
    // Block size lives on the shape. The DEST footprint is block_size * chain_lane_width;
    // the chain clamps block_size so it can never overflow DEST.
    constexpr uint32_t chain_lane_w = chain_lane_width_v<Chain>;
    uint32_t block_size = shape.block_size;
    // InputLifecycle::Streaming CB-reader chains can't multi-tile their DEST window — force
    // block_size to 1 (compile-time gated, so the override emits no code for block-capable chains).
    if constexpr (!chain_supports_block_v<Chain>) {
        block_size = 1;
    } else {
        // Clamp the runtime block_size to the chain's compile-time DEST capacity
        // (chain_max_block_v = DEST_AUTO_LIMIT / chain_lane_width). Clamping down is
        // correctness-safe — more outer iters, same total tile coverage.
        constexpr uint32_t max_block = chain_max_block_v<Chain>;
        if (block_size > max_block) {
            block_size = max_block;
        }
    }

    using IdxSeq = std::make_index_sequence<sizeof...(Es)>;
    const uint32_t Ht = shape.Ht;
    const uint32_t Wt = shape.Wt;

    // Upfront input wait + output reserve — each fires once for the whole Ht*Wt window for its
    // upfront policies (Bulk / HeldBulk / BulkDrain on the input wait; Bulk-reserve on the output),
    // bracketing the loop with the end-of-chain pop_upfront_end / push_at_end folds. The input
    // wait is hoisted here (not sprayed per block-iter) so it's placed exactly once — symmetric
    // with its pop_upfront_end partner — rather than relying on cb_wait_front idempotency.
    (detail::elem_wait_upfront(elts, Ht, Wt), ...);
    (detail::elem_reserve_upfront(elts, Ht, Wt), ...);

    // L1 accumulation is a packer-global mode. A preloaded accumulator enables it before the walk;
    // seed-first starts in overwrite mode and its tile-major pack phase enables it after every
    // output slot has packed logical tile zero. Both modes are reset before publication below.
    if constexpr (detail::ChainTraits<Es...>::any_l1_accumulation) {
        pack_reconfig_l1_acc(detail::ChainTraits<Es...>::any_seed_first_l1_accumulation ? 0 : 1);
    }

    // Packer ReLU (STACC_RELU) is a latched packer-global mode like L1 accumulation. When every pack
    // site shares one mode (homogeneous), program it once here and restore pass-through before
    // publication below — the cheap path. A heterogeneous chain does NOT bracket here; each ReLU pack
    // sets and restores its own mode in elem_apply_pack. `any_pack_relu && !pack_relu_hetero` means
    // "homogeneous and at least one site is ReLU" (all such sites share the single non-None mode).
    if constexpr (detail::ChainTraits<Es...>::any_pack_relu && !detail::ChainTraits<Es...>::pack_relu_hetero) {
        ckernel::pack_relu_config(ckernel::ReluConfig::zero());
    }

    // Outer 2D loop. `flat_base = ht * Wt + wt_base` is computed once per (ht, wt_base) pair.
    // Block-mode elements consume `flat_base + j`; bcast-mode read `ht` / `wt = wt_base + j`.
    for (uint32_t ht = 0; ht < Ht; ++ht) {
        const uint32_t row_base = ht * Wt;
        // Outer-axis streamed input operands (InputLifecycle::OuterStream): wait ONE tile at row
        // entry; the inner loop re-reads it at the front; pop it at row exit. Inert for every
        // other policy.
        (detail::elem_wait_per_row(elts), ...);
        for (uint32_t wt_base = 0; wt_base < Wt; wt_base += block_size) {
            const uint32_t inner_count = (wt_base + block_size <= Wt) ? block_size : (Wt - wt_base);
            const uint32_t i_flat = row_base + wt_base;
            tile_regs_acquire();
            detail::apply_compute_phase<EmitMathInit, EmitSfpuInit, 0>(
                IdxSeq{}, i_flat, ht, wt_base, inner_count, chain_lane_w, Ht, Wt, elts...);
            tile_regs_commit();
            tile_regs_wait();
            if constexpr (detail::ChainTraits<Es...>::any_seed_first_l1_accumulation) {
                detail::apply_seed_first_l1_pack_phase(
                    IdxSeq{}, i_flat, ht, wt_base, inner_count, chain_lane_w, elts...);
            } else {
                detail::apply_pack_phase(
                    IdxSeq{}, i_flat, ht, wt_base, inner_count, chain_lane_w, Ht, Wt, elts...);
            }
            tile_regs_release();
        }
        (detail::elem_pop_per_row(elts), ...);
    }

    // Reset before any output is published, so unrelated pack work after this chain cannot inherit
    // accumulation mode even if the consumer wakes immediately on the push below.
    if constexpr (detail::ChainTraits<Es...>::any_l1_accumulation) {
        pack_reconfig_l1_acc(0);
    }
    // Same escape concern for packer ReLU (homogeneous path only): restore pass-through before the
    // push below so a later, unrelated pack in this kernel isn't silently clamped by our latched
    // STACC_RELU mode. A heterogeneous chain already restored none() after its last ReLU pack.
    if constexpr (detail::ChainTraits<Es...>::any_pack_relu && !detail::ChainTraits<Es...>::pack_relu_hetero) {
        ckernel::pack_relu_config(ckernel::ReluConfig::none());
    }

    // End-of-chain upfront-policy lifecycle.
    (detail::elem_pop_upfront_end(elts, Ht, Wt), ...);
    (detail::elem_push_at_end(elts, Ht, Wt), ...);
}

// DEST-accumulation walk. Each outer row is one independent reduction: D0 stays acquired across
// that row's Wt inputs, is packed once, then is reset by the next row's acquire. Ordinary elements
// retain block-lane parallelism in D1..; the accumulating BinaryFpu ignores that offset and
// remains pinned to D0. A 1D shape (Ht=1) retains the single-output behavior.
template <bool EmitMathInit, bool EmitSfpuInit, class... Es>
ALWI void chain_run_dest_accumulation_loop(EltwiseShape shape, Es... elts) {
    using Chain = EltwiseChain<Es...>;
    constexpr uint32_t transient_lane_w = chain_transient_lane_width_v<Chain>;
    uint32_t block_size = shape.block_size;
    ASSERT(block_size > 0);

    if constexpr (!chain_supports_block_v<Chain>) {
        block_size = 1;
    } else if constexpr (transient_lane_w != 0) {
        constexpr uint32_t max_block = chain_max_block_v<Chain>;
        if (block_size > max_block) {
            block_size = max_block;
        }
    }

    using IdxSeq = std::make_index_sequence<sizeof...(Es)>;
    const uint32_t Ht = shape.Ht;
    const uint32_t Wt = shape.Wt;
    ASSERT(Ht > 0 && Wt > 0);

    (detail::elem_wait_upfront(elts, Ht, Wt), ...);
    (detail::elem_reserve_upfront(elts, Ht, Wt), ...);

    for (uint32_t ht = 0; ht < Ht; ++ht) {
        const uint32_t row_base = ht * Wt;
        (detail::elem_wait_per_row(elts), ...);
        (detail::elem_reserve_per_row(elts), ...);
        tile_regs_acquire();
        for (uint32_t wt_base = 0; wt_base < Wt; wt_base += block_size) {
            const uint32_t inner_count = (wt_base + block_size <= Wt) ? block_size : (Wt - wt_base);
            const uint32_t i_flat = row_base + wt_base;
            detail::apply_compute_phase<EmitMathInit, EmitSfpuInit, 1>(
                IdxSeq{}, i_flat, ht, wt_base, inner_count, transient_lane_w, Ht, Wt, elts...);
        }
        tile_regs_commit();

        tile_regs_wait();
        detail::apply_pack_phase(IdxSeq{}, row_base, ht, 0, 1, 0, Ht, Wt, elts...);
        tile_regs_release();

        (detail::elem_push_per_row(elts), ...);
        (detail::elem_pop_per_row(elts), ...);
    }

    (detail::elem_pop_upfront_end(elts, Ht, Wt), ...);
    (detail::elem_push_at_end(elts, Ht, Wt), ...);
}

// eltwise_chain_impl — the walk. SetupOwner::Chain (default) emits the chain's one-time setup
// (pack boot init + the uniform math-MOP / SFPU init + their srca/srcb reconfig) before walking.
// SetupOwner::Caller skips ALL of it: the caller emitted the chain's whole one-time setup itself,
// once, before its own loop, so this call is pure per-tile compute. SetupOwner is about WHO emits
// the hoistable setup — it never changes which init is hoistable (that's deduced from uniformity).
template <SetupOwner SO = SetupOwner::Chain, class... Es>
ALWI void eltwise_chain_impl(EltwiseShape shape, Es... elts) {
    using Chain = EltwiseChain<Es...>;
    static_assert(
        detail::ChainTraits<Es...>::any_dest_accumulation ==
            detail::ChainTraits<Es...>::any_dest_accumulation_lifecycle,
        "eltwise_chain: DEST accumulation requires OutputLifecycle::DestAccumulation or "
        "OutputLifecycle::DestAccumulationCallerManaged, and those lifecycles require an accumulating BinaryFpu");
    static_assert(
        !detail::ChainTraits<Es...>::any_dest_accumulation ||
            detail::ChainTraits<Es...>::dest_accumulation_slot_count == 1,
        "eltwise_chain: DEST accumulation supports exactly one unique sticky DEST slot");
    static_assert(
        !detail::ChainTraits<Es...>::any_dest_accumulation || detail::ChainTraits<Es...>::pack_writer_count == 1,
        "eltwise_chain: DEST accumulation requires exactly one PackTile");
    static_assert(
        !detail::ChainTraits<Es...>::any_dest_accumulation ||
            detail::ChainTraits<Es...>::dest_accumulation_pack_matches,
        "eltwise_chain: DEST accumulation PackTile must pack the sticky DEST slot");
    static_assert(
        !detail::ChainTraits<Es...>::any_dest_accumulation ||
            detail::ChainTraits<Es...>::all_writers_dest_accumulation_lifecycle,
        "eltwise_chain: DEST accumulation cannot mix accumulating and ordinary output lifecycles");
    static_assert(
        detail::ChainTraits<Es...>::managed_dest_accumulation_lifecycles <= 1,
        "eltwise_chain: only one PackTile may own the DEST-accumulation reserve-one/push-one lifecycle");
    static_assert(
        !detail::ChainTraits<Es...>::any_dest_accumulation ||
            detail::ChainTraits<Es...>::transient_lane_width < DEST_AUTO_LIMIT,
        "eltwise_chain: sticky D0 leaves insufficient DEST capacity for one transient lane");
    static_assert(
        !detail::ChainTraits<Es...>::any_dest_accumulation || !detail::ChainTraits<Es...>::any_l1_accumulation,
        "eltwise_chain: composing DEST and L1 accumulation is not supported yet");
    static_assert(
        !detail::ChainTraits<Es...>::any_l1_accumulation || detail::ChainTraits<Es...>::pack_dfbs_consistent,
        "eltwise_chain: L1 accumulation supports only one output CB per chain");
    static_assert(
        !detail::ChainTraits<Es...>::any_l1_accumulation || detail::ChainTraits<Es...>::all_writers_l1_accumulation,
        "eltwise_chain: a chain using L1 accumulation cannot mix accumulating and ordinary PackTile elements");
    static_assert(
        detail::ChainTraits<Es...>::l1_accumulation_modes_consistent,
        "eltwise_chain: accumulating PackTile elements must all use the same L1 accumulation mode");
    static_assert(
        detail::ChainTraits<Es...>::managed_l1_accumulation_lifecycles <= 1,
        "eltwise_chain: only one PackTile may own the L1-accumulation reserve-one/push-one lifecycle");
    // Packer ReLU (STACC_RELU) is a latched packer-global mode. A HOMOGENEOUS-ReLU chain programs it
    // once before the loop and restores it at exit (chain_run_loop). A HETEROGENEOUS chain (pack sites
    // disagree on ReLU) instead sets each ReLU pack's mode before its pack and restores pass-through
    // after it (elem_apply_pack) — so mixed ReLU is supported, just at a per-pack cost.
    // The packer-ReLU set/reset live only on the ordinary (non-DEST-accumulation) walk; a DEST-
    // accumulation chain routes through chain_run_dest_accumulation_loop and would silently drop the
    // activation. Forbid the combination until it's wired there.
    static_assert(!detail::ChainTraits<Es...>::any_pack_relu || !detail::ChainTraits<Es...>::any_dest_accumulation,
                  "eltwise_chain: packer ReLU combined with DEST accumulation is not supported yet");
    static_assert(!chain_has_duplicate_upfront_cbs_v<Chain>,
                  "eltwise_chain: two CB-reader elements share a CB on upfront-wait policy.");
    static_assert(!chain_pack_writes_collide_v<Chain>,
                  "eltwise_chain: two PackTile elements collide on (dfb, dst_slot).");
    // SetupOwner::Caller means "the caller did the chain's whole one-time setup itself, once,
    // before the loop." That's only achievable if EVERY part of it is boot-hoistable: uniform
    // math MOP + SFPU init (so input/srca-srcb reconfig is boot-only) AND homogeneous pack CBs
    // (so output/pack reconfig is boot-only — no per-stage pack reconfig in the loop). Otherwise
    // some setup must re-emit per tile, the caller can't pre-do it once, and the knob silently
    // skips a needed init/reconfig — a footgun. Forbid it at compile time.
    static_assert(SO == SetupOwner::Chain ||
                      (chain_hoist_math_mop_v<Chain> && chain_hoist_sfpu_v<Chain> &&
                       chain_hoist_pack_v<Chain>),
                  "SetupOwner::Caller requires a chain whose entire one-time setup is boot-hoistable "
                  "(uniform math MOP + SFPU init AND homogeneous pack CBs) so that input AND output "
                  "reconfig are boot-only and nothing self-emits per tile. This chain has setup that "
                  "must re-emit per tile, so the caller cannot own it once — use SetupOwner::Chain.");
    // Honesty: under SetupOwner::Caller the chain emits NO reconfig at all (the caller owns the
    // setup), so a non-None reconfig knob on any element is inert and lies about what runs inside
    // the helper. Forbid it — make the caller declare None, which truthfully says "the chain does
    // no reconfig; my manual setup owns the format."
    static_assert(SO == SetupOwner::Chain || chain_no_reconfig_requested_v<Chain>,
                  "SetupOwner::Caller with a non-None reconfig knob: under Caller the chain emits no "
                  "reconfig (the caller owns the setup), so the knob is inert and misleading. Set "
                  "every element's reconfig to None — the caller's manual setup owns the format.");
    // Per-cohort hoist decisions: math-MOP init can be hoisted at boot even when SFPU isn't
    // uniform; the SFPU side then re-inits per tile.
    constexpr bool hoist_math = chain_hoist_math_mop_v<Chain>;
    constexpr bool hoist_sfpu = chain_hoist_sfpu_v<Chain>;
    using IdxSeq = std::make_index_sequence<sizeof...(Es)>;
    if constexpr (SO == SetupOwner::Chain) {
        detail::pack_init_for_each<Es...>(IdxSeq{});
        detail::hoist_compute_init<hoist_math, hoist_sfpu>(IdxSeq{}, elts...);
    }
    if constexpr (detail::ChainTraits<Es...>::any_dest_accumulation) {
        chain_run_dest_accumulation_loop<!hoist_math, !hoist_sfpu>(shape, elts...);
    } else {
        chain_run_loop<!hoist_math, !hoist_sfpu>(shape, elts...);
    }
}

// =============================================================================
// 11c. Public eltwise_chain — forwards every element straight to eltwise_chain_impl.
//
// A compile-time-disabled optional (`OptionalChainElement<false, _>`) is a members-less,
// tag-less marker. Every op-kind trait (is_cb_reader / is_pack / is_dest_only / is_math_mop
// = std::is_base_of on a tag it lacks) is false for it, and every ElemDesc accessor is
// SFINAE-guarded (dfb_* gate on the op-kind, lane_width defaults to 1, etc.), so it reflects
// into a NEUTRAL ElemDesc — no CB, no reconfig, no pack, lane_width 1 — and every stage
// (planner, hoist, reconfig fold, per-tile loop) treats it as inert. It is transparent to
// the prev-CB sweep too: its NO_PREV_DFB sides never update the running prev, so a later
// element sees the same previous CB as if the marker were absent.
//
// We deliberately do NOT filter the marker out with `std::tuple_cat` / `std::get` (the old
// approach): that pulled the entire std::tuple template family into every chain kernel's -g
// debug info (~2/3 of .debug_str) solely to strip a no-op. Passing it through inert folds
// away to nothing at runtime and in code, and removes that debug bloat.
// =============================================================================

// Public entry. `SetupOwner SO` (default Chain) says who emits the chain's one-time setup:
// Chain = this call emits it; Caller = the caller emitted it once, outside its loop (see the
// SetupOwner enum doc). It never changes which init is hoistable. (default lives on the
// declaration in eltwise_chain.hpp.)
template <SetupOwner SO, class... Es>
ALWI void eltwise_chain(EltwiseShape shape, Es... elts) {
    eltwise_chain_impl<SO>(shape, elts...);
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
