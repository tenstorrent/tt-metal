// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_chain.inl
 * @brief Implementation of the eltwise chain pipeline + chain element types + traits.
 *
 * Included from `eltwise_chain.hpp`. Do NOT include directly.
 */

// (LLK / compute-API headers are pulled in via eltwise_chain.hpp.)

namespace compute_kernel_lib {

// =============================================================================
// Compile-time prev-CB / prev-fp32-dest-acc tracking (D2 + D6 fold infrastructure)
// =============================================================================
//
// Each chain element exposes static accessors describing which CB it routes to
// each side of the math/pack pipeline. InputLifecycle::Streaming and block elements use the same
// uniform accessors so the chain pipeline can compute, at compile time, the most
// recent CB seen on each Side (SrcA / SrcB / Pack) before any given element index.
//
// NO_PREV_CB is the sentinel used by elements that don't touch a side; the fold
// walks Es[0..I-1] backwards and returns the most recent non-NO_PREV_CB.

inline constexpr uint32_t NO_PREV_CB = 0xFFFFFFFFu;

enum class Side : uint8_t { SrcA, SrcB, Pack };

namespace detail {

// =============================================================================
// A0. 2D index-mode helpers (OperandKind → tile index / upfront window)
//
// Pure compile-time-elided helpers. `idx` and `window` are inlined by
// every CB-reader element's `exec` / `wait_upfront`. RISC-V cost: zero
// branches at run time — `if constexpr` collapses to a single arithmetic op.
//
//   Mode      | tile index            | window size
//   ----------|-----------------------|-----------------
//   FirstTile | 0                     | 1
//   BlockIter | i_flat (= ht*Wt+wt)   | Ht * Wt
//   RowBcast  | wt                    | Wt
//   ColBcast  | ht                    | Ht
//
// Runtime offsets are layered on top by `TileBase` (caller-side add to the
// idx result, and an inflation of the wait/pop count via `tile_base_value`).
//
// `is_bcast_mode_v<M>` is the predicate driving the (Policy × Mode) compatibility
// static_asserts on every CB-reader element (Row/Col modes reject streaming
// policies the same way `binary_op_helpers` rejects ROW/SCALAR + WaitAndPopPerTile).
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
//   static constexpr uint32_t cb_a_id();             // primary CB
//   static constexpr uint32_t cb_b_id();             // secondary CB or 0 if N/A
//   static constexpr InputLifecycle a_policy();
//   static constexpr InputLifecycle b_policy();
// (default impls below cover non-CB-reader elements.)
//
// Every CB-writer element must expose:
//   static constexpr uint32_t pack_cb_id();
//   static constexpr Dst pack_dst_slot();
//   static constexpr uint32_t pack_output_index();   // runtime fallback OK; used only for index mode FirstTile/Pinned k
// =============================================================================

template <class T, class = void> struct has_cb_a    : std::false_type {};
template <class T> struct has_cb_a<T, std::void_t<decltype(T::cb_a_id())>> : std::true_type {};

template <class T, class = void> struct has_cb_b    : std::false_type {};
template <class T> struct has_cb_b<T, std::void_t<decltype(T::cb_b_id())>> : std::true_type {};

template <class T, class = void> struct has_pack_cb : std::false_type {};
template <class T> struct has_pack_cb<T, std::void_t<decltype(T::pack_cb_id())>> : std::true_type {};

// Forward declarations — defined below (used by reader_pair_collide / writer_pair_collide
// earlier in the file's flow).
template <class E> constexpr uint32_t cb_a_of();
template <class E> constexpr uint32_t cb_b_of();
template <class E> constexpr uint32_t pack_cb_of();

// ChainTraits<Es...> — the one value-reflection aggregate for the whole chain (defined
// once below, after the per-element accessors it reads). Forward-declared here so the
// trait wrappers (chain_lane_width etc.) can name it.
template <class... Es> struct ChainTraits;

// =============================================================================
// Per-Side prev-CB SFINAE probe (D2)
//
// `cb_for_side<Side, E>` reads `E::reconfig_srca_cb` / `_srcb_cb` / `_pack_cb`
// when present, returns `NO_PREV_CB` otherwise. Block elements that haven't yet
// adopted the accessors (commit 3 / D7) still participate in the fold transparently.
// =============================================================================

template <class E, class = void>
struct has_reconfig_srca : std::false_type {};
template <class E>
struct has_reconfig_srca<E, std::void_t<decltype(E::reconfig_srca_cb)>> : std::true_type {};

template <class E, class = void>
struct has_reconfig_srcb : std::false_type {};
template <class E>
struct has_reconfig_srcb<E, std::void_t<decltype(E::reconfig_srcb_cb)>> : std::true_type {};

template <class E, class = void>
struct has_reconfig_pack : std::false_type {};
template <class E>
struct has_reconfig_pack<E, std::void_t<decltype(E::reconfig_pack_cb)>> : std::true_type {};

template <Side S, class E>
constexpr uint32_t cb_for_side() {
    if constexpr (S == Side::SrcA) {
        if constexpr (has_reconfig_srca<E>::value) return E::reconfig_srca_cb;
        else                                       return NO_PREV_CB;
    } else if constexpr (S == Side::SrcB) {
        if constexpr (has_reconfig_srcb<E>::value) return E::reconfig_srcb_cb;
        else                                       return NO_PREV_CB;
    } else {  // Pack
        if constexpr (has_reconfig_pack<E>::value) return E::reconfig_pack_cb;
        else                                       return NO_PREV_CB;
    }
}

// Per-side prev-CB history, last opt-in pack CB, and heterogeneous-pack detection are
// now single-sweep fields on `ChainTraits` (prev / last_pack_cb / pack_hetero), computed
// once from the reflected ElemDesc array instead of re-walked per emit site (was O(N²)).

}  // namespace detail

// =============================================================================
// 1. CopyTile chain element
// =============================================================================

template <uint32_t Cb,
          Dst DstSlot,
          InputLifecycle Policy,
          OperandKind IndexMode,
          CopyTileReconfig Reconfig,
          TileOffset Offset>
struct CopyTile : CopyTileTag {
    // ---- compile-time validation ----
    static_assert(to_u32(DstSlot) < DEST_AUTO_LIMIT,
                  "CopyTile: DEST slot exceeds DEST_AUTO_LIMIT");
    // Comprehensive (IndexMode, Policy) legality. Block rejects PerTile-pop
    // (InputLifecycle::Streaming/InputLifecycle::BulkDrain/InputLifecycle::NoWaitPop — absolute-idx footgun) and PerTile-wait-of-1
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

    static constexpr uint32_t       cb              = Cb;
    static constexpr uint32_t       cb_a_id()       { return Cb; }
    static constexpr uint32_t       cb_b_id()       { return 0;  }
    static constexpr OperandKind    a_index_mode    = IndexMode;
    static constexpr OperandKind    b_index_mode    = OperandKind::Scalar;
    static constexpr Dst            dst_slot        = DstSlot;
    static constexpr InputLifecycle a_policy()      { return Policy; }
    static constexpr InputLifecycle b_policy()      { return InputLifecycle::CallerManaged; }
    static constexpr bool           is_upfront      = (Policy == InputLifecycle::Bulk) ||
                                                      (Policy == InputLifecycle::HeldBulk) ||
                                                      (Policy == InputLifecycle::Pipelined);
    static constexpr bool           clashes_with_fpu= true;   // copy_tile uses unpacker MOP

    // Prev-CB fold (D2): CopyTile loads CbA only.
    static constexpr uint32_t       reconfig_srca_cb = (Reconfig == CopyTileReconfig::Input) ? Cb : NO_PREV_CB;
    static constexpr uint32_t       reconfig_srcb_cb = NO_PREV_CB;
    static constexpr uint32_t       reconfig_pack_cb = NO_PREV_CB;

    uint32_t tile_base = 0;

    constexpr CopyTile() noexcept = default;
    constexpr explicit CopyTile(uint32_t base) noexcept : tile_base(base) {}

    // ---- chain pipeline hooks ----
    static ALWI void init() {
        if constexpr (Reconfig == CopyTileReconfig::Input) {
            // Single-arg reconfig — no previous-CB tracking. Per design: omitting `old_cb`
            // is good enough vs the two-arg `_with_dt` form.
            reconfig_data_format_srca(Cb);
            copy_tile_to_dst_init_short(Cb);
        } else {
            copy_tile_init(Cb);
        }
    }

    /// Per-iter wait. Element fires at its own granularity — streaming policies wait 1
    /// per iter, Cumulative grows wait count with i (i+1), Upfront fires once via
    /// wait_upfront with full n_tiles. None scale by chain block_size — block_size
    /// only drives the inner DEST-lane loop and slot_offset.
    ALWI void wait_per_tile(uint32_t cumulative_count) const {
        if constexpr (Policy == InputLifecycle::Streaming || Policy == InputLifecycle::HeldStream) {
            cb_wait_front(Cb, 1);
        } else if constexpr (Policy == InputLifecycle::Pipelined ||
                             Policy == InputLifecycle::HeldCumulative) {
            cb_wait_front(Cb, cumulative_count);
        }
    }

    /// Per-outer-iter wait of `inner_count` tiles (chunked streaming).
    /// inner_count == BlockSize for steady iters, == tail size for the last iter.
    ALWI void wait_per_block(uint32_t inner_count) const {
        if constexpr (Policy == InputLifecycle::Chunked) {
            cb_wait_front(Cb, inner_count);
        }
    }



    // 2D variants — Ht/Wt-aware. Routes through `idx` and `window`; TileBase
    // adds the runtime offset on top. InputLifecycle::Streaming policies handled by the same
    // `wait_per_tile` / `pop_per_tile` as 1D.
    ALWI void wait_upfront(uint32_t Ht, uint32_t Wt) const {
        if constexpr (Policy == InputLifecycle::Bulk ||
                      Policy == InputLifecycle::HeldBulk ||
                      Policy == InputLifecycle::BulkDrain) {
            cb_wait_front(Cb, detail::window<IndexMode>(Ht, Wt) + tile_base_value<Offset>(tile_base));
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
            cb_pop_front(Cb, detail::window<IndexMode>(Ht, Wt) + tile_base_value<Offset>(tile_base));
        }
    }

    static constexpr uint32_t lane_width = to_u32(DstSlot) + 1;

    ALWI void pop_per_tile(uint32_t /*i*/) const {
        if constexpr (Policy == InputLifecycle::Streaming ||
                      Policy == InputLifecycle::NoWaitPop ||
                      Policy == InputLifecycle::BulkDrain) {
            cb_pop_front(Cb, 1);
        }
    }

    /// Per-outer-iter pop of `inner_count` tiles (chunked streaming).
    ALWI void pop_per_block(uint32_t inner_count) const {
        if constexpr (Policy == InputLifecycle::Chunked) {
            cb_pop_front(Cb, inner_count);
        }
    }

};

// =============================================================================
// 2. PackTile chain element
// =============================================================================

template <uint32_t Cb,
          Dst DstSlot,
          OutputLifecycle Policy,
          PackTileReconfig Reconfig,
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

    static constexpr uint32_t          cb                  = Cb;
    static constexpr uint32_t          pack_cb_id()        { return Cb; }
    static constexpr Dst               pack_dst_slot       = DstSlot;
    static constexpr bool              is_upfront          = (Policy == OutputLifecycle::Bulk);
    static constexpr bool              uses_per_block_pack = (Policy == OutputLifecycle::Chunked);
    // Walk vs pinned output addressing is DERIVED from the OutputLifecycle (no caller knob):
    // OutputLifecycle::Bulk reserves the whole window upfront and writes distinct tiles into it (walk); every
    // other policy advances the CB front via per-tile/chunk reserve+push, so the write index
    // stays pinned at base. (For a 1-tile output, walk and pinned are identical: base + 0 == base.)
    static constexpr bool              walk                = (Policy == OutputLifecycle::Bulk);

    // Prev-CB fold (D2): PackTile writes pack-side; mark Cb under reconfig only when
    // the user opted into pack reconfig (Output). Otherwise no pack reconfig is
    // emitted — fold keeps prior pack target.
    static constexpr uint32_t          reconfig_srca_cb    = NO_PREV_CB;
    static constexpr uint32_t          reconfig_srcb_cb    = NO_PREV_CB;
    static constexpr uint32_t          reconfig_pack_cb    =
        (Reconfig == PackTileReconfig::Output) ? Cb : NO_PREV_CB;

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
            cb_reserve_back(Cb, 1);
        }
    }

    /// Per-outer-iter reserve of `inner_count` tiles (chunked streaming).
    ALWI void reserve_per_block(uint32_t inner_count) const {
        if constexpr (Policy == OutputLifecycle::Chunked) {
            cb_reserve_back(Cb, inner_count);
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
    ALWI void reserve_upfront(uint32_t Ht, uint32_t Wt) const {
        if constexpr (Policy == OutputLifecycle::Bulk) {
            cb_reserve_back(Cb, (Ht * Wt) + tile_base_value<Offset>(tile_base));
        }
    }
    ALWI void push_at_end(uint32_t Ht, uint32_t Wt) const {
        if constexpr (Policy == OutputLifecycle::DeferredReserve ||
                      Policy == OutputLifecycle::Bulk) {
            cb_push_back(Cb, (walk ? (Ht * Wt) : 1u) + tile_base_value<Offset>(tile_base));
        }
    }

    static constexpr uint32_t lane_width = to_u32(DstSlot) + 1;

    ALWI void push_per_tile(uint32_t /*i*/) const {
        if constexpr (Policy == OutputLifecycle::Streaming) {
            cb_push_back(Cb, 1);
        }
    }

    /// Per-outer-iter push of `inner_count` tiles (chunked streaming).
    ALWI void push_per_block(uint32_t inner_count) const {
        if constexpr (Policy == OutputLifecycle::Chunked) {
            cb_push_back(Cb, inner_count);
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
          BinaryDataFormatReconfig DfReconfig,
          InputLifecycle APolicy,
          InputLifecycle BPolicy,
          OperandKind AIndex,
          Dst DstSlot,
          OperandKind BIndex,
          TileOffset OffsetA,
          TileOffset OffsetB>
struct BinaryFpu : BinaryFpuTag {
    static_assert(to_u32(DstSlot) < DEST_AUTO_LIMIT,
                  "BinaryFpu: DEST slot exceeds DEST_AUTO_LIMIT");
    // Comprehensive per-side (IndexMode, Policy) legality. Block rejects PerTile-pop
    // (InputLifecycle::Streaming/InputLifecycle::BulkDrain/InputLifecycle::NoWaitPop — absolute-idx footgun) and PerTile-wait-of-1
    // (InputLifecycle::HeldStream — never tracks per-iter requirement). Scalar/Row/Col accept every
    // legal lifecycle — caller-sized.
    static_assert(is_legal_kind_lifecycle(AIndex, APolicy),
                  "BinaryFpu: (AIndex, APolicy) is illegal for Block — exclude "
                  "InputLifecycle::Streaming / InputLifecycle::HeldStream / InputLifecycle::BulkDrain / InputLifecycle::NoWaitPop on Block walkers.");
    static_assert(is_legal_kind_lifecycle(BIndex, BPolicy),
                  "BinaryFpu: (BIndex, BPolicy) is illegal for Block — exclude "
                  "InputLifecycle::Streaming / InputLifecycle::HeldStream / InputLifecycle::BulkDrain / InputLifecycle::NoWaitPop on Block walkers.");
    // same_cb dedup safety: when CbA == CbB the B-side wait/pop is skipped, so the
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
    // `needs_per_side_idx`. Same-regime hits the 2-arg fast path identical to
    // pre-extension behaviour.

    static constexpr uint32_t      cb_a_id()  { return CbA; }
    static constexpr uint32_t      cb_b_id()  { return CbB; }
    static constexpr OperandKind   a_index_mode = AIndex;
    static constexpr OperandKind   b_index_mode = BIndex;
    static constexpr InputLifecycle a_policy(){ return APolicy; }
    static constexpr InputLifecycle b_policy(){ return BPolicy; }
    static constexpr Dst           dst_slot   = DstSlot;
    static constexpr bool          is_upfront = (APolicy == InputLifecycle::Bulk) ||
                                                (APolicy == InputLifecycle::HeldBulk) ||
                                                (APolicy == InputLifecycle::Pipelined) ||
                                                (BPolicy == InputLifecycle::Bulk) ||
                                                (BPolicy == InputLifecycle::HeldBulk) ||
                                                (BPolicy == InputLifecycle::Pipelined);
    static constexpr bool          clashes_with_fpu = true;
    static constexpr bool          same_cb    = (CbA == CbB);

    // Per-side local-vs-absolute index resolution. When the two operands declare
    // DIFFERENT regimes (A=PerBlock + B=Upfront, or vice versa), the chain calls
    // the 3-arg exec / exec overload and passes both indices; each side picks.
    // Same-regime falls through to the 2-arg forwarder identical to today's code.
    static constexpr bool a_uses_local_idx = (APolicy == InputLifecycle::Chunked);
    static constexpr bool b_uses_local_idx = (BPolicy == InputLifecycle::Chunked);
    static constexpr bool needs_per_side_idx = (a_uses_local_idx != b_uses_local_idx);

    // Prev-CB fold (D2): BinaryFpu touches srca (CbA) and srcb (CbB) only. Pack-side
    // reconfig is owned by the downstream PackTile element (`PackTileReconfig::Output`)
    // — BinaryFpu writes to DEST, not to a CB, so it has no pack-side responsibility.
    //
    // Per-side selection (Input / SrcA / SrcB) lets the caller opt into a single-side
    // fold when the other side is already programmed (by a previous chain element on
    // the same side, or by external init).
    static constexpr uint32_t      reconfig_srca_cb =
        (DfReconfig == BinaryDataFormatReconfig::Input ||
         DfReconfig == BinaryDataFormatReconfig::SrcA) ? CbA : NO_PREV_CB;
    static constexpr uint32_t      reconfig_srcb_cb =
        (DfReconfig == BinaryDataFormatReconfig::Input ||
         DfReconfig == BinaryDataFormatReconfig::SrcB) ? CbB : NO_PREV_CB;
    static constexpr uint32_t      reconfig_pack_cb = NO_PREV_CB;

    uint32_t tile_base_a = 0;
    uint32_t tile_base_b = 0;

    constexpr BinaryFpu() noexcept = default;
    constexpr BinaryFpu(uint32_t a, uint32_t b) noexcept : tile_base_a(a), tile_base_b(b) {}
    constexpr explicit BinaryFpu(uint32_t a) noexcept : tile_base_a(a) {}

    // Helper: when same_cb, both bases live in the single shared wait window.
    // Wait/pop count uses max(base_a, base_b) — caller must stage that many tiles
    // in front of both reads.
    ALWI uint32_t same_cb_base_max() const noexcept {
        const uint32_t bA = tile_base_value<OffsetA>(tile_base_a);
        const uint32_t bB = tile_base_value<OffsetB>(tile_base_b);
        return bA > bB ? bA : bB;
    }

    // ---- init / reconfig ----
    // F-PERF-3: srca / srcb / pack reconfig are now fold-driven (compile-time-elided
    // when prev_*_cb == cur_*_cb). init() programs only the per-op LLK shape.
    static ALWI void init() {
        // Op-specific init.
        if constexpr (Bcast == BroadcastDim::None) {
            if constexpr      (Op == BinaryFpuOp::Add) add_tiles_init(CbA, CbB);
            else if constexpr (Op == BinaryFpuOp::Sub) sub_tiles_init(CbA, CbB);
            else                                       mul_tiles_init(CbA, CbB);
        } else {
            // Reg A fix v2: use *_init_short form from bcast.h:352-446. The original
            // init_bcast<>() was the BIG init (hw_configure + pack_dest_init + sync_init)
            // — undefined mid-MAIN per D8. The previous patch used non-operand math init
            // which programs DEFAULT_TENSOR_SHAPE; we now switch to the _with_operands form
            // that reads the actual tensor shape from CB metadata via get_operand_tensor_shape.
            // This matches `add_bcast_rows_init_short` / `sub_bcast_cols_init_short` etc.
            // exactly — math init + unpack init only, no hw_configure / pack_dest_init / sync_init.
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
            cb_wait_front(CbA, 1);
        } else if constexpr (APolicy == InputLifecycle::Pipelined ||
                             APolicy == InputLifecycle::HeldCumulative) {
            cb_wait_front(CbA, cumulative_count);
        }
        if constexpr (!same_cb) {
            if constexpr (BPolicy == InputLifecycle::Streaming || BPolicy == InputLifecycle::HeldStream) {
                cb_wait_front(CbB, 1);
            } else if constexpr (BPolicy == InputLifecycle::Pipelined ||
                                 BPolicy == InputLifecycle::HeldCumulative) {
                cb_wait_front(CbB, cumulative_count);
            }
        }
    }

    /// Per-outer-iter chunked wait. Per-side: A waits `inner_count` if APolicy is
    /// per-block; same for B (same_cb dedup).
    ALWI void wait_per_block(uint32_t inner_count) const {
        if constexpr (APolicy == InputLifecycle::Chunked) {
            cb_wait_front(CbA, inner_count);
        }
        if constexpr (!same_cb && BPolicy == InputLifecycle::Chunked) {
            cb_wait_front(CbB, inner_count);
        }
    }


    // 2D: per-side upfront wait — A uses AIndex's window, B uses BIndex's window.
    // Same `same_cb` dedup as 1D (skip B side when CbA == CbB).
    ALWI void wait_upfront(uint32_t Ht, uint32_t Wt) const {
        if constexpr (APolicy == InputLifecycle::Bulk ||
                      APolicy == InputLifecycle::HeldBulk ||
                      APolicy == InputLifecycle::BulkDrain) {
            const uint32_t a_base = same_cb ? same_cb_base_max() : tile_base_value<OffsetA>(tile_base_a);
            cb_wait_front(CbA, detail::window<AIndex>(Ht, Wt) + a_base);
        }
        if constexpr (!same_cb && (BPolicy == InputLifecycle::Bulk ||
                                   BPolicy == InputLifecycle::HeldBulk ||
                                   BPolicy == InputLifecycle::BulkDrain)) {
            cb_wait_front(CbB, detail::window<BIndex>(Ht, Wt) + tile_base_value<OffsetB>(tile_base_b));
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
            cb_pop_front(CbA, 1);
        }
        if constexpr (!same_cb && (BPolicy == InputLifecycle::Streaming ||
                                   BPolicy == InputLifecycle::NoWaitPop ||
                                   BPolicy == InputLifecycle::BulkDrain)) {
            cb_pop_front(CbB, 1);
        }
    }

    ALWI void pop_per_block(uint32_t inner_count) const {
        if constexpr (APolicy == InputLifecycle::Chunked) {
            cb_pop_front(CbA, inner_count);
        }
        if constexpr (!same_cb && BPolicy == InputLifecycle::Chunked) {
            cb_pop_front(CbB, inner_count);
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
            const uint32_t a_base = same_cb ? same_cb_base_max() : tile_base_value<OffsetA>(tile_base_a);
            cb_pop_front(CbA, detail::window<AIndex>(Ht, Wt) + a_base);
        }
        if constexpr (!same_cb && (BPolicy == InputLifecycle::Bulk ||
                                   BPolicy == InputLifecycle::Pipelined ||
                                   BPolicy == InputLifecycle::DeferredPop)) {
            cb_pop_front(CbB, detail::window<BIndex>(Ht, Wt) + tile_base_value<OffsetB>(tile_base_b));
        }
    }
};

// =============================================================================
// 5. DestReuseBinary chain element
// =============================================================================

template <uint32_t Cb,
          BinaryFpuOp Op,
          DestReuseType ReuseType,
          Dst DstIn,
          Dst DstOut,
          DestReuseReconfig Reconfig,
          InputLifecycle Policy,
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

    static constexpr uint32_t       cb_a_id()         { return Cb; }
    static constexpr uint32_t       cb_b_id()         { return 0;  }
    static constexpr OperandKind    a_index_mode     = IndexMode;
    static constexpr OperandKind    b_index_mode     = OperandKind::Scalar;
    static constexpr InputLifecycle a_policy()        { return Policy; }
    static constexpr InputLifecycle b_policy()        { return InputLifecycle::CallerManaged; }
    static constexpr Dst            dst_slot          = DstOut;
    static constexpr bool           is_upfront        = (Policy == InputLifecycle::Bulk) ||
                                                        (Policy == InputLifecycle::HeldBulk) ||
                                                        (Policy == InputLifecycle::Pipelined);
    static constexpr bool           clashes_with_fpu  = true;

    // Prev-CB fold (D2): DestReuseBinary loads CB into srca (when DEST → srcb) or srcb
    // (when DEST → srca). Reconfig only fires when opted in.
    //
    // `Input` follows ReuseType (programs the side the CB actually unpacks into).
    // `SrcA` / `SrcB` explicitly pick a side, decoupled from ReuseType — used when
    // the caller wants to program a specific unpack lane regardless of which lane
    // DEST is feeding into.
    static constexpr uint32_t       reconfig_srca_cb  =
        ((Reconfig == DestReuseReconfig::Input && ReuseType == DestReuseType::DEST_TO_SRCB) ||
         Reconfig == DestReuseReconfig::SrcA) ? Cb : NO_PREV_CB;
    static constexpr uint32_t       reconfig_srcb_cb  =
        ((Reconfig == DestReuseReconfig::Input && ReuseType == DestReuseType::DEST_TO_SRCA) ||
         Reconfig == DestReuseReconfig::SrcB) ? Cb : NO_PREV_CB;
    static constexpr uint32_t       reconfig_pack_cb  = NO_PREV_CB;

    uint32_t tile_base = 0;

    constexpr DestReuseBinary() noexcept = default;
    constexpr explicit DestReuseBinary(uint32_t base) noexcept : tile_base(base) {}

    // F-PERF-3: srca / srcb reconfig is fold-driven; init() programs only the per-op
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
            cb_wait_front(Cb, 1);
        } else if constexpr (Policy == InputLifecycle::Pipelined ||
                             Policy == InputLifecycle::HeldCumulative) {
            cb_wait_front(Cb, cumulative_count);
        }
    }
    ALWI void wait_per_block(uint32_t inner_count) const {
        if constexpr (Policy == InputLifecycle::Chunked) {
            cb_wait_front(Cb, inner_count);
        }
    }

    // 2D variants
    ALWI void wait_upfront(uint32_t Ht, uint32_t Wt) const {
        if constexpr (Policy == InputLifecycle::Bulk ||
                      Policy == InputLifecycle::HeldBulk ||
                      Policy == InputLifecycle::BulkDrain) {
            cb_wait_front(Cb, detail::window<IndexMode>(Ht, Wt) + tile_base_value<Offset>(tile_base));
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
            cb_pop_front(Cb, detail::window<IndexMode>(Ht, Wt) + tile_base_value<Offset>(tile_base));
        }
    }

    static constexpr uint32_t lane_width =
        (to_u32(DstIn) > to_u32(DstOut)) ? (to_u32(DstIn) + 1) : (to_u32(DstOut) + 1);
    ALWI void pop_per_tile(uint32_t /*i*/) const {
        if constexpr (Policy == InputLifecycle::Streaming ||
                      Policy == InputLifecycle::NoWaitPop ||
                      Policy == InputLifecycle::BulkDrain) {
            cb_pop_front(Cb, 1);
        }
    }
    ALWI void pop_per_block(uint32_t inner_count) const {
        if constexpr (Policy == InputLifecycle::Chunked) {
            cb_pop_front(Cb, inner_count);
        }
    }
};

// =============================================================================
// 6. UnaryBcast chain element
// =============================================================================

template <BroadcastDim Dim,
          uint32_t Cb,
          Dst DstSlot,
          InputLifecycle Policy,
          UnaryBcastReconfig Reconfig>
struct UnaryBcast : UnaryBcastTag {
    static_assert(to_u32(DstSlot) < DEST_AUTO_LIMIT,
                  "UnaryBcast: DEST slot exceeds DEST_AUTO_LIMIT");

    static constexpr uint32_t       cb_a_id()         { return Cb; }
    static constexpr uint32_t       cb_b_id()         { return 0;  }
    static constexpr InputLifecycle a_policy()        { return Policy; }
    static constexpr InputLifecycle b_policy()        { return InputLifecycle::CallerManaged; }
    static constexpr Dst            dst_slot          = DstSlot;
    static constexpr bool           is_upfront        = (Policy == InputLifecycle::Bulk) ||
                                                        (Policy == InputLifecycle::HeldBulk) ||
                                                        (Policy == InputLifecycle::Pipelined);
    static constexpr bool           clashes_with_fpu  = true;

    // Prev-CB fold (D2): UnaryBcast binds BOTH srca and srcb to Cb. The broadcast datacopy MOP
    // drives the FPU SrcB lane (ELWADD + SRCB_BCAST_*), so srcb must be reprogrammed too — a
    // srca-only reconfig leaves ALU_FORMAT_SPEC_REG1_SrcB stale from a preceding two-operand op
    // (e.g. layernorm's BinaryFpu(cb_ex2, cb_eps) leaves SrcB = cb_eps), which corrupts the bcast.
    // Declaring both CBs lets the chain's reconfig fold (emit_pre_element_transitions) emit the
    // reconfig before init() AND record Cb as the post-element srca/srcb state for the next
    // element — so a subsequent srca/srcb reader sees the correct prev-CB and won't wrongly elide.
    // Pack-side reconfig is owned by the downstream PackTile (PackTileReconfig::Output), exactly
    // like BinaryFpu — UnaryBcast never configures pack.
    static constexpr uint32_t       reconfig_srca_cb  = (Reconfig == UnaryBcastReconfig::Input) ? Cb : NO_PREV_CB;
    static constexpr uint32_t       reconfig_srcb_cb  = (Reconfig == UnaryBcastReconfig::Input) ? Cb : NO_PREV_CB;
    static constexpr uint32_t       reconfig_pack_cb  = NO_PREV_CB;

    static ALWI void init() {
        constexpr auto bt = static_cast<ckernel::BroadcastType>(static_cast<uint8_t>(Dim));
        // Small per-element init only — the caller owns BIG init (compute_kernel_hw_startup /
        // a boot unary_bcast_init). This does NOT re-run any hw_configure or pack init, and it
        // does NOT do the srca/srcb format reconfig: that is fold-driven (see reconfig_srca_cb /
        // reconfig_srcb_cb above), emitted by emit_pre_element_transitions() before this init().
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
            cb_wait_front(Cb, 1);
        } else if constexpr (Policy == InputLifecycle::Pipelined ||
                             Policy == InputLifecycle::HeldCumulative) {
            cb_wait_front(Cb, cumulative_count);
        }
    }
    ALWI void wait_per_block(uint32_t inner_count) const {
        if constexpr (Policy == InputLifecycle::Chunked) {
            cb_wait_front(Cb, inner_count);
        }
    }

    // 2D variants — UnaryBcast always reads tile 0 (intra-tile bcast LLK), no per-iter
    // tile index. Upfront window in 2D = Ht * Wt (every (ht, wt) iter consumes one tile).
    ALWI void wait_upfront(uint32_t Ht, uint32_t Wt) const {
        if constexpr (Policy == InputLifecycle::Bulk ||
                      Policy == InputLifecycle::HeldBulk ||
                      Policy == InputLifecycle::BulkDrain) {
            cb_wait_front(Cb, Ht * Wt);
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
            cb_pop_front(Cb, Ht * Wt);
        }
    }

    static constexpr uint32_t lane_width = to_u32(DstSlot) + 1;
    ALWI void pop_per_tile(uint32_t /*i*/) const {
        if constexpr (Policy == InputLifecycle::Streaming ||
                      Policy == InputLifecycle::NoWaitPop ||
                      Policy == InputLifecycle::BulkDrain) {
            cb_pop_front(Cb, 1);
        }
    }
    ALWI void pop_per_block(uint32_t inner_count) const {
        if constexpr (Policy == InputLifecycle::Chunked) {
            cb_pop_front(Cb, inner_count);
        }
    }
};

// =============================================================================
// 7. Fill / Rand chain elements — declarations only in v1 core.
//    Implementations live in eltwise_fill.hpp / eltwise_rand.hpp (forthcoming).
//    Forward declarations satisfy the trait predicates without forcing the
//    fill/rand LLK headers into every chain-using kernel's include cone.
// =============================================================================
//
// (FillScalar / FillInt / FillBitcast / RandTile are already declared in eltwise_chain.hpp.
//  When eltwise_fill.hpp ships, it specializes those templates with full bodies.)

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

// chain_lane_width — N-element fold (item 2). Max of per-element `lane_width`. Bounds
// the legal BlockSize at the chain call site via the static_assert
// `BlockSize * chain_lane_width <= DEST_AUTO_LIMIT`. Each element writes to
// DEST[dst_slot + j * chain_lane_width] for lane j in [0, BlockSize).
//
// SFINAE fallback: elements that don't expose a `lane_width` member (caller-defined
// chain elements that inherit directly from `CopyTileTag` / `PackTileTag` / `DestOnlyTag`
// without inheriting from `UnaryOp`/`BinaryOp`/`TernaryOp`/`QuaternaryOp` bases) default
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
        return policy_supports_block(E::a_policy()) && policy_supports_block(E::b_policy());
    } else {
        return true;  // non-CB-reader elements don't constrain block_size
    }
}

// 1D-only chain entry points cannot resolve Row/Col indexing — there is no
// Ht/Wt context to drive `idx<Row>(...) = wt` or `idx<Col>(...) = ht`.
// In 1D, exec collapses Row/Col to `base` (silently degenerate). Banning these
// kinds at the 1D dispatch site forces callers to either pick Block/Scalar
// (the only kinds that make sense without Ht/Wt) or switch to the 2D
// `eltwise_chain(EltwiseShape, ...)` overload.
template <class E, class = void>
struct elem_has_a_index_mode : std::false_type {};
template <class E>
struct elem_has_a_index_mode<E, std::void_t<decltype(E::a_index_mode)>> : std::true_type {};

template <class E, class = void>
struct elem_has_b_index_mode : std::false_type {};
template <class E>
struct elem_has_b_index_mode<E, std::void_t<decltype(E::b_index_mode)>> : std::true_type {};

}  // namespace detail

// =============================================================================
// ChainTraits — reflect each element once into an ElemDesc record, then derive every
// value-based chain property as a field. Replaces the scattered fold / array-scan /
// head-tail-recursion passes (proposal kind ①). Type-uniformity (chain_*_uniform) stays
// separate — it reads element types, not values (kind ②). Emission stays separate (kind ③).
// All compile-time: the whole struct folds to constants (verified — zero runtime cost).
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

// One plain-data descriptor per element — reflected once via the existing accessors.
struct ElemDesc {
    bool is_cb_reader;
    bool is_pack;
    uint32_t srca_cb;      // cb_for_side<SrcA> (NO_PREV_CB when not programmed)  — G1 + prev input
    uint32_t srcb_cb;      // cb_for_side<SrcB>
    uint32_t pack_side_cb; // cb_for_side<Pack> (reconfig_pack_cb) — prev / last_pack / hetero input
    uint32_t cb_a;         // cb_a_of (kNoCb when n/a)  — reader-collision input
    uint32_t cb_b;         // cb_b_of (kNoCb when n/a)
    uint32_t pack_cb;      // pack_cb_of (kNoCb when n/a) — writer-collision input
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
        cb_for_side<Side::SrcA, E>(),
        cb_for_side<Side::SrcB, E>(),
        cb_for_side<Side::Pack, E>(),
        cb_a_of<E>(),
        cb_b_of<E>(),
        pack_cb_of<E>(),
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
    uint32_t seen = NO_PREV_CB;
    for (int i = 0; i < n; ++i) {
        uint32_t cb = d[i].*side;
        if (cb == NO_PREV_CB) continue;
        if (seen == NO_PREV_CB) seen = cb;
        else if (seen != cb) return false;
    }
    return true;
}
constexpr bool ct_reader_collide(const ElemDesc* d, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            if (!(d[i].is_cb_reader && d[j].is_cb_reader)) continue;
            if (!(d[i].is_upfront && d[j].is_upfront)) continue;
            uint32_t a0 = d[i].cb_a, a1 = d[i].cb_b, b0 = d[j].cb_a, b1 = d[j].cb_b;
            if ((a0 != kNoCb && (a0 == b0 || a0 == b1)) || (a1 != kNoCb && (a1 == b0 || a1 == b1)))
                return true;
        }
    return false;
}
constexpr bool ct_writer_collide(const ElemDesc* d, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (d[i].is_pack && d[j].is_pack && d[i].pack_cb == d[j].pack_cb &&
                d[i].pack_dst_slot == d[j].pack_dst_slot)
                return true;
    return false;
}

// Per-side "previous programmed CB at each index" tables, built in ONE forward sweep:
// carry a running prev per side, record it BEFORE each element, update it when the
// element programs that side. prev.srca[I] equals the old back-scan prev_cb_for_idx<SrcA,I>
// (verified byte-identical for all I / all sides) but computed once instead of O(N²).
template <int M>
struct PrevTable {
    uint32_t srca[M];
    uint32_t srcb[M];
    uint32_t pack[M];
};
template <int M>
constexpr PrevTable<M> ct_build_prev(const ElemDesc* d, int n) {
    PrevTable<M> t{};
    uint32_t pa = NO_PREV_CB, pb = NO_PREV_CB, pp = NO_PREV_CB;
    for (int i = 0; i < n; ++i) {
        t.srca[i] = pa;
        t.srcb[i] = pb;
        t.pack[i] = pp;
        if (d[i].srca_cb != NO_PREV_CB) pa = d[i].srca_cb;
        if (d[i].srcb_cb != NO_PREV_CB) pb = d[i].srcb_cb;
        if (d[i].pack_side_cb != NO_PREV_CB) pp = d[i].pack_side_cb;
    }
    return t;
}
// Last opt-in pack CB in chain order (iter-to-iter wraparound prev for pack site 0).
constexpr uint32_t ct_last_pack_cb(const ElemDesc* d, int n) {
    uint32_t last = NO_PREV_CB;
    for (int i = 0; i < n; ++i)
        if (d[i].pack_side_cb != NO_PREV_CB) last = d[i].pack_side_cb;
    return last;
}
// True iff ≥2 opt-in pack sites declare different CBs (boot can't program all).
constexpr bool ct_pack_hetero(const ElemDesc* d, int n) {
    uint32_t first = NO_PREV_CB;
    for (int i = 0; i < n; ++i) {
        if (d[i].pack_side_cb == NO_PREV_CB) continue;
        if (first == NO_PREV_CB) first = d[i].pack_side_cb;
        else if (first != d[i].pack_side_cb) return true;
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

    // Per-side prev-CB history (one sweep), + pack-side metadata. Replaces the O(N²)
    // per-site prev_cb_for_idx and the two standalone pack scans.
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
                         (E::b_policy() == InputLifecycle::Chunked)> {};

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

// chain_has_duplicate_upfront_cbs / chain_pack_writes_collide — pairwise collision
// checks, now flat nested loops in ChainTraits (reader_collide / writer_collide).
template <class... Es> struct chain_has_duplicate_upfront_cbs<EltwiseChain<Es...>>
    : std::bool_constant<detail::ChainTraits<Es...>::reader_collide> {};

template <class... Es> struct chain_pack_writes_collide<EltwiseChain<Es...>>
    : std::bool_constant<detail::ChainTraits<Es...>::writer_collide> {};

// `chain_hoist_math_mop` / `chain_hoist_sfpu` — per-cohort hoist decisions.
// Together they encode the FPU-init hoisting decision tree from
// `ttnn/cpp/ttnn/kernel_lib/docs/fpu_init_hoisting.html` §6, split per
// cohort so the chain can hoist math-MOP init even when SFPU isn't uniform.
// Hoisting means running each element's init() once at boot rather than per
// tile. The two cohorts (math-MOP = ADDR_MOD_0..3 + MATH MOP; SFPU =
// ADDR_MOD_7 + SFPU CSR) are disjoint by hardware design — see
// `llk_math_eltwise_unary_sfpu.h:24-27`, so the two decisions are independent
// in principle. We constrain them to a single-direction implication
// (`chain_hoist_sfpu_v` implies `chain_hoist_math_mop_v`) to keep the
// dispatcher simple — see `eltwise_chain_partial_hoist_proposal.html`.
//
// Eltwise-only scope: matmul / reduce are not modeled as chain elements, so
// gates G2 / G4 / G5 (matmul-specific ADDR_MOD_6 collisions, welford_reinit-
// style UNPACK/MATH reconfig, matmul + binary CLR_DVALID mixing) are
// vacuously satisfied.
//
// Active gates:
//
//   G1. Per-side CB consistency — across all chain elements, every element
//       that programs a side (srcA / srcB) must use the SAME CB id. The
//       boot-time fold programs each side once; if two elements declare
//       different CBs on the same side, the LAST reconfig wins and earlier
//       elements read with the wrong format at per-tile exec time.
//
//   G3. MATH-MOP uniformity — across all `is_math_mop_op_v` elements
//       (`CopyTile`, `BinaryFpu`, `DestReuseBinary`, `UnaryBcast`), all
//       instantiated types must be identical. Each one's init programs
//       MATH MOP / ADDR_MOD_0..3 in a kind-specific way (different FPU
//       ops, different CB args, different bcast modes, the CopyTile vs
//       binary-op init clash from the old `chain_has_non_copy_tile_fpu_clash`
//       check); hoisting more than one type leaves only the last init's
//       MOP programmed and earlier elements run with the wrong MOP.
//
//   SFPU. SFPU-init uniqueness — across all `is_sfpu_op_v` elements, all
//       instantiated types must be identical. This is the regression fix:
//       multiple distinct SFPU `*_tile_init` calls done at boot leave only
//       the LAST MOP programmed. Production failures that drove this gate:
//         - mish_kernel.cpp FP32 path: Exp/Log1p/Tanh chain → tanh saturation,
//           PCC 0.988.
//         - logit_kernel.cpp stage-2: Rsub/DivBinary/Log → ATOL deltas 9–12.
//
// "Identical type" uses `std::is_same_v` per user direction (fpu_init_hoisting
// integration thread). This rejects some chains the doc would consider safe
// (e.g. two `Exp<...>` instances differing only in `Dst::Slot`), but the false
// negatives only cost a per-tile init — never correctness.

namespace detail {

// G1 (per-side CB consistency) is now ChainTraits::srca_consistent / srcb_consistent.

// Trait wrappers (Pred<E>::value form) — `is_sfpu_op_v` / `is_math_mop_op_v`
// are `inline constexpr bool` variable templates; wrap them so they fit the
// `Pred<E>::value` interface used by the uniformity fold.
template <class E>
struct is_sfpu_op_t : std::bool_constant<is_sfpu_op_v<E>> {};
template <class E>
struct is_math_mop_op_t : std::bool_constant<is_math_mop_op_v<E>> {};

// G3 / SFPU helper: across all `Es...` satisfying `Pred<E>`, require every
// instantiated type to be `std::is_same_v` with every other (≤ 1 distinct).
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
struct chain_per_side_cbs_consistent : std::true_type {};

template <class... Es>
struct chain_per_side_cbs_consistent<EltwiseChain<Es...>>
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

// Math-MOP cohort hoist: per-side CB consistency (G1) + math-MOP uniformity (G3).
// True when CopyTile / BinaryFpu / DestReuseBinary / UnaryBcast inits can be
// emitted once at boot instead of per tile. Independent of SFPU uniformity.
template <class Chain>
struct chain_hoist_math_mop : std::false_type {};

template <class... Es>
struct chain_hoist_math_mop<EltwiseChain<Es...>>
    : std::bool_constant<chain_per_side_cbs_consistent_v<EltwiseChain<Es...>> &&
                         chain_math_mop_uniform_v<EltwiseChain<Es...>>> {};

// SFPU cohort hoist: requires math-MOP hoist AND SFPU init uniformity.
// True when every element's init can be emitted at boot. This is the
// fully-hoisted shape (the historical `chain_is_hoist_safe` case).
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
// elements that pre-date the WaitAndPopPerBlock / PerBlockReserveAndPush policies.

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
// emit_pre_element_transitions<E, I, Es...>() (D2)
//
// For element at position I in pack Es..., emit srca / srcb / pack reconfig (each
// compile-time-elided when prev_*_cb == curr_*_cb). Compile-time elision means a
// chain whose elements all share a CB on a side emits the reconfig once (at element
// 0, where prev == NO_PREV_CB) and never again on that side. Run-time cost: zero —
// `if constexpr` resolves at compile time.
//
// Emission shapes:
//   - srca AND srcb both reconfig, both have prev    → reconfig_data_format(prev_a, curr_a, prev_b, curr_b)  (4-arg _with_dt)
//   - srca AND srcb both reconfig, both first-emit   → reconfig_data_format(curr_a, curr_b)                  (2-arg combined)
//   - srca AND srcb both reconfig, mixed prev-state  → independent per-side calls
//   - one side only                                  → reconfig_data_format_src{a,b}(prev, curr) or (curr)
//   - pack                                           → always independent; pack_reconfig_data_format(prev_p, curr_p) or (curr_p)
// The LLK's _with_dt overloads run a format-equality fast-path skip against the CB
// metadata tables, so an emitted reconfig on CBs with matching dtypes is a no-op
// at the hardware level — strictly bounded by a handful of compare instructions.
//
// Pack-side hoist policy (per `docs/pack_reconfig_hoisting_proposal.html` §4.2):
//   - Homogeneous chains (≤1 opt-in pack site, or all sites share a CB): boot
//     emission via `pack_init_for_each`. Subsequent sites fold-elide.
//   - Heterogeneous chains (≥2 opt-in pack sites with different CBs): boot emits
//     only the FIRST opt-in site's reconfig (initializes packer state); later
//     sites' reconfigs are deferred to per-stage emission inside the per-iter
//     pack phase (`emit_per_stage_pack_reconfig`). The 2-arg cache-checked LLK
//     form makes the no-change case ~one compare + branch.
//
// DEST accumulation mode is build-flag-driven (DST_ACCUM_MODE / FP32_DEST_ACC_EN) —
// no per-element fp32 fold here.
// =============================================================================

template <class E, std::size_t I, class... Es>
ALWI void emit_pre_element_transitions() {
    constexpr uint32_t curr_a = cb_for_side<Side::SrcA, E>();
    constexpr uint32_t curr_b = cb_for_side<Side::SrcB, E>();
    constexpr uint32_t curr_p = cb_for_side<Side::Pack, E>();

    constexpr uint32_t prev_a =
        (curr_a != NO_PREV_CB) ? ChainTraits<Es...>::prev.srca[I] : NO_PREV_CB;
    constexpr uint32_t prev_b =
        (curr_b != NO_PREV_CB) ? ChainTraits<Es...>::prev.srcb[I] : NO_PREV_CB;
    constexpr uint32_t prev_p =
        (curr_p != NO_PREV_CB) ? ChainTraits<Es...>::prev.pack[I] : NO_PREV_CB;

    constexpr bool reconf_a = (curr_a != NO_PREV_CB) && (curr_a != prev_a);
    constexpr bool reconf_b = (curr_b != NO_PREV_CB) && (curr_b != prev_b);
    constexpr bool reconf_p = (curr_p != NO_PREV_CB) && (curr_p != prev_p);

    // Pack-side deferral: in heterogeneous chains, only the first opt-in pack
    // site (prev_p == NO_PREV_CB) emits at boot. Later sites defer to per-stage
    // via `emit_per_stage_pack_reconfig`, where the 2-arg LLK form's cache check
    // handles intra-stage transitions cheaply and the per-iter wraparound from
    // last-pack-cb to first-pack-cb is correctly programmed.
    constexpr bool defer_pack_to_per_stage =
        ChainTraits<Es...>::pack_hetero && (prev_p != NO_PREV_CB);

    // ---- srca + srcb: coalesce when both sides share prev-state ----
    if constexpr (reconf_a && reconf_b) {
        if constexpr (prev_a != NO_PREV_CB && prev_b != NO_PREV_CB) {
            // both sides have prev → 4-arg _with_dt
            reconfig_data_format(prev_a, curr_a, prev_b, curr_b);
        } else if constexpr (prev_a == NO_PREV_CB && prev_b == NO_PREV_CB) {
            // first-emit on both sides → 2-arg combined (unconditional reprogram)
            reconfig_data_format(curr_a, curr_b);
        } else {
            // mixed prev-state → independent per-side
            if constexpr (prev_a != NO_PREV_CB) {
                reconfig_data_format_srca(prev_a, curr_a);
            } else {
                reconfig_data_format_srca(curr_a);
            }
            if constexpr (prev_b != NO_PREV_CB) {
                reconfig_data_format_srcb(prev_b, curr_b);
            } else {
                reconfig_data_format_srcb(curr_b);
            }
        }
    } else if constexpr (reconf_a) {
        if constexpr (prev_a != NO_PREV_CB) {
            reconfig_data_format_srca(prev_a, curr_a);
        } else {
            reconfig_data_format_srca(curr_a);
        }
    } else if constexpr (reconf_b) {
        if constexpr (prev_b != NO_PREV_CB) {
            reconfig_data_format_srcb(prev_b, curr_b);
        } else {
            reconfig_data_format_srcb(curr_b);
        }
    }

    // ---- pack: always independent; deferred to per-stage in heterogeneous chains ----
    if constexpr (reconf_p && !defer_pack_to_per_stage) {
        if constexpr (prev_p != NO_PREV_CB) {
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
    constexpr uint32_t curr_p = cb_for_side<Side::Pack, E>();
    if constexpr (curr_p == NO_PREV_CB) return;
    constexpr uint32_t prev_chain = ChainTraits<Es...>::prev.pack[I];
    // Wraparound: first opt-in pack site has no in-chain prev; on iter ≥ 1 the
    // packer ended the previous iter on `last_pack_cb`. The LLK 2-arg form does
    // the right thing on iter 0 too (cache check vs. boot-initialized state).
    constexpr uint32_t prev_p =
        (prev_chain != NO_PREV_CB) ? prev_chain : ChainTraits<Es...>::last_pack_cb;
    if constexpr (prev_p != NO_PREV_CB) {
        pack_reconfig_data_format(prev_p, curr_p);
    }
}

// Pack-phase init (Pack* only — F-PERF-4: hoisted to boot when sound).
// Pack reconfig is fold-driven via `emit_pre_element_transitions`. For
// homogeneous pack chains (≤1 opt-in pack site, or all sites share a CB),
// boot programs the pack engine once and per-stage emission is suppressed.
// For heterogeneous chains (multi-output ops with different pack CBs), boot
// programs only the FIRST opt-in pack site; later sites emit per-stage in
// `apply_pack_phase` via `emit_per_stage_pack_reconfig` using the 2-arg
// cache-checked LLK form. See `docs/pack_reconfig_hoisting_proposal.html` §4.2.
//
// `hoist_compute_init` excludes PackTile from its filtered walk (PACK is its
// own cohort, disjoint from math-MOP and SFPU). `pack_init_for_each` runs the
// boot fold for pack sites; per-stage emission is intentionally separate so
// the heterogeneous case stays correct across per-iter wraparound.
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
// Each element owns its full slice of the outer iteration. The pipeline makes two
// element-iterating calls per outer iter with the commit/wait/release barriers
// between them:
//
//   tile_regs_acquire();
//   apply_compute_phase(...);   // per element: wait + init? + for(j) exec + pop
//   tile_regs_commit();
//   tile_regs_wait();
//   apply_pack_phase(...);      // per pack element: reserve + for(j) pack_exec + push
//   tile_regs_release();
//
// After the outer loop ends, upfront-policy lifecycle fires:
//   elem_pop_upfront_end(...) and elem_push_at_end(...).
//
// pop_per_tile / push_per_tile live INSIDE the apply body — each element owns its
// full lifecycle slice. By the time the LLK exec call returns the unpack-read is
// queued and the framework manages in-flight reads vs producer L1 reuse, so
// cb_pop_front right after exec is safe. push fires right after pack_exec —
// downstream consumer wakes on the new tile while DEST is still held. Both are
// policy-guarded (no-op for upfront / no-pop / no-push policies; those fire after
// the outer loop via elem_pop_upfront_end / elem_push_at_end).
//
// BlockSize == 1 today; commit 7 (auto-block) raises it. exec(i_outer * BlockSize + j)
// passes the absolute tile index — identical to today's exec(i) at BlockSize=1.
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
// D1/D5/D8 caller-init contract:
//   - Caller writes `compute_kernel_hw_startup(cb_a, cb_b, cb_out)` (or its
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
        emit_per_stage_pack_reconfig<ElemT, I, Es...>();
        elem.reserve_per_tile(i_flat);
        elem_reserve_per_block(elem, inner_count);
        elem.reserve_upfront(Ht, Wt);
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
                  "eltwise_chain: two PackTile elements collide on (cb, dst_slot).");

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
// 11c. Public eltwise_chain — strips compile-time-disabled optional elements before
// any stage runs. OptionalChainElement<false, _> carries `is_disabled = true`; we drop
// every such element (type AND instance) from the pack here, then forward the survivors
// to eltwise_chain_impl. "Disabled == absent": the impl, and therefore every stage (the
// invariant static_asserts, hoist decisions, reconfig fold, per-tile loop), only ever
// sees enabled elements. Detection is member-based so the chain needs no knowledge of
// OptionalChainElement (which depends on the chain, not the reverse).
//
// Implementation: tuple-cat the kept elements (each chain_keep yields a 0- or 1-tuple),
// then expand with a direct std::get<I> call into eltwise_chain_impl. We deliberately do
// NOT use std::apply — its INVOKE indirection routes through a callable and can defeat
// `always_inline`, which on a Tensix MATH kernel pushes the compute body out of line and
// miscompiles it. The std::get expansion calls eltwise_chain_impl directly (no closure).
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
// 12. cb_a_of / cb_b_of / pack_cb_of — single-element CB id deducers used by the
// collision-detection static_asserts (chain_has_duplicate_upfront_cbs_v,
// chain_pack_writes_collide_v). Caller-side hw_startup is the caller's
// responsibility — there is no deduced wrapper.
// =============================================================================

namespace detail {

template <class E>
constexpr uint32_t cb_a_of() {
    if constexpr (is_cb_reader_op_v<E>) {
        static_assert(has_cb_a<E>::value,
                      "CbReader element must declare 'static constexpr uint32_t cb_a_id()'");
        return E::cb_a_id();
    } else {
        return kNoCb;
    }
}

template <class E>
constexpr uint32_t cb_b_of() {
    if constexpr (is_binary_fpu_op_v<E> || is_dest_reuse_binary_op_v<E>) {
        static_assert(has_cb_b<E>::value,
                      "Binary CbReader element must declare 'static constexpr uint32_t cb_b_id()'");
        return E::cb_b_id();
    } else {
        return kNoCb;
    }
}

template <class E>
constexpr uint32_t pack_cb_of() {
    if constexpr (is_pack_tile_op_v<E>) {
        static_assert(has_pack_cb<E>::value,
                      "CbWriter element must declare 'static constexpr uint32_t pack_cb_id()'");
        return E::pack_cb_id();
    } else {
        return kNoCb;
    }
}

}  // namespace detail

}  // namespace compute_kernel_lib
