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
// each side of the math/pack pipeline. Streaming and block elements use the same
// uniform accessors so the chain pipeline can compute, at compile time, the most
// recent CB seen on each Side (SrcA / SrcB / Pack) before any given element index.
//
// NO_PREV_CB is the sentinel used by elements that don't touch a side; the fold
// walks Es[0..I-1] backwards and returns the most recent non-NO_PREV_CB.

inline constexpr uint32_t NO_PREV_CB = 0xFFFFFFFFu;

enum class Side : uint8_t { SrcA, SrcB, Pack };

namespace detail {

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
//   static constexpr CopyTilePolicy a_policy();
//   static constexpr CopyTilePolicy b_policy();
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

// =============================================================================
// prev_cb_for_idx<Side, I, Es...>()
//
// Walks Es[0..I-1] backwards, returning the most recent non-NO_PREV_CB on `Side`.
// Implemented as a constexpr fold using std::index_sequence.
// =============================================================================

template <Side S, std::size_t I, class... Es>
constexpr uint32_t prev_cb_for_idx() {
    // Pack into an array; walk indices [0..I) backwards and pick the first non-sentinel.
    if constexpr (I == 0) {
        return NO_PREV_CB;
    } else {
        constexpr uint32_t cbs[] = { cb_for_side<S, Es>()... };
        for (std::size_t k = I; k-- > 0; ) {
            if (cbs[k] != NO_PREV_CB) return cbs[k];
        }
        return NO_PREV_CB;
    }
}

}  // namespace detail

// =============================================================================
// 1. CopyTile chain element
// =============================================================================

template <uint32_t Cb,
          Dst DstSlot,
          CopyTilePolicy Policy,
          CbIndexMode IndexMode,
          CopyTileReconfig Reconfig>
struct CopyTile : CopyTileTag {
    // ---- compile-time validation ----
    static_assert(to_u32(DstSlot) < DEST_AUTO_LIMIT,
                  "CopyTile: DEST slot exceeds DEST_AUTO_LIMIT");
    static_assert(!(Policy == CopyTilePolicy::WaitAndPop  && IndexMode == CbIndexMode::BlockIter),
                  "CopyTile: BlockIter index requires WaitUpfrontPopAtEnd or NoWaitNoPop policy");
    static_assert(!(Policy == CopyTilePolicy::WaitAndPop  && IndexMode == CbIndexMode::Absolute),
                  "CopyTile: Absolute index requires WaitUpfrontPopAtEnd or NoWaitNoPop policy");
    static_assert(!(Policy == CopyTilePolicy::WaitNoPop   && IndexMode == CbIndexMode::BlockIter),
                  "CopyTile: BlockIter index requires WaitUpfrontPopAtEnd or NoWaitNoPop policy");
    static_assert(!(Policy == CopyTilePolicy::WaitNoPop   && IndexMode == CbIndexMode::Absolute),
                  "CopyTile: Absolute index requires WaitUpfrontPopAtEnd or NoWaitNoPop policy");
    static_assert(!(Policy == CopyTilePolicy::NoWaitPop   && IndexMode == CbIndexMode::BlockIter),
                  "CopyTile: BlockIter index requires WaitUpfrontPopAtEnd or NoWaitNoPop policy");
    static_assert(!(Policy == CopyTilePolicy::NoWaitPop   && IndexMode == CbIndexMode::Absolute),
                  "CopyTile: Absolute index requires WaitUpfrontPopAtEnd or NoWaitNoPop policy");

    static constexpr uint32_t       cb              = Cb;
    static constexpr uint32_t       cb_a_id()       { return Cb; }
    static constexpr uint32_t       cb_b_id()       { return 0;  }
    static constexpr Dst            dst_slot        = DstSlot;
    static constexpr CopyTilePolicy a_policy()      { return Policy; }
    static constexpr CopyTilePolicy b_policy()      { return CopyTilePolicy::NoWaitNoPop; }
    static constexpr bool           is_upfront      = (Policy == CopyTilePolicy::WaitUpfrontPopAtEnd);
    static constexpr bool           clashes_with_fpu= true;   // copy_tile uses unpacker MOP

    // Prev-CB fold (D2): CopyTile loads CbA only.
    static constexpr uint32_t       reconfig_srca_cb = (Reconfig == CopyTileReconfig::Input) ? Cb : NO_PREV_CB;
    static constexpr uint32_t       reconfig_srcb_cb = NO_PREV_CB;
    static constexpr uint32_t       reconfig_pack_cb = NO_PREV_CB;

    constexpr CopyTile() noexcept = default;
    constexpr explicit CopyTile(uint32_t cb_tile_idx) noexcept : cb_tile_idx_(cb_tile_idx) {}

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

    /// Wait phase — called once at chain entry per iteration (PerTile policies) or once before
    /// the loop (Upfront policies). The chain pipeline calls the right shape based on `is_upfront`.
    ALWI void wait_per_tile(uint32_t /*i*/) const {
        if constexpr (Policy == CopyTilePolicy::WaitAndPop || Policy == CopyTilePolicy::WaitNoPop) {
            cb_wait_front(Cb, 1);
        }
        // NoWaitPop / NoWaitNoPop / WaitUpfrontPopAtEnd: no per-tile wait.
    }

    ALWI void wait_upfront(uint32_t n) const {
        if constexpr (Policy == CopyTilePolicy::WaitUpfrontPopAtEnd) {
            cb_wait_front(Cb, n);
        }
    }

    ALWI void exec(uint32_t i) const {
        const uint32_t in_idx = [&]() -> uint32_t {
            if constexpr (IndexMode == CbIndexMode::FirstTile) return 0;
            else if constexpr (IndexMode == CbIndexMode::BlockIter) return i;
            else return cb_tile_idx_;  // Pinned / Absolute
        }();
        copy_tile(Cb, in_idx, to_u32(DstSlot));
    }

    ALWI void pop_per_tile(uint32_t /*i*/) const {
        if constexpr (Policy == CopyTilePolicy::WaitAndPop || Policy == CopyTilePolicy::NoWaitPop) {
            cb_pop_front(Cb, 1);
        }
        // WaitNoPop / NoWaitNoPop / WaitUpfrontPopAtEnd: no per-tile pop.
    }

    ALWI void pop_upfront_end(uint32_t n) const {
        if constexpr (Policy == CopyTilePolicy::WaitUpfrontPopAtEnd) {
            cb_pop_front(Cb, n);
        }
    }

private:
    /// Pipeline-driven tile index — set by user via ctor only when IndexMode == Pinned / Absolute.
    /// Not exposed as a public field; pipeline reads through exec().
    uint32_t cb_tile_idx_ = 0;
};

// =============================================================================
// 2. PackTile chain element
// =============================================================================

template <uint32_t Cb,
          Dst DstSlot,
          PackTilePolicy Policy,
          PackTileIndexMode IndexMode,
          PackTileReconfig Reconfig>
struct PackTile : PackTileTag {
    static_assert(to_u32(DstSlot) < DEST_AUTO_LIMIT,
                  "PackTile: DEST slot exceeds DEST_AUTO_LIMIT");
    static_assert(!(Policy == PackTilePolicy::PerTileReserveAndPush && IndexMode == PackTileIndexMode::BlockIter),
                  "PackTile: BlockIter index requires UpfrontReservePushAtEnd or NoReserve* policy");
    static_assert(!(Policy == PackTilePolicy::PerTileReserveAndPush && IndexMode == PackTileIndexMode::Absolute),
                  "PackTile: Absolute index requires UpfrontReservePushAtEnd or NoReserve* policy");
    static_assert(!(Policy == PackTilePolicy::PerTileReserveNoPush  && IndexMode == PackTileIndexMode::BlockIter),
                  "PackTile: BlockIter index requires UpfrontReservePushAtEnd or NoReserve* policy");
    static_assert(!(Policy == PackTilePolicy::NoReservePushAtEnd    && IndexMode == PackTileIndexMode::BlockIter),
                  "PackTile: BlockIter requires Upfront* / NoReserveNoPush");

    static constexpr uint32_t          cb                  = Cb;
    static constexpr uint32_t          pack_cb_id()        { return Cb; }
    static constexpr Dst               pack_dst_slot       = DstSlot;
    static constexpr bool              is_upfront          = (Policy == PackTilePolicy::UpfrontReservePushAtEnd);
    static constexpr PackTileIndexMode index_mode          = IndexMode;

    // Prev-CB fold (D2): PackTile writes pack-side; mark Cb under reconfig only when
    // the user opted into pack reconfig (Output / OutputConditional). Otherwise no
    // pack reconfig is emitted — fold keeps prior pack target.
    static constexpr uint32_t          reconfig_srca_cb    = NO_PREV_CB;
    static constexpr uint32_t          reconfig_srcb_cb    = NO_PREV_CB;
    static constexpr uint32_t          reconfig_pack_cb    =
        (Reconfig == PackTileReconfig::Output || Reconfig == PackTileReconfig::OutputConditional) ? Cb : NO_PREV_CB;

    constexpr PackTile() noexcept = default;
    constexpr explicit PackTile(uint32_t output_tile_idx) noexcept : output_tile_idx_(output_tile_idx) {}

    static ALWI void init() {
        // Pack reconfig is fold-driven (compile-time-elided when prev_pack_cb == Cb).
        // The chain emits the reconfig in `emit_pre_element_transitions()` before this
        // element runs; init() here is a no-op for reconfig.
        // Retained empty so trait-dispatch stays uniform.
    }

    ALWI void reserve_per_tile(uint32_t /*i*/) const {
        if constexpr (Policy == PackTilePolicy::PerTileReserveAndPush ||
                      Policy == PackTilePolicy::PerTileReserveNoPush) {
            cb_reserve_back(Cb, 1);
        }
    }

    ALWI void reserve_upfront(uint32_t n) const {
        if constexpr (Policy == PackTilePolicy::UpfrontReservePushAtEnd) {
            cb_reserve_back(Cb, n);
        }
    }

    ALWI void exec(uint32_t i) const {
        const uint32_t out_idx = [&]() -> uint32_t {
            if constexpr (IndexMode == PackTileIndexMode::FirstTile) return 0;
            else if constexpr (IndexMode == PackTileIndexMode::BlockIter) return i;
            else return output_tile_idx_;  // Pinned / Absolute
        }();
        pack_tile(to_u32(DstSlot), Cb, out_idx);
    }

    ALWI void push_per_tile(uint32_t /*i*/) const {
        if constexpr (Policy == PackTilePolicy::PerTileReserveAndPush) {
            cb_push_back(Cb, 1);
        }
    }

    ALWI void push_at_end(uint32_t n) const {
        if constexpr (Policy == PackTilePolicy::NoReservePushAtEnd ||
                      Policy == PackTilePolicy::UpfrontReservePushAtEnd) {
            cb_push_back(Cb, n);
        }
    }

private:
    /// Pipeline-driven runtime output-tile index for Pinned / Absolute modes — ctor-only.
    uint32_t output_tile_idx_ = 0;
};

// =============================================================================
// 3. PackTileBlock — atomic multi-slot pack
// =============================================================================

template <uint32_t Cb,
          Dst FirstSlot,
          uint32_t NTiles,
          PackTilePolicy Policy,
          PackTileReconfig Reconfig>
struct PackTileBlock : PackTileTag {
    static_assert(NTiles >= 1 && NTiles <= DEST_AUTO_LIMIT,
                  "PackTileBlock: NTiles must be in [1, DEST_AUTO_LIMIT]");
    static_assert(to_u32(FirstSlot) + NTiles <= DEST_AUTO_LIMIT,
                  "PackTileBlock: FirstSlot + NTiles exceeds DEST_AUTO_LIMIT (consecutive slots required)");

    static constexpr uint32_t cb           = Cb;
    static constexpr uint32_t pack_cb_id() { return Cb; }
    static constexpr Dst      pack_dst_slot = FirstSlot;
    static constexpr uint32_t n_tiles      = NTiles;
    static constexpr bool     is_upfront   = (Policy == PackTilePolicy::UpfrontReservePushAtEnd);

    // Prev-CB fold (D2): PackTileBlock writes pack-side.
    static constexpr uint32_t reconfig_srca_cb = NO_PREV_CB;
    static constexpr uint32_t reconfig_srcb_cb = NO_PREV_CB;
    static constexpr uint32_t reconfig_pack_cb =
        (Reconfig == PackTileReconfig::Output || Reconfig == PackTileReconfig::OutputConditional) ? Cb : NO_PREV_CB;

    static ALWI void init() {
        // Pack reconfig is fold-driven; init() is a no-op.
    }

    ALWI void reserve_per_tile(uint32_t /*i*/) const {
        if constexpr (Policy == PackTilePolicy::PerTileReserveAndPush ||
                      Policy == PackTilePolicy::PerTileReserveNoPush) {
            cb_reserve_back(Cb, NTiles);
        }
    }
    ALWI void reserve_upfront(uint32_t n) const {
        if constexpr (Policy == PackTilePolicy::UpfrontReservePushAtEnd) {
            cb_reserve_back(Cb, n * NTiles);
        }
    }
    ALWI void exec(uint32_t /*i*/) const {
        pack_tile_block(to_u32(FirstSlot), Cb, NTiles);
    }
    ALWI void push_per_tile(uint32_t /*i*/) const {
        if constexpr (Policy == PackTilePolicy::PerTileReserveAndPush) {
            cb_push_back(Cb, NTiles);
        }
    }
    ALWI void push_at_end(uint32_t n) const {
        if constexpr (Policy == PackTilePolicy::NoReservePushAtEnd ||
                      Policy == PackTilePolicy::UpfrontReservePushAtEnd) {
            cb_push_back(Cb, n * NTiles);
        }
    }
};

// =============================================================================
// 4. BinaryFpu chain element
// =============================================================================

template <uint32_t CbA,
          uint32_t CbB,
          uint32_t CbOut,
          BinaryFpuOp Op,
          BroadcastDim Bcast,
          BinaryDataFormatReconfig DfReconfig,
          CopyTilePolicy APolicy,
          CopyTilePolicy BPolicy,
          CbIndexMode AIndex,
          Dst DstSlot,
          CbIndexMode BIndex>
struct BinaryFpu : BinaryFpuTag {
    static_assert(to_u32(DstSlot) < DEST_AUTO_LIMIT,
                  "BinaryFpu: DEST slot exceeds DEST_AUTO_LIMIT");
    // Per-side BlockIter requires Upfront / NoWait* on that side (caller pre-pushes the
    // walked range; per-tile wait + pop would not advance the walked index).
    static_assert(!(APolicy == CopyTilePolicy::WaitAndPop && AIndex == CbIndexMode::BlockIter),
                  "BinaryFpu: AIndex=BlockIter requires APolicy in {WaitUpfrontPopAtEnd, WaitNoPop, NoWaitPop, NoWaitNoPop}");
    static_assert(!(BPolicy == CopyTilePolicy::WaitAndPop && BIndex == CbIndexMode::BlockIter),
                  "BinaryFpu: BIndex=BlockIter requires BPolicy in {WaitUpfrontPopAtEnd, WaitNoPop, NoWaitPop, NoWaitNoPop}");
    // same_cb dedup safety: when CbA == CbB the B-side wait/pop is skipped, so the
    // helper would under-wait if A and B walked different ranges of the shared CB.
    static_assert((CbA != CbB) || AIndex == BIndex,
                  "BinaryFpu: when CbA == CbB, AIndex and BIndex must match "
                  "(B-side wait/pop is deduped — asymmetric indices would under-wait).");

    static constexpr uint32_t      cb_a_id()  { return CbA; }
    static constexpr uint32_t      cb_b_id()  { return CbB; }
    static constexpr CopyTilePolicy a_policy(){ return APolicy; }
    static constexpr CopyTilePolicy b_policy(){ return BPolicy; }
    static constexpr Dst           dst_slot   = DstSlot;
    static constexpr bool          is_upfront = (APolicy == CopyTilePolicy::WaitUpfrontPopAtEnd) ||
                                                (BPolicy == CopyTilePolicy::WaitUpfrontPopAtEnd);
    static constexpr bool          clashes_with_fpu = true;
    static constexpr bool          same_cb    = (CbA == CbB);

    // Prev-CB fold (D2): BinaryFpu touches both srca (CbA), srcb (CbB), and pack (CbOut)
    // when the corresponding reconfig is opted in. F-PERF-3 strips the per-element pack
    // reconfig from init(); the chain's compile-time-elided fold drives both input-side
    // and output-side reconfig before this element runs.
    static constexpr uint32_t      reconfig_srca_cb =
        (DfReconfig == BinaryDataFormatReconfig::Input || DfReconfig == BinaryDataFormatReconfig::InputAndOutput)
            ? CbA : NO_PREV_CB;
    static constexpr uint32_t      reconfig_srcb_cb =
        (DfReconfig == BinaryDataFormatReconfig::Input || DfReconfig == BinaryDataFormatReconfig::InputAndOutput)
            ? CbB : NO_PREV_CB;
    static constexpr uint32_t      reconfig_pack_cb =
        ((DfReconfig == BinaryDataFormatReconfig::Output || DfReconfig == BinaryDataFormatReconfig::InputAndOutput)
         && CbOut != 0) ? CbOut : NO_PREV_CB;

    constexpr BinaryFpu() noexcept = default;
    constexpr BinaryFpu(uint32_t a_tile_idx, uint32_t b_tile_idx) noexcept
        : a_tile_idx_(a_tile_idx), b_tile_idx_(b_tile_idx) {}

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
                MATH((llk_math_eltwise_binary_init_with_operands<et, bt, MATH_FIDELITY>(CbA, CbB)));
            } else {
                MATH((llk_math_eltwise_binary_init_with_operands<et, bt, MathFidelity::LoFi>(CbA, CbB)));
            }
            UNPACK((llk_unpack_AB_init<bt>(CbA, CbB)));
        }
    }

    // ---- CB lifecycle (per-tile) ----
    ALWI void wait_per_tile(uint32_t /*i*/) const {
        if constexpr (APolicy == CopyTilePolicy::WaitAndPop || APolicy == CopyTilePolicy::WaitNoPop) {
            cb_wait_front(CbA, 1);
        }
        if constexpr (!same_cb && (BPolicy == CopyTilePolicy::WaitAndPop || BPolicy == CopyTilePolicy::WaitNoPop)) {
            cb_wait_front(CbB, 1);
        }
        // same_cb: skip second wait — dedup.
    }

    ALWI void wait_upfront(uint32_t n) const {
        if constexpr (APolicy == CopyTilePolicy::WaitUpfrontPopAtEnd) cb_wait_front(CbA, n);
        if constexpr (!same_cb && BPolicy == CopyTilePolicy::WaitUpfrontPopAtEnd) cb_wait_front(CbB, n);
    }

    ALWI void exec(uint32_t i) const {
        // Per-side index mode. AIndex drives a_idx, BIndex drives b_idx. The
        // canonical bcast walk is A=BlockIter (walks the tile range) + B=FirstTile
        // (pins the scaler/vector operand at tile 0).
        const uint32_t a_idx = [&]() -> uint32_t {
            if constexpr      (AIndex == CbIndexMode::FirstTile)  return 0;
            else if constexpr (AIndex == CbIndexMode::BlockIter)  return i;
            else                                                  return a_tile_idx_;  // Pinned / Absolute
        }();
        const uint32_t b_idx = [&]() -> uint32_t {
            if constexpr      (BIndex == CbIndexMode::FirstTile)  return 0;
            else if constexpr (BIndex == CbIndexMode::BlockIter)  return i;
            else                                                  return b_tile_idx_;  // Pinned / Absolute
        }();
        if constexpr (Bcast == BroadcastDim::None) {
            if constexpr      (Op == BinaryFpuOp::Add) add_tiles(CbA, CbB, a_idx, b_idx, to_u32(DstSlot));
            else if constexpr (Op == BinaryFpuOp::Sub) sub_tiles(CbA, CbB, a_idx, b_idx, to_u32(DstSlot));
            else                                       mul_tiles(CbA, CbB, a_idx, b_idx, to_u32(DstSlot));
        } else {
            // Broadcast variants via the generic `add/sub/mul_tiles_bcast<BroadcastType>` template
            // — these forward to `any_tiles_bcast<EltwiseBinaryType, BroadcastType>` internally.
            constexpr auto bt = static_cast<ckernel::BroadcastType>(static_cast<uint8_t>(Bcast));
            if constexpr      (Op == BinaryFpuOp::Add) add_tiles_bcast<bt>(CbA, CbB, a_idx, b_idx, to_u32(DstSlot));
            else if constexpr (Op == BinaryFpuOp::Sub) sub_tiles_bcast<bt>(CbA, CbB, a_idx, b_idx, to_u32(DstSlot));
            else                                       mul_tiles_bcast<bt>(CbA, CbB, a_idx, b_idx, to_u32(DstSlot));
        }
    }

    ALWI void pop_per_tile(uint32_t /*i*/) const {
        if constexpr (APolicy == CopyTilePolicy::WaitAndPop || APolicy == CopyTilePolicy::NoWaitPop) {
            cb_pop_front(CbA, 1);
        }
        if constexpr (!same_cb && (BPolicy == CopyTilePolicy::WaitAndPop || BPolicy == CopyTilePolicy::NoWaitPop)) {
            cb_pop_front(CbB, 1);
        }
    }

    ALWI void pop_upfront_end(uint32_t n) const {
        if constexpr (APolicy == CopyTilePolicy::WaitUpfrontPopAtEnd) cb_pop_front(CbA, n);
        if constexpr (!same_cb && BPolicy == CopyTilePolicy::WaitUpfrontPopAtEnd) cb_pop_front(CbB, n);
    }

private:
    /// Pipeline-driven runtime indices for Pinned / Absolute modes — ctor-only.
    uint32_t a_tile_idx_ = 0;
    uint32_t b_tile_idx_ = 0;
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
          CopyTilePolicy Policy,
          CbIndexMode IndexMode>
struct DestReuseBinary : DestReuseBinaryTag {
    static_assert(to_u32(DstIn) < DEST_AUTO_LIMIT && to_u32(DstOut) < DEST_AUTO_LIMIT,
                  "DestReuseBinary: DEST slot exceeds DEST_AUTO_LIMIT");
    static_assert(!(Policy == CopyTilePolicy::WaitAndPop && IndexMode == CbIndexMode::BlockIter),
                  "DestReuseBinary: BlockIter index requires Upfront / NoWaitNoPop policy");

    static constexpr uint32_t       cb_a_id()         { return Cb; }
    static constexpr uint32_t       cb_b_id()         { return 0;  }
    static constexpr CopyTilePolicy a_policy()        { return Policy; }
    static constexpr CopyTilePolicy b_policy()        { return CopyTilePolicy::NoWaitNoPop; }
    static constexpr Dst            dst_slot          = DstOut;
    static constexpr bool           is_upfront        = (Policy == CopyTilePolicy::WaitUpfrontPopAtEnd);
    static constexpr bool           clashes_with_fpu  = true;

    // Prev-CB fold (D2): DestReuseBinary loads CB into srca (when DEST → srcb) or srcb
    // (when DEST → srca). Reconfig only fires when opted in.
    static constexpr uint32_t       reconfig_srca_cb  =
        (Reconfig == DestReuseReconfig::Input && ReuseType == DestReuseType::DEST_TO_SRCB) ? Cb : NO_PREV_CB;
    static constexpr uint32_t       reconfig_srcb_cb  =
        (Reconfig == DestReuseReconfig::Input && ReuseType == DestReuseType::DEST_TO_SRCA) ? Cb : NO_PREV_CB;
    static constexpr uint32_t       reconfig_pack_cb  = NO_PREV_CB;

    constexpr DestReuseBinary() noexcept = default;
    constexpr explicit DestReuseBinary(uint32_t cb_tile_idx) noexcept : cb_tile_idx_(cb_tile_idx) {}

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

    ALWI void wait_per_tile(uint32_t /*i*/) const {
        if constexpr (Policy == CopyTilePolicy::WaitAndPop || Policy == CopyTilePolicy::WaitNoPop) {
            cb_wait_front(Cb, 1);
        }
    }
    ALWI void wait_upfront(uint32_t n) const {
        if constexpr (Policy == CopyTilePolicy::WaitUpfrontPopAtEnd) cb_wait_front(Cb, n);
    }
    ALWI void exec(uint32_t i) const {
        constexpr auto et = (Op == BinaryFpuOp::Add) ? ckernel::EltwiseBinaryType::ELWADD :
                            (Op == BinaryFpuOp::Sub) ? ckernel::EltwiseBinaryType::ELWSUB :
                                                       ckernel::EltwiseBinaryType::ELWMUL;
        constexpr auto reuse = (ReuseType == DestReuseType::DEST_TO_SRCA)
                                   ? ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCA
                                   : ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCB;
        const uint32_t in_idx = [&]() -> uint32_t {
            if constexpr (IndexMode == CbIndexMode::FirstTile) return 0;
            else if constexpr (IndexMode == CbIndexMode::BlockIter) return i;
            else return cb_tile_idx_;
        }();
        binary_dest_reuse_tiles<et, reuse>(Cb, in_idx, to_u32(DstIn));
    }
    ALWI void pop_per_tile(uint32_t /*i*/) const {
        if constexpr (Policy == CopyTilePolicy::WaitAndPop || Policy == CopyTilePolicy::NoWaitPop) {
            cb_pop_front(Cb, 1);
        }
    }
    ALWI void pop_upfront_end(uint32_t n) const {
        if constexpr (Policy == CopyTilePolicy::WaitUpfrontPopAtEnd) cb_pop_front(Cb, n);
    }

private:
    /// Pipeline-driven runtime index for Pinned / Absolute modes — ctor-only.
    uint32_t cb_tile_idx_ = 0;
};

// =============================================================================
// 6. UnaryBcast chain element
// =============================================================================

template <BroadcastDim Dim,
          uint32_t Cb,
          uint32_t CbOut,
          Dst DstSlot,
          CopyTilePolicy Policy,
          UnaryBcastReconfig Reconfig>
struct UnaryBcast : UnaryBcastTag {
    static_assert(to_u32(DstSlot) < DEST_AUTO_LIMIT,
                  "UnaryBcast: DEST slot exceeds DEST_AUTO_LIMIT");

    static constexpr uint32_t       cb_a_id()         { return Cb; }
    static constexpr uint32_t       cb_b_id()         { return 0;  }
    static constexpr CopyTilePolicy a_policy()        { return Policy; }
    static constexpr CopyTilePolicy b_policy()        { return CopyTilePolicy::NoWaitNoPop; }
    static constexpr Dst            dst_slot          = DstSlot;
    static constexpr bool           is_upfront        = (Policy == CopyTilePolicy::WaitUpfrontPopAtEnd);
    static constexpr bool           clashes_with_fpu  = true;

    // Prev-CB fold (D2): UnaryBcast loads Cb to srca; ocb (CbOut or Cb) on pack-side.
    // Reconfig is currently bundled into `unary_bcast_init` so no separate fold-driven
    // reconfig fires here; sentinels remain NO_PREV_CB to keep the fold transparent.
    static constexpr uint32_t       reconfig_srca_cb  = NO_PREV_CB;
    static constexpr uint32_t       reconfig_srcb_cb  = NO_PREV_CB;
    static constexpr uint32_t       reconfig_pack_cb  = NO_PREV_CB;

    static ALWI void init() {
        constexpr auto bt = static_cast<ckernel::BroadcastType>(static_cast<uint8_t>(Dim));
        constexpr uint32_t ocb = (CbOut != 0) ? CbOut : Cb;
        // Reconfig path: single-arg-style — `unary_bcast_init` already does srca + ocb reconfig
        // for the new bcast dim. No previous-CB tracking needed.
        unary_bcast_init<bt>(Cb, ocb);
    }

    ALWI void wait_per_tile(uint32_t /*i*/) const {
        if constexpr (Policy == CopyTilePolicy::WaitAndPop || Policy == CopyTilePolicy::WaitNoPop) {
            cb_wait_front(Cb, 1);
        }
    }
    ALWI void wait_upfront(uint32_t n) const {
        if constexpr (Policy == CopyTilePolicy::WaitUpfrontPopAtEnd) cb_wait_front(Cb, n);
    }
    ALWI void exec(uint32_t /*i*/) const {
        constexpr auto bt = static_cast<ckernel::BroadcastType>(static_cast<uint8_t>(Dim));
        unary_bcast<bt>(Cb, /*in_tile_index=*/0, to_u32(DstSlot));
    }
    ALWI void pop_per_tile(uint32_t /*i*/) const {
        if constexpr (Policy == CopyTilePolicy::WaitAndPop || Policy == CopyTilePolicy::NoWaitPop) {
            cb_pop_front(Cb, 1);
        }
    }
    ALWI void pop_upfront_end(uint32_t n) const {
        if constexpr (Policy == CopyTilePolicy::WaitUpfrontPopAtEnd) cb_pop_front(Cb, n);
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

template <class... Es> struct chain_has_any_copy_tile<EltwiseChain<Es...>>
    : std::bool_constant<(is_copy_tile_op_v<Es> || ...)> {};

template <class... Es> struct chain_has_any_pack_tile<EltwiseChain<Es...>>
    : std::bool_constant<(is_pack_tile_op_v<Es> || ...)> {};

template <class... Es> struct chain_has_any_cb_reader<EltwiseChain<Es...>>
    : std::bool_constant<(is_cb_reader_op_v<Es> || ...)> {};

template <class... Es> struct chain_has_any_cb_writer<EltwiseChain<Es...>>
    : std::bool_constant<(is_cb_writer_op_v<Es> || ...)> {};

template <class... Es> struct chain_has_non_copy_tile_fpu_clash<EltwiseChain<Es...>>
    : std::bool_constant<((is_binary_fpu_op_v<Es>        ||
                           is_dest_reuse_binary_op_v<Es> ||
                           is_unary_bcast_op_v<Es>) || ...)> {};

// chain_loads_share_cb — N-element fold (item 7). True when every CopyTile element in
// the chain reads from the same CB (or chain has ≤1 CopyTile, in which case the
// constraint is vacuously satisfied). Drives the hoist gate together with
// chain_has_non_copy_tile_fpu_clash: hoisting init() is safe only when no element
// reprograms hardware state another element relies on.

namespace detail {
template <class E>
constexpr uint32_t copy_tile_cb_of() {
    if constexpr (is_copy_tile_op_v<E>) return E::cb;
    else                                 return NO_PREV_CB;
}

template <class... Es>
constexpr bool copy_tiles_share_cb_v = []() {
    if constexpr (sizeof...(Es) == 0) {
        return true;
    } else {
        constexpr uint32_t cbs[] = {copy_tile_cb_of<Es>()...};
        uint32_t seen = NO_PREV_CB;
        for (auto cb : cbs) {
            if (cb == NO_PREV_CB) continue;
            if (seen == NO_PREV_CB) seen = cb;
            else if (seen != cb) return false;
        }
        return true;
    }
}();
}  // namespace detail

template <class Chain>
struct chain_loads_share_cb : std::true_type {};  // default (empty chain — vacuously true)

template <class... Es>
struct chain_loads_share_cb<EltwiseChain<Es...>>
    : std::bool_constant<detail::copy_tiles_share_cb_v<Es...>> {};

// chain_has_duplicate_upfront_cbs / chain_pack_writes_collide:
// defined as a runtime fold for now — every CB-reader / CB-writer pair is checked.
// Static assertions in the chain pipeline use these as constexpr-evaluated booleans.

namespace detail {

template <class A, class B>
constexpr bool reader_pair_collide() {
    if constexpr (!is_cb_reader_op_v<A> || !is_cb_reader_op_v<B>) return false;
    else if constexpr (!A::is_upfront || !B::is_upfront)          return false;
    else {
        constexpr uint32_t a0 = cb_a_of<A>();
        constexpr uint32_t a1 = cb_b_of<A>();
        constexpr uint32_t b0 = cb_a_of<B>();
        constexpr uint32_t b1 = cb_b_of<B>();
        return (a0 != 0 && (a0 == b0 || a0 == b1)) ||
               (a1 != 0 && (a1 == b0 || a1 == b1));
    }
}

template <class A, class B>
constexpr bool writer_pair_collide() {
    if constexpr (!is_pack_tile_op_v<A> || !is_pack_tile_op_v<B>) return false;
    else                                                          return (pack_cb_of<A>() == pack_cb_of<B>()) &&
                                                                         (A::pack_dst_slot == B::pack_dst_slot);
}

template <class... Es>
struct any_reader_dup;

template <>
struct any_reader_dup<> : std::false_type {};

template <class E0, class... Rest>
struct any_reader_dup<E0, Rest...>
    : std::bool_constant<((reader_pair_collide<E0, Rest>() || ...) || any_reader_dup<Rest...>::value)> {};

template <class... Es>
struct any_writer_dup;

template <>
struct any_writer_dup<> : std::false_type {};

template <class E0, class... Rest>
struct any_writer_dup<E0, Rest...>
    : std::bool_constant<((writer_pair_collide<E0, Rest>() || ...) || any_writer_dup<Rest...>::value)> {};

}  // namespace detail

template <class... Es> struct chain_has_duplicate_upfront_cbs<EltwiseChain<Es...>>
    : detail::any_reader_dup<Es...> {};

template <class... Es> struct chain_pack_writes_collide<EltwiseChain<Es...>>
    : detail::any_writer_dup<Es...> {};

// `chain_is_hoist_safe` — generic N-element fold (item 7 + lessons §3.4).
// Hoist init() out of the per-tile loop iff:
//   1. No element with `clashes_with_fpu == true` outside the CopyTile family
//      (BinaryFpu / DestReuseBinary / UnaryBcast reprogram unpack MOP per iter).
//   2. All CopyTile elements share a single srca CB (or ≤1 CopyTile in chain).
// SFPU ops program SFPU/SFPI state, not unpack MOP — they are always hoist-safe
// as a group; the chain-shape gate covers the elements that DO touch unpack.
template <class Chain>
struct chain_is_hoist_safe : std::false_type {};

template <class... Es>
struct chain_is_hoist_safe<EltwiseChain<Es...>>
    : std::bool_constant<!chain_has_non_copy_tile_fpu_clash_v<EltwiseChain<Es...>> &&
                         chain_loads_share_cb_v<EltwiseChain<Es...>>> {};

// =============================================================================
// 10. Chain pipeline — per-iteration emit
// =============================================================================

namespace detail {

// Per-element wait dispatch (only CB-readers do anything).
template <class E>
ALWI void elem_wait_per_tile(const E& e, uint32_t i) {
    if constexpr (is_cb_reader_op_v<E>) e.wait_per_tile(i);
}
template <class E>
ALWI void elem_wait_upfront(const E& e, uint32_t n) {
    if constexpr (is_cb_reader_op_v<E>) e.wait_upfront(n);
}
template <class E>
ALWI void elem_pop_per_tile(const E& e, uint32_t i) {
    if constexpr (is_cb_reader_op_v<E>) e.pop_per_tile(i);
}
template <class E>
ALWI void elem_pop_upfront_end(const E& e, uint32_t n) {
    if constexpr (is_cb_reader_op_v<E>) e.pop_upfront_end(n);
}

// Per-element pack dispatch.
template <class E>
ALWI void elem_reserve_per_tile(const E& e, uint32_t i) {
    if constexpr (is_cb_writer_op_v<E>) e.reserve_per_tile(i);
}
template <class E>
ALWI void elem_reserve_upfront(const E& e, uint32_t n) {
    if constexpr (is_cb_writer_op_v<E>) e.reserve_upfront(n);
}
template <class E>
ALWI void elem_push_per_tile(const E& e, uint32_t i) {
    if constexpr (is_cb_writer_op_v<E>) e.push_per_tile(i);
}
template <class E>
ALWI void elem_push_at_end(const E& e, uint32_t n) {
    if constexpr (is_cb_writer_op_v<E>) e.push_at_end(n);
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
// DEST accumulation mode is build-flag-driven (DST_ACCUM_MODE / FP32_DEST_ACC_EN) —
// no per-element fp32 fold here.
// =============================================================================

template <class E, std::size_t I, class... Es>
ALWI void emit_pre_element_transitions() {
    // ---- D2 prev-CB elision ----
    constexpr uint32_t curr_a = cb_for_side<Side::SrcA, E>();
    if constexpr (curr_a != NO_PREV_CB) {
        constexpr uint32_t prev_a = prev_cb_for_idx<Side::SrcA, I, Es...>();
        if constexpr (curr_a != prev_a) {
            reconfig_data_format_srca(curr_a);
        }
    }

    constexpr uint32_t curr_b = cb_for_side<Side::SrcB, E>();
    if constexpr (curr_b != NO_PREV_CB) {
        constexpr uint32_t prev_b = prev_cb_for_idx<Side::SrcB, I, Es...>();
        if constexpr (curr_b != prev_b) {
            reconfig_data_format_srcb(curr_b);
        }
    }

    constexpr uint32_t curr_p = cb_for_side<Side::Pack, E>();
    if constexpr (curr_p != NO_PREV_CB) {
        constexpr uint32_t prev_p = prev_cb_for_idx<Side::Pack, I, Es...>();
        if constexpr (curr_p != prev_p) {
            pack_reconfig_data_format(curr_p);
        }
    }
}

// Compute-phase exec (everything except Pack*).
// Runs between `tile_regs_acquire()` and `tile_regs_commit()`.
// For FPU-clash patterns each element's init() runs immediately before its exec —
// this matches production kernels (`copy_tile_init; copy_tile; sfpu_init; sfpu_tile;
// binary_dest_reuse_init; binary_dest_reuse_tiles`) and avoids state-clobbering when
// multiple chain elements reconfigure the unpacker.
//
// Single dispatch contract (§4.5): every element exposes `void exec(uint32_t) const`.
// CRTP bases (UnaryOp / BinaryOp / TernaryOp / QuaternaryOp) forward to `exec_impl()`;
// runtime-param ops override `exec(uint32_t)` directly. No static-vs-member fork.
//
// `EmitInit`: when true, the per-tile path re-fires `E::init()` before exec (clash
// chains). When false, the chain hoisted init() to boot — per-tile path skips it.
template <bool EmitInit, class E>
ALWI void elem_compute_exec(const E& e, uint32_t i) {
    if constexpr (is_pack_tile_op_v<E>) {
        // Pack runs in the pack phase, not compute. Skip here.
        (void)e; (void)i;
    } else {
        if constexpr (EmitInit) E::init();
        e.exec(i);
    }
}

// Pack-phase exec (Pack* only).
// Runs between `tile_regs_wait()` and `tile_regs_release()`.
template <class E>
ALWI void elem_pack_exec(const E& e, uint32_t i) {
    if constexpr (is_pack_tile_op_v<E>) e.exec(i);
}

// Pack-phase init (Pack* only — F-PERF-4: hoisted to boot, not per-tile).
// Note: post-commit-2 the pack reconfig is fold-driven via `emit_pre_element_transitions`,
// so this is effectively a no-op for PackTile / PackTileBlock. Retained for symmetry
// in case a pack element gains per-op LLK programming in a future commit.
template <class E>
ALWI void elem_pack_init() {
    if constexpr (is_pack_tile_op_v<E>) E::init();
}

// Index-list expander: runs `emit_pre_element_transitions<E_I, I, Es...>()` then
// `elem_compute_exec<EmitInit>(elts[I], i)` for each I in [0, sizeof...(Es)).
//
// `EmitInit` and `EmitTransitions` are independent gates:
//   - EmitTransitions=true: per-tile pre-element reconfig + fp32 transitions
//     (used by FPU-clash chains where the FPU op overwrites unpacker state)
//   - EmitTransitions=false: transitions already hoisted to boot — skip
//   - EmitInit=true: per-tile init() (FPU-clash chains)
//   - EmitInit=false: init() already hoisted to boot — skip
template <bool EmitInit, bool EmitTransitions, std::size_t... Is, class... Es>
ALWI void compute_phase_for_each(std::index_sequence<Is...>, uint32_t i, Es&... elts) {
    auto run_one = [&](auto idx, auto& elem) {
        constexpr std::size_t II = decltype(idx)::value;
        using ElemT = std::remove_reference_t<decltype(elem)>;
        if constexpr (EmitTransitions) {
            emit_pre_element_transitions<ElemT, II, Es...>();
        }
        elem_compute_exec<EmitInit>(elem, i);
    };
    (run_one(std::integral_constant<std::size_t, Is>{}, elts), ...);
}

// Hoisted init+transitions for non-clash chains (F-PERF-1).
// Boot-time emission: per-element pre-element transitions + per-element static init().
template <std::size_t... Is, class... Es>
ALWI void hoisted_init_for_each(std::index_sequence<Is...>, Es&... elts) {
    auto run_one = [&](auto idx, auto& elem) {
        constexpr std::size_t II = decltype(idx)::value;
        using ElemT = std::remove_reference_t<decltype(elem)>;
        // FIX (Reg C): previously skipped Pack elements, but emit_pre_element_transitions
        // is the only emission path for pack_reconfig_data_format declared by
        // PackTile<...PackTileReconfig::Output>. Skipping PackTile here means non-clash
        // chains never emit pack reconfig, leaving stale pack format from the previous
        // chain. Includes PackTile now — its emit_pre_element_transitions fires pack
        // reconfig; PackTile::init() is empty (no-op) so other side-effects are safe.
        emit_pre_element_transitions<ElemT, II, Es...>();
        ElemT::init();
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

template <class... Es>
ALWI void eltwise_chain(uint32_t n_tiles, Es... elts) {
    using Chain = EltwiseChain<Es...>;

    // ---- Compile-time invariant checks ----
    static_assert(!chain_has_duplicate_upfront_cbs_v<Chain>,
                  "eltwise_chain: two CB-reader elements share a CB on upfront-wait policy. "
                  "Each upfront-wait CB must appear in exactly one element.");
    static_assert(!chain_pack_writes_collide_v<Chain>,
                  "eltwise_chain: two PackTile elements collide on (cb, dst_slot). "
                  "Pack writes must target distinct (cb, dst) tuples.");

    // ---- Detect upfront-block path (auto-detected via per-element `is_upfront`) ----
    constexpr bool block_path = ((Es::is_upfront) || ...);

    // ---- Hoist gate (item 7 + lessons §3.4) ----
    //
    // Hoist all per-element init() to chain entry only when both:
    //   - No FPU-clash element (BinaryFpu / DestReuseBinary / UnaryBcast reprogram
    //     unpack MOP each iter — must reinit).
    //   - All CopyTile elements share one CB (multi-CB CopyTile chains need per-iter
    //     reinit because each CopyTile's hoisted init() reprograms srca and the next
    //     element's hoisted init() overwrites it).
    // The trait `chain_is_hoist_safe_v` folds both conditions.
    constexpr bool emit_init_per_tile = !chain_is_hoist_safe_v<Chain>;

    using IdxSeq = std::make_index_sequence<sizeof...(Es)>;

    // ---- F-PERF-4: hoist pack init out of per-tile loop ----
    // Pack reconfig is fold-driven (D2) and idempotent — emit once at chain entry.
    (detail::elem_pack_init<Es>(), ...);

    if constexpr (!emit_init_per_tile) {
        // Non-clash chain: hoist all per-element transitions + init() to boot.
        // The per-tile loop emits only the lifecycle + exec.
        detail::hoisted_init_for_each(IdxSeq{}, elts...);
    }

    // Pack lifecycle ordering: reserve emitted as late as possible (right before
    // pack_exec) and push emitted as early as possible (right after pack_exec) so
    // the downstream consumer sees pushed tiles before this iteration releases
    // DEST. The per-tile reserve/push helpers are policy-guarded internally
    // (no-op for non-PerTile policies), so emitting them inside the pack window
    // is safe in both block and per-tile paths.
    if constexpr (block_path) {
        // Upfront block path: wait/reserve `n_tiles` worth of CB tiles, run the
        // per-tile loop, pop/push at end. Element types with block-size multipliers
        // (e.g. BlockCopyTile<Cb, BlockSize>) scale `n_tiles` internally.
        //
        // Wait-late amendment (item 6 / lessons §11): `cb_wait_front(cb, N)` and
        // `cb_reserve_back(cb, N)` are cumulative-count idempotent — emitting them
        // inside the loop blocks once on iter 0 and short-circuits thereafter,
        // recovering producer/consumer overlap. Pop/push-at-end stay outside the
        // loop (consumer needs the whole block visible until release).
        //
        // Mixed-policy chains (e.g. CopyTile WaitUpfrontPopAtEnd + PackTile
        // PerTileReserveAndPush — the softmax phase 2b pattern) need per-tile
        // reserve/push for the streaming output even though the input drove
        // block_path=true. Per-tile lifecycle ops are policy-guarded internally
        // (no-op on upfront elements), so emitting them every iter is correct.
        for (uint32_t i = 0; i < n_tiles; ++i) {
            // wait late — inside loop, late as possible before unpack reads
            (detail::elem_wait_upfront(elts, n_tiles), ...);
            (detail::elem_wait_per_tile(elts, i), ...);

            tile_regs_acquire();
            // Compute-phase: emit pre-element transitions (when init is per-tile)
            // and per-element exec.
            detail::compute_phase_for_each<emit_init_per_tile, emit_init_per_tile>(IdxSeq{}, i, elts...);
            tile_regs_commit();
            tile_regs_wait();
            // reserve late — inside loop, right before pack
            (detail::elem_reserve_per_tile(elts, i), ...);
            (detail::elem_reserve_upfront(elts, n_tiles), ...);
            (detail::elem_pack_exec(elts, i), ...);
            (detail::elem_push_per_tile(elts, i), ...);
            tile_regs_release();

            (detail::elem_pop_per_tile(elts, i), ...);
        }

        (detail::elem_pop_upfront_end(elts, n_tiles), ...);
        (detail::elem_push_at_end(elts, n_tiles), ...);
    } else {
        // Per-tile path (default — PerTile policies).
        for (uint32_t i = 0; i < n_tiles; ++i) {
            (detail::elem_wait_per_tile(elts, i), ...);

            tile_regs_acquire();
            detail::compute_phase_for_each<emit_init_per_tile, emit_init_per_tile>(IdxSeq{}, i, elts...);
            tile_regs_commit();
            tile_regs_wait();
            (detail::elem_reserve_per_tile(elts, i), ...);
            (detail::elem_pack_exec(elts, i), ...);
            (detail::elem_push_per_tile(elts, i), ...);
            tile_regs_release();

            (detail::elem_pop_per_tile(elts, i), ...);
        }
    }
}

// =============================================================================
// 12. eltwise_chain_with_init — deduced wrapper (U4)
//
// Single-stage convenience that derives (cb_a, cb_b, cb_out) at compile time
// from the chain element pack and emits compute_kernel_hw_startup before the
// chain. Multi-stage kernels (different PACK output CB per stage) MUST keep the
// explicit per-stage compute_kernel_hw_startup pattern.
// =============================================================================

namespace detail {

// `first_*_cb<Es...>()`: pack-fold deducers for the boot's three CBs.
//
// CB ID `0` is a legitimate CB index (`tt::CBIndex::c_0`). We can't use 0 as a
// sentinel for "not found". Instead the deducers walk Es left-to-right and return
// the CB from the first matching element (CB-reader / CB-binary / CB-pack); the
// caller's `static_assert` uses a separate "is found" predicate that walks the
// same predicate set.
//
// Implemented as recursive pack peel.

template <class E>
constexpr uint32_t cb_a_of() {
    if constexpr (is_cb_reader_op_v<E>) {
        static_assert(has_cb_a<E>::value,
                      "CbReader element must declare 'static constexpr uint32_t cb_a_id()'");
        return E::cb_a_id();
    } else {
        return 0u;
    }
}

template <class E>
constexpr uint32_t cb_b_of() {
    if constexpr (is_binary_fpu_op_v<E> || is_dest_reuse_binary_op_v<E>) {
        static_assert(has_cb_b<E>::value,
                      "Binary CbReader element must declare 'static constexpr uint32_t cb_b_id()'");
        return E::cb_b_id();
    } else {
        return 0u;
    }
}

template <class E>
constexpr uint32_t pack_cb_of() {
    if constexpr (is_pack_tile_op_v<E>) {
        static_assert(has_pack_cb<E>::value,
                      "CbWriter element must declare 'static constexpr uint32_t pack_cb_id()'");
        return E::pack_cb_id();
    } else {
        return 0u;
    }
}

// Has-any-* predicates so the caller can static_assert presence without false
// positives on CB index 0.
template <class... Es>
inline constexpr bool has_any_cb_reader_v = ((is_cb_reader_op_v<Es>) || ...);

template <class... Es>
inline constexpr bool has_any_pack_tile_v = ((is_pack_tile_op_v<Es>) || ...);

// First CB on side X, walking Es left-to-right and returning the first element
// of the matching kind. Returns 0 only if no matching element exists (caller is
// expected to gate via the has_any_* predicates above).

template <class... Es>
constexpr uint32_t first_cb_a_impl() {
    uint32_t result = 0;
    bool found = false;
    auto step = [&](bool is_reader, uint32_t cb) {
        if (!found && is_reader) { result = cb; found = true; }
    };
    (step(is_cb_reader_op_v<Es>, cb_a_of<Es>()), ...);
    return result;
}

template <class... Es>
constexpr uint32_t first_cb_b_impl() {
    uint32_t result = 0;
    bool found = false;
    auto step = [&](bool is_bin, uint32_t cb) {
        if (!found && is_bin) { result = cb; found = true; }
    };
    (step(is_binary_fpu_op_v<Es> || is_dest_reuse_binary_op_v<Es>, cb_b_of<Es>()), ...);
    return result;
}

template <class... Es>
constexpr uint32_t first_pack_cb_impl() {
    uint32_t result = 0;
    bool found = false;
    auto step = [&](bool is_pack, uint32_t cb) {
        if (!found && is_pack) { result = cb; found = true; }
    };
    (step(is_pack_tile_op_v<Es>, pack_cb_of<Es>()), ...);
    return result;
}

// Detect "has any binary FPU element" — used to decide cb_b vs falling back to cb_a.
template <class... Es>
inline constexpr bool has_any_binary_v = ((is_binary_fpu_op_v<Es> || is_dest_reuse_binary_op_v<Es>) || ...);

template <class... Es>
constexpr uint32_t first_cb_a()    { return first_cb_a_impl<Es...>(); }

template <class... Es>
constexpr uint32_t first_cb_b()    { return first_cb_b_impl<Es...>(); }

template <class... Es>
constexpr uint32_t first_pack_cb() { return first_pack_cb_impl<Es...>(); }

}  // namespace detail

template <class... Es>
ALWI void eltwise_chain_with_init(uint32_t n_tiles, Es... elts) {
    // Compile-time CB deduction. Note: CB ID 0 is a legitimate index
    // (tt::CBIndex::c_0), so we gate presence on the has_any_* predicates rather
    // than on `cb != 0`.
    static_assert(detail::has_any_pack_tile_v<Es...>,
                  "eltwise_chain_with_init: chain has no PackTile element. Multi-stage kernels "
                  "must use explicit per-stage compute_kernel_hw_startup, not this wrapper.");

    constexpr uint32_t cb_out = detail::first_pack_cb<Es...>();
    // Reader-less chains (fill-only) boot the engine using cb_out for both srca/srcb
    // — matches the legacy EltwiseChainPipelineInit::run() no-reader fallback.
    constexpr uint32_t cb_a   = detail::has_any_cb_reader_v<Es...> ? detail::first_cb_a<Es...>() : cb_out;
    constexpr uint32_t cb_b   = detail::has_any_binary_v<Es...> ? detail::first_cb_b<Es...>() : cb_a;

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);
    eltwise_chain(n_tiles, elts...);
}

}  // namespace compute_kernel_lib
