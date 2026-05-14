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
    // BlockIter / Absolute legal under WaitUpfrontPopAtEnd, CumulativeWaitPopAtEnd, NoWaitNoPop.
    // Single-tile-window policies (WaitAndPop, WaitNoPop, NoWaitPop) limit index to FirstTile
    // (or Pinned k where k == 0 — runtime-asserted, not statically expressible).
    static_assert(to_u32(DstSlot) < DEST_AUTO_LIMIT,
                  "CopyTile: DEST slot exceeds DEST_AUTO_LIMIT");
    static_assert(!(Policy == CopyTilePolicy::WaitAndPop  && IndexMode == CbIndexMode::BlockIter),
                  "CopyTile: BlockIter index requires Upfront / Cumulative / NoWaitNoPop policy");
    static_assert(!(Policy == CopyTilePolicy::WaitAndPop  && IndexMode == CbIndexMode::Absolute),
                  "CopyTile: Absolute index requires Upfront / Cumulative / NoWaitNoPop policy");
    static_assert(!(Policy == CopyTilePolicy::WaitNoPop   && IndexMode == CbIndexMode::BlockIter),
                  "CopyTile: BlockIter index requires Upfront / Cumulative / NoWaitNoPop policy");
    static_assert(!(Policy == CopyTilePolicy::WaitNoPop   && IndexMode == CbIndexMode::Absolute),
                  "CopyTile: Absolute index requires Upfront / Cumulative / NoWaitNoPop policy");
    static_assert(!(Policy == CopyTilePolicy::NoWaitPop   && IndexMode == CbIndexMode::BlockIter),
                  "CopyTile: BlockIter index requires Upfront / Cumulative / NoWaitNoPop policy");
    static_assert(!(Policy == CopyTilePolicy::NoWaitPop   && IndexMode == CbIndexMode::Absolute),
                  "CopyTile: Absolute index requires Upfront / Cumulative / NoWaitNoPop policy");

    static constexpr uint32_t       cb              = Cb;
    static constexpr uint32_t       cb_a_id()       { return Cb; }
    static constexpr uint32_t       cb_b_id()       { return 0;  }
    static constexpr Dst            dst_slot        = DstSlot;
    static constexpr CopyTilePolicy a_policy()      { return Policy; }
    static constexpr CopyTilePolicy b_policy()      { return CopyTilePolicy::NoWaitNoPop; }
    static constexpr bool           is_upfront      = (Policy == CopyTilePolicy::WaitUpfrontPopAtEnd) ||
                                                      (Policy == CopyTilePolicy::CumulativeWaitPopAtEnd);
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

    /// Per-iter wait. Element fires at its own granularity — streaming policies wait 1
    /// per iter, Cumulative grows wait count with i (i+1), Upfront fires once via
    /// wait_upfront with full n_tiles. None scale by chain block_size — block_size
    /// only drives the inner DEST-lane loop and slot_offset.
    ALWI void wait_per_tile(uint32_t i) const {
        if constexpr (Policy == CopyTilePolicy::WaitAndPop || Policy == CopyTilePolicy::WaitNoPop) {
            cb_wait_front(Cb, 1);
        } else if constexpr (Policy == CopyTilePolicy::CumulativeWaitPopAtEnd) {
            cb_wait_front(Cb, i + 1);
        }
    }

    ALWI void wait_upfront(uint32_t n) const {
        if constexpr (Policy == CopyTilePolicy::WaitUpfrontPopAtEnd) {
            cb_wait_front(Cb, n);
        }
    }

    ALWI void exec(uint32_t i, uint32_t slot_offset) const {
        const uint32_t in_idx = [&]() -> uint32_t {
            if constexpr (IndexMode == CbIndexMode::FirstTile) return 0;
            else if constexpr (IndexMode == CbIndexMode::BlockIter) return i;
            else return cb_tile_idx_;  // Pinned / Absolute
        }();
        copy_tile(Cb, in_idx, to_u32(DstSlot) + slot_offset);
    }

    static constexpr uint32_t lane_width = to_u32(DstSlot) + 1;

    ALWI void pop_per_tile(uint32_t /*i*/) const {
        if constexpr (Policy == CopyTilePolicy::WaitAndPop || Policy == CopyTilePolicy::NoWaitPop) {
            cb_pop_front(Cb, 1);
        }
    }

    ALWI void pop_upfront_end(uint32_t n) const {
        if constexpr (Policy == CopyTilePolicy::WaitUpfrontPopAtEnd ||
                      Policy == CopyTilePolicy::CumulativeWaitPopAtEnd) {
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

    ALWI void exec(uint32_t i, uint32_t slot_offset) const {
        const uint32_t out_idx = [&]() -> uint32_t {
            if constexpr (IndexMode == PackTileIndexMode::FirstTile) return 0;
            else if constexpr (IndexMode == PackTileIndexMode::BlockIter) return i;
            else return output_tile_idx_;  // Pinned / Absolute
        }();
        pack_tile(to_u32(DstSlot) + slot_offset, Cb, out_idx);
    }

    static constexpr uint32_t lane_width = to_u32(DstSlot) + 1;

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

    ALWI void reserve_per_tile(uint32_t /*i*/, uint32_t /*block_size*/) const {
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
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const {
        pack_tile_block(to_u32(FirstSlot) + slot_offset, Cb, NTiles);
    }

    static constexpr uint32_t lane_width = to_u32(FirstSlot) + NTiles;
    ALWI void push_per_tile(uint32_t /*i*/, uint32_t /*block_size*/) const {
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
                                                (APolicy == CopyTilePolicy::CumulativeWaitPopAtEnd) ||
                                                (BPolicy == CopyTilePolicy::WaitUpfrontPopAtEnd) ||
                                                (BPolicy == CopyTilePolicy::CumulativeWaitPopAtEnd);
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
    // Streaming policies (WaitAndPop / WaitNoPop) always wait 1 — they are incompatible
    // with BlockSize > 1 per-iter consumption. Cumulative scales `(i+1) * block_size`.
    ALWI void wait_per_tile(uint32_t i) const {
        if constexpr (APolicy == CopyTilePolicy::WaitAndPop || APolicy == CopyTilePolicy::WaitNoPop) {
            cb_wait_front(CbA, 1);
        } else if constexpr (APolicy == CopyTilePolicy::CumulativeWaitPopAtEnd) {
            cb_wait_front(CbA, i + 1);
        }
        if constexpr (!same_cb) {
            if constexpr (BPolicy == CopyTilePolicy::WaitAndPop || BPolicy == CopyTilePolicy::WaitNoPop) {
                cb_wait_front(CbB, 1);
            } else if constexpr (BPolicy == CopyTilePolicy::CumulativeWaitPopAtEnd) {
                cb_wait_front(CbB, i + 1);
            }
        }
    }

    ALWI void wait_upfront(uint32_t n) const {
        if constexpr (APolicy == CopyTilePolicy::WaitUpfrontPopAtEnd) cb_wait_front(CbA, n);
        if constexpr (!same_cb && BPolicy == CopyTilePolicy::WaitUpfrontPopAtEnd) cb_wait_front(CbB, n);
    }

    ALWI void exec(uint32_t i, uint32_t slot_offset) const {
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
        const uint32_t dst = to_u32(DstSlot) + slot_offset;
        if constexpr (Bcast == BroadcastDim::None) {
            if constexpr      (Op == BinaryFpuOp::Add) add_tiles(CbA, CbB, a_idx, b_idx, dst);
            else if constexpr (Op == BinaryFpuOp::Sub) sub_tiles(CbA, CbB, a_idx, b_idx, dst);
            else                                       mul_tiles(CbA, CbB, a_idx, b_idx, dst);
        } else {
            // Broadcast variants via the generic `add/sub/mul_tiles_bcast<BroadcastType>` template
            // — these forward to `any_tiles_bcast<EltwiseBinaryType, BroadcastType>` internally.
            constexpr auto bt = static_cast<ckernel::BroadcastType>(static_cast<uint8_t>(Bcast));
            if constexpr      (Op == BinaryFpuOp::Add) add_tiles_bcast<bt>(CbA, CbB, a_idx, b_idx, dst);
            else if constexpr (Op == BinaryFpuOp::Sub) sub_tiles_bcast<bt>(CbA, CbB, a_idx, b_idx, dst);
            else                                       mul_tiles_bcast<bt>(CbA, CbB, a_idx, b_idx, dst);
        }
    }

    static constexpr uint32_t lane_width = to_u32(DstSlot) + 1;

    ALWI void pop_per_tile(uint32_t /*i*/) const {
        if constexpr (APolicy == CopyTilePolicy::WaitAndPop || APolicy == CopyTilePolicy::NoWaitPop) {
            cb_pop_front(CbA, 1);
        }
        if constexpr (!same_cb && (BPolicy == CopyTilePolicy::WaitAndPop || BPolicy == CopyTilePolicy::NoWaitPop)) {
            cb_pop_front(CbB, 1);
        }
    }

    ALWI void pop_upfront_end(uint32_t n) const {
        if constexpr (APolicy == CopyTilePolicy::WaitUpfrontPopAtEnd ||
                      APolicy == CopyTilePolicy::CumulativeWaitPopAtEnd) {
            cb_pop_front(CbA, n);
        }
        if constexpr (!same_cb && (BPolicy == CopyTilePolicy::WaitUpfrontPopAtEnd ||
                                   BPolicy == CopyTilePolicy::CumulativeWaitPopAtEnd)) {
            cb_pop_front(CbB, n);
        }
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
                  "DestReuseBinary: BlockIter index requires Upfront / Cumulative / NoWaitNoPop policy");

    static constexpr uint32_t       cb_a_id()         { return Cb; }
    static constexpr uint32_t       cb_b_id()         { return 0;  }
    static constexpr CopyTilePolicy a_policy()        { return Policy; }
    static constexpr CopyTilePolicy b_policy()        { return CopyTilePolicy::NoWaitNoPop; }
    static constexpr Dst            dst_slot          = DstOut;
    static constexpr bool           is_upfront        = (Policy == CopyTilePolicy::WaitUpfrontPopAtEnd) ||
                                                        (Policy == CopyTilePolicy::CumulativeWaitPopAtEnd);
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

    ALWI void wait_per_tile(uint32_t i) const {
        if constexpr (Policy == CopyTilePolicy::WaitAndPop || Policy == CopyTilePolicy::WaitNoPop) {
            cb_wait_front(Cb, 1);
        } else if constexpr (Policy == CopyTilePolicy::CumulativeWaitPopAtEnd) {
            cb_wait_front(Cb, i + 1);
        }
    }
    ALWI void wait_upfront(uint32_t n) const {
        if constexpr (Policy == CopyTilePolicy::WaitUpfrontPopAtEnd) cb_wait_front(Cb, n);
    }
    ALWI void exec(uint32_t i, uint32_t slot_offset) const {
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
        binary_dest_reuse_tiles<et, reuse>(Cb, in_idx, to_u32(DstIn) + slot_offset);
    }

    static constexpr uint32_t lane_width =
        (to_u32(DstIn) > to_u32(DstOut)) ? (to_u32(DstIn) + 1) : (to_u32(DstOut) + 1);
    ALWI void pop_per_tile(uint32_t /*i*/) const {
        if constexpr (Policy == CopyTilePolicy::WaitAndPop || Policy == CopyTilePolicy::NoWaitPop) {
            cb_pop_front(Cb, 1);
        }
    }
    ALWI void pop_upfront_end(uint32_t n) const {
        if constexpr (Policy == CopyTilePolicy::WaitUpfrontPopAtEnd ||
                      Policy == CopyTilePolicy::CumulativeWaitPopAtEnd) {
            cb_pop_front(Cb, n);
        }
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
    static constexpr bool           is_upfront        = (Policy == CopyTilePolicy::WaitUpfrontPopAtEnd) ||
                                                        (Policy == CopyTilePolicy::CumulativeWaitPopAtEnd);
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

    ALWI void wait_per_tile(uint32_t i) const {
        if constexpr (Policy == CopyTilePolicy::WaitAndPop || Policy == CopyTilePolicy::WaitNoPop) {
            cb_wait_front(Cb, 1);
        } else if constexpr (Policy == CopyTilePolicy::CumulativeWaitPopAtEnd) {
            cb_wait_front(Cb, i + 1);
        }
    }
    ALWI void wait_upfront(uint32_t n) const {
        if constexpr (Policy == CopyTilePolicy::WaitUpfrontPopAtEnd) cb_wait_front(Cb, n);
    }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const {
        constexpr auto bt = static_cast<ckernel::BroadcastType>(static_cast<uint8_t>(Dim));
        unary_bcast<bt>(Cb, /*in_tile_index=*/0, to_u32(DstSlot) + slot_offset);
    }

    static constexpr uint32_t lane_width = to_u32(DstSlot) + 1;
    ALWI void pop_per_tile(uint32_t /*i*/) const {
        if constexpr (Policy == CopyTilePolicy::WaitAndPop || Policy == CopyTilePolicy::NoWaitPop) {
            cb_pop_front(Cb, 1);
        }
    }
    ALWI void pop_upfront_end(uint32_t n) const {
        if constexpr (Policy == CopyTilePolicy::WaitUpfrontPopAtEnd ||
                      Policy == CopyTilePolicy::CumulativeWaitPopAtEnd) {
            cb_pop_front(Cb, n);
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

// chain_lane_width — N-element fold (item 2). Max of per-element `lane_width`. Drives
// auto-block: chain BlockSize = DEST_AUTO_LIMIT / chain_lane_width when AutoBlock::On.
// Each element writes to DEST[dst_slot + j * chain_lane_width] for lane j in [0, BlockSize).
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

template <class... Es>
constexpr uint32_t chain_lane_width_impl_v = []() {
    if constexpr (sizeof...(Es) == 0) {
        return uint32_t{1};
    } else {
        uint32_t w = 1;
        ((w = (elem_lane_width_v<Es> > w ? elem_lane_width_v<Es> : w)), ...);
        return w;
    }
}();
}  // namespace detail

template <class Chain>
struct chain_lane_width;

template <class... Es>
struct chain_lane_width<EltwiseChain<Es...>>
    : std::integral_constant<uint32_t, detail::chain_lane_width_impl_v<Es...>> {};

template <class Chain>
inline constexpr uint32_t chain_lane_width_v = chain_lane_width<Chain>::value;

// chain_supports_block — N-element fold. True when every CB-reader element uses a
// policy that stages a multi-tile DEST window (Upfront / Cumulative / NoWaitNoPop).
// Streaming policies (WaitAndPop / WaitNoPop / NoWaitPop) consume ONE tile per iter
// and are incompatible with chain BlockSize > 1 (chain consumes BlockSize tiles per
// outer iter). The chain `static_assert`s on this predicate when AutoBlock::On.
namespace detail {
constexpr bool policy_supports_block(CopyTilePolicy p) {
    return p == CopyTilePolicy::WaitUpfrontPopAtEnd ||
           p == CopyTilePolicy::CumulativeWaitPopAtEnd ||
           p == CopyTilePolicy::NoWaitNoPop;
}

template <class E>
constexpr bool element_supports_block() {
    if constexpr (is_cb_reader_op_v<E>) {
        return policy_supports_block(E::a_policy()) && policy_supports_block(E::b_policy());
    } else {
        return true;  // non-CB-reader elements don't constrain block_size
    }
}

template <class... Es>
constexpr bool chain_supports_block_impl_v = (element_supports_block<Es>() && ...);
}  // namespace detail

template <class Chain>
struct chain_supports_block;

template <class... Es>
struct chain_supports_block<EltwiseChain<Es...>>
    : std::bool_constant<detail::chain_supports_block_impl_v<Es...>> {};

template <class Chain>
inline constexpr bool chain_supports_block_v = chain_supports_block<Chain>::value;

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
//
// **Disabled (always false).** The previous predicate assumed SFPU ops were always
// hoist-safe as a group, but multiple distinct SFPU `*_tile_init` calls done up front
// leave only the LAST init's MOP programmed — so subsequent execs reuse the wrong
// SFPU state. Symptoms seen on this branch:
//   - mish_kernel.cpp FP32 path: Exp/Log1p/Tanh chain produced tanh-saturated output
//     (last init clobbered). PCC dropped to 0.988 vs golden.
//   - logit_kernel.cpp stage-2 chain (Rsub → DivBinary → Log): 21 bfloat16 logit
//     tests with ATOL deltas of 9-12 (expected 0.04).
//
// Until the predicate is rewritten to detect SFPU-init heterogeneity (e.g. fingerprint
// init member types and require ≤ 1 unique), force per-tile init for every chain. The
// per-tile init cost is small relative to the SFPU op cost; correctness > micro-perf.
template <class Chain>
struct chain_is_hoist_safe : std::false_type {};

template <class... Es>
struct chain_is_hoist_safe<EltwiseChain<Es...>> : std::false_type {};

// =============================================================================
// 10. Chain pipeline — per-iteration emit
// =============================================================================

namespace detail {

// End-of-outer-loop lifecycle dispatch. Fire AFTER the outer loop ends for upfront-policy
// elements (policy-gated internally — no-op for per-tile-policy elements). The per-tile
// wait / pop / reserve / push pieces live INSIDE the apply_compute / apply_pack body
// (single element-owned lifecycle per outer iter).
template <class E>
ALWI void elem_pop_upfront_end(const E& e, uint32_t n) {
    if constexpr (is_cb_reader_op_v<E>) e.pop_upfront_end(n);
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

// Pack-phase init (Pack* only — F-PERF-4: hoisted to boot, not per-tile).
// Note: post-commit-2 the pack reconfig is fold-driven via `emit_pre_element_transitions`,
// so this is effectively a no-op for PackTile / PackTileBlock. Retained for symmetry
// in case a pack element gains per-op LLK programming in a future commit.
template <class E>
ALWI void elem_pack_init() {
    if constexpr (is_pack_tile_op_v<E>) E::init();
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

template <bool EmitInit, std::size_t I, class ElemT, class... Es>
ALWI void elem_apply_compute(
    const ElemT& elem,
    uint32_t i_outer,
    uint32_t base_tile,
    uint32_t inner_count,
    uint32_t chain_lane_width,
    uint32_t n_tiles) {
    if constexpr (is_pack_tile_op_v<ElemT>) {
        (void)elem; (void)i_outer; (void)base_tile; (void)inner_count; (void)chain_lane_width; (void)n_tiles;
    } else if constexpr (is_cb_reader_op_v<ElemT>) {
        elem.wait_per_tile(i_outer);
        elem.wait_upfront(n_tiles);
        if constexpr (EmitInit) {
            emit_pre_element_transitions<ElemT, I, Es...>();
            ElemT::init();
        }
        // Lane j writes DEST[dst_slot + j * chain_lane_width]; tile index =
        // base_tile + j (absolute, tail-safe — i_outer * BlockSize, not * inner_count).
        for (uint32_t j = 0; j < inner_count; ++j) {
            elem.exec(base_tile + j, j * chain_lane_width);
        }
        elem.pop_per_tile(i_outer);
    } else if constexpr (is_dest_only_op_v<ElemT>) {
        if constexpr (EmitInit) {
            emit_pre_element_transitions<ElemT, I, Es...>();
            ElemT::init();
        }
        for (uint32_t j = 0; j < inner_count; ++j) {
            elem.exec(base_tile + j, j * chain_lane_width);
        }
    }
}

template <std::size_t I, class ElemT, class... Es>
ALWI void elem_apply_pack(
    const ElemT& elem,
    uint32_t i_outer,
    uint32_t base_tile,
    uint32_t inner_count,
    uint32_t chain_lane_width,
    uint32_t n_tiles) {
    if constexpr (is_pack_tile_op_v<ElemT>) {
        elem.reserve_per_tile(i_outer);
        elem.reserve_upfront(n_tiles);
        for (uint32_t j = 0; j < inner_count; ++j) {
            elem.exec(base_tile + j, j * chain_lane_width);
        }
        elem.push_per_tile(i_outer);
    } else {
        (void)elem; (void)i_outer; (void)base_tile; (void)inner_count; (void)chain_lane_width; (void)n_tiles;
    }
}

template <bool EmitInit, std::size_t... Is, class... Es>
ALWI void apply_compute_phase(
    std::index_sequence<Is...>,
    uint32_t i_outer,
    uint32_t base_tile,
    uint32_t inner_count,
    uint32_t chain_lane_width,
    uint32_t n_tiles,
    Es&... elts) {
    auto run_one = [&](auto idx_const, auto& elem) {
        constexpr std::size_t II = decltype(idx_const)::value;
        using ElemT = std::remove_reference_t<decltype(elem)>;
        elem_apply_compute<EmitInit, II, ElemT, Es...>(
            elem, i_outer, base_tile, inner_count, chain_lane_width, n_tiles);
    };
    (run_one(std::integral_constant<std::size_t, Is>{}, elts), ...);
}

template <std::size_t... Is, class... Es>
ALWI void apply_pack_phase(
    std::index_sequence<Is...>,
    uint32_t i_outer,
    uint32_t base_tile,
    uint32_t inner_count,
    uint32_t chain_lane_width,
    uint32_t n_tiles,
    Es&... elts) {
    auto run_one = [&](auto idx_const, auto& elem) {
        constexpr std::size_t II = decltype(idx_const)::value;
        using ElemT = std::remove_reference_t<decltype(elem)>;
        elem_apply_pack<II, ElemT, Es...>(elem, i_outer, base_tile, inner_count, chain_lane_width, n_tiles);
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

template <AutoBlock Block, class... Es>
ALWI void eltwise_chain(uint32_t n_tiles, Es... elts) {
    using Chain = EltwiseChain<Es...>;

    // ---- Compile-time invariant checks ----
    static_assert(!chain_has_duplicate_upfront_cbs_v<Chain>,
                  "eltwise_chain: two CB-reader elements share a CB on upfront-wait policy. "
                  "Each upfront-wait CB must appear in exactly one element.");
    static_assert(!chain_pack_writes_collide_v<Chain>,
                  "eltwise_chain: two PackTile elements collide on (cb, dst_slot). "
                  "Pack writes must target distinct (cb, dst) tuples.");

    constexpr bool emit_init_per_tile = !chain_is_hoist_safe_v<Chain>;

    // ---- Auto-block (item 2 of eltwise_helper_proposal.md) ----
    //
    // AutoBlock::On  → BlockSize = DEST_AUTO_LIMIT / chain_lane_width.
    //                  Each outer iter processes BlockSize tiles in BlockSize DEST lanes
    //                  (lane j at slot dst_slot + j * chain_lane_width). Requires every
    //                  CB-reader policy to stage a multi-tile window — see
    //                  chain_supports_block_v.
    // AutoBlock::Off → BlockSize = 1 (today's per-tile shape).
    static_assert(Block == AutoBlock::Off || chain_supports_block_v<Chain>,
                  "eltwise_chain<AutoBlock::On>: streaming CB-reader policy (WaitAndPop / "
                  "WaitNoPop / NoWaitPop) consumes one tile per iter — incompatible with "
                  "BlockSize > 1. Switch the reader to WaitUpfrontPopAtEnd, "
                  "CumulativeWaitPopAtEnd, or NoWaitNoPop, or call eltwise_chain<AutoBlock::Off>.");
    constexpr uint32_t chain_lane_w = chain_lane_width_v<Chain>;
    constexpr uint32_t auto_block_size = DEST_AUTO_LIMIT / chain_lane_w;
    constexpr uint32_t block_size = (Block == AutoBlock::On) ? auto_block_size : 1u;
    static_assert(block_size >= 1, "eltwise_chain: chain_lane_width exceeds DEST_AUTO_LIMIT");

    using IdxSeq = std::make_index_sequence<sizeof...(Es)>;

    // ---- F-PERF-4: hoist pack init out of per-tile loop ----
    (detail::elem_pack_init<Es>(), ...);

    if constexpr (!emit_init_per_tile) {
        detail::hoisted_init_for_each(IdxSeq{}, elts...);
    }

    // Outer loop processes `block_size` tiles per iter. Runtime tail handles the case
    // where `n_tiles % block_size != 0`: the last iter's inner block size clamps to
    // `n_tiles - i_outer * block_size`.
    for (uint32_t i_outer = 0; i_outer * block_size < n_tiles; ++i_outer) {
        const uint32_t base_tile = i_outer * block_size;
        const uint32_t inner_count =
            (base_tile + block_size <= n_tiles) ? block_size : (n_tiles - base_tile);
        tile_regs_acquire();
        detail::apply_compute_phase<emit_init_per_tile>(
            IdxSeq{}, i_outer, base_tile, inner_count, chain_lane_w, n_tiles, elts...);
        tile_regs_commit();
        tile_regs_wait();
        detail::apply_pack_phase(IdxSeq{}, i_outer, base_tile, inner_count, chain_lane_w, n_tiles, elts...);
        tile_regs_release();
    }

    // End-of-chain upfront-policy lifecycle (policy-gated no-op for non-upfront elts).
    (detail::elem_pop_upfront_end(elts, n_tiles), ...);
    (detail::elem_push_at_end(elts, n_tiles), ...);
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

template <AutoBlock Block, class... Es>
ALWI void eltwise_chain_with_init(uint32_t n_tiles, Es... elts) {
    static_assert(detail::has_any_pack_tile_v<Es...>,
                  "eltwise_chain_with_init: chain has no PackTile element. Multi-stage kernels "
                  "must use explicit per-stage compute_kernel_hw_startup, not this wrapper.");

    constexpr uint32_t cb_out = detail::first_pack_cb<Es...>();
    constexpr uint32_t cb_a   = detail::has_any_cb_reader_v<Es...> ? detail::first_cb_a<Es...>() : cb_out;
    constexpr uint32_t cb_b   = detail::has_any_binary_v<Es...> ? detail::first_cb_b<Es...>() : cb_a;

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);
    eltwise_chain<Block>(n_tiles, elts...);
}

}  // namespace compute_kernel_lib
