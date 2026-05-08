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

}  // namespace detail

// =============================================================================
// 1. CopyTile chain element
// =============================================================================

template <uint32_t Cb,
          Dst DstSlot,
          CopyTilePolicy Policy,
          CbIndexMode IndexMode,
          CopyTileReconfig Reconfig,
          uint32_t OldCb>
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

    /// Runtime tile index — set by user only when IndexMode == Pinned / Absolute.
    uint32_t cb_tile_idx = 0;

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
            else return cb_tile_idx;  // Pinned / Absolute
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
};

// =============================================================================
// 2. PackTile chain element
// =============================================================================

template <uint32_t Cb,
          Dst DstSlot,
          PackTilePolicy Policy,
          PackTileIndexMode IndexMode,
          PackTileReconfig Reconfig,
          uint32_t OldCb>
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

    /// Runtime output-tile index for Pinned / Absolute modes.
    uint32_t output_tile_idx = 0;

    static ALWI void init() {
        // Single-arg pack reconfig — no previous-CB tracking. (`OutputConditional` retained
        // for source compatibility but emits the same single-arg call.)
        if constexpr (Reconfig == PackTileReconfig::Output ||
                      Reconfig == PackTileReconfig::OutputConditional) {
            pack_reconfig_data_format(Cb);
        }
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
            else return output_tile_idx;  // Pinned / Absolute
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
};

// =============================================================================
// 3. PackTileBlock — atomic multi-slot pack
// =============================================================================

template <uint32_t Cb,
          Dst FirstSlot,
          uint32_t NTiles,
          PackTilePolicy Policy,
          PackTileReconfig Reconfig,
          uint32_t OldCb>
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

    static ALWI void init() {
        // Single-arg pack reconfig — no previous-CB tracking.
        if constexpr (Reconfig == PackTileReconfig::Output ||
                      Reconfig == PackTileReconfig::OutputConditional) {
            pack_reconfig_data_format(Cb);
        }
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
          BinaryFpuOp Op,
          BroadcastDim Bcast,
          BinaryFpuOutputPolicy OutPolicy,
          BinaryDataFormatReconfig DfReconfig,
          CopyTilePolicy APolicy,
          CopyTilePolicy BPolicy,
          CbIndexMode AIndex,
          CbIndexMode BIndex,
          Dst DstSlot,
          uint32_t OldCbA,
          uint32_t OldCbB,
          uint32_t OldCbOut,
          uint32_t CbOut>
struct BinaryFpu : BinaryFpuTag {
    static_assert(to_u32(DstSlot) < DEST_AUTO_LIMIT,
                  "BinaryFpu: DEST slot exceeds DEST_AUTO_LIMIT");
    // BinaryFpu reads each operand once per tile — index modes constrained like CopyTile.
    static_assert(!(APolicy == CopyTilePolicy::WaitAndPop && AIndex == CbIndexMode::BlockIter),
                  "BinaryFpu A: BlockIter index requires Upfront / NoWaitNoPop policy");
    static_assert(!(BPolicy == CopyTilePolicy::WaitAndPop && BIndex == CbIndexMode::BlockIter),
                  "BinaryFpu B: BlockIter index requires Upfront / NoWaitNoPop policy");

    static constexpr uint32_t      cb_a_id()  { return CbA; }
    static constexpr uint32_t      cb_b_id()  { return CbB; }
    static constexpr CopyTilePolicy a_policy(){ return APolicy; }
    static constexpr CopyTilePolicy b_policy(){ return BPolicy; }
    static constexpr Dst           dst_slot   = DstSlot;
    static constexpr bool          is_upfront = (APolicy == CopyTilePolicy::WaitUpfrontPopAtEnd) ||
                                                (BPolicy == CopyTilePolicy::WaitUpfrontPopAtEnd);
    static constexpr bool          clashes_with_fpu = true;
    static constexpr bool          same_cb    = (CbA == CbB);

    /// Runtime indices for Pinned / Absolute modes.
    uint32_t a_tile_idx = 0;
    uint32_t b_tile_idx = 0;

    // ---- init / reconfig ----
    static ALWI void init() {
        // Input-side reconfig (single-arg — no previous-CB tracking).
        if constexpr (DfReconfig == BinaryDataFormatReconfig::Input ||
                      DfReconfig == BinaryDataFormatReconfig::InputAndOutput) {
            reconfig_data_format_srca(CbA);
            reconfig_data_format_srcb(CbB);
        }
        // Output-side reconfig (pack, single-arg).
        if constexpr ((DfReconfig == BinaryDataFormatReconfig::Output ||
                       DfReconfig == BinaryDataFormatReconfig::InputAndOutput) && CbOut != 0) {
            pack_reconfig_data_format(CbOut);
        }
        // Op-specific init.
        if constexpr (Bcast == BroadcastDim::None) {
            if constexpr      (Op == BinaryFpuOp::Add) add_tiles_init(CbA, CbB);
            else if constexpr (Op == BinaryFpuOp::Sub) sub_tiles_init(CbA, CbB);
            else                                       mul_tiles_init(CbA, CbB);
        } else {
            // Broadcast init via init_bcast<EltwiseBinaryType, BroadcastType>.
            constexpr auto bt = static_cast<ckernel::BroadcastType>(static_cast<uint8_t>(Bcast));
            constexpr auto et = (Op == BinaryFpuOp::Add) ? ckernel::EltwiseBinaryType::ELWADD :
                                (Op == BinaryFpuOp::Sub) ? ckernel::EltwiseBinaryType::ELWSUB :
                                                           ckernel::EltwiseBinaryType::ELWMUL;
            constexpr uint32_t ocb = (CbOut != 0) ? CbOut : CbA;
            init_bcast<et, bt>(CbA, CbB, ocb);
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
        const uint32_t a_idx = [&]() -> uint32_t {
            if constexpr (AIndex == CbIndexMode::FirstTile) return 0;
            else if constexpr (AIndex == CbIndexMode::BlockIter) return i;
            else return a_tile_idx;
        }();
        const uint32_t b_idx = [&]() -> uint32_t {
            if constexpr (BIndex == CbIndexMode::FirstTile) return 0;
            else if constexpr (BIndex == CbIndexMode::BlockIter) return i;
            else return b_tile_idx;
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
          CbIndexMode IndexMode,
          uint32_t OldCb>
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

    uint32_t cb_tile_idx = 0;

    static ALWI void init() {
        constexpr auto et = (Op == BinaryFpuOp::Add) ? ckernel::EltwiseBinaryType::ELWADD :
                            (Op == BinaryFpuOp::Sub) ? ckernel::EltwiseBinaryType::ELWSUB :
                                                       ckernel::EltwiseBinaryType::ELWMUL;
        constexpr auto reuse = (ReuseType == DestReuseType::DEST_TO_SRCA)
                                   ? ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCA
                                   : ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCB;
        if constexpr (Reconfig == DestReuseReconfig::Input) {
            // Single-arg reconfig — no previous-CB tracking.
            if constexpr (ReuseType == DestReuseType::DEST_TO_SRCB) reconfig_data_format_srca(Cb);
            else                                                     reconfig_data_format_srcb(Cb);
        }
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
            else return cb_tile_idx;
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
};

// =============================================================================
// 6. UnaryBcast chain element
// =============================================================================

template <BroadcastDim Dim,
          uint32_t Cb,
          uint32_t CbOut,
          Dst DstSlot,
          CopyTilePolicy Policy,
          UnaryBcastReconfig Reconfig,
          uint32_t OldCb,
          uint32_t OldCbOut>
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

template <class Chain>
struct chain_loads_share_cb : std::false_type {};   // refined below for size-2 CopyTile-only chains

template <class A, class B>
struct chain_loads_share_cb<EltwiseChain<A, B>>
    : std::bool_constant<is_copy_tile_op_v<A> && is_copy_tile_op_v<B> && (A::cb == B::cb)> {};

// chain_has_duplicate_upfront_cbs / chain_pack_writes_collide:
// defined as a runtime fold for now — every CB-reader / CB-writer pair is checked.
// Static assertions in the chain pipeline use these as constexpr-evaluated booleans.

namespace detail {

template <class A, class B>
constexpr bool reader_pair_collide() {
    if constexpr (!is_cb_reader_op_v<A> || !is_cb_reader_op_v<B>) return false;
    else if constexpr (!A::is_upfront || !B::is_upfront)          return false;
    else {
        constexpr uint32_t a0 = A::cb_a_id(), a1 = A::cb_b_id();
        constexpr uint32_t b0 = B::cb_a_id(), b1 = B::cb_b_id();
        return (a0 != 0 && (a0 == b0 || a0 == b1)) ||
               (a1 != 0 && (a1 == b0 || a1 == b1));
    }
}

template <class A, class B>
constexpr bool writer_pair_collide() {
    if constexpr (!is_pack_tile_op_v<A> || !is_pack_tile_op_v<B>) return false;
    else                                                          return (A::pack_cb_id() == B::pack_cb_id()) &&
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

// Hoist-safe = exactly two elements, CopyTile + DestOnlyTag op (single-input pure SFPU).
template <class Chain> struct chain_is_hoist_safe : std::false_type {};

template <class A, class B>
struct chain_is_hoist_safe<EltwiseChain<A, B>>
    : std::bool_constant<is_copy_tile_op_v<A> && is_dest_only_op_v<B>> {};

template <class A, class B, class C>
struct chain_is_hoist_safe<EltwiseChain<A, B, C>>
    : std::bool_constant<is_copy_tile_op_v<A> && is_dest_only_op_v<B> && is_pack_tile_op_v<C>> {};

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

// SFINAE detector — does `E` have a callable `void exec(uint32_t) const` member?
// Used to route runtime-param SFPU ops (UnaryNe / Clamp / MulUnary / FillScalar / RandTile / ...)
// to member dispatch even though they inherit `DestOnlyTag` via the CRTP base.
template <class T, class = void> struct has_member_exec : std::false_type {};
template <class T>
struct has_member_exec<T, std::void_t<decltype(std::declval<const T>().exec(uint32_t{}))>>
    : std::true_type {};
template <class T> inline constexpr bool has_member_exec_v = has_member_exec<T>::value;

// Compute-phase exec (everything except Pack*).
// Runs between `tile_regs_acquire()` and `tile_regs_commit()`.
// For FPU-clash patterns each element's init() runs immediately before its exec —
// this matches production kernels (`copy_tile_init; copy_tile; sfpu_init; sfpu_tile;
// binary_dest_reuse_init; binary_dest_reuse_tiles`) and avoids state-clobbering when
// multiple chain elements reconfigure the unpacker.
template <class E>
ALWI void elem_compute_exec(const E& e, uint32_t i) {
    if constexpr (is_pack_tile_op_v<E>) {
        // Pack runs in the pack phase, not compute. Skip here.
        (void)e; (void)i;
    } else if constexpr (has_member_exec_v<E>) {
        // CB-bound element, Fill/Rand, OR runtime-param SFPU op (UnaryNe etc.) —
        // dispatch via member exec(i) so runtime payload (param0, value, idx, ...) is captured.
        E::init();
        e.exec(i);
    } else if constexpr (is_dest_only_op_v<E>) {
        // Pure SFPU op via CRTP base — static init() + static exec().
        E::init();
        E::exec();
    } else {
        // Fallback (shouldn't reach for well-formed chain elements).
        E::init();
        (void)e; (void)i;
    }
}

// Pack-phase exec (Pack* only).
// Runs between `tile_regs_wait()` and `tile_regs_release()`.
template <class E>
ALWI void elem_pack_exec(const E& e, uint32_t i) {
    if constexpr (is_pack_tile_op_v<E>) e.exec(i);
}

// Pack-phase init (Pack* only — pack_reconfig_data_format if Reconfig != None).
template <class E>
ALWI void elem_pack_init() {
    if constexpr (is_pack_tile_op_v<E>) E::init();
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

template <EltwiseChainOptions Opts, class... Es>
ALWI void eltwise_chain(uint32_t n_tiles, Es... elts) {
    using Chain = EltwiseChain<Es...>;

    // ---- Compile-time invariant checks ----
    static_assert(!chain_has_duplicate_upfront_cbs_v<Chain>,
                  "eltwise_chain: two CB-reader elements share a CB on upfront-wait policy. "
                  "Each upfront-wait CB must appear in exactly one element.");
    static_assert(!chain_pack_writes_collide_v<Chain>,
                  "eltwise_chain: two PackTile elements collide on (cb, dst_slot). "
                  "Pack writes must target distinct (cb, dst) tuples.");

    // ---- Detect upfront-block path ----
    constexpr bool any_upfront = ((Es::is_upfront) || ...);
    constexpr bool block_path  = (Opts.upfront_block_size > 0) || any_upfront;
    constexpr uint32_t block_n = (Opts.upfront_block_size > 0) ? Opts.upfront_block_size : 0;

    static_assert(!any_upfront || block_path,
                  "eltwise_chain: upfront policy used but EltwiseChainOptions.upfront_block_size == 0");

    if constexpr (block_path) {
        (detail::elem_wait_upfront(elts, block_n), ...);
        (detail::elem_reserve_upfront(elts, block_n), ...);

        for (uint32_t i = 0; i < n_tiles; ++i) {
            tile_regs_acquire();
            // Per-element init+exec: each element's init runs immediately before its
            // exec, mirroring production kernels and avoiding state clobber across the
            // FPU-clash chain.
            (detail::elem_compute_exec(elts, i), ...);
            tile_regs_commit();
            tile_regs_wait();
            // Pack-side init runs once per pack element (idempotent reconfig).
            (detail::elem_pack_init<Es>(), ...);
            (detail::elem_pack_exec(elts, i), ...);
            tile_regs_release();
        }

        (detail::elem_pop_upfront_end(elts, block_n), ...);
        (detail::elem_push_at_end(elts, block_n), ...);
    } else {
        // Per-tile path (default — PerTile policies).
        for (uint32_t i = 0; i < n_tiles; ++i) {
            (detail::elem_wait_per_tile(elts, i), ...);
            (detail::elem_reserve_per_tile(elts, i), ...);

            tile_regs_acquire();
            (detail::elem_compute_exec(elts, i), ...);
            tile_regs_commit();
            tile_regs_wait();
            (detail::elem_pack_init<Es>(), ...);
            (detail::elem_pack_exec(elts, i), ...);
            tile_regs_release();

            (detail::elem_pop_per_tile(elts, i), ...);
            (detail::elem_push_per_tile(elts, i), ...);
        }
    }
}

}  // namespace compute_kernel_lib
