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

// =============================================================================
// fp32_or_default<E, Default> — SFINAE probe for D6 EnableFp32DestAcc member.
//
// Returns `E::EnableFp32DestAcc` when E (a CARRY-list element) carries the flag;
// returns `Default` (the running prev) when E does not (SKIP-list elements).
// Used by the fp32-dest-acc fold to pass the running value through SKIP elements
// transparently.
// =============================================================================

template <class E, bool Default, class = void>
struct fp32_or_default {
    static constexpr bool value = Default;
};
template <class E, bool Default>
struct fp32_or_default<E, Default, std::void_t<decltype(E::EnableFp32DestAcc)>> {
    static constexpr bool value = E::EnableFp32DestAcc;
};

// has_fp32_dest_acc_v<E> — true when E::EnableFp32DestAcc is a member.
// Resolved by the same SFINAE machinery as `fp32_or_default`.
template <class E, class = void>
struct has_fp32_dest_acc : std::false_type {};
template <class E>
struct has_fp32_dest_acc<E, std::void_t<decltype(E::EnableFp32DestAcc)>> : std::true_type {};

// =============================================================================
// prev_fp32_dest_acc_for_idx<I, Es...>()
//
// Walks Es[0..I-1] forwards (recursive pack-peel), threading the running fp32 flag
// through each element. SKIP elements keep the prior value (no member); CARRY
// elements anchor it (`E::EnableFp32DestAcc`). Default (no prior element) is `false`
// per Q6 — chain inherits no fp32 state from the kernel.
//
// `prev_fp32_walk` is the recursion: K = current index, I = stop index, running =
// the value seen so far.
// =============================================================================

template <std::size_t K, std::size_t I, bool Running, class First, class... Rest>
constexpr bool prev_fp32_walk_step() {
    if constexpr (K == I) {
        return Running;
    } else {
        constexpr bool next = has_fp32_dest_acc<First>::value
                                  ? fp32_or_default<First, Running>::value
                                  : Running;
        if constexpr (sizeof...(Rest) == 0) {
            // Walked the whole pack but K < I — should not happen for well-formed I; guard.
            return next;
        } else {
            return prev_fp32_walk_step<K + 1, I, next, Rest...>();
        }
    }
}

template <std::size_t I, class... Es>
constexpr bool prev_fp32_dest_acc_for_idx() {
    if constexpr (I == 0 || sizeof...(Es) == 0) {
        return false;
    } else {
        return prev_fp32_walk_step<0, I, false, Es...>();
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
          bool EnableFp32DestAccV>
struct PackTile : PackTileTag {
    static_assert(!EnableFp32DestAccV || DST_ACCUM_MODE,
                  "PackTile<...EnableFp32DestAcc=true> requires kernel built with FP32_DEST_ACC_EN.");
    // D6 carry: expose member name EnableFp32DestAcc for the SFINAE fold probe.
    static constexpr bool EnableFp32DestAcc = EnableFp32DestAccV;
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

    /// Runtime output-tile index for Pinned / Absolute modes.
    uint32_t output_tile_idx = 0;

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
          bool EnableFp32DestAccV>
struct PackTileBlock : PackTileTag {
    static_assert(NTiles >= 1 && NTiles <= DEST_AUTO_LIMIT,
                  "PackTileBlock: NTiles must be in [1, DEST_AUTO_LIMIT]");
    static_assert(to_u32(FirstSlot) + NTiles <= DEST_AUTO_LIMIT,
                  "PackTileBlock: FirstSlot + NTiles exceeds DEST_AUTO_LIMIT (consecutive slots required)");
    static_assert(!EnableFp32DestAccV || DST_ACCUM_MODE,
                  "PackTileBlock<...EnableFp32DestAcc=true> requires kernel built with FP32_DEST_ACC_EN.");
    static constexpr bool EnableFp32DestAcc = EnableFp32DestAccV;

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
          CbIndexMode Index,
          Dst DstSlot,
          bool EnableFp32DestAccV>
struct BinaryFpu : BinaryFpuTag {
    static_assert(to_u32(DstSlot) < DEST_AUTO_LIMIT,
                  "BinaryFpu: DEST slot exceeds DEST_AUTO_LIMIT");
    // BinaryFpu reads each operand once per tile — index mode constrained like CopyTile.
    // The OR over per-side policies guards both A and B in the collapsed-Index world
    // (Q4 v6 collapse).
    static_assert(!((APolicy == CopyTilePolicy::WaitAndPop || BPolicy == CopyTilePolicy::WaitAndPop)
                    && Index == CbIndexMode::BlockIter),
                  "BinaryFpu: BlockIter index requires Upfront / NoWaitNoPop policy on both sides");
    // D6: EnableFp32DestAcc requires the kernel was built with FP32_DEST_ACC_EN.
    static_assert(!EnableFp32DestAccV || DST_ACCUM_MODE,
                  "BinaryFpu<...EnableFp32DestAcc=true> requires kernel built with FP32_DEST_ACC_EN "
                  "(DST_ACCUM_MODE must be 1).");
    // D6 carry: expose member name EnableFp32DestAcc for the SFINAE fold probe.
    static constexpr bool EnableFp32DestAcc = EnableFp32DestAccV;

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

    /// Runtime indices for Pinned / Absolute modes.
    uint32_t a_tile_idx = 0;
    uint32_t b_tile_idx = 0;

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
            // Reg A fix: replace init_bcast<>() (full HW configure mid-MAIN) with short-init
            // forms: llk_math_eltwise_binary_init + llk_unpack_AB_init. No hw_configure,
            // no pack_dest_init, no math_pack_sync_init — preserves D8 invariant
            // (BIG init only at compute_kernel_hw_startup boot, never per-tile mid-MAIN).
            constexpr auto bt = static_cast<ckernel::BroadcastType>(static_cast<uint8_t>(Bcast));
            constexpr auto et = (Op == BinaryFpuOp::Add) ? ckernel::EltwiseBinaryType::ELWADD :
                                (Op == BinaryFpuOp::Sub) ? ckernel::EltwiseBinaryType::ELWSUB :
                                                           ckernel::EltwiseBinaryType::ELWMUL;
            if constexpr (Op == BinaryFpuOp::Mul) {
                MATH((llk_math_eltwise_binary_init<et, bt, MATH_FIDELITY>()));
            } else {
                MATH((llk_math_eltwise_binary_init<et, bt, MathFidelity::LoFi>()));
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
        // Q4 v6 collapse: single Index mode drives both A-side and B-side index
        // computation. Per-side runtime tile index member fields (a_tile_idx,
        // b_tile_idx) remain independent — only the MODE template collapsed.
        const auto idx_for = [&](uint32_t pinned_val) -> uint32_t {
            if constexpr (Index == CbIndexMode::FirstTile) return 0;
            else if constexpr (Index == CbIndexMode::BlockIter) return i;
            else return pinned_val;  // Pinned / Absolute
        };
        const uint32_t a_idx = idx_for(a_tile_idx);
        const uint32_t b_idx = idx_for(b_tile_idx);
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
          bool EnableFp32DestAccV>
struct DestReuseBinary : DestReuseBinaryTag {
    static_assert(to_u32(DstIn) < DEST_AUTO_LIMIT && to_u32(DstOut) < DEST_AUTO_LIMIT,
                  "DestReuseBinary: DEST slot exceeds DEST_AUTO_LIMIT");
    static_assert(!(Policy == CopyTilePolicy::WaitAndPop && IndexMode == CbIndexMode::BlockIter),
                  "DestReuseBinary: BlockIter index requires Upfront / NoWaitNoPop policy");
    static_assert(!EnableFp32DestAccV || DST_ACCUM_MODE,
                  "DestReuseBinary<...EnableFp32DestAcc=true> requires kernel built with FP32_DEST_ACC_EN.");
    static constexpr bool EnableFp32DestAcc = EnableFp32DestAccV;

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

    uint32_t cb_tile_idx = 0;

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
          bool EnableFp32DestAccV>
struct UnaryBcast : UnaryBcastTag {
    static_assert(to_u32(DstSlot) < DEST_AUTO_LIMIT,
                  "UnaryBcast: DEST slot exceeds DEST_AUTO_LIMIT");
    static_assert(!EnableFp32DestAccV || DST_ACCUM_MODE,
                  "UnaryBcast<...EnableFp32DestAcc=true> requires kernel built with FP32_DEST_ACC_EN.");
    static constexpr bool EnableFp32DestAcc = EnableFp32DestAccV;

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

// =============================================================================
// emit_pre_element_transitions<E, I, Es...>() (D2 + D6)
//
// For element at position I in pack Es..., emit:
//   1. fp32-dest-acc transition (if curr_fp32 != prev_fp32)
//   2. srca / srcb / pack reconfig (each compile-time-elided when prev_*_cb == curr_*_cb)
//
// Compile-time elision means a chain whose elements all share a CB on a side emits
// the reconfig once (at element 0, where prev == NO_PREV_CB) and never again on that
// side. Run-time cost: zero — `if constexpr` resolves at compile time.
//
// D6 transition fold: the fp32 fold walks Es[0..I-1] threading SKIP elements through
// transparently and anchoring on CARRY-list elements. CARRY elements that don't
// declare `EnableFp32DestAcc` (yet — D6 lands in commit 5) are treated as SKIP via
// the SFINAE probe, so this commit's fold is a structural no-op for fp32 transitions
// until D6 elements ship.
// =============================================================================

template <class E, std::size_t I, class... Es>
ALWI void emit_pre_element_transitions() {
    // ---- D6 fp32 transition ----
    constexpr bool prev_fp32 = prev_fp32_dest_acc_for_idx<I, Es...>();
    constexpr bool curr_fp32 = fp32_or_default<E, prev_fp32>::value;
    if constexpr (curr_fp32 != prev_fp32) {
        if constexpr (curr_fp32) { enable_fp32_dest_acc(); }
        else                     { disable_fp32_dest_acc(); }
    }

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
// `EmitInit`: when true, the per-tile path re-fires `E::init()` before exec (clash
// chains). When false, the chain hoisted init() to boot — per-tile path skips it.
template <bool EmitInit, class E>
ALWI void elem_compute_exec(const E& e, uint32_t i) {
    if constexpr (is_pack_tile_op_v<E>) {
        // Pack runs in the pack phase, not compute. Skip here.
        (void)e; (void)i;
    } else if constexpr (has_member_exec_v<E>) {
        // CB-bound element, Fill/Rand, OR runtime-param SFPU op (UnaryNe etc.) —
        // dispatch via member exec(i) so runtime payload (param0, value, idx, ...) is captured.
        if constexpr (EmitInit) E::init();
        e.exec(i);
    } else if constexpr (is_dest_only_op_v<E>) {
        // Pure SFPU op via CRTP base — static init() + static exec().
        if constexpr (EmitInit) E::init();
        E::exec();
    } else {
        // Fallback (shouldn't reach for well-formed chain elements).
        if constexpr (EmitInit) E::init();
        (void)e; (void)i;
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
        // Skip Pack elements (their reconfig is fold-driven and pack init() is
        // already a no-op post-commit-2; pack-side transitions are handled by the
        // boot pack_init pass and the per-element fold below covers any non-pack
        // pack-side reconfig ordering).
        if constexpr (!is_pack_tile_op_v<ElemT>) {
            emit_pre_element_transitions<ElemT, II, Es...>();
            ElemT::init();
        }
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

    // ---- F-PERF-1: per-tile init gate on FPU clash ----
    //
    // Chains without FPU clash (pure CopyTile + SFPU + PackTile) only need init()
    // emitted once at boot — subsequent tiles re-use the programmed state. Chains
    // with FPU clash (BinaryFpu, DestReuseBinary, UnaryBcast) need per-tile re-init
    // because the FPU op overwrites unpacker state that copy_tile / SFPU programs.
    constexpr bool has_clash = chain_has_non_copy_tile_fpu_clash_v<Chain>;
    constexpr bool emit_init_per_tile = has_clash;

    using IdxSeq = std::make_index_sequence<sizeof...(Es)>;

    // ---- F-PERF-4: hoist pack init out of per-tile loop ----
    // Pack reconfig is fold-driven (D2) and idempotent — emit once at chain entry.
    (detail::elem_pack_init<Es>(), ...);

    if constexpr (!emit_init_per_tile) {
        // Non-clash chain: hoist all per-element transitions + init() to boot.
        // The per-tile loop emits only the lifecycle + exec.
        detail::hoisted_init_for_each(IdxSeq{}, elts...);
    }

    if constexpr (block_path) {
        // Upfront block path: wait/reserve `n_tiles` worth of CB tiles upfront,
        // run the per-tile loop, pop/push at end. Element types with block-size
        // multipliers (e.g. BlockCopyTile<Cb, BlockSize>) scale `n_tiles` internally.
        (detail::elem_wait_upfront(elts, n_tiles), ...);
        (detail::elem_reserve_upfront(elts, n_tiles), ...);

        for (uint32_t i = 0; i < n_tiles; ++i) {
            tile_regs_acquire();
            // Compute-phase: emit pre-element transitions (when init is per-tile)
            // and per-element exec.
            detail::compute_phase_for_each<emit_init_per_tile, emit_init_per_tile>(IdxSeq{}, i, elts...);
            tile_regs_commit();
            tile_regs_wait();
            (detail::elem_pack_exec(elts, i), ...);
            tile_regs_release();
        }

        (detail::elem_pop_upfront_end(elts, n_tiles), ...);
        (detail::elem_push_at_end(elts, n_tiles), ...);
    } else {
        // Per-tile path (default — PerTile policies).
        for (uint32_t i = 0; i < n_tiles; ++i) {
            (detail::elem_wait_per_tile(elts, i), ...);
            (detail::elem_reserve_per_tile(elts, i), ...);

            tile_regs_acquire();
            detail::compute_phase_for_each<emit_init_per_tile, emit_init_per_tile>(IdxSeq{}, i, elts...);
            tile_regs_commit();
            tile_regs_wait();
            (detail::elem_pack_exec(elts, i), ...);
            tile_regs_release();

            (detail::elem_pop_per_tile(elts, i), ...);
            (detail::elem_push_per_tile(elts, i), ...);
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
    if constexpr (is_cb_reader_op_v<E>) return E::cb_a_id();
    else                                return 0u;
}

template <class E>
constexpr uint32_t cb_b_of() {
    if constexpr (is_binary_fpu_op_v<E> || is_dest_reuse_binary_op_v<E>) return E::cb_b_id();
    else                                                                  return 0u;
}

template <class E>
constexpr uint32_t pack_cb_of() {
    if constexpr (is_pack_tile_op_v<E>) return E::pack_cb_id();
    else                                return 0u;
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
