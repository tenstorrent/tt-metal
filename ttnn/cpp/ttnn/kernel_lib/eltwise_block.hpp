// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_block.hpp
 * @brief Block-mode chain elements for multi-tile DEST scratch patterns.
 *
 * Production kernels frequently use a "process N tiles per acquire/release window"
 * pattern (binary_ng, eltwise_binary_*, ternary_*_bcast, etc.):
 *
 *   for (uint32_t i = 0; i < N; ++i) {
 *       add_tiles(cb_a, cb_b, i, i, i);   // N FPU ops, N consecutive DEST slots
 *   }
 *   // commit/wait
 *   for (uint32_t i = 0; i < N; ++i) {
 *       pack_tile(i, cb_out);             // N packs from D0..DN-1
 *   }
 *
 * The streaming chain in `eltwise_chain.hpp` is one-DEST-slot-per-iteration.
 * Block-mode chain elements pack N operations into a single chain iteration:
 *
 *   eltwise_chain(num_blocks,
 *       BlockCopyTile<cb_a, N>{},                  // N copies into D0..DN-1
 *       BlockBinaryFpu<cb_a, cb_b, Op, N>{},       // N FPU ops, ditto
 *       BlockPackTile<cb_out, N>{});               // N packs
 *
 * Each block element waits/pops `N * num_blocks` lifecycle (N tiles per outer iter),
 * acquires DEST once per outer iter, runs N inner ops, commits, packs N, releases.
 *
 * Element kinds:
 *  - `BlockCopyTile<Cb, BlockSize, BaseDst, Policy, Reconfig>` — N CB → DEST loads.
 *  - `BlockBinaryFpu<CbA, CbB, Op, BlockSize, BaseDst, ...>` — N FPU binary ops.
 *  - `BlockPackTile<Cb, BlockSize, BaseDst, Policy, Reconfig>` — N DEST → CB packs.
 *
 * The streaming chain pipeline already supports these — they just present a different
 * `wait_per_tile` / `pop_per_tile` / `exec` shape (waits N, pops N, runs N inner ops).
 */

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace compute_kernel_lib {

// =============================================================================
// BlockCopyTile — N CB → DEST loads in one acquire/release window.
// =============================================================================

template <
    uint32_t Cb,
    uint32_t BlockSize,
    Dst BaseDst = Dst::D0,
    CopyTilePolicy Policy = CopyTilePolicy::WaitAndPop,
    CopyTileReconfig Reconfig = CopyTileReconfig::None>
struct BlockCopyTile : CopyTileTag {
    static_assert(to_u32(BaseDst) + BlockSize <= DEST_AUTO_LIMIT,
                  "BlockCopyTile: BaseDst + BlockSize exceeds DEST_AUTO_LIMIT");
    static_assert(BlockSize >= 1, "BlockCopyTile: BlockSize must be >= 1");

    static constexpr uint32_t       cb              = Cb;
    static constexpr uint32_t       cb_a_id()       { return Cb; }
    static constexpr uint32_t       cb_b_id()       { return 0;  }
    static constexpr Dst            dst_slot        = BaseDst;
    static constexpr CopyTilePolicy a_policy()      { return Policy; }
    static constexpr CopyTilePolicy b_policy()      { return CopyTilePolicy::NoWaitNoPop; }
    static constexpr bool           is_upfront      = (Policy == CopyTilePolicy::WaitUpfrontPopAtEnd);
    static constexpr bool           clashes_with_fpu= true;
    static constexpr uint32_t       block_size      = BlockSize;

    // Prev-CB fold (D7): BlockCopyTile loads Cb to srca only.
    static constexpr uint32_t reconfig_srca_cb = (Reconfig == CopyTileReconfig::Input) ? Cb : NO_PREV_CB;
    static constexpr uint32_t reconfig_srcb_cb = NO_PREV_CB;
    static constexpr uint32_t reconfig_pack_cb = NO_PREV_CB;

    // F-PERF-3 (D7): srca reconfig is fold-driven via the chain's
    // `emit_pre_element_transitions`. init() programs only `copy_tile_init`
    // — `copy_tile_to_dst_init_short_with_dt` is no longer needed because the
    // fold emits the equivalent `reconfig_data_format_srca(curr) + copy_tile_to_dst_init_short(curr)`
    // sequence at chain-derived prev-CB boundaries. For chains that didn't opt
    // into reconfig (Reconfig == None) we keep the simple `copy_tile_init`.
    static ALWI void init() { copy_tile_init(Cb); }

    ALWI void wait_per_tile(uint32_t /*i*/) const {
        if constexpr (Policy == CopyTilePolicy::WaitAndPop || Policy == CopyTilePolicy::WaitNoPop) {
            cb_wait_front(Cb, BlockSize);
        }
    }
    ALWI void wait_upfront(uint32_t n) const {
        if constexpr (Policy == CopyTilePolicy::WaitUpfrontPopAtEnd) cb_wait_front(Cb, n * BlockSize);
    }

    ALWI void exec(uint32_t /*i*/) const {
        for (uint32_t j = 0; j < BlockSize; ++j) {
            copy_tile(Cb, j, to_u32(BaseDst) + j);
        }
    }

    ALWI void pop_per_tile(uint32_t /*i*/) const {
        if constexpr (Policy == CopyTilePolicy::WaitAndPop || Policy == CopyTilePolicy::NoWaitPop) {
            cb_pop_front(Cb, BlockSize);
        }
    }
    ALWI void pop_upfront_end(uint32_t n) const {
        if constexpr (Policy == CopyTilePolicy::WaitUpfrontPopAtEnd) cb_pop_front(Cb, n * BlockSize);
    }
};

// =============================================================================
// BlockBinaryFpu — N FPU binary ops in one acquire/release window.
// =============================================================================

template <
    uint32_t CbA,
    uint32_t CbB,
    BinaryFpuOp Op,
    uint32_t BlockSize,
    uint32_t CbOut = 0,
    Dst BaseDst = Dst::D0,
    BroadcastDim Bcast = BroadcastDim::None,
    BinaryDataFormatReconfig DF = BinaryDataFormatReconfig::None,
    CopyTilePolicy APolicy = CopyTilePolicy::WaitAndPop,
    CopyTilePolicy BPolicy = CopyTilePolicy::WaitAndPop,
    CbIndexMode Index = CbIndexMode::BlockIter,
    uint32_t BWaitTiles = 0,  // override for B wait count when Index == FirstTile (e.g. 1 for scalar)
    bool EnableFp32DestAccV = false>
struct BlockBinaryFpu : BinaryFpuTag {
    static_assert(to_u32(BaseDst) + BlockSize <= DEST_AUTO_LIMIT,
                  "BlockBinaryFpu: BaseDst + BlockSize exceeds DEST_AUTO_LIMIT");

    static constexpr uint32_t        cb_a_id()         { return CbA; }
    static constexpr uint32_t        cb_b_id()         { return CbB; }
    static constexpr CopyTilePolicy  a_policy()        { return APolicy; }
    static constexpr CopyTilePolicy  b_policy()        { return BPolicy; }
    static constexpr Dst             dst_slot          = BaseDst;
    static constexpr bool            is_upfront        = (APolicy == CopyTilePolicy::WaitUpfrontPopAtEnd) ||
                                                         (BPolicy == CopyTilePolicy::WaitUpfrontPopAtEnd);
    static constexpr bool            clashes_with_fpu  = true;
    static constexpr bool            same_cb           = (CbA == CbB);
    static constexpr uint32_t        block_size        = BlockSize;

    // Prev-CB fold (D7): BlockBinaryFpu touches srca (CbA), srcb (CbB), and pack
    // (CbOut) when the corresponding reconfig is opted in. Reconfig is fold-driven
    // — `emit_pre_element_transitions` emits the elided sequence before init().
    static constexpr uint32_t reconfig_srca_cb =
        (DF == BinaryDataFormatReconfig::Input || DF == BinaryDataFormatReconfig::InputAndOutput) ? CbA : NO_PREV_CB;
    static constexpr uint32_t reconfig_srcb_cb =
        (DF == BinaryDataFormatReconfig::Input || DF == BinaryDataFormatReconfig::InputAndOutput) ? CbB : NO_PREV_CB;
    static constexpr uint32_t reconfig_pack_cb =
        ((DF == BinaryDataFormatReconfig::Output || DF == BinaryDataFormatReconfig::InputAndOutput) && CbOut != 0)
            ? CbOut
            : NO_PREV_CB;

    // F-PERF-3 (D7): srca / srcb / pack reconfig are fold-driven; init() programs
    // only the per-op LLK shape.
    static ALWI void init() {
        if constexpr (Bcast == BroadcastDim::None) {
            if constexpr      (Op == BinaryFpuOp::Add) add_tiles_init(CbA, CbB);
            else if constexpr (Op == BinaryFpuOp::Sub) sub_tiles_init(CbA, CbB);
            else                                       mul_tiles_init(CbA, CbB);
        } else {
            constexpr auto bt = static_cast<ckernel::BroadcastType>(static_cast<uint8_t>(Bcast));
            constexpr auto et = (Op == BinaryFpuOp::Add) ? ckernel::EltwiseBinaryType::ELWADD :
                                (Op == BinaryFpuOp::Sub) ? ckernel::EltwiseBinaryType::ELWSUB :
                                                           ckernel::EltwiseBinaryType::ELWMUL;
            constexpr uint32_t ocb = (CbOut != 0) ? CbOut : CbA;
            init_bcast<et, bt>(CbA, CbB, ocb);
        }
    }

    // D6 static_assert (CARRY) + member exposure for fold SFINAE probe.
    static_assert(
        !EnableFp32DestAccV || DST_ACCUM_MODE,
        "BlockBinaryFpu<...EnableFp32DestAcc=true> requires kernel built with FP32_DEST_ACC_EN "
        "(DST_ACCUM_MODE must be 1).");
    static constexpr bool EnableFp32DestAcc = EnableFp32DestAccV;

    // Q4 v6 collapse: b_wait_count keys on the single Index template.
    static constexpr uint32_t b_wait_count =
        (Index == CbIndexMode::FirstTile) ? (BWaitTiles == 0 ? 1u : BWaitTiles) : BlockSize;

    ALWI void wait_per_tile(uint32_t /*i*/) const {
        if constexpr (APolicy == CopyTilePolicy::WaitAndPop || APolicy == CopyTilePolicy::WaitNoPop) {
            cb_wait_front(CbA, BlockSize);
        }
        if constexpr (!same_cb && (BPolicy == CopyTilePolicy::WaitAndPop || BPolicy == CopyTilePolicy::WaitNoPop)) {
            cb_wait_front(CbB, b_wait_count);
        }
    }
    ALWI void wait_upfront(uint32_t n) const {
        if constexpr (APolicy == CopyTilePolicy::WaitUpfrontPopAtEnd) cb_wait_front(CbA, n * BlockSize);
        if constexpr (!same_cb && BPolicy == CopyTilePolicy::WaitUpfrontPopAtEnd) cb_wait_front(CbB, n * BlockSize);
    }

    ALWI void exec(uint32_t /*i*/) const {
        for (uint32_t j = 0; j < BlockSize; ++j) {
            const uint32_t dst = to_u32(BaseDst) + j;
            // Q4 v6 collapse: single Index drives both sides. (Per-side runtime
            // tile-index member fields aren't a feature on BlockBinaryFpu — block
            // walks always start at j=0.)
            const uint32_t a_idx = (Index == CbIndexMode::BlockIter) ? j : 0u;
            const uint32_t b_idx = (Index == CbIndexMode::BlockIter) ? j : 0u;
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
    }

    ALWI void pop_per_tile(uint32_t /*i*/) const {
        if constexpr (APolicy == CopyTilePolicy::WaitAndPop || APolicy == CopyTilePolicy::NoWaitPop) {
            cb_pop_front(CbA, BlockSize);
        }
        if constexpr (!same_cb && (BPolicy == CopyTilePolicy::WaitAndPop || BPolicy == CopyTilePolicy::NoWaitPop)) {
            cb_pop_front(CbB, b_wait_count);
        }
    }
    ALWI void pop_upfront_end(uint32_t n) const {
        if constexpr (APolicy == CopyTilePolicy::WaitUpfrontPopAtEnd) cb_pop_front(CbA, n * BlockSize);
        if constexpr (!same_cb && BPolicy == CopyTilePolicy::WaitUpfrontPopAtEnd) cb_pop_front(CbB, n * BlockSize);
    }
};

// =============================================================================
// BlockPackTile — N DEST → CB packs in one acquire/release window.
// =============================================================================

template <
    uint32_t Cb,
    uint32_t BlockSize,
    Dst BaseDst = Dst::D0,
    PackTilePolicy Policy = PackTilePolicy::PerTileReserveAndPush,
    PackTileReconfig Reconfig = PackTileReconfig::None,
    bool EnableFp32DestAccV = false>
struct BlockPackTile : PackTileTag {
    static_assert(to_u32(BaseDst) + BlockSize <= DEST_AUTO_LIMIT,
                  "BlockPackTile: BaseDst + BlockSize exceeds DEST_AUTO_LIMIT");
    static_assert(
        !EnableFp32DestAccV || DST_ACCUM_MODE,
        "BlockPackTile<...EnableFp32DestAcc=true> requires kernel built with FP32_DEST_ACC_EN.");
    static constexpr bool EnableFp32DestAcc = EnableFp32DestAccV;

    static constexpr uint32_t       cb                  = Cb;
    static constexpr uint32_t       pack_cb_id()        { return Cb; }
    static constexpr Dst            pack_dst_slot       = BaseDst;
    static constexpr bool           is_upfront          = (Policy == PackTilePolicy::UpfrontReservePushAtEnd);
    static constexpr uint32_t       block_size          = BlockSize;

    // Prev-CB fold (D7): BlockPackTile writes pack-side; mark Cb under reconfig
    // only when the user opted in. The fold then handles single-arg vs two-arg
    // pack_reconfig_data_format dispatch based on the chain-derived prev_pack_cb.
    static constexpr uint32_t reconfig_srca_cb = NO_PREV_CB;
    static constexpr uint32_t reconfig_srcb_cb = NO_PREV_CB;
    static constexpr uint32_t reconfig_pack_cb =
        (Reconfig == PackTileReconfig::Output || Reconfig == PackTileReconfig::OutputConditional) ? Cb : NO_PREV_CB;

    // F-PERF-3 (D7): pack reconfig is fold-driven; init() is a no-op.
    static ALWI void init() {
        // Pack reconfig fold-driven via `emit_pre_element_transitions`.
    }

    ALWI void reserve_per_tile(uint32_t /*i*/) const {
        if constexpr (Policy == PackTilePolicy::PerTileReserveAndPush ||
                      Policy == PackTilePolicy::PerTileReserveNoPush) {
            cb_reserve_back(Cb, BlockSize);
        }
    }
    ALWI void reserve_upfront(uint32_t n) const {
        if constexpr (Policy == PackTilePolicy::UpfrontReservePushAtEnd) cb_reserve_back(Cb, n * BlockSize);
    }

    ALWI void exec(uint32_t /*i*/) const {
        for (uint32_t j = 0; j < BlockSize; ++j) {
            pack_tile(to_u32(BaseDst) + j, Cb, j);
        }
    }

    ALWI void push_per_tile(uint32_t /*i*/) const {
        if constexpr (Policy == PackTilePolicy::PerTileReserveAndPush) {
            cb_push_back(Cb, BlockSize);
        }
    }
    ALWI void push_at_end(uint32_t n) const {
        if constexpr (Policy == PackTilePolicy::NoReservePushAtEnd ||
                      Policy == PackTilePolicy::UpfrontReservePushAtEnd) {
            cb_push_back(Cb, n * BlockSize);
        }
    }
};

}  // namespace compute_kernel_lib
