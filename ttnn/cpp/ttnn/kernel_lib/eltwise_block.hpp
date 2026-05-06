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

template <uint32_t Cb,
          uint32_t BlockSize,
          Dst BaseDst                = Dst::D0,
          CopyTilePolicy Policy      = CopyTilePolicy::WaitAndPop,
          CopyTileReconfig Reconfig  = CopyTileReconfig::None,
          uint32_t OldCb             = 0>
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

    static ALWI void init() {
        if constexpr (Reconfig == CopyTileReconfig::Input) {
            copy_tile_to_dst_init_short_with_dt(OldCb, Cb, /*transpose=*/0);
        } else {
            copy_tile_init(Cb);
        }
    }

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

template <uint32_t CbA,
          uint32_t CbB,
          BinaryFpuOp Op,
          uint32_t BlockSize,
          Dst BaseDst                  = Dst::D0,
          BroadcastDim Bcast           = BroadcastDim::None,
          BinaryDataFormatReconfig DF  = BinaryDataFormatReconfig::None,
          CopyTilePolicy APolicy       = CopyTilePolicy::WaitAndPop,
          CopyTilePolicy BPolicy       = CopyTilePolicy::WaitAndPop,
          CbIndexMode AIndex           = CbIndexMode::BlockIter,
          CbIndexMode BIndex           = CbIndexMode::BlockIter,
          uint32_t BWaitTiles          = 0,  // override for B wait count when BIndex == FirstTile (e.g. 1 for scalar)
          uint32_t OldCbA              = 0,
          uint32_t OldCbB              = 0,
          uint32_t OldCbOut            = 0,
          uint32_t CbOut               = 0>
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

    static ALWI void init() {
        if constexpr (DF == BinaryDataFormatReconfig::Input ||
                      DF == BinaryDataFormatReconfig::InputAndOutput) {
            if constexpr (OldCbA != 0 && OldCbA != CbA) reconfig_data_format_srca(OldCbA, CbA);
            else                                        reconfig_data_format_srca(CbA);
            if constexpr (OldCbB != 0 && OldCbB != CbB) reconfig_data_format_srcb(OldCbB, CbB);
            else                                        reconfig_data_format_srcb(CbB);
        }
        if constexpr ((DF == BinaryDataFormatReconfig::Output ||
                       DF == BinaryDataFormatReconfig::InputAndOutput) && CbOut != 0) {
            if constexpr (OldCbOut != 0 && OldCbOut != CbOut) {
                pack_reconfig_data_format(OldCbOut, CbOut);
            } else {
                pack_reconfig_data_format(CbOut);
            }
        }
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

    static constexpr uint32_t b_wait_count = (BIndex == CbIndexMode::FirstTile)
                                               ? (BWaitTiles == 0 ? 1u : BWaitTiles)
                                               : BlockSize;

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
            const uint32_t a_idx = (AIndex == CbIndexMode::BlockIter) ? j : 0u;
            const uint32_t b_idx = (BIndex == CbIndexMode::BlockIter) ? j : 0u;
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

template <uint32_t Cb,
          uint32_t BlockSize,
          Dst BaseDst                  = Dst::D0,
          PackTilePolicy Policy        = PackTilePolicy::PerTileReserveAndPush,
          PackTileReconfig Reconfig    = PackTileReconfig::None,
          uint32_t OldCb               = 0>
struct BlockPackTile : PackTileTag {
    static_assert(to_u32(BaseDst) + BlockSize <= DEST_AUTO_LIMIT,
                  "BlockPackTile: BaseDst + BlockSize exceeds DEST_AUTO_LIMIT");

    static constexpr uint32_t       cb                  = Cb;
    static constexpr uint32_t       pack_cb_id()        { return Cb; }
    static constexpr Dst            pack_dst_slot       = BaseDst;
    static constexpr bool           is_upfront          = (Policy == PackTilePolicy::UpfrontReservePushAtEnd);
    static constexpr uint32_t       block_size          = BlockSize;

    static ALWI void init() {
        if constexpr (Reconfig == PackTileReconfig::Output) {
            pack_reconfig_data_format(Cb);
        } else if constexpr (Reconfig == PackTileReconfig::OutputConditional) {
            pack_reconfig_data_format(OldCb, Cb);
        }
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
