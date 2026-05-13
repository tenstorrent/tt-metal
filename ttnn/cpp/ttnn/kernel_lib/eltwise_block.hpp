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
 *  - `BlockBinaryFpu<CbA, CbB, Op, BlockSize, ...>` — N FPU binary ops.
 *  - `BlockPackTile<Cb, BlockSize, BaseDst, Policy, Reconfig>` — N DEST → CB packs.
 *
 * The streaming chain pipeline already supports these — they just present a different
 * `wait_per_tile` / `pop_per_tile` / `exec` shape (waits N, pops N, runs N inner ops).
 *
 * @section block_path_fold Compile-time prev-CB / prev-fp32 fold (D7)
 *
 * Block elements expose the same `reconfig_srca_cb` / `reconfig_srcb_cb` /
 * `reconfig_pack_cb` static accessors as streaming elements (post-commit-3). The
 * chain's `emit_pre_element_transitions<E, I, Es...>()` walks the element pack at
 * compile time and emits per-element reconfig + fp32 transitions ahead of each
 * element's `init()`. Reconfig is compile-time-elided when the running prev value
 * equals the element's required value.
 *
 * Block-element `init()` bodies no longer emit reconfig — they program only the
 * per-op LLK shape (`add_tiles_init`, math+unpack bcast short init, `copy_tile_init`).
 *
 * @section block_path_fp32 FP32 DEST accumulation
 *
 * Block elements inherit the kernel's build-time DST_ACCUM_MODE (from FP32_DEST_ACC_EN).
 * No per-element opt-in; no mid-kernel enable/disable. DEST_AUTO_LIMIT already accounts
 * for the halved slot count under fp32 mode.
 *
 * @section block_caller_init_contract Caller-init contract — reminder
 *
 * Block-using kernels follow the same caller-init contract as streaming chains. See
 * `eltwise_chain.hpp` `@section caller_init_contract` for the table. `BlockBinaryFpu::init()`
 * uses the math+unpack short bcast init (mirrors streaming `BinaryFpu`); pack-side
 * configure stays as `compute_kernel_hw_startup` programmed it.
 *
 * @section block_asymmetric_bcast Asymmetric A=BlockIter / B=FirstTile bcast walk
 *
 * `BlockBinaryFpu` exposes per-side `AIndex` / `BIndex` template parameters. The
 * canonical broadcast pattern walks the streamed operand while pinning the scaler/vector
 * operand at tile 0 — `AIndex=BlockIter, BIndex=FirstTile` (the default `BIndex=AIndex`
 * preserves back-compat with single-Index callers):
 *
 *   // softmax phase 2a (stable): out[t] = exp(in[t] - max), max pinned at tile 0
 *   using SubBcast = BlockBinaryFpu<
 *       cb_input_tiles, cb_max, BinaryFpuOp::Sub, stripe_tiles,
 *       Dst::D0, BroadcastDim::COL, BinaryDataFormatReconfig::None,
 *       CopyTilePolicy::WaitUpfrontPopAtEnd,   // A: wait N upfront, pop at end
 *       CopyTilePolicy::WaitNoPop,             // B: wait 1, never pop (caller pops)
 *       CbIndexMode::BlockIter,                // AIndex — A walks 0..stripe_tiles-1
 *       0,                                     // BWaitTiles (deprecated)
 *       CbIndexMode::FirstTile>;               // BIndex — B pinned at tile 0
 *
 *   eltwise_chain(num_stripes, SubBcast{}, Exp<>{}, BlockPackTile<cb_exps, stripe_tiles>{});
 *
 * Wait counts auto-derive: `a_wait_count = BlockSize` when AIndex=BlockIter,
 * `b_wait_count = 1` when BIndex=FirstTile.
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
    static_assert(
        to_u32(BaseDst) + BlockSize <= DEST_AUTO_LIMIT, "BlockCopyTile: BaseDst + BlockSize exceeds DEST_AUTO_LIMIT");
    static_assert(BlockSize >= 1, "BlockCopyTile: BlockSize must be >= 1");

    static constexpr uint32_t cb = Cb;
    static constexpr uint32_t cb_a_id() { return Cb; }
    static constexpr uint32_t cb_b_id() { return 0; }
    static constexpr Dst dst_slot = BaseDst;
    static constexpr CopyTilePolicy a_policy() { return Policy; }
    static constexpr CopyTilePolicy b_policy() { return CopyTilePolicy::NoWaitNoPop; }
    static constexpr bool is_upfront = (Policy == CopyTilePolicy::WaitUpfrontPopAtEnd);
    static constexpr bool clashes_with_fpu = true;
    static constexpr uint32_t block_size = BlockSize;

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
        if constexpr (Policy == CopyTilePolicy::WaitUpfrontPopAtEnd) {
            cb_wait_front(Cb, n * BlockSize);
        }
    }

    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const {
        for (uint32_t j = 0; j < BlockSize; ++j) {
            copy_tile(Cb, j, to_u32(BaseDst) + j + slot_offset);
        }
    }

    static constexpr uint32_t lane_width = to_u32(BaseDst) + BlockSize;

    ALWI void pop_per_tile(uint32_t /*i*/) const {
        if constexpr (Policy == CopyTilePolicy::WaitAndPop || Policy == CopyTilePolicy::NoWaitPop) {
            cb_pop_front(Cb, BlockSize);
        }
    }
    ALWI void pop_upfront_end(uint32_t n) const {
        if constexpr (Policy == CopyTilePolicy::WaitUpfrontPopAtEnd) {
            cb_pop_front(Cb, n * BlockSize);
        }
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
    Dst BaseDst = Dst::D0,
    BroadcastDim Bcast = BroadcastDim::None,
    BinaryDataFormatReconfig DF = BinaryDataFormatReconfig::None,
    CopyTilePolicy APolicy = CopyTilePolicy::WaitAndPop,
    CopyTilePolicy BPolicy = CopyTilePolicy::WaitAndPop,
    CbIndexMode AIndex = CbIndexMode::BlockIter,
    uint32_t BWaitTiles = 0,  // deprecated B-wait override; prefer BIndex=FirstTile (kept for back-compat)
    CbIndexMode BIndex = AIndex>
struct BlockBinaryFpu : BinaryFpuTag {
    static_assert(
        to_u32(BaseDst) + BlockSize <= DEST_AUTO_LIMIT, "BlockBinaryFpu: BaseDst + BlockSize exceeds DEST_AUTO_LIMIT");
    static_assert(
        DF != BinaryDataFormatReconfig::Output && DF != BinaryDataFormatReconfig::InputAndOutput,
        "BlockBinaryFpu: pack-side DF reconfig must live on the BlockPackTile element "
        "(PackTileReconfig::Output), not on BlockBinaryFpu. Only None / Input are valid here.");
    // same_cb dedup safety: when CbA == CbB the B-side wait/pop is skipped, so the
    // helper would under-wait if A and B walked different ranges of the shared CB.
    static_assert(
        (CbA != CbB) || AIndex == BIndex,
        "BlockBinaryFpu: when CbA == CbB, AIndex and BIndex must match "
        "(B-side wait/pop is deduped — asymmetric indices would under-wait).");

    static constexpr uint32_t cb_a_id() { return CbA; }
    static constexpr uint32_t cb_b_id() { return CbB; }
    static constexpr CopyTilePolicy a_policy() { return APolicy; }
    static constexpr CopyTilePolicy b_policy() { return BPolicy; }
    static constexpr Dst dst_slot = BaseDst;
    static constexpr bool is_upfront =
        (APolicy == CopyTilePolicy::WaitUpfrontPopAtEnd) || (BPolicy == CopyTilePolicy::WaitUpfrontPopAtEnd);
    static constexpr bool clashes_with_fpu = true;
    static constexpr bool same_cb = (CbA == CbB);
    static constexpr uint32_t block_size = BlockSize;

    // Prev-CB fold (D7): BlockBinaryFpu touches srca (CbA) and srcb (CbB) only.
    // Pack-side reconfig is owned by BlockPackTile (PackTileReconfig::Output) — not
    // duplicated here. `emit_pre_element_transitions` emits the elided srca/srcb
    // sequence before init().
    static constexpr uint32_t reconfig_srca_cb = (DF == BinaryDataFormatReconfig::Input) ? CbA : NO_PREV_CB;
    static constexpr uint32_t reconfig_srcb_cb = (DF == BinaryDataFormatReconfig::Input) ? CbB : NO_PREV_CB;
    static constexpr uint32_t reconfig_pack_cb = NO_PREV_CB;

    // F-PERF-3 (D7): srca / srcb reconfig is fold-driven; init() programs only the
    // per-op LLK shape. Bcast uses `_init_short`-equivalent (math+unpack only) — the
    // original `init_bcast<>()` BIG init (hw_configure + pack_dest_init + sync_init)
    // is undefined mid-MAIN; pack-side configure was already programmed by
    // `compute_kernel_hw_startup`. Mirrors streaming `BinaryFpu` (eltwise_chain.inl).
    static ALWI void init() {
        if constexpr (Bcast == BroadcastDim::None) {
            if constexpr (Op == BinaryFpuOp::Add) {
                add_tiles_init(CbA, CbB);
            } else if constexpr (Op == BinaryFpuOp::Sub) {
                sub_tiles_init(CbA, CbB);
            } else {
                mul_tiles_init(CbA, CbB);
            }
        } else {
            constexpr auto bt = static_cast<ckernel::BroadcastType>(static_cast<uint8_t>(Bcast));
            constexpr auto et = (Op == BinaryFpuOp::Add)   ? ckernel::EltwiseBinaryType::ELWADD
                                : (Op == BinaryFpuOp::Sub) ? ckernel::EltwiseBinaryType::ELWSUB
                                                           : ckernel::EltwiseBinaryType::ELWMUL;
            if constexpr (Op == BinaryFpuOp::Mul) {
                MATH((llk_math_eltwise_binary_init_with_operands<et, bt, MATH_FIDELITY>(CbA, CbB)));
            } else {
                MATH((llk_math_eltwise_binary_init_with_operands<et, bt, MathFidelity::LoFi>(CbA, CbB)));
            }
            UNPACK((llk_unpack_AB_init<bt>(CbA, CbB)));
        }
    }

    // Per-side wait counts. BlockIter walks tiles 0..BlockSize-1 → wait BlockSize.
    // FirstTile pins at tile 0 → wait 1 (or BWaitTiles override for legacy callers
    // who passed an explicit B-wait count under the collapsed-Index API).
    static constexpr uint32_t a_wait_count = (AIndex == CbIndexMode::FirstTile) ? 1u : BlockSize;
    static constexpr uint32_t b_wait_count =
        (BIndex == CbIndexMode::FirstTile) ? (BWaitTiles == 0 ? 1u : BWaitTiles) : BlockSize;

    ALWI void wait_per_tile(uint32_t /*i*/) const {
        if constexpr (APolicy == CopyTilePolicy::WaitAndPop || APolicy == CopyTilePolicy::WaitNoPop) {
            cb_wait_front(CbA, a_wait_count);
        }
        if constexpr (!same_cb && (BPolicy == CopyTilePolicy::WaitAndPop || BPolicy == CopyTilePolicy::WaitNoPop)) {
            cb_wait_front(CbB, b_wait_count);
        }
    }
    ALWI void wait_upfront(uint32_t n) const {
        if constexpr (APolicy == CopyTilePolicy::WaitUpfrontPopAtEnd) {
            cb_wait_front(CbA, n * a_wait_count);
        }
        if constexpr (!same_cb && BPolicy == CopyTilePolicy::WaitUpfrontPopAtEnd) {
            cb_wait_front(CbB, n * b_wait_count);
        }
    }

    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const {
        for (uint32_t j = 0; j < BlockSize; ++j) {
            const uint32_t dst = to_u32(BaseDst) + j + slot_offset;
            const uint32_t a_idx = (AIndex == CbIndexMode::BlockIter) ? j : 0u;
            const uint32_t b_idx = (BIndex == CbIndexMode::BlockIter) ? j : 0u;
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
    }

    static constexpr uint32_t lane_width = to_u32(BaseDst) + BlockSize;

    ALWI void pop_per_tile(uint32_t /*i*/) const {
        if constexpr (APolicy == CopyTilePolicy::WaitAndPop || APolicy == CopyTilePolicy::NoWaitPop) {
            cb_pop_front(CbA, a_wait_count);
        }
        if constexpr (!same_cb && (BPolicy == CopyTilePolicy::WaitAndPop || BPolicy == CopyTilePolicy::NoWaitPop)) {
            cb_pop_front(CbB, b_wait_count);
        }
    }
    ALWI void pop_upfront_end(uint32_t n) const {
        if constexpr (APolicy == CopyTilePolicy::WaitUpfrontPopAtEnd) {
            cb_pop_front(CbA, n * a_wait_count);
        }
        if constexpr (!same_cb && BPolicy == CopyTilePolicy::WaitUpfrontPopAtEnd) {
            cb_pop_front(CbB, n * b_wait_count);
        }
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
    PackTileReconfig Reconfig = PackTileReconfig::None>
struct BlockPackTile : PackTileTag {
    static_assert(
        to_u32(BaseDst) + BlockSize <= DEST_AUTO_LIMIT, "BlockPackTile: BaseDst + BlockSize exceeds DEST_AUTO_LIMIT");

    static constexpr uint32_t cb = Cb;
    static constexpr uint32_t pack_cb_id() { return Cb; }
    static constexpr Dst pack_dst_slot = BaseDst;
    static constexpr bool is_upfront = (Policy == PackTilePolicy::UpfrontReservePushAtEnd);
    static constexpr uint32_t block_size = BlockSize;

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
        if constexpr (
            Policy == PackTilePolicy::PerTileReserveAndPush || Policy == PackTilePolicy::PerTileReserveNoPush) {
            cb_reserve_back(Cb, BlockSize);
        }
    }
    ALWI void reserve_upfront(uint32_t n) const {
        if constexpr (Policy == PackTilePolicy::UpfrontReservePushAtEnd) {
            cb_reserve_back(Cb, n * BlockSize);
        }
    }

    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const {
        for (uint32_t j = 0; j < BlockSize; ++j) {
            pack_tile(to_u32(BaseDst) + j + slot_offset, Cb, j);
        }
    }

    static constexpr uint32_t lane_width = to_u32(BaseDst) + BlockSize;

    ALWI void push_per_tile(uint32_t /*i*/) const {
        if constexpr (Policy == PackTilePolicy::PerTileReserveAndPush) {
            cb_push_back(Cb, BlockSize);
        }
    }
    ALWI void push_at_end(uint32_t n) const {
        if constexpr (
            Policy == PackTilePolicy::NoReservePushAtEnd || Policy == PackTilePolicy::UpfrontReservePushAtEnd) {
            cb_push_back(Cb, n * BlockSize);
        }
    }
};

}  // namespace compute_kernel_lib
