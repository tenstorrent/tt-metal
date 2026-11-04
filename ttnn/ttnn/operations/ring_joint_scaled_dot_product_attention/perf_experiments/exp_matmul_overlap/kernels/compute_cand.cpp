// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// exp_matmul_overlap CANDIDATE (graduation variant = cand_split: -DEMO_EXP_GROUP -DEMO_BLOCKED_PACK -DEMO_SPLIT).
// The op's compute kernel with the key block's score phases STREAMED in q tile-rows and the softmax exp issued from
// the PACK thread (TRISC2), so the SFPU exp of row i runs while the MATH thread (TRISC1) does FPU work of other rows.
// bf16 DEST only; fp32 DEST keeps the op's helper body verbatim (stock exp_tile, different algorithm).
//
// RAW-LLK JUSTIFICATION (capability, not ergonomics). Bypassed helpers: matmul_block (Q.K^T, P.V),
// reduce<MAX/SUM, REDUCE_ROW>, binary_sfpu<BinaryMax>, eltwise_chain(Sub bcast col -> ScaledExp -> PackTile).
//  (1) The exp must run on TRISC2 between tile_regs_wait() and the pack of the SAME DEST half while TRISC1 fills the
//      other half: no helper can place an SFPU op on the pack thread (eltwise_chain runs every op on MATH).
//  (2) The overlap needs several different ops per DEST section packed to different CBs (copy + block max reduce +
//      row sum + P.V in one section -> m_blk, l, O) and Q.K^T / P rows written out of order into a held-wr_ptr score
//      block (pack_tile<true>, blocked pack width); helpers own one phase over one whole block, one output CB.
//  (3) The running max is folded into the row max by pre-loading m_acc (within-face transposed) into the max
//      reduce's DEST accumulator, replacing BinaryMax (an SFPU op on MATH that would race the pack-thread SFPU
//      stream's LREG/LOADMACRO state); the row max uses reduce_block_max_row (one MOP per row; bf16 operands only).
//  The exp itself is ScaledExp's instruction stream unchanged (bit-identical to exp_tile<false, true>), issued for a
//  whole DEST section (sbw tiles) with ONE STALLWAIT(SFPU, MATH) + set_dst_write_addr and ONE RWC reset per section
//  (each tile's stream advances the DEST RWC by exactly one tile). Without any SFPU-side stall the per-tile SETRWC
//  (a math-pipe instruction) races the next tile's SFPU loads: measured INCORRECT (cand_nostall).
//
// Per key block (R rows, skv valid key tiles, sbw = DEST-half column chunk), steps i = -2 .. R:
//   E0  exp chunk 0 of row i: sub (S_i - m_i) on MATH -> exp(s .) on PACK -> pack in place      [overlaps S2]
//   S2  m_blk(i+1) = max(m_acc, rowmax S_{i+1}); l(i-1) = rowsum P_{i-1}; O(i-1) = P_{i-1}.V     [MATH, FPU]
//   E1  exp chunk 1 of row i (EMO_SPLIT: a one-section row is exp'd in two halves)              [overlaps QK]
//   QK  S_{i+2} = Q_{i+2} . K^T (+ key-tail mask via dest-reuse add)                             [MATH, FPU]
// P rows are written in place; a credit-only token CB (cb_p_tok) orders "P row packed" (pack) before "P row
// unpacked" (P.V / rowsum). Score rows are made visible with push-and-hold-wr_ptr (the CB holds exactly one block).
// Output is bit-identical to the op's kernel (all regimes tested).

#include <cstdint>
#include <type_traits>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/reduce_custom.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/minmax.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ring_joint_sdpa_common.hpp"
#include "emo_scaled_exp.hpp"

using namespace compute_kernel_lib;

#ifndef EMO_ZONE_KB0_DEF
#define EMO_ZONE_KB0_DEF 8
#endif
constexpr uint32_t EMO_ZONE_KB0 = EMO_ZONE_KB0_DEF;
constexpr uint32_t EMO_ZONE_NKB = 16;

#ifdef EMO_SFPU_STALL
constexpr bool kExpStall = true;
#else
constexpr bool kExpStall = false;
#endif
#ifdef EMO_EXP_ON_MATH
constexpr bool kExpOnMath = true;
#else
constexpr bool kExpOnMath = false;
#endif
#ifdef EMO_STOCK_MAX
constexpr bool kStockMax = true;
#else
constexpr bool kStockMax = false;
#endif

#ifdef EMO_EXP_GROUP
constexpr bool kExpGroup = true;
#else
constexpr bool kExpGroup = false;
#endif
#ifdef EMO_GROUP_STALL_CFG
constexpr uint32_t kGroupStallCond = p_stall::TRISC_CFG;
#else
constexpr uint32_t kGroupStallCond = p_stall::MATH;
#endif
#ifdef EMO_NO_EXP
constexpr bool kNoExp = true;  // ABLATION ONLY (wrong output): the FPU/pack floor without the exp
#else
constexpr bool kNoExp = false;
#endif

struct ValidKeyTiles {
    uint32_t n;
    ALWI uint32_t operator()(uint32_t /*block*/, uint32_t /*block_k*/) const { return n; }
};

inline uint32_t largest_divisor_le(uint32_t n, uint32_t cap) {
    uint32_t w = n < cap ? n : cap;
    while (n % w) {
        --w;
    }
    return w;
}

inline uint32_t subblock_w(uint32_t n) { return largest_divisor_le(n, DEST_AUTO_LIMIT); }

// Push n tiles (visible to UNPACK) but keep the pack write pointer on the block base, so later pack_tile<true>
// indices stay relative to the block (the CB holds exactly one block).
ALWI void push_hold_wr_ptr(uint32_t cb, uint32_t n) {
    cb_push_back(cb, n);
    PACK(({
        auto& intf = get_local_cb_interface(cb);
        intf.fifo_wr_ptr -= n * intf.fifo_page_size;
        const uint32_t fifo_start = intf.fifo_limit - intf.fifo_size;
        if (intf.fifo_wr_ptr < fifo_start) {
            intf.fifo_wr_ptr += intf.fifo_size;
        }
    }));
}

#ifdef EMO_SPLIT
constexpr bool kSplit = true;
#else
constexpr bool kSplit = false;
#endif
#ifdef EMO_BLOCKED_PACK
constexpr bool kBlockedPack = true;
#else
constexpr bool kBlockedPack = false;
#endif
uint32_t g_pack_width = 1;  // PACK thread: the packer MOP's current tile count
// Re-program the pack MOP for w tiles per pack call (no addrmod / stride reconfig: same formats, MOP-only).
ALWI void set_pack_width(uint32_t cb, uint32_t w) {
    if constexpr (kBlockedPack) {
        PACK(({
            if (g_pack_width != w) {
                llk_pack_init<PackMode::Default, false, true, true>(cb, w);
                g_pack_width = w;
            }
        }));
    }
}
// Pack n contiguous DEST tiles (from dst0) to n contiguous tiles of cb at out-of-order index idx (relative to the
// reserved write pointer): one blocked pack call (kBlockedPack) or n single-tile calls.
ALWI void pack_row(uint32_t cb, uint32_t dst0, uint32_t idx, uint32_t n) {
    if constexpr (kBlockedPack) {
        set_pack_width(cb, n);
        pack_tile<true>(dst0, cb, idx);
    } else {
        for (uint32_t j = 0; j < n; ++j) {
            pack_tile<true>(dst0 + j, cb, idx + j);
        }
    }
}

// Unpacker source tracker: (srcA cb, srcB cb) currently configured; reconfig only on a format change.
struct Src {
    uint32_t a, b;
    ALWI void use(uint32_t na, uint32_t nb) {
        reconfig_data_format(a, na, b, nb);
        a = na;
        b = nb;
    }
};

void kernel_main() {
    constexpr uint32_t cb_q = get_compile_time_arg_val(0);
    constexpr uint32_t cb_k = get_compile_time_arg_val(1);
    constexpr uint32_t cb_v = get_compile_time_arg_val(2);
    constexpr uint32_t cb_mask = get_compile_time_arg_val(3);
    constexpr uint32_t cb_max_scaler = get_compile_time_arg_val(4);
    constexpr uint32_t cb_sum_scaler = get_compile_time_arg_val(5);
    constexpr uint32_t cb_scores = get_compile_time_arg_val(6);
    constexpr uint32_t cb_max_blk = get_compile_time_arg_val(7);
    constexpr uint32_t cb_max_acc = get_compile_time_arg_val(8);
    constexpr uint32_t cb_sum_blk = get_compile_time_arg_val(9);
    constexpr uint32_t cb_sum_acc = get_compile_time_arg_val(10);
    constexpr uint32_t cb_row_factor = get_compile_time_arg_val(11);
    constexpr uint32_t cb_out_blk = get_compile_time_arg_val(12);
    constexpr uint32_t cb_out_acc = get_compile_time_arg_val(13);
    constexpr uint32_t cb_out_tiles = get_compile_time_arg_val(14);
    constexpr uint32_t Sq = get_compile_time_arg_val(15);
    constexpr uint32_t Sk = get_compile_time_arg_val(16);
    constexpr uint32_t DHt = get_compile_time_arg_val(17);
    constexpr uint32_t DVt = get_compile_time_arg_val(18);
    constexpr uint32_t scale_bits = get_compile_time_arg_val(26);
    constexpr uint32_t cb_p_tok = get_compile_time_arg_val(27);

    const uint32_t num_segments = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q, cb_k, cb_scores);

    CircularBuffer q_buf(cb_q), k_buf(cb_k), v_buf(cb_v), scores_buf(cb_scores);
    CircularBuffer out_acc_buf(cb_out_acc), out_blk_buf(cb_out_blk);

    const uint32_t W = get_arg_val<uint32_t>(2 + 2 * num_segments);
    const uint32_t my_units = W + get_arg_val<uint32_t>(3 + 2 * num_segments);
    const uint32_t tail_rows = get_arg_val<uint32_t>(4 + 2 * num_segments);
    const uint32_t pv_sbw = subblock_w(DVt);

    // P.V output chunk width inside the S2 section (2 DEST slots go to the max and the row sum).
    constexpr uint32_t PV_W = DEST_AUTO_LIMIT > 2 ? DEST_AUTO_LIMIT - 2 : 1;
    const uint32_t pv_w = largest_divisor_le(DVt, PV_W);

    using PExp = PackThreadScaledExp<scale_bits, kExpStall>;

    // ------------------------------------------------------------------------------------------------------------
    // Streamed score phases of one key block (bf16 DEST). Leaves cb_max_blk (R, fronted), cb_sum_{acc|blk} (R),
    // cb_out_{acc|blk} (R*DVt) exactly as the op's phase-by-phase body does; cb_scores is consumed.
    // ------------------------------------------------------------------------------------------------------------
    auto stream_block = [&](uint32_t R, uint32_t skv, bool masked, bool first) {
        const uint32_t sbw = subblock_w(skv);
        const uint32_t sum_cb = first ? cb_sum_acc : cb_sum_blk;
        const uint32_t out_cb = first ? cb_out_acc : cb_out_blk;
        Src src{cb_k, cb_q};
        reconfig_data_format(cb_k, cb_q);  // unconditional: the previous block's helpers left any state
        pack_reconfig_data_format(cb_scores);
        PACK((g_pack_width = 0));  // unknown after the previous block's helpers: the first set_pack_width re-programs

        if constexpr (!kExpOnMath && !kNoExp) {
            PExp::init();
        }
        cb_wait_front(cb_k, Sk * DHt);
        cb_wait_front(cb_q, Sq * DHt);
        if (!first) {
            cb_wait_front(cb_max_acc, R);
        }
        if (masked) {
            cb_wait_front(cb_mask, 1);
        }
        cb_reserve_back(cb_scores, Sq * Sk);

        // ---- S3 chunk: score row r cols [c0, c0+sbw) = Q_r . K^T (+ key-tail mask) -> cb_scores (absolute) ----
        auto qk_chunk = [&](uint32_t r, uint32_t c0) {
            src.use(cb_k, cb_q);
            matmul_block_init(cb_q, cb_k, 1, sbw, 1, DHt);
            tile_regs_acquire();
            for (uint32_t k = 0; k < DHt; ++k) {
                matmul_block(cb_q, cb_k, r * DHt + k, k * Sk + c0, 0, 1, sbw, 1, DHt);
            }
            if (masked && (skv - 1) >= c0 && (skv - 1) < c0 + sbw) {
                src.use(src.a, cb_mask);
                add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_mask);
                add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_mask, 0, skv - 1 - c0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_row(cb_scores, 0, r * Sk + c0, sbw);
            tile_regs_release();
            if (c0 + sbw >= skv) {  // row complete
                push_hold_wr_ptr(cb_scores, Sk);
                if (r + 1 == R) {
                    cb_pop_front(cb_k, Sk * DHt);
                    if (R < Sq) {
                        push_hold_wr_ptr(cb_scores, (Sq - R) * Sk);  // nominal credit (tail piece)
                    }
                }
            }
        };

        // ---- S1 chunk: P_i cols [c0, c0+w) = exp(s (S_i - m_i)) in place; exp on the PACK thread ----
        auto exp_chunk = [&](uint32_t i, uint32_t c0, uint32_t w) {
            if (c0 == 0) {
                cb_wait_front(cb_max_blk, i + 1);
            }
            src.use(cb_scores, cb_max_blk);
            sub_bcast_cols_init(cb_scores, cb_max_blk);
            if constexpr (kExpOnMath) {
                ScaledExp<scale_bits>::init();
            }
            tile_regs_acquire();
            for (uint32_t j = 0; j < w; ++j) {
                sub_tiles_bcast_cols(cb_scores, cb_max_blk, i * Sk + c0 + j, i, j);
            }
            if constexpr (kExpOnMath) {
                for (uint32_t j = 0; j < w; ++j) {
                    ScaledExp<scale_bits>::exec_impl(j);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            if constexpr (!kExpOnMath && !kNoExp) {
                if constexpr (kExpGroup) {
                    PExp::tiles<kGroupStallCond>(w);
                } else {
                    for (uint32_t j = 0; j < w; ++j) {
                        PExp::tile(j);
                    }
                }
                PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
            }
            pack_row(cb_scores, 0, i * Sk + c0, w);
            tile_regs_release();
            if (c0 + w >= skv) {  // P row complete -> token for P.V / rowsum
                cb_reserve_back(cb_p_tok, 1);
                cb_push_back(cb_p_tok, 1);
            }
        };

        // ---- S2: m_blk(mrow) = max(m_acc(mrow), rowmax S_mrow); l(prow) = rowsum P_prow; O(prow) = P_prow . V ----
        auto s2 = [&](uint32_t mrow, bool do_max, uint32_t prow, bool do_pv) {
            tile_regs_acquire();
            if (do_max) {
                cb_wait_front(cb_scores, (mrow + 1) * Sk);
                if (!first) {
                    src.use(cb_max_acc, src.b);
                    copy_init(cb_max_acc, 0, 1);  // within-face transpose: col 0 -> the GMPOOL row-0 accumulator
                    copy_tile(cb_max_acc, mrow, 0);
                }
                src.use(cb_scores, cb_max_scaler);
                if constexpr (kStockMax) {
                    reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_scores, cb_max_scaler, cb_max_blk);
                    for (uint32_t k = 0; k < skv; ++k) {
                        reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(
                            cb_scores, cb_max_scaler, mrow * Sk + k, 0, 0);
                    }
                    reduce_uninit();
                } else {
                    reduce_block_max_row_init_runtime(cb_max_blk, skv, cb_scores, cb_max_scaler);
                    reduce_block_max_row_runtime(cb_scores, cb_max_scaler, mrow * Sk, 0);
                    reduce_block_max_row_uninit_runtime(cb_scores);
                }
            }
            if (do_pv) {
                cb_wait_front(cb_p_tok, 1);
                if (prow == 0) {
                    cb_wait_front(cb_v, Sk * DVt);
                }
                src.use(cb_sum_scaler, cb_scores);
                reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_scores, cb_sum_scaler, sum_cb);
                for (uint32_t k = 0; k < skv; ++k) {
                    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_scores, cb_sum_scaler, prow * Sk + k, 0, 1);
                }
                reduce_uninit();
                src.use(cb_v, cb_scores);
                matmul_block_init(cb_scores, cb_v, 0, pv_w, 1, Sk);
                for (uint32_t k = 0; k < skv; ++k) {
                    matmul_block(cb_scores, cb_v, prow * Sk + k, k * DVt, 2, 0, pv_w, 1, Sk);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            PACK((llk_pack_reduce_mask_config<ReduceDim::REDUCE_ROW, PackMode::Default>(cb_max_blk)));
            set_pack_width(cb_max_blk, 1);
            if (do_max) {
                cb_reserve_back(cb_max_blk, 1);
                pack_tile(0, cb_max_blk);
                cb_push_back(cb_max_blk, 1);
            }
            if (do_pv) {
                cb_reserve_back(sum_cb, 1);
                pack_tile(1, sum_cb);
                cb_push_back(sum_cb, 1);
            }
            PACK((llk_pack_reduce_mask_clear()));
            if (do_pv) {
                cb_reserve_back(out_cb, DVt);
                pack_row(out_cb, 2, 0, pv_w);
            }
            tile_regs_release();
            if (do_pv) {
                for (uint32_t c0 = pv_w; c0 < DVt; c0 += pv_w) {  // remaining P.V column chunks (DVt > PV_W)
                    tile_regs_acquire();
                    for (uint32_t k = 0; k < skv; ++k) {
                        matmul_block(cb_scores, cb_v, prow * Sk + k, k * DVt + c0, 0, 0, pv_w, 1, Sk);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_row(out_cb, 0, c0, pv_w);
                    tile_regs_release();
                }
                cb_push_back(out_cb, DVt);
                cb_pop_front(cb_p_tok, 1);
                if (prow + 1 == R) {
                    cb_pop_front(cb_v, Sk * DVt);
                }
            }
        };

        // exp chunk width: split a single-section row in two so each half's exp overlaps a different MATH section
        // (S2, then the Q.K^T of row i+2) instead of one long exp against one section (EMO_SPLIT).
        const uint32_t n_qk = skv / sbw;
        const uint32_t ew = (kSplit && n_qk == 1 && (sbw % 2) == 0) ? sbw / 2 : sbw;
        const uint32_t n_ex = skv / ew;
        const int32_t Ri = static_cast<int32_t>(R);
        for (int32_t i = -2; i <= Ri; ++i) {
            const bool has_exp = i >= 0 && i < Ri;
            const int32_t m = i + 1, p = i - 1, q = i + 2;
            const bool do_max = m >= 0 && m < Ri;
            const bool do_pv = p >= 0 && p < Ri;
            const bool has_qk = q >= 0 && q < Ri;
            const uint32_t ne = has_exp ? n_ex : 0, nq = has_qk ? n_qk : 0;
            if (ne) {
                exp_chunk(i, 0, ew);
            }
            if (do_max || do_pv) {
                s2(do_max ? m : 0, do_max, do_pv ? p : 0, do_pv);
            }
            for (uint32_t t = 1; t < ne || t <= nq; ++t) {
                if (t < ne) {
                    exp_chunk(i, t * ew, ew);
                }
                if (t <= nq) {
                    qk_chunk(q, (t - 1) * sbw);
                }
            }
        }
        cb_wait_front(cb_scores, Sq * Sk);
        cb_pop_front(cb_scores, Sq * Sk);
        set_pack_width(cb_scores, 1);  // the helpers that follow pack one tile per call
    };

    // One unit's key walk + finalize on R compute rows (Sq for whole units, tail_rows for the tail piece).
    auto run_unit = [&](uint32_t R, uint32_t i) {
        MaybeDeviceZoneScope("compute_unit");
        uint32_t kb = 0;  // processed key blocks of this unit
        for (uint32_t s = 0; s < num_segments; ++s) {
            const uint32_t valid = get_arg_val<uint32_t>(2 + 2 * s);
            const bool seg_masked = get_arg_val<uint32_t>(3 + 2 * s) != 0;
            for (uint32_t t0 = 0; t0 < valid; t0 += Sk, ++kb) {
                const uint32_t skv = (valid - t0) < Sk ? (valid - t0) : Sk;
                const bool masked = seg_masked && (t0 + skv == valid);
#ifdef EMO_BLOCK_ZONES
                MaybeDeviceZoneScopeIf(
                    "emo_block", (i == 0) && (kb >= EMO_ZONE_KB0) && (kb < EMO_ZONE_KB0 + EMO_ZONE_NKB));
#endif
                if constexpr (!DST_ACCUM_MODE) {
                    stream_block(R, skv, masked, kb == 0);
                } else {
                    // ---- fp32 DEST: the op's phase-by-phase body (helpers), unchanged ----
                    k_buf.wait_front(Sk * DHt);
                    {
                        const uint32_t sbw = subblock_w(skv);
                        matmul_block<
                            true,
                            false,
                            LastBlockTarget::Out,
                            OutputCBLayout::TileRowMajor,
                            matmul_config::InitMode::Short,
                            InputPolicy::WaitAndRetainOnLastBlock,
                            InputPolicy::NoWaitNoPop>(
                            q_buf,
                            k_buf,
                            scores_buf,
                            scores_buf,
                            MatmulBlockShape::of(R, skv / sbw, 1, sbw, DHt, 1),
                            NoPostCompute{},
                            NoPreKBlock{},
                            Sk,
                            Sk);
                    }
                    k_buf.pop_front(Sk * DHt);
                    if (R < Sq) {
                        cb_reserve_back(cb_scores, (Sq - R) * Sk);
                        cb_push_back(cb_scores, (Sq - R) * Sk);
                    }
                    cb_wait_front(cb_scores, Sq * Sk);
                    if (masked) {
                        eltwise_chain(
                            IterationShape::grid(R, 1),
                            BinaryFpu<
                                BinaryFpuOp::Add,
                                input(
                                    cb_scores,
                                    WaitPolicy::None,
                                    PopPolicy::None,
                                    InputTileMapping::Block,
                                    DataFormatReconfig::Enabled,
                                    TileAddressing::Strided),
                                input(cb_mask, BroadcastDim::None, WaitPolicy::Upfront, PopPolicy::None)>{
                                StridedTileRange{skv - 1, Sk}},
                            PackTile<output(
                                cb_scores,
                                ReservePolicy::None,
                                PushPolicy::None,
                                DataFormatReconfig::Enabled,
                                TileAddressing::Strided)>{StridedTileRange{skv - 1, Sk}});
                        cb_pop_front(cb_scores, Sq * Sk);
                        cb_reserve_back(cb_scores, Sq * Sk);
                        cb_push_back(cb_scores, Sq * Sk);
                        cb_wait_front(cb_scores, Sq * Sk);
                    }
                    reduce<
                        PoolType::MAX,
                        ReduceDim::REDUCE_ROW,
                        cb_scores,
                        cb_max_scaler,
                        cb_max_blk,
                        ReduceInputPolicy::NoWaitNoPop>(
                        ReduceInputBlockShape::of(R, skv), ReduceInputMemoryLayout::with_row_stride(Sk));
                    if (kb > 0) {
                        binary_sfpu<
                            BinaryMax<>,
                            input(cb_max_blk),
                            input(cb_max_acc, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Block),
                            output(cb_max_blk)>(IterationShape::tiles(R));
                    }
                    eltwise_chain(
                        IterationShape::grid(R, skv),
                        BinaryFpu<
                            BinaryFpuOp::Sub,
                            input(
                                cb_scores,
                                WaitPolicy::None,
                                PopPolicy::None,
                                InputTileMapping::Block,
                                DataFormatReconfig::Enabled,
                                TileAddressing::Strided),
                            input(
                                cb_max_blk,
                                BroadcastDim::Col,
                                WaitPolicy::Upfront,
                                PopPolicy::None,
                                InputTileMapping::Col)>{StridedTileRange{0, Sk}},
                        ScaledExp<scale_bits>{},
                        PackTile<output(
                            cb_scores,
                            ReservePolicy::None,
                            PushPolicy::None,
                            DataFormatReconfig::Enabled,
                            TileAddressing::Strided)>{StridedTileRange{0, Sk}});
                    cb_pop_front(cb_scores, Sq * Sk);
                    cb_reserve_back(cb_scores, Sq * Sk);
                    cb_push_back(cb_scores, Sq * Sk);
                    cb_wait_front(cb_scores, Sq * Sk);
                    if (kb == 0) {
                        reduce<
                            PoolType::SUM,
                            ReduceDim::REDUCE_ROW,
                            cb_scores,
                            cb_sum_scaler,
                            cb_sum_acc,
                            ReduceInputPolicy::NoWaitNoPop>(
                            ReduceInputBlockShape::of(R, skv), ReduceInputMemoryLayout::with_row_stride(Sk));
                    } else {
                        reduce<
                            PoolType::SUM,
                            ReduceDim::REDUCE_ROW,
                            cb_scores,
                            cb_sum_scaler,
                            cb_sum_blk,
                            ReduceInputPolicy::NoWaitNoPop>(
                            ReduceInputBlockShape::of(R, skv), ReduceInputMemoryLayout::with_row_stride(Sk));
                    }
                    v_buf.wait_front(Sk * DVt);
                    matmul_block<
                        false,
                        false,
                        LastBlockTarget::Out,
                        OutputCBLayout::TileRowMajor,
                        matmul_config::InitMode::Short,
                        InputPolicy::WaitAndPopPerKBlock,
                        InputPolicy::NoWaitNoPop>(
                        scores_buf,
                        v_buf,
                        kb == 0 ? out_acc_buf : out_blk_buf,
                        kb == 0 ? out_acc_buf : out_blk_buf,
                        MatmulBlockShape::of(R, DVt / pv_sbw, 1, pv_sbw, Sk, 1),
                        NoPostCompute{},
                        NoPreKBlock{},
                        DVt,
                        DVt,
                        NoPostKBlock{},
                        ValidKeyTiles{skv});
                    v_buf.pop_front(Sk * DVt);
                    if (R < Sq) {
                        cb_pop_front(cb_scores, (Sq - R) * Sk);
                    }
                }

                // ---- rescale_block + m_acc <- m_blk (the op's helpers, unchanged) ----
                auto rescale_copy = [&](auto rows_t) {
                    const uint32_t RR = rows_t;
                    if (kb > 0) {
                        eltwise_chain(
                            IterationShape::tiles(RR),
                            BinaryFpu<
                                BinaryFpuOp::Sub,
                                input(cb_max_acc),
                                input(cb_max_blk, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Block)>{},
                            ScaledExp<scale_bits>{},
                            PackTile<output(cb_row_factor)>{});
                        eltwise_chain(
                            IterationShape::tiles(RR),
                            BinaryFpu<
                                BinaryFpuOp::Mul,
                                input(cb_sum_acc),
                                input(
                                    cb_row_factor,
                                    BroadcastDim::Col,
                                    WaitPolicy::Upfront,
                                    PopPolicy::None,
                                    InputTileMapping::Block)>{},
                            DestReuseBinary<BinaryFpuOp::Add, input(cb_sum_blk), DestReuseType::DEST_TO_SRCA>{},
                            PackTile<output(cb_sum_acc)>{});
                        eltwise_chain(
                            IterationShape::grid(RR, DVt),
                            BinaryFpu<
                                BinaryFpuOp::Mul,
                                input(cb_out_acc),
                                input(
                                    cb_row_factor,
                                    BroadcastDim::Col,
                                    WaitPolicy::Upfront,
                                    PopPolicy::AtEnd,
                                    InputTileMapping::Col)>{},
                            DestReuseBinary<BinaryFpuOp::Add, input(cb_out_blk), DestReuseType::DEST_TO_SRCA>{},
                            PackTile<output(cb_out_acc)>{});
                    }
                    copy<input(cb_max_blk), output(cb_max_acc)>(IterationShape::tiles(RR));
                };
                if (R == Sq) {
                    rescale_copy(std::integral_constant<uint32_t, Sq>{});
                } else {
                    rescale_copy(R);
                }
            }
        }

        {
            MaybeDeviceZoneScope("compute_finalize");
            unary<Recip<>, input(cb_sum_acc), output(cb_row_factor)>(IterationShape::tiles(R));
            mul<input(cb_out_acc),
                input(cb_row_factor, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col),
                output(cb_out_tiles)>(IterationShape::grid(R, DVt));
        }
        cb_wait_front(cb_max_acc, R);
        cb_pop_front(cb_max_acc, R);
        cb_pop_front(cb_q, Sq * DHt);
    };

    for (uint32_t i = 0; i < my_units; ++i) {
        run_unit(i < W ? Sq : tail_rows, i);
    }
}
