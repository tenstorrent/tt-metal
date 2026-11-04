// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// pv_rescale_fusion bench copy of the op's compute kernel. VARIANT (ct arg 27):
//   0 base   : the op's current pv_block + rescale_block (O_blk -> cb_out_blk, then O_acc = a O_acc + O_blk chain).
//   1 fused  : DEST preload. Per DEST chunk: a*O_acc (mul_tiles_bcast_cols) into DEST, the P.V matmul accumulates on
//              top (DST += P.V), pack straight back to cb_out_acc in place. No cb_out_blk, no 64-tile add pass.
//   3 fused_mmfirst : as 1, but the matmul runs first into the fresh DEST chunk and the ELWMUL (which accumulates
//              into DEST in hardware) adds a*O_acc on top — keeps the baseline's P.V rounding at |O_blk| scale.
//   4 fused_l : 3 + the l update folded into the row sum (bf16 DEST): a computed right after the exp, then per DEST
//              chunk FPU row-reduce of P into DEST + ELWMUL a*l_acc on top, packed in place (no cb_sum_blk, no l
//              chain).
//   2 l1acc  : helper-only, production-SDPA style. P.V -> the free O buffer (ping-pong cb_out_acc / cb_out_blk), then
//              one eltwise_chain Mul(O_prev, a bcast col) packed with L1Accumulation::AddToExisting onto it.
//
// RAW-LLK JUSTIFICATION (variants 1/3/4: fused_pv_correct, fused_rowsum_l, l_update_acc).
//  Mechanism: on Wormhole/Blackhole both MVMUL (ckernel::matmul_block: DST += A.B) and ELWMUL (mul_tiles_bcast_cols:
//  DST += A*B, the FPU accumulates every fidelity phase into DEST — see the ZEROACC note in tt-llk
//  llk_math_eltwise_binary.h, "ELWMUL accumulates into DEST") ADD into the DEST tile they target, and so does the FPU
//  row reduce (reduce_tile). So the online-softmax correction x_new = a * x_acc + x_blk needs no separate pass: run
//  the producer of x_blk (P.V matmul / P row-sum reduce) into a fresh DEST chunk, then ELWMUL a (bcast col) * x_acc
//  on top, and pack the chunk straight back over x_acc. Matmul FIRST keeps the baseline's P.V rounding (at |O_blk|
//  scale); seeding a*O_acc first and letting the 8 matmul steps accumulate on it rounds every step at |O_acc| scale
//  (variant 1: rel-RMS 3.6x the baseline — disqualified).
//  Helpers bypassed — both CAPABILITY gaps, not ergonomics:
//  * compute_kernel_lib::matmul_block owns tile_regs_acquire/commit/pack per output subblock and has no per-subblock
//    hook INSIDE the DEST window after the K loop that may unpack a second CB pair (PostComputeFn runs on the MATH
//    thread only, so it cannot issue the ELWMUL's unpack; PreKBlockFn runs per K-block outside the DEST window), and
//    it cannot pack in place onto a fronted CB (its pack target is always reserved/pushed FIFO-style).
//    Missing: a per-subblock "epilogue with unpack" functor (all three TRISCs) + an in-place / caller-owned pack
//    window that is NOT L1-accumulating.
//  * compute_kernel_lib::reduce<SUM, REDUCE_ROW> acquires DEST per output row (one tile), so a post_reduce_op ELWMUL
//    would re-init mul + reduce twice per row, and its output reserve/push cannot target the accumulator CB the
//    ELWMUL is reading (in place). Missing: multi-row DEST chunks + the same unpacking epilogue + in-place output.
//  * the l update at fp32 DEST (l_update_acc) is copy_tile + ELWMUL-accumulate + in-place pack; eltwise_chain's
//    DestReuse ops overwrite DEST (DEST -> src, op, write), there is no "ELWMUL accumulate onto a DEST seed" element.
//  In-place pack contract (all three): the CB holds exactly the accumulator (RR == Sq: wr_ptr == rd_ptr) or the
//  accumulator plus free space (RR | Sq, RR <= Sq/2: the pack lands in the next free window); pack_tile<true> writes
//  each tile after its own unpack, then pop RR / reserve RR / push RR makes the new values the front.
//  Kept noinline: inlining the two row-count instantiations overflowed the 70656 B kernel config buffer (profiler on).

// Compute-core flash attention (Q-stationary, online softmax), one attention unit = Sq q tile-rows of one
// (batch, q head), walking every key segment in blocks of Sk key tiles.
//
// Per key block (skv <= Sk valid key tiles; CB credits are always nominal: Sq*Sk scores, Sk*DHt K, Sk*DVt V):
//   score_block   S = Q . K^T                      matmul_block (transpose, TileRowMajor, row stride Sk)
//   mask_block    S[:, skv-1] += key-tail mask      eltwise_chain BinaryFpu Add, Strided in place (boundary block)
//   row_max_block m_blk = rowmax(S) [max m_acc]     reduce<MAX, REDUCE_ROW> (row stride Sk) + binary_sfpu BinaryMax
//   exp_block     P = exp(s (S - m_blk)) in place   eltwise_chain BinaryFpu Sub (bcast col) -> ScaledExp, Strided
//   sum           l_blk = rowsum(P)                 reduce<SUM, REDUCE_ROW> (row stride Sk)
//   pv_block      O_blk = P . V                     matmul_block (K = Sk stride, skv inner steps)
//   rescale_block (kb > 0) a = exp(s (m_acc - m_blk)); l_acc = a l_acc + l_blk; O_acc = a O_acc + O_blk
//                 eltwise_chain BinaryFpu Mul (bcast col) -> DestReuseBinary Add, in place on the accumulators
// finalize_block: out = O_acc * (1 / l_acc)  (Recip, then Mul bcast col) -> cb_out_tiles (bf16)
//
// Deviations from the design's API table (recorded in the commit):
//  * the row sum is a reduce<SUM> over P (the design's "sum via reduce" perf lamp) instead of a second
//    L1-accumulating pack in the exp chain — l is then column-0 valid and needs no in-tile row sum at finalize;
//  * the running accumulators have fixed roles (m_acc / l_acc / O_acc) updated in place, with a per-block scratch
//    (m_blk / l_blk / O_blk), instead of a ping-pong pair — same CB count, no runtime CB-id swap;
//  * scores keep a nominal Sk row stride so every CB credit is uniform (ragged blocks narrow only the compute
//    extent: the matmul N, the reduce cols, the exp grid and the P.V inner steps).

#include <cstdint>
#include <type_traits>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_unary/exp.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/minmax.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "ring_joint_sdpa_common.hpp"
#include "ring_joint_sdpa_scaled_exp.hpp"

using namespace compute_kernel_lib;

// Sampled per-phase zone window (key blocks of the first unit); see the zone budget note at the loop.
constexpr uint32_t ZONE_SAMPLE_KB0 = 4;
constexpr uint32_t ZONE_SAMPLE_NKB = 6;

// exp(scale * x): ScaledExp<scale_bf16> (ring_joint_sdpa_scaled_exp.hpp) — a raw-LLK SFPLOADMACRO exp_21f at bf16 DEST
// (bit-identical to exp_tile<false, true>, 1.85x faster; justification at that file's head), stock exp_tile at fp32
// DEST.

// P.V inner steps: only the valid key tiles of a (possibly ragged) block.
struct ValidKeyTiles {
    uint32_t n;
    ALWI uint32_t operator()(uint32_t /*block*/, uint32_t /*block_k*/) const { return n; }
};

// In-place phases on cb_scores (mask, exp) write the fronted block without moving CB credit; cycle the whole
// block's credit (unpack pop -> pack reserve/push -> unpack wait) so the next phase's unpacker reads only after the
// packer finished. The CB holds exactly one block, so the block stays on the same slots.
ALWI void inplace_fence(uint32_t cb, uint32_t n) {
    cb_pop_front(cb, n);
    cb_reserve_back(cb, n);
    cb_push_back(cb, n);
    cb_wait_front(cb, n);
}

// Largest subblock width dividing n that fits DEST (subblock height 1).
inline uint32_t subblock_w(uint32_t n) {
    uint32_t w = n < DEST_AUTO_LIMIT ? n : DEST_AUTO_LIMIT;
    while (n % w) {
        --w;
    }
    return w;
}

constexpr uint32_t largest_divisor_le(uint32_t n, uint32_t cap) {
    uint32_t w = n < cap ? n : cap;
    while (n % w) {
        --w;
    }
    return w;
}

// O_acc (RR x DVt, fronted in cb_out_acc) <- a (bcast col) * O_acc + P (RR x skv, row stride Sk) . V (skv x DVt).
// DEST chunk = h q-rows x w value columns (h*w <= DEST). Packs in place: for a full CB (RR == Sq) the pack window
// (fifo_wr_ptr) equals the fronted window; for an RR < Sq tail piece (RR | Sq, so RR <= Sq/2) the pack lands in the
// free window right after it. Either way the credit cycle at the end (pop RR*DVt, reserve, push) makes the new
// tiles the front. Each tile is unpacked (mul) before its own pack is issued, so the in-place overwrite is safe.
template <
    bool mm_first,
    uint32_t cb_scores,
    uint32_t cb_v,
    uint32_t cb_out_acc,
    uint32_t cb_row_factor,
    uint32_t Sk,
    uint32_t DVt>
__attribute__((noinline)) void fused_pv_correct(uint32_t RR, uint32_t skv) {
    constexpr uint32_t w = largest_divisor_le(DVt, DEST_AUTO_LIMIT);
    uint32_t h = DEST_AUTO_LIMIT / w;
    while (RR % h) {
        --h;
    }
    cb_wait_front(cb_out_acc, RR * DVt);
    cb_wait_front(cb_row_factor, RR);
    PACK((pack_reconfig_data_format(cb_out_acc)));
    // Loop invariant at each chunk start: unpacker / math formats are those of the op issued LAST in a chunk.
    if constexpr (mm_first) {
        reconfig_data_format(cb_out_acc, cb_row_factor);
    } else {
        reconfig_data_format(cb_v, cb_scores);
    }
    for (uint32_t g = 0; g < RR; g += h) {
        for (uint32_t c0 = 0; c0 < DVt; c0 += w) {
            auto mul_part = [&]() {
                reconfig_data_format(cb_v, cb_out_acc, cb_scores, cb_row_factor);
                mul_bcast_cols_init(cb_out_acc, cb_row_factor);
                for (uint32_t r = 0; r < h; ++r) {
                    for (uint32_t c = 0; c < w; ++c) {
                        mul_tiles_bcast_cols(cb_out_acc, cb_row_factor, (g + r) * DVt + c0 + c, g + r, r * w + c);
                    }
                }
            };
            auto mm_part = [&]() {
                reconfig_data_format(cb_out_acc, cb_v, cb_row_factor, cb_scores);
                matmul_block_init(cb_scores, cb_v, false, w, h, Sk);
                for (uint32_t k = 0; k < skv; ++k) {
                    matmul_block(cb_scores, cb_v, g * Sk + k, k * DVt + c0, 0, false, w, h, Sk);
                }
            };
            tile_regs_acquire();
            if constexpr (mm_first) {
                // DEST = P.V (fresh, the baseline O_blk rounding), then ELWMUL accumulates a*O_acc on top
                // (the FPU ELWMUL adds into DEST: DST += A*B, fidelity phases included).
                mm_part();
                mul_part();
            } else {
                // DEST = a*O_acc, then the matmul accumulates P.V on top (DST += P.V)
                mul_part();
                mm_part();
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t r = 0; r < h; ++r) {
                for (uint32_t c = 0; c < w; ++c) {
                    pack_tile<true>(r * w + c, cb_out_acc, (g + r) * DVt + c0 + c);
                }
            }
            tile_regs_release();
        }
    }
    cb_pop_front(cb_out_acc, RR * DVt);
    cb_reserve_back(cb_out_acc, RR * DVt);
    cb_push_back(cb_out_acc, RR * DVt);
}

// l_acc (RR tiles, fronted in cb_sum_acc) <- rowsum(P) + a (bcast col) * l_acc, one DEST chunk per DEST_AUTO_LIMIT
// rows: the FPU row reduce (MVMUL against the scaler tile) accumulates the row sum into a fresh DEST slot, then ELWMUL
// (which accumulates into DEST) adds a*l_acc on top; packed in place (same credit cycle as fused_pv_correct). bf16 DEST
// only (the reduce helper takes an SFPU path for fp32 SUM, so the fp32 DEST build keeps the helper + l chain).
template <uint32_t cb_scores, uint32_t cb_sum_scaler, uint32_t cb_sum_acc, uint32_t cb_row_factor, uint32_t Sk>
__attribute__((noinline)) void fused_rowsum_l(uint32_t RR, uint32_t skv) {
    cb_wait_front(cb_sum_acc, RR);
    cb_wait_front(cb_row_factor, RR);
    cb_wait_front(cb_sum_scaler, 1);
    PACK((pack_reconfig_data_format(cb_sum_acc)));
    for (uint32_t g = 0; g < RR; g += DEST_AUTO_LIMIT) {
        const uint32_t n = (RR - g) < DEST_AUTO_LIMIT ? (RR - g) : DEST_AUTO_LIMIT;
        tile_regs_acquire();
        reconfig_data_format(cb_sum_scaler, cb_scores);  // REDUCE_ROW SUM swaps operands: scaler -> SrcA
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_scores, cb_sum_scaler, cb_sum_acc);
        for (uint32_t r = 0; r < n; ++r) {
            for (uint32_t k = 0; k < skv; ++k) {
                reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_scores, cb_sum_scaler, (g + r) * Sk + k, 0, r);
            }
        }
        MATH((llk_math_reduce_uninit()));  // math state only; the packer keeps the row-reduce mask for these packs
        reconfig_data_format(cb_sum_acc, cb_row_factor);
        mul_bcast_cols_init(cb_sum_acc, cb_row_factor);
        for (uint32_t r = 0; r < n; ++r) {
            mul_tiles_bcast_cols(cb_sum_acc, cb_row_factor, g + r, g + r, r);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t r = 0; r < n; ++r) {
            pack_tile<true>(r, cb_sum_acc, g + r);
        }
        tile_regs_release();
    }
    PACK((llk_pack_reduce_mask_clear()));
    cb_pop_front(cb_sum_acc, RR);
    cb_reserve_back(cb_sum_acc, RR);
    cb_push_back(cb_sum_acc, RR);
}

// l_acc <- l_blk + a (bcast col) * l_acc: l_blk copied into DEST, ELWMUL accumulates a*l_acc on top, packed in place.
// (fp32 DEST path of variant 4, where the row sum stays on the reduce helper's SFPU path.)
template <uint32_t cb_sum_blk, uint32_t cb_sum_acc, uint32_t cb_row_factor>
__attribute__((noinline)) void l_update_acc(uint32_t RR) {
    cb_wait_front(cb_sum_acc, RR);
    cb_wait_front(cb_sum_blk, RR);
    cb_wait_front(cb_row_factor, RR);
    PACK((pack_reconfig_data_format(cb_sum_acc)));
    for (uint32_t g = 0; g < RR; g += DEST_AUTO_LIMIT) {
        const uint32_t n = (RR - g) < DEST_AUTO_LIMIT ? (RR - g) : DEST_AUTO_LIMIT;
        tile_regs_acquire();
        reconfig_data_format_srca(cb_sum_blk);
        copy_tile_to_dst_init_short(cb_sum_blk);
        for (uint32_t r = 0; r < n; ++r) {
            copy_tile(cb_sum_blk, g + r, r);
        }
        reconfig_data_format(cb_sum_acc, cb_row_factor);
        mul_bcast_cols_init(cb_sum_acc, cb_row_factor);
        for (uint32_t r = 0; r < n; ++r) {
            mul_tiles_bcast_cols(cb_sum_acc, cb_row_factor, g + r, g + r, r);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t r = 0; r < n; ++r) {
            pack_tile<true>(r, cb_sum_acc, g + r);
        }
        tile_regs_release();
    }
    cb_pop_front(cb_sum_blk, RR);
    cb_pop_front(cb_sum_acc, RR);
    cb_reserve_back(cb_sum_acc, RR);
    cb_push_back(cb_sum_acc, RR);
}

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
    constexpr uint32_t n_q_t = get_compile_time_arg_val(19);
    constexpr uint32_t Lq_t = get_compile_time_arg_val(20);
    constexpr uint32_t BH = get_compile_time_arg_val(22);
    constexpr uint32_t C = get_compile_time_arg_val(23);
    constexpr uint32_t scale_bits = get_compile_time_arg_val(26);
    constexpr uint32_t VARIANT = get_compile_time_arg_val(27);

    const uint32_t core_idx = get_arg_val<uint32_t>(0);
    const uint32_t num_segments = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q, cb_k, cb_scores);

    CircularBuffer q_buf(cb_q), k_buf(cb_k), v_buf(cb_v), scores_buf(cb_scores);
    CircularBuffer out_acc_buf(cb_out_acc), out_blk_buf(cb_out_blk);

    // Balanced tail split: W whole units (nominal Sq compute rows), then an optional tail piece computed on its
    // R = tail_rows rows only (R | Sq, so every R-sized credit cycle stays contiguous in its CB; the piece is the
    // core's last unit). cb_q keeps its nominal Sq*DHt credit (the matmul waits only R*DHt of it).
    const uint32_t W = get_arg_val<uint32_t>(2 + 2 * num_segments);
    const uint32_t my_units = W + get_arg_val<uint32_t>(3 + 2 * num_segments);
    const uint32_t tail_rows = get_arg_val<uint32_t>(4 + 2 * num_segments);
    constexpr uint32_t pv_sbw_n = DVt;
    const uint32_t pv_sbw = subblock_w(pv_sbw_n);

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
                // Per-phase zones on a sampled window (first unit, steady-state local blocks): the marker budget
                // (~125 zone executions per RISC) cannot hold every block.
                const bool zs = (i == 0) && (kb >= ZONE_SAMPLE_KB0) && (kb < ZONE_SAMPLE_KB0 + ZONE_SAMPLE_NKB);

                // ---- score_block: S (R x skv, row stride Sk) = Q . K^T ----
                {
                    MaybeDeviceZoneScopeIf("compute_k_wait", zs);
                    k_buf.wait_front(Sk * DHt);
                }
                {
                    MaybeDeviceZoneScopeIf("compute_qk", zs);
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
                // cb_scores keeps the NOMINAL Sq*Sk credit per block even on an R-row tail piece: the in-place
                // phases (mask, exp) rely on the CB holding exactly one block (pack write ptr == unpack read ptr);
                // an R*Sk credit would make them ping-pong between CB halves. Pad the matmul's R*Sk push.
                if (R < Sq) {
                    cb_reserve_back(cb_scores, (Sq - R) * Sk);
                    cb_push_back(cb_scores, (Sq - R) * Sk);
                }
                cb_wait_front(cb_scores, Sq * Sk);

                // ---- mask_block: last valid key tile column of the boundary block ----
                if (masked) {
                    MaybeDeviceZoneScopeIf("compute_mask", zs);
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
                    inplace_fence(cb_scores, Sq * Sk);
                }

                // ---- row_max_block ----
                {
                    MaybeDeviceZoneScopeIf("compute_rowmax", zs);
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
                }

                // ---- exp_block: P = exp(s (S - m_blk)) in place ----
                {
                    MaybeDeviceZoneScopeIf("compute_sub_exp", zs);
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
                    inplace_fence(cb_scores, Sq * Sk);
                }

                // ---- row sums of P ----
                constexpr bool fold_l = VARIANT == 4 && !DST_ACCUM_MODE;
                if (fold_l && kb > 0) {
                    MaybeDeviceZoneScopeIf("compute_rowsum", zs);
                    auto a_and_l = [&](auto rows_t) {
                        const uint32_t RR = rows_t;
                        {
                            MaybeDeviceZoneScopeIf("compute_a", zs);
                            eltwise_chain(
                                IterationShape::tiles(RR),
                                BinaryFpu<
                                    BinaryFpuOp::Sub,
                                    input(cb_max_acc),
                                    input(cb_max_blk, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Block)>{},
                                ScaledExp<scale_bits>{},
                                PackTile<output(cb_row_factor)>{});
                        }
                        fused_rowsum_l<cb_scores, cb_sum_scaler, cb_sum_acc, cb_row_factor, Sk>(RR, skv);
                    };
                    if (R == Sq) {
                        a_and_l(std::integral_constant<uint32_t, Sq>{});
                    } else {
                        a_and_l(R);
                    }
                } else {
                    MaybeDeviceZoneScopeIf("compute_rowsum", zs);
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
                }

                // ---- pv_block + rescale_block + m_acc <- m_blk (the part under study) ----
                MaybeDeviceZoneScopeIf("compute_pv_rescale", zs);
                v_buf.wait_front(Sk * DVt);
                // variant 2 ping-pong: the accumulator after block kb lives in cb_out_acc for even kb, cb_out_blk
                // for odd kb. T = this block's P.V target.
                const bool odd = (kb & 1) != 0;
                const bool pv_fused = (VARIANT == 1 || VARIANT == 3 || VARIANT == 4) && kb > 0;
                if (!pv_fused) {
                    MaybeDeviceZoneScopeIf("compute_pv", zs);
                    const uint32_t pv_cb = (kb == 0 || (VARIANT == 2 && !odd)) ? cb_out_acc : cb_out_blk;
                    CircularBuffer pv_target(pv_cb);
                    matmul_block<
                        false,
                        false,
                        LastBlockTarget::Out,
                        OutputCBLayout::TileRowMajor,
                        matmul_config::InitMode::Short,
                        InputPolicy::WaitAndRetainOnLastBlock,
                        InputPolicy::NoWaitNoPop>(
                        scores_buf,
                        v_buf,
                        pv_target,
                        pv_target,
                        MatmulBlockShape::of(R, DVt / pv_sbw, 1, pv_sbw, Sk, 1),
                        NoPostCompute{},
                        NoPreKBlock{},
                        DVt,
                        DVt,
                        NoPostKBlock{},
                        ValidKeyTiles{skv});
                    if (VARIANT == 2 && R < Sq) {  // pad credit: keep the O CB full so wr_ptr == rd_ptr
                        cb_reserve_back(pv_cb, (Sq - R) * DVt);
                        cb_push_back(pv_cb, (Sq - R) * DVt);
                    }
                }

                auto rescale_copy = [&](auto rows_t) {
                    const uint32_t RR = rows_t;
                    if (kb > 0) {
                        MaybeDeviceZoneScopeIf("compute_rescale", zs);
                        if constexpr (!fold_l) {
                            // a = exp(s (m_acc - m_blk)); consumes m_acc
                            {
                                MaybeDeviceZoneScopeIf("compute_a", zs);
                                eltwise_chain(
                                    IterationShape::tiles(RR),
                                    BinaryFpu<
                                        BinaryFpuOp::Sub,
                                        input(cb_max_acc),
                                        input(
                                            cb_max_blk,
                                            WaitPolicy::Upfront,
                                            PopPolicy::None,
                                            InputTileMapping::Block)>{},
                                    ScaledExp<scale_bits>{},
                                    PackTile<output(cb_row_factor)>{});
                            }
                            // l_acc = a l_acc + l_blk
                            {
                                MaybeDeviceZoneScopeIf("compute_l", zs);
                                if constexpr (VARIANT == 4) {
                                    l_update_acc<cb_sum_blk, cb_sum_acc, cb_row_factor>(RR);
                                } else {
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
                                        DestReuseBinary<
                                            BinaryFpuOp::Add,
                                            input(cb_sum_blk),
                                            DestReuseType::DEST_TO_SRCA>{},
                                        PackTile<output(cb_sum_acc)>{});
                                }
                            }
                        }  // !fold_l
                        if constexpr (VARIANT == 0) {
                            // O_acc = a O_acc + O_blk; releases a
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
                        } else if constexpr (VARIANT == 1 || VARIANT == 3 || VARIANT == 4) {
                            fused_pv_correct<VARIANT != 1, cb_scores, cb_v, cb_out_acc, cb_row_factor, Sk, DVt>(
                                RR, skv);
                            cb_pop_front(cb_row_factor, RR);
                            cb_pop_front(cb_scores, R * Sk);  // P consumed (the helper popped it per K-block)
                        } else {
                            // T += a * S (packer L1 accumulation onto the fronted P.V block); S released
                            auto corr = [&](auto src_t, auto dst_t) {
                                constexpr uint32_t S = decltype(src_t)::value;
                                constexpr uint32_t T = decltype(dst_t)::value;
                                eltwise_chain(
                                    IterationShape::grid(RR, DVt),
                                    BinaryFpu<
                                        BinaryFpuOp::Mul,
                                        input(S),
                                        input(
                                            cb_row_factor,
                                            BroadcastDim::Col,
                                            WaitPolicy::Upfront,
                                            PopPolicy::AtEnd,
                                            InputTileMapping::Col)>{},
                                    PackTile<output(
                                        T,
                                        ReservePolicy::None,
                                        PushPolicy::None,
                                        DataFormatReconfig::Enabled,
                                        TileAddressing::Strided,
                                        DestAccumulation::Disabled,
                                        L1Accumulation::AddToExisting)>{StridedTileRange{0, DVt}});
                                if (R < Sq) {
                                    cb_pop_front(S, (Sq - R) * DVt);  // S's pad credit
                                }
                                inplace_fence(T, Sq * DVt);
                            };
                            if (odd) {
                                corr(
                                    std::integral_constant<uint32_t, cb_out_acc>{},
                                    std::integral_constant<uint32_t, cb_out_blk>{});
                            } else {
                                corr(
                                    std::integral_constant<uint32_t, cb_out_blk>{},
                                    std::integral_constant<uint32_t, cb_out_acc>{});
                            }
                        }
                    }
                    // m_acc <- m_blk
                    MaybeDeviceZoneScopeIf("compute_copy_max", zs);
                    copy<input(cb_max_blk), output(cb_max_acc)>(IterationShape::tiles(RR));
                };
                if (R == Sq) {
                    rescale_copy(std::integral_constant<uint32_t, Sq>{});
                } else {
                    rescale_copy(R);
                }
                if (!pv_fused) {
                    cb_pop_front(cb_scores, R * Sk);
                }
                v_buf.pop_front(Sk * DVt);
                if (R < Sq) {
                    cb_pop_front(cb_scores, (Sq - R) * Sk);  // the pad credit
                }
            }
        }

        // ---- finalize_block: out = O_acc / l_acc ----
        {
            MaybeDeviceZoneScope("compute_finalize");
            unary<Recip<>, input(cb_sum_acc), output(cb_row_factor)>(IterationShape::tiles(R));
            auto fin = [&](auto acc_t) {
                constexpr uint32_t A = decltype(acc_t)::value;
                mul<input(A),
                    input(
                        cb_row_factor, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col),
                    output(cb_out_tiles)>(IterationShape::grid(R, DVt));
                if (VARIANT == 2 && R < Sq) {
                    cb_pop_front(A, (Sq - R) * DVt);
                }
            };
            if (VARIANT == 2 && ((kb - 1) & 1)) {
                fin(std::integral_constant<uint32_t, cb_out_blk>{});
            } else {
                fin(std::integral_constant<uint32_t, cb_out_acc>{});
            }
        }
        cb_wait_front(cb_max_acc, R);
        cb_pop_front(cb_max_acc, R);
        cb_pop_front(cb_q, Sq * DHt);
    };

    for (uint32_t i = 0; i < my_units; ++i) {
        run_unit(i < W ? Sq : tail_rows, i);
    }
}
