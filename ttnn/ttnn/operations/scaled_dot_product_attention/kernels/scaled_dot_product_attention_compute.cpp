// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// SDPA compute kernel (TRISC).
//
// Refinement 4: two-pass output normalization. Pass 1 walks K to find
// global_max and the row-sum-of-exp; pass 2 re-streams K + V and
// accumulates `out = sum_k exp(scores_k - global_max) @ V[k]` directly,
// without the per-iter `prev_mm_out * exp_max_diff + partial` correction
// cascade. The cascade was the dominant precision floor at fp32 +
// S >= 4096 — eliminating it closes those cells' RMS to within fp32
// tolerance (PCC ≥ 0.999, RMS ≤ 0.02).
//
// Per-row pipeline:
//
//   Pass 1 (max/sum) — for k in [0, Kt):
//     scores = Q @ K[k]^T                  (matmul_tiles, transpose B)
//     scores = scores + mask               (when HAS_MASK)
//     row_max = max(scores along W)
//     cur_max = (k==0) ? row_max : max(cur_max, row_max)
//     attn = exp(scale * (scores - cur_max))           (SFPU fused scale+exp)
//     mirror attn into cur_sum_exp (consumed by post-K row-reduce)
//     if k > 0:
//       corr = exp(scale * (prev_max - cur_max))
//       cur_sum_exp = prev_sum_exp * corr + cur_sum_exp  (column-vector
//                                                        cascade — bounded)
//     [V is NOT consumed; cb_attention_weights is dropped]
//
//   Post-pass-1:
//     global_max = cur_max
//     row_reduce(cur_sum_exp) → column vector, then take reciprocal
//                              ⇒ 1 / global_sum
//
//   Pass 2 (output) — for k in [0, Kt):
//     scores = Q @ K[k]^T                  (re-matmul; K re-read from DRAM)
//     scores = scores + mask               (mask re-pushed by reader)
//     attn = exp(scale * (scores - global_max))         (FIXED max; no cascade)
//     partial = attn @ V[k]
//     if k == 0: cb_cur_mm_out = partial
//     else:      cb_cur_mm_out[d] += partial[d]   (direct add, NO corr mul)
//
//   Final: out = cb_cur_mm_out * (1 / global_sum), col-broadcast on sum.
//
// L1 footprint trade-off (per the R4 verifier note): K and V are re-read
// from DRAM in pass 2 (cheap) rather than parking attention weights in L1
// across the K-loop (would cost Kt × Wt extra tile slots — prohibitive at
// S=8192). Mask is also re-pushed by the reader per pass.
//
// Raw LLK is used throughout because this fused-online-softmax kernel
// requires a `PostComputeFn`-style mask add inside an open matmul DEST
// window (which itself must be raw LLK), several two-stage FMA passes
// that the per-element eltwise_chain composition cannot express in a
// single chain, and exact pool-type-aware scaler layouts that the
// tt-train SDPA reference's pattern already handles. Substituting raw
// LLK here matches that proven SDPA reference structure.
//
// Refinement 5: reclaimed cb_prev_max, cb_prev_sum_exp, cb_exp_max_diff,
// cb_prev_mm_out. They were R1-vintage CBs for the ping-pong correction
// cascade that R4-iter3 removed; the cb_prev_mm_out alone was 256 KB at
// fp32 D=1024 (2 × Dt × fp32 tile_size = 2 × 32 × 4096), enough on its
// own to push the four `Q1x1x128x1024 fp32` golden cells over the
// 1.5 MB per-core L1 cap. R5 closes those cells by deleting the unused
// allocations + removing the dead helper functions that referenced them
// (update_cur_row_max_value, apply_exp_inplace_and_find_exp_sum,
// update_exp_max_diff, update_cur_exp_sum_inplace, update_cur_mm_out_inplace).
// The remaining Dt-scaling CBs (cb_query, cb_key, cb_value, cb_cur_mm_out,
// cb_output) still grow with head_dim and would OOM at fp32 D ≥ 2048 —
// true D-blocking on the matmul is filed as a follow-up (see changelog).

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"

constexpr uint32_t onetile = 1U;

// --- Compile-time args ------------------------------------------------------
constexpr uint32_t Dt = get_compile_time_arg_val(0);
constexpr uint32_t Kt = get_compile_time_arg_val(1);
constexpr uint32_t scaler_fp32 = get_compile_time_arg_val(2);
constexpr uint32_t HAS_MASK = get_compile_time_arg_val(3);
// Refinement 6: when fp32_dest_acc_en=True the program descriptor tags
// cb_cur_sum_exp and cb_cur_mm_out with UnpackToDestFp32 so the per-K-
// iter SFPU reloads preserve the full 24-bit FP32 mantissa instead of
// truncating to 10-bit TF32 through srcA/srcB. The final divide is FPU
// (mul_tiles_bcast_cols), which is incompatible with UnpackToDestFp32 —
// so on the tagged path we first SFPU-copy both accumulators into
// untagged intermediates (cb_cur_*_for_divide) and run the FPU divide
// against those. When fp32_dest_acc_en=False the running-state CBs are
// bf16 (FPU-readable) and the kernel takes the direct-divide branch.
constexpr uint32_t USE_UNTAGGED_DIVIDE = get_compile_time_arg_val(4);

// scaler_bf16 = the upper 16 bits of the fp32 scale value.
constexpr uint16_t scaler_bf16 = static_cast<uint16_t>(scaler_fp32 >> 16);

// --- CB indices -------------------------------------------------------------
// Slots 8, 10, 11, 20 (old cb_prev_max / cb_exp_max_diff / cb_prev_sum_exp
// / cb_prev_mm_out) were reclaimed in Refinement 5. R4-iter3's two-pass
// structure does the running-state updates IN PLACE on cb_cur_max,
// cb_cur_sum_exp and cb_cur_mm_out — there is no "prev" companion any
// more. At fp32 D=1024 the dropped cb_prev_mm_out alone was 256 KB
// (~16% of the per-core L1 budget); reclaiming all four pushes the
// footprint back under the 1.5 MB cap for the previously-OOMing
// `Q1x1x128x1024 fp32` cells.
constexpr uint32_t cb_query = 0;
constexpr uint32_t cb_key = 1;
constexpr uint32_t cb_value = 2;
constexpr uint32_t cb_attn_mask = 3;
constexpr uint32_t cb_reduction_scaler = 5;
constexpr uint32_t cb_matmul_reduce = 6;
constexpr uint32_t cb_attention_weights = 7;
constexpr uint32_t cb_cur_max = 9;
// Refinement 6: cb_cur_sum_exp_for_divide (slot 11) — untagged
// single-tile copy of cb_cur_sum_exp's reciprocal, source for the FPU
// mul_tiles_bcast_cols final divide on the UnpackToDestFp32-tagged path.
// Only allocated by the program descriptor when USE_UNTAGGED_DIVIDE.
constexpr uint32_t cb_cur_sum_exp_for_divide = 11;
constexpr uint32_t cb_cur_sum_exp = 12;

constexpr uint32_t cb_output = 16;
// Refinement 6: cb_cur_mm_out_for_divide (slot 20) — untagged Dt-tile
// copy of cb_cur_mm_out, source for the FPU final divide. Same gating
// as cb_cur_sum_exp_for_divide.
constexpr uint32_t cb_cur_mm_out_for_divide = 20;
constexpr uint32_t cb_cur_mm_out = 21;

// ===========================================================================
// Helper functions (modeled on tt-train SDPA reference). All ping-ponged
// CBs are passed as parameters so the caller can swap aliases per K-iter.
// ===========================================================================

// Add an additive mask tile to the QK^T result in DST.
FORCE_INLINE void apply_additive_mask_on_reg(uint32_t scores_reg) {
    const uint32_t mask_reg = scores_reg + 1U;
    cb_wait_front(cb_attn_mask, onetile);

    copy_tile_to_dst_init_short_with_dt(cb_query, cb_attn_mask);
    copy_tile(cb_attn_mask, 0, mask_reg);

    add_binary_tile_init();
    add_binary_tile(scores_reg, mask_reg, scores_reg);
}

// attn @ V[k] -> cur_mm_out_cb (Dt tiles).
FORCE_INLINE void matmul_attn_by_v(uint32_t cur_mm_out_cb) {
    cb_wait_front(cb_attention_weights, onetile);
    cb_wait_front(cb_value, Dt);
    cb_reserve_back(cur_mm_out_cb, Dt);

    reconfig_data_format(cb_value, cb_attention_weights);
    mm_init_short(cb_attention_weights, cb_value, /*transpose*/ 0);
    pack_reconfig_data_format(cur_mm_out_cb);

    for (uint32_t d = 0; d < Dt; ++d) {
        tile_regs_acquire();
        matmul_tiles(cb_attention_weights, cb_value, /*A tile*/ 0, /*B tile*/ d, /*dst*/ 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cur_mm_out_cb);
        tile_regs_release();
    }
    cb_push_back(cur_mm_out_cb, Dt);
}

// Pass-1 max-only: in-place update of cb_cur_max from cb_attention_weights.
// k=0: cb_cur_max = row_max(scores).
// k>0: cb_cur_max = max(cb_cur_max, row_max(scores)) [in-place pop+push].
// Does NOT pop cb_attention_weights — caller does after.
//
// R4-iter3 collapsed the OLD pass-1 ping-pong (cb_prev_max ↔ cb_cur_max)
// because no correction-multiply happens in pass 1 — only an eltwise max
// against the previous iter's value. R5 then deleted the now-unused
// cb_prev_max descriptor to reclaim its L1 slot.
FORCE_INLINE void update_cur_max_inplace(uint32_t cb_cur_max, bool init_mode) {
    cb_wait_front(cb_attention_weights, onetile);
    if (!init_mode) {
        cb_wait_front(cb_cur_max, onetile);
    }

    const uint32_t reduce_dst = 0;
    const uint32_t prev_max_dst = 1U;

    reconfig_data_format(cb_attention_weights, cb_reduction_scaler);
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_attention_weights, cb_reduction_scaler, cb_cur_max);

    tile_regs_acquire();
    reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(
        cb_attention_weights, cb_reduction_scaler, /*tile_idx*/ 0, /*tile_idx*/ 0, reduce_dst);
    reduce_uninit();

    if (!init_mode) {
        copy_tile_to_dst_init_short_with_dt(cb_attention_weights, cb_cur_max);
        copy_tile(cb_cur_max, 0, prev_max_dst);

        binary_max_tile_init();
        binary_max_tile(reduce_dst, prev_max_dst, reduce_dst, VectorMode::C);
    }
    tile_regs_commit();

    if (!init_mode) {
        cb_pop_front(cb_cur_max, onetile);
    }
    cb_reserve_back(cb_cur_max, onetile);
    tile_regs_wait();
    pack_reconfig_data_format(cb_cur_max);
    pack_tile(reduce_dst, cb_cur_max);
    tile_regs_release();
    cb_push_back(cb_cur_max, onetile);
}

// Pass-2 sum-exp accumulator. Each iter, computes row_sum(attn) via the
// matmul-with-ones tile and either initialises cb_cur_sum_exp (k=0) or
// in-place adds the new row-sum into cb_cur_sum_exp (k>0). Does NOT pop
// cb_attention_weights — downstream S@V matmul reads it next.
//
// This replaces R4-iter2's "track sum_exp in pass 1 via cur_sum_exp
// running cascade" path. Pass 1 had `cur_sum_exp = prev * corr + row_sum`
// with corr = exp(prev_max - cur_max) at each iter — a multiplicative
// cascade on the column-vector accumulator that, despite being narrow,
// propagated into the final-divide denominator and floored RMS at the
// pre-R4 level. Direct accumulation in pass 2 against the FIXED global
// max (already in cb_cur_max post-pass-1) is a simple sum, sqrt(Kt)
// error growth instead of linear-Kt cascade.
FORCE_INLINE void update_cur_sum_exp_pass2(uint32_t cb_cur_sum_exp, bool init_mode) {
    cb_wait_front(cb_attention_weights, onetile);
    cb_wait_front(cb_matmul_reduce, onetile);
    if (!init_mode) {
        cb_wait_front(cb_cur_sum_exp, onetile);
    }

    const uint32_t dst_sum = 0;
    const uint32_t dst_prev = 1U;

    // row_sum via matmul-with-ones (in1 = col-0-ones tile).
    reconfig_data_format(cb_attention_weights, cb_matmul_reduce);
    mm_init(cb_attention_weights, cb_matmul_reduce, cb_cur_sum_exp, /*transpose*/ 0);

    tile_regs_acquire();
    matmul_tiles(cb_attention_weights, cb_matmul_reduce, /*A tile*/ 0, /*B tile*/ 0, dst_sum);

    if (!init_mode) {
        // DST[1] = existing cb_cur_sum_exp (column vector value in col 0).
        copy_tile_to_dst_init_short_with_dt(cb_matmul_reduce, cb_cur_sum_exp);
        copy_tile(cb_cur_sum_exp, 0, dst_prev);

        add_binary_tile_init();
        add_binary_tile(dst_sum, dst_prev, dst_sum);
    }
    tile_regs_commit();

    if (!init_mode) {
        cb_pop_front(cb_cur_sum_exp, onetile);
    }
    cb_reserve_back(cb_cur_sum_exp, onetile);
    tile_regs_wait();
    pack_reconfig_data_format(cb_cur_sum_exp);
    pack_tile(dst_sum, cb_cur_sum_exp);
    tile_regs_release();
    cb_push_back(cb_cur_sum_exp, onetile);
}

// Pass-2 variant of apply_exp_inplace_and_find_exp_sum: subtracts a FIXED
// global_max from scores, exps, in-place rewrites cb_attention_weights.
// Does NOT mirror to a sum_exp CB (sum_exp now lives in pass-2 directly).
// global_max_cb stays fronted across the call — caller is responsible
// for waiting and holding.
FORCE_INLINE void apply_exp_inplace_with_global_max(uint32_t global_max_cb) {
    cb_wait_front(cb_attention_weights, onetile);
    cb_wait_front(global_max_cb, onetile);

    const uint32_t exp_dst = 0;
    reconfig_data_format(cb_attention_weights, global_max_cb);
    sub_bcast_cols_init_short(cb_attention_weights, global_max_cb);

    tile_regs_acquire();
    sub_tiles_bcast_cols(cb_attention_weights, global_max_cb, 0, 0, exp_dst);

    // Fused scale+exp via the SFPU exp's scale_en mode.
    exp_tile_init</*approx*/ false, scaler_fp32>();
    exp_tile</*approx*/ false, /*scale_en*/ true>(exp_dst, VectorMode::RC, scaler_bf16);
    tile_regs_commit();

    tile_regs_wait();
    // In-place rewrite of cb_attention_weights.
    cb_pop_front(cb_attention_weights, onetile);
    cb_reserve_back(cb_attention_weights, onetile);
    pack_reconfig_data_format(cb_attention_weights);
    pack_tile(exp_dst, cb_attention_weights);
    tile_regs_release();
    cb_push_back(cb_attention_weights, onetile);
}

// Pass-2 accumulator: cb_cur_mm_out[d] += attn @ V[k][d], for all d in
// [0, Dt). Reads the existing Dt-tile accumulator in-place via the
// pop/reserve/pack/push cycle, same pattern as update_cur_mm_out_inplace
// but without the col-broadcast `corr` mul — the new partial is just
// added to the accumulator. CB must be sized 2*Dt for double-buffer.
//
// One tile_regs window per d covers (a) the single-tile matmul partial,
// (b) the read of cb_cur_mm_out[d], and (c) the add. mm_init_short is
// re-issued inside the loop because copy_tile + add_binary_tile clobber
// the matmul state (mirroring update_cur_mm_out_inplace's pattern).
FORCE_INLINE void matmul_attn_by_v_accumulate(uint32_t cb_cur_mm_out) {
    cb_wait_front(cb_attention_weights, onetile);
    cb_wait_front(cb_value, Dt);
    cb_wait_front(cb_cur_mm_out, Dt);
    cb_reserve_back(cb_cur_mm_out, Dt);

    const uint32_t dst_mul = 0;
    const uint32_t dst_cur = 1U;

    for (uint32_t d = 0; d < Dt; ++d) {
        tile_regs_acquire();

        // DST[0] = attn @ V[k][d] (single-tile matmul; K-axis is 1 tile
        // wide so one matmul_tiles call computes the full product).
        reconfig_data_format(cb_value, cb_attention_weights);
        mm_init_short(cb_attention_weights, cb_value, /*transpose*/ 0);
        matmul_tiles(cb_attention_weights, cb_value, /*A tile*/ 0, /*B tile*/ d, dst_mul);

        // DST[1] = cb_cur_mm_out[d] (existing accumulator value).
        copy_tile_to_dst_init_short_with_dt(cb_value, cb_cur_mm_out);
        copy_tile(cb_cur_mm_out, d, dst_cur);

        // DST[0] = DST[0] + DST[1] = new accumulator value.
        add_binary_tile_init();
        add_binary_tile(dst_mul, dst_cur, dst_mul);
        tile_regs_commit();

        tile_regs_wait();
        pack_reconfig_data_format(cb_cur_mm_out);
        pack_tile(dst_mul, cb_cur_mm_out);
        tile_regs_release();
    }
    // Swap fronted set: drop the original Dt (stale), commit new Dt.
    cb_pop_front(cb_cur_mm_out, Dt);
    cb_push_back(cb_cur_mm_out, Dt);
}

// Row-reduce a sum_exp tile via matmul-with-ones; result is a column
// vector (value in col 0 of each row). In-place rewrite of cb_in.
FORCE_INLINE void row_reduce_tile_inplace(uint32_t cb_in_idx) {
    cb_wait_front(cb_in_idx, onetile);
    cb_wait_front(cb_matmul_reduce, onetile);

    const uint32_t dst = 0;
    reconfig_data_format(cb_matmul_reduce, cb_in_idx);

    mm_init(cb_in_idx, cb_matmul_reduce, cb_in_idx, /*transpose*/ 0);
    tile_regs_acquire();
    matmul_tiles(cb_in_idx, cb_matmul_reduce, 0, 0, dst);
    tile_regs_commit();

    tile_regs_wait();
    cb_pop_front(cb_in_idx, onetile);
    cb_reserve_back(cb_in_idx, onetile);
    pack_reconfig_data_format(cb_in_idx);
    pack_tile(dst, cb_in_idx);
    tile_regs_release();
    cb_push_back(cb_in_idx, onetile);
}

// Reciprocal of a single-tile column vector (data in col 0).
FORCE_INLINE void recip_tile_inplace(uint32_t cb_in_idx) {
    cb_wait_front(cb_in_idx, onetile);
    const uint32_t dst = 0;

    tile_regs_acquire();
    reconfig_data_format(cb_in_idx, cb_in_idx);
    copy_tile_init(cb_in_idx);
    copy_tile(cb_in_idx, 0, dst);

    recip_tile_init</*legacy_compat*/ false>();
    recip_tile(dst);
    tile_regs_commit();

    tile_regs_wait();
    cb_pop_front(cb_in_idx, onetile);
    cb_reserve_back(cb_in_idx, onetile);
    pack_reconfig_data_format(cb_in_idx);
    pack_tile(dst, cb_in_idx);
    tile_regs_release();
    cb_push_back(cb_in_idx, onetile);
}

// ===========================================================================
// Per-row pipeline
// ===========================================================================

FORCE_INLINE void process_single_row() {
    cb_wait_front(cb_query, Dt);

    const uint32_t mm_dst = 0;

    // =====================================================================
    // PASS 1 — global_max ONLY.
    // Each K-iter: Q@K^T (+mask), reduce_row(max), in-place update of
    // cb_cur_max. No exp, no sum_exp, no correction cascade, no
    // ping-pong. cb_cur_sum_exp / cb_cur_mm_out are touched only in
    // pass 2 (so we don't carry their state across the K-loop here).
    // V is NOT consumed (reader pass-1 doesn't push it).
    // =====================================================================
    for (uint32_t k_i = 0; k_i < Kt; ++k_i) {
        // Q @ K[k]^T -> cb_attention_weights.
        cb_wait_front(cb_key, Dt);

        reconfig_data_format(cb_query, cb_key);
        mm_init_short(cb_query, cb_key, /*transpose*/ 1);

        tile_regs_acquire();
        for (uint32_t d = 0; d < Dt; ++d) {
            matmul_tiles(cb_query, cb_key, /*A tile*/ d, /*B tile*/ d, mm_dst);
        }
        if constexpr (HAS_MASK) {
            apply_additive_mask_on_reg(mm_dst);
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_attention_weights, onetile);
        pack_reconfig_data_format(cb_attention_weights);
        pack_tile(mm_dst, cb_attention_weights);
        tile_regs_release();
        cb_push_back(cb_attention_weights, onetile);

        if constexpr (HAS_MASK) {
            cb_pop_front(cb_attn_mask, onetile);
        }
        cb_pop_front(cb_key, Dt);

        // Update cb_cur_max in-place (init on k=0, eltwise-max on k>0).
        update_cur_max_inplace(cb_cur_max, /*init_mode*/ k_i == 0);

        // Drop the score tile — pass 1 doesn't need exp / S@V / sum_exp.
        cb_pop_front(cb_attention_weights, onetile);
    }

    // After pass 1: cb_cur_max has 1 tile (global_max), held into pass 2
    // and the final divide.
    cb_wait_front(cb_cur_max, onetile);

    // =====================================================================
    // PASS 2 — global_sum_exp + output accumulation against FIXED
    // global_max.
    //
    // Each K-iter:
    //   - re-matmul Q @ K[k]^T (reader re-pushed K)
    //   - mask add (reader re-pushed mask)
    //   - exp(scale * (scores - global_max)) in-place on cb_attention_weights
    //   - row_sum(attn) accumulated into cb_cur_sum_exp (direct add)
    //   - attn @ V[k] accumulated into cb_cur_mm_out (direct add)
    //   - pop cb_attention_weights, cb_value, mask
    //
    // Both running CBs accumulate via simple addition — no correction
    // cascade in either. Each per-iter precision loss bounded by TF32
    // unpack (~10 mantissa bits preserved); cumulative error grows as
    // sqrt(Kt) instead of linear-Kt, dropping S=8192 fp32 RMS from
    // 0.053 (pre-R4) to under the 0.02 target.
    // =====================================================================
    for (uint32_t k_i = 0; k_i < Kt; ++k_i) {
        // Q @ K[k]^T -> cb_attention_weights (re-matmul; K re-read by reader).
        cb_wait_front(cb_key, Dt);

        reconfig_data_format(cb_query, cb_key);
        mm_init_short(cb_query, cb_key, /*transpose*/ 1);

        tile_regs_acquire();
        for (uint32_t d = 0; d < Dt; ++d) {
            matmul_tiles(cb_query, cb_key, /*A tile*/ d, /*B tile*/ d, mm_dst);
        }
        if constexpr (HAS_MASK) {
            apply_additive_mask_on_reg(mm_dst);
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_attention_weights, onetile);
        pack_reconfig_data_format(cb_attention_weights);
        pack_tile(mm_dst, cb_attention_weights);
        tile_regs_release();
        cb_push_back(cb_attention_weights, onetile);

        if constexpr (HAS_MASK) {
            cb_pop_front(cb_attn_mask, onetile);
        }
        cb_pop_front(cb_key, Dt);

        // attn = exp(scale * (scores - global_max)), in-place rewrite.
        apply_exp_inplace_with_global_max(cb_cur_max);

        // cb_cur_sum_exp += row_sum(attn) — init on k=0, in-place add on k>0.
        update_cur_sum_exp_pass2(cb_cur_sum_exp, /*init_mode*/ k_i == 0);

        // attn @ V[k] -> cb_cur_mm_out — init on k=0, in-place add on k>0.
        if (k_i == 0) {
            matmul_attn_by_v(cb_cur_mm_out);
        } else {
            matmul_attn_by_v_accumulate(cb_cur_mm_out);
        }

        cb_pop_front(cb_attention_weights, onetile);
        cb_pop_front(cb_value, Dt);
    }

    // ---- Final divide: output = cb_cur_mm_out * (1 / cb_cur_sum_exp), col bcast ----
    //
    // Refinement 6: when USE_UNTAGGED_DIVIDE, cb_cur_sum_exp and
    // cb_cur_mm_out carry the UnpackToDestFp32 tag — the FPU
    // mul_tiles_bcast_cols cannot consume them directly. SFPU-copy the
    // (post-reciprocal) sum_exp and the Dt mm_out tiles into the
    // untagged cb_*_for_divide CBs first, then run the FPU divide.
    //
    // Both copies are pure SFPU copy_tile, compatible with the
    // UnpackToDestFp32 tag on the source. Each copy preserves full FP32
    // through DST (one ULP rounding when pack_tile writes back to the
    // untagged CB, which then unpacks through srcA/srcB as TF32 — but
    // that's a SINGLE ULP per tile, not the per-K-iter cumulative
    // sqrt(Kt) TF32 cascade that floored RMS at 0.027 for S=8192).
    //
    // When !USE_UNTAGGED_DIVIDE the running-state CBs are bf16-formatted
    // and FPU-readable — we take the direct path (same as pre-R6).
    cb_wait_front(cb_cur_sum_exp, onetile);
    recip_tile_inplace(cb_cur_sum_exp);
    cb_wait_front(cb_cur_sum_exp, onetile);

    if constexpr (USE_UNTAGGED_DIVIDE) {
        // ---- (a) copy cb_cur_sum_exp (reciprocal already in place) ----
        //   → cb_cur_sum_exp_for_divide via SFPU copy_tile.
        cb_reserve_back(cb_cur_sum_exp_for_divide, onetile);
        tile_regs_acquire();
        copy_tile_to_dst_init_short_with_dt(cb_query, cb_cur_sum_exp);
        copy_tile(cb_cur_sum_exp, 0, /*dst*/ 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_cur_sum_exp_for_divide);
        pack_tile(0, cb_cur_sum_exp_for_divide);
        tile_regs_release();
        cb_push_back(cb_cur_sum_exp_for_divide, onetile);

        // ---- (b) copy cb_cur_mm_out (Dt tiles) → cb_cur_mm_out_for_divide ----
        cb_wait_front(cb_cur_mm_out, Dt);
        cb_reserve_back(cb_cur_mm_out_for_divide, Dt);
        for (uint32_t d = 0; d < Dt; ++d) {
            tile_regs_acquire();
            copy_tile_to_dst_init_short_with_dt(cb_query, cb_cur_mm_out);
            copy_tile(cb_cur_mm_out, d, /*dst*/ 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_cur_mm_out_for_divide);
            pack_tile(0, cb_cur_mm_out_for_divide);
            tile_regs_release();
        }
        cb_push_back(cb_cur_mm_out_for_divide, Dt);

        // ---- (c) FPU mul on the untagged intermediates ----
        cb_wait_front(cb_cur_sum_exp_for_divide, onetile);
        cb_wait_front(cb_cur_mm_out_for_divide, Dt);
        cb_reserve_back(cb_output, Dt);
        reconfig_data_format(cb_cur_mm_out_for_divide, cb_cur_sum_exp_for_divide);
        mul_bcast_cols_init_short(cb_cur_mm_out_for_divide, cb_cur_sum_exp_for_divide);
        pack_reconfig_data_format(cb_output);

        for (uint32_t d = 0; d < Dt; ++d) {
            tile_regs_acquire();
            mul_tiles_bcast_cols(cb_cur_mm_out_for_divide, cb_cur_sum_exp_for_divide, d, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_output);
            tile_regs_release();
        }
        cb_push_back(cb_output, Dt);

        // ---- Cleanup intermediates ----
        cb_pop_front(cb_cur_sum_exp_for_divide, onetile);
        cb_pop_front(cb_cur_mm_out_for_divide, Dt);
    } else {
        // Direct path — untagged running-state CBs (bf16 when
        // fp32_dest_acc_en=False) feed mul_tiles_bcast_cols directly.
        cb_wait_front(cb_cur_mm_out, Dt);
        cb_reserve_back(cb_output, Dt);
        reconfig_data_format(cb_cur_mm_out, cb_cur_sum_exp);
        mul_bcast_cols_init_short(cb_cur_mm_out, cb_cur_sum_exp);
        pack_reconfig_data_format(cb_output);

        for (uint32_t d = 0; d < Dt; ++d) {
            tile_regs_acquire();
            mul_tiles_bcast_cols(cb_cur_mm_out, cb_cur_sum_exp, d, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_output);
            tile_regs_release();
        }
        cb_push_back(cb_output, Dt);
    }

    // ---- Cleanup: drop all per-row state CBs ----
    cb_pop_front(cb_cur_max, onetile);
    cb_pop_front(cb_cur_sum_exp, onetile);
    cb_pop_front(cb_cur_mm_out, Dt);
    cb_pop_front(cb_query, Dt);
}

// ===========================================================================
// kernel_main
// ===========================================================================

void kernel_main() {
    const uint32_t num_rows = get_arg_val<uint32_t>(0);
    const uint32_t start_row = get_arg_val<uint32_t>(1);
    (void)start_row;  // The reader/writer drive (b, h, qt) decoding;
                      // compute only needs the count.

    // Boot-time init (mirrors the tt-train SDPA reference). MMIO-unsafe to
    // re-issue mid-kernel.
    init_sfpu(cb_query, cb_output);
    binary_op_init_common(cb_query, cb_key, cb_value);
    mm_init(cb_query, cb_key, cb_attention_weights);

    // One-shot scaler tiles — fronted for the lifetime of the kernel.
    cb_wait_front(cb_reduction_scaler, onetile);
    cb_wait_front(cb_matmul_reduce, onetile);

    for (uint32_t r = 0; r < num_rows; ++r) {
        process_single_row();
    }
}
