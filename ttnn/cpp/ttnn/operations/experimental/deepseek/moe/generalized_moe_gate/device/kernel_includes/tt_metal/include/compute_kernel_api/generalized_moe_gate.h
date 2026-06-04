// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/transpose_wh.h"
#ifdef TRISC_MATH
#ifdef ARCH_BLACKHOLE
#include "../../hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_generalized_moe_gate_topk_single_face.h"
#include "../../hw/ckernels/blackhole/metal/llk_api/llk_math_generalized_moe_gate_eltwise_binary_api.h"
#include "../../hw/ckernels/blackhole/metal/llk_api/llk_math_generalized_moe_gate_transpose_dest_single_face_api.h"
#else
#include "../../hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_generalized_moe_gate_topk_single_face.h"
#include "../../hw/ckernels/wormhole_b0/metal/llk_api/llk_math_generalized_moe_gate_eltwise_binary_api.h"
#include "../../hw/ckernels/wormhole_b0/metal/llk_api/llk_math_generalized_moe_gate_transpose_dest_single_face_api.h"
#endif
#endif

namespace ckernel {

template <bool enable_sigmoid = false, bool is_32bit = false>
ALWI void generalized_moe_gate_init(uint32_t icb0, uint32_t icb1) {
    if constexpr (enable_sigmoid) {
        // Init sigmoid (SFPU)
        sigmoid_tile_init<false>();
        // Init transpose wh (FPU)
        transpose_wh_init_short(icb0);
    } else {
        // Init copy add (FPU)
        UNPACK((llk_unpack_AB_init<BroadcastType::NONE>(icb0, icb1, Transpose::Both)));
        MATH((llk_math_generalized_moe_gate_eltwise_binary_init_with_operands<
              EltwiseBinaryType::ELWADD,
              GeneralizedMoeGateEltwiseBinaryMode::COPY,
              MATH_FIDELITY>(icb0, icb1, false)));
        // Init transpose dest addrmods (does not conflict with copy add)
        MATH((llk_math_generalized_moe_gate_transpose_dest_single_face_common_init<is_32bit>()));
        // Init topk (SFPU)
        MATH((llk_math_sfpu_generalized_moe_gate_topk_init<APPROX, DST_ACCUM_MODE>()));
    }
}

template <bool enable_sigmoid = false, bool is_32bit = false>
ALWI void generalized_moe_gate(uint32_t icb0, uint32_t icb1, uint32_t eps, uint32_t scale) {
    if constexpr (enable_sigmoid) {
        // Transpose wh (FPU)
        transpose_wh_tile(icb0, 0, 0);
        // Sigmoid (SFPU)
        sigmoid_tile<VectorMode::RC_custom, false>(0);
        // Init add binary reuse (FPU)
        UNPACK((llk_unpack_A_init<BroadcastType::NONE, true, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
            false, false, icb1)));
        MATH((llk_math_generalized_moe_gate_eltwise_binary_init_with_operands<
              EltwiseBinaryType::ELWADD,
              GeneralizedMoeGateEltwiseBinaryMode::RELOAD,
              MATH_FIDELITY>(icb1, icb1, false)));
        // Add binary reuse (FPU)
        UNPACK((llk_unpack_A<BroadcastType::NONE, true, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(icb1, 0)));
        MATH((llk_math_generalized_moe_gate_eltwise_binary<EltwiseBinaryType::ELWADD, DST_ACCUM_MODE, MATH_FIDELITY>(
            icb1, icb1, 0, true)));
        // Init transpose dest addrmods (does not conflict with add binary reuse)
        MATH((llk_math_generalized_moe_gate_transpose_dest_single_face_common_init<is_32bit>()));
        // Init topk (SFPU)
        MATH((llk_math_sfpu_generalized_moe_gate_topk_init<APPROX, DST_ACCUM_MODE>()));
    } else {
        // Copy add (FPU)
        UNPACK((llk_unpack_AB(icb0, icb1, 0, 0)));
        MATH((llk_math_generalized_moe_gate_eltwise_binary<EltwiseBinaryType::ELWADD, DST_ACCUM_MODE, MATH_FIDELITY>(
            icb0, icb1, 0, true)));
    }
    // Set srcb dummy valid for transpose wh (FPU)
    UNPACK((llk_unpack_set_srcb_dummy_valid()));
    // Sum top2 (SFPU)
    MATH((llk_math_sfpu_generalized_moe_gate_sum_top2<APPROX, DST_ACCUM_MODE>(0)));
#ifdef GMG_DUMP_AFTER_SUM_TOP2
    // DEBUG PROBE: stop here so the post-sum_top2 DEST layout (per-group top-8) can be
    // packed out and inspected. See generalized_moe_gate_kernel.cpp.
    return;
#endif
    // Transpose dest step 0 (FPU) — always runs; puts each group g at DEST row g.
    MATH((llk_math_generalized_moe_gate_transpose_dest_single_face_step0_init<is_32bit>()));
    MATH((llk_math_generalized_moe_gate_transpose_dest_single_face_step0<DST_ACCUM_MODE, is_32bit>()));
#ifdef GMG_DUMP_AFTER_STEP0
    // DEBUG PROBE: inspect the cross-group transpose — where do the 8 groups land after step0?
    return;
#endif
#ifdef GMG_UNGROUPED_TOP8
    // TRUE GLOBAL TOP-8 over all 256 experts (ungrouped). post-step0: group g at DEST row g.
    // SFPU merges stay in rows 0-7 (only rows 0-7 are SFPU-addressable); FPU copy4rows (a plain
    // MOVD2B->MOVB2D copy) stashes 4-row blocks in rows 8-15. topA -> {0,2} (rows 0-3), topB -> {4,6}
    // (rows 4-7): row-disjoint. Each copy4rows uses a DISJOINT SrcB scratch window (16/20/24/28) so a
    // later MOVB2D can't read a previous (back-to-back) copy's SrcB leftover.
    //
    // save groups 4-7 source (rows 4-7) -> rows 8-11 (step1<0> below clobbers rows 0-7).
    MATH((llk_math_generalized_moe_gate_copy4rows_init<4, 8, is_32bit, 16>()));
    MATH((llk_math_generalized_moe_gate_copy4rows<DST_ACCUM_MODE, is_32bit>()));
    // topA = top8(groups 0-3): step1<d2b_dst=0> -> run at rows 0-7 -> merge -> topA at {0,2}.
    MATH((llk_math_generalized_moe_gate_transpose_dest_single_face_step1_hi_init<0, 0, is_32bit>()));
    MATH((llk_math_generalized_moe_gate_transpose_dest_single_face_step1_hi<DST_ACCUM_MODE, is_32bit>()));
    MATH((llk_math_sfpu_generalized_moe_gate_merge4_top8<APPROX, DST_ACCUM_MODE, 0, 0, 2>(0)));
    // park topA (rows 0-3) -> rows 12-15; restore groups 4-7 (rows 8-11) -> rows 4-7.
    MATH((llk_math_generalized_moe_gate_copy4rows_init<0, 12, is_32bit, 20>()));
    MATH((llk_math_generalized_moe_gate_copy4rows<DST_ACCUM_MODE, is_32bit>()));
    MATH((llk_math_generalized_moe_gate_copy4rows_init<8, 4, is_32bit, 24>()));
    MATH((llk_math_generalized_moe_gate_copy4rows<DST_ACCUM_MODE, is_32bit>()));
    // topB = top8(groups 4-7): step1_hi<d2b_dst=4> -> run at rows 0-7 -> merge -> topB at {4,6}.
    MATH((llk_math_generalized_moe_gate_transpose_dest_single_face_step1_hi_init<4, 0, is_32bit>()));
    MATH((llk_math_generalized_moe_gate_transpose_dest_single_face_step1_hi<DST_ACCUM_MODE, is_32bit>()));
    MATH((llk_math_sfpu_generalized_moe_gate_merge4_top8<APPROX, DST_ACCUM_MODE, 0, 4, 6>(0)));
    // restore topA (rows 12-15) -> rows 0-3; now topA@{0,2} (rows 0-3), topB@{4,6} (rows 4-7).
    MATH((llk_math_generalized_moe_gate_copy4rows_init<12, 0, is_32bit, 28>()));
    MATH((llk_math_generalized_moe_gate_copy4rows<DST_ACCUM_MODE, is_32bit>()));
#ifdef GMG_DIAG_TOPA
    // ISOLATION DIAGNOSTIC (debug aid for generalization): output topA ALONE. Move topA {0,2} -> {0,4}
    // + normalize; pair with a top8(groups 0-3) golden in op.py.
    MATH((llk_math_sfpu_generalized_moe_gate_copy_topk_run<APPROX, DST_ACCUM_MODE, 0, 2, 0, 4>(0)));
    MATH((llk_math_sfpu_generalized_moe_gate_normalize_run<APPROX, DST_ACCUM_MODE>(0, eps, scale)));
#elif defined(GMG_DIAG_TOPB)
    // ISOLATION DIAGNOSTIC (debug aid): output topB ALONE (sits at {4,6}); pair with a top8(groups 4-7)
    // golden. NOTE: copy_topk_run/normalize_run reset Dst RWC (SETRWC) — required after the FPU copy4rows.
    MATH((llk_math_sfpu_generalized_moe_gate_copy_topk_run<APPROX, DST_ACCUM_MODE, 4, 6, 0, 4>(0)));
    MATH((llk_math_sfpu_generalized_moe_gate_normalize_run<APPROX, DST_ACCUM_MODE>(0, eps, scale)));
#else
    // finalize: full bitonic sort of the 16 candidates (topA{0,2} + topB{4,6}) -> global top-8 + normalize.
    MATH((llk_math_sfpu_generalized_moe_gate_finalize_ungrouped<APPROX, DST_ACCUM_MODE>(0, eps, scale)));
#endif
#else
    // Grouped DeepSeek gate: sort_top4 selects top-4 groups, step1 lays them out, top8 merges.
    MATH((llk_math_sfpu_generalized_moe_gate_sort_top4_groups<APPROX, DST_ACCUM_MODE>(0)));
    MATH((llk_math_generalized_moe_gate_transpose_dest_single_face_step1_init<is_32bit>()));
    MATH((llk_math_generalized_moe_gate_transpose_dest_single_face_step1<DST_ACCUM_MODE, is_32bit>()));
#ifdef GMG_DUMP_AFTER_STEP1
    // DEBUG PROBE: stop here to inspect the orientation the (proven) top8 bitonic merge consumes.
    return;
#endif
    MATH((llk_math_sfpu_generalized_moe_gate_top8<APPROX, DST_ACCUM_MODE>(0, eps, scale)));
#endif  // GMG_UNGROUPED_TOP8
    // Transpose dest step 2 (FPU)
    MATH((llk_math_generalized_moe_gate_transpose_dest_single_face_step2_init<is_32bit>()));
    MATH((llk_math_generalized_moe_gate_transpose_dest_single_face_step2<DST_ACCUM_MODE, is_32bit>()));
}

}  // namespace ckernel
