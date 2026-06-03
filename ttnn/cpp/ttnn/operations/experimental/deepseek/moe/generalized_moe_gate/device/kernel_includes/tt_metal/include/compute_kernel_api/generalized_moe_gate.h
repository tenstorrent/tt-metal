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
#ifdef GMG_PROBE_LANEMAP
    // P1: pure load/store lane-map probe (see ckernel_sfpu_..._single_face.h). Dump bias region.
    MATH((llk_math_sfpu_generalized_moe_gate_probe_lanemap<APPROX, DST_ACCUM_MODE>(0)));
    return;
#endif
#ifdef GMG_DUMP_AFTER_SUM_TOP2
    // DEBUG PROBE: stop here so the post-sum_top2 DEST layout (per-group top-8) can be
    // packed out and inspected. See generalized_moe_gate_kernel.cpp.
    return;
#endif
#ifdef GMG_UNGROUPED_TOP8
    // UNGROUPED (DRAFT): true global top-8 over all 8 groups. Skip group selection
    // (step0 / sort_top4 / step1) entirely and merge all 8 per-group top-8 runs that
    // sum_top2 left at cols {0,2,...,14}. Hypothesis: no transpose is needed before the
    // merge (step0/step1 existed only for the group sort). VERIFY against the golden test.
    MATH((llk_math_sfpu_generalized_moe_gate_top8_ungrouped<APPROX, DST_ACCUM_MODE>(0, eps, scale)));
#else
    // Transpose dest step 0 (FPU)
    MATH((llk_math_generalized_moe_gate_transpose_dest_single_face_step0_init<is_32bit>()));
    MATH((llk_math_generalized_moe_gate_transpose_dest_single_face_step0<DST_ACCUM_MODE, is_32bit>()));
#ifdef GMG_DUMP_AFTER_STEP0
    // DEBUG PROBE: inspect the cross-group transpose — where do the 8 groups land after step0?
    return;
#endif
#ifdef GMG_PROBE_OFFSETS
    // COORDINATE PROBE: overwrite the indices region with (1000+offset) at offsets {0,2,4,6,16,18,
    // 20,22}, then dump -> maps each SFPLOAD offset to its display footprint at the post-step0 layout.
    MATH((llk_math_sfpu_generalized_moe_gate_probe_offsets<APPROX, DST_ACCUM_MODE>(0)));
    return;
#endif
#ifdef GMG_TEST_HI_GROUPS
    // P4 (step1_hi): instead of sort_top4 + collapsing the LOW half, run a tunable step1 variant
    // that collapses the HIGH 4 groups (4-7) into runs. Tune GMG_HI_D2B_DST / GMG_HI_B2D_BASE via
    // the GMG_DUMP_AFTER_STEP1 dump until the merge slot (cols 0,2,4,6) holds groups 4-7's runs.
    MATH((llk_math_generalized_moe_gate_transpose_dest_single_face_step1_hi_init<
          GMG_HI_D2B_DST,
          GMG_HI_B2D_BASE,
          is_32bit>()));
    MATH((llk_math_generalized_moe_gate_transpose_dest_single_face_step1_hi<DST_ACCUM_MODE, is_32bit>()));
#else
    // Sort top4 groups (SFPU)
#if !defined(GMG_SKIP_SORT_TOP4)
    MATH((llk_math_sfpu_generalized_moe_gate_sort_top4_groups<APPROX, DST_ACCUM_MODE>(0)));
#endif
    // Transpose dest step 1 (FPU)
    MATH((llk_math_generalized_moe_gate_transpose_dest_single_face_step1_init<is_32bit>()));
    MATH((llk_math_generalized_moe_gate_transpose_dest_single_face_step1<DST_ACCUM_MODE, is_32bit>()));
#endif  // GMG_TEST_HI_GROUPS
#ifdef GMG_DUMP_AFTER_STEP1
    // DEBUG PROBE: stop here to inspect the orientation the (proven) top8 bitonic merge consumes.
    return;
#endif
    // Top8 (SFPU)
    MATH((llk_math_sfpu_generalized_moe_gate_top8<APPROX, DST_ACCUM_MODE>(0, eps, scale)));
#endif  // GMG_UNGROUPED_TOP8
    // Transpose dest step 2 (FPU)
    MATH((llk_math_generalized_moe_gate_transpose_dest_single_face_step2_init<is_32bit>()));
    MATH((llk_math_generalized_moe_gate_transpose_dest_single_face_step2<DST_ACCUM_MODE, is_32bit>()));
}

}  // namespace ckernel
