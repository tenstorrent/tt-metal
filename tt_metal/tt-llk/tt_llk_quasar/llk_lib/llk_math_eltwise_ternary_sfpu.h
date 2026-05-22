// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <utility>

#include "ckernel_sfpu.h"
#include "ckernel_trisc_common.h"
#include "llk_defs.h"
#include "llk_math_eltwise_sfpu_common.h"

/**
 * @brief Initializes shared SFPU state for ternary element-wise operations.
 *
 * Programs ADDR_MOD_7 with dest.incr=0 via the common SFPU init. Per-op
 * state (e.g. ADDR_MOD_6 for ops that auto-advance dest) is set up by the
 * op's own @c _init_<op>_ call after this one.
 *
 * @tparam sfpu_op  The ternary SFPU operation type (e.g. @c SfpuType::where).
 */
template <SfpuType sfpu_op>
inline void _llk_math_eltwise_ternary_sfpu_init_()
{
    _llk_math_sfpu_init_();
}

/**
 * @brief Dispatches a ternary SFPU kernel over all faces of the current DEST section.
 *
 * Sets the DEST section base to tile 0, calls @p sfpu_func once per face with the
 * supplied tile indices, advances the face pointer between calls, then signals
 * SFPU done.
 *
 * @tparam F              Callable type matching the ternary SFPU kernel signature.
 * @tparam ARGS           Any extra arguments forwarded verbatim to @p sfpu_func.
 *
 * @param sfpu_func       Ternary SFPU kernel (e.g. @c _calculate_where_<false>).
 * @param dst_index_in0   DEST tile index for the first input operand (e.g. condition).
 * @param dst_index_in1   DEST tile index for the second input operand (e.g. true_val).
 * @param dst_index_in2   DEST tile index for the third input operand (e.g. false_val).
 * @param dst_index_out   DEST tile index that receives the result.
 * @param args            Extra arguments forwarded to @p sfpu_func after the tile indices.
 */
template <class F, class... ARGS>
inline void _llk_math_eltwise_ternary_sfpu_params_(
    F&& sfpu_func, std::uint32_t dst_index_in0, std::uint32_t dst_index_in1, std::uint32_t dst_index_in2, std::uint32_t dst_index_out, ARGS&&... args)
{
    _llk_math_sfpu_start_(0);

    for (std::uint32_t face = 0; face < NUM_FACES; face++)
    {
        sfpu_func(dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, std::forward<ARGS>(args)...);
        _llk_math_sfpu_inc_dst_face_addr_();
    }

    _llk_math_sfpu_done_();
}
