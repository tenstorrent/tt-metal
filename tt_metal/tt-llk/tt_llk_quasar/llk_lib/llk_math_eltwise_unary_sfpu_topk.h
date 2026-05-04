// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

#include "ckernel_addrmod.h"
#include "cmath_common.h"
#include "experimental/ckernel_sfpu_topk.h"
#include "llk_defs.h"
#include "llk_math_eltwise_unary_sfpu_common.h"

using namespace ckernel::math;

/**
 * @brief SfpuType-templated init overload for topk.
 *
 * The shared topk test calls `_llk_math_eltwise_unary_sfpu_init_<SfpuType::topk_local_sort>()`
 * directly, then `ckernel::sfpu::_init_topk()`. The base un-templated init in
 * llk_math_eltwise_unary_sfpu_common.h takes no SfpuType parameter, so this is an overload.
 *
 * On top of the standard SFPU init (config reg, addrmod 7, RWC reset), this configures
 * ADDR_MOD_6 with dest.incr=32 — required because `bitonic_topk_store16<is_fp32, alt_addr_mod=true>`
 * uses ADDR_MOD_6 on the FINAL index store to auto-advance Dest by 32 rows. ADDR_MOD_6 is
 * otherwise undefined in default SFPU init.
 *
 * The caller must follow up with `ckernel::sfpu::_init_topk()` to set LaneConfig bit [2]
 * (ENABLE_DEST_INDEX) — that is a separate step left to the caller so a single
 * init_<topk_local_sort>() does not silently configure features the caller may not want.
 */
template <SfpuType sfpu_op>
inline void _llk_math_eltwise_unary_sfpu_init_()
{
    // Common SFPU init: config reg, ADDR_MOD_7, RWC reset.
    _llk_math_eltwise_sfpu_init_();

    if constexpr (sfpu_op == SfpuType::topk_local_sort || sfpu_op == SfpuType::topk_merge || sfpu_op == SfpuType::topk_rebuild)
    {
        // ADDR_MOD_6: dest.incr=32 — used by alt_addr_mod=true store16 path.
        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 32},
        }
            .set(ADDR_MOD_6, csr_read<CSR::TRISC_ID>());
    }
}

namespace ckernel
{

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_topk_init()
{
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::topk_local_sort>();
    ckernel::sfpu::_init_topk();
}

// Topk LLK wrappers inline `_llk_math_eltwise_sfpu_start_` / `_done_` rather than
// going through `_llk_math_eltwise_sfpu_params_`, because the params helper
// iterates over NUM_FACES and topk manages its own (face, col) walk internally
// via `set_dst_write_addr` — calling sfpu_func once per tile is required.
template <bool APPROXIMATE, bool is_fp32_dest_acc_en, bool STABLE_SORT = false>
inline void llk_math_eltwise_unary_sfpu_topk_local_sort(std::uint32_t dst_index, int idir, int i_end_phase, int i_start_phase, int i_end_step, int i_start_step)
{
    _llk_math_eltwise_sfpu_start_(dst_index);
    ckernel::sfpu::calculate_bitonic_topk_phases_steps<APPROXIMATE, is_fp32_dest_acc_en, STABLE_SORT>(
        idir, i_end_phase, i_start_phase, i_end_step, i_start_step);
    _llk_math_eltwise_sfpu_done_();
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en, bool top_min = false, bool STABLE_SORT = false>
inline void llk_math_eltwise_unary_sfpu_topk_merge(std::uint32_t dst_index, int m_iter, int k)
{
    _llk_math_eltwise_sfpu_start_(dst_index);
    ckernel::sfpu::calculate_bitonic_topk_merge<APPROXIMATE, is_fp32_dest_acc_en, top_min, STABLE_SORT>(m_iter, k);
    _llk_math_eltwise_sfpu_done_();
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en, bool STABLE_SORT = false>
inline void llk_math_eltwise_unary_sfpu_topk_rebuild(std::uint32_t dst_index, int idir, int m_iter, int k, int logk, int skip_second)
{
    _llk_math_eltwise_sfpu_start_(dst_index);
    ckernel::sfpu::calculate_bitonic_topk_rebuild<APPROXIMATE, is_fp32_dest_acc_en, STABLE_SORT>(idir, m_iter, k, logk, skip_second);
    _llk_math_eltwise_sfpu_done_();
}

} // namespace ckernel
