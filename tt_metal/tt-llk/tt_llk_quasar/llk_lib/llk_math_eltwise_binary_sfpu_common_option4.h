// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Option 4 — binary SFPU params wrapper.
//
// Mirrors the unary wrapper shape: per-slot SfpuReg template parameter,
// `if constexpr` on register identity to pick offset formula and inter-slice
// sync. Difference from unary: each Dest operand carries its own dest tile
// index (idst0/idst1/odst), so per-slot offsets depend on a runtime tile
// index. The choice of which formula to apply is compile-time; the resulting
// offset value is runtime whenever that slot maps to Dest.

#pragma once

#include <cstdint>

#include "cmath_common.h"
#include "csfpu_common.h"
#include "llk_defs.h"

namespace ckernel
{

using ckernel::isolate_sfpu::SfpuReg;
using ckernel::isolate_sfpu::SfpuSlice;
using ckernel::isolate_sfpu::srcs_dims;

namespace detail
{

// Returns the row offset of one operand of a binary op at the current slice.
//   - Dest slot:  dst_index * rows_per_dest_tile + slice * YDIM (runtime)
//   - SrcS slot:  fixed HW slot for that operand (constexpr)
// The choice of formula is compile-time via SfpuReg; only the resulting
// value is runtime for Dest.
template <SfpuReg REG, int YDIM>
inline int operand_offset(int srcs_slot, int slice, int rows_per_dest_tile, std::uint32_t dst_tile_index)
{
    if constexpr (REG == SfpuReg::Dest)
    {
        return static_cast<int>(dst_tile_index) * rows_per_dest_tile + slice * YDIM;
    }
    else
    {
        constexpr SfpuSlice<REG> sl {YDIM};
        (void)slice;
        (void)rows_per_dest_tile;
        (void)dst_tile_index;
        return sl[srcs_slot];
    }
}

template <SfpuReg IN0_REG, SfpuReg IN1_REG, SfpuReg OUT_REG, bool IS_32BIT_MODE, class FN, class... ARGS>
inline void binary_impl(FN sfpu_func, std::uint32_t dst_index_in0, std::uint32_t dst_index_in1, std::uint32_t dst_index_out, ARGS&&... rt_args)
{
    constexpr int YDIM               = srcs_dims::ydim(IS_32BIT_MODE);
    constexpr int SLICE_COUNT        = srcs_dims::slice_count(IS_32BIT_MODE);
    constexpr int ROWS_PER_DEST_TILE = SLICE_COUNT * YDIM;

    constexpr bool DEST_INVOLVED = (IN0_REG == SfpuReg::Dest) || (IN1_REG == SfpuReg::Dest) || (OUT_REG == SfpuReg::Dest);
    constexpr bool SRCS_INVOLVED = (IN0_REG == SfpuReg::SrcS) || (IN1_REG == SfpuReg::SrcS) || (OUT_REG == SfpuReg::SrcS);

    if constexpr (DEST_INVOLVED)
    {
        TTI_STALLWAIT(p_stall::STALL_SFPU, 0, 0, p_stall::MATH);
    }

    for (int slice = 0; slice < SLICE_COUNT; slice++)
    {
        const int input0_offset = operand_offset<IN0_REG, YDIM>(/*srcs_hw_slot=*/0, slice, ROWS_PER_DEST_TILE, dst_index_in0);
        const int input1_offset = operand_offset<IN1_REG, YDIM>(/*srcs_hw_slot=*/1, slice, ROWS_PER_DEST_TILE, dst_index_in1);
        const int output_offset = operand_offset<OUT_REG, YDIM>(/*srcs_hw_slot=*/2, slice, ROWS_PER_DEST_TILE, dst_index_out);

        sfpu_func(input0_offset, input1_offset, output_offset, static_cast<ARGS&&>(rt_args)...);

        if constexpr (SRCS_INVOLVED)
        {
            constexpr bool rd_done = (IN0_REG == SfpuReg::SrcS) || (IN1_REG == SfpuReg::SrcS);
            constexpr bool wr_done = (OUT_REG == SfpuReg::SrcS);
            TTI_SFPNOP(static_cast<std::uint32_t>(wr_done), static_cast<std::uint32_t>(rd_done), 0);
        }
    }

    if constexpr (DEST_INVOLVED)
    {
        _reset_counters_<p_setrwc::SET_D>();
    }
}

} // namespace detail

template <SfpuReg IN0_REG, SfpuReg IN1_REG, SfpuReg OUT_REG, class FN, class... ARGS>
inline void _llk_math_eltwise_binary_sfpu_params_(
    FN sfpu_func, std::uint32_t dst_index_in0, std::uint32_t dst_index_in1, std::uint32_t dst_index_out, bool is_32bit_mode, ARGS&&... rt_args)
{
    if (is_32bit_mode)
    {
        detail::binary_impl<IN0_REG, IN1_REG, OUT_REG, /*IS_32BIT=*/true>(
            sfpu_func, dst_index_in0, dst_index_in1, dst_index_out, static_cast<ARGS&&>(rt_args)...);
    }
    else
    {
        detail::binary_impl<IN0_REG, IN1_REG, OUT_REG, /*IS_32BIT=*/false>(
            sfpu_func, dst_index_in0, dst_index_in1, dst_index_out, static_cast<ARGS&&>(rt_args)...);
    }
}

} // namespace ckernel
