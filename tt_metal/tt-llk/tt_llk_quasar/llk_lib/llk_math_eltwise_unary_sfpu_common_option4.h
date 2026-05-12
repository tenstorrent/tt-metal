// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Option 4 — unary SFPU params wrapper (slice loop, register-aware sync).

#pragma once

#include <cstdint>

#include "cmath_common.h"
#include "csfpu_common.h"
#include "llk_defs.h"

namespace ckernel
{

// ---------------------------------------------------------------------------
// Sketch — addition to csfpu_common.h. Replaces the existing SfpuSrcsSlice
// and SfpuDestSlice structs with one templated form. Existing call sites
// continue to compile via the type aliases at the bottom.
//
//   namespace ckernel::isolate_sfpu {
//
//   enum class SfpuReg : std::uint8_t { Dest, SrcS };
//
//   template <SfpuReg REG>
//   struct SfpuSlice {
//       int ydim;
//       constexpr int operator[](int slot) const {
//           if constexpr (REG == SfpuReg::Dest) {
//               return SFPU_DEST_BASE_ADDR + slot * ydim;   // base = 0
//           } else {
//               return SFPU_SRCS_BASE_ADDR + slot * ydim;
//           }
//       }
//   };
//
//   // Back-compat aliases for existing call sites (e.g. isolate-SFPU tests).
//   using SfpuSrcsSlice = SfpuSlice<SfpuReg::SrcS>;
//   using SfpuDestSlice = SfpuSlice<SfpuReg::Dest>;
//
//   } // namespace ckernel::isolate_sfpu
// ---------------------------------------------------------------------------

using ckernel::isolate_sfpu::SfpuReg;
using ckernel::isolate_sfpu::SfpuSlice;
using ckernel::isolate_sfpu::srcs_dims;

namespace detail
{

// Fully-templated unary impl. All geometry constexpr.
template <SfpuReg IN_REG, SfpuReg OUT_REG, bool IS_32BIT_MODE, class FN, class... ARGS>
inline void unary_impl(FN sfpu_func, std::uint32_t dst_tile_index, ARGS&&... rt_args)
{
    constexpr int YDIM        = srcs_dims::ydim(IS_32BIT_MODE);
    constexpr int SLICE_COUNT = srcs_dims::slice_count(IS_32BIT_MODE);

    constexpr SfpuSlice<IN_REG> in_slice {YDIM};
    constexpr SfpuSlice<OUT_REG> out_slice {YDIM};
    constexpr int input_offset  = in_slice[0];
    constexpr int output_offset = out_slice[OUT_REG == SfpuReg::SrcS ? 2 : 0];

    constexpr bool DEST_INVOLVED = (IN_REG == SfpuReg::Dest) || (OUT_REG == SfpuReg::Dest);
    constexpr bool SRCS_INVOLVED = (IN_REG == SfpuReg::SrcS) || (OUT_REG == SfpuReg::SrcS);

    if constexpr (DEST_INVOLVED)
    {
        _set_dst_write_addr_<TRISC_ID, trisc::DstTileShape::Tile32x32>(dst_tile_index);
        TTI_STALLWAIT(p_stall::STALL_SFPU, 0, 0, p_stall::MATH);
    }

    for (int slice = 0; slice < SLICE_COUNT; slice++)
    {
        sfpu_func(input_offset, output_offset, static_cast<ARGS&&>(rt_args)...);

        if constexpr (SRCS_INVOLVED)
        {
            constexpr bool rd_done = (IN_REG == SfpuReg::SrcS);
            constexpr bool wr_done = (OUT_REG == SfpuReg::SrcS);
            TTI_SFPNOP(static_cast<std::uint32_t>(wr_done), static_cast<std::uint32_t>(rd_done), 0);
        }
        if constexpr (DEST_INVOLVED)
        {
            _inc_dst_addr_<YDIM>();
        }
    }

    if constexpr (DEST_INVOLVED)
    {
        _reset_counters_<p_setrwc::SET_D>();
    }
}

} // namespace detail

// ---------------------------------------------------------------------------
// Public unary params wrapper. is_32bit_mode is a runtime bool; internal
// 2-way dispatch picks the constexpr-templated impl. When the bool is in
// fact constexpr at the call site (low-level kernels with known format),
// the unused branch dead-codes.
// ---------------------------------------------------------------------------
template <SfpuReg IN_REG, SfpuReg OUT_REG, class FN, class... ARGS>
inline void _llk_math_eltwise_unary_sfpu_params_(FN sfpu_func, std::uint32_t dst_tile_index, bool is_32bit_mode, ARGS&&... rt_args)
{
    if (is_32bit_mode)
    {
        detail::unary_impl<IN_REG, OUT_REG, /*IS_32BIT=*/true>(sfpu_func, dst_tile_index, static_cast<ARGS&&>(rt_args)...);
    }
    else
    {
        detail::unary_impl<IN_REG, OUT_REG, /*IS_32BIT=*/false>(sfpu_func, dst_tile_index, static_cast<ARGS&&>(rt_args)...);
    }
}

} // namespace ckernel
