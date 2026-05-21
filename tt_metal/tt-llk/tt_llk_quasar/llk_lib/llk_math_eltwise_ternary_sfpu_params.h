// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <utility>

#include "ckernel_sfpu.h"
#include "ckernel_trisc_common.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "llk_math_eltwise_ternary_sfpu.h"

// Ternary SFPU dispatch — drives a kernel func that consumes three input tiles
// from DEST and writes one output tile. Mirrors the Blackhole API surface so
// shared ttnn-style ternary tests look the same on both arches.
//
// Section base is set to DEST tile 0 (`_llk_math_sfpu_start_(0)`); the kernel
// resolves the per-tile DEST address space via the four tile indices it
// receives. Per-face advancement uses the shared
// `_llk_math_sfpu_inc_dst_face_addr_` step (16 rows / one face).
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
