// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <tuple>

// Ported VERBATIM from native
//   ttnn/cpp/ttnn/operations/matmul/device/config/matmul_program_config.cpp
//   SUBBLOCK_HW_CHOICES (lines 16-25) + get_subblock_sizes (lines 164-177).
//
// Used by the matmul_decode program factories to derive a REAL fat systolic fill
// (out_subblock_h/w) for the compute kernel, mirroring native so that at equal
// fill the resident-weight op converges to native (deep-plan_13 sec 0/4.3).
//
// DST-capacity bound (Blackhole): DST holds 8 bf16 tiles; fp32_dest_acc halves the
// usable DST to 4. get_subblock_sizes enforces out_h*out_w <= 4 when fp32_dest_acc_en,
// else <= 8 (the (h*w)<=4 || !fp32_dest_acc_en gate).

namespace ttnn::operations::matmul_decode {

// Stored as (w, h), descending area -- identical to native SUBBLOCK_HW_CHOICES.
inline constexpr std::array<std::tuple<uint32_t, uint32_t>, 20> MMD_SUBBLOCK_HW_CHOICES = {{
    {4, 2}, {2, 4}, {8, 1}, {1, 8},  // subblock_hw = 8
    {7, 1}, {1, 7},                  // subblock_hw = 7
    {3, 2}, {2, 3}, {6, 1}, {1, 6},  // subblock_hw = 6
    {5, 1}, {1, 5},                  // subblock_hw = 5
    {2, 2}, {4, 1}, {1, 4},          // subblock_hw = 4
    {3, 1}, {1, 3},                  // subblock_hw = 3
    {2, 1}, {1, 2},                  // subblock_hw = 2
    {1, 1},                          // subblock_hw = 1
}};

// Returns (out_subblock_h, out_subblock_w). Verbatim native semantics.
inline std::tuple<uint32_t, uint32_t> mmd_get_subblock_sizes(
    uint32_t m_tiles_per_core, uint32_t n_tiles_per_core, bool fp32_dest_acc_en) {
    uint32_t out_subblock_h = 1, out_subblock_w = 1;
    for (const auto& subblock_hw : MMD_SUBBLOCK_HW_CHOICES) {
        out_subblock_w = std::get<0>(subblock_hw);
        out_subblock_h = std::get<1>(subblock_hw);
        if ((out_subblock_h * out_subblock_w) <= 4 || !fp32_dest_acc_en) {
            if (m_tiles_per_core % out_subblock_h == 0 && n_tiles_per_core % out_subblock_w == 0) {
                return {out_subblock_h, out_subblock_w};
            }
        }
    }
    return {1, 1};  // graceful degrade (awkward dims) -- never throws
}

// deep-plan_13 P0-A fallback: WIDTH one-shot full_in0 is SENDER-MAJOR (A tiles for a
// fixed K-col across M-rows are at stride inA_K_tiles_per_core, NOT contiguous), so
// matmul_block with rt_dim=out_h>1 (which reads out_h CONSECUTIVE in0 tiles per kt-col)
// cannot consume them as a contiguous out_h x kt rectangle. N-fill (out_w>1) is safe:
// B is [K_tiles x N_tiles_per_core] row-major so the out_w in1 tiles for a fixed K-row
// ARE contiguous. Until M-fill is proven (P0-A), clamp out_h=1 and only fatten on out_w.
inline std::tuple<uint32_t, uint32_t> mmd_get_subblock_sizes_out_w_only(
    uint32_t /*m_tiles_per_core*/, uint32_t n_tiles_per_core, bool fp32_dest_acc_en) {
    const uint32_t max_w = fp32_dest_acc_en ? 4u : 8u;
    for (uint32_t w = max_w; w >= 1; --w) {
        if (n_tiles_per_core % w == 0) {
            return {1u, w};
        }
    }
    return {1, 1};
}

}  // namespace ttnn::operations::matmul_decode
