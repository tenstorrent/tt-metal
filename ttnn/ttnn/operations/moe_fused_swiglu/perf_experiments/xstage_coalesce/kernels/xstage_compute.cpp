// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED BAKE-OFF — moe_fused_swiglu x-activation staging, compute half.
//
// Verbatim reconstruction of moe_fused_swiglu_compute.cpp's `compute_tilize` block
// (~line 176): ONE fused tilize call, asymmetric page mode (32 row-major stick slices in,
// KR_PAD bfp8 tiles out). Identical for every reader VARIANT that uses the row-major path
// (0,1,2,3,5) — only the reader's read STRATEGY changes, never this compute step, which is
// exactly what makes the reader-side ns delta attributable to the read strategy alone.
// Not instantiated at all for VARIANT 4 (bfp8_tile_direct — no tilize in that path).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

using namespace compute_kernel_lib;

constexpr uint32_t KR_PAD = get_compile_time_arg_val(0);

constexpr uint32_t CB_X_IN = 0;
constexpr uint32_t CB_X_STAGE = 1;

constexpr uint32_t TILE_H = 32;

void kernel_main() {
    compute_kernel_hw_startup(CB_X_IN, CB_X_STAGE);
    tilize<KR_PAD, CB_X_IN, CB_X_STAGE>(1, TILE_H);
}
