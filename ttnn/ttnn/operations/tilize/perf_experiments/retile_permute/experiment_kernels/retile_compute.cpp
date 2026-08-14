// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Bake-off compute arm. Variants 3 / 4 / 7 are the "retile-direct" scheme: the
// reader produces cb_output_tiles itself (the permutation IS the tilize for a
// same-dtype retile), so the tilize LLK has nothing left to do and compute
// collapses to nothing. Every other variant keeps the op's real compute kernel
// verbatim, so the row-major arms are measured against the true pipeline.

#ifndef RETILE_VARIANT
#define RETILE_VARIANT 0
#endif

#if RETILE_VARIANT == 3 || RETILE_VARIANT == 4 || RETILE_VARIANT == 7 || RETILE_VARIANT == 8
#include <cstdint>
void kernel_main() {}
#else
#include "ttnn/ttnn/operations/tilize/kernels/tilize_compute.cpp"
#endif
