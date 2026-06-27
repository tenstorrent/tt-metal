// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for scaled_dot_product_attention (Flash Attention).
//
// Stage 0 (init): The reader initializes all running-state CBs (cb_max_old,
// cb_sum_old, cb_o) via raw constant fills. The compute kernel only needs
// to boot the hardware (compute_kernel_hw_startup) so the TRISC pipeline is
// ready for the matmul/eltwise/reduce helpers that later stages add.
//
// The full Flash Attention recurrence (phases 1-14) is added incrementally
// by later stages. Each phase uses a helper from compute_kernel_lib; raw
// LLK calls are avoided.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"

void kernel_main() {
    // CB indices (match op_design.md CB layout).
    constexpr uint32_t cb_q = tt::CBIndex::c_0;
    constexpr uint32_t cb_scores = 24;

    // Boot the Tensix compute pipeline. Later stages will issue matmul,
    // eltwise, and reduce helpers against cb_q / cb_scores / etc.
    compute_kernel_hw_startup(cb_q, cb_scores);
}
