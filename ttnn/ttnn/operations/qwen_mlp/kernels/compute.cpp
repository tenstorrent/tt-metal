// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_hw_startup.h"
#include "matmul_compute.hpp"
#include "rmsnorm_compute.hpp"

void kernel_main() {
    constexpr RmsnormCbs norm_cbs = RMSNORM_CBS;
    constexpr MatmulShape gate_up = MATMUL_SHAPE("mm1");
    constexpr MatmulShape down = MATMUL_SHAPE("mm2");

    compute_kernel_hw_startup<SrcOrder::Reverse>(gate_up.cb_in0, gate_up.cb_in1, gate_up.cb_partial);

    CircularBuffer(norm_cbs.scaler).wait_front(1);
    for (uint32_t row = 0; row < RT_ARG("norm_rows"); row++) {
        rmsnorm_row(
            norm_cbs,
            gate_up.in0_row_tiles,
            CT_ARG("norm_group"),
            CT_ARG("norm_inv_width_bits"),
            CT_ARG("norm_eps_bits"));
    }

    matmul_phase(gate_up);
    matmul_phase(down);
}
