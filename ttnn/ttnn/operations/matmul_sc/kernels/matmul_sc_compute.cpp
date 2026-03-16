// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// matmul_sc - Compute Kernel
// Full matmul C = A x B using the matmul_1d helper.
//
// Runtime args:
//   [0] Mt    -- tile rows of C
//   [1] Kt    -- inner dimension tiles
//   [2] Nt    -- tile columns of C
//   [3] batch -- always 1 for rank-2

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/cb_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_1d_helpers.hpp"

constexpr uint32_t cb_in0 = 0;
constexpr uint32_t cb_in1 = 1;
constexpr uint32_t cb_out = 16;

namespace NAMESPACE {
void MAIN {
    uint32_t Mt = get_arg_val<uint32_t>(0);
    uint32_t Kt = get_arg_val<uint32_t>(1);
    uint32_t Nt = get_arg_val<uint32_t>(2);
    uint32_t batch = get_arg_val<uint32_t>(3);

    // Three-arg form required because srcA (cb_in0) and srcB (cb_in1) are different CBs
    compute_kernel_hw_startup(cb_in0, cb_in1, cb_out);

    compute_kernel_lib::matmul_1d<cb_in0, cb_in1, cb_out>(Mt, Nt, Kt, batch);
}
}  // namespace NAMESPACE
