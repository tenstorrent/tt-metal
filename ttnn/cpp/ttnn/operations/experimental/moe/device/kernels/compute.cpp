// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"

namespace NAMESPACE {
void MAIN {
    // CBs
    constexpr auto cb_r2c_w0 = tt::CBIndex::c_0;
    constexpr auto cb_s2c_in = tt::CBIndex::c_1;
    constexpr auto cb_c2c_mm0 = tt::CBIndex::c_2;
    constexpr auto cb_c2c_mm1 = tt::CBIndex::c_3;
    constexpr auto cb_c2w_elt = tt::CBIndex::c_4;
    constexpr auto cb_r2c_in2 = tt::CBIndex::c_5;
    constexpr auto cb_c2w_mm2 = tt::CBIndex::c_6;

    // CB Aliases
    constexpr auto cb_r2c_w1 = tt::CBIndex::c_0;
    constexpr auto cb_r2c_w2 = tt::CBIndex::c_0;

    // Constants for MoE
    constexpr uint32_t num_w0_tiles = 224;
    constexpr uint32_t num_w1_tiles = 224;
    constexpr uint32_t num_w2_tiles = 224;

    // Read W0 from CB into registers
    for (uint32_t i = 0; i < num_w0_tiles; ++i) {
        cb_wait_front(cb_r2c_w0, 1);
        cb_pop_front(cb_r2c_w0, 1);
    }
}
}  // namespace NAMESPACE
