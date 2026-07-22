// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Pack-side _with_dt across heterogeneous output CBs.
//
// CopyTile(CbA) -> PackTile(CbOut1) -> PackTile(CbOut2), both packing D0 to distinct dtypes
// (CbOut1=bf16, CbOut2=bfp8). The chain hoists only the first pack's reconfig to boot and emits
// 2-arg pack_reconfig_data_format(prev_p, curr_p) per-stage for later sites: PackTile #2 reprograms
// bf16->bfp8, and PackTile #1 reprograms bfp8->bf16 on wraparound (iter >= 1).
//
// Runtime-validated by tests/ttnn/unit_tests/kernel_lib/test_chain_reconfig.py::test_pack_to_bfp8.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_out1 = tt::CBIndex::c_16;
    constexpr uint32_t cb_out2 = tt::CBIndex::c_17;

    constexpr uint32_t total_tiles = get_compile_time_arg_val(0);

    compute_kernel_hw_startup(cb_a, cb_a, cb_out1);

    compute_kernel_lib::eltwise_chain(
        compute_kernel_lib::EltwiseShape::tiles(total_tiles),
        compute_kernel_lib::CopyTile<compute_kernel_lib::input(cb_a)>{},
        compute_kernel_lib::PackTile<compute_kernel_lib::output(cb_out1)>{},
        compute_kernel_lib::PackTile<compute_kernel_lib::output(cb_out2)>{});
}
