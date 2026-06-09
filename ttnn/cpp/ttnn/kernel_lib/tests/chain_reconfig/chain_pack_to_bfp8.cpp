// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Case E — pack-side _with_dt across heterogeneous output CBs.
//
// Chain shape: CopyTile(CbA, D0) -> PackTile(CbOut1, D0) -> PackTile(CbOut2, D0).
// Both PackTiles read D0 (the CopyTile result = CbA) and pack to their own output CBs with
// distinct dtypes (CbOut1=bf16, CbOut2=bfp8). Per `docs/pack_reconfig_hoisting_proposal.html`
// §4.2, the chain detects heterogeneous opt-in pack CBs, hoists only the first site's reconfig
// to boot, and emits 2-arg `pack_reconfig_data_format(prev_p, curr_p)` at per-stage for later
// sites. PackTile #2 sees prev_p=CbOut1, curr_p=CbOut2 → bf16 -> bfp8 reprogram fires; PackTile
// #1 sees wraparound prev_p=CbOut2, curr_p=CbOut1 → bfp8 -> bf16 reprogram on iter ≥ 1.
//
// Runtime-validated by tests/ttnn/unit_tests/kernel_lib/test_chain_reconfig.py::test_case_e_pack_to_bfp8.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_out1 = tt::CBIndex::c_16;
    constexpr uint32_t cb_out2 = tt::CBIndex::c_17;

    constexpr uint32_t total_tiles = get_compile_time_arg_val(0);

    compute_kernel_hw_startup(cb_a, cb_a, cb_out1);

    compute_kernel_lib::eltwise_chain(
        total_tiles,
        compute_kernel_lib::CopyTile<cb_a>{},
        compute_kernel_lib::PackTile<cb_out1>{},
        compute_kernel_lib::PackTile<cb_out2>{});
}
