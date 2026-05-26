// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Case E — pack-side _with_dt. Pack-side rotates output CB across different pack dtypes.
//
// Chain shape: CopyTile(CbA, D0) -> PackTile(CbOut1, D0) -> PackTile(CbOut2, D0).
// Both PackTiles read D0 (the CopyTile result = CbA) and pack to their respective output CBs.
// At PackTile #2: prev_p=CbOut1 (set by PackTile #1), curr_p=CbOut2 with different dtype →
// pack_reconfig_data_format(prev_p, curr_p) fires. CbOut1=bf16, CbOut2=bfp8 → pack-side rotates
// IEEE -> block-float, exercising the pack engine's shared-exponent programming on the new with_dt
// pack overload.
//
// STRUCTURAL DOCUMENTATION ONLY — not currently runtime-validated.
// -----------------------------------------------------------------
// eltwise_chain hoists ALL pack reconfigs to boot via `pack_init_for_each` (eltwise_chain.inl:1804,
// comment "F-PERF-4: hoisted to boot, not per-tile"). With 2 PackTile elements, the LAST hoisted
// reconfig wins — pack format stays set to PackTile #2's CB format (bfp8) for the entire main loop,
// so PackTile #1's pack_tile(D0, CbOut1, idx) writes bfp8-format bytes into CbOut1 (which expects
// bf16) → CbOut1 reads as garbage downstream. Real chain consumers all use exactly 1 PackTile per
// chain call, so this is an unhit code path in production. The 2-arg pack _with_dt emission itself
// IS correct (verified by code review of emit_pre_element_transitions and by build success of this
// file); only the *runtime distinguishability* via a single-chain test is structurally blocked.
//
// The pytest test_case_e_pack_to_bfp8 is therefore @pytest.mark.skip with reason; see
// tests/ttnn/unit_tests/kernel_lib/test_chain_reconfig.py for the full explanation.

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
        compute_kernel_lib::CopyTile<
            cb_a,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::Streaming,
            compute_kernel_lib::OperandKind::Scalar,
            compute_kernel_lib::CopyTileReconfig::Input>{},
        compute_kernel_lib::PackTile<
            cb_out1,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::OutStreaming,
            compute_kernel_lib::OperandKind::Scalar,
            compute_kernel_lib::PackTileReconfig::Output>{},
        compute_kernel_lib::PackTile<
            cb_out2,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::OutStreaming,
            compute_kernel_lib::OperandKind::Scalar,
            compute_kernel_lib::PackTileReconfig::Output>{});
}
