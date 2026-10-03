// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "experimental/kernel_args.h"

// TEMP DIAGNOSTIC (cross-test hang): print the launch-populated DFB base this TRISC will unpack from.
// See project_quasar_graphops_to_torch_double_readback — quasar.tilize faults MEM_READ_NO_RESPONSE on
// the compute TRISC when run after test_add because g_dfb_interface[dfb::in].tc_slots[0].base_addr is
// stale (inherited from the prior program's freed buffer) and not reset by the emulator. Comparing this
// base standalone vs after test_add disambiguates a firmware-gap (trisc.cc) from an emulator-reset gap.
#include "api/debug/dprint.h"
#include "api/dataflow/dataflow_buffer.h"  // g_dfb_interface + get_local_dfb_interface

void kernel_main() {
    constexpr auto per_core_block_cnt = get_arg(args::per_core_block_cnt);
    constexpr auto per_core_block_tile_cnt = get_arg(args::per_core_block_tile_cnt);

#ifdef ARCH_QUASAR
    // DPRINT_UNPACK/_PACK (not plain DPRINT): g_dfb_interface is defined ONLY on the UNPACK/PACK
    // TRISCs (guarded UCK_CHLKC_UNPACK||UCK_CHLKC_PACK in trisc.cc) — a plain DPRINT compiles the
    // symbol reference on the MATH TRISC too and fails to link. On Quasar TRISC2 == PACK
    // (build.cpp:1130); the cross-test hang faults there (MEM_READ_NO_RESPONSE), so we also print the
    // pack's view. g_dfb_config_base_addr is the per-RISC base the BD programming reads config from —
    // a stale value (freed test_add region) is a prime MEM_READ_NO_RESPONSE suspect.
    DPRINT_UNPACK(
        "QSR tilize UNPACK base: in={} out={} cfg={}\n",
        get_local_dfb_interface(static_cast<uint32_t>(dfb::in)).tc_slots[0].base_addr,
        get_local_dfb_interface(static_cast<uint32_t>(dfb::out)).tc_slots[0].base_addr,
        static_cast<uint32_t>(g_dfb_config_base_addr));
    DPRINT_PACK(
        "QSR tilize PACK base: in={} out={} cfg={}\n",
        get_local_dfb_interface(static_cast<uint32_t>(dfb::in)).tc_slots[0].base_addr,
        get_local_dfb_interface(static_cast<uint32_t>(dfb::out)).tc_slots[0].base_addr,
        static_cast<uint32_t>(g_dfb_config_base_addr));
#endif

    compute_kernel_hw_startup(dfb::in, dfb::out);

    // Use lossless tilize for fp32 inputs to preserve exact values (fast tilize truncates fp32 → tf32)
    constexpr auto fp32_mode = compute_kernel_lib::is_fp32_input_format<dfb::in>()
                                   ? compute_kernel_lib::tilize_config::Fp32Mode::Lossless
                                   : compute_kernel_lib::tilize_config::Fp32Mode::Fast;

    compute_kernel_lib::tilize<
        per_core_block_tile_cnt,
        dfb::in,
        dfb::out,
        compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure,
        fp32_mode>(per_core_block_cnt);
}
