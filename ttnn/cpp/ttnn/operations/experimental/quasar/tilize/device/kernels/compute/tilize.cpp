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
    // Gate ALL diagnostics to SMALL outputs only (<=32 tiles total). test_concat.py builds 16 wide
    // (256-tile) inputs whose DPRINTs saturate the device print buffer and drown out the actual hang in
    // test_concat_small_grid.py's small (16/8-tile) tilizes. This compiles the prints into ONLY the
    // small-tilize programs, so the buffer captures the hanging op. Compute is already exonerated (it
    // completes every run); these markers show how far the small tilize gets after the poison.
    constexpr bool qsr_trace = (per_core_block_cnt * per_core_block_tile_cnt) <= 32;
#if defined(UCK_CHLKC_PACK)
    if constexpr (qsr_trace) {
        const uint8_t out_ptc =
            get_local_dfb_interface(static_cast<uint32_t>(dfb::out)).tc_slots[0].packed_tile_counter;
        const uint32_t out_tc = dfb::get_counter_id(out_ptc);
        volatile ckernel::trisc::tile_counter_u* tcs = &ckernel::trisc::tile_counters[out_tc];
        DPRINT(
            "QSR tilize PACK base={} out-TC tc={} posted={} acked={} cap={}\n",
            get_local_dfb_interface(static_cast<uint32_t>(dfb::out)).tc_slots[0].base_addr,
            out_tc,
            static_cast<uint32_t>(tcs->f.posted),
            static_cast<uint32_t>(tcs->f.acked),
            static_cast<uint32_t>(tcs->f.buf_capacity));
    }
#endif
#if defined(UCK_CHLKC_UNPACK)
    if constexpr (qsr_trace) {
        const uint8_t in_ptc = get_local_dfb_interface(static_cast<uint32_t>(dfb::in)).tc_slots[0].packed_tile_counter;
        const uint32_t in_tc = dfb::get_counter_id(in_ptc);
        volatile ckernel::trisc::tile_counter_u* tcs = &ckernel::trisc::tile_counters[in_tc];
        DPRINT(
            "QSR tilize UNPACK base={} in-TC tc={} posted={} acked={} cap={}\n",
            get_local_dfb_interface(static_cast<uint32_t>(dfb::in)).tc_slots[0].base_addr,
            in_tc,
            static_cast<uint32_t>(tcs->f.posted),
            static_cast<uint32_t>(tcs->f.acked),
            static_cast<uint32_t>(tcs->f.buf_capacity));
    }
#endif
    if constexpr (qsr_trace) {
        DPRINT_UNPACK("QSR tilize UNPACK: A pre-hw_startup\n");
        DPRINT_PACK("QSR tilize PACK: A pre-hw_startup\n");
    }
#endif

    compute_kernel_hw_startup(dfb::in, dfb::out);

#ifdef ARCH_QUASAR
    if constexpr (qsr_trace) {
        DPRINT_UNPACK("QSR tilize UNPACK: B post-hw_startup, pre-tilize\n");
        DPRINT_PACK("QSR tilize PACK: B post-hw_startup, pre-tilize\n");
    }
#endif

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

#ifdef ARCH_QUASAR
    if constexpr (qsr_trace) {
        DPRINT_UNPACK("QSR tilize UNPACK: C post-tilize (done)\n");
        DPRINT_PACK("QSR tilize PACK: C post-tilize (done)\n");
    }
#endif
}
