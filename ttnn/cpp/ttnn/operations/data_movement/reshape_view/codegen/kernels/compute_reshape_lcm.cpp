// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute kernel of the LCM TILE reshape program (build_lcm).
//
// Per chunk: untilize in_tile_rows x Wt_in tiles (cb_in -> cb_mid), then tilize out_tile_rows x Wt_out tiles
// (cb_mid -> cb_out). cb_mid holds the chunk's row-major data, which both widths read in the same order.
#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = get_named_compile_time_arg_val("cb_in");
    constexpr uint32_t cb_mid = get_named_compile_time_arg_val("cb_mid");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t Wt_in = get_named_compile_time_arg_val("wt_in");
    constexpr uint32_t Wt_out = get_named_compile_time_arg_val("wt_out");
    constexpr uint32_t in_tile_rows = get_named_compile_time_arg_val("in_tile_rows");
    constexpr uint32_t out_tile_rows = get_named_compile_time_arg_val("out_tile_rows");

    const uint32_t num_chunks = get_arg_val<uint32_t>(0);

    DataflowBuffer in_buffer(cb_in);
    DataflowBuffer mid_buffer(cb_mid);
    DataflowBuffer out_buffer(cb_out);

    compute_kernel_hw_startup(cb_in, cb_mid);

    // All three CBs share one data format, so neither phase reconfigures. Lossless keeps fp32 off fast tilize, which
    // truncates to tf32; 16-bit data arrives as UInt16, which neither fast path takes.
    namespace ckl = compute_kernel_lib;
    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        ckl::untilize<
            Wt_in,
            cb_in,
            cb_mid,
            ckl::untilize_config::InitUninitMode::InitAndUninit,
            ckl::untilize_config::WaitMode::WaitBlock,
            ckl::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(in_buffer, mid_buffer, in_tile_rows);
        ckl::tilize<
            Wt_out,
            cb_mid,
            cb_out,
            ckl::tilize_config::InitUninitMode::InitAndUninit,
            ckl::tilize_config::WaitMode::WaitBlock,
            ckl::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure,
            ckl::tilize_config::Fp32Mode::Lossless>(mid_buffer, out_buffer, out_tile_rows);
    }
}
