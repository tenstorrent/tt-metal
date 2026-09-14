// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

// Per band: untilize every input tile into its own row-major 32x32 block (width 1 keeps the
// helper geometry independent of per-input widths -- the writer's assembler does all the
// column math), then retilize the assembled full-width band the writer built in cb_asm.
void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_rm = get_compile_time_arg_val(1);
    constexpr uint32_t cb_asm = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);
    constexpr uint32_t total_in_wt = get_compile_time_arg_val(4);
    constexpr uint32_t out_wt = get_compile_time_arg_val(5);
    constexpr bool fp32_lossless = get_compile_time_arg_val(6) != 0;

    const uint32_t num_bands = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_in, cb_rm);

    for (uint32_t b = 0; b < num_bands; ++b) {
        compute_kernel_lib::untilize<1, cb_in, cb_rm>(total_in_wt);
        compute_kernel_lib::tilize<
            out_wt,
            cb_asm,
            cb_out,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure,
            fp32_lossless ? compute_kernel_lib::tilize_config::Fp32Mode::Lossless
                          : compute_kernel_lib::tilize_config::Fp32Mode::Fast>(1);
    }
}
