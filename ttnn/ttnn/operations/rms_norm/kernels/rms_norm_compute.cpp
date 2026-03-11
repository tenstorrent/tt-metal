// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// RMS Norm - Compute Kernel
// Stage 1 (data_pipeline): tilize(RM), identity copy c_1->c_16, untilize(RM)

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/copy_tile_helpers.hpp"

constexpr uint32_t cb_input_rm = 0;
constexpr uint32_t cb_tilized = 1;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_out = 16;
constexpr uint32_t cb_untilized = 17;

// Compile-time args
constexpr uint32_t Ht_max = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);
constexpr uint32_t input_is_rm = get_compile_time_arg_val(2);
constexpr uint32_t has_gamma = get_compile_time_arg_val(3);

void kernel_main() {
    // Runtime arg: actual Ht for this core
    uint32_t Ht = get_arg_val<uint32_t>(0);
    if (Ht == 0) {
        return;
    }

    // Hardware startup: srcA=cb_tilized, srcB=cb_scaler, ocb=cb_out
    compute_kernel_hw_startup(cb_tilized, cb_scaler, cb_out);

    for (uint32_t row = 0; row < Ht; ++row) {
        // Phase 1: Tilize (RM path only) c_0 -> c_1
        if constexpr (input_is_rm) {
            compute_kernel_lib::tilize<
                cb_input_rm,
                cb_tilized,
                compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
                compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
                compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(Wt, 1);
        }

        // Identity copy: c_1 -> c_16 (tile-by-tile)
        compute_kernel_lib::copy_tiles<
            compute_kernel_lib::CopyInputPolicy::WaitAndPop,
            compute_kernel_lib::CopyDataFormatReconfig::INPUT_AND_OUTPUT>(cb_tilized, cb_out, Wt);

        // Phase 7: Untilize (RM path only) c_16 -> c_17
        if constexpr (input_is_rm) {
            compute_kernel_lib::untilize<
                Wt,
                cb_out,
                cb_untilized,
                compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
                compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
                compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(1);
        }
    }
}
