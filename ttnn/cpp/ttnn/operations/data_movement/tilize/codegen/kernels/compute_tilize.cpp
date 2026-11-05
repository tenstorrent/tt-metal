// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = get_named_compile_time_arg_val("cb_in");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t num_col_chunks = get_named_compile_time_arg_val("num_col_chunks");
    constexpr uint32_t chunk_Wt = get_named_compile_time_arg_val("chunk_wt");

    uint32_t num_tile_rows = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_in, cb_out);
    // tilize_init leaves the Src zero-substitution flag alone; copy_init sets it to preserve. With it, fp32 keeps
    // -0.0 and subnormals bit-exact. Fast tilize for bf16 still writes them as +0, as the native kernel does.
    copy_init(cb_in);

    // Same mode choice as the native kernel: fast tilize for bf16 (the math thread otherwise bounds L1-resident
    // inputs), the exact path for fp32, which fast tilize would truncate to tf32.
    constexpr auto fp32_mode = compute_kernel_lib::is_fp32_input_format<cb_in>()
                                   ? compute_kernel_lib::tilize_config::Fp32Mode::Lossless
                                   : compute_kernel_lib::tilize_config::Fp32Mode::Fast;
    compute_kernel_lib::tilize<
        chunk_Wt,
        cb_in,
        cb_out,
        compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure,
        fp32_mode>(num_tile_rows * num_col_chunks);
}
