// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Fused pixel_unshuffle TILE compute kernel (Approach B).
//
// Per work unit (one output tile-row), this compute kernel runs the two
// hardware format conversions and lets the BRISC dataflow kernel do the cheap
// stride-r gather in between:
//
//   c_in (TILE band) --untilize--> c_rm (RM band)
//        [ BRISC gathers c_rm -> c_gathered (RM output tile-row) ]
//   c_gathered (RM) --tilize--> c_out (TILE output tile-row)
//
// The two conversions are the expensive part and run on the compute engine
// (tilize/pack_untilize LLKs); the gather is a contiguous RM stick copy on BRISC.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t WiC = get_compile_time_arg_val(0);        // input sub-band width in tiles (untilize block width)
    constexpr uint32_t band_rows = get_compile_time_arg_val(1);  // r + 1  (untilize num_blocks)
    constexpr uint32_t WoC = get_compile_time_arg_val(2);        // output chunk width in tiles (tilize block width)
    constexpr uint32_t cb_in = get_compile_time_arg_val(3);      // TILE input sub-band
    constexpr uint32_t cb_rm = get_compile_time_arg_val(4);      // RM untilized sub-band
    constexpr uint32_t cb_gathered = get_compile_time_arg_val(5);  // RM gathered output chunk
    constexpr uint32_t cb_out = get_compile_time_arg_val(6);       // TILE output chunk

    const uint32_t num_items = get_arg_val<uint32_t>(0);  // (tile-row, width-chunk) items for this core

    compute_kernel_hw_startup(cb_in, cb_out);

    for (uint32_t u = 0; u < num_items; u++) {
        // Phase A: untilize the (band_rows × WiC) TILE sub-band into row-major.
        compute_kernel_lib::untilize<
            WiC,
            cb_in,
            cb_rm,
            compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(band_rows);

        // (BRISC gathers c_rm -> c_gathered here, sequenced by the CB handshake.)

        // Phase B: tilize the gathered RM output chunk (1 tile-row × WoC) into TILE.
        compute_kernel_lib::tilize<
            WoC,
            cb_gathered,
            cb_out,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(1);
    }
}
