// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Tilize compute kernel, uniform-width variant: tilize_block per tile-row chunk.
//
// Same pipeline as compute_tilize.cpp but with num_col_chunks/chunk_Wt as compile-time args. Used
// when every core in the program shares one block width (row path, uniform 2D splits): the
// constexpr width lets tilize_init/tilize_block constant-fold their LLK setup, which a runtime
// width cannot. Ragged splits (mixed widths across cores) must keep compute_tilize.cpp's runtime
// ABI so they share one kernel binary.
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tilize.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    // Host-gated bf16->bf16 selects the fast-tilize LLK datapath; fp32 input
    // must stay on the standard path (fast tilize truncates fp32 to tf32).
    constexpr bool use_fast = get_compile_time_arg_val(2) != 0;
    constexpr uint32_t num_col_chunks = get_compile_time_arg_val(3);
    constexpr uint32_t chunk_Wt = get_compile_time_arg_val(4);

    uint32_t num_tile_rows = get_arg_val<uint32_t>(0);

    CircularBuffer cb_in_exp(cb_in);
    CircularBuffer cb_out_exp(cb_out);

    // Native's init shape: hw_startup only (no state_configure, no A2D unpack/math init) —
    // tilize_init/fast_tilize_init below re-program all of it.
    compute_kernel_hw_startup(cb_in, cb_out);
    if constexpr (use_fast) {
        fast_tilize_init(cb_in, chunk_Wt, cb_out);
    } else {
        tilize_init(cb_in, chunk_Wt, cb_out);
    }

    for (uint32_t b = 0; b < num_tile_rows; ++b) {
        for (uint32_t c = 0; c < num_col_chunks; ++c) {
            cb_in_exp.wait_front(chunk_Wt);
            cb_out_exp.reserve_back(chunk_Wt);

            if constexpr (use_fast) {
                fast_tilize_block(cb_in, chunk_Wt, cb_out);
            } else {
                tilize_block(cb_in, chunk_Wt, cb_out);
            }

            cb_out_exp.push_back(chunk_Wt);
            cb_in_exp.pop_front(chunk_Wt);
        }
    }

    if constexpr (use_fast) {
        fast_tilize_uninit(cb_in, cb_out, chunk_Wt);
    }
}
