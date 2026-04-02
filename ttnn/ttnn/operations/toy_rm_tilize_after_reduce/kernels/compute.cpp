// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Probe kernel:
//   1. tilize row-major input into cb_tilized_out
//   2. run REDUCE_ROW while preserving cb_tilized_out
//   3. discard the reduce result
//
// The writer drains cb_tilized_out after reduce has already run.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/debug/dprint.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t width_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(1);
    constexpr uint32_t post_tilize_nops = get_compile_time_arg_val(2);
    constexpr uint32_t insert_tensix_sync = get_compile_time_arg_val(3);

    constexpr uint32_t cb_rm_in = 0;
    constexpr uint32_t cb_scaler = 8;
    constexpr uint32_t cb_tilized_out = 24;
    constexpr uint32_t cb_reduce = 25;

    compute_kernel_hw_startup(cb_rm_in, cb_scaler, cb_tilized_out);

    for (uint32_t block = 0; block < num_blocks; ++block) {
        compute_kernel_lib::tilize<width_tiles, cb_rm_in, cb_tilized_out>(1);

        if constexpr (post_tilize_nops > 0) {
            for (uint32_t i = 0; i < post_tilize_nops; ++i) {
                TTI_NOP;
            }
        }

        if constexpr (insert_tensix_sync) {
            tensix_sync();
        }

        compute_kernel_lib::reduce<SUM, REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
            cb_tilized_out, cb_scaler, cb_reduce, compute_kernel_lib::ReduceInputBlockShape::row(width_tiles));

        if (block == num_blocks - 1) {
            SliceRange mid_rows = SliceRange{.h0 = 16, .h1 = 20, .hs = 1, .w0 = 0, .w1 = 8, .ws = 1};
            SliceRange tail_rows_0 = SliceRange{.h0 = 24, .h1 = 28, .hs = 1, .w0 = 0, .w1 = 8, .ws = 1};
            SliceRange tail_rows_1 = SliceRange{.h0 = 28, .h1 = 32, .hs = 1, .w0 = 0, .w1 = 8, .ws = 1};

            DPRINT_UNPACK({
                DPRINT << "after reduce nops=" << post_tilize_nops << " block=" << block << " cb24 tile0 mid" << ENDL();
                DPRINT << TSLICE(cb_tilized_out, 0, mid_rows, true, false) << ENDL();
                DPRINT << "after reduce nops=" << post_tilize_nops << " block=" << block << " cb24 tile1 mid" << ENDL();
                DPRINT << TSLICE(cb_tilized_out, 1, mid_rows, true, false) << ENDL();
                DPRINT << "after reduce nops=" << post_tilize_nops << " block=" << block << " cb24 tile0 tail0"
                       << ENDL();
                DPRINT << TSLICE(cb_tilized_out, 0, tail_rows_0, true, false) << ENDL();
                DPRINT << "after reduce nops=" << post_tilize_nops << " block=" << block << " cb24 tile0 tail1"
                       << ENDL();
                DPRINT << TSLICE(cb_tilized_out, 0, tail_rows_1, true, false) << ENDL();
                DPRINT << "after reduce nops=" << post_tilize_nops << " block=" << block << " cb24 tile3 tail0"
                       << ENDL();
                DPRINT << TSLICE(cb_tilized_out, 3, tail_rows_0, true, false) << ENDL();
                DPRINT << "after reduce nops=" << post_tilize_nops << " block=" << block << " cb24 tile3 tail1"
                       << ENDL();
                DPRINT << TSLICE(cb_tilized_out, 3, tail_rows_1, true, false) << ENDL();
            });
        }

        cb_pop_front(cb_reduce, 1);
    }

    cb_pop_front(cb_scaler, 1);
}
