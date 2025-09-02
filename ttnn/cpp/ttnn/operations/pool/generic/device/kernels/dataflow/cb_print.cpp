// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

#define DEBUG_PRINT 1

#if DEBUG_PRINT == 1
#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#endif

#define TILE_WIDTH 32
#define TILE_HEIGHT 32
#define FACE_WIDTH 16
#define FACE_HEIGHT 16

void kernel_main() {
    // NOTE: here it is assumed that in_ntiles_hw == 1. General cases not handled yet. When ntiles_hw > 1 the large
    // kernel is called
    constexpr uint32_t in_ntiles_c = get_compile_time_arg_val(0);
    constexpr uint32_t window_size_hw = get_compile_time_arg_val(1);

    constexpr uint32_t split_reader = get_compile_time_arg_val(2);

    constexpr uint32_t nsticks_per_core_by_nblocks = get_compile_time_arg_val(3);
    constexpr uint32_t in_c = get_compile_time_arg_val(4);
    constexpr uint32_t in_nblocks_c = get_compile_time_arg_val(5);
    constexpr uint32_t max_sticks_for_reduction = get_compile_time_arg_val(6);

    constexpr uint32_t in_cb_id_0 = get_compile_time_arg_val(7);
    constexpr uint32_t in_cb_id_1 = get_compile_time_arg_val(8);  // for split reader
    constexpr uint32_t in_scalar_cb_id_0 = get_compile_time_arg_val(9);
    constexpr uint32_t in_scalar_cb_id_1 = get_compile_time_arg_val(10);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(11);
    constexpr uint32_t one_scalar_per_core = get_compile_time_arg_val(12);

    DPRINT << "CB_PRINT START: in_ntiles_c=" << in_ntiles_c << " window_size_hw=" << window_size_hw
           << " split_reader=" << split_reader << " nsticks=" << nsticks_per_core_by_nblocks << " in_c=" << in_c
           << " in_nblocks_c=" << in_nblocks_c << " max_sticks=" << max_sticks_for_reduction
           << " in_cb_0=" << in_cb_id_0 << " in_cb_1=" << in_cb_id_1 << " scalar_cb_0=" << in_scalar_cb_id_0
           << " scalar_cb_1=" << in_scalar_cb_id_1 << " out_cb=" << out_cb_id << " one_scalar=" << one_scalar_per_core
           << ENDL();

    constexpr uint32_t remaining_elems = window_size_hw % max_sticks_for_reduction;
    constexpr uint32_t interm_reduction_chunks =
        remaining_elems ? window_size_hw / max_sticks_for_reduction + 1 : window_size_hw / max_sticks_for_reduction;

    constexpr uint32_t tile_size = get_tile_size(in_cb_id_0);

    DPRINT << "CB_PRINT: chunks=" << interm_reduction_chunks << " tile_size=" << tile_size << ENDL();

    // Wait for scalar initialization
    if constexpr (one_scalar_per_core) {
        cb_wait_front(in_scalar_cb_id_0, 1);
        // DPRINT << "CB_PRINT: Scalar CB " << in_scalar_cb_id_0 << " data:" << ENDL();
        // #if DEBUG_PRINT == 1
        // tt::data_movement::common::print_bf16_pages(get_read_ptr(in_scalar_cb_id_0), 32, 1, 0);
        // #endif
    }

    // Main processing loop
    for (uint32_t n = 0; n < nsticks_per_core_by_nblocks; ++n) {
        const bool reader0 = !(split_reader && (n & 0x1));
        const uint32_t curr_scalar_cb_id = (!reader0 && !one_scalar_per_core) ? in_scalar_cb_id_1 : in_scalar_cb_id_0;
        const uint32_t curr_in_cb_id = !reader0 ? in_cb_id_1 : in_cb_id_0;

        if constexpr (!one_scalar_per_core) {
            cb_wait_front(curr_scalar_cb_id, 1);
            // DPRINT << "CB_PRINT: Scalar CB " << curr_scalar_cb_id << " data:" << ENDL();
            // #if DEBUG_PRINT == 1
            // tt::data_movement::common::print_bf16_pages(get_read_ptr(curr_scalar_cb_id), 32, 1, 0);
            // #endif
        }

        for (uint32_t c_i = 0; c_i < in_nblocks_c; c_i++) {
            for (uint32_t chunk = 0; chunk < interm_reduction_chunks; chunk++) {
                // Wait for input data
                cb_wait_front(curr_in_cb_id, 1);

                DPRINT << "CB_PRINT: Input CB " << curr_in_cb_id << " [n=" << n << " c_i=" << c_i << " chunk=" << chunk
                       << "]:" << ENDL();

                // Get read pointer and print input data
                uint32_t l1_read_addr = get_read_ptr(curr_in_cb_id);
#if DEBUG_PRINT == 1
                tt::data_movement::common::print_bf16_pages(l1_read_addr, tile_size, 1, 0);
#endif

                // Pass data through to output
                cb_reserve_back(out_cb_id, 1);
                uint32_t l1_write_addr = get_write_ptr(out_cb_id);

                // Copy data from input to output using NOC
                noc_async_read(get_noc_addr(l1_read_addr), l1_write_addr, tile_size);
                noc_async_read_barrier();

                cb_push_back(out_cb_id, 1);
                cb_pop_front(curr_in_cb_id, 1);
            }
        }

        if constexpr (!one_scalar_per_core) {
            cb_pop_front(curr_scalar_cb_id, 1);
        }
    }

    // Pop scalar if one per core
    if constexpr (one_scalar_per_core) {
        cb_pop_front(in_scalar_cb_id_0, 1);
    }

    // DPRINT << "CB_PRINT: Finished processing" << ENDL();
}
