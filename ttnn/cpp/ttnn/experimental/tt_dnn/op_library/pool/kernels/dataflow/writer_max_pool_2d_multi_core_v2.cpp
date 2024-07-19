// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

#define ENABLE_DEBUG_PRINT 0

#if ENABLE_DEBUG_PRINT == 1
#include "debug/dprint.h"
SliceRange srt = SliceRange{ .h0 = 0, .h1 = 32, .hs = 8, .w0 = 0, .w1 = 32, .ws = 8 };
SliceRange srr = SliceRange{ .h0 = 0, .h1 = 1, .hs = 8, .w0 = 0, .w1 = 32, .ws = 1 };
SliceRange srr2 = SliceRange{ .h0 = 0, .h1 = 1, .hs = 8, .w0 = 0, .w1 = 64, .ws = 2 };

inline void print_pages(uint32_t l1_addr, uint32_t pagelen, uint32_t npages, uint32_t start = 0) {
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr) + start * pagelen;
    for (uint32_t page = 0; page < npages; ++ page) {
        DPRINT << start + page << ": ";
        for (uint32_t j = 0; j < pagelen; ++ j, ++ ptr) {
            DPRINT << BF16(*ptr) << " ";
        }
        DPRINT << ENDL();
    }
}
#endif

/**
 * Max-pool 2D.
 */
void kernel_main() {
    constexpr uint32_t nblocks = get_compile_time_arg_val(3);

    constexpr uint32_t out_cb_id = tt::CB::c_out0;
    constexpr uint32_t sharded_out_cb_id = tt::CB::c_out1;

    const uint32_t out_nbytes_c = get_arg_val<uint32_t>(1);
    const uint32_t out_ntiles_c = get_arg_val<uint32_t>(2);
    const uint32_t nsticks_per_core = get_arg_val<uint32_t>(3);
    // uint32_t core_offset_out_row_id = get_arg_val<uint32_t>(4);
    const uint32_t nsticks_per_core_by_nblocks = get_arg_val<uint32_t>(5);
    const uint32_t out_c = get_arg_val<uint32_t>(6);
    const bool is_partial_tile = out_c < 32;

    uint64_t out_noc_addr = get_noc_addr(get_read_ptr(sharded_out_cb_id));

    cb_reserve_back(sharded_out_cb_id, nsticks_per_core);

    uint32_t npages_to_wait_on = is_partial_tile ? 1 : nblocks;

    for (uint32_t stick = 0; stick < nsticks_per_core_by_nblocks; ++ stick) {
        cb_wait_front(out_cb_id, npages_to_wait_on * out_ntiles_c);
        uint32_t out_l1_read_addr = get_read_ptr(out_cb_id);

        if (is_partial_tile) {
            for (uint32_t block = 0; block < nblocks; ++ block) {
                noc_async_write(out_l1_read_addr, out_noc_addr, out_nbytes_c);
                out_noc_addr += out_nbytes_c;
                out_l1_read_addr += out_nbytes_c;
            }
        } else {
            noc_async_write(out_l1_read_addr, out_noc_addr, out_nbytes_c);
            out_noc_addr += out_nbytes_c;
        }

        noc_async_write_barrier();

        cb_pop_front(out_cb_id, npages_to_wait_on * out_ntiles_c);
    }
    cb_push_back(sharded_out_cb_id, nsticks_per_core);
    cb_wait_front(sharded_out_cb_id, nsticks_per_core);
} // kernel_main()
