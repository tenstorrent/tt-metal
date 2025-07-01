// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "reader_pool2d_sharded_common.hpp"
#define ENABLE_DEBUG_PRINT 1

#if ENABLE_DEBUG_PRINT == 1
#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#endif

void kernel_main() {
    constexpr uint32_t cb_src = get_compile_time_arg_val(0);
    constexpr uint32_t cb_dst = get_compile_time_arg_val(1);
    constexpr uint32_t nblocks = get_compile_time_arg_val(2);
    constexpr uint32_t window_hw = 0;    // get_compile_time_arg_val(3);
    constexpr uint32_t in_c_padded = 0;  // get_compile_time_arg_val(4);
    uint32_t in_l1_read_base_addr = get_write_ptr(cb_src);
    uint32_t out_l1_write_addr = get_write_ptr(cb_dst);
    DPRINT << "in_l1_read_base_addr:" << in_l1_read_base_addr << ENDL();
    // print_bf16_pages(in_l1_read_base_addr, 32*8, 32);
    for (uint32_t i = 0; i < nblocks; ++i) {
        cb_wait_front(cb_src, 1);
        cb_reserve_back(cb_dst, 1);

        uint32_t read_c = 8;

        for (uint32_t h = 0; h < window_hw; ++h) {
            for (uint32_t c = 0; c < in_c_padded / read_c; ++c) {
                const uint32_t read_offset = in_l1_read_base_addr + h * in_c_padded + c * read_c;
                noc_async_read_one_packet(get_noc_addr(read_offset), out_l1_write_addr, read_c);
                out_l1_write_addr += read_c;
            }
        }
        noc_async_read_barrier();  // At this line, read is complete.

        cb_push_back(cb_src, 1);
    }
}  // kernel_main()
