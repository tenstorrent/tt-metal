// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#define ENABLE_DEBUG_PRINT 0

#if ENABLE_DEBUG_PRINT == 1
#include "debug/dprint.h"

inline void print_pages(uint32_t l1_addr, uint32_t pagelen, uint32_t npages, uint32_t start = 0) {
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr) + start * pagelen;
    for (uint32_t page = 0; page < npages; ++page) {
        DPRINT << start + page << ": ";
        for (uint32_t j = 0; j < pagelen; ++j, ++ptr) {
            DPRINT << BF16(*ptr) << " ";
        }
        DPRINT << ENDL();
    }
}
#endif

void kernel_main() {
    uint32_t stick_nbytes = get_arg_val<uint32_t>(0);
    uint32_t in_nsticks_per_core = get_arg_val<uint32_t>(1);
    uint32_t scale_h = get_arg_val<uint32_t>(2);
    uint32_t scale_w = get_arg_val<uint32_t>(3);
    uint32_t in_w = get_arg_val<uint32_t>(4);
    uint32_t out_w = get_arg_val<uint32_t>(5);

    constexpr uint32_t in_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t is_reader = get_compile_time_arg_val(2);
    constexpr uint32_t config_cb_id = get_compile_time_arg_val(3);

    uint32_t reader_nsticks_per_core = (in_nsticks_per_core + is_reader) / 2;
    uint32_t writer_nsticks_per_core = in_nsticks_per_core / 2;
    uint32_t image_row_begin = is_reader ? 0 : reader_nsticks_per_core;
    uint32_t image_row_end = is_reader ? reader_nsticks_per_core : in_nsticks_per_core;
    uint32_t l1_read_addr = get_read_ptr(in_cb_id);
    uint32_t l1_write_addr = get_write_ptr(out_cb_id) + image_row_begin * scale_h * scale_w * stick_nbytes;

    uint32_t config_l1_addr = get_read_ptr(config_cb_id);
    volatile tt_l1_ptr uint16_t* config_data = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(config_l1_addr);

    uint32_t reader_idx = 0;
    if (!is_reader) {
        reader_idx = 4 * (scale_h * image_row_begin);
    }
    cb_reserve_back(out_cb_id, out_w);

    for (uint32_t row_begin = image_row_begin; row_begin < image_row_end; ++row_begin) {
        for (uint32_t sh = 0; sh < scale_h; sh++) {
            uint16_t corex = config_data[reader_idx++];
            uint16_t corey = config_data[reader_idx++];
            uint16_t offset = config_data[reader_idx++];
            reader_idx++;
            uint64_t src_remote_addr = get_noc_addr(corex, corey, l1_read_addr + offset * stick_nbytes);
            // replicate stick scale_w times.
            for (uint32_t sw = 0; sw < scale_w; sw++) {
                noc_async_read(src_remote_addr, l1_write_addr, stick_nbytes);
                l1_write_addr += stick_nbytes;
            }
        }
    }

    cb_push_back(out_cb_id, out_w);

    noc_async_write_barrier();
    noc_async_read_barrier();
}
