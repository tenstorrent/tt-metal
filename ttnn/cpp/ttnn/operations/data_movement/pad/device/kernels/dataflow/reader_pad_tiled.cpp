// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <algorithm>
#include "dataflow_api.h"

static inline int next_index_u32(volatile tt_l1_ptr uint32_t* idx, volatile tt_l1_ptr uint32_t* dims, uint32_t ndims) {
    // increment least-significant dim first
    for (uint32_t d = ndims; d-- > 0;) {
        uint32_t v = idx[d] + 1;
        if (v < dims[d]) {
            idx[d] = v;
            return 1;
        }
        idx[d] = 0;  // wrap and carry
    }
    return 0;  // overflowed most-significant dim
}

void kernel_main() {
    constexpr uint32_t input_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    constexpr uint32_t num_dims = get_compile_time_arg_val(2);

    uint32_t rt_ind = 0;
    const uint32_t input_addr = get_arg_val<uint32_t>(rt_ind++);
    const uint32_t num_pages_to_write = get_arg_val<uint32_t>(rt_ind++);
    const uint32_t start_offset = get_arg_val<uint32_t>(rt_ind++);
    volatile tt_l1_ptr uint32_t* input_page_shape = (tt_l1_ptr uint32_t*)(get_arg_addr(rt_ind));
    volatile tt_l1_ptr uint32_t* output_page_shape = input_page_shape + num_dims;
    volatile tt_l1_ptr uint32_t* input_odo = output_page_shape + num_dims;
    volatile tt_l1_ptr uint32_t* output_odo = input_odo + num_dims;

    constexpr auto dst_args = TensorAccessorArgs<3>();

    const auto s0 = TensorAccessor(dst_args, input_addr, page_size);

    bool within_input_region;
    uint32_t input_page_offset = start_offset;

    for (uint32_t out_pages_written = 0; out_pages_written < num_pages_to_write; out_pages_written++) {
        within_input_region = true;
        for (uint32_t d = 0; d < num_dims; d++) {
            if (input_odo[d] < output_odo[d]) {
                within_input_region = false;
                break;
            }
        }
        // DPRINT << "within_input_region: " << (uint32_t)within_input_region << ENDL();
        DPRINT << "written pages: " << out_pages_written << ENDL();

        if (within_input_region) {
            cb_reserve_back(input_cb_id, 1);
            uint32_t l1_write_addr = get_write_ptr(input_cb_id);
            uint64_t src_noc_addr = get_noc_addr(input_page_offset, s0);
            noc_async_read(src_noc_addr, l1_write_addr, page_size);
            noc_async_read_barrier();
            cb_push_back(input_cb_id, 1);
            DPRINT << "SENT PAGE" << ENDL();
            input_page_offset++;
            next_index_u32(input_odo, input_page_shape, num_dims);
        }
        next_index_u32(output_odo, output_page_shape, num_dims);
    }
    DPRINT << "Finished!" << ENDL();
}
