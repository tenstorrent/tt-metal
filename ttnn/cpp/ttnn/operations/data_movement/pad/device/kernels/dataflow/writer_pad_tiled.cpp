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
    constexpr uint32_t output_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t pad_val_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t page_size = get_compile_time_arg_val(3);
    constexpr uint32_t num_dims = get_compile_time_arg_val(4);
    constexpr uint32_t pad_value = get_compile_time_arg_val(5);
    const uint32_t element_size = get_compile_time_arg_val(6);

    uint32_t rt_ind = 0;
    const uint32_t output_addr = get_arg_val<uint32_t>(rt_ind++);
    const uint32_t num_pages_to_write = get_arg_val<uint32_t>(rt_ind++);
    const uint32_t start_offset = get_arg_val<uint32_t>(rt_ind++);
    volatile tt_l1_ptr uint32_t* input_page_shape = (tt_l1_ptr uint32_t*)(get_arg_addr(rt_ind));
    volatile tt_l1_ptr uint32_t* output_page_shape = input_page_shape + num_dims;
    volatile tt_l1_ptr uint32_t* input_odo = output_page_shape + num_dims;
    volatile tt_l1_ptr uint32_t* output_odo = input_odo + num_dims;

    constexpr auto dst_args = TensorAccessorArgs<6>();

    const auto s0 = TensorAccessor(dst_args, output_addr, page_size);

    // Reserve and push the pad value into the circular buffer, generalized for any contiguous dtype
    cb_reserve_back(pad_val_cb_id, 1);
    uint32_t l1_write_addr = get_write_ptr(pad_val_cb_id);
    volatile tt_l1_ptr uint8_t* pad_val_page = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(l1_write_addr);
    const volatile tt_l1_ptr uint8_t* pad_val = reinterpret_cast<const volatile tt_l1_ptr uint8_t*>(&pad_value);
    // volatile tt_l1_ptr uint32_t* l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_write_addr);
    for (uint32_t i = 0; i < page_size / element_size; i++) {
        for (uint32_t b = 0; b < element_size; b++) {
            pad_val_page[i * element_size + b] = pad_val[b];
        }
    }
    cb_push_back(pad_val_cb_id, 1);
    // tt::data_movement::common::print_bf16_pages(l1_write_addr, page_size / 2, 1);

    bool within_input_region;
    uint32_t output_page_offset = start_offset;

    for (uint32_t out_pages_written = 0; out_pages_written < num_pages_to_write; out_pages_written++) {
        within_input_region = true;
        for (uint32_t d = 0; d < num_dims; d++) {
            if (input_odo[d] < output_odo[d]) {
                within_input_region = false;
                break;
            }
        }

        uint64_t dst_noc_addr = get_noc_addr(output_page_offset, s0);
        if (within_input_region) {
            cb_wait_front(input_cb_id, 1);
            uint32_t l1_read_addr = get_read_ptr(input_cb_id);
            noc_async_write(l1_read_addr, dst_noc_addr, page_size);
            noc_async_write_barrier();
            next_index_u32(input_odo, input_page_shape, num_dims);
            cb_pop_front(input_cb_id, 1);
        } else {
            noc_async_write(l1_write_addr, dst_noc_addr, page_size);
            noc_async_write_barrier();
        }
        next_index_u32(output_odo, output_page_shape, num_dims);
        output_page_offset++;
    }
}
