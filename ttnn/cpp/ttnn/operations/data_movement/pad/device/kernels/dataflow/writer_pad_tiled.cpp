// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <algorithm>
#include "api/dataflow/dataflow_api.h"
#include "common.hpp"

// This kernel keeps track of which page (tile) we are on from a logical tensor perspective, and fills the output with
// either the input or padding respectively
// For example, if we are padding (2, 2, 32, 32) -> (4, 4, 64, 64), then we condense the inner dims to tiles:
// (2, 2, 1, 1) -> (4, 4, 2, 2) and as incrementing through writing the output, [0:2, 0:2, 0:1, 0:1] will be
// tiles read from input, and the rest will be padding. So for this writer kernel, if we are within
// [0:2, 0:2, 0:1, 0:1] we wait for the reader to send us the correct tile, and then write it, otherwise we
// write padding.
void kernel_main() {
    constexpr uint32_t input_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t output_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t pad_val_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t page_size = get_compile_time_arg_val(3);
    constexpr uint32_t num_dims = get_compile_time_arg_val(4);
    constexpr uint32_t pad_value = get_compile_time_arg_val(5);
    constexpr uint32_t element_size = get_compile_time_arg_val(6);
    constexpr uint32_t num_elements = page_size / element_size;

    uint32_t rt_ind = 0;
    const uint32_t output_addr = get_arg_val<uint32_t>(rt_ind++);
    const uint32_t num_pages_to_write = get_arg_val<uint32_t>(rt_ind++);
    const uint32_t start_offset = get_arg_val<uint32_t>(rt_ind++);
    volatile tt_l1_ptr uint32_t* input_page_shape = (tt_l1_ptr uint32_t*)(get_arg_addr(rt_ind));
    volatile tt_l1_ptr uint32_t* output_page_shape = input_page_shape + num_dims;
    volatile tt_l1_ptr uint32_t* input_id_per_dim = output_page_shape + num_dims;
    volatile tt_l1_ptr uint32_t* output_id_per_dim = input_id_per_dim + num_dims;

    constexpr auto dst_args = TensorAccessorArgs<7>();

    const auto s0 = TensorAccessor(dst_args, output_addr, page_size);

    // Reserve and push the pad value into the circular buffer, generalized for any contiguous dtype
    cb_reserve_back(pad_val_cb_id, 1);
    uint32_t l1_write_addr = get_write_ptr(pad_val_cb_id);
    volatile tt_l1_ptr uint8_t* pad_val_page = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(l1_write_addr);
    const volatile tt_l1_ptr uint8_t* pad_val = reinterpret_cast<const volatile tt_l1_ptr uint8_t*>(&pad_value);
    for (uint32_t i = 0; i < num_elements; i++) {
        for (uint32_t b = 0; b < element_size; b++) {
            pad_val_page[i * element_size + b] = pad_val[b];
        }
    }
    cb_push_back(pad_val_cb_id, 1);
    // Our scratchpad cb is now a tile full of padding.

    bool within_input_region;
    uint32_t output_page_offset = start_offset;

    // Loop over all output pages to write
    for (uint32_t out_pages_written = 0; out_pages_written < num_pages_to_write; out_pages_written++) {
        within_input_region = true;
        for (uint32_t d = 0; d < num_dims; d++) {
            if (input_id_per_dim[d] < output_id_per_dim[d]) {
                within_input_region = false;
                break;
            }
        }

        // We have two cases, if we are within the input region, we wait for the reader to send us the correct tile
        // Otherwise we simply write the padding tile we have in our circular buffer
        uint64_t dst_noc_addr = get_noc_addr(output_page_offset, s0);
        if (within_input_region) {
            cb_wait_front(input_cb_id, 1);
            uint32_t l1_read_addr = get_read_ptr(input_cb_id);
            noc_async_write(l1_read_addr, dst_noc_addr, page_size);
            noc_async_write_barrier();
            advance_tensor_index(input_id_per_dim, input_page_shape, num_dims);
            cb_pop_front(input_cb_id, 1);
        } else {
            noc_async_write(l1_write_addr, dst_noc_addr, page_size);
            noc_async_write_barrier();
        }
        advance_tensor_index(output_id_per_dim, output_page_shape, num_dims);
        output_page_offset++;
    }
}
