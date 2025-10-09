// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <algorithm>
#include "dataflow_api.h"
#include "common.hpp"

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
    volatile tt_l1_ptr uint32_t* input_id_per_dim = output_page_shape + num_dims;
    volatile tt_l1_ptr uint32_t* output_id_per_dim = input_id_per_dim + num_dims;

    constexpr auto dst_args = TensorAccessorArgs<3>();

    const auto s0 = TensorAccessor(dst_args, input_addr, page_size);

    bool within_input_region;
    uint32_t input_page_offset = start_offset;

    // This kernel keeps track of which page (tile) we are on from a logical tensor perspective
    // and reads from the input tensor only when we are within the input region
    // The writer will be waiting for the correct page to be available in the input circular buffer
    // For example, if we are padding (2, 2, 32, 32) -> (4, 4, 64, 64), then we condense the inner dims to tiles:
    // (2, 2, 1, 1) -> (4, 4, 2, 2) and as incrementing through writing the output, [0:2, 0:2, 0:1, 0:1] will be
    // tiles read from input, and the rest will be padding. So for this reader kernel, we will only read when
    // [0:2, 0:2, 0:1, 0:1] is reached, and skip reads otherwise.

    for (uint32_t out_pages_written = 0; out_pages_written < num_pages_to_write; out_pages_written++) {
        within_input_region = true;
        for (uint32_t d = 0; d < num_dims; d++) {
            if (input_id_per_dim[d] < output_id_per_dim[d]) {
                within_input_region = false;
                break;
            }
        }

        if (within_input_region) {
            cb_reserve_back(input_cb_id, 1);
            uint32_t l1_write_addr = get_write_ptr(input_cb_id);
            uint64_t src_noc_addr = get_noc_addr(input_page_offset, s0);
            noc_async_read(src_noc_addr, l1_write_addr, page_size);
            noc_async_read_barrier();
            cb_push_back(input_cb_id, 1);
            input_page_offset++;
            advance_tensor_index(input_id_per_dim, input_page_shape, num_dims);
        }
        advance_tensor_index(output_id_per_dim, output_page_shape, num_dims);
    }
}
