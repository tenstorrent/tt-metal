// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    constexpr bool src0_is_dram = (bool)get_compile_time_arg_val(0);
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    constexpr uint32_t dims = get_compile_time_arg_val(2);

    const uint32_t src_addr = get_arg_val<uint32_t>(0);

    // Initialize shape, starts, ends, strides
    uint32_t shape[dims], starts[dims], ends[dims], strides[dims];
    for (uint32_t i = 1; i <= dims; i++) {
        shape[i - 1] = get_arg_val<uint32_t>(i);
        starts[i - 1] = get_arg_val<uint32_t>(i + dims);
        ends[i - 1] = get_arg_val<uint32_t>(i + 2 * dims);
        strides[i - 1] = get_arg_val<uint32_t>(i + 3 * dims);
    }

    // Calculate the product array, excluding the last dimension
    uint32_t prod[dims];
    for (uint32_t i = 0; i < dims - 1; i++) {
        prod[i] = 1;
        for (uint32_t j = i + 1; j < dims - 1; j++) {  // Exclude the last dimension
            prod[i] *= shape[j];
        }
    }
    prod[dims - 1] = 1;  // Not used, but set to 1 for completeness

    const InterleavedAddrGen<src0_is_dram> s0 = {.bank_base_address = src_addr, .page_size = page_size};

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_out0 = 24;
    uint32_t src_buffer_l1_addr = get_write_ptr(cb_id_in0);
    volatile tt_l1_ptr uint16_t* in_stick = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(src_buffer_l1_addr);

    uint32_t index[dims];  // To hold current index in each of the first dims-1 dimensions
    index[dims - 1] = 0;   // Initialize the last index to 0
    for (uint32_t i = 0; i < dims - 1; i++) {
        index[i] = starts[i];  // Initialize the index with the start values
    }

    // Flag to determine when to terminate the loop
    bool done = false;

    while (!done) {
        // Calculate the base linear index based on the first dims-1 indices
        uint32_t base_linear_index = 0;
        for (uint32_t i = 0; i < dims - 1; i++) {
            base_linear_index += index[i] * prod[i];
        }

        // Now iterate over the last dimension
        uint32_t out_stick_id = 0;
        // Perform the read operation
        noc_async_read_page(base_linear_index, s0, src_buffer_l1_addr);
        // Reserve space in the output buffer
        cb_reserve_back(cb_id_out0, 1);
        noc_async_read_barrier();
        for (uint32_t l = starts[dims - 1]; l < ends[dims - 1]; l += strides[dims - 1]) {
            // Write the element into the output buffer
            volatile tt_l1_ptr uint16_t* out_stick =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_id_out0));
            out_stick[out_stick_id] = in_stick[l];  // Assuming you write one element at a time
            out_stick_id++;
        }
        cb_push_back(cb_id_out0, 1);
        if constexpr (dims == 1) {
            break;  // If there's only one dimension, we're done
        } else {
            // Increment the indices for the first dims-1 dimensions
            for (int32_t i = dims - 2; i >= 0; i--) {  // Start from the last of the first dims-1
                index[i] += strides[i];
                if (index[i] < ends[i]) {
                    break;  // Successfully incremented this dimension, no carry over
                } else {
                    index[i] = starts[i];  // Reset this dimension and carry over to the next
                    if (i == 0) {
                        done = true;  // If the first dimension is reset, we've completed all iterations
                    }
                }
            }
        }
    }
}
