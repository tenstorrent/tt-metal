// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"

uint32_t compute_src_row_id(
    uint32_t dst_row_id, uint32_t rank, const uint32_t* shape, const uint32_t* dims_to_flip, const uint32_t* strides) {
    // Step 1: dst_linear_id -> multi-dimensional index
    uint32_t dst_multi_dim[rank - 1];  // TODO: do not use VLAs
    uint32_t remainder = dst_row_id;
    for (uint32_t i = 0; i < rank - 1; i++) {
        dst_multi_dim[i] = remainder / strides[i];
        remainder = remainder % strides[i];
    }

    // Step 2: Flip dst multi dim to find src multi dim
    uint32_t src_multi_dim[rank - 1];  // TODO: do not use VLAs
    for (uint32_t i = 0; i < rank - 1; ++i) {
        if (dims_to_flip[i]) {
            src_multi_dim[i] = shape[i] - dst_multi_dim[i] - 1;
        } else {
            src_multi_dim[i] = dst_multi_dim[i];
        }
    }

    // Step 3: src multi dim to linear
    uint32_t src_row_id = 0;
    for (uint32_t i = 0; i < rank - 1; i++) {
        src_row_id += src_multi_dim[i] * strides[i];
    }
    return src_row_id;
}

void kernel_main() {
    // Compile time arguments
    constexpr bool src_is_dram = static_cast<bool>(get_compile_time_arg_val(0));
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    constexpr uint32_t rank = get_compile_time_arg_val(2);
    constexpr uint32_t element_size = get_compile_time_arg_val(3);

    // Runtime arguments
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_row = get_arg_val<uint32_t>(1);
    const uint32_t end_row = get_arg_val<uint32_t>(2);

    uint32_t input_shape[rank], input_row_strides[rank], dims_to_flip[rank];
    for (uint32_t i = 0; i < rank; i++) {
        input_shape[i] = get_arg_val<uint32_t>(i + 3);
        input_row_strides[i] = get_arg_val<uint32_t>(i + rank + 3);
        dims_to_flip[i] = get_arg_val<uint32_t>(i + rank + rank + 3);
    }

    // Derived constants
    const uint32_t row_width = input_shape[rank - 1];
    // const bool is_vertical_flip = static_cast<bool>(dims_to_flip[rank - 2]);
    const bool is_horizontal_flip = static_cast<bool>(dims_to_flip[rank - 1]);

    constexpr uint32_t cb_id = tt::CBIndex::c_0;
    const InterleavedAddrGen<src_is_dram> s0 = {.bank_base_address = src_addr, .page_size = page_size};

    for (uint32_t row_id = start_row; row_id < end_row; row_id++) {
        uint32_t src_row_id = compute_src_row_id(row_id, rank, input_shape, dims_to_flip, input_row_strides);

        // DPRINT << src_row_id << " | " << row_id << ENDL();

        cb_reserve_back(cb_id, 1);
        uint32_t l1_buffer_addr = get_write_ptr(cb_id);
        uint64_t read_noc_addr = get_noc_addr(src_row_id, s0);
        noc_async_read(read_noc_addr, l1_buffer_addr, page_size);
        noc_async_read_barrier();

        // for (uint32_t col_id = 0; col_id < row_width; ++col_id) {
        //     DPRINT << uint32_t(reinterpret_cast<uint32_t*>(l1_buffer_addr)[col_id]) << ", ";
        // }
        // DPRINT << ENDL();

        if (is_horizontal_flip) {
            // flip elements within the row
            uint8_t* row_bytes = reinterpret_cast<uint8_t*>(l1_buffer_addr);
            for (uint32_t col_id = 0; col_id < row_width / 2; ++col_id) {
                uint32_t left = col_id * element_size;
                uint32_t right = (row_width - 1 - col_id) * element_size;
                for (uint32_t b = 0; b < element_size; ++b) {
                    uint8_t tmp = row_bytes[left + b];
                    row_bytes[left + b] = row_bytes[right + b];
                    row_bytes[right + b] = tmp;
                }
            }
        }
        cb_push_back(cb_id, 1);
    }
}
