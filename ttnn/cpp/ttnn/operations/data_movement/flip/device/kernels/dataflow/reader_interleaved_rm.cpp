// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

uint32_t compute_src_row_id(
    const uint32_t* dst_multi_dim,
    uint32_t rank,
    const uint32_t* input_shape,
    const uint32_t* dims_to_flip,
    const uint32_t* strides) {
    uint32_t src_row_id = 0;
    for (uint32_t i = 0; i < rank - 1; ++i) {
        uint32_t src_dim = dims_to_flip[i] ? (input_shape[i] - dst_multi_dim[i] - 1) : dst_multi_dim[i];
        src_row_id += src_dim * strides[i];
    }
    return src_row_id;
}

void kernel_main() {
    constexpr uint32_t page_size = get_named_compile_time_arg_val("page_size");
    constexpr uint32_t rank = get_named_compile_time_arg_val("rank");
    constexpr uint32_t element_size = get_named_compile_time_arg_val("element_size");
    constexpr auto src_args = TensorAccessorArgs<0>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_row = get_arg_val<uint32_t>(1);
    const uint32_t end_row = get_arg_val<uint32_t>(2);

    uint32_t input_shape[rank], input_row_strides[rank], dims_to_flip[rank];
    for (uint32_t i = 0; i < rank; i++) {
        input_shape[i] = get_arg_val<uint32_t>(i + 3);
        input_row_strides[i] = get_arg_val<uint32_t>(i + rank + 3);
        dims_to_flip[i] = get_arg_val<uint32_t>(i + rank + rank + 3);
    }

    const uint32_t row_width = input_shape[rank - 1];
    const bool is_horizontal_flip = static_cast<bool>(dims_to_flip[rank - 1]);
    constexpr uint32_t cb_id = tt::CBIndex::c_0;
    const auto s0 = TensorAccessor(src_args, src_addr, page_size);

    uint32_t dst_multi_dim[rank - 1];
    uint32_t remainder = start_row;
    for (uint32_t i = 0; i < rank - 1; i++) {
        dst_multi_dim[i] = remainder / input_row_strides[i];
        remainder = remainder % input_row_strides[i];
    }

    for (uint32_t row_id = start_row; row_id < end_row; row_id++) {
        uint32_t src_row_id = compute_src_row_id(dst_multi_dim, rank, input_shape, dims_to_flip, input_row_strides);

        cb_reserve_back(cb_id, 1);
        uint32_t l1_buffer_addr = get_write_ptr(cb_id);
        noc_async_read_page(src_row_id, s0, l1_buffer_addr);
        noc_async_read_barrier();

        if (is_horizontal_flip) {
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

        for (int j = rank - 2; j >= 0; j--) {
            dst_multi_dim[j]++;
            if (dst_multi_dim[j] == input_shape[j]) {
                dst_multi_dim[j] = 0;
            } else {
                break;
            }
        }
    }
}
