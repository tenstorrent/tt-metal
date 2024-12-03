// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp"

// Function template to swap two elements in a uint32_t array
template <size_t N>
FORCE_INLINE void swap_elements(uint32_t (&array)[N], size_t i, size_t j) {
    // Perform the swap
    uint32_t temp = array[i];
    array[i] = array[j];
    array[j] = temp;
}

inline void print_pages(uint32_t l1_addr, uint32_t pagelen, uint32_t npages, uint32_t start = 0) {
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr) + start * pagelen;
    for (uint32_t page = 0; page < npages; ++page) {
        DPRINT << start + page << ": ";
        for (uint32_t j = 0; j < pagelen; ++j, ++ptr) {
            DPRINT << BF16(*ptr) << " ";
        }
        DPRINT << ENDL();
    }
    DPRINT << ENDL();
}

FORCE_INLINE void transpose_XW_to_WX(uint32_t input_l1_addr, uint32_t output_l1_addr, uint32_t X, uint32_t W, uint32_t element_size, uint32_t input_page_size, uint32_t output_page_size) {
    volatile tt_l1_ptr uint8_t* input_ptr = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(input_l1_addr);
    volatile tt_l1_ptr uint8_t* output_ptr = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(output_l1_addr);
    // transpose from XW, where X is outer and W inner, to WX, where W is outer and X is inner
    // each element is element_size bytes
    // each row is W elements, and each row is separated by input_page_size bytes
    // each output row is X elements, and each row is separated by output_page_size bytes

    for (uint32_t x = 0; x < X; ++x) {
        for (uint32_t w = 0; w < W; ++w) {
            // Compute the input and output addresses
            uint32_t input_addr = x * input_page_size + w * element_size;
            uint32_t output_addr = w * output_page_size + x * element_size;
            // Copy the element - do we have memcpy? use this for now
            for (uint32_t i = 0; i < element_size; ++i) {
                output_ptr[output_addr + i] = input_ptr[input_addr + i];
            }
        }
    }
}

void kernel_main() {
    constexpr bool dst_is_dram = (bool)get_compile_time_arg_val(0);
    constexpr uint32_t N = get_compile_time_arg_val(1);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(2);
    constexpr uint32_t num_rows = get_compile_time_arg_val(3);

    constexpr uint32_t X = get_compile_time_arg_val(4);
    // X_stride and x_dim along the input tensor
    constexpr uint32_t X_stride = get_compile_time_arg_val(5);
    constexpr uint32_t x_dim = get_compile_time_arg_val(6);

    constexpr uint32_t W = get_compile_time_arg_val(7);
    // W_stride and w_dim along the output tensor
    constexpr uint32_t W_stride = get_compile_time_arg_val(8);
    constexpr uint32_t input_page_size = get_compile_time_arg_val(9);
    constexpr uint32_t element_size_bytes = get_compile_time_arg_val(10);

    constexpr uint32_t w_dim = N - 1;

    // // DPRINT << "N = " << N << ENDL();
    // DPRINT << "output_page_size = " << output_page_size << ENDL();
    // // DPRINT << "num_rows = " << num_rows << ENDL();
    // // DPRINT << "X = " << X << ENDL();
    // // DPRINT << "X_stride = " << X_stride << ENDL();
    // // DPRINT << "x_dim = " << x_dim << ENDL();
    // // DPRINT << "W = " << W << ENDL();
    // DPRINT << "W_stride = " << W_stride << ENDL();
    // // DPRINT << "w_dim = " << w_dim << ENDL() << ENDL();
    // DPRINT << "input_page_size = " << input_page_size << ENDL();
    // DPRINT << "element_size_bytes = " << element_size_bytes << ENDL();


    const uint32_t dst_addr = get_arg_val<uint32_t>(0);

    const InterleavedAddrGen<dst_is_dram> s0 = {.bank_base_address = dst_addr, .page_size = output_page_size};

    uint32_t input_shape[N], perm[N], dest_strides[N];
    for (uint32_t i = 1; i <= N; i++) {
        input_shape[i - 1] = get_arg_val<uint32_t>(i);
        perm[i - 1] = get_arg_val<uint32_t>(i + N);
        dest_strides[i - 1] = get_arg_val<uint32_t>(i + 2 * N);
    }

    // after we transpose w and x, the perm is different
    // perm[i] = x_dim becomes perm[i] == N - 1 as it's been in the correct position (i == N - 1)
    // perm[j] = N - 1 becomes perm[j] = x_dim as x_dim has been moved to the end
    // input shape[i] = X becomes input_shape[i] = W as we're transposing the X dimension
    // input shape[j] = W becomes input_shape[j] = X as we're transposing the W dimension
    // dest strides is slightly incorrect as it assumes W is in its final position
    swap_elements(input_shape, x_dim, w_dim);
    for (uint32_t i = 0; i < N; i++) {
        if (perm[i] == x_dim) {
            perm[i] = N - 1;
        } else if (perm[i] == N - 1) {
            perm[i] = x_dim;
        }
    }

    // for (uint32_t i = 0; i < N; i++) {
    //     DPRINT << "input_shape[" << i << "] = " << input_shape[i] << " ";
    // }
    // DPRINT << ENDL();
    // for (uint32_t i = 0; i < N; i++) {
    //     DPRINT << "perm[" << i << "] = " << perm[i] << " ";
    // }
    // DPRINT << ENDL();
    // for (uint32_t i = 0; i < N; i++) {
    //     DPRINT << "dest_strides[" << i << "] = " << dest_strides[i] << " ";
    // }
    // DPRINT << ENDL() << ENDL();


    uint32_t curr_addr = dst_addr;
    for (uint32_t block = 0; block < num_rows/X; ++block) {
        // Compute multi-dimensional index for the source block
        uint32_t src_multi_idx[N];
        size_t remaining = block;
        for (int32_t d = N - 2; d >= 0; --d) { // Exclude W dimension
            if (d == (int32_t)x_dim) {
                continue; // Skip post-XW transpose w dimension
            }
            src_multi_idx[d] = remaining % input_shape[d];
            remaining /= input_shape[d];
        }
        // Apply permutation to get destination multi-dimensional index

        src_multi_idx[N - 1] = 0;  // Row dimension index
        src_multi_idx[N - 1] = 0;  // Row dimension index
        cb_wait_front(tt::CBIndex::c_0, X);
        uint32_t src_buffer_l1_addr = get_read_ptr(tt::CBIndex::c_0);
        print_pages(src_buffer_l1_addr, tt::data_movement::common::round_up<W, 16>(), X, 0);
        // Transpose the X*W*element_size block
        uint32_t transposed_buffer_read_addr = get_read_ptr(tt::CBIndex::c_1);
        transpose_XW_to_WX(src_buffer_l1_addr, transposed_buffer_read_addr, X, W, element_size_bytes, input_page_size, output_page_size);
        print_pages(transposed_buffer_read_addr, tt::data_movement::common::round_up<X, 16>(), W, 0);
        for (uint32_t w = 0; w < W; ++w) {
            src_multi_idx[x_dim] = w;
            uint32_t dest_multi_idx[N];
            for (uint32_t i = 0; i < N; ++i) {
                dest_multi_idx[i] = src_multi_idx[perm[i]];
            }
            for (uint32_t i = 0; i < N; i++) {
                DPRINT << "src_multi_idx[" << i << "] = " << src_multi_idx[i] << " ";
            }
            DPRINT << ENDL();
            for (uint32_t i = 0; i < N; i++) {
                DPRINT << "dest_multi_idx[" << i << "] = " << dest_multi_idx[i] << " ";
            }
            DPRINT << ENDL();

            // Convert destination multi-dimensional index to linear index
            uint32_t dest_linear_idx = 0;
            for (uint32_t i = 0; i < N - 1; ++i) {
                dest_linear_idx += dest_multi_idx[i] * dest_strides[i];
            }

            uint64_t dst_noc_addr = get_noc_addr(dest_linear_idx, s0);
            noc_async_write(transposed_buffer_read_addr + w * output_page_size, dst_noc_addr, output_page_size);
            noc_async_write_barrier();
        }
        cb_pop_front(tt::CBIndex::c_0, X);
    }
}
