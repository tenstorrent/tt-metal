// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp"

void kernel_main() {
    constexpr bool dst_is_dram = (bool)get_compile_time_arg_val(0);
    constexpr uint32_t N = get_compile_time_arg_val(1);
    constexpr uint32_t output_cb_page_size = get_compile_time_arg_val(2);
    constexpr uint32_t num_rows = get_compile_time_arg_val(3);

    constexpr uint32_t X = get_compile_time_arg_val(4);
    constexpr uint32_t X_stride = get_compile_time_arg_val(5);
    constexpr uint32_t x_dim = get_compile_time_arg_val(6);

    constexpr uint32_t W_stride = get_compile_time_arg_val(7);
    constexpr uint32_t input_cb_page_size = get_compile_time_arg_val(8);
    constexpr uint32_t element_size_bytes = get_compile_time_arg_val(9);

    constexpr uint32_t num_blocks_total = get_compile_time_arg_val(10);
    constexpr uint32_t x_blocks = get_compile_time_arg_val(11);
    constexpr uint32_t w_blocks = get_compile_time_arg_val(12);
    constexpr uint32_t x_block_size = get_compile_time_arg_val(13);
    constexpr uint32_t w_block_size = get_compile_time_arg_val(14);
    constexpr uint32_t W = get_compile_time_arg_val(15);
    constexpr uint32_t output_tensor_page_size = get_compile_time_arg_val(16);
    constexpr uint32_t cb_id_in = tt::CBIndex::c_2;

    constexpr uint32_t x_block_size_bytes = x_block_size * element_size_bytes;
    constexpr uint32_t w_dim = N - 1;

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_block = get_arg_val<uint32_t>(1);
    const uint32_t end_block = get_arg_val<uint32_t>(2);

    const InterleavedAddrGen<dst_is_dram> s0 = {.bank_base_address = dst_addr, .page_size = output_tensor_page_size};

    uint32_t input_shape[N], perm[N], dest_strides[N];
    for (uint32_t i = 3; i < N + 3; i++) {
        input_shape[i - 3] = get_arg_val<uint32_t>(i);
        perm[i - 3] = get_arg_val<uint32_t>(i + N);
        dest_strides[i - 3] = get_arg_val<uint32_t>(i + 2 * N);
    }

    // Adjust for the transpose between X and W dimensions
    tt::data_movement::common::swap_elements(input_shape, x_dim, w_dim);
    for (uint32_t i = 0; i < N; i++) {
        if (perm[i] == x_dim) {
            perm[i] = w_dim;
        } else if (perm[i] == w_dim) {
            perm[i] = x_dim;
        }
    }

    uint32_t x_dim_in_dest = N;  // Invalid index
    for (uint32_t i = 0; i < N; ++i) {
        if (perm[i] == x_dim) {
            x_dim_in_dest = i;
            break;
        }
    }

    uint32_t src_multi_idx[N] = {0};
    uint32_t dest_multi_idx[N] = {0};
    for (uint32_t block = start_block; block < end_block; ++block) {
        // Compute block indices
        uint32_t w_block = block % w_blocks;
        uint32_t rem = block / w_blocks;
        uint32_t x_block = rem % x_blocks;
        rem = rem / x_blocks;
        uint32_t xw_block = rem % (num_rows / X);

        // Map linear index xw_block to multidimensional indices idxs[]
        uint32_t remainder = xw_block;

        uint32_t x_start = x_block * x_block_size;
        uint32_t x_end = min(x_start + x_block_size, X);
        uint32_t x_offset = x_start * element_size_bytes;

        uint32_t w_start = w_block * w_block_size;
        uint32_t w_end = min(w_start + w_block_size, W);

        uint32_t x_read_size_bytes = (x_end - x_start) * element_size_bytes;

        // Compute source indices
        size_t remaining = xw_block;
        for (int32_t d = N - 2; d >= 0; --d) {  // Exclude W dimension
            if (d == (int32_t)x_dim) {
                continue;  // Skip x_dim
            }
            src_multi_idx[d] = remaining % input_shape[d];
            remaining /= input_shape[d];
        }

        // Precompute dest_multi_idx and dest_linear_idx_base
        uint32_t dest_linear_idx_base = 0;
        for (uint32_t i = 0; i < N; ++i) {
            uint32_t src_idx = perm[i];
            if (src_idx != x_dim) {
                dest_multi_idx[i] = src_multi_idx[src_idx];
                if (i < N - 1) {  // Exclude W dimension
                    dest_linear_idx_base += dest_multi_idx[i] * dest_strides[i];
                }
            }
        }

        // Wait for transposed block
        cb_wait_front(cb_id_in, w_block_size);
        uint32_t transposed_buffer_read_addr = get_read_ptr(cb_id_in);
        for (uint32_t w = w_start; w < w_end; ++w) {
            src_multi_idx[x_dim] = w;
            dest_multi_idx[x_dim_in_dest] = w;

            // Update dest_linear_idx
            uint32_t dest_linear_idx = dest_linear_idx_base;
            if (x_dim_in_dest < N - 1) {  // Exclude W dimension
                dest_linear_idx += dest_multi_idx[x_dim_in_dest] * dest_strides[x_dim_in_dest];
            }
            uint64_t dst_noc_addr = get_noc_addr(dest_linear_idx, s0, x_offset);
            uint32_t l1_addr = transposed_buffer_read_addr + (w - w_start) * output_cb_page_size;
            noc_async_write(l1_addr, dst_noc_addr, x_read_size_bytes);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_id_in, w_block_size);
    }
}
