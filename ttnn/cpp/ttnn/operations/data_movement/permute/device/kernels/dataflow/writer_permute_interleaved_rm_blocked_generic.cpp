// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <algorithm>
#include "dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

void kernel_main() {
    // Compile-time constants
    constexpr bool dst_is_dram = (bool)get_compile_time_arg_val(0);
    constexpr uint32_t N = get_compile_time_arg_val(1);
    constexpr uint32_t output_cb_page_size = get_compile_time_arg_val(2);
    constexpr uint32_t num_rows = get_compile_time_arg_val(3);

    constexpr uint32_t X = get_compile_time_arg_val(4);
    constexpr uint32_t X_stride = get_compile_time_arg_val(5);
    constexpr uint32_t x_dim = get_compile_time_arg_val(6);

    constexpr uint32_t W_stride = get_compile_time_arg_val(7);
    constexpr uint32_t input_cb_page_size = get_compile_time_arg_val(8);
    constexpr uint32_t element_size = get_compile_time_arg_val(9);

    constexpr uint32_t num_blocks_total = get_compile_time_arg_val(10);
    constexpr uint32_t x_blocks = get_compile_time_arg_val(11);
    constexpr uint32_t w_blocks = get_compile_time_arg_val(12);
    constexpr uint32_t x_block_size = get_compile_time_arg_val(13);
    constexpr uint32_t w_block_size = get_compile_time_arg_val(14);
    constexpr uint32_t W = get_compile_time_arg_val(15);
    constexpr uint32_t output_tensor_page_size = get_compile_time_arg_val(16);

    constexpr uint32_t cb_id_in = tt::CBIndex::c_2;

    // Precompute bytes-per-block along X
    constexpr uint32_t x_block_size_bytes = x_block_size * element_size;

    // W dimension is always the last dimension
    constexpr uint32_t w_dim = N - 1;

    // Calculate how many "non_x_rows" we have (these are the combinations of all dimensions except X)
    constexpr uint32_t non_x_rows = num_rows / X;

    // Destination base address
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_block = get_arg_val<uint32_t>(1);
    const uint32_t end_block = get_arg_val<uint32_t>(2);

    // Interleaved address configuration for the destination
    const InterleavedAddrGen<dst_is_dram> s0 = {.bank_base_address = dst_addr, .page_size = output_tensor_page_size};

    // Input shape, permutation, and destination strides
    // start at runtime arg 3 since address/start_block/end_block make up the first 3 args
    uint32_t input_shape[N], perm[N], dest_strides[N];
    for (uint32_t i = 3; i < N + 3; i++) {
        input_shape[i - 3] = get_arg_val<uint32_t>(i);
        perm[i - 3] = get_arg_val<uint32_t>(i + N);
        dest_strides[i - 3] = get_arg_val<uint32_t>(i + 2 * N);
    }

    // The source data was transposed between W and X by the previous kernel.
    // Adjust input_shape and perm to reflect that swap.
    tt::data_movement::common::swap_elements(input_shape, x_dim, w_dim);
    for (uint32_t i = 0; i < N; i++) {
        if (perm[i] == x_dim) {
            perm[i] = w_dim;
        } else if (perm[i] == w_dim) {
            perm[i] = x_dim;
        }
    }

    // Find where the original X dimension ended up in the permuted output
    uint32_t x_dim_in_dest = N;  // Will hold the position of x_dim in the permuted array
    for (uint32_t i = 0; i < N; ++i) {
        if (perm[i] == x_dim) {
            x_dim_in_dest = i;
            break;
        }
    }

    uint32_t src_multi_idx[N] = {0};
    uint32_t dest_multi_idx[N] = {0};

    // Process each block of data from start_block to end_block
    for (uint32_t block = start_block; block < end_block; ++block) {
        // Decompose linear block index into w_block, x_block, and xw_block
        uint32_t rem = block;

        // w_block: portion of the W dimension handled by this block
        const uint32_t w_block = rem % w_blocks;
        rem /= w_blocks;

        // x_block: portion of the X dimension handled by this block
        const uint32_t x_block = rem % x_blocks;
        rem /= x_blocks;

        // xw_block: which "non-X row set" we are in
        const uint32_t xw_block = rem % non_x_rows;

        // Compute start/end boundaries for the current X and W blocks
        const uint32_t x_start = x_block * x_block_size;
        const uint32_t x_end = std::min(x_start + x_block_size, X);

        const uint32_t w_start = w_block * w_block_size;
        const uint32_t w_end = std::min(w_start + w_block_size, W);

        // Compute the read size for the X dimension
        const uint32_t x_read_size_bytes = (x_end - x_start) * element_size;
        const uint32_t x_offset = x_start * element_size;

        // Decode xw_block into multi-dimensional indices excluding the W dimension and X dimension
        uint32_t remainder = xw_block;
        for (int32_t d = N - 2; d >= 0; --d) {
            if (d == (int32_t)x_dim) {
                // Skip the original X dimension index during this mapping
                continue;
            }
            src_multi_idx[d] = remainder % input_shape[d];
            remainder /= input_shape[d];
        }

        // Compute dest_multi_idx (excluding W dimension), and a base linear index
        // for all dimensions except W and X. We'll add W and X offsets later.
        uint32_t dest_linear_idx_base = 0;
        for (uint32_t i = 0; i < N; ++i) {
            uint32_t src_idx = perm[i];
            if (src_idx != x_dim) {
                dest_multi_idx[i] = src_multi_idx[src_idx];
                // Accumulate partial index product for all dimensions except W
                if (i < w_dim) {
                    dest_linear_idx_base += dest_multi_idx[i] * dest_strides[i];
                }
            }
        }

        // Wait for the transposed block data to be ready in the input CB
        cb_wait_front(cb_id_in, w_block_size);
        uint32_t transposed_buffer_read_addr = get_read_ptr(cb_id_in);

        // Iterate over the W dimension elements
        for (uint32_t w = w_start; w < w_end; ++w) {
            // Update indices for the current W
            src_multi_idx[x_dim] = w;
            dest_multi_idx[x_dim_in_dest] = w;

            // Compute final linear index for the current W
            uint32_t dest_linear_idx = dest_linear_idx_base;
            if (x_dim_in_dest < w_dim) {
                dest_linear_idx += dest_multi_idx[x_dim_in_dest] * dest_strides[x_dim_in_dest];
            }

            // Compute the NoC address for the output
            uint64_t dst_noc_addr = get_noc_addr(dest_linear_idx, s0, x_offset);

            // Compute the L1 address from which to write (offset by W-block pages)
            uint32_t l1_addr = transposed_buffer_read_addr + (w - w_start) * output_cb_page_size;

            // Perform an asynchronous write of the X-block to the destination
            noc_async_write(l1_addr, dst_noc_addr, x_read_size_bytes);
        }

        // Wait until all writes are completed before proceeding to the next block
        noc_async_write_barrier();

        // Pop the block from the input circular buffer, as we're done writing it
        cb_pop_front(cb_id_in, w_block_size);
    }
}
