// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <algorithm>
#include "dataflow_api.h"

void kernel_main() {
    constexpr bool src0_is_dram = (bool)get_compile_time_arg_val(0);
    constexpr uint32_t N = get_compile_time_arg_val(1);
    constexpr uint32_t input_cb_page_size = get_compile_time_arg_val(2);
    constexpr uint32_t num_rows = get_compile_time_arg_val(3);
    constexpr uint32_t x_dim = get_compile_time_arg_val(4);
    constexpr uint32_t num_blocks_total = get_compile_time_arg_val(5);
    constexpr uint32_t x_blocks = get_compile_time_arg_val(6);
    constexpr uint32_t w_blocks = get_compile_time_arg_val(7);
    constexpr uint32_t x_block_size = get_compile_time_arg_val(8);
    constexpr uint32_t w_block_size = get_compile_time_arg_val(9);
    constexpr uint32_t element_size = get_compile_time_arg_val(10);
    constexpr uint32_t input_tensor_page_size = get_compile_time_arg_val(11);

    // Precomputed constants: size of a 32 element block along the W dimension (measured in bytes)
    constexpr uint32_t w_block_size_bytes = w_block_size * element_size;

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t start_block = get_arg_val<uint32_t>(1);
    uint32_t end_block = get_arg_val<uint32_t>(2);

    // Input shape and strides (excluding W dimension and measured in rows, not bytes)
    // start at runtime arg 3 since address/start_block/end_block make up the first 3 args
    uint32_t input_shape[N], src_strides[N];
    for (uint32_t i = 3; i < N + 3; i++) {
        input_shape[i - 3] = get_arg_val<uint32_t>(i);
        src_strides[i - 3] = get_arg_val<uint32_t>(i + N);
    }

    /**
     * We have a multidimensional tensor:
     * - num_blocks_total = (rows * x_blocks * w_blocks) where rows = num_rows / X
     *   Here, 'rows' represent the combination of all rows before and after X dimension.
     *   So: rows * X * W_dimension = total number of elements (conceptually).
     *
     * For each 'block':
     *   - Compute which w_block and x_block this corresponds to.
     *   - Then compute which row set (xw_block) we are in.
     */

    // x_dim is the dimension along which we are reading the tensor, as it's the new W dimension in the output tensor
    uint32_t X = input_shape[x_dim];
    uint32_t X_stride = src_strides[x_dim];

    const InterleavedAddrGen<src0_is_dram> s0 = {.bank_base_address = src_addr, .page_size = input_tensor_page_size};

    uint32_t idxs[N];
    idxs[N - 1] = 0;
    uint32_t non_x_rows = num_rows / X;

    for (uint32_t block = start_block; block < end_block; ++block) {
        // Decompose block into w_block, x_block, and xw_block indices
        uint32_t rem = block;
        const uint32_t w_block = rem % w_blocks;  // Which W block are we in?
        rem /= w_blocks;

        const uint32_t x_block = rem % x_blocks;  // Which X block?
        rem /= x_blocks;

        uint32_t xw_block = rem % (non_x_rows);  // Which row set (beyond X dimension)?
        uint32_t remainder = xw_block;

        // Compute X block boundaries
        uint32_t x_start = x_block * x_block_size;
        uint32_t x_end = std::min(x_start + x_block_size, X);

        // Compute W block boundaries
        uint32_t w_start = w_block * w_block_size;
        uint32_t w_end = std::min(w_start + w_block_size, input_shape[N - 1]);
        uint32_t w_offset = w_start * element_size;

        uint32_t w_read_size_bytes = (w_end - w_start) * element_size;

        // Map linear index i to multidimensional indices idxs[]
        // We skip x_dim when doing this mapping and set it separately later
        for (int32_t d = N - 2; d >= 0; --d) {  // Exclude W dimension
            if (d == (int32_t)x_dim) {
                idxs[d] = 0;  // Initialize x_dim to zero (will be set in inner loop)
                continue;     // Skip x_dim during mapping
            }
            idxs[d] = remainder % input_shape[d];
            remainder /= input_shape[d];
        }
        idxs[N - 1] = 0;  // Initialize W dimension index to zero if not already set

        // Precompute the base address offset (excluding x_dim)
        uint64_t base_addr_offset = 0;
        for (uint32_t d = 0; d < N; ++d) {
            if (d != x_dim) {
                base_addr_offset += idxs[d] * src_strides[d];
            }
        }

        // Reserve space in the circular buffer for the X-block length
        cb_reserve_back(tt::CBIndex::c_0, x_block_size);
        uint32_t src_buffer_l1_addr = get_write_ptr(tt::CBIndex::c_0);

        // We read in 'x_block_len' chunks along the X dimension
        uint32_t page_offset = 0;
        // Read along the X dimension
        for (uint32_t x = x_start; x < x_end; ++x) {
            // Compute the address offset for this index
            uint64_t addr_offset = base_addr_offset + x * X_stride;
            uint64_t src_noc_addr = get_noc_addr(addr_offset, s0, w_offset);

            // Perform async read of the current line (w_block_len elements) into L1
            noc_async_read(src_noc_addr, src_buffer_l1_addr + page_offset, w_read_size_bytes);

            // Advance output pointer by one page size for next row
            page_offset += input_cb_page_size;
        }
        // Wait for all async reads to complete before proceeding
        noc_async_read_barrier();
        // Push the filled block into the circular buffer
        cb_push_back(tt::CBIndex::c_0, x_block_size);
    }
}
