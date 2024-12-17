// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    constexpr bool dst_is_dram = (bool)get_compile_time_arg_val(0);
    constexpr uint32_t N = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr uint32_t num_rows = get_compile_time_arg_val(3);

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_row = get_arg_val<uint32_t>(1);
    const uint32_t end_row = get_arg_val<uint32_t>(2);

    const InterleavedAddrGen<dst_is_dram> s0 = {.bank_base_address = dst_addr, .page_size = page_size};

    // start at runtime arg 3 since address/start_block/end_block make up the first 3 args
    uint32_t input_shape[N], perm[N], dest_strides[N];
    for (uint32_t i = 3; i < N + 3; i++) {
        input_shape[i - 3] = get_arg_val<uint32_t>(i);
        perm[i - 3] = get_arg_val<uint32_t>(i + N);
        dest_strides[i - 3] = get_arg_val<uint32_t>(i + 2 * N);
    }

    uint32_t src_buffer_l1_addr = get_write_ptr(tt::CBIndex::c_0);
    uint32_t curr_addr = dst_addr;
    for (uint32_t row = start_row; row < end_row; ++row) {
        // Compute multi-dimensional index for the source row
        uint32_t src_multi_idx[N];
        size_t remaining = row;
        for (uint32_t i = 0; i < N - 1; ++i) {
            size_t dim = N - 2 - i;  // Start from the second last dimension
            src_multi_idx[dim] = remaining % input_shape[dim];
            remaining /= input_shape[dim];
        }
        src_multi_idx[N - 1] = 0;  // Row dimension index

        // Apply permutation to get destination multi-dimensional index
        uint32_t dest_multi_idx[N];
        for (uint32_t i = 0; i < N; ++i) {
            dest_multi_idx[i] = src_multi_idx[perm[i]];
        }

        // Convert destination multi-dimensional index to linear index
        uint32_t dest_linear_idx = 0;
        for (uint32_t i = 0; i < N - 1; ++i) {
            dest_linear_idx += dest_multi_idx[i] * dest_strides[i];
        }
        cb_wait_front(tt::CBIndex::c_0, 1);
        uint32_t l1_read_addr = get_read_ptr(tt::CBIndex::c_0);
        uint64_t dst_noc_addr = get_noc_addr(dest_linear_idx, s0);
        noc_async_write(l1_read_addr, dst_noc_addr, page_size);
        noc_async_write_barrier();
        cb_pop_front(tt::CBIndex::c_0, 1);
    }
}
