// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "permute_struct.hpp"

void kernel_main() {
    constexpr auto PermuteWriterArgs = PERMUTE_WRITER_STRUCT;

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_row = get_arg_val<uint32_t>(1);
    const uint32_t end_row = get_arg_val<uint32_t>(2);

    const InterleavedAddrGen<PermuteWriterArgs.dst_is_dram> s0 = {
        .bank_base_address = dst_addr, .page_size = PermuteWriterArgs.output_rm_page_size};

    uint32_t src_buffer_l1_addr = get_write_ptr(tt::CBIndex::c_0);
    uint32_t curr_addr = dst_addr;
    uint32_t src_multi_idx[PermuteWriterArgs.rank];
    uint32_t dest_multi_idx[PermuteWriterArgs.rank];

    for (uint32_t row = start_row; row < end_row; ++row) {
        // Compute multi-dimensional index for the source row
        size_t remaining = row;
        for (uint32_t i = 0; i < PermuteWriterArgs.rank - 1; ++i) {
            size_t dim = PermuteWriterArgs.rank - 2 - i;  // Start from the second last dimension
            src_multi_idx[dim] = remaining % PermuteWriterArgs.input_shape[dim];
            remaining /= PermuteWriterArgs.input_shape[dim];
        }
        src_multi_idx[PermuteWriterArgs.rank - 1] = 0;  // Row dimension index

        // Apply permutation to get destination multi-dimensional index
        for (uint32_t i = 0; i < PermuteWriterArgs.rank; ++i) {
            dest_multi_idx[i] = src_multi_idx[PermuteWriterArgs.perm[i]];
        }

        // Convert destination multi-dimensional index to linear index
        uint32_t dest_linear_idx = 0;
        for (uint32_t i = 0; i < PermuteWriterArgs.rank - 1; ++i) {
            dest_linear_idx += dest_multi_idx[i] * PermuteWriterArgs.dest_strides[i];
        }
        cb_wait_front(tt::CBIndex::c_0, 1);
        uint32_t l1_read_addr = get_read_ptr(tt::CBIndex::c_0);
        uint64_t dst_noc_addr = get_noc_addr(dest_linear_idx, s0);
        noc_async_write(l1_read_addr, dst_noc_addr, PermuteWriterArgs.output_rm_page_size);
        noc_async_write_barrier();
        cb_pop_front(tt::CBIndex::c_0, 1);
    }
}
