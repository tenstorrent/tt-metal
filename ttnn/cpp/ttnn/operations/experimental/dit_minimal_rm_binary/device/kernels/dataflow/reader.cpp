// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Reads 1024-element chunks from two row-major DRAM tensors into double-buffered CBs.
//
// Work is split on row boundaries: each core is assigned a contiguous range of
// complete rows, so the cursor always starts at byte offset 0 of a row.
// Because DRAM page size equals one full row, cross-page reads within a chunk
// are still handled by the while-loop inside read_chunks when row_size <
// STICK_SIZE_BYTES (multiple rows packed per stick).
//
// CT args: [STICK_SIZE_BYTES, TensorAccessorArgs_A..., TensorAccessorArgs_B...]
// RT args: [src_a_addr, src_b_addr, num_full_sticks, start_row,
//           last_chunk_bytes, row_size_bytes]

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// Read chunk_bytes bytes for both tensors A and B simultaneously into their
// respective CBs, advancing the shared (cur_row, cur_offset) cursor across
// DRAM pages of size row_size_bytes.  Both cur_row and cur_offset are updated
// so that successive calls continue from the right position (needed when
// row_size does not evenly divide STICK_SIZE_BYTES).
template <typename Accessor>
FORCE_INLINE void read_chunks(
    const Accessor& acc_a,
    const Accessor& acc_b,
    uint32_t& cur_row,
    uint32_t& cur_offset,
    uint32_t cb_a,
    uint32_t cb_b,
    uint32_t chunk_bytes,
    uint32_t row_size_bytes) {
    uint32_t l1_write_addr_cba = get_write_ptr(cb_a);
    uint32_t l1_write_addr_cbb = get_write_ptr(cb_b);

    uint32_t remaining = chunk_bytes;

    while (remaining > 0) {
        uint32_t avail = row_size_bytes - cur_offset;
        uint32_t n = avail < remaining ? avail : remaining;

        noc_async_read(acc_a.get_noc_addr(cur_row, cur_offset), l1_write_addr_cba, n);
        noc_async_read(acc_b.get_noc_addr(cur_row, cur_offset), l1_write_addr_cbb, n);

        remaining -= n;
        cur_offset += n;
        if (cur_offset == row_size_bytes) {
            cur_row++;
            cur_offset = 0;
        }
        l1_write_addr_cba += n;
        l1_write_addr_cbb += n;
    }
}

void kernel_main() {
    // CT arg 0 = STICK_SIZE_BYTES; TensorAccessorArgs start at index 1.
    constexpr uint32_t STICK_SIZE_BYTES = get_compile_time_arg_val(0);

    const uint32_t src_a_addr = get_arg_val<uint32_t>(0);
    const uint32_t src_b_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_full_sticks = get_arg_val<uint32_t>(2);
    uint32_t cur_row = get_arg_val<uint32_t>(3);
    const uint32_t last_chunk_bytes = get_arg_val<uint32_t>(4);
    const uint32_t bytes_per_row = get_arg_val<uint32_t>(5);

    // Cursor always starts at offset 0 since work is split on row boundaries.
    uint32_t cur_offset = 0;

    constexpr auto cb_a = tt::CBIndex::c_0;
    constexpr auto cb_b = tt::CBIndex::c_1;

    // CT arg 0 = STICK_SIZE_BYTES; a_args starts at CT index 1.
    constexpr auto a_args = TensorAccessorArgs<1>();
    const auto a = TensorAccessor(a_args, src_a_addr, bytes_per_row);

    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, src_b_addr, bytes_per_row);

    for (uint32_t i = 0; i < num_full_sticks; ++i) {
        cb_reserve_back(cb_a, 1);
        cb_reserve_back(cb_b, 1);

        read_chunks(a, b, cur_row, cur_offset, cb_a, cb_b, STICK_SIZE_BYTES, bytes_per_row);

        noc_async_read_barrier();

        cb_push_back(cb_a, 1);
        cb_push_back(cb_b, 1);
    }

    if (last_chunk_bytes > 0) {
        cb_reserve_back(cb_a, 1);
        cb_reserve_back(cb_b, 1);

        read_chunks(a, b, cur_row, cur_offset, cb_a, cb_b, last_chunk_bytes, bytes_per_row);

        noc_async_read_barrier();

        cb_push_back(cb_a, 1);
        cb_push_back(cb_b, 1);
    }
}
