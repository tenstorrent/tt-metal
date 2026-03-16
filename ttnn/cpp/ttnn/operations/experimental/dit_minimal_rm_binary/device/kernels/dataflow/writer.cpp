// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Writes 1024-element chunks from the output CB back to the DRAM destination.
//
// Mirrors reader.cpp: a (row, in_row_byte_offset) cursor advances across DRAM
// rows so that chunks that span row boundaries are written correctly.
//
// CT args: [STICK_SIZE_BYTES, TensorAccessorArgs_out...]
// RT args: [dst_addr, num_full_sticks, start_row,
//           start_in_row_offset_bytes, last_chunk_bytes, row_size_bytes]

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// Write chunk_bytes bytes from the CB read pointer to DRAM, advancing cursor.
template <typename Accessor>
FORCE_INLINE void write_chunk(
    const Accessor& acc,
    uint32_t& current_row,
    uint32_t& current_offset,
    uint32_t cb_id,
    uint32_t chunk_bytes,
    uint32_t row_size_bytes) {
    uint32_t remaining = chunk_bytes;

    uint32_t l1_read_addr = get_read_ptr(cb_id);

    while (remaining > 0) {
        uint32_t avail = row_size_bytes - current_offset;
        uint32_t n = avail < remaining ? avail : remaining;
        noc_async_write(l1_read_addr, acc.get_noc_addr(current_row, current_offset), n);
        remaining -= n;
        current_offset += n;
        if (current_offset == row_size_bytes) {
            current_row++;
            current_offset = 0;
        }
        l1_read_addr += n;
    }
}

void kernel_main() {
    // CT arg 0 = STICK_SIZE_BYTES; TensorAccessorArgs start at index 1.
    constexpr uint32_t STICK_SIZE_BYTES = get_compile_time_arg_val(0);

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_full_sticks = get_arg_val<uint32_t>(1);
    uint32_t start_row = get_arg_val<uint32_t>(2);
    uint32_t start_offset = get_arg_val<uint32_t>(3);
    const uint32_t last_chunk_bytes = get_arg_val<uint32_t>(4);
    const uint32_t row_size_bytes = get_arg_val<uint32_t>(5);

    constexpr auto cb_out = tt::CBIndex::c_2;

    // TensorAccessorArgs start at CT index 1.
    constexpr auto out_args = TensorAccessorArgs<1>();
    const auto out = TensorAccessor(out_args, dst_addr, row_size_bytes);

    uint32_t current_row = start_row;
    uint32_t current_offset = start_offset;

    // Full 1024-element sticks.
    for (uint32_t i = 0; i < num_full_sticks; ++i) {
        cb_wait_front(cb_out, 1);
        write_chunk(out, current_row, current_offset, cb_out, STICK_SIZE_BYTES, row_size_bytes);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }

    // Partial last chunk.
    if (last_chunk_bytes > 0) {
        cb_wait_front(cb_out, 1);
        write_chunk(out, current_row, current_offset, cb_out, last_chunk_bytes, row_size_bytes);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
