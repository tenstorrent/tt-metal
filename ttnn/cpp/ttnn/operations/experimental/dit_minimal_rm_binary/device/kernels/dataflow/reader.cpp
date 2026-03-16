// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Reads 1024-element chunks from two row-major DRAM tensors into double-buffered CBs.
//
// The tensor is treated as a flat 1-D array divided into STICK_SIZE-element
// chunks (one hardware tile = 32x32 = 1024 elements).  Because padded_shape[-1]
// is not necessarily a divisor of 1024, a chunk may span multiple DRAM rows.
// The inner read loop advances a (row, in_row_byte_offset) cursor and issues one
// noc_async_read per row fragment, exactly as in the built-in clone read_kernel_rm.
//
// CT args: [STICK_SIZE_BYTES, TensorAccessorArgs_A..., TensorAccessorArgs_B...]
// RT args: [src_a_addr, src_b_addr, num_full_sticks, start_row,
//           start_in_row_offset_bytes, last_chunk_bytes, row_size_bytes]

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// Read chunk_bytes bytes into the current CB write pointer, advancing the
// (cur_row, cur_offset) cursor across DRAM pages of size row_size_bytes.
template <typename Accessor>
FORCE_INLINE void read_chunk(
    const Accessor& acc,
    uint32_t& cur_row,
    uint32_t& cur_offset,
    uint32_t cb_id,
    uint32_t chunk_bytes,
    uint32_t row_size_bytes) {
    uint32_t l1_write_addr = get_write_ptr(cb_id);

    uint32_t cb_write_offset = 0;
    uint32_t remaining = chunk_bytes;

    while (remaining > 0) {
        uint32_t avail = row_size_bytes - cur_offset;
        uint32_t n = avail < remaining ? avail : remaining;
        DPRINT << "Reading " << n << " bytes from (" << cur_row << ", " << cur_offset << ") " << ENDL();
        noc_async_read(acc.get_noc_addr(cur_row, cur_offset), l1_write_addr, n);
        remaining -= n;
        cur_offset += n;
        if (cur_offset == row_size_bytes) {
            cur_row++;
            cur_offset = 0;
        }
        l1_write_addr += n;
    }
    DPRINT << "pushed " << cb_write_offset << " bytes to cb" << ENDL();
}

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
        // IDEA: read page if enough; read bytes otherwise

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
    uint32_t cur_row_a = get_arg_val<uint32_t>(3);
    uint32_t cur_offset_a = get_arg_val<uint32_t>(4);
    const uint32_t last_chunk_bytes = get_arg_val<uint32_t>(5);
    const uint32_t bytes_per_row = get_arg_val<uint32_t>(6);

    // b cursor mirrors a (same shape, same starting position).
    uint32_t cur_row_b = cur_row_a;
    uint32_t cur_offset_b = cur_offset_a;

    constexpr auto cb_a = tt::CBIndex::c_0;
    constexpr auto cb_b = tt::CBIndex::c_1;

    // CT arg 0 = STICK_SIZE_BYTES; a_args starts at CT index 1.
    // next_compile_time_args_offset() returns CTA_OFFSET + num_compile_time_args(),
    // i.e. the absolute CT index of the first arg AFTER a_args — so b_args uses
    // that value directly, with no extra +1.
    constexpr auto a_args = TensorAccessorArgs<1>();
    const auto a = TensorAccessor(a_args, src_a_addr, bytes_per_row);

    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, src_b_addr, bytes_per_row);

    // Full 1024-element sticks.
    for (uint32_t i = 0; i < num_full_sticks; ++i) {
        cb_reserve_back(cb_a, 1);
        cb_reserve_back(cb_b, 1);

        read_chunks(a, b, cur_row_a, cur_offset_a, cb_a, cb_b, STICK_SIZE_BYTES, bytes_per_row);

        noc_async_read_barrier();

        cb_push_back(cb_a, 1);
        cb_push_back(cb_b, 1);
    }

    // Partial last chunk (< 1024 elements); only present when
    // total_elements % STICK_SIZE != 0.
    if (last_chunk_bytes > 0) {
        DPRINT << "Reading last chunk of " << last_chunk_bytes << " bytes" << ENDL();

        cb_reserve_back(cb_a, 1);
        cb_reserve_back(cb_b, 1);

        read_chunk(a, cur_row_a, cur_offset_a, cb_a, last_chunk_bytes, bytes_per_row);
        read_chunk(b, cur_row_b, cur_offset_b, cb_b, last_chunk_bytes, row_size_bytes);

        noc_async_read_barrier();

        cb_push_back(cb_a, 1);
        cb_push_back(cb_b, 1);
    }
}
