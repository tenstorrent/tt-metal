// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Reads row-major sticks from two DRAM tensors (A and B) into CBs using the
// tilize split-rows reader pattern.
//
// For each row-block of TILE_HEIGHT rows the reader iterates over ntiles_per_row
// tile-columns, producing one CB push per tile-column per tensor:
//   - Each push: 1 page = TILE_HEIGHT rows × tile_width_bytes laid out row-by-row
// This layout is consumed directly by tilize_block in the compute kernel.
//
// The last block may be partial (num_sticks % TILE_HEIGHT != 0).  In that case
// the reader only issues NOC reads for the real rows; the remainder of the
// tile-sized CB page is left as uninitialized L1.  The writer mirrors this by
// only writing back the real rows to DRAM.
//
// Stick (DRAM page) == one full tensor row == row_size_bytes.
//
// CT args: [stick_size, TensorAccessorArgs_A..., TensorAccessorArgs_B...,
//           ntiles_per_row, tile_width_bytes]
//   stick_size       = row_size_bytes (DRAM page for both A and B)
//   ntiles_per_row   = last_dim / TILE_WIDTH
//   tile_width_bytes = TILE_WIDTH * dtype_bytes
//
// RT args: [src_a_addr, src_b_addr, num_sticks, start_stick_id]
//   num_sticks     = number of rows assigned to this core
//   start_stick_id = first row index for this core

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t TILE_HEIGHT = 32;

void print_data(uint16_t* ptr, uint32_t len) {
    for (uint32_t i = 0; i < len; ++i) {
        DPRINT << BF16(ptr[i]) << " ";
    }
    DPRINT << ENDL();
}

void kernel_main() {
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    constexpr auto a_args = TensorAccessorArgs<1>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();

    constexpr uint32_t ntiles_per_row = get_compile_time_arg_val(b_args.next_compile_time_args_offset());
    constexpr uint32_t tile_width_bytes = get_compile_time_arg_val(b_args.next_compile_time_args_offset() + 1);

    const uint32_t src_a_addr = get_arg_val<uint32_t>(0);
    const uint32_t src_b_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_sticks = get_arg_val<uint32_t>(2);      // number of sticks ~ number of rows
    const uint32_t start_stick_id = get_arg_val<uint32_t>(3);  // first row for this core

    const auto a = TensorAccessor(a_args, src_a_addr, stick_size);
    const auto b = TensorAccessor(b_args, src_b_addr, stick_size);

    constexpr auto cb_a = tt::CBIndex::c_0;
    constexpr auto cb_b = tt::CBIndex::c_1;

    uint32_t end_stick = start_stick_id + num_sticks;

    // DPRINT << "stick size = " << stick_size << ENDL();
    // DPRINT << "ntiles per row = " << ntiles_per_row << ENDL();
    // DPRINT << "tile width bytes = " << tile_width_bytes << ENDL();
    // DPRINT << "num sticks = " << num_sticks << ENDL();
    // DPRINT << "start stick id = " << start_stick_id << ENDL();
    // DPRINT << "end stick = " << end_stick << ENDL();

    // DPRINT << "cb a id = " << static_cast<uint32_t>(cb_a) << ENDL();
    // DPRINT << "cb b id = " << static_cast<uint32_t>(cb_b) << ENDL();

    uint32_t stick_id = start_stick_id;
    for (stick_id = start_stick_id; stick_id < end_stick; stick_id += TILE_HEIGHT) {
        uint32_t nrows = std::min(TILE_HEIGHT, end_stick - stick_id);

        // read_blocks(a, b, cb_a, cb_b, stick_id, ntiles_per_row * tile_width_bytes, nrows, ntiles_per_row);

        // DPRINT << "Reading " << ntiles_per_row << " tiles from CB_A" << ENDL();

        // IDEAS:
        // 1) Switch back to noc_async_read but prefect next address (hide some latency)

        cb_reserve_back(cb_a, ntiles_per_row);
        // cb_reserve_back(cb_b, ntiles_per_row);
        uint32_t l1_a = get_write_ptr(cb_a);
        for (uint32_t k = 0; k < nrows; ++k) {
            noc_async_read_page(stick_id + k, a, l1_a);
            l1_a += stick_size;  // each page/stick is stick_size bytes (= ntiles_per_row * tile_width_bytes)
        }
        noc_async_read_barrier();

        print_data(reinterpret_cast<uint16_t*>(l1_a), 10);

        cb_push_back(cb_a, ntiles_per_row);

        // DPRINT << "Push done" << ENDL();
    }
    // DPRINT << "Reader completed" << ENDL();
}
