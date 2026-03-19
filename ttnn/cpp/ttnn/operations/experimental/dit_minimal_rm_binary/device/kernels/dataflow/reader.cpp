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

// Read one tile-column of num_rows rows from 'acc' into 'cb'.
// For each of the num_rows rows in the block, reads tile_width_bytes starting
// at col_byte_offset within that row.  NOC addresses are computed on-the-fly.
// The CB page is always tile-sized (TILE_HEIGHT rows); rows beyond num_rows
// are left as uninitialized L1 (harmless — the writer will not emit them).
template <typename Accessor>
void read_tile_col(
    const Accessor& acc,
    tt::CBIndex cb,
    uint32_t block_start_stick,
    uint32_t col_byte_offset,
    uint32_t tile_width_bytes,
    uint32_t num_rows) {
    cb_reserve_back(cb, 1);
    uint32_t l1 = get_write_ptr(cb);

    // DEBUG: Print address of NoC for k=0
    // DPRINT << "NoC address for k=0 = " << acc.get_noc_addr(block_start_stick) + col_byte_offset << ENDL();
    // DPRINT << "L1 address = " << l1 << ENDL();

    for (uint32_t k = 0; k < num_rows; ++k) {
        noc_async_read(acc.get_noc_addr(block_start_stick + k) + col_byte_offset, l1, tile_width_bytes);
        l1 += tile_width_bytes;
    }
    noc_async_read_barrier();

    // if (dbg_j == 15) {
    //     uint16_t* l1_start = reinterpret_cast<uint16_t*>(get_write_ptr(cb));
    //     print_data(l1_start, tile_width_bytes);  //
    // }

    cb_push_back(cb, 1);
}

template <typename Accessor>
void read_blocks(
    const Accessor& acc_a,
    const Accessor& acc_b,
    tt::CBIndex cb_a,
    tt::CBIndex cb_b,
    uint32_t block_start_stick,
    uint32_t tile_width_bytes,
    uint32_t num_rows,
    uint32_t tiles_per_block) {
    cb_reserve_back(cb_a, tiles_per_block);
    cb_reserve_back(cb_b, tiles_per_block);
    uint32_t l1_a = get_write_ptr(cb_a);
    uint32_t l1_b = get_write_ptr(cb_b);

    // DEBUG: Print address of NoC for k=0
    DPRINT << "block start stick = " << block_start_stick << ENDL();
    DPRINT << "num rows = " << num_rows << ENDL();

    for (uint32_t k = 0; k < num_rows; ++k) {
        // DPRINT << " reading stick " << block_start_stick + k << ENDL();
        // noc_async_read(acc_a.get_noc_addr(block_start_stick + k), l1_a, tile_width_bytes);
        // noc_async_read(acc_b.get_noc_addr(block_start_stick + k), l1_b, tile_width_bytes);

        noc_async_read_page(block_start_stick + k, acc_a, l1_a);
        noc_async_read_page(block_start_stick + k, acc_b, l1_b);

        // if (block_start_stick + k == 33) {

        //     print_data(reinterpret_cast<uint16_t*>(l1_a), 23);
        //     print_data(reinterpret_cast<uint16_t*>(l1_b), 23);
        // }

        l1_a += tile_width_bytes;
        l1_b += tile_width_bytes;
    }
    noc_async_read_barrier();

    // if (dbg_j == 15) {
    //     uint16_t* l1_start = reinterpret_cast<uint16_t*>(get_write_ptr(cb));
    //     print_data(l1_start, tile_width_bytes);  //
    // }

    cb_push_back(cb_a, tiles_per_block);
    cb_push_back(cb_b, tiles_per_block);
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

    const uint32_t num_full_blocks = num_sticks / TILE_HEIGHT;
    const uint32_t tail_rows = num_sticks % TILE_HEIGHT;

    uint32_t end_stick = start_stick_id + num_sticks;

    DPRINT << "tile width bytes = " << tile_width_bytes << ENDL();
    DPRINT << "ntiles/row = " << ntiles_per_row << ENDL();
    DPRINT << "num full blocks = " << num_full_blocks << ENDL();
    DPRINT << "tail rows = " << tail_rows << ENDL();
    DPRINT << "start stick id = " << start_stick_id << ENDL();
    DPRINT << "end stick = " << end_stick << ENDL();

    uint32_t stick_id = start_stick_id;
    for (stick_id = start_stick_id; stick_id < end_stick; stick_id += TILE_HEIGHT) {
        DPRINT << "[main] Pushing " << ntiles_per_row << " tiles to CB_A and " << ntiles_per_row << " tiles to CB_B"
               << ENDL();

        uint32_t nrows = std::min(TILE_HEIGHT, end_stick - stick_id);
        DPRINT << "stick id = " << stick_id << ", nrows = " << nrows << ENDL();

        read_blocks(a, b, cb_a, cb_b, stick_id, ntiles_per_row * tile_width_bytes, nrows, ntiles_per_row);
        DPRINT << "Push done" << ENDL();
    }
    DPRINT << "Reader completed" << ENDL();
}
