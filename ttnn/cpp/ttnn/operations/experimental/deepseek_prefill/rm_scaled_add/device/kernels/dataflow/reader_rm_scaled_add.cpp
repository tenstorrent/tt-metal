// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// Compile-time arguments
constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);        // A
constexpr uint32_t cb_in1 = get_compile_time_arg_val(1);        // B
constexpr uint32_t cb_scalar = get_compile_time_arg_val(2);     // scalar
constexpr bool src0_is_dram = get_compile_time_arg_val(3) == 1;
constexpr bool src1_is_dram = get_compile_time_arg_val(4) == 1;

void kernel_main() {
    // Runtime arguments
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t n_tiles = get_arg_val<uint32_t>(2);
    uint32_t packed_scalar = get_arg_val<uint32_t>(3);
    uint32_t buffer_page_size = get_arg_val<uint32_t>(4);
    uint32_t num_buffer_pages = get_arg_val<uint32_t>(5);

    // CB tile size (should be 2048 for bfloat16)
    uint32_t tile_size = get_tile_size(cb_in0);

    // Generate scalar tile - only the first value matters for SCALAR broadcast
    cb_reserve_back(cb_scalar, 1);
    volatile tt_l1_ptr uint32_t* scalar_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_scalar));
    scalar_ptr[0] = packed_scalar >> 16;  // Store the bfloat16 scalar value
    cb_push_back(cb_scalar, 1);

    // Set up address generators with the actual buffer page size
    const InterleavedAddrGenFast<src0_is_dram> src0_addr_gen = {
        .bank_base_address = src0_addr,
        .page_size = buffer_page_size,
        .data_format = DataFormat::Float16_b
    };

    const InterleavedAddrGenFast<src1_is_dram> src1_addr_gen = {
        .bank_base_address = src1_addr,
        .page_size = buffer_page_size,
        .data_format = DataFormat::Float16_b
    };

    // Reserve space for all tiles in CBs (read all data first, then compute processes it)
    cb_reserve_back(cb_in0, n_tiles);
    cb_reserve_back(cb_in1, n_tiles);

    uint32_t cb_in0_base_addr = get_write_ptr(cb_in0);
    uint32_t cb_in1_base_addr = get_write_ptr(cb_in1);

    // Read all pages using proper interleaved addressing
    uint32_t l1_write_offset = 0;
    for (uint32_t page_id = 0; page_id < num_buffer_pages; page_id++) {
        // Read page from A
        uint64_t src0_noc_addr = get_noc_addr(page_id, src0_addr_gen);
        noc_async_read(src0_noc_addr, cb_in0_base_addr + l1_write_offset, buffer_page_size);

        // Read page from B
        uint64_t src1_noc_addr = get_noc_addr(page_id, src1_addr_gen);
        noc_async_read(src1_noc_addr, cb_in1_base_addr + l1_write_offset, buffer_page_size);

        l1_write_offset += buffer_page_size;
    }
    noc_async_read_barrier();

    // Push all tiles to CBs
    cb_push_back(cb_in0, n_tiles);
    cb_push_back(cb_in1, n_tiles);
}
