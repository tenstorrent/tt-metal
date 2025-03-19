// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
/**
 * add a cb full of indices for the tile
 * each row is identical in the index tensor, so we just need to add an offset based on which row tile it is
 * first 32 elements are {0,..31}, then next 32 are {32,..64}
 * wt is which tile it is along the row [0, Wt) so j + 32*wt is the value in the tile at each element
 */
FORCE_INLINE void generate_index_tile(const uint32_t cb_id, const uint32_t wt) {
    // TODO: investigate moving to compile time (binary size is at risk)
    cb_reserve_back(cb_id, 1);
    uint32_t write_addr = get_write_ptr(cb_id);
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_addr);
    uint16_t wt_offset = wt << 5;

    uint32_t count = 0;
    for (uint32_t i = 0; i < 2; ++i) {
        for (uint32_t j = 0; j < 2; ++j) {
            for (uint32_t k = 0; k < 16; ++k) {
                for (uint32_t l = 0; l < 16; l += 2) {
                    uint16_t value = l + 16 * j + wt_offset;
                    ptr[count] = (value + 1) << 16 | value;
                    count++;
                }
            }
        }
    }
    cb_push_back(cb_id, 1);
}

void kernel_main() {
    uint32_t values_addr = get_arg_val<uint32_t>(0);
    uint32_t indices_addr = get_arg_val<uint32_t>(1);

    constexpr uint32_t input_values_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t input_indices_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t cb_intermed_index = get_compile_time_arg_val(2);

    constexpr bool input_values_is_dram = get_compile_time_arg_val(3);
    constexpr bool input_indices_is_dram = get_compile_time_arg_val(4);
    constexpr uint32_t Ht = get_compile_time_arg_val(5);
    constexpr uint32_t Wt = get_compile_time_arg_val(6);
    constexpr uint32_t input_indices_page_size = get_compile_time_arg_val(7);

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    constexpr uint32_t TILE_HEIGHT = 32;
    constexpr uint32_t tile_bytes_input_values = get_tile_size(input_values_cb_index);
    constexpr DataFormat data_format_input_values = get_dataformat(input_values_cb_index);

    const InterleavedAddrGenFast<input_values_is_dram> s0 = {
        .bank_base_address = values_addr,
        .page_size = tile_bytes_input_values,
        .data_format = data_format_input_values};

    const InterleavedAddrGen<input_indices_is_dram> s1 = {
        .bank_base_address = indices_addr, .page_size = input_indices_page_size};

    uint32_t tile_id_input_values = 0;
    uint32_t tile_id_input_indices = 0;
    for (uint32_t i = 0; i < Ht; ++i) {
        // input values TILE
        for (uint32_t j = 0; j < Wt; ++j) {
            cb_reserve_back(input_values_cb_index, onetile);
            uint32_t l1_write_addr_values = get_write_ptr(input_values_cb_index);
            noc_async_read_tile(tile_id_input_values, s0, l1_write_addr_values);
            l1_write_addr_values += tile_bytes_input_values;
            tile_id_input_values++;
            generate_index_tile(cb_intermed_index, j);
            noc_async_read_barrier();
            cb_push_back(input_values_cb_index, onetile);
        }
    }

    // input indices RM
    for (uint32_t j = 0; j < Ht * TILE_HEIGHT; ++j) {
        cb_reserve_back(input_indices_cb_index, onetile);
        uint32_t l1_write_addr_indices = get_write_ptr(input_indices_cb_index);
        uint64_t input_noc_addr = get_noc_addr(j, s1);
        noc_async_read(input_noc_addr, l1_write_addr_indices, input_indices_page_size);
        noc_async_read_barrier();
        cb_push_back(input_indices_cb_index, onetile);
    }
}
