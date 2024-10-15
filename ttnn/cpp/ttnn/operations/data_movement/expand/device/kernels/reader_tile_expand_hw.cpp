// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"

inline uint32_t div_up(uint32_t dividend, uint32_t divisor) { return (dividend + divisor - 1) / divisor; }
inline uint32_t row_tile_idx(uint32_t row_idx, uint32_t tile_height) { return row_idx / tile_height; }
inline uint32_t col_tile_idx(uint32_t col_idx, uint32_t tile_width) { return col_idx / tile_width; }

void kernel_main() {
    std::uint32_t mem_buffer_src_addr = get_arg_val<uint32_t>(0);
    std::uint32_t num_rows = get_arg_val<uint32_t>(1);
    std::uint32_t element_per_row = get_arg_val<uint32_t>(2);
    std::uint32_t horz_expand_count = get_arg_val<uint32_t>(3);
    std::uint32_t vert_expand_count = get_arg_val<uint32_t>(4);

    constexpr std::uint32_t scratch_cb_id = get_compile_time_arg_val(0);
    constexpr std::uint32_t io_cb_id = get_compile_time_arg_val(1);
    constexpr std::uint32_t datasize_bytes = get_compile_time_arg_val(2);
    constexpr std::uint32_t src_is_dram = get_compile_time_arg_val(3);
    constexpr std::uint32_t tile_width = get_compile_time_arg_val(4);
    constexpr std::uint32_t tile_height = get_compile_time_arg_val(5);

    auto num_tile_h = div_up(element_per_row, tile_width);
    auto num_tile_v = div_up(num_rows, tile_height);

    auto tile_size = get_tile_size(scratch_cb_id);

    auto mem_src_addr_gen =
        InterleavedAddrGen<src_is_dram>{
            .bank_base_address = mem_buffer_src_addr,
            .page_size = tile_size * datasize_bytes,
        }

    /* We're cheating a little bit here:
    You can only expand a singleton dimension (aka that dimension can only be `1`)
    As such we copy the first element (in case of W) or the first row (in case of H) of a 2D tensor and call it good.
    */

    cb_reserve_back(scratch_cb_id, 1);
    auto tmp_buf = get_write_ptr(scratch_cb_id);

    // Horizontal expansion
    for (uint32_t i = 0; i < num_tile_v; i++) {
        // read the tile into scratch buffer
        auto tile_addr = get_noc_addr(i * num_tile_h, mem_src_addr_gen);
        noc_async_read(tile_addr, tmp_buf, tile_size * datasize_bytes);
        noc_async_read_barrier();

        char values[datasize_bytes];

        for (uint32_t j = 0; j < tile_height; j++) {
            // Copy data into values
            char* ptr = reinterpret_cast<char*>(tmp_buf) + j * tile_width * datasize_bytes;
            for (uint32_t k = 0; k < datasize_bytes; k++) {
                values[k] = ptr[k];
            }

            for (uint32_t k = 0; k < num_tile_h; k++) {
                // read the tile into the io buffer
                cb_reserve_back(io_cb_id, 1);
                auto l1_addr = get_write_ptr(io_cb_id);
                noc_async_read(get_noc_addr(i * num_tile_h + k, mem_src_addr_gen), l1_addr, tile_size * datasize_bytes);
                noc_async_read_barrier();

                for (uint32_t l = horz_expand_count; l > 0; l -= 32) {
                    for (uint32_t m = 0; m < datasize_bytes; m++) {
                        l1_addr[j * tile_width * datasize_bytes + l * datasize_bytes + m] = values[m];
                    }
                }

                noc_async_write(l1_addr, get_noc_addr(i * num_tile_h + k, mem_src_addr_gen),  tile_size * datasize_bytes);
                noc_async_write_barrier();
            }
        }
    }
}
