// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

FORCE_INLINE void fill_with_val(uint32_t begin_addr, uint32_t n, uint32_t val) {
    auto* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(begin_addr);
    for (uint32_t i = 0; i < n; ++i) {
        ptr[i] = val;
    }
}

void kernel_main() {
    constexpr uint32_t cb_id_in0 = 0;

    constexpr uint32_t num_padding_rows = get_compile_time_arg_val(3);
    const uint32_t total_num_rows = get_compile_time_arg_val(4);
    const uint32_t ncores = get_compile_time_arg_val(5);
    const uint32_t third_dim = get_compile_time_arg_val(6);
    const uint32_t tile_width = get_compile_time_arg_val(7);

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t unpadded_X_size = get_arg_val<uint32_t>(1);
    const uint32_t pad_value = get_arg_val<uint32_t>(2);
    const uint32_t core_number = get_arg_val<uint32_t>(3);

    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;

#define stick_size_is_pow2 get_compile_time_arg_val(1) == 1
#if (stick_size_is_pow2)
    constexpr uint32_t log_base_2_of_page_size = get_compile_time_arg_val(2);
    const InterleavedPow2AddrGen<src0_is_dram> s = {
        .bank_base_address = src_addr, .log_base_2_of_page_size = log_base_2_of_page_size};
#else
    const InterleavedAddrGen<src0_is_dram> s = {.bank_base_address = src_addr, .page_size = unpadded_X_size};
#endif

    auto read_block = [&](uint32_t num_rows,
                          uint32_t mul,
                          uint32_t size_per_row_per_block,
                          uint32_t start_id,
                          uint32_t width_size,
                          uint32_t size_2d) {
        uint32_t onetile = 1;
        uint32_t padding_rows = num_padding_rows & 31;
        bool has_rows = (num_rows + padding_rows) > 0;

        cb_reserve_back(cb_id_in0, onetile * has_rows);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);

        for (uint32_t k = 0; k < num_rows; k++) {
            uint64_t src_noc_addr = get_noc_addr(size_2d + k, s);

            // Read from DRAM to tmp buffer
            noc_async_read(src_noc_addr + start_id + mul * size_per_row_per_block, l1_write_addr, width_size);

            // pad the row for the last core if needed
            uint32_t prev_size = mul * size_per_row_per_block + start_id;
            uint32_t this_block_size = unpadded_X_size - prev_size;
            if (mul == ncores - 1 && this_block_size < width_size) {
                uint32_t to_pad = width_size - this_block_size;
                fill_with_val(l1_write_addr + this_block_size, (to_pad) >> 2, pad_value);
            } else if (mul == ncores - 1 && prev_size > unpadded_X_size) {
                fill_with_val(l1_write_addr, (width_size) >> 2, pad_value);
            }

            // Block before copying data from tmp to cb buffer
            noc_async_read_barrier();
            l1_write_addr += width_size;

            // Pushing one tile at a time because the current LLK tilize implementation doesn't support tilizing more
            // than one tile per column at the same time.
            // This needs to be fixed in the future
            if (k > 0 && k % tile_width == 0) {
                cb_push_back(cb_id_in0, onetile * has_rows);
                cb_reserve_back(cb_id_in0, onetile * has_rows);
            }
        }

        // pad in the height dim if needed
        fill_with_val(l1_write_addr, padding_rows * (width_size >> 2), pad_value);
        l1_write_addr += padding_rows * width_size;

        cb_push_back(cb_id_in0, onetile * has_rows);
    };

    const uint32_t size_per_row_per_block = get_arg_val<uint32_t>(4);
    const uint32_t blocks_per_core = get_arg_val<uint32_t>(5);
    const uint32_t width_size = get_arg_val<uint32_t>(6);

    uint32_t size_2d = 0;
    for (uint32_t dim3 = 0; dim3 < third_dim; dim3++) {
        uint32_t start_id = 0;
        for (uint32_t b = 0; b < blocks_per_core; b++) {
            read_block(total_num_rows, core_number, size_per_row_per_block, start_id, width_size, size_2d);
            start_id += width_size;
        }
        size_2d += total_num_rows;
    }
}
