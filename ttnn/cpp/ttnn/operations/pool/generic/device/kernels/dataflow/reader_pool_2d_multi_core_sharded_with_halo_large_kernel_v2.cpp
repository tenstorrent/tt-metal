// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <sys/types.h>

#include <cstdint>
#include <cstring>

#include "dataflow_api.h"

#define ENABLE_DEBUG_PRINT 0

#if ENABLE_DEBUG_PRINT == 1
#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#endif

#define ALWI inline __attribute__((always_inline))

// Fill an L1 buffer with the given val
// WARNING: Use with caution as there's no memory protection. Make sure size is within limits
ALWI bool fill_with_val(uint32_t begin_addr, uint32_t n, uint16_t val) {
    // simplest impl:
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(begin_addr);
    for (uint32_t i = 0; i < n / 2; ++i) {
        ptr[i] = (val | (val << 16));
    }
    return true;
}

/**
 * Pool 2D (Max pool 2D and Avg pool 2D)
 */
void kernel_main() {
    const uint32_t reader_nindices = get_compile_time_arg_val(0);
    const uint32_t window_h = get_compile_time_arg_val(1);
    const uint32_t window_w = get_compile_time_arg_val(2);

    const int32_t pad_w = get_compile_time_arg_val(3);

    // channel size in bytes
    const uint32_t in_nbytes_c = get_compile_time_arg_val(4);

    // input tensor height / width / channels
    const int32_t in_w = get_compile_time_arg_val(5);
    const uint32_t in_cb_nsticks = get_compile_time_arg_val(6);

    const uint32_t in_c = get_compile_time_arg_val(7);

    const uint32_t split_reader = get_compile_time_arg_val(9);
    const uint32_t reader_id = get_compile_time_arg_val(10);

    // compile time args
    // BF16 value packed in UINT32. For maxpool, value is 1.
    constexpr uint32_t bf16_scalar = get_compile_time_arg_val(11);
    constexpr uint32_t bf16_one_u32 = get_compile_time_arg_val(12);
    constexpr uint32_t bf16_init_value = get_compile_time_arg_val(13);

    constexpr uint32_t in_nblocks_c = get_compile_time_arg_val(14);
    constexpr uint32_t in_cb_sz = get_compile_time_arg_val(15);
    constexpr uint32_t max_rows_for_reduction = get_compile_time_arg_val(16);
    constexpr uint32_t ceil_pad_w = get_compile_time_arg_val(17);

    constexpr uint32_t TILE_WIDTH = 32;
    constexpr uint32_t TILE_SIZE = 32 * 32;
    constexpr uint32_t MAX_TILES_PER_REDUCTION = 8;
    constexpr uint32_t MAX_ELE_PER_REDUCTION = 512;  // TILE_WIDTH * 8 * numbytes

    constexpr uint32_t in_cb_id = (reader_id == 1) ? get_compile_time_arg_val(19) : get_compile_time_arg_val(18);
    constexpr uint32_t in_shard_cb_id = get_compile_time_arg_val(20);
    constexpr uint32_t in_reader_indices_cb_id = get_compile_time_arg_val(21);
    constexpr uint32_t in_scalar_cb_id = get_compile_time_arg_val(22);
    constexpr uint32_t interm_reduction_cb_id = get_compile_time_arg_val(23);
    constexpr uint32_t in_one_cb_id = get_compile_time_arg_val(24);

    // Reduce scalar = 1
    if (reader_id == 0) {
        cb_reserve_back(in_scalar_cb_id, 1);

        uint32_t bf16_one_u16 = bf16_one_u32 >> 16;
        // fill interm buffer with init_value
        fill_with_val(get_write_ptr(interm_reduction_cb_id), in_cb_sz, bf16_init_value);
        fill_with_val(get_write_ptr(in_scalar_cb_id), TILE_WIDTH, bf16_scalar >> 16);
        fill_with_val(get_write_ptr(in_one_cb_id), TILE_WIDTH, bf16_one_u16);
        cb_push_back(in_scalar_cb_id, 1);
    }

    uint32_t in_l1_read_base_addr = get_read_ptr(in_shard_cb_id);
    uint32_t reader_indices_l1_addr = get_read_ptr(in_reader_indices_cb_id);
    volatile tt_l1_ptr uint16_t* reader_indices_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(reader_indices_l1_addr);

    uint32_t in_w_padded = in_w + pad_w + ceil_pad_w;

    uint32_t counter = reader_id;
    uint32_t total_elems_to_reduce = window_h * window_w;
    uint32_t remaining_elems = total_elems_to_reduce % max_rows_for_reduction;
    bool wide_reduction = in_nblocks_c > 1;
    uint32_t read_bytes =
        wide_reduction ? MAX_ELE_PER_REDUCTION : in_nbytes_c;  // in_cb is MAX_ELE_PER_REDUCTION for wide reductions
    while (counter < reader_nindices) {
        for (uint32_t c_i = 0; c_i < in_nblocks_c; c_i++) {
            uint16_t top_left_local_index = reader_indices_ptr[counter];
            uint32_t processed_rows = 0;
            cb_reserve_back(in_cb_id, 1);
            uint32_t out_l1_write_addr_base = get_write_ptr(in_cb_id);
            uint32_t out_l1_write_addr = out_l1_write_addr_base;

            // Fill interim buffer with init_value if we have only one chunk.
            if ((total_elems_to_reduce - processed_rows) < max_rows_for_reduction) {
                fill_with_val(out_l1_write_addr, in_cb_sz, bf16_init_value);
            }

            for (uint32_t h = 0; h < window_h; ++h) {
                for (uint32_t w = 0; w < window_w; w++) {
                    uint32_t stick_offset = top_left_local_index + w + h * in_w_padded;
                    uint32_t read_offset =
                        in_l1_read_base_addr + (stick_offset * in_nbytes_c + c_i * MAX_ELE_PER_REDUCTION);
                    noc_async_read_one_packet(get_noc_addr(read_offset), out_l1_write_addr, read_bytes);
                    out_l1_write_addr += read_bytes;
                    processed_rows++;
                    if ((processed_rows % max_rows_for_reduction) == 0) {
                        noc_async_read_barrier();
                        cb_push_back(in_cb_id, 1);
                        cb_reserve_back(in_cb_id, 1);
                        out_l1_write_addr_base = get_write_ptr(in_cb_id);
                        out_l1_write_addr = out_l1_write_addr_base;
                        // If next is last chunk, fill whole buffer with the init_value.
                        if ((total_elems_to_reduce - processed_rows) < max_rows_for_reduction) {
                            fill_with_val(out_l1_write_addr, in_cb_sz, bf16_init_value);
                        }
                    }
                }
            }
            if (remaining_elems) {
                noc_async_read_barrier();
                cb_push_back(in_cb_id, 1);
            }
        }
        counter++;
        if (split_reader) {
            counter++;  // interleave the indices
        }
    }
}  // kernel_main()
