// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <sys/types.h>

#include <cstdint>
#include "dataflow_api.h"
#include "reader_pool2d_sharded_common.hpp"

#define ENABLE_DEBUG_PRINT 0

#if ENABLE_DEBUG_PRINT == 1
#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#endif

/**
 * Pool 2D (Max pool 2D and Avg pool 2D)
 */
void kernel_main() {
    constexpr uint32_t reader_nindices = get_compile_time_arg_val(0);
    constexpr uint32_t window_h = get_compile_time_arg_val(1);
    constexpr uint32_t window_w = get_compile_time_arg_val(2);

    constexpr int32_t pad_w = get_compile_time_arg_val(3);

    // channel size in bytes
    constexpr uint32_t in_nbytes_c = get_compile_time_arg_val(4);

    // input tensor height / width / channels
    constexpr int32_t in_w = get_compile_time_arg_val(5);

    constexpr uint32_t in_c = get_compile_time_arg_val(6);

    constexpr uint32_t split_reader = get_compile_time_arg_val(7);
    constexpr uint32_t reader_id = get_compile_time_arg_val(8);

    constexpr uint32_t bf16_scalar = get_compile_time_arg_val(9);
    constexpr uint32_t bf16_one_u32 = get_compile_time_arg_val(10);
    constexpr uint32_t bf16_init_value = get_compile_time_arg_val(11);

    constexpr uint32_t in_nblocks_c = get_compile_time_arg_val(12);
    constexpr uint32_t in_cb_sz = get_compile_time_arg_val(13);
    constexpr uint32_t max_rows_for_reduction = get_compile_time_arg_val(14);
    constexpr uint32_t ceil_pad_w = get_compile_time_arg_val(15);

    constexpr uint32_t TILE_WIDTH = 32;
    constexpr uint32_t MAX_ELE_PER_REDUCTION = 512;  // TILE_WIDTH * 8 * numbytes

    constexpr uint32_t in_cb_id = (reader_id == 1) ? get_compile_time_arg_val(17) : get_compile_time_arg_val(16);
    constexpr uint32_t in_shard_cb_id = get_compile_time_arg_val(18);
    constexpr uint32_t in_reader_indices_cb_id = get_compile_time_arg_val(19);
    constexpr uint32_t in_scalar_cb_id_0 = get_compile_time_arg_val(20);
    constexpr uint32_t in_scalar_cb_id_1 = get_compile_time_arg_val(21);
    constexpr uint32_t interm_reduction_cb_id = get_compile_time_arg_val(22);
    constexpr uint32_t in_one_cb_id = get_compile_time_arg_val(23);
    constexpr bool one_scalar_per_core = get_compile_time_arg_val(26);
    constexpr uint32_t config_cb_id = get_compile_time_arg_val(27);
    constexpr uint32_t in_scalar_cb_id =
        split_reader && reader_id == 1 && !one_scalar_per_core ? in_scalar_cb_id_1 : in_scalar_cb_id_0;

    uint32_t scalar_index = 0;
    uint32_t scalar_start = 0;
    uint32_t scalar_end = 1;
    uint32_t scalar_value = 0;

    if constexpr (reader_id == 0) {
        constexpr uint32_t bf16_one_u16 = bf16_one_u32 >> 16;
        // fill interm buffer with init_value
        fill_with_val(get_write_ptr(interm_reduction_cb_id), in_cb_sz, bf16_init_value);
        if constexpr (one_scalar_per_core) {
            cb_reserve_back(in_scalar_cb_id_0, 1);
            fill_with_val(get_write_ptr(in_scalar_cb_id_0), TILE_WIDTH, bf16_scalar >> 16);
            cb_push_back(in_scalar_cb_id_0, 1);
        }
        if (bf16_scalar != bf16_one_u32 || !one_scalar_per_core) {
            // Pool operation is not maxpool
            fill_with_val(get_write_ptr(in_one_cb_id), TILE_WIDTH, bf16_one_u16);
        }
    }

    const uint32_t in_l1_read_base_addr = get_read_ptr(in_shard_cb_id);
    uint32_t reader_indices_l1_addr = get_read_ptr(in_reader_indices_cb_id);
    volatile tt_l1_ptr uint16_t* reader_indices_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(reader_indices_l1_addr);
    uint32_t config_l1_addr;
    volatile tt_l1_ptr uint16_t* config_ptr;

    constexpr uint32_t in_w_padded = in_w + pad_w + ceil_pad_w;

    uint32_t counter = reader_id;
    constexpr uint32_t total_elems_to_reduce = window_h * window_w;
    constexpr uint32_t remaining_elems = total_elems_to_reduce % max_rows_for_reduction;
    constexpr bool wide_reduction = in_nblocks_c > 1;
    constexpr uint32_t read_bytes =
        wide_reduction ? MAX_ELE_PER_REDUCTION : in_nbytes_c;  // in_cb is MAX_ELE_PER_REDUCTION for wide reductions

    if constexpr (!one_scalar_per_core) {
        config_l1_addr = get_read_ptr(config_cb_id);
        config_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(config_l1_addr);
        scalar_start = config_ptr[3 * scalar_index];
        scalar_value = config_ptr[3 * scalar_index + 1];
        scalar_end = config_ptr[3 * scalar_index + 2];
        scalar_index++;
    }

    while (counter < reader_nindices) {
        if constexpr (!one_scalar_per_core) {
            cb_reserve_back(in_scalar_cb_id, 1);
            while ((counter >= scalar_end) && scalar_end != reader_nindices) {
                scalar_start = scalar_end;
                scalar_value = config_ptr[3 * scalar_index + 1];
                scalar_end = config_ptr[3 * scalar_index + 2];
                scalar_index++;
            }
            // We want to fill the scalar CB at most only the fisrt 2 times since the number of pages is 2, only for the
            // intervals [x, y) where y >= x + 3 exactly 2 times and when y < x + 3 only once. When split reader is
            // enabled counter takes even or odd values only depennding on the reader id so if the scalar start is even
            // and counter is even it will fullfill the first half of the condition counter == scalar_start || counter
            // == scalar_start + 2. When reader is even and scalar_start is odd or vice versa we will fullfill the
            // second half of the condition counter == scalar_start + 1 || counter == scalar_start + 3.
            if (counter < scalar_end &&
                (counter == scalar_start || counter == scalar_start + 1 ||
                 (split_reader && (counter == scalar_start + 2 || counter == scalar_start + 3)))) {
                fill_with_val(get_write_ptr(in_scalar_cb_id), TILE_WIDTH, scalar_value, false);
            }
            cb_push_back(in_scalar_cb_id, 1);
        }

        for (uint32_t c_i = 0; c_i < in_nblocks_c; c_i++) {
            const uint16_t top_left_local_index = reader_indices_ptr[counter];
            uint32_t processed_rows = 0;
            cb_reserve_back(in_cb_id, 1);
            uint32_t out_l1_write_addr = get_write_ptr(in_cb_id);
            for (uint32_t h = 0; h < window_h; ++h) {
                for (uint32_t w = 0; w < window_w; w++) {
                    const uint32_t stick_offset = top_left_local_index + w + h * in_w_padded;
                    const uint32_t read_offset =
                        in_l1_read_base_addr + (stick_offset * in_nbytes_c + c_i * MAX_ELE_PER_REDUCTION);
                    noc_async_read_one_packet(get_noc_addr(read_offset), out_l1_write_addr, read_bytes);
                    out_l1_write_addr += read_bytes;
                    processed_rows++;
                    if ((processed_rows % max_rows_for_reduction) == 0) {
                        noc_async_read_barrier();
                        cb_push_back(in_cb_id, 1);
                        cb_reserve_back(in_cb_id, 1);
                        out_l1_write_addr = get_write_ptr(in_cb_id);
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
        if constexpr (split_reader) {
            counter++;  // interleave the indices
        }
    }
}  // kernel_main()
