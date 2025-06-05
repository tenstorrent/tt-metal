// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "reader_pool2d_sharded_common.hpp"

#define ENABLE_DEBUG_PRINT 0

#if ENABLE_DEBUG_PRINT == 1
#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#endif

template <uint32_t cb_id, uint32_t clear_value_cb_id>
FORCE_INLINE void clear_out_tiles() {
    constexpr uint32_t tile_size = get_tile_size(cb_id);
    const uint32_t num_pages = get_local_cb_interface(cb_id).fifo_num_pages;
    const uint32_t num_tiles = get_local_cb_interface(cb_id).fifo_page_size / tile_size;
    const uint64_t clear_value_addr = get_noc_addr(get_read_ptr(clear_value_cb_id));
    uint64_t write_addr = get_noc_addr(get_write_ptr(cb_id));

    for (uint32_t i = 0; i < num_tiles * num_pages; ++i) {
        noc_async_read(clear_value_addr, write_addr, tile_size);
        write_addr += tile_size;
    }
    noc_async_read_barrier();
}

template <uint32_t clear_value_cb_id, uint32_t num_tiles>
FORCE_INLINE void clear_out_tiles(uint64_t write_addr, uint64_t clear_value_addr) {
    constexpr uint32_t tile_size = get_tile_size(clear_value_cb_id);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        noc_async_read(clear_value_addr, write_addr, tile_size);
        write_addr += tile_size;
    }
    noc_async_write_barrier();
}

/**
 * Pool 2D (Max pool 2D and Avg pool 2D)
 */
void kernel_main() {
    constexpr uint32_t reader_nindices = get_compile_time_arg_val(0);
    constexpr uint32_t window_h = get_compile_time_arg_val(1);
    constexpr uint32_t window_w = get_compile_time_arg_val(2);

    constexpr int32_t pad_w = get_compile_time_arg_val(3);

    // channel size in bytes, multiple of 32
    constexpr uint32_t in_nbytes_c = get_compile_time_arg_val(4);

    // input tensor height / width / channels
    constexpr int32_t in_w = get_compile_time_arg_val(5);

    constexpr uint32_t in_c = get_compile_time_arg_val(6);

    constexpr uint32_t split_reader = get_compile_time_arg_val(7);
    constexpr uint32_t reader_id = get_compile_time_arg_val(8);

    // compile time args
    // BF16 value packed in UINT32. For maxpool, value is 1, for avgpool value is 1/kernel_size.
    constexpr uint32_t bf16_scalar = get_compile_time_arg_val(9);
    constexpr uint32_t bf16_init_value = get_compile_time_arg_val(11);

    constexpr uint32_t in_nblocks_c = get_compile_time_arg_val(12);

    constexpr uint32_t ceil_pad_w = get_compile_time_arg_val(15);

    constexpr uint32_t TILE_HEIGHT = 32;
    constexpr uint32_t TILE_WIDTH = 32;
    constexpr uint32_t MAX_TILES_PER_REDUCTION = 8;
    constexpr uint32_t BYTES_PER_DATUM = 2;
    constexpr uint32_t MAX_BYTES_PER_REDUCTION = TILE_WIDTH * MAX_TILES_PER_REDUCTION * BYTES_PER_DATUM;

    constexpr uint32_t in_cb_id = (reader_id == 1) ? get_compile_time_arg_val(17) : get_compile_time_arg_val(16);
    constexpr uint32_t in_shard_cb_id = get_compile_time_arg_val(18);
    constexpr uint32_t in_reader_indices_cb_id = get_compile_time_arg_val(19);
    constexpr uint32_t in_scalar_cb_id_0 = get_compile_time_arg_val(20);
    constexpr uint32_t in_scalar_cb_id_1 = get_compile_time_arg_val(21);
    constexpr uint32_t clear_value_cb_id = get_compile_time_arg_val(24);
    constexpr uint32_t pool_type = (bool)get_compile_time_arg_val(25);
    constexpr bool one_scalar_per_core = get_compile_time_arg_val(26);
    constexpr uint32_t config_cb_id = get_compile_time_arg_val(27);
    constexpr uint32_t in_scalar_cb_id =
        split_reader && reader_id == 1 && !one_scalar_per_core ? in_scalar_cb_id_1 : in_scalar_cb_id_0;

    constexpr uint32_t in_nbytes_leftover = (in_c % (TILE_WIDTH * MAX_TILES_PER_REDUCTION)) * BYTES_PER_DATUM;

    uint32_t scalar_index = 0;
    uint32_t scalar_start = 0;
    uint32_t scalar_end = 1;
    uint32_t scalar_value = 0;

    if constexpr (reader_id == 0 && one_scalar_per_core) {
        cb_reserve_back(in_scalar_cb_id_0, 1);
        fill_with_val(get_write_ptr(in_scalar_cb_id_0), TILE_WIDTH, bf16_scalar >> 16);
        cb_push_back(in_scalar_cb_id_0, 1);
    }

    constexpr uint32_t window_hw = window_h * window_w;
    constexpr bool full_dest_width = in_c % (TILE_WIDTH * MAX_TILES_PER_REDUCTION) == 0;
    constexpr uint32_t leftover_num_tiles = (in_c % (TILE_WIDTH * MAX_TILES_PER_REDUCTION)) / TILE_WIDTH;

    // We only need to clear out temp CB tiles if the window size is larger than 16.
    // If <= 16, than we use only upper to faces of the tile, and we can configure
    // reduce to only process as many rows as needed.
    // In case we need bottom two faces, than we have to configure reduce to process all rows,
    // as number of valid rows in upper two faces will be 16 and in bottom two some different number.
    // In that case not all rows will have valid data, so we need to clear them out.
    if constexpr (window_hw > 16) {
        fill_with_val(get_read_ptr(clear_value_cb_id), TILE_HEIGHT * TILE_WIDTH, bf16_init_value);
        clear_out_tiles<in_cb_id, clear_value_cb_id>();
    }
    const uint64_t clear_value_addr = get_noc_addr(get_read_ptr(clear_value_cb_id));

    const uint32_t in_l1_read_base_addr = get_read_ptr(in_shard_cb_id);
    uint32_t reader_indices_l1_addr = get_read_ptr(in_reader_indices_cb_id);
    volatile tt_l1_ptr uint16_t* reader_indices_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(reader_indices_l1_addr);
    uint32_t config_l1_addr;
    volatile tt_l1_ptr uint16_t* config_ptr;

    constexpr uint32_t in_w_padded = in_w + pad_w + ceil_pad_w;
    constexpr uint32_t is_wide_reduction = in_c > MAX_TILES_PER_REDUCTION * TILE_WIDTH;

    if constexpr (!one_scalar_per_core) {
        config_l1_addr = get_read_ptr(config_cb_id);
        config_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(config_l1_addr);
        scalar_start = config_ptr[3 * scalar_index];
        scalar_value = config_ptr[3 * scalar_index + 1];
        scalar_end = config_ptr[3 * scalar_index + 2];
        scalar_index++;
    }

    constexpr uint32_t npages_to_reserve = 1;
    uint32_t counter = reader_id;
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
        if constexpr (is_wide_reduction) {
            const uint16_t top_left_local_index = reader_indices_ptr[counter++];
            for (uint32_t c_i = 0; c_i < in_nblocks_c; ++c_i) {
                cb_reserve_back(in_cb_id, npages_to_reserve);
                uint32_t out_l1_write_addr = get_write_ptr(in_cb_id);

                uint32_t read_bytes = MAX_BYTES_PER_REDUCTION;
                if constexpr (!full_dest_width) {
                    if (c_i == in_nblocks_c - 1) {
                        read_bytes = in_nbytes_leftover;
                        clear_out_tiles<clear_value_cb_id, leftover_num_tiles>(out_l1_write_addr, clear_value_addr);
                    }
                }

                for (uint32_t h = 0; h < window_h; ++h) {
                    for (uint32_t w = 0; w < window_w; ++w) {
                        const uint32_t stick_offset = top_left_local_index + w + h * in_w_padded;
                        const uint32_t read_offset =
                            in_l1_read_base_addr + (stick_offset * in_nbytes_c + c_i * MAX_BYTES_PER_REDUCTION);
                        noc_async_read_one_packet(get_noc_addr(read_offset), out_l1_write_addr, read_bytes);
                        out_l1_write_addr += read_bytes;
                    }
                }
                noc_async_read_barrier();  // At this line, read is complete.

                cb_push_back(in_cb_id, npages_to_reserve);
            }
        } else {
            cb_reserve_back(in_cb_id, npages_to_reserve);
            uint32_t out_l1_write_addr = get_write_ptr(in_cb_id);
            uint16_t top_left_local_index = reader_indices_ptr[counter++];
            uint32_t h_multiples = 0;
            for (uint32_t h = 0; h < window_h; ++h, h_multiples += in_w_padded) {
                const uint32_t stick_offset = top_left_local_index + h_multiples;
                const uint32_t read_offset = in_l1_read_base_addr + (stick_offset * in_nbytes_c);
                noc_async_read_one_packet(get_noc_addr(read_offset), out_l1_write_addr, in_nbytes_c * window_w);
                out_l1_write_addr += in_nbytes_c * window_w;
            }
            noc_async_read_barrier();
            cb_push_back(in_cb_id, npages_to_reserve);
        }

        if constexpr (split_reader) {
            counter++;  // interleave the indices
        }
    }
}  // kernel_main()
