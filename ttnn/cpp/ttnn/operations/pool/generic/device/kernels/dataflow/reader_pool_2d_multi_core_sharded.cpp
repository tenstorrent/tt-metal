// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
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

template <
    bool is_wide_reduction,
    uint32_t in_nblocks_c,
    uint32_t in_cb_id,
    uint32_t npages_to_reserve,
    uint32_t MAX_BYTES_PER_REDUCTION,
    bool full_dest_width,
    uint32_t in_nbytes_leftover,
    uint32_t clear_value_cb_id,
    uint32_t leftover_num_tiles,
    uint32_t window_h,
    uint32_t window_w,
    uint32_t in_w_padded,
    uint32_t in_nbytes_c>
FORCE_INLINE void read_window_with_top_left_index(
    uint64_t clear_value_addr, uint64_t in_l1_read_base_addr, uint64_t ind) {
    if constexpr (is_wide_reduction) {
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
                    const uint32_t stick_offset = ind + w + h * in_w_padded;
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
        uint32_t h_multiples = 0;
        for (uint32_t h = 0; h < window_h; ++h, h_multiples += in_w_padded) {
            const uint32_t stick_offset = ind + h_multiples;
            const uint32_t read_offset = in_l1_read_base_addr + (stick_offset * in_nbytes_c);
            noc_async_read_one_packet(get_noc_addr(read_offset), out_l1_write_addr, in_nbytes_c * window_w);
            out_l1_write_addr += in_nbytes_c * window_w;
        }
        noc_async_read_barrier();
        cb_push_back(in_cb_id, npages_to_reserve);
    }
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
    constexpr uint32_t in_scalar_cb_id = get_compile_time_arg_val(20);
    constexpr uint32_t clear_value_cb_id = get_compile_time_arg_val(23);

    constexpr uint32_t stride_w = get_compile_time_arg_val(24);

    constexpr uint32_t in_nbytes_leftover = (in_c % (TILE_WIDTH * MAX_TILES_PER_REDUCTION)) * BYTES_PER_DATUM;

    if (reader_id == 0) {
        cb_reserve_back(in_scalar_cb_id, 1);
        fill_with_val(get_write_ptr(in_scalar_cb_id), TILE_WIDTH, bf16_scalar >> 16);
        cb_push_back(in_scalar_cb_id, 1);
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
    volatile tt_l1_ptr uint32_t* reader_indices_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reader_indices_l1_addr);

    constexpr uint32_t in_w_padded = in_w + pad_w + ceil_pad_w;
    constexpr uint32_t is_wide_reduction = in_c > MAX_TILES_PER_REDUCTION * TILE_WIDTH;

    constexpr uint32_t npages_to_reserve = 1;
    uint32_t counter = 1;

    uint16_t num_segments = reader_indices_ptr[0] & 0xffff;
    bool first_row_value = reader_id == 0;
    uint32_t reader_indices_on_core = 0;

    if (split_reader) {
        if (reader_id == 0) {
            reader_indices_on_core = (reader_nindices + 1) / 2;
        } else {
            reader_indices_on_core = reader_nindices / 2;
        }
    } else {
        reader_indices_on_core = reader_nindices;
    }

    while (num_segments--) {
        uint32_t start_end_segment = reader_indices_ptr[counter++];
        uint16_t start = start_end_segment & 0xffff;
        uint16_t end = start_end_segment >> 16;

        if (!first_row_value) {
            start += stride_w;
            first_row_value = true;
        }

        for (uint16_t ind = start; ind <= end; ind += 2 * stride_w) {
            reader_indices_on_core--;

            read_window_with_top_left_index<
                is_wide_reduction,
                in_nblocks_c,
                in_cb_id,
                npages_to_reserve,
                MAX_BYTES_PER_REDUCTION,
                full_dest_width,
                in_nbytes_leftover,
                clear_value_cb_id,
                leftover_num_tiles,
                window_h,
                window_w,
                in_w_padded,
                in_nbytes_c>(clear_value_addr, in_l1_read_base_addr, ind);

            if (split_reader && ind == end) {
                first_row_value = false;
            }
        }
    }

    while (reader_indices_on_core--) {
        read_window_with_top_left_index<
            is_wide_reduction,
            in_nblocks_c,
            in_cb_id,
            npages_to_reserve,
            MAX_BYTES_PER_REDUCTION,
            full_dest_width,
            in_nbytes_leftover,
            clear_value_cb_id,
            leftover_num_tiles,
            window_h,
            window_w,
            in_w_padded,
            in_nbytes_c>(clear_value_addr, in_l1_read_base_addr, 0);
    }
}  // kernel_main()
