// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <sys/types.h>

#include <cstdint>
#include "dataflow_api.h"

#define ENABLE_DEBUG_PRINT 0

#if ENABLE_DEBUG_PRINT == 1
#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#endif

#define ALWI inline __attribute__((always_inline))

#define TILE_HEIGHT 32
#define TILE_WIDTH 32
#define FACE_WIDTH 16
#define FACE_HEIGHT 16

// Fill an L1 buffer with the given val
// WARNING: Use with caution as there's no memory protection. Make sure size is within limits
ALWI bool fill_with_val(uint32_t begin_addr, uint32_t n, uint16_t val, bool unconditionally = true) {
    // simplest impl:
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(begin_addr);
    uint32_t value = val | (val << 16);
    if (ptr[0] != value || unconditionally) {
        for (uint32_t i = 0; i < n / 2; ++i) {
            ptr[i] = (value);
        }
    }

    return true;
}

template <uint32_t cb_id, uint32_t clear_value_cb_id>
ALWI void clear_out_tiles() {
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
ALWI void clear_out_tiles(uint64_t write_addr, uint64_t clear_value_addr) {
    constexpr uint32_t tile_size = get_tile_size(clear_value_cb_id);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        noc_async_read(clear_value_addr, write_addr, tile_size);
        write_addr += tile_size;
    }
    noc_async_read_barrier();
}

// Fill an L1 buffer with the given val
// WARNING: Use with caution as there's no memory protection. Make sure size is within limits
template <
    uint32_t in_nblocks_c,
    uint32_t in_cb_id,
    uint32_t window_h,
    uint32_t window_w,
    uint32_t in_w_padded,
    uint32_t in_nbytes_leftover,  // in_aligned_nbytes_c
    uint32_t in_c,
    uint32_t max_sticks_for_reduction,
    uint32_t total_elems_to_reduce,
    bool is_avg_pool,
    bool wide_reduction,
    uint32_t clear_value_cb_id,
    uint32_t in_cb_ntiles,
    uint32_t in_nbytes_c,
    bool is_large_kernel,
    bool last_tile_is_partial>
ALWI void read_window_with_top_left_index(uint32_t ind, uint32_t in_l1_read_base_addr) {
    constexpr uint32_t BYTES_PER_ELEM = 2;
    // average pool with large kernels requires fp32 accumulation so we can only reduce 4 tiles at a time,
    // otherwise we can reduce 8 tiles at a time.
    constexpr uint32_t MAX_TILES_PER_REDUCTION = (is_avg_pool && is_large_kernel) ? 4 : 8;
    constexpr uint32_t MAX_BYTES_PER_REDUCTION = MAX_TILES_PER_REDUCTION * TILE_WIDTH * BYTES_PER_ELEM;
    constexpr uint32_t in_ntiles_c = in_c / TILE_WIDTH;
    constexpr bool tilize_reconfig = in_nblocks_c > 1 && in_ntiles_c % MAX_TILES_PER_REDUCTION != 0 &&
                                     (window_h * window_w) <= 16 && !last_tile_is_partial;
    constexpr uint32_t max_write_inc = wide_reduction ? MAX_BYTES_PER_REDUCTION : in_nbytes_leftover;

    uint32_t in_l1_write_addr_base = get_write_ptr(in_cb_id);

    for (uint32_t c_i = 0; c_i < in_nblocks_c; c_i++) {
        uint32_t read_bytes = in_nbytes_c;
        if constexpr (wide_reduction) {
            read_bytes =
                (c_i == in_nblocks_c - 1) ? in_nbytes_c - c_i * MAX_BYTES_PER_REDUCTION : MAX_BYTES_PER_REDUCTION;
        }
        uint32_t in_l1_write_addr = get_write_ptr(in_cb_id);
        uint32_t processed_sticks = 0;
        cb_reserve_back(in_cb_id, 1);
        for (uint32_t h = 0; h < window_h; ++h) {
            auto process_h = [&](uint32_t w_offset, uint32_t w_multiple) __attribute__((always_inline)) {
                const uint32_t stick_offset = ind + w_offset + h * in_w_padded;
                const uint32_t read_offset =
                    in_l1_read_base_addr + (stick_offset * in_nbytes_c + c_i * MAX_BYTES_PER_REDUCTION);
                noc_async_read_one_packet(get_noc_addr(read_offset), in_l1_write_addr, read_bytes * w_multiple);
                // if compute is using tilize_reconfig we will only untilize the needed number of tiles rather
                // than the entire MAX_TILES_PER_REDUCTION, thus we use a different offset for the write address
                if constexpr (tilize_reconfig) {
                    in_l1_write_addr += read_bytes * w_multiple;
                } else {
                    in_l1_write_addr += max_write_inc * w_multiple;
                }
                processed_sticks += w_multiple;
                if constexpr (is_large_kernel) {
                    if ((processed_sticks % max_sticks_for_reduction) == 0 ||
                        processed_sticks == total_elems_to_reduce) {
                        noc_async_read_barrier();
                        cb_push_back(in_cb_id, 1);
                        cb_reserve_back(in_cb_id, 1);
                        in_l1_write_addr = get_write_ptr(in_cb_id);
                        // If next is last chunk, fill whole buffer with the init_value. note for max pool we do
                        // not need to fill the CB for the partial chunk since as long as we have N>1 chunks we
                        // are guaranteed that the junk data remaining from chunk N-1 will fill the entire CB and
                        // cannot contain values greater than the max value, and if we have N=1 chunks we already
                        // initialized the entire CB with the init value, but for avg pool we need to fill the
                        // entire CB with the init value since the junk data will contribute to the average.
                        if constexpr (is_avg_pool) {
                            // clear the in CB
                            if ((total_elems_to_reduce - processed_sticks) < max_sticks_for_reduction &&
                                processed_sticks != total_elems_to_reduce) {
                                clear_out_tiles<clear_value_cb_id, in_cb_ntiles>(
                                    get_noc_addr(in_l1_write_addr), get_noc_addr(get_read_ptr(clear_value_cb_id)));
                            }
                        }
                    }
                }
            };

            // Case where in_nbytes_leftover and in_nbytes_c is different is when we are dealing with
            // tesnors that have last tile as partial. Cb page size is multiple of tile but when the last
            // tile is partial we have to read the smaller stick width. Therefore we need to write out the next stick
            // right bellow the previous one and this is when increment of the write pointer and the read stick size is
            // not compliant.
            bool use_contiguous_read = !wide_reduction && in_nbytes_leftover == in_nbytes_c;
            if constexpr (is_large_kernel) {
                bool whole_row_remaining =
                    window_w <= max_sticks_for_reduction - (processed_sticks % max_sticks_for_reduction);
                use_contiguous_read &= whole_row_remaining;
            }

            if (use_contiguous_read) {  // read entire row as one chunk
                process_h(0, window_w);
            } else {  // read rows stick by stick
                for (uint32_t w = 0; w < window_w; ++w) {
                    process_h(w, 1);
                }
            }
        }
        if constexpr (!is_large_kernel) {
            noc_async_read_barrier();
            cb_push_back(in_cb_id, 1);
        }
    }
}

template <bool one_scalar_per_core, uint32_t in_scalar_cb_id, uint32_t reader_nindices, bool split_reader>
ALWI void fill_scalar(
    uint32_t& scalar_start,
    uint32_t& scalar_end,
    uint32_t& scalar_value,
    uint32_t& scalar_index,
    uint32_t& counter,
    volatile uint16_t* config_ptr) {
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
    if (counter < scalar_end && (counter == scalar_start || counter == scalar_start + 1 ||
                                 (split_reader && (counter == scalar_start + 2 || counter == scalar_start + 3)))) {
        fill_with_val(get_write_ptr(in_scalar_cb_id), TILE_WIDTH, scalar_value, false);
    }
    cb_push_back(in_scalar_cb_id, 1);
    counter++;
    if constexpr (split_reader) {
        counter++;
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

    // channel size in bytes
    constexpr uint32_t in_aligned_nbytes_c = get_compile_time_arg_val(4);

    // input tensor height / width / channels
    constexpr int32_t in_w = get_compile_time_arg_val(5);

    constexpr uint32_t in_c = get_compile_time_arg_val(6);

    constexpr uint32_t split_reader = get_compile_time_arg_val(7);
    constexpr uint32_t reader_id = get_compile_time_arg_val(8);

    constexpr uint32_t bf16_scalar = get_compile_time_arg_val(9);
    constexpr uint32_t bf16_init_value = get_compile_time_arg_val(10);

    constexpr uint32_t in_nblocks_c = get_compile_time_arg_val(11);
    constexpr uint32_t in_cb_sz = get_compile_time_arg_val(12);
    constexpr uint32_t max_sticks_for_reduction = get_compile_time_arg_val(13);
    constexpr uint32_t ceil_pad_w = get_compile_time_arg_val(14);

    constexpr uint32_t in_cb_id = (reader_id == 1) ? get_compile_time_arg_val(16) : get_compile_time_arg_val(15);
    constexpr uint32_t in_shard_cb_id = get_compile_time_arg_val(17);
    constexpr uint32_t in_reader_indices_cb_id = get_compile_time_arg_val(18);
    constexpr uint32_t in_scalar_cb_id_0 = get_compile_time_arg_val(19);
    constexpr uint32_t in_scalar_cb_id_1 = get_compile_time_arg_val(20);
    constexpr uint32_t clear_value_cb_id = get_compile_time_arg_val(21);
    constexpr bool is_avg_pool = (bool)get_compile_time_arg_val(22);
    constexpr bool one_scalar_per_core = get_compile_time_arg_val(23);
    constexpr uint32_t config_cb_id = get_compile_time_arg_val(24);
    constexpr uint32_t in_nbytes_c = get_compile_time_arg_val(25);
    constexpr uint32_t in_nbytes_padded_c = get_compile_time_arg_val(26);
    constexpr uint32_t multi_buffering_factor = get_compile_time_arg_val(27);
    constexpr uint32_t stride_w = get_compile_time_arg_val(28);
    constexpr bool last_tile_is_partial = in_c % TILE_WIDTH != 0 && in_c % TILE_WIDTH <= FACE_WIDTH;

    if constexpr (last_tile_is_partial) {
        clear_out_tiles<in_cb_id, clear_value_cb_id>();
    }

    constexpr uint32_t in_scalar_cb_id =
        split_reader && reader_id == 1 && !one_scalar_per_core ? in_scalar_cb_id_1 : in_scalar_cb_id_0;

    uint32_t scalar_index = 0;
    uint32_t scalar_start = 0;
    uint32_t scalar_end = 1;
    uint32_t scalar_value = 0;

    constexpr uint32_t window_size_hw = window_h * window_w;
    constexpr uint32_t face_r_dim = window_size_hw < FACE_HEIGHT ? window_size_hw : FACE_HEIGHT;
    constexpr uint32_t num_faces_in_input_tile =
        (max_sticks_for_reduction < TILE_WIDTH || window_size_hw <= FACE_HEIGHT) ? 2 : 4;
    constexpr bool is_large_kernel = (window_h * window_w) > max_sticks_for_reduction;
    constexpr uint32_t remaining_elems = window_size_hw % max_sticks_for_reduction;
    constexpr uint32_t interm_reduction_chunks =
        remaining_elems ? window_size_hw / max_sticks_for_reduction + 1 : window_size_hw / max_sticks_for_reduction;
    // we only need to initialize the in_cb if we will not fill each reduction chunk with valid data
    constexpr bool need_to_initialize_in_cb = remaining_elems && face_r_dim == FACE_HEIGHT &&
                                              (num_faces_in_input_tile == 4 || last_tile_is_partial) &&
                                              interm_reduction_chunks <= multi_buffering_factor;
    constexpr uint32_t in_cb_ntiles = in_cb_sz / (TILE_WIDTH * TILE_HEIGHT);  // only use the non-multi buffering size

    // fill the clear cb
    if constexpr (is_avg_pool || need_to_initialize_in_cb) {
        if constexpr (reader_id == 0) {
            fill_with_val(get_write_ptr(clear_value_cb_id), TILE_HEIGHT * TILE_WIDTH, bf16_init_value);
            cb_push_back(clear_value_cb_id, 1);
        }
        if constexpr (reader_id == 1) {
            cb_wait_front(clear_value_cb_id, 1);
        }
        // for average pool clear out tiles runs in loop, no need to initialize here
        if constexpr (!is_avg_pool || !is_large_kernel) {
            clear_out_tiles<in_cb_id, clear_value_cb_id>();
        }
    }

    // initialize the scalar CB
    if constexpr (reader_id == 0 && one_scalar_per_core) {
        fill_with_val(get_write_ptr(in_scalar_cb_id_0), TILE_WIDTH, bf16_scalar >> 16);
        cb_push_back(in_scalar_cb_id_0, 1);
    }

    const uint32_t in_l1_read_base_addr = get_read_ptr(in_shard_cb_id);
    uint32_t reader_indices_l1_addr = get_read_ptr(in_reader_indices_cb_id);
    volatile tt_l1_ptr uint32_t* reader_indices_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reader_indices_l1_addr);
    uint32_t config_l1_addr;
    volatile tt_l1_ptr uint16_t* config_ptr;

    constexpr uint32_t in_w_padded = in_w + pad_w + ceil_pad_w;

    uint32_t segments_counter = 1;
    uint32_t counter = reader_id;
    constexpr uint32_t total_elems_to_reduce = window_h * window_w;
    constexpr bool wide_reduction = in_nblocks_c > 1;

    if constexpr (!one_scalar_per_core) {
        config_l1_addr = get_read_ptr(config_cb_id);
        config_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(config_l1_addr);
        scalar_start = config_ptr[3 * scalar_index];
        scalar_value = config_ptr[3 * scalar_index + 1];
        scalar_end = config_ptr[3 * scalar_index + 2];
        scalar_index++;
    }

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
        uint32_t start_end_segment = reader_indices_ptr[segments_counter++];
        uint16_t start = start_end_segment & 0xffff;
        uint16_t end = start_end_segment >> 16;

        if (!first_row_value) {
            start += stride_w;
            first_row_value = true;
        }

        constexpr uint32_t stride_multiple = split_reader ? 2 : 1;
        for (uint16_t ind = start; ind <= end; ind += stride_multiple * stride_w) {
            if constexpr (!one_scalar_per_core) {
                fill_scalar<one_scalar_per_core, in_scalar_cb_id, reader_nindices, split_reader>(
                    scalar_start, scalar_end, scalar_value, scalar_index, counter, config_ptr);
            }
            reader_indices_on_core--;
            read_window_with_top_left_index<
                in_nblocks_c,
                in_cb_id,
                window_h,
                window_w,
                in_w_padded,
                in_aligned_nbytes_c,
                in_c,
                max_sticks_for_reduction,
                total_elems_to_reduce,
                is_avg_pool,
                wide_reduction,
                clear_value_cb_id,
                in_cb_ntiles,
                in_nbytes_padded_c,
                is_large_kernel,
                last_tile_is_partial>(ind, in_l1_read_base_addr);
            if (split_reader && ind == end) {
                first_row_value = false;
            }
        }
    }

    while (reader_indices_on_core--) {
        if constexpr (!one_scalar_per_core) {
            fill_scalar<one_scalar_per_core, in_scalar_cb_id, reader_nindices, split_reader>(
                scalar_start, scalar_end, scalar_value, scalar_index, counter, config_ptr);
        }
        read_window_with_top_left_index<
            in_nblocks_c,
            in_cb_id,
            window_h,
            window_w,
            in_w_padded,
            in_aligned_nbytes_c,
            in_c,
            max_sticks_for_reduction,
            total_elems_to_reduce,
            is_avg_pool,
            wide_reduction,
            clear_value_cb_id,
            in_cb_ntiles,
            in_nbytes_padded_c,
            is_large_kernel,
            last_tile_is_partial>(0, in_l1_read_base_addr);
    }
}  // kernel_main()
