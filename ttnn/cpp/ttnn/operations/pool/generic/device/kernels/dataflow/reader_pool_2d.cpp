// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <sys/types.h>

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

#define ENABLE_DEBUG_PRINT 0

#if ENABLE_DEBUG_PRINT == 1
#include "api/debug/dprint.h"
#include "api/debug/dprint_pages.h"
#endif

#define ALWI inline __attribute__((always_inline))

#define TILE_HEIGHT 32
#define TILE_WIDTH 32
#define FACE_WIDTH 16
#define FACE_HEIGHT 16
#define FACE_SIZE (FACE_WIDTH * FACE_HEIGHT)
#define FACES_PER_TILE_WIDTH (TILE_WIDTH / FACE_WIDTH)

// Zero out a single page (where wr ptr points) for a given circular buffer.
template <uint32_t cb_id>
ALWI void zero_out_page() {
    uint32_t page_size = get_local_cb_interface(cb_id).fifo_page_size;
    const uint32_t num_zeros_reads = page_size / MEM_ZEROS_SIZE;
    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    uint32_t write_addr = get_write_ptr(cb_id);

    noc_async_read_one_packet_set_state(zeros_noc_addr, MEM_ZEROS_SIZE);
    for (uint32_t i = 0; i < num_zeros_reads; ++i) {
        noc_async_read_one_packet_with_state<true>(zeros_noc_addr, write_addr);
        write_addr += MEM_ZEROS_SIZE;
    }
    noc_async_read_barrier();
}

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

template <uint32_t config_dram_addr, uint32_t config_page_size, uint32_t tensor_args_index, uint32_t cb_reader_index>
void load_config_tensor_if_in_dram(uint32_t core_index) {
    // TODO: Instead of all cores reading from dram, only the first column reads, and does an MCAST to all the other
    // cores in the row.
    constexpr auto config_tensor_args = TensorAccessorArgs<tensor_args_index>();
    const auto config_accessor = TensorAccessor(config_tensor_args, config_dram_addr, config_page_size);
    uint64_t src_noc_addr = get_noc_addr(core_index, config_accessor);

    noc_async_read(src_noc_addr, get_write_ptr(cb_reader_index), config_page_size);
    noc_async_read_barrier();
    cb_push_back(cb_reader_index, 1);
}

// Initialize indices and increment tiles for return_indices functionality
template <
    uint32_t kernel_h,
    uint32_t kernel_w,
    uint32_t in_w,
    uint32_t in_c,
    uint32_t pad_t,
    uint32_t pad_l,
    uint32_t dilation_h,
    uint32_t dilation_w,
    uint16_t right_inc,
    uint16_t down_left_wrap_inc,
    uint16_t up_left_wrap_inc,
    uint32_t in_idx_cb_id,
    uint32_t right_inc_cb_id,
    uint32_t down_left_wrap_inc_cb_id,
    uint32_t up_left_wrap_inc_cb_id>
ALWI void initialize_return_indices_data() {
    // since kernels can start in padded regions we need to have "indexes" in these regions
    // we choose a paradigm where we padding indexes must satisfy the following conditions:
    //   1. they are 1 less than the index to the right (whether it's padding or not)
    //   2. they are in_w less than the index below (whether it's padding or not)
    // this results in repeat indexes, negative indexes and other such effects, but since
    // padding is never chosen as a max index, validity of the padding index is unimportant
    // we only care that when they are incremented they result in the correct index which
    // this paradigm guarantees
    // Note we also use unsigned integers and allow wrapping for negatives to preserve range
    // and since all negative values correspond to padding indexes which will never be a max

    // Calculate initial index based on padding conditions
    uint16_t init_index = 0;
    constexpr uint32_t eff_kernel_w = (kernel_w - 1) * dilation_w + 1;
    const uint16_t start_row = (uint16_t)get_arg_val<uint32_t>(2);
    const uint16_t start_col = (uint16_t)get_arg_val<uint32_t>(3);

    if (start_row <= pad_t) {
        // top left is in top padding, we increment from the padding index in the top left
        // of the padded tensor
        uint16_t global_top_left_pad_idx = -(uint16_t)pad_l - (uint16_t)pad_t * (uint16_t)in_w;
        init_index = global_top_left_pad_idx + start_col + start_row * in_w;
    } else if (start_col <= pad_l) {
        // top left is in left padding, we increment from the padding index in the leftmost
        // column of the starting row of the padded tensor
        uint16_t leftmost_valid_index = (start_row - (uint16_t)pad_t) * (uint16_t)in_w;
        uint16_t start_row_left_pad_idx = leftmost_valid_index - (uint16_t)pad_l;
        init_index = start_row_left_pad_idx + start_col;
    } else {
        // top left is in valid region, we choose the valid index
        init_index = (start_row - (uint16_t)pad_t) * (uint16_t)in_w + (start_col - (uint16_t)pad_l);
    }

    constexpr uint32_t fill_c = in_c <= TILE_WIDTH ? in_c : TILE_WIDTH;

    // initialize the index CB - TODO for c > 1 we could optimize by storing two values per write
    volatile tt_l1_ptr uint16_t* idx_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(in_idx_cb_id));
    uint16_t kernel_idx = 0;
    for (uint32_t h = 0; h < kernel_h; ++h) {
        for (uint32_t w = 0; w < kernel_w; ++w) {
            uint16_t hw = h * kernel_w + w;
            for (uint32_t c = 0; c < fill_c; ++c) {
                uint16_t index = init_index + kernel_idx;
                idx_ptr[hw * TILE_WIDTH + c] = index;
            }
            kernel_idx += dilation_w;
        }
        kernel_idx += dilation_h * in_w - eff_kernel_w - (dilation_w - 1);
    }

    // initialize the increment CBs
    auto fill_inc = [&](uint32_t cb_id, uint16_t inc) __attribute__((always_inline)) {
        uint32_t inc_32_bit = (uint32_t)inc | ((uint32_t)inc << 16);
        volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id));
        for (uint32_t k = 0; k < kernel_h * kernel_w; ++k) {
            constexpr uint32_t fill_c_32_bit = fill_c % 2 == 0 ? fill_c / 2 : (fill_c / 2) + 1;
            for (uint32_t c = 0; c <= fill_c_32_bit; ++c) {
                ptr[k * TILE_WIDTH / 2 + c] = inc_32_bit;
            }
        }
    };

    cb_reserve_back(right_inc_cb_id, 1);
    fill_inc(right_inc_cb_id, right_inc);
    cb_push_back(right_inc_cb_id, 1);

    cb_reserve_back(down_left_wrap_inc_cb_id, 1);
    fill_inc(down_left_wrap_inc_cb_id, down_left_wrap_inc);
    cb_push_back(down_left_wrap_inc_cb_id, 1);

    cb_reserve_back(up_left_wrap_inc_cb_id, 1);
    fill_inc(up_left_wrap_inc_cb_id, up_left_wrap_inc);
    cb_push_back(up_left_wrap_inc_cb_id, 1);
}

// Fill an L1 buffer with the given val
// WARNING: Use with caution as there's no memory protection. Make sure size is within limits
template <
    uint32_t in_nblocks_c,
    uint32_t in_cb_id,
    uint32_t kernel_h,
    uint32_t kernel_w,
    uint32_t in_w_padded,
    uint32_t in_nbytes_leftover,
    uint32_t in_c,
    uint32_t max_sticks_for_reduction,
    uint32_t total_elems_to_reduce,
    bool is_avg_pool,
    bool wide_reduction,
    uint32_t clear_value_cb_id,
    uint32_t in_cb_ntiles,
    uint32_t in_nbytes_c,
    uint32_t shard_width_bytes,
    bool is_large_kernel,
    bool last_tile_is_partial,
    uint32_t dilation_h,
    uint32_t dilation_w,
    bool return_indices,
    bool zero_pages,
    uint32_t out_cb_id,
    uint32_t out_idx_cb_id,
    uint32_t reader_id,
    uint32_t pack_tmp_cb_id,
    uint32_t pack_idx_tmp_cb_id>
ALWI void read_kernel_with_top_left_index(uint32_t ind, uint32_t in_l1_read_base_addr) {
    constexpr uint32_t BYTES_PER_ELEM = 2;
    // average pool with large kernels requires fp32 accumulation so we can only reduce 4 tiles at a time,
    // return_indices requires 1 tile at a time, otherwise we can reduce 8 tiles at a time.
    constexpr uint32_t MAX_TILES_PER_REDUCTION = return_indices ? 1 : (is_avg_pool && is_large_kernel) ? 4 : 8;
    constexpr uint32_t MAX_BYTES_PER_REDUCTION = MAX_TILES_PER_REDUCTION * TILE_WIDTH * BYTES_PER_ELEM;
    constexpr uint32_t in_ntiles_c = (in_c + TILE_WIDTH - 1) / TILE_WIDTH;
    constexpr bool tilize_reconfig = in_nblocks_c > 1 && in_ntiles_c % MAX_TILES_PER_REDUCTION != 0 &&
                                     (kernel_h * kernel_w) <= 16 && !last_tile_is_partial;
    uint32_t max_write_inc = wide_reduction ? MAX_BYTES_PER_REDUCTION : in_nbytes_leftover;
    if constexpr (return_indices) {
        static_assert(MAX_TILES_PER_REDUCTION == 1, "MAX_TILES_PER_REDUCTION must be 1 for return indices");
        max_write_inc = TILE_WIDTH * BYTES_PER_ELEM;
    }
    for (uint32_t c_i = 0; c_i < in_nblocks_c; c_i++) {
        if constexpr (reader_id == 0 || !return_indices) {
            uint32_t read_bytes = in_nbytes_c;
            if constexpr (wide_reduction) {
                read_bytes =
                    (c_i == in_nblocks_c - 1) ? in_nbytes_c - c_i * MAX_BYTES_PER_REDUCTION : MAX_BYTES_PER_REDUCTION;
            }

            uint32_t in_l1_write_addr = get_write_ptr(in_cb_id);
            cb_reserve_back(in_cb_id, 1);
            uint32_t processed_sticks = 0;
            // page zeroing is only necessary for tiled block output format so that scale is not affected by
            // junk/padding data
            if constexpr (zero_pages) {
                if (c_i == in_nblocks_c - 1 && last_tile_is_partial) {
                    zero_out_page<in_cb_id>();
                }
            }
            for (uint32_t h = 0; h < kernel_h; ++h) {
                auto process_h = [&](uint32_t w_offset, uint32_t w_multiple) __attribute__((always_inline)) {
                    const uint32_t stick_offset = ind + w_offset + h * dilation_h * in_w_padded;
                    const uint32_t read_offset =
                        in_l1_read_base_addr + (stick_offset * shard_width_bytes + c_i * MAX_BYTES_PER_REDUCTION);
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
                // tile is partial we have to read the smaller stick width. Therefore we need to write out the next
                // stick right bellow the previous one and this is when increment of the write pointer and the read
                // stick size is not compliant.
                bool use_contiguous_read = !wide_reduction && in_nbytes_leftover == in_nbytes_c &&
                                           dilation_w == 1;  // read entire row as one chunk (only if no width dilation)
                if constexpr (is_large_kernel) {
                    bool whole_row_remaining =
                        kernel_w <= max_sticks_for_reduction - (processed_sticks % max_sticks_for_reduction);
                    use_contiguous_read &= whole_row_remaining;
                }

                if (use_contiguous_read) {
                    process_h(0, kernel_w);
                } else {  // read rows stick by stick with dilation
                    for (uint32_t w = 0; w < kernel_w; ++w) {
                        process_h(w * dilation_w, 1);
                    }
                }
            }
        }
        if constexpr (!is_large_kernel) {
            if (reader_id == 0 || !return_indices) {
                noc_async_read_barrier();
                cb_push_back(in_cb_id, 1);
            }
            if constexpr (reader_id == 1 && return_indices) {
                constexpr uint32_t num_faces_in_output_tile = 2;
                constexpr uint32_t num_faces_in_last_output_tile =
                    last_tile_is_partial && in_c % TILE_WIDTH <= FACE_WIDTH ? 1 : 2;
                uint32_t output_faces =
                    c_i == in_nblocks_c - 1 ? num_faces_in_last_output_tile : num_faces_in_output_tile;

                cb_wait_front(pack_tmp_cb_id, 1);
                noc_async_read_one_packet(
                    get_noc_addr(get_read_ptr(pack_tmp_cb_id)),
                    get_write_ptr(out_cb_id),
                    output_faces * FACE_WIDTH * BYTES_PER_ELEM);
                cb_wait_front(pack_idx_tmp_cb_id, 1);
                noc_async_read_one_packet(
                    get_noc_addr(get_read_ptr(pack_idx_tmp_cb_id)),
                    get_write_ptr(out_idx_cb_id),
                    output_faces * FACE_WIDTH * BYTES_PER_ELEM);
                noc_async_read_barrier();
                cb_pop_front(pack_tmp_cb_id, 1);
                cb_pop_front(pack_idx_tmp_cb_id, 1);

                cb_push_back(out_cb_id, output_faces);
                if constexpr (return_indices) {
                    cb_push_back(out_idx_cb_id, output_faces);
                }
            }
        }
    }
}

template <
    bool one_scalar_per_core,
    uint32_t in_scalar_cb_id,
    uint32_t reader_nindices,
    bool split_reader,
    uint32_t multi_buffering_factor>
ALWI void fill_scalar(
    uint32_t& scalar_start,
    uint32_t& scalar_end,
    uint32_t& scalar_value,
    uint32_t& scalar_index,
    uint32_t& counter,
    volatile uint16_t* config_ptr) {
    constexpr uint32_t num_readers = split_reader ? 2 : 1;
    cb_reserve_back(in_scalar_cb_id, 1);

    while (counter >= scalar_end && scalar_end < reader_nindices) {
        scalar_index++;
        scalar_start = scalar_end;
        scalar_value = config_ptr[3 * scalar_index + 1];
        scalar_end = config_ptr[3 * scalar_index + 2];
    }

    // We want to fill the scalar CB the fewest times possible, this will be min(scalar_end - scalar_start, num_readers
    // * multi_buffering_factor)
    if (counter < scalar_start + num_readers * multi_buffering_factor) {
        // Fill only the first FACE_WIDTH, since we set reload_srcB = true in unpack_tilizeA_B_block, meaning the values
        // for the remaining faces will be reused from the first one. This is safe here because there’s no difference
        // between the first and second face.
        fill_with_val(get_write_ptr(in_scalar_cb_id), FACE_WIDTH, scalar_value, false);
    }
    counter += num_readers;

    cb_push_back(in_scalar_cb_id, 1);
}

/**
 * Pool 2D (Max pool 2D and Avg pool 2D)
 */
void kernel_main() {
    constexpr uint32_t reader_nindices = get_compile_time_arg_val(0);
    constexpr uint32_t kernel_h = get_compile_time_arg_val(1);
    constexpr uint32_t kernel_w = get_compile_time_arg_val(2);

    constexpr int32_t pad_w = get_compile_time_arg_val(3);

    // channel size in bytes
    constexpr uint32_t in_nbytes_leftover = get_compile_time_arg_val(4);

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
    constexpr uint32_t in_idx_cb_id = get_compile_time_arg_val(21);
    constexpr uint32_t pack_tmp_cb_id = get_compile_time_arg_val(22);
    constexpr uint32_t pack_idx_tmp_cb_id = get_compile_time_arg_val(23);
    constexpr uint32_t right_inc_cb_id = get_compile_time_arg_val(24);
    constexpr uint32_t down_left_wrap_inc_cb_id = get_compile_time_arg_val(25);
    constexpr uint32_t up_left_wrap_inc_cb_id = get_compile_time_arg_val(26);
    constexpr uint32_t clear_value_cb_id = get_compile_time_arg_val(27);
    constexpr bool is_avg_pool = (bool)get_compile_time_arg_val(28);
    constexpr bool one_scalar_per_core = get_compile_time_arg_val(29);
    constexpr uint32_t config_cb_id = get_compile_time_arg_val(30);
    constexpr uint32_t in_nbytes_c = get_compile_time_arg_val(31);
    constexpr uint32_t shard_width_bytes = get_compile_time_arg_val(32);
    constexpr uint32_t multi_buffering_factor = get_compile_time_arg_val(33);
    constexpr uint32_t stride_w = get_compile_time_arg_val(34);
    constexpr uint32_t dilation_h = get_compile_time_arg_val(35);
    constexpr uint32_t dilation_w = get_compile_time_arg_val(36);
    constexpr bool return_indices = (bool)get_compile_time_arg_val(37);
    constexpr uint32_t pad_t = get_compile_time_arg_val(38);
    constexpr uint32_t pad_l = get_compile_time_arg_val(39);
    constexpr uint16_t right_inc = get_compile_time_arg_val(40);
    constexpr uint16_t down_left_wrap_inc = get_compile_time_arg_val(41);
    constexpr uint16_t up_left_wrap_inc = get_compile_time_arg_val(42);
    constexpr bool zero_pages = (bool)get_compile_time_arg_val(43);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(44);
    constexpr uint32_t out_idx_cb_id = get_compile_time_arg_val(45);
    constexpr uint32_t config_in_dram = get_compile_time_arg_val(46);
    constexpr uint32_t config_dram_addr = get_compile_time_arg_val(47);
    constexpr uint32_t config_page_size = get_compile_time_arg_val(48);
    constexpr uint32_t reader_dram_addr = get_compile_time_arg_val(49);
    constexpr uint32_t reader_page_size = get_compile_time_arg_val(50);
    constexpr uint32_t reader_tensor_args_index = 51;

    constexpr bool use_split_reader = split_reader && !return_indices;
    constexpr uint32_t eff_kernel_w = (kernel_w - 1) * dilation_w + 1;

    constexpr uint32_t in_w_padded = in_w + pad_w + ceil_pad_w;
    constexpr bool last_tile_is_partial = in_c % TILE_WIDTH != 0;
    constexpr uint32_t in_scalar_cb_id =
        use_split_reader && reader_id == 1 && !one_scalar_per_core ? in_scalar_cb_id_1 : in_scalar_cb_id_0;

    constexpr uint32_t window_size_hw = kernel_h * kernel_w;
    constexpr uint32_t face_r_dim = window_size_hw < FACE_HEIGHT && !return_indices ? window_size_hw : FACE_HEIGHT;
    constexpr uint32_t num_faces_in_input_tile =
        (max_sticks_for_reduction < TILE_WIDTH || window_size_hw <= FACE_HEIGHT) && !return_indices ? 2 : 4;
    constexpr bool is_large_kernel = window_size_hw > max_sticks_for_reduction;
    constexpr bool wide_reduction = in_nblocks_c > 1;
    constexpr uint32_t remaining_elems = window_size_hw % max_sticks_for_reduction;
    constexpr uint32_t interm_reduction_chunks =
        remaining_elems ? window_size_hw / max_sticks_for_reduction + 1 : window_size_hw / max_sticks_for_reduction;
    // we only need to initialize the in_cb if we will not fill each reduction chunk with valid data
    constexpr bool need_to_initialize_in_cb =
        return_indices ||
        (remaining_elems && face_r_dim == FACE_HEIGHT && (num_faces_in_input_tile == 4 || last_tile_is_partial) &&
         interm_reduction_chunks <= multi_buffering_factor);
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
        // TODO do we really need to init the in CB for return indices?
        // - yes until LLK is updated for arbitrary kernel sizes, for now we rely on init
        if constexpr (!is_avg_pool || !is_large_kernel || return_indices) {
            clear_out_tiles<in_cb_id, clear_value_cb_id>();
        }
    }

    if constexpr (reader_id == 0 && return_indices) {
        initialize_return_indices_data<
            kernel_h,
            kernel_w,
            in_w,
            in_c,
            pad_t,
            pad_l,
            dilation_h,
            dilation_w,
            right_inc,
            down_left_wrap_inc,
            up_left_wrap_inc,
            in_idx_cb_id,
            right_inc_cb_id,
            down_left_wrap_inc_cb_id,
            up_left_wrap_inc_cb_id>();
    }

    // initialize the scalar CB
    if constexpr (reader_id == 0 && one_scalar_per_core) {
        // Fill only the first FACE_WIDTH, since we set reload_srcB = true in unpack_tilizeA_B_block, meaning the values
        // for the remaining faces will be reused from the first one. This is safe here because there’s no difference
        // between the first and second face.
        fill_with_val(get_write_ptr(in_scalar_cb_id_0), FACE_WIDTH, bf16_scalar >> 16);
        cb_push_back(in_scalar_cb_id_0, 1);
    }
    const uint32_t core_nhw_index = get_arg_val<uint32_t>(1);

    const uint32_t in_l1_read_base_addr = get_read_ptr(in_shard_cb_id);
    if constexpr (config_in_dram) {
        if (reader_id == 0) {
            load_config_tensor_if_in_dram<
                reader_dram_addr,
                reader_page_size,
                reader_tensor_args_index,
                in_reader_indices_cb_id>(core_nhw_index);

        } else {
            cb_wait_front(in_reader_indices_cb_id, 1);
        }
    }
    uint32_t reader_indices_l1_addr = get_read_ptr(in_reader_indices_cb_id);
    volatile tt_l1_ptr uint32_t* reader_indices_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reader_indices_l1_addr);

    uint32_t segments_counter = 1;
    constexpr uint32_t total_elems_to_reduce = kernel_h * kernel_w;

    volatile tt_l1_ptr uint16_t* config_ptr;
    uint32_t scalar_index = 0;
    uint32_t scalar_start;
    uint32_t scalar_value;
    uint32_t scalar_end;
    uint32_t counter = reader_id;
    if constexpr (!one_scalar_per_core) {
        uint32_t config_l1_addr = get_read_ptr(config_cb_id);
        if constexpr (config_in_dram) {
            if (reader_id == 0) {
                constexpr uint32_t config_tensor_args_index =
                    TensorAccessorArgs<reader_tensor_args_index>().next_compile_time_args_offset();
                load_config_tensor_if_in_dram<
                    config_dram_addr,
                    config_page_size,
                    config_tensor_args_index,
                    config_cb_id>(core_nhw_index);
            } else {
                cb_wait_front(config_cb_id, 1);
            }
        }
        config_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(config_l1_addr);
        scalar_start = config_ptr[0];
        scalar_value = config_ptr[1];
        scalar_end = config_ptr[2];
    }

    uint16_t num_segments = reader_indices_ptr[0] & 0xffff;
    bool first_row_value = reader_id == 0 || !use_split_reader;

    while (num_segments--) {
        uint32_t start_end_segment = reader_indices_ptr[segments_counter++];
        uint16_t start = start_end_segment & 0xffff;
        uint16_t end = start_end_segment >> 16;

        if (!first_row_value) {
            start += stride_w;
            first_row_value = true;
        }

        constexpr uint32_t stride_multiple = use_split_reader ? 2 : 1;
        for (uint16_t ind = start; ind <= end; ind += stride_multiple * stride_w) {
            if constexpr (!one_scalar_per_core) {
                fill_scalar<
                    one_scalar_per_core,
                    in_scalar_cb_id,
                    reader_nindices,
                    use_split_reader,
                    multi_buffering_factor>(scalar_start, scalar_end, scalar_value, scalar_index, counter, config_ptr);
            }
            read_kernel_with_top_left_index<
                in_nblocks_c,
                in_cb_id,
                kernel_h,
                kernel_w,
                in_w_padded,
                in_nbytes_leftover,
                in_c,
                max_sticks_for_reduction,
                total_elems_to_reduce,
                is_avg_pool,
                wide_reduction,
                clear_value_cb_id,
                in_cb_ntiles,
                in_nbytes_c,
                shard_width_bytes,
                is_large_kernel,
                last_tile_is_partial,
                dilation_h,
                dilation_w,
                return_indices,
                zero_pages,
                out_cb_id,
                out_idx_cb_id,
                reader_id,
                pack_tmp_cb_id,
                pack_idx_tmp_cb_id>(ind, in_l1_read_base_addr);
            if (use_split_reader && ind == end) {
                first_row_value = false;
            }
        }
    }
}  // kernel_main()
