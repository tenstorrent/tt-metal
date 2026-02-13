// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <sys/types.h>

#include <cstdint>
#include <api/dataflow/dataflow_api.h>
#include <ttnn/cpp/ttnn/operations/pool/device/kernels/pool_kernels_common.hpp>

#define ENABLE_DEBUG_PRINT 0

#if ENABLE_DEBUG_PRINT == 1
#include "api/debug/dprint.h"
#include "api/debug/dprint_pages.h"
#endif

// Read kernel data for normal max/average pooling (without indices)
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
    bool zero_pages>
ALWI void read_kernel_with_top_left_index(uint32_t ind, uint32_t in_l1_read_base_addr) {
    constexpr uint32_t BYTES_PER_ELEM = 2;
    // average pool with large kernels requires fp32 accumulation so we can only reduce 4 tiles at a time,
    // otherwise we can reduce 8 tiles at a time.
    constexpr uint32_t MAX_TILES_PER_REDUCTION = (is_avg_pool && is_large_kernel) ? 4 : 8;
    constexpr uint32_t MAX_BYTES_PER_REDUCTION = MAX_TILES_PER_REDUCTION * TILE_WIDTH * BYTES_PER_ELEM;
    constexpr uint32_t in_ntiles_c = (in_c + TILE_WIDTH - 1) / TILE_WIDTH;
    constexpr bool tilize_reconfig = in_nblocks_c > 1 && in_ntiles_c % MAX_TILES_PER_REDUCTION != 0 &&
                                     (kernel_h * kernel_w) <= 16 && !last_tile_is_partial;
    uint32_t max_write_inc = wide_reduction ? MAX_BYTES_PER_REDUCTION : in_nbytes_leftover;
    for (uint32_t c_i = 0; c_i < in_nblocks_c; c_i++) {
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
                zero_out_page<in_cb_id>(get_write_ptr(in_cb_id));
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
        if constexpr (!is_large_kernel) {
            noc_async_read_barrier();
            cb_push_back(in_cb_id, 1);
        }
    }
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
    constexpr uint32_t pad_t = get_compile_time_arg_val(37);
    constexpr uint32_t pad_l = get_compile_time_arg_val(38);
    constexpr bool zero_pages = (bool)get_compile_time_arg_val(39);
    constexpr uint32_t config_in_dram = get_compile_time_arg_val(50);
    constexpr uint32_t config_dram_addr = get_compile_time_arg_val(51);
    constexpr uint32_t config_page_size = get_compile_time_arg_val(52);
    constexpr uint32_t reader_dram_addr = get_compile_time_arg_val(53);
    constexpr uint32_t reader_page_size = get_compile_time_arg_val(54);
    constexpr uint32_t reader_tensor_args_index = 55;

    constexpr bool use_split_reader = split_reader;
    constexpr uint32_t eff_kernel_w = (kernel_w - 1) * dilation_w + 1;

    constexpr uint32_t in_w_padded = in_w + pad_w + ceil_pad_w;
    constexpr bool last_tile_is_partial = in_c % TILE_WIDTH != 0;
    constexpr uint32_t in_scalar_cb_id =
        use_split_reader && reader_id == 1 && !one_scalar_per_core ? in_scalar_cb_id_1 : in_scalar_cb_id_0;

    constexpr uint32_t window_size_hw = kernel_h * kernel_w;
    constexpr uint32_t face_r_dim = window_size_hw < FACE_HEIGHT ? window_size_hw : FACE_HEIGHT;
    constexpr uint32_t num_faces_in_input_tile =
        (max_sticks_for_reduction < TILE_WIDTH || window_size_hw <= FACE_HEIGHT) ? 2 : 4;
    constexpr bool is_large_kernel = window_size_hw > max_sticks_for_reduction;
    constexpr bool wide_reduction = in_nblocks_c > 1;
    constexpr uint32_t remaining_elems = window_size_hw % max_sticks_for_reduction;
    constexpr uint32_t interm_reduction_chunks =
        remaining_elems ? window_size_hw / max_sticks_for_reduction + 1 : window_size_hw / max_sticks_for_reduction;
    // we only need to initialize the in_cb if we will not fill each reduction chunk with valid data
    constexpr bool need_to_initialize_in_cb =
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
        if constexpr (!is_avg_pool || !is_large_kernel) {
            clear_out_tiles<in_cb_id, clear_value_cb_id>();
        }
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
                zero_pages>(ind, in_l1_read_base_addr);
            if (use_split_reader && ind == end) {
                first_row_value = false;
            }
        }
    }
}  // kernel_main()
