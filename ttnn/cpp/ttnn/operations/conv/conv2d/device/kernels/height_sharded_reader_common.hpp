// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/constants.hpp>
#include "dataflow_api.h"

// Zero out all tiles for a given circular buffer.
template <uint32_t cb_id>
FORCE_INLINE void zero_out_tiles() {
    constexpr uint32_t tile_size = get_tile_size(cb_id);
    static_assert(
        tile_size % MEM_ZEROS_SIZE == 0, "Tile size must be a multiple of MEM_ZEROS_BASE for zeroing out tiles");
    const uint32_t num_tiles = get_local_cb_interface(cb_id).fifo_num_pages;
    const uint32_t num_zeros_reads = (tile_size / MEM_ZEROS_SIZE) * num_tiles;
    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    uint32_t write_addr = get_write_ptr(cb_id);

    noc_async_read_one_packet_set_state(zeros_noc_addr, MEM_ZEROS_SIZE);
    for (uint32_t i = 0; i < num_zeros_reads; ++i) {
        noc_async_read_one_packet_with_state<true>(zeros_noc_addr, write_addr);
        write_addr += MEM_ZEROS_SIZE;
    }
    noc_async_read_barrier();
}

template <
    uint32_t dilation_w,
    uint32_t coalesced_read_bytes,
    uint32_t conv_act_c_read_bytes,
    uint32_t act_block_w_extra_align_bytes,
    uint32_t stride_w_bytes,
    uint32_t weight_size_w,
    uint32_t stride_w>
FORCE_INLINE void read_sticks(
    volatile tt_l1_ptr uint32_t* packed_reader_indices_ptr,
    uint32_t reader_offset,
    uint32_t& l1_write_addr_act,
    uint32_t& reader_idx) {
    uint16_t num_elems = packed_reader_indices_ptr[reader_idx] & 0xffff;

    while (num_elems--) {
        reader_idx++;
        uint16_t start_ind = packed_reader_indices_ptr[reader_idx] & 0xffff;
        uint16_t end_ind = packed_reader_indices_ptr[reader_idx] >> 16;

        if constexpr (dilation_w == 1) {
            for (uint16_t ind = start_ind; ind <= end_ind; ind += stride_w) {
                uint32_t act_l1_offset = reader_offset + (ind * conv_act_c_read_bytes);
                noc_async_read_one_packet_with_state<true>(act_l1_offset, l1_write_addr_act);
                l1_write_addr_act += (coalesced_read_bytes + act_block_w_extra_align_bytes);
            }
        } else {
            for (uint16_t ind = start_ind; ind <= end_ind; ind += stride_w) {
                uint32_t act_l1_offset = reader_offset + (ind * conv_act_c_read_bytes);
                for (uint32_t inner = 0; inner < weight_size_w; inner++) {
                    noc_async_read_one_packet_with_state<true>(act_l1_offset, l1_write_addr_act);
                    l1_write_addr_act += conv_act_c_read_bytes;
                    act_l1_offset += stride_w_bytes;
                }
                l1_write_addr_act += act_block_w_extra_align_bytes;
            }
        }
    }
    reader_idx++;
}

template <uint32_t coalesced_read_bytes, uint32_t stride_h_bytes>
FORCE_INLINE void read_kernel_w(uint32_t& l1_write_addr_act, uint32_t& act_l1_offset) {
    noc_async_read_one_packet_with_state<true>(act_l1_offset, l1_write_addr_act);
    l1_write_addr_act += coalesced_read_bytes;
    act_l1_offset += stride_h_bytes;
}

#ifdef ACTIVATION_REUSE
template <uint32_t cb_id_act, uint32_t act_cb_tiles, uint32_t window_reuse_offset>
FORCE_INLINE void pass_to_the_next_image_width(
    uint32_t& l1_write_addr_act, uint32_t cb_start_addr, uint32_t& pixel_row, uint32_t& pixel_column) {
    pixel_column = 0;
    pixel_row++;
    cb_reserve_back(cb_id_act, act_cb_tiles);
    l1_write_addr_act = cb_start_addr + pixel_row * window_reuse_offset;
    get_local_cb_interface(cb_id_act).fifo_wr_ptr = l1_write_addr_act;
}

template <uint32_t cb_id_act, uint32_t act_cb_w_tiles>
FORCE_INLINE void push_full_tile_height() {
    noc_async_read_barrier();
    cb_push_back(cb_id_act, act_cb_w_tiles);
}

template <uint32_t cb_id_act, uint32_t act_cb_w_tiles, uint32_t image_width_tiles>
FORCE_INLINE void push_remaining_tiles(uint32_t remaining_tiles_to_push, uint32_t cb_start_addr) {
    constexpr uint32_t tiles_to_push = image_width_tiles * act_cb_w_tiles;
    for (uint32_t i = 0; i < remaining_tiles_to_push; i += image_width_tiles) {
        get_local_cb_interface(cb_id_act).fifo_wr_ptr = cb_start_addr;
        cb_push_back(cb_id_act, tiles_to_push);
    }
}

// Function to read the windows in the first output image row where we don't reuse anything
template <
    uint32_t coalesced_read_bytes,
    uint32_t stride_h_bytes,
    uint32_t window_outer,
    uint32_t cb_id_act,
    uint32_t act_cb_w_tiles,
    uint32_t conv_act_c_read_bytes,
    uint32_t act_block_w_extra_align_bytes>
FORCE_INLINE void read_first_image_row_window(
    uint32_t& l1_write_addr_act, uint32_t reader_offset, uint16_t ind, uint32_t& pixel_column) {
    uint32_t act_l1_offset = reader_offset + (ind * conv_act_c_read_bytes);
    for (uint32_t outer = 0; outer < window_outer; outer++) {
        read_kernel_w<coalesced_read_bytes, stride_h_bytes>(l1_write_addr_act, act_l1_offset);
    }
    l1_write_addr_act += act_block_w_extra_align_bytes;

    pixel_column++;
    // if full tile, push back
    if ((pixel_column & 31) == 0) {
        push_full_tile_height<cb_id_act, act_cb_w_tiles>();
    }
}

template <
    uint32_t coalesced_read_bytes,
    uint32_t stride_h_bytes,
    uint32_t outer_coalesced_read_bytes,
    uint32_t outer_stride_h_bytes>
FORCE_INLINE void read_image_row_window_with_reuse(uint32_t& l1_write_addr_act, uint32_t& act_l1_offset) {
    l1_write_addr_act += outer_coalesced_read_bytes;
    act_l1_offset += outer_stride_h_bytes;
    read_kernel_w<coalesced_read_bytes, stride_h_bytes>(l1_write_addr_act, act_l1_offset);
}

template <
    uint32_t coalesced_read_bytes,
    uint32_t conv_act_c_read_bytes,
    uint32_t act_block_w_extra_align_bytes,
    uint32_t stride_h_bytes,
    uint32_t window_inner,
    uint32_t stride_w,
    uint32_t window_outer,
    uint32_t cb_id_act,
    uint32_t act_cb_tiles,
    uint32_t act_cb_w_tiles,
    bool readers_process_full_image_widths,
    uint32_t image_width_tiles,
    uint32_t output_image_width,
    uint32_t window_reuse_offset>
FORCE_INLINE void read_sticks_activation_reuse(
    volatile tt_l1_ptr uint32_t* packed_reader_indices_ptr,
    uint32_t reader_offset,
    uint32_t& l1_write_addr_act,
    uint32_t& reader_idx,
    uint32_t cb_start_addr) {
    constexpr uint32_t image_width_padded_to_tile = image_width_tiles * tt::constants::TILE_HEIGHT;
    constexpr bool output_image_width_full_tile = output_image_width == image_width_padded_to_tile;
    constexpr uint32_t reuse_outer = window_outer - 1;
    constexpr uint32_t outer_coalesced_read_bytes = reuse_outer * coalesced_read_bytes;
    constexpr uint32_t outer_stride_h_bytes = reuse_outer * stride_h_bytes;

    uint16_t num_elems = packed_reader_indices_ptr[reader_idx] & 0xffff;
    uint32_t pixel_row = 0, pixel_column = 0;

    cb_reserve_back(cb_id_act, act_cb_tiles);

    if (num_elems == 0) {
        return;
    }

    reader_idx++;
    uint16_t start_ind = packed_reader_indices_ptr[reader_idx] & 0xffff;
    uint16_t end_ind = packed_reader_indices_ptr[reader_idx] >> 16;

    // ------ HANDLE FIRST OUTPUT IMAGE WIDTH SEPARATELY, WHERE WE NEED TO READ THE FULL WINDOW ------
    for (uint16_t ind = start_ind; ind <= end_ind; ind += stride_w) {
        read_first_image_row_window<
            coalesced_read_bytes,
            stride_h_bytes,
            window_outer,
            cb_id_act,
            act_cb_w_tiles,
            conv_act_c_read_bytes,
            act_block_w_extra_align_bytes>(l1_write_addr_act, reader_offset, ind, pixel_column);
    }

    num_elems--;
    reader_idx++;
    start_ind = packed_reader_indices_ptr[reader_idx] & 0xffff;
    end_ind = packed_reader_indices_ptr[reader_idx] >> 16;

    if constexpr (!readers_process_full_image_widths) {
        if (num_elems) {
            // The first image width might be split between two rows
            uint16_t leftover_row_width = image_width_padded_to_tile - pixel_column;
            uint16_t second_row_width = leftover_row_width, third_row_width = 0;
            if constexpr (!output_image_width_full_tile) {
                // If the output image width is not a multiple of the tile width, the first 'image width' might be split
                // between three rows since we padd image width to tile size; otherwise, it is always split between
                // maximum two rows
                const uint16_t interval_width = end_ind - start_ind + 1;
                if (leftover_row_width > interval_width) {
                    second_row_width = interval_width;
                    third_row_width = leftover_row_width - second_row_width;
                }
            }

            for (uint16_t ind = start_ind; ind < start_ind + second_row_width; ind += stride_w) {
                read_first_image_row_window<
                    coalesced_read_bytes,
                    stride_h_bytes,
                    window_outer,
                    cb_id_act,
                    act_cb_w_tiles,
                    conv_act_c_read_bytes,
                    act_block_w_extra_align_bytes>(l1_write_addr_act, reader_offset, ind, pixel_column);
            }

            start_ind = start_ind + second_row_width;

            if constexpr (!output_image_width_full_tile) {
                if (third_row_width > 0) {
                    num_elems--;
                    reader_idx++;
                    start_ind = packed_reader_indices_ptr[reader_idx] & 0xffff;
                    end_ind = packed_reader_indices_ptr[reader_idx] >> 16;

                    for (uint16_t ind = start_ind; ind < start_ind + third_row_width; ind += stride_w) {
                        read_first_image_row_window<
                            coalesced_read_bytes,
                            stride_h_bytes,
                            window_outer,
                            cb_id_act,
                            act_cb_w_tiles,
                            conv_act_c_read_bytes,
                            act_block_w_extra_align_bytes>(l1_write_addr_act, reader_offset, ind, pixel_column);
                    }

                    start_ind = start_ind + third_row_width;
                }
            }
        }
    }
    // Move on to the next output image width
    pass_to_the_next_image_width<cb_id_act, act_cb_tiles, window_reuse_offset>(
        l1_write_addr_act, cb_start_addr, pixel_row, pixel_column);

    // ------ HANDLE REMAINING INPUT, WHERE WE READ JUST THE LAST KERNEL WIDTH OF THE WINDOW ------
    while (num_elems--) {
        for (uint16_t ind = start_ind; ind <= end_ind; ind += stride_w) {
            uint32_t act_l1_offset = reader_offset + (ind * conv_act_c_read_bytes);
            if constexpr (output_image_width_full_tile) {
                // In this case, everything is reused after the first image width
                read_image_row_window_with_reuse<
                    coalesced_read_bytes,
                    stride_h_bytes,
                    outer_coalesced_read_bytes,
                    outer_stride_h_bytes>(l1_write_addr_act, act_l1_offset);
            } else {
                // In this case, we need to check if we are in the limits of image width since we pad it to tile size
                if (pixel_column < output_image_width) {
                    read_image_row_window_with_reuse<
                        coalesced_read_bytes,
                        stride_h_bytes,
                        outer_coalesced_read_bytes,
                        outer_stride_h_bytes>(l1_write_addr_act, act_l1_offset);
                } else {
                    for (uint32_t outer = 0; outer < window_outer; outer++) {
                        read_kernel_w<coalesced_read_bytes, stride_h_bytes>(l1_write_addr_act, act_l1_offset);
                    }
                }
            }

            l1_write_addr_act += act_block_w_extra_align_bytes;

            pixel_column++;
            if ((pixel_column & 31) == 0) {
                push_full_tile_height<cb_id_act, act_cb_w_tiles>();

                // Move on to the next output image width
                if constexpr (!readers_process_full_image_widths) {
                    if (pixel_column == image_width_padded_to_tile) {
                        pass_to_the_next_image_width<cb_id_act, act_cb_tiles, window_reuse_offset>(
                            l1_write_addr_act, cb_start_addr, pixel_row, pixel_column);
                    }
                }
            }
        }

        if constexpr (readers_process_full_image_widths) {
            // Move on to the next output image width
            pass_to_the_next_image_width<cb_id_act, act_cb_tiles, window_reuse_offset>(
                l1_write_addr_act, cb_start_addr, pixel_row, pixel_column);
        }

        reader_idx++;
        start_ind = packed_reader_indices_ptr[reader_idx] & 0xffff;
        end_ind = packed_reader_indices_ptr[reader_idx] >> 16;
    }
}
#endif
template <uint32_t dram_addr_index, uint32_t page_size_index, uint32_t tensor_args_index, uint32_t cb_reader_index>
void load_config_tensor_if_in_dram(uint32_t core_index) {
#ifdef CONFIG_TENSOR_IN_DRAM
    // TODO: Instead of all cores reading from dram, only the first column reads, and does an MCAST to all the other
    // cores in the row.
    constexpr uint32_t config_dram_addr = get_compile_time_arg_val(dram_addr_index);
    constexpr uint32_t config_page_size = get_compile_time_arg_val(page_size_index);
    const auto config_tensor_args = TensorAccessorArgs<tensor_args_index>();
    const auto config_accessor = TensorAccessor(config_tensor_args, config_dram_addr, config_page_size);
    uint64_t src_noc_addr = get_noc_addr(core_index, config_accessor);

    noc_async_read(src_noc_addr, get_write_ptr(cb_reader_index), config_page_size);
    noc_async_read_barrier();
    cb_push_back(cb_reader_index, 1);
#endif
}
