// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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

#ifdef ACTIVATION_REUSE
template <
    uint32_t coalesced_read_bytes,
    uint32_t conv_act_c_read_bytes,
    uint32_t act_block_w_extra_align_bytes,
    uint32_t stride_w_bytes,
    uint32_t stride_h_bytes,
    uint32_t window_inner,
    uint32_t stride_w,
    uint32_t window_outer,
    uint32_t cb_id_act,
    uint32_t act_cb_tiles,
    uint32_t act_cb_w_tiles,
    bool output_image_starts_from_row_beginning,
    uint32_t image_width_tiles,
    uint32_t output_image_width,
    uint32_t window_reuse_offset,
    uint32_t expected_total_tiles_pushed>
FORCE_INLINE void read_sticks_activation_reuse(
    volatile tt_l1_ptr uint32_t* packed_reader_indices_ptr,
    uint32_t reader_offset,
    uint32_t& l1_write_addr_act,
    uint32_t& reader_idx,
    uint32_t cb_start_addr) {
    constexpr uint32_t image_width_padded_to_tile = image_width_tiles * 32;
    constexpr uint32_t output_image_width_full_tile = output_image_width == image_width_padded_to_tile;

    uint16_t num_elems = packed_reader_indices_ptr[reader_idx] & 0xffff;
    uint32_t pixel_row = 0, pixel_column = 0, total_tiles_pushed = 0;
    bool first_pixel_row = true;

    cb_reserve_back(cb_id_act, act_cb_tiles);

    while (num_elems--) {
        reader_idx++;
        uint16_t start_ind = packed_reader_indices_ptr[reader_idx] & 0xffff;
        uint16_t end_ind = packed_reader_indices_ptr[reader_idx] >> 16;

        for (uint16_t ind = start_ind; ind <= end_ind; ind += stride_w) {
            uint32_t act_l1_offset = reader_offset + (ind * conv_act_c_read_bytes);
            uint32_t outer = 0;
            if constexpr (output_image_width_full_tile) {
                if (!first_pixel_row) {
                    outer = window_outer - 1;
                    l1_write_addr_act += outer * coalesced_read_bytes;
                    act_l1_offset += outer * stride_h_bytes;
                }
            } else {
                if (!first_pixel_row && (pixel_column < output_image_width)) {
                    outer = window_outer - 1;
                    l1_write_addr_act += outer * coalesced_read_bytes;
                    act_l1_offset += outer * stride_h_bytes;
                }
            }
            for (; outer < window_outer; outer++) {
                // read full inner dim at once
                noc_async_read_one_packet_with_state<true>(act_l1_offset, l1_write_addr_act);
                l1_write_addr_act += coalesced_read_bytes;
                act_l1_offset += stride_h_bytes;
            }
            l1_write_addr_act += act_block_w_extra_align_bytes;

            pixel_column++;
            // if full tile, push back
            if ((pixel_column & 31) == 0) {
                noc_async_read_barrier();
                cb_push_back(cb_id_act, act_cb_w_tiles);
                total_tiles_pushed += act_cb_w_tiles;

                // move on to the next output image row
                if constexpr (!output_image_starts_from_row_beginning) {
                    if (pixel_column == image_width_padded_to_tile) {
                        pixel_column = 0;
                        pixel_row++;
                        first_pixel_row = false;

                        cb_reserve_back(cb_id_act, act_cb_tiles);
                        l1_write_addr_act = cb_start_addr + pixel_row * window_reuse_offset;
                        get_local_cb_interface(cb_id_act).fifo_wr_ptr = l1_write_addr_act;
                    }
                }
            }
        }

        // move on to the next output image row
        if constexpr (output_image_starts_from_row_beginning) {
            pixel_column = 0;
            pixel_row++;
            first_pixel_row = false;

            cb_reserve_back(cb_id_act, act_cb_tiles);
            l1_write_addr_act = cb_start_addr + pixel_row * window_reuse_offset;
            get_local_cb_interface(cb_id_act).fifo_wr_ptr = l1_write_addr_act;
        }
    }
    reader_idx++;

    // needed for the last core that might have less data than what compute expects
    if (total_tiles_pushed < expected_total_tiles_pushed) {
        uint32_t remaining_tiles_to_push = expected_total_tiles_pushed - total_tiles_pushed;

        for (uint32_t i = 0; i < remaining_tiles_to_push; i += image_width_tiles) {
            get_local_cb_interface(cb_id_act).fifo_wr_ptr = cb_start_addr;
            cb_push_back(cb_id_act, image_width_tiles);
        }
    }
}
#endif
