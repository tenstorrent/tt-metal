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
        noc_async_read_one_packet_with_state(zeros_noc_addr, write_addr);
        write_addr += MEM_ZEROS_SIZE;
    }
    noc_async_write_barrier();
}

template <
    uint32_t dilation_w,
    uint32_t coalesced_read_bytes,
    uint32_t conv_act_c_read_bytes,
    uint32_t act_block_w_extra_align_bytes,
    uint32_t stride_w_bytes,
    uint32_t weight_size_w>
FORCE_INLINE void read_sticks(
    uint32_t act_block_h_datums_read_curr,
    volatile tt_l1_ptr uint32_t* packed_reader_indices_ptr,
    uint32_t reader_offset,
    uint32_t& l1_write_addr_act,
    uint32_t& reader_idx) {
    for (uint32_t bhd = 0; bhd < act_block_h_datums_read_curr; bhd++) {
        // local read from reader_index + reader_offset;
        uint32_t two_reader_indices = packed_reader_indices_ptr[reader_idx];
        uint32_t reader_idx_1 = two_reader_indices & 0xffff;
        uint32_t reader_idx_2 = two_reader_indices >> 16;

        if constexpr (dilation_w == 1) {
            uint32_t act_l1_offset = reader_offset + (reader_idx_1 * conv_act_c_read_bytes);
            noc_async_read_one_packet_with_state<true>(act_l1_offset, l1_write_addr_act);
            l1_write_addr_act += (coalesced_read_bytes + act_block_w_extra_align_bytes);

            act_l1_offset = reader_offset + (reader_idx_2 * conv_act_c_read_bytes);
            noc_async_read_one_packet_with_state<true>(act_l1_offset, l1_write_addr_act);
            l1_write_addr_act += (coalesced_read_bytes + act_block_w_extra_align_bytes);
        } else {
            uint32_t act_l1_offset = reader_offset + (reader_idx_1 * conv_act_c_read_bytes);
            for (uint32_t inner = 0; inner < weight_size_w; inner++) {
                noc_async_read_one_packet_with_state<true>(act_l1_offset, l1_write_addr_act);
                l1_write_addr_act += conv_act_c_read_bytes;
                act_l1_offset += stride_w_bytes;
            }
            l1_write_addr_act += act_block_w_extra_align_bytes;

            act_l1_offset = reader_offset + (reader_idx_2 * conv_act_c_read_bytes);
            for (uint32_t inner = 0; inner < weight_size_w; inner++) {
                noc_async_read_one_packet_with_state<true>(act_l1_offset, l1_write_addr_act);
                l1_write_addr_act += conv_act_c_read_bytes;
                act_l1_offset += stride_w_bytes;
            }
            l1_write_addr_act += act_block_w_extra_align_bytes;
        }
        reader_idx++;
    }
}
