// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/constants.hpp>
#include "dataflow_api.h"

template <int window_height, int window_width>
FORCE_INLINE void read_dilated_channels(
    uint32_t& l1_write_addr_act,
    const uint32_t act_l1_read_addr,
    const uint32_t reader_channel_idx,
    const uint32_t conv_act_c_bytes,
    const uint32_t stride_h_bytes,
    const uint32_t stride_w_bytes) {
    uint32_t act_l1_read_addr_plus_offset = act_l1_read_addr + (reader_channel_idx * conv_act_c_bytes);
#pragma GCC unroll(window_height)
    for (uint32_t outer = 0; outer < window_height; outer++) {
        uint32_t act_l1_read_addr_row_offset = act_l1_read_addr_plus_offset;
#pragma GCC unroll(window_width)
        for (uint32_t inner = 0; inner < window_width; inner++) {
            // Read the partial depth.
            noc_async_read_one_packet_with_state<true>(act_l1_read_addr_row_offset, l1_write_addr_act);
            // Increment by full depth to go to the next pixel
            l1_write_addr_act += conv_act_c_bytes;
            act_l1_read_addr_row_offset += stride_w_bytes;
        }
        // Go to the next row
        act_l1_read_addr_plus_offset += stride_h_bytes;
    }
}

template <int window_height>
FORCE_INLINE void read_channels(
    uint32_t& l1_write_addr_act,
    const uint32_t act_l1_read_addr,
    const uint32_t reader_channel_idx,
    const uint32_t conv_act_c_read_bytes,
    const uint32_t coalesced_read_bytes,
    const uint32_t stride_h_bytes) {
    uint32_t act_l1_read_addr_plus_offset = act_l1_read_addr + (reader_channel_idx * conv_act_c_read_bytes);
#pragma GCC unroll(window_height)
    for (uint32_t inner = 0; inner < window_height; inner++) {
        noc_async_read_one_packet_with_state<true>(act_l1_read_addr_plus_offset, l1_write_addr_act);
        l1_write_addr_act += coalesced_read_bytes;
        act_l1_read_addr_plus_offset += stride_h_bytes;
    }
}
