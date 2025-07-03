// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#define ALWI inline __attribute__((always_inline))

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
    noc_async_read_barrier();
}
