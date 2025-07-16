// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "debug/dprint_tile.h"

// Tile is assumed to have 16-bit elements
// Scaler is assumed to be a 16-bit value double packed into a u32
FORCE_INLINE void generate_reduce_scaler(const uint32_t cb_id, const uint32_t scaler) {
    cb_reserve_back(cb_id, 1);

    constexpr uint32_t num_zeros_reads = 2048 / MEM_ZEROS_SIZE;
    // DPRINT << " num_zeros_reads: " << num_zeros_reads << ENDL();
    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    // DPRINT << " zeros_noc_addr: " << zeros_noc_addr << ENDL();
    uint32_t write_addr = get_write_ptr(cb_id);
    // DPRINT << " write_addr: " << write_addr << ENDL();
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_addr);

    // Fill tile with zeros
    // TODO: src addr does not need to be rewritten. Update/add api for this
    // noc_async_read_one_packet_set_state(zeros_noc_addr, MEM_ZEROS_SIZE);
    for (uint32_t i = 0; i < num_zeros_reads; ++i) {
        // noc_async_read_one_packet_with_state(zeros_noc_addr, write_addr);
        noc_async_read_one_packet(zeros_noc_addr, write_addr, MEM_ZEROS_SIZE);
        write_addr += MEM_ZEROS_SIZE;
    }
    noc_async_read_barrier();

    if (scaler != 0) {
        for (int k = 0; k < 4; ++k) {
            uint32_t idx = k << 7;
            for (int j = 0; j < 8; ++j) {
                ptr[idx + j] = scaler;
            }
        }
    }

    DPRINT << " cb_id: " << cb_id << ENDL();
    DPRINT << " scaler: " << scaler << ENDL();
    for (uint8_t iii = 0; iii < 32; ++iii) {
        DPRINT << TileSlice(
                      cb_id,
                      0,
                      SliceRange{.h0 = iii, .h1 = (uint8_t)(iii + 1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1},
                      TSLICE_OUTPUT_CB,
                      TSLICE_WR_PTR,
                      true,
                      true)
               << ENDL();
    }

    cb_push_back(cb_id, 1);
}
template <bool needs_zeroing = true>
FORCE_INLINE void wh_generate_reduce_scaler(const uint32_t cb_id, const uint32_t scaler) {
    // This is much faster but WILL NOT WORK IN BLACKHOLE since it assumes 32B allignment noc reads are allowed, done
    // for llama effort
    cb_reserve_back(cb_id, 1);
    uint32_t write_addr_base = get_write_ptr(cb_id);
    uint64_t target_address = get_noc_addr(write_addr_base);
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_addr_base);

    // Fill tile with zeros
    // TODO: src addr does not need to be rewritten. Update/add api for this
    if constexpr (needs_zeroing) {
        uint32_t write_addr = write_addr_base;
        constexpr uint32_t num_zeros_reads = 2048 / MEM_ZEROS_SIZE;
        uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
        noc_async_read_one_packet_set_state(zeros_noc_addr, MEM_ZEROS_SIZE);
        for (uint32_t i = 0; i < num_zeros_reads; ++i) {
            noc_async_read_one_packet_with_state(zeros_noc_addr, write_addr);
            write_addr += MEM_ZEROS_SIZE;
        }
        noc_async_read_barrier();
    }

    if (scaler != 0 || !needs_zeroing) {
        for (int j = 0; j < 8; ++j) {
            ptr[j] = scaler;
        }
        noc_async_read_one_packet_set_state(target_address, 32);
        noc_async_read_one_packet_with_state(target_address, write_addr_base + (1 << 9));
        noc_async_read_one_packet_with_state(target_address, write_addr_base + (2 << 9));
        noc_async_read_one_packet_with_state(target_address, write_addr_base + (3 << 9));
        noc_async_read_barrier();
    }
    cb_push_back(cb_id, 1);
}
