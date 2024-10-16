// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dataflow_api.h"

// Tile is assumed to have 16-bit elements
// Scaler is assumed to be a 16-bit value double packed into a u32
FORCE_INLINE void generate_reduce_scaler(const uint32_t cb_id, const uint32_t scaler) {
    cb_reserve_back(cb_id, 1);

    constexpr uint32_t num_zeros_reads = 2048 / MEM_ZEROS_SIZE;
    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    uint32_t write_addr = get_write_ptr(cb_id);
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_addr);

    // Fill tile with zeros
    // TODO: src addr does not need to be rewritten. Update/add api for this
    noc_async_read_one_packet_set_state(zeros_noc_addr, MEM_ZEROS_SIZE);
    for (uint32_t i = 0; i < num_zeros_reads; ++i) {
        noc_async_read_one_packet_with_state(zeros_noc_addr, write_addr);
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

    cb_push_back(cb_id, 1);
}

// Tile is assumed to have 32-bit elements
// Scaler is assumed to be a 32-bit value
FORCE_INLINE void generate_reduce_scaler_fp32(const uint32_t cb_id, uint32_t scaler) {
    cb_reserve_back(cb_id, 1);

    constexpr uint32_t num_zeros_reads = 4096 / MEM_ZEROS_SIZE;
    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    uint32_t write_addr = get_write_ptr(cb_id);
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_addr);

    // Fill tile with zeros
    noc_async_read_one_packet_set_state(zeros_noc_addr, MEM_ZEROS_SIZE);
    for (uint32_t i = 0; i < num_zeros_reads; ++i) {
        noc_async_read_one_packet_with_state(zeros_noc_addr, write_addr);
        write_addr += MEM_ZEROS_SIZE;
    }
    noc_async_read_barrier();

    if (scaler != 0) {
        for (int k = 0; k < 4; ++k) {
            uint32_t idx = k << 8;
            for (int j = 0; j < 16; ++j) {
                ptr[idx + j] = scaler;
            }
        }
    }

    cb_push_back(cb_id, 1);
}
