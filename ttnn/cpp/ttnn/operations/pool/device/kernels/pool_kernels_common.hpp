// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// This file contains common utilities for pool operation kernels (rotate, upsample, grid_sample).
// It is intended for use in device kernels only, not host code.

#pragma once

#include <api/dataflow/dataflow_api.h>

#define ALWI inline __attribute__((always_inline))

template <uint32_t cb_id>
FORCE_INLINE void zero_out_page(uint32_t write_addr) {
    const uint32_t page_size = get_local_cb_interface(cb_id).fifo_page_size;
    const uint32_t num_zeros_reads = page_size / MEM_ZEROS_SIZE;
    const uint32_t remainder_bytes = page_size % MEM_ZEROS_SIZE;
    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);

    noc_async_read_one_packet_set_state(zeros_noc_addr, MEM_ZEROS_SIZE);
    for (uint32_t i = 0; i < num_zeros_reads; ++i) {
        noc_async_read_one_packet_with_state<true>(zeros_noc_addr, write_addr);
        write_addr += MEM_ZEROS_SIZE;
    }
    if (remainder_bytes > 0) {
        noc_async_read(zeros_noc_addr, write_addr, remainder_bytes);
    }
}

FORCE_INLINE void zero_out_nbytes(uint32_t write_addr, uint32_t nbytes) {
    const uint32_t num_zeros_reads = nbytes / MEM_ZEROS_SIZE;
    const uint32_t remainder_bytes = nbytes % MEM_ZEROS_SIZE;
    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);

    noc_async_read_one_packet_set_state(zeros_noc_addr, MEM_ZEROS_SIZE);
    for (uint32_t i = 0; i < num_zeros_reads; ++i) {
        noc_async_read_one_packet_with_state<true>(zeros_noc_addr, write_addr);
        write_addr += MEM_ZEROS_SIZE;
    }
    if (remainder_bytes > 0) {
        noc_async_read(zeros_noc_addr, write_addr, remainder_bytes);
    }
}
