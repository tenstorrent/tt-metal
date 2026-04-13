// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#ifdef ARCH_QUASAR
#include "experimental/dataflow_buffer.h"
#endif

// Tile is assumed to have 16-bit elements
// Scaler is assumed to be a 16-bit value double packed into a u32
#ifdef ARCH_QUASAR
FORCE_INLINE void generate_mm_scaler(experimental::DataflowBuffer& dfb, const uint32_t scaler) {
    dfb.reserve_back(1);
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        dfb.get_write_ptr() + MEMORY_PORT_NONCACHEABLE_MEM_PORT_MEM_BASE_ADDR);

    constexpr uint32_t num_words = 2048 / sizeof(uint32_t);
    for (uint32_t i = 0; i < num_words; ++i) {
        ptr[i] = 0;
    }

    uint32_t single_packed_scalar = scaler & 0xFFFF;
    for (int i = 0; i < 128; i += 8) {
        ptr[i] = single_packed_scalar;
    }
    for (int i = 256; i < 384; i += 8) {
        ptr[i] = single_packed_scalar;
    }

    dfb.push_back(1);
}
#else
FORCE_INLINE void generate_mm_scaler(const uint32_t cb_id, const uint32_t scaler) {
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

    uint32_t single_packed_scalar = scaler & 0xFFFF;
    for (int i = 0; i < 128; i += 8) {
        ptr[i] = single_packed_scalar;
    }
    for (int i = 256; i < 384; i += 8) {
        ptr[i] = single_packed_scalar;
    }

    cb_push_back(cb_id, 1);
}
#endif
