// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Reader for Sequential Eltwise Test
 *
 * Reads input data for all phases of the sequential eltwise operation:
 * Phase 0: Read A, B into cb_in0, cb_in1
 * Phase 1: Read C into cb_in2
 * Phase 2: Read D into cb_in3
 *
 * Uses InterleavedAddrGen to properly read from interleaved DRAM buffers.
 */

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime args
    uint32_t src_a_addr = get_arg_val<uint32_t>(0);
    uint32_t src_b_addr = get_arg_val<uint32_t>(1);
    uint32_t src_c_addr = get_arg_val<uint32_t>(2);
    uint32_t src_d_addr = get_arg_val<uint32_t>(3);
    uint32_t num_tiles = get_arg_val<uint32_t>(4);
    uint32_t num_phases = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_id_in0 = 0;  // input A
    constexpr uint32_t cb_id_in1 = 1;  // input B
    constexpr uint32_t cb_id_in2 = 3;  // input C
    constexpr uint32_t cb_id_in3 = 4;  // input D

    uint32_t tile_size_bytes = get_tile_size(cb_id_in0);

    // Create interleaved address generators for each input buffer
    const InterleavedAddrGen<true> src_a_addrgen = {
        .bank_base_address = src_a_addr,
        .page_size = tile_size_bytes,
    };
    const InterleavedAddrGen<true> src_b_addrgen = {
        .bank_base_address = src_b_addr,
        .page_size = tile_size_bytes,
    };
    const InterleavedAddrGen<true> src_c_addrgen = {
        .bank_base_address = src_c_addr,
        .page_size = tile_size_bytes,
    };
    const InterleavedAddrGen<true> src_d_addrgen = {
        .bank_base_address = src_d_addr,
        .page_size = tile_size_bytes,
    };

    // ========== PHASE 0: Read A and B ==========
    for (uint32_t i = 0; i < num_tiles; ++i) {
        uint64_t src_a_noc_addr = src_a_addrgen.get_noc_addr(i);
        uint64_t src_b_noc_addr = src_b_addrgen.get_noc_addr(i);

        cb_reserve_back(cb_id_in0, 1);
        cb_reserve_back(cb_id_in1, 1);

        uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
        uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);

        noc_async_read(src_a_noc_addr, l1_write_addr_in0, tile_size_bytes);
        noc_async_read(src_b_noc_addr, l1_write_addr_in1, tile_size_bytes);
        noc_async_read_barrier();

        cb_push_back(cb_id_in0, 1);
        cb_push_back(cb_id_in1, 1);
    }

    if (num_phases == 1) {
        return;
    }

    // ========== PHASE 1: Read C ==========
    for (uint32_t i = 0; i < num_tiles; ++i) {
        uint64_t src_c_noc_addr = src_c_addrgen.get_noc_addr(i);

        cb_reserve_back(cb_id_in2, 1);
        uint32_t l1_write_addr_in2 = get_write_ptr(cb_id_in2);

        noc_async_read(src_c_noc_addr, l1_write_addr_in2, tile_size_bytes);
        noc_async_read_barrier();

        cb_push_back(cb_id_in2, 1);
    }

    if (num_phases == 2) {
        return;
    }

    // ========== PHASE 2: Read D ==========
    for (uint32_t i = 0; i < num_tiles; ++i) {
        uint64_t src_d_noc_addr = src_d_addrgen.get_noc_addr(i);

        cb_reserve_back(cb_id_in3, 1);
        uint32_t l1_write_addr_in3 = get_write_ptr(cb_id_in3);

        noc_async_read(src_d_noc_addr, l1_write_addr_in3, tile_size_bytes);
        noc_async_read_barrier();

        cb_push_back(cb_id_in3, 1);
    }
}
