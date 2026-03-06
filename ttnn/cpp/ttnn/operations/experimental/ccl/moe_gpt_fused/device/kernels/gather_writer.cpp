// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Gather Writer (RISCV_0 / NOC_0) for moe_gpt_fused
//
// Gathers tilized data from all 3 tilize cores onto the drain core,
// then pushes 90 tiles to each matmul core's c_1 and signals them.
//
// Non-drain cores: send their 30 tiles to drain core at the appropriate offset.
// Drain core: wait for non-drain cores, receive matmul c_1 address, push tiles.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "moe_gpt_fused_ring_common.h"

void kernel_main() {
    constexpr uint32_t num_gather_cores = moe_gpt_fused_ring::NUM_GATHER_CORES;            // 3
    constexpr uint32_t tiles_per_gather_core = moe_gpt_fused_ring::TILES_PER_GATHER_CORE;  // 30
    constexpr uint32_t k_tiles = moe_gpt_fused_ring::K_TILES;                              // 90

    constexpr auto tilize_output_cb = tt::CBIndex::c_16;
    constexpr uint32_t tile_size = get_tile_size(tilize_output_cb);
    constexpr uint32_t total_bytes = k_tiles * tile_size;  // 90 * 2048 = 184320

    // Runtime args
    uint32_t argidx = 0;
    const auto gather_core_id = get_arg_val<uint32_t>(argidx++);
    const auto gather_semaphore_id = get_arg_val<uint32_t>(argidx++);
    const auto tilize_ready_semaphore_id = get_arg_val<uint32_t>(argidx++);
    const auto drain_noc_x = get_arg_val<uint32_t>(argidx++);
    const auto drain_noc_y = get_arg_val<uint32_t>(argidx++);
    const auto num_matmul_cores = get_arg_val<uint32_t>(argidx++);

    uint32_t matmul_noc_x[12], matmul_noc_y[12];
    for (uint32_t i = 0; i < num_matmul_cores; ++i) {
        matmul_noc_x[i] = get_arg_val<uint32_t>(argidx++);
        matmul_noc_y[i] = get_arg_val<uint32_t>(argidx++);
    }

    const auto addr_exchange_semaphore_id = get_arg_val<uint32_t>(argidx++);

    const bool is_drain = (gather_core_id == 0);

    // Wait for compute to push tilized tiles
    cb_wait_front(tilize_output_cb, k_tiles);
    const uint32_t local_output_addr = get_read_ptr(tilize_output_cb);

    if (!is_drain) {
        // Non-drain core: send our 30 tiles to drain core at the appropriate offset
        const uint32_t tile_offset = gather_core_id * tiles_per_gather_core;
        const uint32_t byte_offset = tile_offset * tile_size;

        const uint64_t drain_dest_addr = get_noc_addr(drain_noc_x, drain_noc_y, local_output_addr + byte_offset);

        noc_async_write(local_output_addr, drain_dest_addr, tiles_per_gather_core * tile_size);
        noc_async_write_barrier();

        // Signal drain core that our tiles are sent
        uint32_t gather_sem_addr = get_semaphore(gather_semaphore_id);
        uint64_t drain_gather_sem_addr = get_noc_addr(drain_noc_x, drain_noc_y, gather_sem_addr);
        noc_semaphore_inc(drain_gather_sem_addr, 1);
    } else {
        // Drain core: wait for non-drain cores to send their tiles
        uint32_t gather_sem_addr = get_semaphore(gather_semaphore_id);
        volatile tt_l1_ptr uint32_t* gather_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gather_sem_addr);
        *gather_sem_ptr = 0;
        noc_semaphore_wait(gather_sem_ptr, num_gather_cores - 1);

        // All 90 tiles are now in local c_16 buffer:
        //   tiles 0-29: our own (from compute)
        //   tiles 30-59: from core 1
        //   tiles 60-89: from core 2

        // Wait for matmul core 0 to send us its c_1 base address via addr_exchange
        uint32_t addr_exchange_sem_addr = get_semaphore(addr_exchange_semaphore_id);
        volatile tt_l1_ptr uint32_t* addr_exchange_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr_exchange_sem_addr);
        // Wait for non-zero value (the c_1 address)
        while (*addr_exchange_ptr == 0) {
        };
        const uint32_t matmul_c1_addr = *addr_exchange_ptr;

        // Push 90 tiles to each matmul core's c_1
        for (uint32_t i = 0; i < num_matmul_cores; ++i) {
            uint64_t dest_noc_addr = get_noc_addr(matmul_noc_x[i], matmul_noc_y[i], matmul_c1_addr);
            noc_async_write(local_output_addr, dest_noc_addr, total_bytes);
        }
        noc_async_write_barrier();

        // Signal all matmul cores that tilized input is in their c_1
        uint32_t tilize_ready_sem_addr = get_semaphore(tilize_ready_semaphore_id);
        for (uint32_t i = 0; i < num_matmul_cores; ++i) {
            uint64_t matmul_sem_noc_addr = get_noc_addr(matmul_noc_x[i], matmul_noc_y[i], tilize_ready_sem_addr);
            noc_semaphore_inc(matmul_sem_noc_addr, 1);
        }
        noc_async_atomic_barrier();
    }

    cb_pop_front(tilize_output_cb, k_tiles);
}
