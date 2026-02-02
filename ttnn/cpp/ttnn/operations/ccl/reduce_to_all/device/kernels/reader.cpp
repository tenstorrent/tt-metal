// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// ZERO-COPY OPTIMIZATIONS:
// 1. LOCAL INPUT: CB aliased to input tensor shard (set_globally_allocated_address)
//    - Data already at CB address, just cb_push_back()
//
// 2. NEIGHBOR DATA (L only): CB aliased to MeshBuffer base (offset 0)
//    - Sender packs [L|MS] and sends to MeshBuffer address
//    - L is at offset 0, so cb_r1_neighbor_l can be aliased directly
//    - Wait for semaphore, then cb_push_back() - NO memcpy for L!
//
// 3. NEIGHBOR DATA (MS): Small memcpy needed
//    - CB aliasing doesn't support offsets, so MS CB is regular CB
//    - After packet arrives, we memcpy MS from buffer to its CB
//    - MS is tiny (~512 bytes) - negligible overhead

#include "api/dataflow/dataflow_api.h"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "tools/profiler/kernel_profiler.hpp"
#include <cstdint>

using tt::data_movement::common::tt_memmove;

void kernel_main() {
    // ==========================================================================
    // Compile-time args (combined MS format)
    // ==========================================================================
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(0);
    constexpr uint32_t vDHt = get_compile_time_arg_val(1);
    [[maybe_unused]] constexpr uint32_t page_size_bytes = get_compile_time_arg_val(2);
    [[maybe_unused]] constexpr uint32_t alignment = get_compile_time_arg_val(3);

    // CBs for local input (aliased to input tensor shard - zero-copy)
    constexpr uint32_t cb_local_l = get_compile_time_arg_val(4);
    constexpr uint32_t cb_local_ms = get_compile_time_arg_val(5);

    // CBs for R1 neighbor data
    // cb_r1_neighbor_l is ALIASED to R1 MeshBuffer (zero-copy for L)
    // cb_r1_neighbor_ms is regular CB (need memcpy)
    constexpr uint32_t cb_r1_neighbor_l = get_compile_time_arg_val(6);
    constexpr uint32_t cb_r1_neighbor_ms = get_compile_time_arg_val(7);

    // CBs for R2 neighbor data (separate buffer for race condition prevention)
    constexpr uint32_t cb_r2_neighbor_l = get_compile_time_arg_val(8);
    constexpr uint32_t cb_r2_neighbor_ms = get_compile_time_arg_val(9);

    // Offsets and sizes for memcpy
    constexpr uint32_t ms_offset_in_buffer = get_compile_time_arg_val(10);  // payload_size_bytes
    constexpr uint32_t ms_size_bytes = get_compile_time_arg_val(11);

    // Derived constants
    constexpr uint32_t out_tiles = Sq_chunk_t * vDHt;

    // ==========================================================================
    // Runtime args
    // ==========================================================================
    size_t arg_idx = 0;
    const uint32_t r1_neighbor_sem_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t r2_neighbor_sem_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t r1_recv_buffer_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t r2_recv_buffer_addr = get_arg_val<uint32_t>(arg_idx++);

    // =========================================================================
    // ROUND 1: Local input (zero-copy) + Neighbor data
    // =========================================================================
    {
        DeviceZoneScopedN("R1-LOCAL-INPUT");
        // LOCAL INPUT: CB aliased to input tensor shard - data already there!
        cb_reserve_back(cb_local_l, out_tiles);
        cb_push_back(cb_local_l, out_tiles);

        cb_reserve_back(cb_local_ms, 1);  // Single combined MS tile
        cb_push_back(cb_local_ms, 1);
    }

    // R1 NEIGHBOR DATA:
    // - L: cb_r1_neighbor_l is aliased to R1 MeshBuffer base (zero-copy!)
    // - MS: Need memcpy from buffer offset to CB address
    volatile tt_l1_ptr uint32_t* r1_neighbor_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(r1_neighbor_sem_addr);

    // Reserve CB space
    cb_reserve_back(cb_r1_neighbor_l, out_tiles);
    cb_reserve_back(cb_r1_neighbor_ms, 1);

    DPRINT << "Reader waiting for R1 neighbor data semaphore." << ENDL();
    {
        DeviceZoneScopedN("R1-WAIT-NEIGHBOR");
        // Wait for R1 data arrival
        noc_semaphore_wait(r1_neighbor_sem_ptr, 1);
        noc_semaphore_set(r1_neighbor_sem_ptr, 0);
    }

    DPRINT << "Reader R1 neighbor data semaphore acquired." << ENDL();
    {
        DeviceZoneScopedN("R1-MEMCPY-MS");
        // L is zero-copy (aliased to buffer base), just push
        cb_push_back(cb_r1_neighbor_l, out_tiles);

        // MS needs memcpy from buffer to CB
        uint32_t r1_ms_cb_addr = get_write_ptr(cb_r1_neighbor_ms);
        tt_memmove<true, false, false, 0>(r1_ms_cb_addr, r1_recv_buffer_addr + ms_offset_in_buffer, ms_size_bytes);

        cb_push_back(cb_r1_neighbor_ms, 1);
    }

    // =========================================================================
    // ROUND 2: Neighbor data from DIFFERENT sender (separate buffer)
    // =========================================================================
    volatile tt_l1_ptr uint32_t* r2_neighbor_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(r2_neighbor_sem_addr);

    cb_reserve_back(cb_r2_neighbor_l, out_tiles);
    cb_reserve_back(cb_r2_neighbor_ms, 1);

    DPRINT << "Reader waiting for R2 neighbor data semaphore." << ENDL();
    {
        DeviceZoneScopedN("R2-WAIT-NEIGHBOR");
        // Wait for R2 data arrival
        noc_semaphore_wait(r2_neighbor_sem_ptr, 1);
        noc_semaphore_set(r2_neighbor_sem_ptr, 0);
    }

    DPRINT << "Reader R2 neighbor data semaphore acquired." << ENDL();
    {
        DeviceZoneScopedN("R2-MEMCPY-MS");
        // L is zero-copy
        cb_push_back(cb_r2_neighbor_l, out_tiles);

        // MS needs memcpy
        uint32_t r2_ms_cb_addr = get_write_ptr(cb_r2_neighbor_ms);
        tt_memmove<true, false, false, 0>(r2_ms_cb_addr, r2_recv_buffer_addr + ms_offset_in_buffer, ms_size_bytes);

        cb_push_back(cb_r2_neighbor_ms, 1);
    }

    DPRINT << "Reader finished processing R2 neighbor data." << ENDL();
    // Done! Compute kernel handles R1 reduction, R2 reduction, and final output.
    DPRINT << "Reader kernel completed." << ENDL();
}
