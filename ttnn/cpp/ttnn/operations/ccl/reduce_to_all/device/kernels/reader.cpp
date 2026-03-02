// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// BUFFER LAYOUT:
// Buffer: [L_data][MS]
// - L data at offset 0 (contiguous) - CB aliased for zero-copy
// - MS at offset total_l_bytes (end of buffer) - copied to cb_ms
//
// TWO TRANSFER MODES (selected at compile time via single_shot_l flag):
//
// Single-shot mode (single_shot_l = 1):
//   Packet arrival order: MS first (sem >= 1), then full L (sem >= 2)
//   Reader waits for 2 semaphore increments total.
//
// Chunked streaming mode (single_shot_l = 0):
//   Packet arrival order: MS first (sem >= 1), then L chunks (sem >= 2, 3, ...)
//   Reader streams L chunks to compute as they arrive.

#include "api/dataflow/dataflow_api.h"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "tools/profiler/kernel_profiler.hpp"
#include <cstdint>

using tt::data_movement::common::tt_memmove;

// =============================================================================
// Compile-time arguments
// =============================================================================
static constexpr uint32_t cb_local_l = get_compile_time_arg_val(0);
static constexpr uint32_t cb_local_ms = get_compile_time_arg_val(1);
static constexpr uint32_t cb_r1_neighbor_l = get_compile_time_arg_val(2);
static constexpr uint32_t cb_r1_neighbor_ms = get_compile_time_arg_val(3);
static constexpr uint32_t cb_r2_neighbor_l = get_compile_time_arg_val(4);
static constexpr uint32_t cb_r2_neighbor_ms = get_compile_time_arg_val(5);

static constexpr uint32_t ms_tile_size_bytes = get_compile_time_arg_val(6);
static constexpr uint32_t single_shot_l = get_compile_time_arg_val(7);  // 0 = chunked, 1 = single-shot
static constexpr uint32_t total_l_bytes = get_compile_time_arg_val(8);  // Full L size (always valid)
static constexpr uint32_t out_tiles = get_compile_time_arg_val(9);      // Total L tiles (always valid)
// Chunked mode parameters (only meaningful when single_shot_l == 0)
static constexpr uint32_t l_chunk_size_bytes = get_compile_time_arg_val(10);
static constexpr uint32_t num_l_chunks = get_compile_time_arg_val(11);
static constexpr uint32_t tiles_per_l_chunk = get_compile_time_arg_val(12);

// Semaphore thresholds:
// MS = 1 (always first)
// L base = 2 (single-shot: just 2; chunked: 2 + chunk_idx)
static constexpr uint32_t MS_SEM_THRESHOLD = 1;
static constexpr uint32_t L_SEM_BASE_THRESHOLD = 2;

// =============================================================================
// Helper functions
// =============================================================================

/**
 * Prepare MS data for compute.
 * MS arrives first (sem >= 1), located at end of buffer.
 */
FORCE_INLINE void prepare_ms_for_compute(
    uint32_t cb_ms, volatile tt_l1_ptr uint32_t* sem_ptr, uint32_t recv_buffer_addr) {
    cb_reserve_back(cb_ms, 1);

    // Wait for MS packet (sem >= 1)
    noc_semaphore_wait_min(sem_ptr, MS_SEM_THRESHOLD);

    // MS is at end of buffer (offset = total_l_bytes)
    tt_memmove<true, false, false, 0>(get_write_ptr(cb_ms), recv_buffer_addr + total_l_bytes, ms_tile_size_bytes);
    cb_push_back(cb_ms, 1);
}

/**
 * Prepare all data for one round (MS first, then L).
 * In single-shot mode: waits for full L as one packet (sem >= 2).
 * In chunked mode: streams L chunks as they arrive (sem >= 2, 3, ...).
 */
FORCE_INLINE void prepare_data_for_compute(
    uint32_t cb_l, uint32_t cb_ms, uint32_t sem_addr, uint32_t recv_buffer_addr) {
    volatile tt_l1_ptr uint32_t* sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);

    // MS first (sem >= 1)
    prepare_ms_for_compute(cb_ms, sem_ptr, recv_buffer_addr);

    if (single_shot_l) {
        // Single-shot: wait for full L (sem >= 2), zero-copy
        cb_reserve_back(cb_l, out_tiles);
        noc_semaphore_wait_min(sem_ptr, L_SEM_BASE_THRESHOLD);
        cb_push_back(cb_l, out_tiles);
    } else {
        // Chunked: stream L chunks as they arrive
        for (uint32_t i = 0; i < num_l_chunks; i++) {
            cb_reserve_back(cb_l, tiles_per_l_chunk);
            // Wait for this L chunk (sem >= 2 + i)
            noc_semaphore_wait_min(sem_ptr, L_SEM_BASE_THRESHOLD + i);
            // L CB is aliased to buffer, just push (zero-copy)
            cb_push_back(cb_l, tiles_per_l_chunk);
        }
    }

    noc_semaphore_set(sem_ptr, 0);
}

// =============================================================================
// Main kernel
// =============================================================================

void kernel_main() {
    size_t arg_idx = 0;
    const uint32_t r1_neighbor_sem_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t r2_neighbor_sem_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t r1_recv_buffer_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t r2_recv_buffer_addr = get_arg_val<uint32_t>(arg_idx++);

    // =========================================================================
    // Push local input (aliased CBs, no copy needed)
    // =========================================================================
    {
        DeviceZoneScopedN("R1-LOCAL-INPUT");
        cb_reserve_back(cb_local_l, out_tiles);
        cb_push_back(cb_local_l, out_tiles);

        cb_reserve_back(cb_local_ms, 1);
        cb_push_back(cb_local_ms, 1);
    }

    // =========================================================================
    // Prepare R1 neighbor data for compute
    // =========================================================================
    {
        DeviceZoneScopedN("R1-WAIT-NEIGHBOR");
        prepare_data_for_compute(cb_r1_neighbor_l, cb_r1_neighbor_ms, r1_neighbor_sem_addr, r1_recv_buffer_addr);
    }

    // =========================================================================
    // Prepare R2 neighbor data for compute
    // =========================================================================
    {
        DeviceZoneScopedN("R2-WAIT-NEIGHBOR");
        prepare_data_for_compute(cb_r2_neighbor_l, cb_r2_neighbor_ms, r2_neighbor_sem_addr, r2_recv_buffer_addr);
    }
}
