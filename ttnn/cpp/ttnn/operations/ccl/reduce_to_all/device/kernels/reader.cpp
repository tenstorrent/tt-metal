// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "tools/profiler/kernel_profiler.hpp"
#include <cstdint>

using tt::data_movement::common::tt_memmove;

template <uint32_t l_tensor_size_bytes, uint32_t ms_tile_size_bytes, uint32_t out_tiles>
FORCE_INLINE void prepare_data_for_compute(
    uint32_t cb_l, uint32_t cb_ms, uint32_t sem_addr, uint32_t recv_buffer_addr) {
    cb_reserve_back(cb_l, out_tiles);
    cb_reserve_back(cb_ms, 1);

    // wait for data arrival
    volatile tt_l1_ptr uint32_t* sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    noc_semaphore_wait(sem_ptr, 1);
    noc_semaphore_set(sem_ptr, 0);

    // push L data to CB (zero-copy, aliased to buffer base)
    cb_push_back(cb_l, out_tiles);
    // memcpy MS data from buffer to CB
    tt_memmove<true, false, false, 0>(get_write_ptr(cb_ms), recv_buffer_addr + l_tensor_size_bytes, ms_tile_size_bytes);
    cb_push_back(cb_ms, 1);
}

// CB IDs
static constexpr uint32_t cb_local_l = get_compile_time_arg_val(0);
static constexpr uint32_t cb_local_ms = get_compile_time_arg_val(1);
static constexpr uint32_t cb_r1_neighbor_l = get_compile_time_arg_val(2);
static constexpr uint32_t cb_r1_neighbor_ms = get_compile_time_arg_val(3);
static constexpr uint32_t cb_r2_neighbor_l = get_compile_time_arg_val(4);
static constexpr uint32_t cb_r2_neighbor_ms = get_compile_time_arg_val(5);

static constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(6);
static constexpr uint32_t vDHt = get_compile_time_arg_val(7);

static constexpr uint32_t l_tensor_size_bytes = get_compile_time_arg_val(8);
static constexpr uint32_t ms_tile_size_bytes = get_compile_time_arg_val(9);

void kernel_main() {
    size_t arg_idx = 0;
    const uint32_t r1_neighbor_sem_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t r2_neighbor_sem_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t r1_recv_buffer_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t r2_recv_buffer_addr = get_arg_val<uint32_t>(arg_idx++);

    constexpr uint32_t out_tiles = Sq_chunk_t * vDHt;

    // =========================================================================
    // Push local input
    // =========================================================================
    cb_reserve_back(cb_local_l, out_tiles);
    cb_push_back(cb_local_l, out_tiles);

    cb_reserve_back(cb_local_ms, 1);
    cb_push_back(cb_local_ms, 1);

    // =========================================================================
    // Prepare R1 neighbor data for compute
    // =========================================================================
    prepare_data_for_compute<l_tensor_size_bytes, ms_tile_size_bytes, out_tiles>(
        cb_r1_neighbor_l, cb_r1_neighbor_ms, r1_neighbor_sem_addr, r1_recv_buffer_addr);

    // =========================================================================
    // Prepare R2 neighbor data for compute
    // =========================================================================
    prepare_data_for_compute<l_tensor_size_bytes, ms_tile_size_bytes, out_tiles>(
        cb_r2_neighbor_l, cb_r2_neighbor_ms, r2_neighbor_sem_addr, r2_recv_buffer_addr);
}
