// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused gate+up partial-width-sharded matmul cross-core K-reduction.
//
// Mirrors writer_partial_width_sharded.cpp but ships TWO partials (gate, up) to the base core: each
// core writes its gate partial into slot `k_idx` of the base core's gate_reduce_cb and its up
// partial into slot `k_idx` of the base core's up_reduce_cb, bumping the matching reduce semaphore
// for each. Base cores wait for all K_blocks of BOTH and publish both reduce CBs to compute. The
// outputs are width-sharded (buffer-backed out CBs), so no interleaved-output scatter is needed.
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/noc_semaphore.h"
#include "experimental/endpoints.h"
using experimental::CircularBuffer;
using experimental::Noc;
using experimental::Semaphore;
using experimental::UnicastEndpoint;

void kernel_main() {
    constexpr uint32_t gate_partial_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t gate_reduce_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t block_num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t tile_size_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t K_blocks = get_compile_time_arg_val(4);
    constexpr uint32_t gate_reduce_sem_id = get_compile_time_arg_val(5);
    constexpr uint32_t up_partial_cb_index = get_compile_time_arg_val(6);
    constexpr uint32_t up_reduce_cb_index = get_compile_time_arg_val(7);
    constexpr uint32_t up_reduce_sem_id = get_compile_time_arg_val(8);

    const uint32_t k_idx = get_arg_val<uint32_t>(0);
    const uint32_t base_noc_x = get_arg_val<uint32_t>(1);
    const uint32_t base_noc_y = get_arg_val<uint32_t>(2);
    const uint32_t is_base = get_arg_val<uint32_t>(3);

    constexpr uint32_t block_size_bytes = block_num_tiles * tile_size_bytes;

    Noc noc;
    CircularBuffer gate_partial_cb(gate_partial_cb_index);
    CircularBuffer gate_reduce_cb(gate_reduce_cb_index);
    CircularBuffer up_partial_cb(up_partial_cb_index);
    CircularBuffer up_reduce_cb(up_reduce_cb_index);
    Semaphore<> gate_reduce_sem(gate_reduce_sem_id);
    Semaphore<> up_reduce_sem(up_reduce_sem_id);
    UnicastEndpoint base_core;

    // Reserve both reduce regions up front on the base core so incoming unicast writes land in
    // valid CB space.
    if (is_base) {
        gate_reduce_cb.reserve_back(K_blocks * block_num_tiles);
        up_reduce_cb.reserve_back(K_blocks * block_num_tiles);
    }

    // ---- gate partial -> base core's gate_reduce_cb slot k_idx ----
    gate_partial_cb.wait_front(block_num_tiles);
    const uint32_t gate_dst_addr = gate_reduce_cb.get_write_ptr() + k_idx * block_size_bytes;
    noc.async_write(
        gate_partial_cb,
        base_core,
        block_size_bytes,
        {.offset_bytes = 0},
        {.noc_x = base_noc_x, .noc_y = base_noc_y, .addr = gate_dst_addr});
    noc.async_write_barrier();
    gate_reduce_sem.up(noc, base_noc_x, base_noc_y, 1);
    noc.async_atomic_barrier();
    gate_partial_cb.pop_front(block_num_tiles);

    // ---- up partial -> base core's up_reduce_cb slot k_idx ----
    up_partial_cb.wait_front(block_num_tiles);
    const uint32_t up_dst_addr = up_reduce_cb.get_write_ptr() + k_idx * block_size_bytes;
    noc.async_write(
        up_partial_cb,
        base_core,
        block_size_bytes,
        {.offset_bytes = 0},
        {.noc_x = base_noc_x, .noc_y = base_noc_y, .addr = up_dst_addr});
    noc.async_write_barrier();
    up_reduce_sem.up(noc, base_noc_x, base_noc_y, 1);
    noc.async_atomic_barrier();
    up_partial_cb.pop_front(block_num_tiles);

    // Base core: once all K_blocks partials of BOTH have arrived, publish them to compute.
    if (is_base) {
        gate_reduce_sem.wait(K_blocks);
        gate_reduce_cb.push_back(K_blocks * block_num_tiles);
        up_reduce_sem.wait(K_blocks);
        up_reduce_cb.push_back(K_blocks * block_num_tiles);
    }
}
