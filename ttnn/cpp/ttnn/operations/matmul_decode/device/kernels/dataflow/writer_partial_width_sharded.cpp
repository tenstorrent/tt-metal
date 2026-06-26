// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"

// Partial-width-sharded matmul cross-core K-reduction.
//
// Each B core produces a partial product (compute -> partial_cb). The K_blocks cores
// that share an N-slice all reduce onto the *base* core (the k_idx == 0 core, which
// owns that output N-slice):
//
//   1. Every core ships its partial into slot `k_idx` of the base core's reduce_cb via
//      a unicast NoC write, then atomically bumps the base core's reduce semaphore.
//   2. The base core waits until all K_blocks partials have arrived, then publishes the
//      reduce_cb to its compute kernel (which sums the blocks into the output).
//
// reduce_cb is allocated identically on every core, so a sender can use its own local
// reduce_cb write pointer as the (matching) destination L1 address on the base core.
void kernel_main() {
    constexpr uint32_t partial_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t reduce_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t block_num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t tile_size_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t K_blocks = get_compile_time_arg_val(4);
    constexpr uint32_t reduce_sem_id = get_compile_time_arg_val(5);
    constexpr uint32_t interleaved_output = get_compile_time_arg_val(6);
    constexpr uint32_t out_cb_index = get_compile_time_arg_val(7);
    constexpr uint32_t M_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t Nc_tiles = get_compile_time_arg_val(9);
    constexpr uint32_t N_tiles = get_compile_time_arg_val(10);
    constexpr auto out_args = TensorAccessorArgs<11>();

    const uint32_t k_idx = get_arg_val<uint32_t>(0);
    const uint32_t base_noc_x = get_arg_val<uint32_t>(1);
    const uint32_t base_noc_y = get_arg_val<uint32_t>(2);
    const uint32_t is_base = get_arg_val<uint32_t>(3);
    const uint32_t n_idx = get_arg_val<uint32_t>(4);
    const uint32_t out_buffer_addr = get_arg_val<uint32_t>(5);

    constexpr uint32_t block_size_bytes = block_num_tiles * tile_size_bytes;

    Noc noc;
    CircularBuffer partial_cb(partial_cb_index);
    CircularBuffer reduce_cb(reduce_cb_index);
    Semaphore<> reduce_sem(reduce_sem_id);
    UnicastEndpoint base_core;

    // Reserve the whole reduce region up front on the base core so incoming unicast
    // writes (which target a fixed L1 address) land in valid CB space.
    if (is_base) {
        reduce_cb.reserve_back(K_blocks * block_num_tiles);
    }

    // Wait for this core's partial product from compute.
    partial_cb.wait_front(block_num_tiles);

    // reduce_cb has the same L1 address on every core, so our local write pointer is
    // also the base core's reduce_cb base address. Write into this core's k_idx slot.
    const uint32_t dst_addr = reduce_cb.get_write_ptr() + k_idx * block_size_bytes;
    noc.async_write(
        partial_cb,
        base_core,
        block_size_bytes,
        {.offset_bytes = 0},
        {.noc_x = base_noc_x, .noc_y = base_noc_y, .addr = dst_addr});
    noc.async_write_barrier();

    reduce_sem.up(noc, base_noc_x, base_noc_y, 1);
    noc.async_atomic_barrier();

    partial_cb.pop_front(block_num_tiles);
    // Base core: once all K_blocks partials have arrived, publish them to compute.
    if (is_base) {
        reduce_sem.wait(K_blocks);
        reduce_cb.push_back(K_blocks * block_num_tiles);

        // Interleaved-output fold: compute packs the reduced (+gelu) output into out_cb;
        // NoC-scatter this base core's [M_tiles x Nc_tiles] N-slice into the interleaved
        // output buffer (tile (mt, nc) -> page mt*N_tiles + n_idx*Nc_tiles + nc). This
        // replaces the caller's separate sharded->interleaved reshard.
        if (interleaved_output) {
            const auto out_acc = TensorAccessor(out_args, out_buffer_addr, tile_size_bytes);
            cb_wait_front(out_cb_index, block_num_tiles);
            uint32_t l1_read_addr = get_read_ptr(out_cb_index);
            for (uint32_t mt = 0; mt < M_tiles; ++mt) {
                for (uint32_t nc = 0; nc < Nc_tiles; ++nc) {
                    const uint32_t page = mt * N_tiles + n_idx * Nc_tiles + nc;
                    noc_async_write_page(page, out_acc, l1_read_addr);
                    l1_read_addr += tile_size_bytes;
                }
            }
            noc_async_write_barrier();
            cb_pop_front(out_cb_index, block_num_tiles);
        }
    }
}
