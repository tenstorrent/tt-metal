// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/prefetcher_pipe.h"
#include "api/semaphore.h"
#include "experimental/kernel_args.h"

// A q_a or kv core: this core's [K, 32] weight slab from DRAM (or W_PIPE: its prefetcher ring), then its one-tile
// result into slot `slot` of the hub's qkv_row -- and a q_a core its tile times gamma into slot scaled_q_a_first +
// slot. The hub multicasts the whole row over the q_b and kv cores once every slot is in; it is off that rectangle, so
// the multicast excludes it. IS_KV: this core is in the rectangle, and hands the row to its compute once it lands.
void kernel_main() {
    constexpr uint32_t k_tiles = get_arg(args::k_tiles);
    constexpr uint32_t row_tiles = get_arg(args::row_tiles);
    constexpr uint32_t row_tile_bytes = get_arg(args::row_tile_bytes);
    constexpr uint32_t hub_noc_x = get_arg(args::hub_noc_x);
    constexpr uint32_t hub_noc_y = get_arg(args::hub_noc_y);
    constexpr uint32_t row_start_noc_x = get_arg(args::row_start_noc_x);
    constexpr uint32_t row_start_noc_y = get_arg(args::row_start_noc_y);
    constexpr uint32_t row_end_noc_x = get_arg(args::row_end_noc_x);
    constexpr uint32_t row_end_noc_y = get_arg(args::row_end_noc_y);
    constexpr uint32_t num_row_dests = get_arg(args::num_row_dests);
    const uint32_t slot = get_arg(args::slot);
    const bool is_hub = get_arg(args::is_hub) != 0;

    Noc noc;
    CircularBuffer in0_cb(dfb::in0);
    CircularBuffer x_cb(dfb::x);
    CircularBuffer row_cb(dfb::qkv_row);
    Semaphore<> arrive_sem(sem::arrive);
    Semaphore<> ready_sem(sem::ready);

#ifdef W_PIPE
    // The slab is one entry of the prefetcher's ring; w relays it to compute in place.
    experimental::PrefetcherPipe pipe(pipe::w);
    auto w_relay = pipe.bind_relay();
    w_relay.reserve_back(k_tiles);
#else
    CircularBuffer w_cb(dfb::w);
    const auto w = TensorAccessor(tensor::w);
    w_cb.reserve_back(k_tiles);
    noc_async_read_shard(get_arg(args::shard_id), w, w_cb.get_write_ptr());
#endif

    // in0 is the tokens replica itself.
    in0_cb.reserve_back(k_tiles);
    in0_cb.push_back(k_tiles);

#ifndef IS_KV
    // gamma is resident in the tensor it borrows.
    CircularBuffer q_a_gamma_cb(dfb::q_a_gamma);
    q_a_gamma_cb.reserve_back(1);
    q_a_gamma_cb.push_back(1);
#endif

#ifdef W_PIPE
    pipe.wait_front(1);
    w_relay.push_back(k_tiles);
#else
    noc.async_read_barrier();
    w_cb.push_back(k_tiles);
#endif

    const uint32_t row_addr = row_cb.get_write_ptr();
#ifdef IS_KV
    constexpr uint32_t x_tiles = 1;
#else
    constexpr uint32_t x_tiles = 2;
#endif
    x_cb.wait_front(x_tiles);
    noc_async_write(
        x_cb.get_read_ptr(), get_noc_addr(hub_noc_x, hub_noc_y, row_addr + slot * row_tile_bytes), row_tile_bytes);
#ifndef IS_KV
    constexpr uint32_t scaled_q_a_first = get_arg(args::scaled_q_a_first);
    noc_async_write(
        x_cb.get_read_ptr() + row_tile_bytes,
        get_noc_addr(hub_noc_x, hub_noc_y, row_addr + (scaled_q_a_first + slot) * row_tile_bytes),
        row_tile_bytes);
#endif
    noc.async_write_barrier();
    arrive_sem.up(noc, hub_noc_x, hub_noc_y, x_tiles);
    x_cb.pop_front(x_tiles);
#ifdef W_PIPE
    // Compute popped the slab before producing x; hand the entry back to the prefetcher.
    pipe.pop_front(1, noc);
#endif

    if (is_hub) {
        arrive_sem.wait(row_tiles);
        noc_async_write_multicast(
            row_addr,
            get_noc_multicast_addr(row_start_noc_x, row_start_noc_y, row_end_noc_x, row_end_noc_y, row_addr),
            row_tiles * row_tile_bytes,
            num_row_dests);
        noc.async_write_barrier();
        ready_sem.inc_multicast(noc, row_start_noc_x, row_start_noc_y, row_end_noc_x, row_end_noc_y, 1, num_row_dests);
    }
    noc.async_atomic_barrier();

#ifdef IS_KV
    ready_sem.wait(1);
    row_cb.reserve_back(row_tiles);
    row_cb.push_back(row_tiles);
#endif
}
