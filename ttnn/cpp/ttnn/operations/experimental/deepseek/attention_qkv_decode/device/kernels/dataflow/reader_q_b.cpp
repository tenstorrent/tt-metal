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

// A q_b core: its [q_lora, Nc] weight slab from DRAM (or W_PIPE: its prefetcher ring), then the hub's
// multicast row.
void kernel_main() {
    constexpr uint32_t w_tiles = get_arg(args::w_tiles);
    constexpr uint32_t row_tiles = get_arg(args::row_tiles);

    Noc noc;
    CircularBuffer row_cb(dfb::qkv_row);
    Semaphore<> ready_sem(sem::ready);

#ifdef W_PIPE
    experimental::PrefetcherPipe pipe(pipe::w);
    auto w_relay = pipe.bind_relay();
    w_relay.reserve_back(w_tiles);
#else
    CircularBuffer w_cb(dfb::w);
    const auto w = TensorAccessor(tensor::w);
    w_cb.reserve_back(w_tiles);
    noc_async_read_shard(get_arg(args::shard_id), w, w_cb.get_write_ptr());
#endif

    ready_sem.wait(1);
    row_cb.reserve_back(row_tiles);
    row_cb.push_back(row_tiles);

#ifdef W_PIPE
    pipe.wait_front(1);
    w_relay.push_back(w_tiles);
    // Waits for compute to pop the slab before handing the entry back.
    pipe.pop_front(1, noc);
    noc.async_atomic_barrier();
#else
    noc.async_read_barrier();
    w_cb.push_back(w_tiles);
#endif
}
