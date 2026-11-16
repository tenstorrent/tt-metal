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
//
// COMP (on is_comp cores): first its [K, 32] slab of w_comp from DRAM and the tokens replica, for the
// compressor-column tile compute runs before q_b. The q_b slab's DRAM read starts once that slab is in, so
// compute can start on it while the q_b slab streams.
//
// QA_OUT: a core with a qa_out_page then copies the row's q_a * gamma tiles (1x32 tiles are row-major bytes)
// into its row of q_a_output, last, so q_b's compute is not held up. Nothing rewrites the row in this launch.
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
#endif

#ifdef COMP
    if (get_arg(args::is_comp) != 0) {
        constexpr uint32_t k_tiles = get_arg(args::k_tiles);
        CircularBuffer in0_cb(dfb::in0);
        CircularBuffer w_comp_cb(dfb::w_comp);
        const auto w_comp = TensorAccessor(tensor::w_comp);
        w_comp_cb.reserve_back(k_tiles);
        noc_async_read_shard(get_arg(args::comp_shard_id), w_comp, w_comp_cb.get_write_ptr());
        // in0 is the tokens replica itself.
        in0_cb.reserve_back(k_tiles);
        in0_cb.push_back(k_tiles);
        noc.async_read_barrier();
        w_comp_cb.push_back(k_tiles);
    }
#endif

#ifndef W_PIPE
    CircularBuffer w_cb(dfb::w);
    const auto w = TensorAccessor(tensor::w);
    w_cb.reserve_back(w_tiles);
    noc_async_read_shard(get_arg(args::shard_id), w, w_cb.get_write_ptr());
#endif

    ready_sem.wait(1);
    row_cb.reserve_back(row_tiles);
#ifdef QA_OUT
    const uint32_t row_addr = row_cb.get_write_ptr();
#endif
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

#ifdef QA_OUT
    const uint32_t qa_out_page = get_arg(args::qa_out_page);
    if (qa_out_page != 0xFFFFFFFF) {
        constexpr uint32_t row_tile_bytes = get_arg(args::row_tile_bytes);
        const auto q_a_output = TensorAccessor(tensor::q_a_output);
        noc_async_write(
            row_addr + get_arg(args::scaled_q_a_first) * row_tile_bytes,
            q_a_output.get_noc_addr(qa_out_page),
            get_arg(args::q_lora_tiles) * row_tile_bytes);
        noc.async_write_barrier();
    }
#endif
}
