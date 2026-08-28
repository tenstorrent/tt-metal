// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Reader for the WIDTH-SHARDED toy_variance path.
//
// It reads no tensor data at all -- the input shard is already resident in this core's L1 and the
// DFB is bound to it with `borrowed_from`, so there is nothing to move. What this kernel owns is:
//
//   1. crediting the borrowed shard;
//   2. the mean broadcast -- the root sends, everyone else receives.
//
// The reduce scaler is not in that list: reduce<> builds and owns it on the compute side via
// ReduceScaler::compute_managed(), normalizing by the FULL row width so each core emits its share of
// the mean directly and the root's combine stays a plain add.
//
// (1) is the one shape here that is an artifact rather than a design choice. Every DFB must have
// exactly one producer instance on every node hosting it, and there is no "already filled" state a
// DFB can be declared in, so resident data needs a producer that writes nothing. One
// reserve_back/push_back pair over the whole shard is the whole cost: nothing has been pushed, so
// every entry is free and it returns immediately.
//
// (2) is in this kernel, and not split with the writer, because `dfb::mean` may have only ONE
// producer binding per node -- so the send half and the receive half have to be the same kernel,
// selected at runtime.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe_spec.hpp"

void kernel_main() {
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t shard_tiles = get_arg(args::shard_tiles);
    const uint32_t is_root = get_arg(args::is_root);

    Noc noc;
    DataflowBuffer dfb_in(dfb::in_shard);
    DataflowBuffer dfb_mean_src(dfb::mean_src);
    DataflowBuffer dfb_mean(dfb::mean);
    constexpr auto mc = MCAST_ARGS(mean_bcast);

    // Credit the resident shard. No write -- the bytes are already there.
    dfb_in.reserve_back(shard_tiles);
    dfb_in.push_back(shard_tiles);

    // ---------- broadcast the mean ----------
    // Every core reserves first, so `mean_entry` is the same address on all of them (DFB base
    // addresses are program-global) -- which is what lets the root multicast into the receivers'
    // entry without ever being told where that is.
    const uint32_t tile_bytes = dfb_mean.get_tile_size();
    dfb_mean.reserve_back(Ht);
    const uint32_t mean_entry = dfb_mean.get_write_ptr();

    if (is_root) {
        dfb_mean_src.wait_front(Ht);

        // mean_src -> mean, L1 to L1 on this core. It is a self-aimed READ, not a write: a
        // DataflowBuffer is a legal read DESTINATION but not a unicast write destination.
        noc.async_read(
            UnicastEndpoint{},
            dfb_mean,
            Ht * tile_bytes,
            dataflow_kernel_lib::local_addr(dfb_mean_src.get_read_ptr(), noc.get_noc_id()),
            {.offset_bytes = 0});
        noc.async_read_barrier();

        auto pipe = mc.sender(noc);
        if constexpr (mc.active) {
            pipe.send(mean_entry, mean_entry, Ht * tile_bytes);
        }
        dfb_mean_src.pop_front(Ht);
    } else {
        auto pipe = mc.receiver(noc);
        pipe.receive();
    }

    dfb_mean.push_back(Ht);
}
