// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared page / packet enumeration for the Quasar all-gather kernels. Every kernel walks the same packet
// sequence, so in the RING arm the writer thread t drains exactly the entries reader thread t produced
// and the (worker, thread) that sends a packet is the (worker, thread) that receives it.
//
// A packet is a run of consecutive shard-local tiles that is contiguous in L1 at BOTH ends, so it moves
// as ONE NoC transaction of tiles_per_packet x page bytes:
//   - source: a shard is stored shard-row-major, so consecutive shard-local tiles are consecutive pages;
//   - destination: an output shard is the full tensor. A height slice is a contiguous page range of it;
//     a width slice is one run per tile row, so packets never cross a shard row when width-sharded.

#pragma once

#include <cstdint>

#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"
#include "api/dataflow/semaphore_binding_token.h"

namespace all_gather {

constexpr uint32_t num_ranks = get_arg(args::num_ranks);
constexpr uint32_t num_workers = get_arg(args::num_workers);
constexpr uint32_t tiles_per_packet = get_arg(args::tiles_per_packet);
constexpr uint32_t chunks_per_sync = get_arg(args::chunks_per_sync);
constexpr uint32_t pages_total = get_arg(args::pages_total);
constexpr uint32_t pages_per_slice = get_arg(args::pages_per_slice);
constexpr uint32_t shard_rows_t = get_arg(args::shard_rows_t);
constexpr uint32_t shard_cols_t = get_arg(args::shard_cols_t);
constexpr uint32_t tensor_cols_t = get_arg(args::tensor_cols_t);
constexpr bool width_sharded = get_arg(args::width_sharded) != 0;

// Packet geometry: a slice is `rows` runs of `row_len` tiles that are contiguous at both ends.
constexpr uint32_t row_len = width_sharded ? shard_cols_t : pages_per_slice;
constexpr uint32_t rows = pages_per_slice / row_len;
constexpr uint32_t packets_per_row = (row_len + tiles_per_packet - 1) / tiles_per_packet;
constexpr uint32_t num_packets = rows * packets_per_row;

// Input page id of shard-local tile k (0 <= k < pages_per_slice) of slice s. The same expression gives
// the tile's page inside any full-tensor output shard, offset by that shard's base page.
inline uint32_t slice_page(uint32_t s, uint32_t k) {
    if constexpr (width_sharded) {
        return (k / shard_cols_t) * tensor_cols_t + s * shard_cols_t + (k % shard_cols_t);
    } else {
        return s * pages_per_slice + k;
    }
}

struct Packet {
    uint32_t k0;  // first shard-local tile
    uint32_t n;   // tiles in the packet (a row's tail packet may be short)
};

inline Packet packet(uint32_t j) {
    const uint32_t row = j / packets_per_row;
    const uint32_t col0 = (j % packets_per_row) * tiles_per_packet;
    const uint32_t left = row_len - col0;
    return {row * row_len + col0, left < tiles_per_packet ? left : tiles_per_packet};
}

// Packet j is owned by global thread j % (num_workers * num_threads).
struct ThreadPackets {
    uint32_t first;  // worker * num_threads + thread
    uint32_t step;   // num_workers * num_threads

    explicit ThreadPackets(uint32_t worker) :
        first(worker * get_num_threads() + get_my_thread_id()), step(num_workers * get_num_threads()) {}
};

#if defined(ALL_GATHER_RING)
// The ready semaphore of thread t: the host binds ready0..ready<T-1> (ALL_GATHER_THREADS = T), and a
// binding token is a plain {id, scope} pair, so pick the token first and build one Semaphore from it.
inline SemaphoreBindingToken ready_token(uint32_t tid) {
    SemaphoreBindingToken token = sem::ready0;
#if ALL_GATHER_THREADS > 1
    if (tid == 1) {
        token = sem::ready1;
    }
#endif
#if ALL_GATHER_THREADS > 2
    if (tid == 2) {
        token = sem::ready2;
    }
#endif
#if ALL_GATHER_THREADS > 3
    if (tid == 3) {
        token = sem::ready3;
    }
#endif
    return token;
}
#endif

}  // namespace all_gather
