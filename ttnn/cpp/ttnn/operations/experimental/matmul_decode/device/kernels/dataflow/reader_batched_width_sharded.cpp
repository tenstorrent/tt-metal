// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/local_tensor_accessor.h"
#include "internal/local_mem/address_access.h"
#include "experimental/kernel_args.h"
#ifdef IN1_PIPE
#include "ttnn/operations/matmul/device/kernels/dataflow/prefetcher_pipe_in1_window.hpp"
#endif

// Gathers this core's batch-block rows of width(K)-sharded A into full_in0 (sender-major layout).
// The senders' NOC coordinates are the compile-time varargs, as (x, y) pairs.
//
// IN0_REPLICATED: A is ROW_MAJOR HEIGHT_SHARDED and already holds the full [M, batch * K] on this
// core (row m is every batch's K side by side, so M = 1 is also [batch, K]). Copy this core's Bc
// batches into full_in0 as [Bc][M][K] instead of the width-sharded gather.
//
// in1 (weights) arrive one of three ways:
//   - default: in1 borrows the L1-resident weight shard, so the reader only has to declare its tiles
//     available. A packed weight starts in1_tile_offset tiles into the shard; compute skips those.
//   - IN1_FROM_DRAM: the weight is DRAM ND-sharded, one [Bc*K, Nc] slab per core. The reader
//     issues the slab's read before the gather so the two overlap, and waits for it before
//     publishing full_in0: compute takes in1 only after full_in0, so that gates it.
//   - IN1_PIPE: the tensor prefetcher streams this core's [Bc*K, Nc] slab over a PrefetcherPipe as
//     num_k_blocks entries, each a whole number of rows, and in1 is a relay over the pipe's ring. An
//     entry goes back to the prefetcher only once compute has popped its tiles.
// OUT_CONSUMER: no writer drains compute's output, so the reader waits for it, giving the buffer
// the consumer every buffer needs.
void kernel_main() {
    constexpr uint32_t block_slice_tiles = get_arg(args::block_slice_tiles);
    constexpr uint32_t tile_size_bytes = get_arg(args::tile_size_bytes);
    constexpr uint32_t num_senders = get_arg(args::num_senders);
    constexpr uint32_t in1_page_tiles = get_arg(args::in1_page_tiles);
    constexpr uint32_t Bc = get_arg(args::Bc);
    constexpr uint32_t inA_K_tiles_per_core = get_arg(args::inA_K_tiles_per_core);
    constexpr uint32_t M_tiles = get_arg(args::M_tiles);
    constexpr uint32_t batch = get_arg(args::batch);
#ifdef IN1_PIPE
    constexpr uint32_t num_k_blocks = get_arg(args::num_k_blocks);
#else
    constexpr uint32_t in1_tile_offset = get_arg(args::in1_tile_offset);
#endif
#ifdef OUT_CONSUMER
    constexpr uint32_t out_num_tiles = get_arg(args::out_num_tiles);
#endif
    constexpr uint32_t K_tiles = num_senders * inA_K_tiles_per_core;
    // One in0 tile row of this core's K slice (the whole K on the replicated path).
    constexpr uint32_t k_slice_bytes = inA_K_tiles_per_core * tile_size_bytes;

    const uint32_t b_idx = get_arg(args::b_idx);

    constexpr uint32_t block_slice_bytes = block_slice_tiles * tile_size_bytes;

    Noc noc;
    const LocalTensorAccessor<uint32_t> in0(tensor::in0);
    CircularBuffer in1_cb(dfb::in1);
    CircularBuffer full_in0_cb(dfb::full_in0);
    UnicastEndpoint sender;

#ifdef IN1_PIPE
    // One accessor names every pipe; the one whose receiver is this core is the one bound here.
    experimental::PrefetcherPipe pipe(pipe::in1);
    auto in1_relay = pipe.bind_relay();
    // Publish the first page before the gather so the prefetcher's transfer overlaps it.
    publish_in1_block_from_pipe(pipe, in1_relay, noc, 0, in1_page_tiles);
#elif defined(IN1_FROM_DRAM)
    // TODO: compute waits for the whole slab, so the matmul does not overlap the DRAM read (o_a:
    // ~85 us read, then the per-row compute on top). Publishing the slab in K-blocks, as the pipe
    // path does, would hide the compute and make M > 1 cost about what M = 1 does.
    const auto in1 = TensorAccessor(tensor::in1);
    in1_cb.reserve_back(in1_page_tiles);
    noc_async_read_shard(get_arg(args::in1_shard_id), in1, in1_cb.get_write_ptr());
#else
    in1_cb.reserve_back(in1_tile_offset + in1_page_tiles);
    in1_cb.push_back(in1_tile_offset + in1_page_tiles);
#endif
    full_in0_cb.reserve_back(num_senders * block_slice_tiles);

    // A's shard is at the same L1 address on every core.
    const uint32_t src_addr =
        static_cast<uint32_t>(local_mem::detail::AddressAccess::address(in0.local_mem())) + b_idx * block_slice_bytes;
#ifdef IN0_REPLICATED
    // A is already the full [M, batch * K] replica on this core (ROW_MAJOR, 1x32 faces).
    // custom_mm and matmul_block both consume full_in0 as [bc][m][k] (inA_K_tiles_per_core is
    // the full K), so a single row is one contiguous copy and several are transposed row by row.
    if constexpr (M_tiles == 1) {
        noc_async_write(src_addr, get_noc_addr(full_in0_cb.get_write_ptr()), block_slice_bytes);
    } else {
        const uint32_t in0_base = static_cast<uint32_t>(local_mem::detail::AddressAccess::address(in0.local_mem()));
        const uint32_t dst_base = full_in0_cb.get_write_ptr();
        for (uint32_t m = 0; m < M_tiles; ++m) {
            for (uint32_t bc_i = 0; bc_i < Bc; ++bc_i) {
                const uint32_t src_row = m * batch + b_idx * Bc + bc_i;
                noc_async_write(
                    in0_base + src_row * k_slice_bytes,
                    get_noc_addr(dst_base + (bc_i * M_tiles + m) * k_slice_bytes),
                    k_slice_bytes);
            }
        }
    }
#elif defined(USE_CUSTOM_MM)
    // Restripe sender-major [sender][bc][m][k_local] into [bc][m][k_global] so custom_mm sees
    // kt_dim consecutive K tiles per in0 tile row.
    for (uint32_t s = 0; s < num_senders; ++s) {
        const uint32_t sender_x = get_compile_time_vararg(2 * s);
        const uint32_t sender_y = get_compile_time_vararg(2 * s + 1);
        for (uint32_t row = 0; row < Bc * M_tiles; ++row) {
            noc.async_read(
                sender,
                full_in0_cb,
                k_slice_bytes,
                {.noc_x = sender_x, .noc_y = sender_y, .addr = src_addr + row * k_slice_bytes},
                {.offset_bytes = (row * K_tiles + s * inA_K_tiles_per_core) * tile_size_bytes});
        }
    }
#else
    for (uint32_t s = 0; s < num_senders; ++s) {
        const uint32_t sender_x = get_compile_time_vararg(2 * s);
        const uint32_t sender_y = get_compile_time_vararg(2 * s + 1);
        noc.async_read(
            sender,
            full_in0_cb,
            block_slice_bytes,
            {.noc_x = sender_x, .noc_y = sender_y, .addr = src_addr},
            {.offset_bytes = s * block_slice_bytes});
    }
#endif
#ifdef IN0_REPLICATED
    noc.async_write_barrier();
#endif
#if !defined(IN0_REPLICATED) || defined(IN1_FROM_DRAM)
    noc.async_read_barrier();
#endif
#ifdef IN1_FROM_DRAM
    in1_cb.push_back(in1_page_tiles);
#endif

    full_in0_cb.push_back(num_senders * block_slice_tiles);

#ifdef IN1_PIPE
    for (uint32_t page = 1; page < num_k_blocks; ++page) {
        publish_in1_block_from_pipe(pipe, in1_relay, noc, page, in1_page_tiles);
    }
    drain_in1_pipe_window(pipe, noc, num_k_blocks);
    // pop_front acks the prefetcher with non-posted NOC atomics; retire them before the kernel exits.
    noc.async_atomic_barrier();
#endif

#ifdef OUT_CONSUMER
    CircularBuffer out_cb(dfb::out);
    out_cb.wait_front(out_num_tiles);
    out_cb.pop_front(out_num_tiles);
#endif
}
