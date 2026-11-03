// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/semaphore.h"
#include "api/tensor/local_tensor_accessor.h"
#include "internal/local_mem/address_access.h"
#include "experimental/kernel_args.h"
#ifdef IN1_PIPE
#include "ttnn/operations/matmul/device/kernels/dataflow/prefetcher_pipe_in1_window.hpp"
#endif

// Gathers width(K)-sharded A onto every core via two-hub gather/broadcast. Shared by the full and the
// partial width-sharded factories; for partial a weight core's slab is [Kc, Nc] rather than [K, Nc].
//
// IN0_REPLICATED: A is ROW_MAJOR HEIGHT_SHARDED and already holds the full [M, K] on this core.
// Fill full_in0 from the local shard as 1x32 tiles (K-major) instead of the hub gather.
//
// IN0_ROW_RESTRIPE: A is ROW_MAJOR width-sharded with several rows. The gather lands sender-major
// ([sender][M][K / senders] 1x32 tiles) in gather_in0, and each compute core copies it into
// full_in0 as [M][K], the whole-K rows custom_mm takes one at a time.
//
// IN1_RECEIVER marks the weight cores, where compute runs; the rest of the bounding box only takes
// part in the gather, and a drain kernel consumes the A it receives. On a weight core in1 (weights) arrive one of three
// ways:
//   - default: in1 borrows the L1-resident weight shard, so the reader only has to declare its tiles
//     available. A packed weight starts in1_tile_offset tiles into the shard; compute skips those.
//   - IN1_FROM_DRAM: the weight is DRAM ND-sharded, one [K, N/num_weight_cores] slab per weight
//     core. The reader issues the slab's read before the gather so the two overlap, and waits for it
//     before publishing full_in0: compute takes in1 only after full_in0, so that gates it.
//   - IN1_PIPE: the tensor prefetcher streams this core's [K, N/num_weight_cores] slab over a
//     PrefetcherPipe as num_k_blocks entries, each a whole number of K-rows, and in1 is a relay over
//     the pipe's ring. An entry goes back to the prefetcher only once compute has popped its tiles.
// OUT_CONSUMER: no writer drains compute's output, so the reader waits for it, giving the buffer
// the consumer every buffer needs.
void kernel_main() {
    constexpr uint32_t shard_num_tiles = get_arg(args::shard_num_tiles);
    constexpr uint32_t tile_size_bytes = get_arg(args::tile_size_bytes);
    constexpr uint32_t num_senders = get_arg(args::num_senders);
    constexpr uint32_t num_receivers = get_arg(args::num_receivers);
    constexpr uint32_t hub0_noc_x = get_arg(args::hub0_noc_x);
    constexpr uint32_t hub0_noc_y = get_arg(args::hub0_noc_y);
    constexpr uint32_t hub1_noc_x = get_arg(args::hub1_noc_x);
    constexpr uint32_t hub1_noc_y = get_arg(args::hub1_noc_y);
    constexpr uint32_t split_H = get_arg(args::split_H);
    constexpr uint32_t in0_M_tiles = get_arg(args::in0_M_tiles);
    constexpr uint32_t in0_K_tiles = get_arg(args::in0_K_tiles);
#ifdef IN1_RECEIVER
    constexpr uint32_t in1_page_tiles = get_arg(args::in1_page_tiles);
#ifdef IN1_PIPE
    constexpr uint32_t num_k_blocks = get_arg(args::num_k_blocks);
#else
    constexpr uint32_t in1_tile_offset = get_arg(args::in1_tile_offset);
#endif
#ifdef OUT_CONSUMER
    constexpr uint32_t out_num_tiles = get_arg(args::out_num_tiles);
#endif
#endif

    const uint32_t is_sender = get_arg(args::is_sender);
    const uint32_t sender_id = get_arg(args::sender_id);
    const uint32_t role = get_arg(args::role);

#ifdef IN0_REPLICATED
    constexpr uint32_t full_num_tiles = in0_M_tiles * in0_K_tiles;
#else
    constexpr uint32_t full_num_tiles = num_senders * shard_num_tiles;
#endif
    const uint32_t shard_size_bytes = shard_num_tiles * tile_size_bytes;

    // The two hubs are the corners of the multicast rectangle. NOC_1 uses an inverted coordinate system.
    uint32_t mcast_x_start = hub0_noc_x;
    uint32_t mcast_y_start = hub0_noc_y;
    uint32_t mcast_x_end = hub1_noc_x;
    uint32_t mcast_y_end = hub1_noc_y;
    if (noc_index == 1) {
        std::swap(mcast_x_start, mcast_x_end);
        std::swap(mcast_y_start, mcast_y_end);
    }

    Noc noc;
    const LocalTensorAccessor<uint32_t> in0(tensor::in0);
    CircularBuffer full_in0_cb(dfb::full_in0);
    Semaphore<> stage_sem(sem::stage);
    Semaphore<> done_sem(sem::done);
    UnicastEndpoint hub;

#ifdef IN1_RECEIVER
    CircularBuffer in1_cb(dfb::in1);
#ifdef IN1_PIPE
    // One accessor names every pipe; the one whose receiver is this core is the one bound here.
    experimental::PrefetcherPipe pipe(pipe::in1);
    auto in1_relay = pipe.bind_relay();
    // Publish the first K-block before the gather so the prefetcher's transfer overlaps it.
    publish_in1_block_from_pipe(pipe, in1_relay, noc, 0, in1_page_tiles);
#elif defined(IN1_FROM_DRAM)
    const auto in1 = TensorAccessor(tensor::in1);
    in1_cb.reserve_back(in1_page_tiles);
    noc_async_read_shard(get_arg(args::in1_shard_id), in1, in1_cb.get_write_ptr());
#else
    in1_cb.reserve_back(in1_tile_offset + in1_page_tiles);
    in1_cb.push_back(in1_tile_offset + in1_page_tiles);
#endif
#endif
    full_in0_cb.reserve_back(full_num_tiles);

#ifdef IN0_REPLICATED
    const uint32_t in0_read_addr = static_cast<uint32_t>(local_mem::detail::AddressAccess::address(in0.local_mem()));
#ifdef USE_CUSTOM_MM
    // The replicated row-major shard is already M-major in 1x32 faces, which is
    // the layout consumed by custom_mm (one complete K row per invocation).
    noc_async_write(in0_read_addr, get_noc_addr(full_in0_cb.get_write_ptr()), full_num_tiles * tile_size_bytes);
    noc.async_write_barrier();
#else
    // General matmul_block consumes the replicated input in K-major order.
    {
        const uint32_t dst_base = full_in0_cb.get_write_ptr();
        for (uint32_t k = 0; k < in0_K_tiles; ++k) {
            for (uint32_t m = 0; m < in0_M_tiles; ++m) {
                const uint32_t src = in0_read_addr + (m * in0_K_tiles + k) * tile_size_bytes;
                const uint32_t dst = dst_base + (k * in0_M_tiles + m) * tile_size_bytes;
                noc_async_write(src, get_noc_addr(dst), tile_size_bytes);
            }
        }
        noc.async_write_barrier();
    }
#endif
#if defined(IN1_RECEIVER) && defined(IN1_FROM_DRAM)
    noc.async_read_barrier();
    in1_cb.push_back(in1_page_tiles);
#endif
    full_in0_cb.push_back(full_num_tiles);
#else
#ifdef IN0_ROW_RESTRIPE
    CircularBuffer gather_cb(dfb::gather_in0);
#else
    CircularBuffer& gather_cb = full_in0_cb;
#endif
    const bool is_hub0 = (role == 1);
    const bool is_hub1 = (role == 2);
    if (is_sender) {
        const bool owned_by_hub0 = sender_id < split_H;
        const uint32_t hub_x = owned_by_hub0 ? hub0_noc_x : hub1_noc_x;
        const uint32_t hub_y = owned_by_hub0 ? hub0_noc_y : hub1_noc_y;
        const uint32_t dst_offset_bytes = sender_id * shard_size_bytes;

        // gather_cb is at the same L1 offset on every core, so the local write ptr is the remote dst addr.
        const uint32_t dst_l1_addr = gather_cb.get_write_ptr() + dst_offset_bytes;
        noc.async_write(
            in0, hub, shard_size_bytes, {.offset_bytes = 0}, {.noc_x = hub_x, .noc_y = hub_y, .addr = dst_l1_addr});
        noc.async_write_barrier();
        stage_sem.up(noc, hub_x, hub_y, 1);
        noc.async_atomic_barrier();
    }

    if (is_hub0 || is_hub1) {
        const uint32_t region_first = is_hub0 ? 0 : split_H;
        const uint32_t region_count = is_hub0 ? split_H : (num_senders - split_H);
        const uint32_t region_offset_bytes = region_first * shard_size_bytes;
        const uint32_t region_size_bytes = region_count * shard_size_bytes;

        if (region_count > 0) {
            stage_sem.wait(region_count);

            noc.async_write_multicast<NocOptions::MCAST_INCL_SRC>(
                use<CircularBuffer::AddrSelector::WRITE_PTR>(gather_cb),
                gather_cb,
                region_size_bytes,
                num_receivers,
                {.offset_bytes = region_offset_bytes},
                {.noc_x_start = mcast_x_start,
                 .noc_y_start = mcast_y_start,
                 .noc_x_end = mcast_x_end,
                 .noc_y_end = mcast_y_end,
                 .offset_bytes = region_offset_bytes});
            noc.async_write_barrier();
        }

        // inc_multicast excludes the sender; self must use atomic NOC up() (local up() can race with the other hub).
        const uint32_t self_noc_x = is_hub0 ? hub0_noc_x : hub1_noc_x;
        const uint32_t self_noc_y = is_hub0 ? hub0_noc_y : hub1_noc_y;
        done_sem.inc_multicast(noc, mcast_x_start, mcast_y_start, mcast_x_end, mcast_y_end, 1, num_receivers - 1);
        done_sem.up(noc, self_noc_x, self_noc_y, 1);
        noc.async_atomic_barrier();
    }

    done_sem.wait(2);
#if defined(IN0_ROW_RESTRIPE) && defined(IN1_RECEIVER)
    // Only the weight cores run compute; the rest of the bounding box just relays the multicast.
    {
        constexpr uint32_t k_tiles_per_sender = shard_num_tiles / in0_M_tiles;
        constexpr uint32_t row_chunk_bytes = k_tiles_per_sender * tile_size_bytes;
        const uint32_t src_base = gather_cb.get_write_ptr();
        const uint32_t dst_base = full_in0_cb.get_write_ptr();
        for (uint32_t s = 0; s < num_senders; ++s) {
            for (uint32_t m = 0; m < in0_M_tiles; ++m) {
                const uint32_t src = src_base + s * shard_size_bytes + m * row_chunk_bytes;
                const uint32_t dst = dst_base + (m * in0_K_tiles + s * k_tiles_per_sender) * tile_size_bytes;
                noc_async_write(src, get_noc_addr(dst), row_chunk_bytes);
            }
        }
        noc.async_write_barrier();
    }
#endif
#if defined(IN1_RECEIVER) && defined(IN1_FROM_DRAM)
    noc.async_read_barrier();
    in1_cb.push_back(in1_page_tiles);
#endif
    full_in0_cb.push_back(full_num_tiles);
#endif  // IN0_REPLICATED

#if defined(IN1_RECEIVER) && defined(IN1_PIPE)
    for (uint32_t block = 1; block < num_k_blocks; ++block) {
        publish_in1_block_from_pipe(pipe, in1_relay, noc, block, in1_page_tiles);
    }
    drain_in1_pipe_window(pipe, noc, num_k_blocks);
    // pop_front acks the prefetcher with non-posted NOC atomics; retire them before the kernel exits.
    noc.async_atomic_barrier();
#endif

#if defined(IN1_RECEIVER) && defined(OUT_CONSUMER)
    CircularBuffer out_cb(dfb::out);
    out_cb.wait_front(out_num_tiles);
    out_cb.pop_front(out_num_tiles);
#endif

    noc.async_write_barrier();
    noc.async_read_barrier();
}
