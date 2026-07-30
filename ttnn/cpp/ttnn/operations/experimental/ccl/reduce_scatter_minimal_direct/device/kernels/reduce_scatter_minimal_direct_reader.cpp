// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/core_local_mem.h"

#include <cstdint>
#include <algorithm>

using address_t = uint32_t;

// Direct (one-shot) reduce-scatter reader, CB producer, no fabric. Two phases:
//
//   1. SEND phase: for every remote destination j (in the writer's send order, farthest ring distance
//      first), read this core's tile range of local input slice j into cb_send. The writer drains it
//      straight onto the fabric as a multi-hop unicast into device j's staging.
//   2. REDUCE phase: wait until every other device's contribution to OUR slice has landed (one atomic
//      inc per source, fused onto that source's last payload packet -- per-source counters, so an
//      absolute wait cannot be satisfied by a neighbour that has raced ahead), then feed the reducer
//      num_devices blocks per chunk: block 0 = our own local slice read from the input tensor,
//      blocks 1..N-1 = the other devices' contributions read out of staging (one coalesced read each).
//      Our own block is read BEFORE the arrival waits (it depends on no arrival), so it lands for free
//      while we spin instead of adding a DRAM round trip after the last packet arrives.
//
// Staging is double-buffered by invocation parity: a device that is one invocation ahead of us writes
// into the other half, so it cannot clobber data we have not reduced yet. Two invocations ahead is
// impossible (it would require our previous program to have completed). Parity comes from this core's
// private invocation counter (`gen`), which this reader owns and bumps at the end; the writer keeps its
// own identical counter, so the two kernels always agree without any cross-kernel handshake.
void kernel_main() {
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(0);        // single tile bytes (real dtype)
    constexpr uint32_t tile_granularity = get_compile_time_arg_val(1);  // tiles per chunk
    constexpr uint32_t chunks_per_slice = get_compile_time_arg_val(2);
    // Whole-slice tile count; this core's tile budget arrives as a runtime arg (tile_count) instead.
    [[maybe_unused]] constexpr uint32_t pages_per_slice = get_compile_time_arg_val(3);
    // Slice walk in page space: a slice is `slice_run_pages` contiguous input pages every `stride_pages`.
    // Scattering the last dim gives run = the slice's width in tiles and stride = the full row; any other
    // dim just changes these two numbers (see ReduceScatterDirectGeometry on the host side).
    constexpr uint32_t slice_run_pages = get_compile_time_arg_val(4);
    constexpr uint32_t stride_pages = get_compile_time_arg_val(5);
    constexpr uint32_t num_devices = get_compile_time_arg_val(6);
    constexpr uint32_t cb_send_id = get_compile_time_arg_val(7);
    constexpr uint32_t cb_reduce_id = get_compile_time_arg_val(8);
    // Arrivals land directly in cb_reduce (staging is this core's L1 shard, aliased into the CB), so the
    // whole staging readback below disappears -- see the factory's CB comment.
    constexpr bool arrivals_in_cb = get_compile_time_arg_val(9) != 0;
    constexpr uint32_t half_stride_tiles = get_compile_time_arg_val(10);  // tiles in one parity half
    constexpr auto input_args = TensorAccessorArgs<11>();
    constexpr auto staging_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

    constexpr uint32_t num_dests = num_devices - 1;  // remote destinations == remote sources

    size_t ai = 0;
    const address_t input_addr = get_arg_val<address_t>(ai++);
    const address_t staging_addr = get_arg_val<address_t>(ai++);
    const uint32_t device_idx = get_arg_val<uint32_t>(ai++);
    // This core's partition: chunks [chunk_start, chunk_start + chunk_count) == tiles
    // [tile_start, tile_start + tile_count) of every slice.
    const uint32_t chunk_start = get_arg_val<uint32_t>(ai++);
    const uint32_t chunk_count = get_arg_val<uint32_t>(ai++);
    const uint32_t tile_start = get_arg_val<uint32_t>(ai++);
    const uint32_t tile_count = get_arg_val<uint32_t>(ai++);
    const address_t gen_addr = get_arg_val<uint32_t>(ai++);  // this core's private invocation counter
    // Arrival counters, one per remote source (never reset -- see the parity note above). Followed by
    // the send-order destination slice list, shared with the writer.
    const size_t arrival_sems = ai;
    ai += num_dests;
    const size_t dest_slices = ai;
    ai += num_dests;

    auto input_acc = TensorAccessor(input_args, input_addr);  // tiled (page = tile_bytes)

    Noc noc;
    CircularBuffer cb_send(cb_send_id);
    CircularBuffer cb_reduce(cb_reduce_id);

    auto* gen_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gen_addr);
    const uint32_t invocation = *gen_ptr;
    // Staging half for this invocation (see the parity note above), in whichever unit the path needs:
    // interleaved staging indexes by page, the aliased CB by a constant tile-index offset.
    const uint32_t staging_half = (invocation & 1u) * (num_devices * chunks_per_slice);
    const uint32_t half_off_bytes = (invocation & 1u) * half_stride_tiles * tile_bytes;

    // Where this core's tile range starts in the per-slice input walk: tile tile_start of a slice sits
    // in run tile_start/slice_run_pages, at offset tile_start%slice_run_pages (loop-invariant, hoisted).
    const uint32_t in_run_off_init = (tile_start / slice_run_pages) * stride_pages;
    const uint32_t in_page_in_run_init = tile_start % slice_run_pages;

    // Reads this core's tile range of local input slice `j`, chunk `k`, into L1 at `l1` (per tile: the
    // input is bank-interleaved and a slice is strided). Walk state is carried by the caller.
    auto read_local_chunk = [&](uint32_t j, uint32_t& run_off, uint32_t& page_in_run, uint32_t l1, uint32_t tiles) {
        const uint32_t tile_id_start = j * slice_run_pages;
        for (uint32_t t = 0; t < tiles; ++t) {
            const uint32_t tid = tile_id_start + run_off + page_in_run;
            if (++page_in_run == slice_run_pages) {
                run_off += stride_pages;
                page_in_run = 0;
            }
            noc.async_read(input_acc, CoreLocalMem<uint32_t>(l1), tile_bytes, {.page_id = tid}, {}, {});
            l1 += tile_bytes;
        }
    };

    // ---- Phase 1: local input slices -> cb_send (one slice per remote destination, send order) ----
    // CB granularity is fixed at tile_granularity (a chunk always occupies a full tg-page slot so slots
    // stay aligned and no read ever wraps the CB); only tiles_in_chunk tiles are valid in a partial chunk.
    for (uint32_t dst = 0; dst < num_dests; ++dst) {
        const uint32_t j = get_arg_val<uint32_t>(dest_slices + dst);
        uint32_t run_off = in_run_off_init;
        uint32_t page_in_run = in_page_in_run_init;
        uint32_t tiles_done = 0;
        for (uint32_t k = 0; k < chunk_count; ++k) {
            const uint32_t tiles_in_chunk = std::min(tile_granularity, tile_count - tiles_done);
            cb_send.reserve_back(tile_granularity);
            read_local_chunk(j, run_off, page_in_run, cb_send.get_write_ptr(), tiles_in_chunk);
            noc.async_read_barrier();
            cb_send.push_back(tile_granularity);
            tiles_done += tiles_in_chunk;
        }
    }

    // ---- Phase 2: feed the reducer num_devices blocks per chunk ----
    {
        uint32_t run_off = in_run_off_init;
        uint32_t page_in_run = in_page_in_run_init;
        uint32_t tiles_done = 0;
        for (uint32_t k = 0; k < chunk_count; ++k) {
            const uint32_t tiles_in_chunk = std::min(tile_granularity, tile_count - tiles_done);
            cb_reduce.reserve_back(num_devices * tile_granularity);
            // On the aliased path the CB write pointer walks half 0, and the parity half is a constant
            // byte offset on top of it -- the same offset compute applies to its tile indices.
            const uint32_t base = cb_reduce.get_write_ptr() + half_off_bytes;

            // block 0: our own contribution, straight out of the input tensor. Issued before the arrival
            // waits below so it lands while we spin.
            read_local_chunk(device_idx, run_off, page_in_run, base, tiles_in_chunk);

            if (k == 0) {
                // Every remote contribution to our own slice has landed (whole tile range, all chunks).
                for (uint32_t s = 0; s < num_dests; ++s) {
                    auto* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(arrival_sems + s));
                    noc_semaphore_wait_min(sem, invocation + 1);
                }
            }

            // blocks 1..N-1: the remote contributions. On the aliased path the senders already wrote them
            // into these exact CB slots, so there is nothing to do -- which is the point of that path:
            // these reads sit strictly AFTER the last arrival, so even out of L1 they are N-1 NoC round
            // trips of pure post-gate latency.
            if constexpr (!arrivals_in_cb) {
                auto staging_acc = TensorAccessor(staging_args, staging_addr);  // chunk-paged
                uint32_t block = 1;
                for (uint32_t s = 0; s < num_devices; ++s) {
                    if (s == device_idx) {
                        continue;
                    }
                    noc.async_read(
                        staging_acc,
                        CoreLocalMem<uint32_t>(base + block * tile_granularity * tile_bytes),
                        tiles_in_chunk * tile_bytes,
                        {.page_id = staging_half + s * chunks_per_slice + chunk_start + k},
                        {},
                        {});
                    ++block;
                }
            }

            noc.async_read_barrier();
            cb_reduce.push_back(num_devices * tile_granularity);
            tiles_done += tiles_in_chunk;
        }
    }

    // This invocation is done consuming the arrival counters; advance our private generation so the next
    // one waits on absolute position invocation+2 (the counters themselves are deliberately never reset:
    // a device that has already moved on would have its increments silently destroyed).
    noc_semaphore_set(gen_ptr, invocation + 1);
}
