// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//
// Reader kernel for untilizer cores.
// Each untilizer core is permanently bound to ONE sender core (its owning sender).
// The owning sender multicasts the expert token counts + its receive_buf_addr to this untilizer
// core's group, then each untilizer core untilizes the tiles for its assigned expert batches.
//
// Expert range: every untilizer group now handles EVERY expert [expert_start_idx, expert_end_idx) =
// [0, experts_per_chip).  The work is split across ALL untilizer cores by DATA via one global
// round-robin: each core has a global interleaved position (rank-major, sender-minor:
// [S0.U0, S1.U0, S0.U1, S1.U1, …]) and processes batches global_pos, +G, +2G, … of every expert,
// where G = total untilizer cores.  Consecutive batches of an expert thus fan out across senders,
// so neither sender's forwarder gets a monopoly of local (or remote) rows however dispatch
// clustered them; a sender's batch share is proportional to its untilizer-core count.
//
// For each assigned batch:
//   1. Read this batch's metadata pages from DRAM into cb_metadata_batch_id (consumed by
//      writer_untilize).
//   2. TILE_LAYOUT: read dispatched_buffer tiles into cb_dispatched_buffer_id (c_0) for the
//      compute kernel to untilize into cb_untilize_id (c_2).
//      ROW_MAJOR: dispatched_buffer is already row-major, so read its rows page-per-page
//      directly into cb_untilize_id (c_2); no compute kernel runs and c_0 is unused.
// Compute (untilize_combine, TILE only) and writer_untilize on this same core independently walk
// the same expert/batch loop using the multicasted counter in c_1, so no per-batch signal CB
// is needed — producer-consumer ordering on c_0 / c_2 / c_9 keeps everyone in lock-step.
//

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/debug/dprint.h"

#define ENABLE_COMBINE_DEBUG 0
#if ENABLE_COMBINE_DEBUG
#define DPRINT_COMBINE(...) DPRINT(__VA_ARGS__)
#else
#define DPRINT_COMBINE(...)
#endif

void kernel_main() {
    // ===== Compile-time args =====
    //   0: cb_experts_tok_counter_id             - CB that receives the multicasted expert token counts
    //   1: experts_tok_counter_pages             - number of pages in the expert_token_counts tensor
    //   2: experts_per_chip                      - number of experts assigned to this chip
    //   3: counter_offset                        - uint32_t offset into the token count buffer for this chip
    //   4: cb_dispatched_buffer_id               - CB for reading dispatched_buffer tiles
    //   5: cb_untilize_id                        - CB for untilized output produced by the paired compute kernel
    //   6: hidden_size                           - hidden dimension (e.g. 7168)
    //   7: read_batch_size                       - number of rows per untilize batch (e.g. 32)
    //   8: aligned_dispatched_buffer_page_size   - aligned page size of dispatched_buffer tensor
    //   9: tile_height                           - tile height in rows (e.g. 32)
    //  10: tile_width                            - tile width in columns (e.g. 32)
    //  11: max_dispatch_buffer_token_size        - total per-chip dispatch buffer capacity (overflow guard)
    //  12: aligned_output_page_size              - aligned page size of output tensor (bytes per untilized row)
    //  13: aligned_experts_tok_counter_page_size - aligned page size of expert_token_counts tensor
    //  14: cb_metadata_batch_id                  - CB this kernel pushes per-batch metadata pages into
    //                                              (consumed by writer_untilize on the same core)
    //  15: aligned_dispatched_metadata_page_size - aligned page size of dispatched_metadata tensor
    //  16: block_ct_dim                          - tiles per chunk pushed to cb_dispatched_buffer_id;
    //                                              matches the compute kernel's per-block consumption
    //                                              size so producer/consumer line up 1:1
    //  17: cb_counter_total_pages                - full page capacity of c_1 (counter + trailer)
    //                                              used for cb_reserve_back / cb_push_back / cb_wait_front
    //  18+: TensorAccessorArgs for dispatched_buffer, then TensorAccessorArgs for dispatched_metadata
    constexpr uint32_t cb_experts_tok_counter_id = get_compile_time_arg_val(0);
    CircularBuffer cb_experts_tok_counter(cb_experts_tok_counter_id);
    constexpr uint32_t experts_tok_counter_pages = get_compile_time_arg_val(1);
    constexpr uint32_t experts_per_chip = get_compile_time_arg_val(2);
    constexpr uint32_t counter_offset = get_compile_time_arg_val(3);
    constexpr uint32_t cb_dispatched_buffer_id = get_compile_time_arg_val(4);
    CircularBuffer cb_dispatched_buffer(cb_dispatched_buffer_id);
    constexpr uint32_t cb_untilize_id = get_compile_time_arg_val(5);
    constexpr uint32_t hidden_size = get_compile_time_arg_val(6);
    constexpr uint32_t read_batch_size = get_compile_time_arg_val(7);
    constexpr uint32_t aligned_dispatched_buffer_page_size = get_compile_time_arg_val(8);
    constexpr uint32_t tile_height = get_compile_time_arg_val(9);
    constexpr uint32_t tile_width = get_compile_time_arg_val(10);
    constexpr uint32_t max_dispatch_buffer_token_size = get_compile_time_arg_val(11);
    constexpr uint32_t aligned_output_page_size = get_compile_time_arg_val(12);
    constexpr uint32_t aligned_experts_tok_counter_page_size = get_compile_time_arg_val(13);
    constexpr uint32_t cb_metadata_batch_id = get_compile_time_arg_val(14);
    CircularBuffer cb_metadata_batch(cb_metadata_batch_id);
    constexpr uint32_t aligned_dispatched_metadata_page_size = get_compile_time_arg_val(15);
    constexpr uint32_t block_ct_dim = get_compile_time_arg_val(16);
    constexpr uint32_t cb_counter_total_pages = get_compile_time_arg_val(17);

    Noc noc;
    constexpr auto dispatched_buffer_args = TensorAccessorArgs<18>();
    constexpr auto dispatched_metadata_args =
        TensorAccessorArgs<dispatched_buffer_args.next_compile_time_args_offset()>();

    constexpr uint32_t tiles_per_batch = hidden_size / tile_width;

    // ===== Runtime args =====
    //   0: counter_ready_semaphore_id  - untilizer core waits for this before reading token counts
    //                                    (incremented by the owning sender after its multicast)
    //   1: dispatched_buffer_addr
    //   2: expert_start_idx
    //   3: expert_end_idx
    //   4: dispatched_metadata_addr    - DRAM base of the dispatched_metadata tensor; this kernel
    //                                    reads it locally so the sender no longer has to unicast
    //                                    per-batch metadata to this core
    //   5: untilizer_global_pos        - this core's position in the global interleaved untilizer
    //                                    ordering; its batches are global_pos, +G, +2G, … per expert
    //   6: total_untilizers            - G, total untilizer cores across all senders (global stride)
    //   7: routed_expert_sem_addr      - absolute L1 address of the routed-expert global semaphore
    //                                    used to overlap the routed expert with the combine.
    //                                    0 => not provided (no overlap) => skip the wait.
    // (sender NOC coords, data_ready and start semaphores are now consumed by the
    //  writer_untilize kernel on the same core — they no longer belong here.)
    uint32_t rt_idx = 0;
    uint32_t counter_ready_semaphore_id = get_arg_val<uint32_t>(rt_idx++);
    uint32_t dispatched_buffer_addr = get_arg_val<uint32_t>(rt_idx++);
    uint32_t expert_start_idx = get_arg_val<uint32_t>(rt_idx++);
    uint32_t expert_end_idx = get_arg_val<uint32_t>(rt_idx++);
    uint32_t dispatched_metadata_addr = get_arg_val<uint32_t>(rt_idx++);
    uint32_t untilizer_global_pos = get_arg_val<uint32_t>(rt_idx++);
    uint32_t total_untilizers = get_arg_val<uint32_t>(rt_idx++);
    uint32_t routed_expert_sem_addr = get_arg_val<uint32_t>(rt_idx++);
    // Routed-expert semaphore per-forward reset protocol (only meaningful when overlap is enabled,
    // i.e. routed_expert_sem_addr != 0). Every untilizer core signals the leader after its final
    // wait_min; the leader (is_reset_leader) collects all signals, then loopback-multicasts 0 to
    // every combine core's copy.
    uint32_t is_reset_leader = get_arg_val<uint32_t>(rt_idx++);
    uint32_t reset_leader_noc_x = get_arg_val<uint32_t>(rt_idx++);
    uint32_t reset_leader_noc_y = get_arg_val<uint32_t>(rt_idx++);
    uint32_t reset_barrier_sem_id = get_arg_val<uint32_t>(rt_idx++);
    uint32_t reset_num_untilizer_cores = get_arg_val<uint32_t>(rt_idx++);
    uint32_t reset_mcast_start_x = get_arg_val<uint32_t>(rt_idx++);
    uint32_t reset_mcast_start_y = get_arg_val<uint32_t>(rt_idx++);
    uint32_t reset_mcast_end_x = get_arg_val<uint32_t>(rt_idx++);
    uint32_t reset_mcast_end_y = get_arg_val<uint32_t>(rt_idx++);
    uint32_t reset_num_combine_cores = get_arg_val<uint32_t>(rt_idx++);

    // ===== Step 1: Wait for the owning sender to multicast expert token counts + receive_buf_addr =====
    // Note: don't reset counter_ready_sem — writer_untilize on this same core also waits on it
    // to read the sender's receive_buf_addr from c_1. Since neither kernel re-uses the sem within
    // a single invocation, leaving it latched at >=1 is safe.
    cb_experts_tok_counter.reserve_back(cb_counter_total_pages);

    Semaphore<> counter_ready_sem(counter_ready_semaphore_id);
    counter_ready_sem.wait(1);

    cb_experts_tok_counter.push_back(cb_counter_total_pages);

    // ===== Step 2: Read per-expert token counts =====
    // writer_untilize independently reads the sender's receive_buf_addr from c_1 at offset
    // experts_tok_counter_pages * aligned_experts_tok_counter_page_size (same L1 layout).
    cb_experts_tok_counter.wait_front(cb_counter_total_pages);
    uint32_t token_counter_base = cb_experts_tok_counter.get_read_ptr();
    const volatile tt_l1_ptr uint32_t* counter_l1_src =
        reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(token_counter_base) + counter_offset;
    uint32_t local_expert_counts[experts_per_chip];
    for (uint32_t e = 0; e < experts_per_chip; e++) {
        local_expert_counts[e] = counter_l1_src[e];
        DPRINT_COMBINE("Expert {}: tokens={}\n", e, counter_l1_src[e]);
    }

    // ===== Step 3: For each assigned batch: trigger compute to untilize and read tiles =====
    const auto dispatched_buffer_addr_gen =
        TensorAccessor(dispatched_buffer_args, dispatched_buffer_addr, aligned_dispatched_buffer_page_size);
    const auto dispatched_metadata_addr_gen =
        TensorAccessor(dispatched_metadata_args, dispatched_metadata_addr, aligned_dispatched_metadata_page_size);

    // Dynamic dispatch buffer: experts are packed contiguously, each occupying
    // ceil(local_expert_counts[e] / tile_height) tile rows. Compute the per-expert
    // tile-row offset as a running prefix sum over the local counts.  The metadata
    // tensor is laid out with the same tile-aligned per-expert stride (host computes
    // expert_region_offsets from cumsum of ceil(count, tile_h)*tile_h), so we can
    // reuse `start_token = (start_page_tiled / tiles_per_batch) * tile_height` below
    // as the per-expert metadata start page.
    uint32_t start_page_tiled = 0;
    for (uint32_t e = 0; e < expert_start_idx; e++) {
        start_page_tiled += ((local_expert_counts[e] + tile_height - 1) / tile_height) * tiles_per_batch;
    }

    for (uint32_t local_expert = expert_start_idx; local_expert < expert_end_idx; local_expert++) {
        // DeviceZoneScopedN("combine_expert_iter");  // per-iteration: this expert's combine work
        // Overlap handshake with the routed expert: before processing this expert, wait until the
        // routed-expert global semaphore reaches a value >= local_expert + 1.
        if (routed_expert_sem_addr != 0) {
            // DeviceZoneScopedN("combine_overlap_handshake");  // per-expert wait on routed-expert semaphore
            volatile tt_l1_ptr uint32_t* routed_expert_sem_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(routed_expert_sem_addr);
            noc_semaphore_wait_min(routed_expert_sem_ptr, local_expert + 1);
            // After clearing the final expert's wait, this core is past ALL of its routed-expert
            // semaphore waits.
            if (local_expert == expert_end_idx - 1) {
                const uint64_t leader_barrier_noc_addr =
                    get_noc_addr(reset_leader_noc_x, reset_leader_noc_y, get_semaphore(reset_barrier_sem_id));
                noc_semaphore_inc(leader_barrier_noc_addr, 1);
                noc_async_atomic_barrier();
            }
        }

        uint32_t expert_tokens = local_expert_counts[local_expert];
        // Clamp to the dispatch buffer capacity to mirror reader_dispatch's overflow guard.
        // start_page_tiled is in tiles; convert via tile_height to compare with the row-count cap.
        // start_token doubles as this expert's metadata start page since dispatch lays the
        // metadata out with tile-aligned per-expert stride (same as expert_region_offsets).
        uint32_t start_token = (start_page_tiled / tiles_per_batch) * tile_height;
        if (start_token >= max_dispatch_buffer_token_size) {
            expert_tokens = 0;
        } else if (start_token + expert_tokens > max_dispatch_buffer_token_size) {
            expert_tokens = max_dispatch_buffer_token_size - start_token;
        }
        DPRINT_COMBINE("Expert {}: tokens={}\n", local_expert, expert_tokens);

        uint32_t actual_batches = (expert_tokens + read_batch_size - 1) / read_batch_size;

        // Global round-robin: this core handles batches untilizer_global_pos, +G, +2G, … of every
        // expert (G = total_untilizers across all senders) — disjoint across cores, covering
        // [0, actual_batches) exactly.  Must match writer_untilize / compute exactly so the
        // c_0 / c_2 producer-consumer protocol stays in lockstep.
        for (uint32_t batch_idx = untilizer_global_pos; batch_idx < actual_batches; batch_idx += total_untilizers) {
            uint32_t batch_tile_start = start_page_tiled + batch_idx * tiles_per_batch;
            uint32_t batch_token_start = batch_idx * read_batch_size;
            uint32_t batch_count = ((batch_token_start + read_batch_size) <= expert_tokens)
                                       ? read_batch_size
                                       : (expert_tokens - batch_token_start);

            // Compute and writer_untilize independently walk this same expert/batch loop
            // (reading the multicasted counter from c_1) — no per-batch signal needed.
            //
            // 1. Read this batch's metadata pages from DRAM into the local metadata batch CB.
            //    writer_untilize pops `batch_count` pages and decides the per-batch path
            //    (all-local vs non-local) locally — sender no longer writes to c_9.
            //    Per-expert metadata stride is tile-aligned, hence reusing start_token.

            {
                uint32_t metadata_batch_start = start_token + batch_token_start;
                // Always reserve/push read_batch_size pages so cb_metadata_batch_id wraps
                // cleanly (tt-metal CBs require fifo_wr_ptr to hit fifo_limit exactly to
                // wrap).  Only the first batch_count pages contain valid metadata read from
                // DRAM; the trailing (read_batch_size - batch_count) pages are unused and
                // will not be read by the consumer.
                cb_metadata_batch.reserve_back(read_batch_size);
                {
                    // DeviceZoneScopedN("METADATA-read");
                    for (uint32_t t = 0; t < batch_count; t++) {
                        noc.async_read(
                            dispatched_metadata_addr_gen,
                            cb_metadata_batch,
                            aligned_dispatched_metadata_page_size,
                            {.page_id = metadata_batch_start + t},
                            {.offset_bytes = t * aligned_dispatched_metadata_page_size});
                    }
                    noc.async_read_barrier();
                }
                cb_metadata_batch.push_back(read_batch_size);
            }

#if IS_TILE_LAYOUT
            // 2. TILE: read tiles for this batch in block_ct_dim-sized chunks so each push matches
            //    one compute-side pack_untilize_block consumption of block_ct_dim tiles.  The CB
            //    write pointer advances with every push, so re-fetch it after each
            //    `cb_reserve_back` — capturing it once before the loop lands every chunk in
            //    the same slot and leaves the other slot uninitialized.  The paired compute
            //    kernel untilizes cb_dispatched_buffer_id (c_0) -> cb_untilize_id (c_2).
            {
                constexpr uint32_t num_blocks = tiles_per_batch / block_ct_dim;
                for (uint32_t cnt = 0; cnt < num_blocks; cnt++) {
                    uint32_t batch_tile = batch_tile_start + cnt * block_ct_dim;
                    cb_dispatched_buffer.reserve_back(block_ct_dim);
                    {
                        // DeviceZoneScopedN("DISPATCHED-BUFFER-read");
                        for (uint32_t t = 0; t < block_ct_dim; t++) {
                            noc.async_read(
                                dispatched_buffer_addr_gen,
                                cb_dispatched_buffer,
                                aligned_dispatched_buffer_page_size,
                                {.page_id = batch_tile + t},
                                {.offset_bytes = t * aligned_dispatched_buffer_page_size});
                        }
                        noc.async_read_barrier();
                    }
                    cb_dispatched_buffer.push_back(block_ct_dim);
                }
                // Steps 3-7 (wait for untilize, wait for sender's send signal, NOC-write to
                // sender, signal sender, pop untilize CB) now run on writer_untilize.
            }
#else
            // 2. ROW_MAJOR: dispatched_buffer is already row-major, so there is no untilization to
            //    do.  Read the batch's rows page-per-page straight into cb_untilize_id (c_2) — the
            //    same CB the compute kernel fills in the TILE path — so writer_untilize consumes it
            //    identically.  No compute kernel runs and cb_dispatched_buffer_id (c_0) is unused.
            //    start_token is this expert's (tile-aligned) start row, == expert_region_offsets[e];
            //    batch_token_start offsets to this batch within the expert.  Always reserve/push
            //    read_batch_size pages so c_2 wraps cleanly (matching writer_untilize's
            //    cb_wait_front(cb_untilize_id, read_batch_size)); only the first batch_count pages
            //    hold valid rows.
            {
                uint32_t batch_row_start = start_token + batch_token_start;
                cb_reserve_back(cb_untilize_id, read_batch_size);
                uint32_t untilize_base = get_write_ptr(cb_untilize_id);
                {
                    // DeviceZoneScopedN("DISPATCHED-BUFFER-row-read");
                    for (uint32_t t = 0; t < batch_count; t++) {
                        noc_async_read_page(
                            batch_row_start + t,
                            dispatched_buffer_addr_gen,
                            untilize_base + t * aligned_output_page_size);
                    }
                    noc_async_read_barrier();
                }
                cb_push_back(cb_untilize_id, read_batch_size);
            }
#endif
        }
        // Advance to the next expert's region.  Buffer and metadata both use the same
        // tile-aligned per-expert stride, so start_page_tiled (and start_token derived from
        // it) tracks both for the next iteration.
        start_page_tiled += ((expert_tokens + tile_height - 1) / tile_height) * tiles_per_batch;
    }

    // Reset of the routed-expert overlap semaphore, performed once by a single leader
    // untilizer core.
    if (routed_expert_sem_addr != 0 && is_reset_leader != 0) {
        volatile tt_l1_ptr uint32_t* routed_expert_sem_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(routed_expert_sem_addr);
        volatile tt_l1_ptr uint32_t* reset_barrier_sem_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(reset_barrier_sem_id));
        noc_semaphore_wait(reset_barrier_sem_ptr, reset_num_untilizer_cores);
        noc_semaphore_set(reset_barrier_sem_ptr, 0);
        // Zero this (leader's) copy so it can serve as the 0 source for the broadcast, then
        // loopback-multicast 0 to every combine core's copy (num_dests counts self for loopback).
        noc_semaphore_set(routed_expert_sem_ptr, 0);
        if (reset_num_combine_cores > 1) {
            const uint64_t reset_mcast_noc_addr = get_noc_multicast_addr(
                reset_mcast_start_x, reset_mcast_start_y, reset_mcast_end_x, reset_mcast_end_y, routed_expert_sem_addr);
            noc_semaphore_set_multicast_loopback_src(
                routed_expert_sem_addr, reset_mcast_noc_addr, reset_num_combine_cores);
        }
        noc_async_write_barrier();
    }
}
