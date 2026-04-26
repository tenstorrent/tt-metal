// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//
// Reader kernel for idle cores.
// Each idle core is permanently bound to ONE sender core (its owning sender).
// The owning sender multicasts the expert token counts + its receive_buf_addr to this idle
// core's group, then each idle core untilizes the tiles for its assigned expert batches.
//
// Expert range: each idle group handles experts [expert_start_idx, expert_end_idx) which maps
// 1:1 to the owning sender's expert range.
//
// Batch splitting within the group: batches are distributed round-robin across the k_s idle
// cores in the group.  Core i (local 0-based) processes batches i, i+k_s, i+2*k_s, …
//
// For each assigned batch:
//   1. Signal compute to start untilizing this batch (via cb_signal_id).
//   2. Read dispatched_buffer tiles into cb_dispatched_buffer_id for compute to consume.
// The actual send of untilized data to the sender (former steps 3-7) now runs on the
// zero_init_writer kernel on the same core, driven by compute via cb_stop_signal_id.
//

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"

#define ENABLE_COMBINE_DEBUG 0
#if ENABLE_COMBINE_DEBUG
#define DPRINT_COMBINE DPRINT
#else
#define DPRINT_COMBINE \
    if (0)             \
    DebugPrinter()
#endif

// Signal last element to compute to break out of loop
constexpr uint32_t ROUTE_INFO_SENTINEL = 0xFFFFFFFF;

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
    //   8: cb_signal_id                          - CB for reader->compute signalling (one page per batch)
    //   9: aligned_dispatched_buffer_page_size   - aligned page size of dispatched_buffer tensor
    //  10: cb_factor                             - CB size reduction factor (1 for Blackhole, 4 for Wormhole)
    //  11: tile_height                            - tile height in rows (e.g. 32)
    //  12: tile_width                             - tile width in columns (e.g. 32)
    //  13: max_dispatch_buffer_token_size        - total per-chip dispatch buffer capacity (overflow guard)
    //  14: core_id                               - local index within the owning sender's idle group (0-based)
    //  15: num_idle_cores                        - size of the owning sender's idle group (k_s)
    //  16: aligned_output_page_size              - aligned page size of output tensor (bytes per untilized row)
    //  17: aligned_experts_tok_counter_page_size - aligned page size of expert_token_counts tensor
    //  18+: TensorAccessorArgs for dispatched_buffer
    constexpr uint32_t cb_experts_tok_counter_id = get_compile_time_arg_val(0);
    constexpr uint32_t experts_tok_counter_pages = get_compile_time_arg_val(1);
    constexpr uint32_t experts_per_chip = get_compile_time_arg_val(2);
    constexpr uint32_t counter_offset = get_compile_time_arg_val(3);
    constexpr uint32_t cb_dispatched_buffer_id = get_compile_time_arg_val(4);
    constexpr uint32_t cb_untilize_id = get_compile_time_arg_val(5);
    constexpr uint32_t hidden_size = get_compile_time_arg_val(6);
    constexpr uint32_t read_batch_size = get_compile_time_arg_val(7);
    constexpr uint32_t cb_signal_id = get_compile_time_arg_val(8);
    constexpr uint32_t aligned_dispatched_buffer_page_size = get_compile_time_arg_val(9);
    constexpr uint32_t cb_factor = get_compile_time_arg_val(10);
    constexpr uint32_t tile_height = get_compile_time_arg_val(11);
    constexpr uint32_t tile_width = get_compile_time_arg_val(12);
    constexpr uint32_t max_dispatch_buffer_token_size = get_compile_time_arg_val(13);
    constexpr uint32_t core_id = get_compile_time_arg_val(14);
    constexpr uint32_t num_idle_cores = get_compile_time_arg_val(15);
    constexpr uint32_t aligned_output_page_size = get_compile_time_arg_val(16);
    constexpr uint32_t aligned_experts_tok_counter_page_size = get_compile_time_arg_val(17);
    constexpr auto dispatched_buffer_args = TensorAccessorArgs<18>();

    constexpr uint32_t tiles_per_batch = hidden_size / tile_width;

    // ===== Runtime args =====
    //   0: counter_ready_semaphore_id  - idle core waits for this before reading token counts
    //                                    (incremented by the owning sender after its multicast)
    //   1: dispatched_buffer_addr
    //   2: expert_start_idx
    //   3: expert_end_idx
    // (sender NOC coords, data_ready and start semaphores are now consumed by the
    //  zero_init_writer kernel on the same core — they no longer belong here.)
    uint32_t rt_idx = 0;
    uint32_t counter_ready_semaphore_id = get_arg_val<uint32_t>(rt_idx++);
    uint32_t dispatched_buffer_addr = get_arg_val<uint32_t>(rt_idx++);
    uint32_t expert_start_idx = get_arg_val<uint32_t>(rt_idx++);
    uint32_t expert_end_idx = get_arg_val<uint32_t>(rt_idx++);

    // ===== Step 1: Wait for the owning sender to multicast expert token counts + receive_buf_addr =====
    // Note: don't reset counter_ready_sem — zero_init_writer on this same core also waits on it
    // to read the sender's receive_buf_addr from c_1. Since neither kernel re-uses the sem within
    // a single invocation, leaving it latched at >=1 is safe.
    cb_reserve_back(cb_experts_tok_counter_id, experts_tok_counter_pages);

    volatile tt_l1_ptr uint32_t* counter_ready_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(counter_ready_semaphore_id));
    noc_semaphore_wait(counter_ready_sem_ptr, 1);

    cb_push_back(cb_experts_tok_counter_id, experts_tok_counter_pages);

    // ===== Step 2: Read per-expert token counts =====
    // zero_init_writer independently reads the sender's receive_buf_addr from c_1 at offset
    // experts_tok_counter_pages * aligned_experts_tok_counter_page_size (same L1 layout).
    cb_wait_front(cb_experts_tok_counter_id, experts_tok_counter_pages);
    uint32_t token_counter_base = get_read_ptr(cb_experts_tok_counter_id);
    const volatile tt_l1_ptr uint32_t* counter_l1_src =
        reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(token_counter_base) + counter_offset;
    uint32_t local_expert_counts[experts_per_chip];
    for (uint32_t e = 0; e < experts_per_chip; e++) {
        local_expert_counts[e] = counter_l1_src[e];
        DPRINT_COMBINE << "Expert " << e << ": tokens=" << counter_l1_src[e] << ENDL();
    }

    // ===== Step 3: For each assigned batch: trigger compute to untilize and read tiles =====
    const auto dispatched_buffer_addr_gen =
        TensorAccessor(dispatched_buffer_args, dispatched_buffer_addr, aligned_dispatched_buffer_page_size);
    uint32_t buffer_base = get_write_ptr(cb_dispatched_buffer_id);

    // Dynamic dispatch buffer: experts are packed contiguously, each occupying
    // ceil(local_expert_counts[e] / tile_height) tile rows. Compute the per-expert
    // tile-row offset as a running prefix sum over the local counts.
    uint32_t start_page_tiled = 0;
    for (uint32_t e = 0; e < expert_start_idx; e++) {
        start_page_tiled += ((local_expert_counts[e] + tile_height - 1) / tile_height) * tiles_per_batch;
    }

    for (uint32_t local_expert = expert_start_idx; local_expert < expert_end_idx; local_expert++) {
        uint32_t expert_tokens = local_expert_counts[local_expert];
        // Clamp to the dispatch buffer capacity to mirror reader_dispatch's overflow guard.
        // start_page_tiled is in tiles; convert via tile_height to compare with the row-count cap.
        uint32_t start_token = (start_page_tiled / tiles_per_batch) * tile_height;
        if (start_token >= max_dispatch_buffer_token_size) {
            expert_tokens = 0;
        } else if (start_token + expert_tokens > max_dispatch_buffer_token_size) {
            expert_tokens = max_dispatch_buffer_token_size - start_token;
        }
        DPRINT_COMBINE << "Expert " << local_expert << ": tokens=" << expert_tokens << ENDL();

        uint32_t actual_batches = (expert_tokens + read_batch_size - 1) / read_batch_size;

        // Round-robin batch assignment across the idle cores in this sender's group.
        // Each idle core starts at its own core_id and strides by num_idle_cores, so:
        //   core 0 processes batches 0, k, 2k, ...
        //   core 1 processes batches 1, k+1, 2k+1, ...
        //   core j processes batches j, j+k, j+2k, ...
        // where k = num_idle_cores.  This spreads the work evenly and ensures the
        // owning sender can predict which idle core handles each batch (batch_idx % k).
        for (uint32_t batch_idx = core_id; batch_idx < actual_batches; batch_idx += num_idle_cores) {
            uint32_t batch_tile_start = start_page_tiled + batch_idx * tiles_per_batch;

            // 1. Signal compute to start untilizing this batch
            cb_reserve_back(cb_signal_id, 1);
            volatile tt_l1_ptr uint32_t* signal_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_signal_id));
            signal_ptr[0] = 0x00000000;
            cb_push_back(cb_signal_id, 1);

            // 2. Reading tiles for this batch
            constexpr uint32_t tiles_per_batch_per_cb = tiles_per_batch / cb_factor;
            for (uint32_t cnt = 0; cnt < cb_factor; cnt++) {
                uint32_t batch_tile = batch_tile_start + cnt * tiles_per_batch_per_cb;
                cb_reserve_back(cb_dispatched_buffer_id, tiles_per_batch_per_cb);
                for (uint32_t t = 0; t < tiles_per_batch_per_cb; t++) {
                    noc_async_read_page(
                        batch_tile + t,
                        dispatched_buffer_addr_gen,
                        buffer_base + t * aligned_dispatched_buffer_page_size);
                }
                noc_async_read_barrier();
                cb_push_back(cb_dispatched_buffer_id, tiles_per_batch_per_cb);
            }
            // Steps 3-7 (wait for untilize, wait for sender's send signal, NOC-write to
            // sender, signal sender, pop untilize CB) now run on zero_init_writer.
        }
        // Advance to the next expert's region in the packed dispatch buffer.
        start_page_tiled += ((expert_tokens + tile_height - 1) / tile_height) * tiles_per_batch;
    }

    // Send sentinel to compute to break out of its loop
    cb_reserve_back(cb_signal_id, 1);
    volatile tt_l1_ptr uint32_t* signal_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_signal_id));
    signal_ptr[0] = ROUTE_INFO_SENTINEL;
    cb_push_back(cb_signal_id, 1);
}
