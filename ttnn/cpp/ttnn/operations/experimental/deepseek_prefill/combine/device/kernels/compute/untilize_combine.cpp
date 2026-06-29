// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack_untilize.h"
#include "api/dataflow/circular_buffer.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "api/debug/dprint.h"
#include "internal/circular_buffer_interface.h"
#include "tools/profiler/kernel_profiler.hpp"

#define ENABLE_COMBINE_DEBUG 0
#if ENABLE_COMBINE_DEBUG
#define DPRINT_COMBINE(...) DPRINT(__VA_ARGS__)
#else
#define DPRINT_COMBINE(...)
#endif

// Compile-time args (shared across untilizer cores):
//   0: cb_untilize_id                  - CB for compute's untilized output (c_2)
//   1: cb_in_id                        - CB for untilize input tile data (c_0)
//   2: cb_experts_tok_counter_id       - CB holding multicasted per-expert token counts (c_1)
//   3: experts_tok_counter_pages       - number of pages in the counter CB
//   4: experts_per_chip                - count of experts mapped to this chip
//   5: counter_offset                  - uint32 offset into the counter buffer for this chip
//   6: max_dispatch_buffer_token_size  - per-chip dispatch capacity (overflow clamp)
//   7: read_batch_size                 - rows per untilize batch (== tile_height for this op)
//   8: full_ct_dim                     - hidden_size / tile_width (tiles per batch)
//   9: block_ct_dim                    - tiles per pack call (largest divisor of full_ct_dim <= 8)
//  10: cb_counter_total_pages          - full page capacity of c_1 (counter + trailer); used
//                                        for cb_wait_front on the multicasted CB
//
// Runtime args (per untilizer core):
//   0: expert_start_idx                - first expert handled (now 0; every group handles all experts)
//   1: expert_end_idx                  - one past the last expert handled (now experts_per_chip)
//   2: core_id                         - local index in the owning sender's untilizer group (0..k_s-1)
//   3: num_untilizer_cores                  - k_s, size of the owning sender's untilizer group
//   4: untilizer_global_pos            - this core's position in the global interleaved untilizer
//                                        ordering; its batches are global_pos, +G, +2G, … per expert
//   5: total_untilizers                - G, total untilizer cores across all senders (global stride)
void kernel_main() {
    constexpr uint32_t cb_untilize_id = get_compile_time_arg_val(0);
    CircularBuffer cb_untilize(cb_untilize_id);
    constexpr uint32_t cb_in_id = get_compile_time_arg_val(1);
    CircularBuffer cb_in(cb_in_id);
    constexpr uint32_t cb_experts_tok_counter_id = get_compile_time_arg_val(2);
    CircularBuffer cb_experts_tok_counter(cb_experts_tok_counter_id);
    constexpr uint32_t experts_tok_counter_pages = get_compile_time_arg_val(3);
    constexpr uint32_t experts_per_chip = get_compile_time_arg_val(4);
    constexpr uint32_t counter_offset = get_compile_time_arg_val(5);
    constexpr uint32_t max_dispatch_buffer_token_size = get_compile_time_arg_val(6);
    constexpr uint32_t read_batch_size = get_compile_time_arg_val(7);
    constexpr uint32_t full_ct_dim = get_compile_time_arg_val(8);
    constexpr uint32_t block_ct_dim = get_compile_time_arg_val(9);
    constexpr uint32_t cb_counter_total_pages = get_compile_time_arg_val(10);
    constexpr uint32_t num_blocks = full_ct_dim / block_ct_dim;
    // read_batch_size doubles as tile_height: one tile-row of input -> read_batch_size element rows.
    constexpr uint32_t tile_height = read_batch_size;
    constexpr uint32_t tiles_per_batch = full_ct_dim;

    uint32_t rt_idx = 0;
    uint32_t expert_start_idx = get_arg_val<uint32_t>(rt_idx++);
    uint32_t expert_end_idx = get_arg_val<uint32_t>(rt_idx++);
    uint32_t core_id = get_arg_val<uint32_t>(rt_idx++);
    uint32_t num_untilizer_cores = get_arg_val<uint32_t>(rt_idx++);
    uint32_t untilizer_global_pos = get_arg_val<uint32_t>(rt_idx++);
    uint32_t total_untilizers = get_arg_val<uint32_t>(rt_idx++);
    // core_id / num_untilizer_cores no longer drive batch assignment under the global round-robin.
    (void)core_id;
    (void)num_untilizer_cores;

    compute_kernel_hw_startup(cb_in_id, cb_untilize_id);
    pack_untilize_init<block_ct_dim, full_ct_dim>(cb_in_id, cb_untilize_id);

    // Wait for the owning sender's expert-token-count multicast.  reader_untilize on this same
    // core pushes the counter CB once after counter_ready_sem fires, so cb_wait_front here
    // doubles as the gate for "counter data is live in L1".  The CB is never popped — both
    // this kernel and writer_untilize rely on the data staying resident.
    cb_experts_tok_counter.wait_front(cb_counter_total_pages);

    // Snapshot per-expert token counts.  read_tile_value has UNPACK read from L1 and broadcast
    // to MATH / PACK via mailbox, so all three TRISCs end up with identical counts and walk the
    // same iteration sequence.
    uint32_t local_expert_counts[experts_per_chip];
    for (uint32_t e = 0; e < experts_per_chip; e++) {
        local_expert_counts[e] = read_tile_value(cb_experts_tok_counter_id, 0, counter_offset + e);
    }

    // Accumulate dispatch-buffer offset for experts below this core's range using raw counts
    // (host laid the dispatch buffer out with tile-aligned per-expert stride; the same applies
    // to skipped experts).
    uint32_t start_page_tiled = 0;
    for (uint32_t e = 0; e < expert_start_idx; e++) {
        start_page_tiled += ((local_expert_counts[e] + tile_height - 1) / tile_height) * tiles_per_batch;
    }

    for (uint32_t local_expert = expert_start_idx; local_expert < expert_end_idx; local_expert++) {
        uint32_t expert_tokens = local_expert_counts[local_expert];
        uint32_t start_token = (start_page_tiled / tiles_per_batch) * tile_height;
        // Mirror reader_dispatch's overflow guard so we never untilize past the dispatch buffer.
        if (start_token >= max_dispatch_buffer_token_size) {
            expert_tokens = 0;
        } else if (start_token + expert_tokens > max_dispatch_buffer_token_size) {
            expert_tokens = max_dispatch_buffer_token_size - start_token;
        }

        uint32_t actual_batches = (expert_tokens + read_batch_size - 1) / read_batch_size;

        // Global round-robin: this core handles batches untilizer_global_pos, +G, +2G, … of every
        // expert (G = total_untilizers across all senders).  Must match reader_untilize /
        // writer_untilize exactly so the c_0 / c_2 producer-consumer protocol stays in lockstep.
        for (uint32_t batch_idx = untilizer_global_pos; batch_idx < actual_batches; batch_idx += total_untilizers) {
            cb_untilize.reserve_back(read_batch_size);
            for (uint32_t block = 0; block < num_blocks; block++) {
                cb_in.wait_front(block_ct_dim);
                {
                    // DeviceZoneScopedN("UNTILIZING");
                    pack_untilize_block<block_ct_dim, full_ct_dim>(cb_in_id, 1, cb_untilize_id, block);
                }
                cb_in.pop_front(block_ct_dim);
            }
            cb_untilize.push_back(read_batch_size);
        }
        start_page_tiled += ((expert_tokens + tile_height - 1) / tile_height) * tiles_per_batch;
    }

    pack_untilize_uninit(cb_untilize_id);
}
