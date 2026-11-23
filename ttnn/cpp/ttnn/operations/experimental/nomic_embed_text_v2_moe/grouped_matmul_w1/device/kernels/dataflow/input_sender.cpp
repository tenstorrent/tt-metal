// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RISCV_1 (NOC1) of a dedicated input sender, a core east of the compute grid. The lane's num_senders senders take
// its blocks in turn, sender c the lane blocks i with i % num_senders == c. Each reads its next block into its own
// staging ring once the current one is out, multicasts each into the receivers' input slot i % input_slots, west,
// then sets the receivers' count to i + 1 from its private word. A receiver grants block i only once block i - 1 has
// arrived, so only one sender multicasts into the lane at a time and the count only grows.

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "expert_schedule.hpp"
#include "input_block.hpp"

void kernel_main() {
    constexpr uint32_t num_experts = get_named_compile_time_arg_val("num_experts");
    constexpr uint32_t num_lanes = get_named_compile_time_arg_val("num_lanes");
    constexpr uint32_t k_tiles = get_named_compile_time_arg_val("k_tiles");
    constexpr uint32_t counts_page_bytes = get_named_compile_time_arg_val("counts_page_bytes");
    constexpr uint32_t num_senders = get_named_compile_time_arg_val("num_senders");
    constexpr uint32_t staging_slots = get_named_compile_time_arg_val("staging_slots");
    constexpr uint32_t input_slots = get_named_compile_time_arg_val("input_slots");
    constexpr uint32_t input_banks = get_named_compile_time_arg_val("input_banks");
    constexpr uint32_t input_run = get_named_compile_time_arg_val("input_run");
    constexpr uint32_t input_cb_id = get_named_compile_time_arg_val("input_cb");
    constexpr uint32_t staging_cb_id = get_named_compile_time_arg_val("staging_cb");
    constexpr auto input_args = TensorAccessorArgs<0>();
    constexpr auto counts_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile_bytes = get_tile_size(input_cb_id);
    constexpr uint32_t block_bytes = k_tiles * tile_bytes;
    static_assert(staging_slots >= 1 && staging_slots <= 8);

    uint32_t arg = 0;
    const uint32_t input_address = get_arg_val<uint32_t>(arg++);
    const uint32_t counts_address = get_arg_val<uint32_t>(arg++);
    const uint32_t input_blocks = get_arg_val<uint32_t>(arg++);
    const uint32_t lane = get_arg_val<uint32_t>(arg++);
    const uint32_t sender_index = get_arg_val<uint32_t>(arg++);
    const uint32_t start_x = get_arg_val<uint32_t>(arg++);
    const uint32_t start_y = get_arg_val<uint32_t>(arg++);
    const uint32_t end_x = get_arg_val<uint32_t>(arg++);
    const uint32_t end_y = get_arg_val<uint32_t>(arg++);
    const uint32_t receivers = get_arg_val<uint32_t>(arg++);

    const Noc noc;
    DataflowBuffer input_cb(input_cb_id);
    DataflowBuffer staging_cb(staging_cb_id);
    DataflowBuffer counts_cb(get_named_compile_time_arg_val("counts_cb"));
    Semaphore<> granted(get_named_compile_time_arg_val("granted"));
    Semaphore<> arrived(get_named_compile_time_arg_val("arrived"));
    Semaphore<> flag(get_named_compile_time_arg_val("flag"));

    if constexpr (get_named_compile_time_arg_val("has_counts") == 1) {
        const auto counts_accessor = TensorAccessor(counts_args, counts_address);
        noc.async_read(counts_accessor, counts_cb, counts_page_bytes, {.page_id = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
    } else {
        // No counts: one expert takes every row.
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counts_cb.get_write_ptr()) =
            input_blocks * grouped_matmul::kBlockRows;
    }
    uint32_t first_block[num_experts];
    uint32_t num_blocks[num_experts];
    grouped_matmul::expert_regions<num_experts>(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counts_cb.get_write_ptr()),
        input_blocks,
        first_block,
        num_blocks);
    uint32_t order[num_experts];
    grouped_matmul::expert_order<num_experts>(num_blocks, order);

    // This sender's blocks in the lane's order, walked once.
    uint32_t count = 0;
    for (uint32_t expert = 0; expert < num_experts; ++expert) {
        count +=
            grouped_matmul::lane_blocks<num_lanes, false>(expert, first_block[expert], num_blocks[expert], lane).count;
    }
    count = count > sender_index ? (count - sender_index + num_senders - 1) / num_senders : 0;
    uint32_t walk_position = 0;
    uint32_t walk_step = 0;
    uint32_t walk_index = 0;
    auto next = [&](uint32_t& lane_index, uint32_t& block) {
        while (true) {
            const uint32_t expert = order[walk_position];
            const auto blocks =
                grouped_matmul::lane_blocks<num_lanes, false>(expert, first_block[expert], num_blocks[expert], lane);
            if (walk_step >= blocks.count) {
                ++walk_position;
                walk_step = 0;
                continue;
            }
            const uint32_t index = walk_index++;
            const uint32_t step = walk_step++;
            if (index % num_senders == sender_index) {
                lane_index = index;
                block = blocks.first + step * blocks.stride;
                return;
            }
        }
    };

    const auto input = TensorAccessor(input_args, input_address);
    // The receivers' slot i % input_slots sits at the same address on every core: the input ring is declared first
    // everywhere.
    const uint32_t ring_base = input_cb.get_write_ptr();
    const uint32_t staging_base = staging_cb.get_write_ptr();
    uint32_t staged_index[staging_slots];
    uint32_t staged_block[staging_slots];

    auto issue = [&](uint32_t mine) {
        const uint32_t slot = mine % staging_slots;
        next(staged_index[slot], staged_block[slot]);
        grouped_matmul::read_input_block<NocOptions::TXN_ID, k_tiles, tile_bytes, input_banks, input_run>(
            noc, input, input_address, staged_block[slot], staging_base + slot * block_bytes, slot + 1);
    };

    // Up to staging_slots - 1 blocks read ahead of the one being sent.
    for (uint32_t mine = 0; mine + 1 < staging_slots && mine < count; ++mine) {
        issue(mine);
    }
    for (uint32_t mine = 0; mine < count; ++mine) {
        noc.async_read_barrier<NocOptions::TXN_ID>({.trid = mine % staging_slots + 1});
        granted.wait_min((mine + 1) * receivers);
        const uint32_t lane_index = staged_index[mine % staging_slots];
        noc.async_write_multicast(
            CoreLocalMem<uint32_t>(staging_base + (mine % staging_slots) * block_bytes),
            MulticastEndpoint{},
            block_bytes,
            receivers,
            {},
            {.noc_x_start = start_x,
             .noc_y_start = start_y,
             .noc_x_end = end_x,
             .noc_y_end = end_y,
             .addr = ring_base + (lane_index % input_slots) * block_bytes},
            true);
        // The count must follow the data, and nothing else may go out on this NOC between the two.
        noc.async_writes_flushed();
        flag.set(lane_index + 1);
        flag.relay_multicast(noc, arrived, start_x, start_y, end_x, end_y, receivers);
        // Read ahead only now: issued before the send, every sender's next block competed with the first blocks for
        // the NoC, and the lanes' first multicasts waited 9 us at 8x512. Its slot's block was flushed an iteration
        // ago.
        if (mine + staging_slots - 1 < count) {
            issue(mine + staging_slots - 1);
        }
    }
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
