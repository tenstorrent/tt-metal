// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RISCV_1 of every core: the input rows of each of this lane's blocks. The lane's first core reads a
// block's K tiles and multicasts them along the lane; the other cores take them into the same ring
// slot. Blocks come in expert order, the order the compute kernel walks them in.

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "expert_schedule.hpp"

void kernel_main() {
    constexpr bool is_sender = get_named_compile_time_arg_val("is_sender") == 1;
    constexpr bool per_expert_lanes = get_named_compile_time_arg_val("per_expert_lanes") == 1;
    constexpr uint32_t num_experts = get_named_compile_time_arg_val("num_experts");
    constexpr uint32_t num_lanes = get_named_compile_time_arg_val("num_lanes");
    constexpr uint32_t k_tiles = get_named_compile_time_arg_val("k_tiles");
    constexpr uint32_t num_receivers = get_named_compile_time_arg_val("num_receivers");
    constexpr uint32_t counts_page_bytes = get_named_compile_time_arg_val("counts_page_bytes");
    constexpr uint32_t input_cb_id = get_named_compile_time_arg_val("input_cb");
    constexpr uint32_t counts_cb_id = get_named_compile_time_arg_val("counts_cb");
    constexpr auto input_args = TensorAccessorArgs<0>();
    constexpr auto counts_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

    const uint32_t input_address = get_arg_val<uint32_t>(0);
    const uint32_t counts_address = get_arg_val<uint32_t>(1);
    const uint32_t input_blocks = get_arg_val<uint32_t>(2);
    const uint32_t lane = get_arg_val<uint32_t>(3);
    // A receiver signals the lane's sender; the sender multicasts to the rectangle of receivers.
    const uint32_t sender_x = get_arg_val<uint32_t>(4);
    const uint32_t sender_y = get_arg_val<uint32_t>(5);
    const uint32_t receivers_start_x = get_arg_val<uint32_t>(6);
    const uint32_t receivers_start_y = get_arg_val<uint32_t>(7);
    const uint32_t receivers_end_x = get_arg_val<uint32_t>(8);
    const uint32_t receivers_end_y = get_arg_val<uint32_t>(9);

    const Noc noc;
    DataflowBuffer input_cb(input_cb_id);
    DataflowBuffer counts_cb(counts_cb_id);
    Semaphore<> sender_semaphore(get_named_compile_time_arg_val("sender_semaphore"));
    Semaphore<> receiver_semaphore(get_named_compile_time_arg_val("receiver_semaphore"));

    const auto counts_accessor = TensorAccessor(counts_args, counts_address);
    noc.async_read(counts_accessor, counts_cb, counts_page_bytes, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    uint32_t first_block[num_experts];
    uint32_t num_blocks[num_experts];
    grouped_matmul::expert_regions<num_experts>(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counts_cb.get_write_ptr()),
        input_blocks,
        first_block,
        num_blocks);
    auto lane_blocks = [&](uint32_t expert) {
        return grouped_matmul::lane_blocks<num_lanes, per_expert_lanes>(
            expert, first_block[expert], num_blocks[expert], lane);
    };

    if constexpr (is_sender) {
        const auto input_accessor = TensorAccessor(input_args, input_address);
        constexpr uint32_t tile_bytes = get_tile_size(input_cb_id);
        // The ring's pages are padded to the DRAM alignment, the stride a tile is read and multicast at.
        constexpr uint32_t aligned_tile_bytes = (tile_bytes + DRAM_ALIGNMENT - 1) & ~(DRAM_ALIGNMENT - 1);
        constexpr uint32_t block_bytes = k_tiles * aligned_tile_bytes;
        receiver_semaphore.set(VALID);
        for (uint32_t expert = 0; expert < num_experts; ++expert) {
            const auto blocks = lane_blocks(expert);
            for (uint32_t index = 0, block = blocks.first; index < blocks.count; ++index, block += blocks.stride) {
                input_cb.reserve_back(k_tiles);
                const uint32_t block_address = input_cb.get_write_ptr();
                uint32_t page = block * k_tiles;
                for (uint32_t tile = 0; tile < k_tiles; ++tile, ++page) {
                    noc.async_read(
                        input_accessor,
                        input_cb,
                        tile_bytes,
                        {.page_id = page},
                        {.offset_bytes = tile * aligned_tile_bytes});
                }
                noc.async_read_barrier();
                if constexpr (num_receivers > 0) {
                    sender_semaphore.wait(num_receivers);
                    sender_semaphore.set(0);
                    noc.async_write_multicast(
                        CoreLocalMem<uint32_t>(block_address),
                        MulticastEndpoint{},
                        block_bytes,
                        num_receivers,
                        {},
                        {.noc_x_start = receivers_start_x,
                         .noc_y_start = receivers_start_y,
                         .noc_x_end = receivers_end_x,
                         .noc_y_end = receivers_end_y,
                         .addr = block_address},
                        true);
#ifdef ARCH_BLACKHOLE
                    noc.async_writes_flushed();
#endif
                    receiver_semaphore.set_multicast(
                        noc, receivers_start_x, receivers_start_y, receivers_end_x, receivers_end_y, num_receivers);
                }
                input_cb.push_back(k_tiles);
            }
        }
    } else {
        uint32_t lane_block_total = 0;
        for (uint32_t expert = 0; expert < num_experts; ++expert) {
            lane_block_total += lane_blocks(expert).count;
        }
        for (uint32_t block = 0; block < lane_block_total; ++block) {
            input_cb.reserve_back(k_tiles);
            receiver_semaphore.set(INVALID);
            sender_semaphore.up(noc, sender_x, sender_y, 1);
            receiver_semaphore.wait(VALID);
            input_cb.push_back(k_tiles);
        }
        noc.async_atomic_barrier();
    }
}
