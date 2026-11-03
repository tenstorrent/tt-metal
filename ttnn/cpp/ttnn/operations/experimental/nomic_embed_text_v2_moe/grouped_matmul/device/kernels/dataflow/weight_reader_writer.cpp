// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RISCV_0 of every core: each expert's weight slice for this column, and this core's outputs. Every
// core of the column reads its share of expert e's K rows, the others write theirs into the first
// core's slot, and the first core multicasts the whole slice down the column into a two-slot ring,
// so expert e + 1's slice loads while e computes. One core reading a slice alone was bound by issuing
// one read per tile. This kernel also writes the outputs of expert e - 1's blocks after it hands
// over expert e's slice: the output ring then never waits on a slice that cannot arrive, which would
// hang the device. With per-expert lanes every core reads its whole share of its own expert's slice
// instead, so the lanes' slices load at once, with no gather or multicast.

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "ckernel.h"
#include "hostdevcommon/common_values.hpp"
#include "expert_schedule.hpp"

void kernel_main() {
    constexpr bool is_sender = get_named_compile_time_arg_val("is_sender") == 1;
    constexpr bool per_expert_lanes = get_named_compile_time_arg_val("per_expert_lanes") == 1;
    constexpr uint32_t num_experts = get_named_compile_time_arg_val("num_experts");
    constexpr uint32_t num_lanes = get_named_compile_time_arg_val("num_lanes");
    constexpr uint32_t k_tiles = get_named_compile_time_arg_val("k_tiles");
    // A slot holds k_tiles rows of slot_n_tiles tiles; a column with fewer N tiles leaves the rest.
    constexpr uint32_t slot_n_tiles = get_named_compile_time_arg_val("slot_n_tiles");
    constexpr uint32_t num_receivers = get_named_compile_time_arg_val("num_receivers");
    constexpr uint32_t counts_page_bytes = get_named_compile_time_arg_val("counts_page_bytes");
    // Weight tile (k, n) of expert e is page k * weight_stride_k + n * weight_stride_n + e * weight_stride_expert.
    constexpr uint32_t weight_stride_k = get_named_compile_time_arg_val("weight_stride_k");
    constexpr uint32_t weight_stride_n = get_named_compile_time_arg_val("weight_stride_n");
    constexpr uint32_t weight_stride_expert = get_named_compile_time_arg_val("weight_stride_expert");
    constexpr uint32_t output_n_tiles = get_named_compile_time_arg_val("output_n_tiles");
    constexpr bool row_major_output = get_named_compile_time_arg_val("row_major_output") == 1;
    constexpr uint32_t subblock_w = get_named_compile_time_arg_val("subblock_w");
    constexpr uint32_t weight_cb_id = get_named_compile_time_arg_val("weight_cb");
    constexpr uint32_t output_cb_id = get_named_compile_time_arg_val("output_cb");
    constexpr uint32_t counts_cb_id = get_named_compile_time_arg_val("counts_cb");
    constexpr uint32_t schedule_cb_id = get_named_compile_time_arg_val("schedule_cb");
    constexpr auto weight_args = TensorAccessorArgs<0>();
    constexpr auto output_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();
    constexpr auto counts_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();

    const uint32_t weight_address = get_arg_val<uint32_t>(0);
    const uint32_t output_address = get_arg_val<uint32_t>(1);
    const uint32_t counts_address = get_arg_val<uint32_t>(2);
    const uint32_t input_blocks = get_arg_val<uint32_t>(3);
    const uint32_t lane = get_arg_val<uint32_t>(4);
    const uint32_t first_n_tile = get_arg_val<uint32_t>(5);
    const uint32_t n_tiles = get_arg_val<uint32_t>(6);
    // A receiver signals the column's sender; the sender multicasts to the rectangle of receivers.
    const uint32_t sender_x = get_arg_val<uint32_t>(7);
    const uint32_t sender_y = get_arg_val<uint32_t>(8);
    const uint32_t receivers_start_x = get_arg_val<uint32_t>(9);
    const uint32_t receivers_start_y = get_arg_val<uint32_t>(10);
    const uint32_t receivers_end_x = get_arg_val<uint32_t>(11);
    const uint32_t receivers_end_y = get_arg_val<uint32_t>(12);

    const Noc noc;
    DataflowBuffer weight_cb(weight_cb_id);
    DataflowBuffer output_cb(output_cb_id);
    DataflowBuffer counts_cb(counts_cb_id);
    DataflowBuffer schedule_cb(schedule_cb_id);
    Semaphore<> sender_semaphore(get_named_compile_time_arg_val("sender_semaphore"));
    Semaphore<> receiver_semaphore(get_named_compile_time_arg_val("receiver_semaphore"));
    Semaphore<> slot_free_semaphore(get_named_compile_time_arg_val("slot_free_semaphore"));

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
    // Whether this core takes expert e's slice. In the shared schedule a lane takes every slice the
    // column shares, also of an expert it has no block of; a per-expert lane takes its own expert's
    // only, and none when that expert has no rows.
    auto takes_slice = [&](uint32_t expert) {
        return per_expert_lanes ? lane_blocks(expert).count > 0 : num_blocks[expert] > 0;
    };

    // The compute kernel's schedule: this core's blocks of each expert, or kSkipExpert where no slice
    // comes.
    schedule_cb.reserve_back(1);
    auto* schedule = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(schedule_cb.get_write_ptr());
    for (uint32_t expert = 0; expert < num_experts; ++expert) {
        schedule[expert] = takes_slice(expert) ? lane_blocks(expert).count : grouped_matmul::kSkipExpert;
    }
    // A store can retire before L1 has it; a blocking load of the last word returns once all have
    // landed, so the compute kernel never reads the schedule ahead of it.
    (void)ckernel::load_blocking(&schedule[num_experts - 1]);
    schedule_cb.push_back(1);

    const auto weight_accessor = TensorAccessor(weight_args, weight_address);
    const auto output_accessor = TensorAccessor(output_args, output_address);
    constexpr uint32_t weight_tile_bytes = get_tile_size(weight_cb_id);
    // The ring's pages are padded to the DRAM alignment, the stride a tile is read and multicast at.
    constexpr uint32_t aligned_weight_tile_bytes = (weight_tile_bytes + DRAM_ALIGNMENT - 1) & ~(DRAM_ALIGNMENT - 1);
    constexpr uint32_t slot_tiles = k_tiles * slot_n_tiles;
    constexpr uint32_t slot_bytes = slot_tiles * aligned_weight_tile_bytes;
    constexpr uint32_t output_tile_bytes = get_tile_size(output_cb_id);
    // A row-major block is 32 rows of n_tiles * 32 bf16 values, one row segment of each output row.
    const uint32_t row_segment_bytes = n_tiles * output_tile_bytes / grouped_matmul::kBlockRows;

    // An output slot is free once its writes have left L1; their acknowledgements are awaited once,
    // at the end. Waiting for them per block cost w2 15 us at 4096 tokens.
    auto write_outputs = [&](uint32_t expert) {
        const auto blocks = lane_blocks(expert);
        for (uint32_t index = 0, block = blocks.first; index < blocks.count; ++index, block += blocks.stride) {
            if constexpr (row_major_output) {
                output_cb.wait_front(n_tiles);
                uint32_t row = block * grouped_matmul::kBlockRows;
                for (uint32_t row_in_block = 0; row_in_block < grouped_matmul::kBlockRows; ++row_in_block, ++row) {
                    noc.async_write(
                        output_cb,
                        output_accessor,
                        row_segment_bytes,
                        {.offset_bytes = row_in_block * row_segment_bytes},
                        {.page_id = row,
                         .offset_bytes = first_n_tile * output_tile_bytes / grouped_matmul::kBlockRows});
                }
                noc.async_writes_flushed();
                output_cb.pop_front(n_tiles);
            } else {
                uint32_t page = block * output_n_tiles + first_n_tile;
                for (uint32_t subblock = 0; subblock < n_tiles; subblock += subblock_w) {
                    output_cb.wait_front(subblock_w);
                    for (uint32_t tile = 0; tile < subblock_w; ++tile, ++page) {
                        noc.async_write(
                            output_cb,
                            output_accessor,
                            output_tile_bytes,
                            {.offset_bytes = tile * output_tile_bytes},
                            {.page_id = page});
                    }
                    noc.async_writes_flushed();
                    output_cb.pop_front(subblock_w);
                }
            }
        }
    };

    // This core's share of every slice: K rows [first_k, end_k), contiguous in the slot.
    const uint32_t first_k = lane * k_tiles / num_lanes;
    const uint32_t end_k = (lane + 1) * k_tiles / num_lanes;
    const uint32_t part_offset = first_k * slot_n_tiles * aligned_weight_tile_bytes;
    const uint32_t part_bytes = (end_k - first_k) * slot_n_tiles * aligned_weight_tile_bytes;
    auto read_part = [&](uint32_t expert) {
        for (uint32_t k = first_k; k < end_k; ++k) {
            uint32_t page = k * weight_stride_k + first_n_tile * weight_stride_n + expert * weight_stride_expert;
            uint32_t offset = k * slot_n_tiles * aligned_weight_tile_bytes;
            for (uint32_t n = 0; n < n_tiles; ++n, page += weight_stride_n, offset += aligned_weight_tile_bytes) {
                noc.async_read(
                    weight_accessor, weight_cb, weight_tile_bytes, {.page_id = page}, {.offset_bytes = offset});
            }
        }
        noc.async_read_barrier();
    };

    if constexpr (per_expert_lanes) {
        // Pages go to the DRAM banks in turn. Where the strides are multiples of the bank count, as
        // in both of the model's projections, every core walking its tiles from (0, 0) would read the
        // same bank at the same time, so each core starts at another K row and N tile. A tile's place
        // in the slot depends on (k, n) alone, so the walk's order changes nothing else.
        const uint32_t column = first_n_tile / slot_n_tiles;
        const uint32_t start_k = (lane + column) % k_tiles;
        const uint32_t start_n = lane % n_tiles;
        auto read_slice = [&](uint32_t expert) {
            uint32_t k = start_k;
            for (uint32_t k_step = 0; k_step < k_tiles; ++k_step) {
                uint32_t n = start_n;
                for (uint32_t n_step = 0; n_step < n_tiles; ++n_step) {
                    const uint32_t page =
                        k * weight_stride_k + (first_n_tile + n) * weight_stride_n + expert * weight_stride_expert;
                    noc.async_read(
                        weight_accessor,
                        weight_cb,
                        weight_tile_bytes,
                        {.page_id = page},
                        {.offset_bytes = (k * slot_n_tiles + n) * aligned_weight_tile_bytes});
                    n = n + 1 == n_tiles ? 0 : n + 1;
                }
                k = k + 1 == k_tiles ? 0 : k + 1;
            }
            noc.async_read_barrier();
        };
        for (uint32_t expert = 0; expert < num_experts; ++expert) {
            if (!takes_slice(expert)) {
                continue;
            }
            weight_cb.reserve_back(slot_tiles);
            read_slice(expert);
            weight_cb.push_back(slot_tiles);
            write_outputs(expert);
        }
        noc.async_write_barrier();
        return;
    }

    if constexpr (is_sender) {
        receiver_semaphore.set(VALID);
    }
    uint32_t previous_expert = num_experts;
    // Slices sent so far, the same count on every core of the column.
    uint32_t slice = 0;
    for (uint32_t expert = 0; expert < num_experts; ++expert) {
        if (!takes_slice(expert)) {
            continue;
        }
        ++slice;
        weight_cb.reserve_back(slot_tiles);
        const uint32_t slot_address = weight_cb.get_write_ptr();
        if constexpr (is_sender) {
            if constexpr (num_receivers > 0) {
                // The others may write their shares into this slot only once its last slice is done
                // with here.
                slot_free_semaphore.set(slice);
                slot_free_semaphore.set_multicast(
                    noc, receivers_start_x, receivers_start_y, receivers_end_x, receivers_end_y, num_receivers);
            }
            read_part(expert);
            if constexpr (num_receivers > 0) {
                // Every receiver has its slot free and its share in this one.
                sender_semaphore.wait(num_receivers);
                sender_semaphore.set(0);
                noc.async_write_multicast(
                    CoreLocalMem<uint32_t>(slot_address),
                    MulticastEndpoint{},
                    slot_bytes,
                    num_receivers,
                    {},
                    {.noc_x_start = receivers_start_x,
                     .noc_y_start = receivers_start_y,
                     .noc_x_end = receivers_end_x,
                     .noc_y_end = receivers_end_y,
                     .addr = slot_address},
                    true);
#ifdef ARCH_BLACKHOLE
                noc.async_writes_flushed();
#endif
                receiver_semaphore.set_multicast(
                    noc, receivers_start_x, receivers_start_y, receivers_end_x, receivers_end_y, num_receivers);
            }
        } else {
            read_part(expert);
            slot_free_semaphore.wait(slice);
            // A NoC write of 0 bytes hangs the device; a core gets no rows only when K is under the lanes.
            if (part_bytes > 0) {
                noc.async_write(
                    CoreLocalMem<uint32_t>(slot_address + part_offset),
                    UnicastEndpoint{},
                    part_bytes,
                    {},
                    {.noc_x = sender_x, .noc_y = sender_y, .addr = slot_address + part_offset});
                // The share must be in the sender's slot before the count of arrivals says so.
                noc.async_write_barrier();
            }
            receiver_semaphore.set(INVALID);
            sender_semaphore.up(noc, sender_x, sender_y, 1);
            receiver_semaphore.wait(VALID);
        }
        weight_cb.push_back(slot_tiles);
        if (previous_expert < num_experts) {
            write_outputs(previous_expert);
        }
        previous_expert = expert;
    }
    if (previous_expert < num_experts) {
        write_outputs(previous_expert);
    }
    noc.async_write_barrier();
    if constexpr (!is_sender) {
        noc.async_atomic_barrier();
    }
}
