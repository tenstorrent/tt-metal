// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RISCV_1 (NOC1) of every compute core: the input stream of the core's lane, and this core's outputs. Either the
// lane's east core reads each block and multicasts it west (stream_protocol.hpp), or dedicated senders east of the
// lane take its blocks in turn (input_sender.cpp) and every compute core receives, granting lane block i to sender
// i % num_senders once block i - 1 has arrived.
//
// Output block i - 2 is written right after block i is granted (or its reads issued): input slot i is free only once
// compute has popped input i - 2, which it does after packing output i - 2, so the write never waits and never
// delays a grant. Each output block is flushed before its slot is popped. Subblock-outer calls (one expert) write
// each group's outputs once the next group is delivered, in compute's order: subblock by subblock, block by block.

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "ckernel.h"
#include "expert_schedule.hpp"
#include "input_block.hpp"
#include "ttnn/operations/experimental/nomic_embed_text_v2_moe/pirotski_matmul/device/kernels/dataflow/stream_protocol.hpp"

void kernel_main() {
    constexpr uint32_t num_experts = get_named_compile_time_arg_val("num_experts");
    constexpr uint32_t num_lanes = get_named_compile_time_arg_val("num_lanes");
    constexpr bool per_expert_lanes = get_named_compile_time_arg_val("per_expert_lanes") == 1;
    constexpr uint32_t k_tiles = get_named_compile_time_arg_val("k_tiles");
    constexpr uint32_t counts_page_bytes = get_named_compile_time_arg_val("counts_page_bytes");
    constexpr uint32_t input_cb_id = get_named_compile_time_arg_val("input_cb");
    constexpr uint32_t output_cb_id = get_named_compile_time_arg_val("output_cb");
    constexpr uint32_t output_n_tiles = get_named_compile_time_arg_val("output_n_tiles");
    constexpr bool row_major_output = get_named_compile_time_arg_val("row_major_output") == 1;
    // An L1-interleaved input over input_banks banks: the sender steps the bank and address itself.
    constexpr uint32_t input_banks = get_named_compile_time_arg_val("input_banks");
    // A sharded input whose shards hold input_run tiles of a row contiguously: one read a run.
    constexpr uint32_t input_run = get_named_compile_time_arg_val("input_run");
    constexpr uint32_t num_senders = get_named_compile_time_arg_val("num_senders");
    // The dedicated senders' branch compiles for every call; without them it never runs.
    constexpr uint32_t rotation = num_senders > 0 ? num_senders : 1;
    constexpr bool subblock_outer = get_named_compile_time_arg_val("subblock_outer") == 1;
    constexpr uint32_t subblock_w = get_named_compile_time_arg_val("subblock_w");
    // Blocks a subblock-outer group; 0: one group of all of the lane's blocks.
    constexpr uint32_t group_blocks = get_named_compile_time_arg_val("group_blocks");
    constexpr uint32_t max_held = 16;
    static_assert(group_blocks * 2 <= max_held);
    constexpr uint32_t group = group_blocks == 0 ? max_held : group_blocks;
    constexpr bool tail_split = get_named_compile_time_arg_val("tail_split") == 1;
    constexpr uint32_t subblocks_max = get_named_compile_time_arg_val("subblocks_max");
    static_assert(!tail_split || (subblock_outer && num_experts == 1));
    constexpr auto input_args = TensorAccessorArgs<0>();
    constexpr auto output_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto counts_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr uint32_t input_tile_bytes = get_tile_size(input_cb_id);
    constexpr uint32_t output_tile_bytes = get_tile_size(output_cb_id);

    uint32_t arg = 0;
    const uint32_t input_address = get_arg_val<uint32_t>(arg++);
    const uint32_t output_address = get_arg_val<uint32_t>(arg++);
    const uint32_t counts_address = get_arg_val<uint32_t>(arg++);
    const uint32_t input_blocks = get_arg_val<uint32_t>(arg++);
    const uint32_t lane = get_arg_val<uint32_t>(arg++);
    const uint32_t first_n_tile = get_arg_val<uint32_t>(arg++);
    const uint32_t n_tiles = get_arg_val<uint32_t>(arg++);
    const bool is_sender = get_arg_val<uint32_t>(arg++) == 1;
    const uint32_t sender_x = get_arg_val<uint32_t>(arg++);
    const uint32_t sender_y = get_arg_val<uint32_t>(arg++);
    pirotski_matmul::Multicast receivers;
    receivers.start_x = get_arg_val<uint32_t>(arg++);
    receivers.start_y = get_arg_val<uint32_t>(arg++);
    receivers.end_x = get_arg_val<uint32_t>(arg++);
    receivers.end_y = get_arg_val<uint32_t>(arg++);
    receivers.count = get_arg_val<uint32_t>(arg++);
    // The dedicated senders' NoC coordinates, (x, y) pairs in turn order.
    uint32_t senders_x[rotation];
    uint32_t senders_y[rotation];
    for (uint32_t sender = 0; sender < num_senders; ++sender) {
        senders_x[sender] = get_arg_val<uint32_t>(arg++);
        senders_y[sender] = get_arg_val<uint32_t>(arg++);
    }

    const Noc noc;
    DataflowBuffer input_cb(input_cb_id);
    DataflowBuffer output_cb(output_cb_id);
    DataflowBuffer counts_cb(get_named_compile_time_arg_val("counts_cb"));
    Semaphore<> granted(get_named_compile_time_arg_val("granted"));
    Semaphore<> arrived(get_named_compile_time_arg_val("arrived"));

    // With a bias, compute's tile of ones: 1.0 in column 0 of every row (faces 0 and 2), 0 elsewhere, so that ones x
    // (a tile whose row 0 is the bias) adds the bias to every row. Made while the first block is on its way: filled
    // first, it held a lane's first read back by about 2 us.
    auto make_ones = [&]() {
        if constexpr (get_named_compile_time_arg_val("has_bias") != 1) {
            return;
        }
        DataflowBuffer ones_cb(get_named_compile_time_arg_val("ones_cb"));
        ones_cb.reserve_back(1);
        auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ones_cb.get_write_ptr());
        for (uint32_t i = 0; i < 32 * 32 / 2; ++i) {
            words[i] = 0;
        }
        auto* ones = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ones_cb.get_write_ptr());
        constexpr uint16_t kOne = 0x3F80;
        for (uint32_t row = 0; row < 16; ++row) {
            ones[row * 16] = kOne;
            ones[2 * 16 * 16 + row * 16] = kOne;
        }
        (void)ckernel::load_blocking(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(&ones[2 * 16 * 16 + 15 * 16]));
        ones_cb.push_back(1);
    };

    const auto input = TensorAccessor(input_args, input_address);
    // The block whose read is already out, into the next slot under the next delivery's transaction id.
    uint32_t ahead_block = grouped_matmul::kNoUnit;
    // One expert: a lane's first block (its index with the tail split, else the start of its run below) follows from
    // the input's blocks when the count covers them all (fc1's does), so its read goes out first. (A counts read
    // under its own transaction id hung the next program under the device profiler; the count is polled.)
    if constexpr (subblock_outer) {
        const uint32_t first = tail_split ? lane
                                          : lane * (input_blocks / num_lanes) +
                                                (lane < input_blocks % num_lanes ? lane : input_blocks % num_lanes);
        if (is_sender && first < input_blocks) {
            input_cb.reserve_back(k_tiles);
            grouped_matmul::read_input_block<NocOptions::TXN_ID, k_tiles, input_tile_bytes, input_banks, input_run>(
                noc, input, input_address, first, input_cb.get_write_ptr(), 1);
            ahead_block = first;
        }
    }
    const auto counts_accessor = TensorAccessor(counts_args, counts_address);
    if constexpr (get_named_compile_time_arg_val("has_counts") == 0) {
        // No counts: one expert takes every row.
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counts_cb.get_write_ptr()) =
            input_blocks * grouped_matmul::kBlockRows;
    } else if constexpr (num_experts == 1) {
        // Wait for the one count itself rather than for every read: a barrier would also wait for the reads issued
        // ahead above. No row count is all ones.
        auto* count = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counts_cb.get_write_ptr());
        *count = 0xFFFFFFFF;
        (void)ckernel::load_blocking(count);
        noc.async_read(counts_accessor, counts_cb, counts_page_bytes, {.page_id = 0}, {.offset_bytes = 0});
        while (*count == 0xFFFFFFFF) {
        }
    } else {
        noc.async_read(counts_accessor, counts_cb, counts_page_bytes, {.page_id = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
    }
    uint32_t first_block[num_experts];
    uint32_t num_blocks[num_experts];
    grouped_matmul::expert_regions<num_experts>(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counts_cb.get_write_ptr()),
        input_blocks,
        first_block,
        num_blocks);

    const auto output = TensorAccessor(output_args, output_address);
    // A row-major block is 32 rows of n_tiles * 32 values, one row segment of each output row.
    const uint32_t row_segment_bytes = n_tiles * output_tile_bytes / grouped_matmul::kBlockRows;

    // One expert: the lane's east core reads each block under transaction id 1 + its delivery's parity, so that the
    // next block's read can be in flight while this one is multicast. With several experts (w2's blocks are 96 tile
    // reads) that form cost w2 4 to 8 us at 2x512 and 4x288: plain reads, one block at a time.
    constexpr bool pipelined = subblock_outer;
    constexpr NocOptions read_mode = pipelined ? NocOptions::TXN_ID : NocOptions::DEFAULT;
    auto read_block = [&](uint32_t block, uint32_t address, uint32_t trid) {
        grouped_matmul::read_input_block<read_mode, k_tiles, input_tile_bytes, input_banks, input_run>(
            noc, input, input_address, block, address, trid);
    };

    auto write_block = [&](uint32_t block) {
        output_cb.wait_front(n_tiles);
        const uint32_t source = output_cb.get_read_ptr();
        if constexpr (row_major_output) {
            uint32_t row = block * grouped_matmul::kBlockRows;
            const uint32_t column_offset = first_n_tile * output_tile_bytes / grouped_matmul::kBlockRows;
            for (uint32_t row_in_block = 0; row_in_block < grouped_matmul::kBlockRows; ++row_in_block, ++row) {
                noc.async_write(
                    CoreLocalMem<uint32_t>(source + row_in_block * row_segment_bytes),
                    output,
                    row_segment_bytes,
                    {},
                    {.page_id = row, .offset_bytes = column_offset});
            }
        } else {
            uint32_t page = block * output_n_tiles + first_n_tile;
            for (uint32_t tile = 0; tile < n_tiles; ++tile, ++page) {
                noc.async_write(
                    CoreLocalMem<uint32_t>(source + tile * output_tile_bytes),
                    output,
                    output_tile_bytes,
                    {},
                    {.page_id = page});
            }
        }
        noc.async_writes_flushed();
        output_cb.pop_front(n_tiles);
    };

    uint32_t delivered = 0;
    uint32_t history[2] = {0, 0};                   // block ids of the last two delivered, newest first
    uint32_t held[2 * max_held];                    // subblock-outer: block ids by delivery, two groups
    uint32_t tail_begin = grouped_matmul::kNoUnit;  // the tail block's subblocks [begin, end) this lane computes
    uint32_t tail_end = grouped_matmul::kNoUnit;
    auto write_lagged = [&]() {
        if (!subblock_outer && delivered >= 2) {
            write_block(history[1]);
        }
    };
    // Group n's outputs, in compute's order. The tail block is the last block of the last group and has the lane's
    // run of its subblocks only.
    uint32_t written_groups = 0;
    auto write_group = [&](uint32_t index) {
        const uint32_t first = index * group;
        const uint32_t count = delivered - first < group ? delivered - first : group;
        for (uint32_t subblock = 0; subblock * subblock_w < n_tiles; ++subblock) {
            for (uint32_t block = 0; block < count; ++block) {
                if (tail_begin != grouped_matmul::kNoUnit && first + block + 1 == delivered &&
                    (subblock < tail_begin || subblock >= tail_end)) {
                    continue;
                }
                output_cb.wait_front(subblock_w);
                const uint32_t source = output_cb.get_read_ptr();
                uint32_t page =
                    held[(first + block) % (2 * group)] * output_n_tiles + first_n_tile + subblock * subblock_w;
                for (uint32_t tile = 0; tile < subblock_w; ++tile, ++page) {
                    noc.async_write(
                        CoreLocalMem<uint32_t>(source + tile * output_tile_bytes),
                        output,
                        output_tile_bytes,
                        {},
                        {.page_id = page});
                }
                noc.async_writes_flushed();
                output_cb.pop_front(subblock_w);
            }
        }
    };

    auto deliver = [&](uint32_t block, uint32_t next) {
        input_cb.reserve_back(k_tiles);
        const uint32_t slot = input_cb.get_write_ptr();
        if (is_sender) {
            const uint32_t trid = delivered % 2 + 1;
            if (ahead_block != block) {
                // A first block read ahead that the lane does not take lands before this one replaces it.
                if (ahead_block != grouped_matmul::kNoUnit) {
                    noc.async_read_barrier<NocOptions::TXN_ID>({.trid = trid});
                }
                read_block(block, slot, trid);
            }
            if (delivered == 0) {
                make_ones();
            }
            write_lagged();
            if constexpr (pipelined) {
                noc.async_read_barrier<NocOptions::TXN_ID>({.trid = trid});
            } else {
                noc.async_read_barrier();
            }
            input_cb.push_back(k_tiles);
            // The next block's read goes out before this one's multicast, into its slot if compute has freed it.
            ahead_block = grouped_matmul::kNoUnit;
            if (pipelined && next != grouped_matmul::kNoUnit && input_cb.pages_reservable_at_back(k_tiles)) {
                input_cb.reserve_back(k_tiles);
                read_block(next, input_cb.get_write_ptr(), (delivered + 1) % 2 + 1);
                ahead_block = next;
            }
            if (receivers.count > 0) {
                pirotski_matmul::send_block(
                    noc, granted, arrived, receivers, slot, k_tiles * input_tile_bytes, delivered);
            }
        } else {
            // Block i - 1 has arrived (the last iteration waited for it) and slot i is free.
            const uint32_t grant_x = num_senders > 0 ? senders_x[delivered % rotation] : sender_x;
            const uint32_t grant_y = num_senders > 0 ? senders_y[delivered % rotation] : sender_y;
            granted.up(noc, grant_x, grant_y, 1);
            if (delivered == 0) {
                make_ones();
            }
            write_lagged();
            arrived.wait_min(delivered + 1);
            input_cb.push_back(k_tiles);
        }
        history[1] = history[0];
        history[0] = block;
        if constexpr (subblock_outer) {
            ASSERT(group_blocks > 0 || delivered < max_held);
            held[delivered % (2 * group)] = block;
        }
        ++delivered;
        // Group n goes out once group n + 1 is delivered: compute has finished group n - 1 to free its input slots,
        // and group n's outputs fill the ring beside group n + 1's being made.
        if constexpr (subblock_outer && group_blocks > 0) {
            if (delivered % group == 0 && delivered / group >= 2) {
                write_group(written_groups++);
            }
        }
    };

    // Each block is delivered once the next is known, so the sender can read it ahead.
    uint32_t pending = grouped_matmul::kNoUnit;
    auto visit = [&](uint32_t block) {
        if (pending != grouped_matmul::kNoUnit) {
            deliver(pending, block);
        }
        pending = block;
    };
    if constexpr (tail_split) {
        const auto work = grouped_matmul::lane_work(num_blocks[0], num_lanes, lane, subblocks_max);
        tail_begin = work.tail_subblock_begin;
        tail_end = work.tail_subblock_end;
        for (uint32_t index = 0, block = lane; index < work.full_count; ++index, block += num_lanes) {
            visit(block);
        }
        if (work.tail_block != grouped_matmul::kNoUnit) {
            visit(work.tail_block);
        }
    } else {
        uint32_t order[num_experts];
        grouped_matmul::expert_order<num_experts>(num_blocks, order);
        for (uint32_t position = 0; position < num_experts; ++position) {
            const uint32_t expert = order[position];
            const auto blocks = grouped_matmul::lane_blocks<num_lanes, per_expert_lanes>(
                expert, first_block[expert], num_blocks[expert], lane);
            if constexpr (subblock_outer) {
                // One expert: the lane's count, as a contiguous run. Dealt round-robin, the lanes' senders read
                // neighbouring blocks at once, which a split norm output keeps on the same few cores.
                const uint32_t share = num_blocks[expert] / num_lanes;
                const uint32_t extra = num_blocks[expert] % num_lanes;
                uint32_t block = first_block[expert] + lane * share + (lane < extra ? lane : extra);
                for (uint32_t index = 0; index < blocks.count; ++index, ++block) {
                    visit(block);
                }
            } else {
                for (uint32_t index = 0, block = blocks.first; index < blocks.count; ++index, block += blocks.stride) {
                    visit(block);
                }
            }
        }
    }
    if (pending != grouped_matmul::kNoUnit) {
        deliver(pending, grouped_matmul::kNoUnit);
    }
    // A lane with no blocks still owes compute its ones, and lets a first block it read ahead land.
    if (delivered == 0) {
        make_ones();
        if (ahead_block != grouped_matmul::kNoUnit) {
            noc.async_read_barrier<NocOptions::TXN_ID>({.trid = 1});
        }
    }
    if constexpr (subblock_outer) {
        while (written_groups * group < delivered) {
            write_group(written_groups++);
        }
    } else {
        // The blocks still unwritten, oldest first.
        for (uint32_t pending = delivered < 2 ? delivered : 2; pending > 0; --pending) {
            write_block(history[pending - 1]);
        }
    }
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
