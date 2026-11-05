// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Turns the window's top-k expert ids into gather_index, combine_index and expert_counts. Each core
// of a rectangle takes a contiguous range of the window's tile rows:
//   1. every core counts the slots of each expert among its tokens and sends the counts to core 0,
//      which multicasts the table of all counts back;
//   2. every core sums the table: the totals give each expert's region, starting on a 32-row block
//      (the exclusive scan of the totals, each rounded up to 32), and the counts of the cores
//      before it give the core's first row inside each region;
//   3. every core hands out its rows in token order (k inner) and writes its slice of
//      combine_index;
//   4. expert e's region of gather_index is assembled on core e % num_cores: the other cores send
//      it their rows of that expert, and it writes the region to DRAM.
// The rows follow from the counts alone, so the result is exact and does not depend on timing.

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "ckernel.h"
#include "routing_metadata_faces.hpp"

namespace {

using namespace routing_metadata;

constexpr uint32_t round_up_to_block(uint32_t rows) { return (rows + kBlockRows - 1) / kBlockRows * kBlockRows; }

// A RISC-V store can retire before L1 has processed it, and the NoC reads L1 as a separate client
// with no ordering against this core. L1 serves this core's requests in order, so a blocking load
// of a word just stored returns only once every earlier store has landed.
void drain_stores(volatile tt_l1_ptr uint32_t* scratch_word) {
    *scratch_word = 0;
    (void)ckernel::load_blocking(scratch_word);
}

// Sets words [begin, end) of an L1 array to value. Unrolled, so the loop overhead does not outweigh
// the stores, which L1 takes four to a write.
void fill_rows(volatile tt_l1_ptr uint32_t* words, uint32_t begin, uint32_t end, uint32_t value) {
#pragma GCC unroll 4
    for (uint32_t index = begin; index < end; ++index) {
        words[index] = value;
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t num_experts = get_named_compile_time_arg_val("num_experts");
    constexpr uint32_t top_k = get_named_compile_time_arg_val("top_k");
    constexpr uint32_t table_row_words = get_named_compile_time_arg_val("table_row_words");
    constexpr uint32_t packed_row_words = (num_experts + 1) / 2;
    constexpr uint32_t totals_row_words = get_named_compile_time_arg_val("totals_row_words");
    // A load from L1 takes at least 8 cycles. The ids of this many tokens are loaded before any is
    // used, so their latencies overlap instead of adding up.
    constexpr uint32_t group_tokens = top_k <= 2 ? 4 : (top_k <= 4 ? 2 : 1);
    constexpr uint32_t group_slots = group_tokens * top_k;
    constexpr auto indices_args = TensorAccessorArgs<0>();
    constexpr auto gather_index_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto combine_index_args = TensorAccessorArgs<gather_index_args.next_compile_time_args_offset()>();
    constexpr auto expert_counts_args = TensorAccessorArgs<combine_index_args.next_compile_time_args_offset()>();

    const uint32_t indices_address = get_arg_val<uint32_t>(0);
    const uint32_t gather_index_address = get_arg_val<uint32_t>(1);
    const uint32_t combine_index_address = get_arg_val<uint32_t>(2);
    const uint32_t expert_counts_address = get_arg_val<uint32_t>(3);
    const uint32_t token_start = get_arg_val<uint32_t>(4);
    const uint32_t num_tokens = get_arg_val<uint32_t>(5);
    const uint32_t gathered_rows = get_arg_val<uint32_t>(6);
    const uint32_t gather_index_page_bytes = get_arg_val<uint32_t>(7);
    const uint32_t combine_index_page_bytes = get_arg_val<uint32_t>(8);
    const uint32_t expert_counts_page_bytes = get_arg_val<uint32_t>(9);
    const uint32_t core_index = get_arg_val<uint32_t>(10);
    const uint32_t num_cores = get_arg_val<uint32_t>(11);
    const uint32_t first_tile_row = get_arg_val<uint32_t>(12);
    const uint32_t num_tile_rows = get_arg_val<uint32_t>(13);
    const uint32_t combine_stride = get_arg_val<uint32_t>(14);
    // NoC coordinates of the rectangle's first and last cores, which are contiguous, in NOC0's
    // multicast order. Cores are numbered row by row, so core c sits c places after core 0.
    const uint32_t first_core_x = get_arg_val<uint32_t>(15);
    const uint32_t first_core_y = get_arg_val<uint32_t>(16);
    const uint32_t last_core_x = get_arg_val<uint32_t>(17);
    const uint32_t last_core_y = get_arg_val<uint32_t>(18);

    const auto indices = TensorAccessor(indices_args, indices_address);
    const auto gather_index = TensorAccessor(gather_index_args, gather_index_address, gather_index_page_bytes);
    const auto combine_index = TensorAccessor(combine_index_args, combine_index_address, combine_index_page_bytes);
    const auto expert_counts = TensorAccessor(expert_counts_args, expert_counts_address, expert_counts_page_bytes);

    Noc noc;
    const uint32_t index_faces = CircularBuffer(get_named_compile_time_arg_val("index_faces_cb")).get_write_ptr();
    const uint32_t gather_scratch = CircularBuffer(get_named_compile_time_arg_val("gather_cb")).get_write_ptr();
    const uint32_t combine_scratch = CircularBuffer(get_named_compile_time_arg_val("combine_cb")).get_write_ptr();
    const uint32_t table_scratch = CircularBuffer(get_named_compile_time_arg_val("table_cb")).get_write_ptr();
    Semaphore<> counts_ready(get_named_compile_time_arg_val("counts_ready_semaphore"));
    Semaphore<> starts_ready(get_named_compile_time_arg_val("starts_ready_semaphore"));
    Semaphore<> rows_ready(get_named_compile_time_arg_val("rows_ready_semaphore"));

    const bool is_coordinator = core_index == 0;
    if (is_coordinator) {
        // The value core 0's semaphore multicast copies into every other core's starts_ready.
        starts_ready.set(1);
    }

    read_face_pairs(noc, indices, index_faces, token_start / kBlockRows + first_tile_row, num_tile_rows);
    noc.async_read_barrier();

    // The core's tokens, numbered from 0 here; the last core's last tile row may be partial.
    const uint32_t first_token = first_tile_row * kBlockRows;
    const uint32_t core_slice_tokens = num_tile_rows * kBlockRows;
    const uint32_t remaining_tokens = num_tokens - first_token;
    const uint32_t core_tokens = remaining_tokens < core_slice_tokens ? remaining_tokens : core_slice_tokens;

    const volatile tt_l1_ptr uint32_t* choices = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(index_faces);
    // topk never returns an id at or above num_experts; one that does counts as the last expert, so
    // no row is ever written out of bounds.
    const auto expert_of = [](uint32_t id) { return id < num_experts ? id : num_experts - 1; };
    const auto load_group = [choices](uint32_t token, uint32_t(&ids)[group_slots]) {
#pragma GCC unroll 16
        for (uint32_t slot = 0; slot < group_slots; ++slot) {
            ids[slot] = choices[(token + slot / top_k) * kWordsPerToken + slot % top_k];
        }
    };

    uint32_t counts[num_experts] = {};
    uint32_t token = 0;
    for (; token + group_tokens <= core_tokens; token += group_tokens) {
        uint32_t ids[group_slots];
        load_group(token, ids);
#pragma GCC unroll 16
        for (uint32_t slot = 0; slot < group_slots; ++slot) {
            ++counts[expert_of(ids[slot])];
        }
    }
    for (; token < core_tokens; ++token) {
        for (uint32_t k = 0; k < top_k; ++k) {
            ++counts[expert_of(choices[token * kWordsPerToken + k])];
        }
    }

    // The table: a row of counts per core, two to a word, gathered on core 0 and multicast back; then
    // the totals that core 0 writes out, one to a word, and a scratch word for draining stores.
    // With two or more cores, a core holds at most half a window's tile rows, rounded up, so about
    // half of its at most 65536 slots: its counts fit in 16 bits. A single core never reads the table.
    volatile tt_l1_ptr uint32_t* table = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(table_scratch);
    volatile tt_l1_ptr uint32_t* own_row = table + core_index * table_row_words;
    volatile tt_l1_ptr uint32_t* totals_row = table + num_cores * table_row_words;
    volatile tt_l1_ptr uint32_t* drain_word = totals_row + totals_row_words;
#pragma GCC unroll 32
    for (uint32_t word = 0; word < packed_row_words; ++word) {
        own_row[word] = counts[2 * word] | (2 * word + 1 < num_experts ? counts[2 * word + 1] << 16 : 0);
    }

    if (is_coordinator) {
        if (num_cores > 1) {
            counts_ready.wait(num_cores - 1);
            drain_stores(drain_word);
            // Linked, so the semaphore write reaches each core only after the table.
            noc.async_write_multicast(
                CoreLocalMem<uint32_t>(table_scratch),
                MulticastEndpoint{},
                num_cores * table_row_words * kWordBytes,
                num_cores - 1,
                {},
                {.noc_x_start = first_core_x,
                 .noc_y_start = first_core_y,
                 .noc_x_end = last_core_x,
                 .noc_y_end = last_core_y,
                 .addr = table_scratch},
                true);
            starts_ready.set_multicast(noc, first_core_x, first_core_y, last_core_x, last_core_y, num_cores - 1);
        }
    } else {
        drain_stores(drain_word);
        const uint32_t own_row_address = table_scratch + core_index * table_row_words * kWordBytes;
        noc.async_write(
            CoreLocalMem<uint32_t>(own_row_address),
            UnicastEndpoint{},
            packed_row_words * kWordBytes,
            {},
            {.noc_x = first_core_x, .noc_y = first_core_y, .addr = own_row_address});
        // The counts must be in core 0's L1 before the count of arrivals says so.
        noc.async_write_barrier();
        counts_ready.up(noc, first_core_x, first_core_y, 1);
        starts_ready.wait(1);
    }

    // Every core sums the table itself: the slots of each expert in the cores before it and in the
    // cores after it, two experts to an add. Each of the two sums leaves out this core's own slots,
    // at least one, out of at most 65536 in the window, so neither 16-bit half can carry into the
    // other. A row's loads are issued together, so their latencies overlap.
    const auto add_rows = [table](uint32_t first_core, uint32_t end_core, uint32_t(&sums)[packed_row_words]) {
        for (uint32_t core = first_core; core < end_core; ++core) {
            volatile tt_l1_ptr uint32_t* row = table + core * table_row_words;
            uint32_t words[packed_row_words];
#pragma GCC unroll 32
            for (uint32_t word = 0; word < packed_row_words; ++word) {
                words[word] = row[word];
            }
#pragma GCC unroll 32
            for (uint32_t word = 0; word < packed_row_words; ++word) {
                sums[word] += words[word];
            }
        }
    };
    uint32_t packed_before[packed_row_words] = {};
    uint32_t packed_after[packed_row_words] = {};
    add_rows(0, core_index, packed_before);
    add_rows(core_index + 1, num_cores, packed_after);
    const auto half = [](const uint32_t(&packed)[packed_row_words], uint32_t expert) {
        return expert % 2 == 0 ? packed[expert / 2] & 0xFFFF : packed[expert / 2] >> 16;
    };
    uint32_t before[num_experts];
    uint32_t totals[num_experts];
#pragma GCC unroll 64
    for (uint32_t expert = 0; expert < num_experts; ++expert) {
        before[expert] = half(packed_before, expert);
        totals[expert] = before[expert] + counts[expert] + half(packed_after, expert);
    }

    uint32_t region_starts[num_experts];
    uint32_t next_row[num_experts];
    uint32_t first_row[num_experts];
    uint32_t region_start = 0;
    for (uint32_t expert = 0; expert < num_experts; ++expert) {
        region_starts[expert] = region_start;
        region_start += round_up_to_block(totals[expert]);
        first_row[expert] = region_starts[expert] + before[expert];
        next_row[expert] = first_row[expert];
    }
    const uint32_t regions_end = region_start;

    if (is_coordinator) {
        for (uint32_t expert = 0; expert < num_experts; ++expert) {
            totals_row[expert] = totals[expert];
        }
        drain_stores(drain_word);
        noc.async_write(
            CoreLocalMem<uint32_t>(table_scratch + num_cores * table_row_words * kWordBytes),
            expert_counts,
            num_experts * kWordBytes,
            {},
            {.page_id = 0});
    }

    volatile tt_l1_ptr uint32_t* gather = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gather_scratch);
    volatile tt_l1_ptr uint32_t* combine = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(combine_scratch);
    const uint32_t first_global_token = token_start + first_token;
    token = 0;
    for (; token + group_tokens <= core_tokens; token += group_tokens) {
        uint32_t rows[group_slots];
        load_group(token, rows);
        // A scattered store takes L1 five cycles. One per slot, between the slots' counter updates,
        // drains while the core keeps working, where a burst of them would stall it.
#pragma GCC unroll 16
        for (uint32_t slot = 0; slot < group_slots; ++slot) {
            rows[slot] = next_row[expert_of(rows[slot])]++;
            gather[rows[slot]] = first_global_token + token + slot / top_k;
        }
        // Consecutive tokens of one choice fill a 16-byte block, which L1 takes as one write.
#pragma GCC unroll 16
        for (uint32_t k = 0; k < top_k; ++k) {
#pragma GCC unroll 16
            for (uint32_t member = 0; member < group_tokens; ++member) {
                combine[k * combine_stride + token + member] = rows[member * top_k + k];
            }
        }
    }
    for (; token < core_tokens; ++token) {
        for (uint32_t k = 0; k < top_k; ++k) {
            const uint32_t row = next_row[expert_of(choices[token * kWordsPerToken + k])]++;
            gather[row] = first_global_token + token;
            combine[k * combine_stride + token] = row;
        }
    }
    for (uint32_t k = 0; k < top_k; ++k) {
        for (uint32_t padding = core_tokens; padding < core_slice_tokens; ++padding) {
            combine[k * combine_stride + padding] = 0;
        }
    }
    drain_stores(drain_word);

    // Send each expert's rows to the core assembling its region, core e % num_cores, at the same
    // address there. A run can start or end inside a 16-byte block that another core also writes;
    // L1 applies such a narrow write as an atomic read-modify-write, so neither write is lost. The
    // runs go before the combine_index writes, so the barrier below waits for them alone. The
    // assembler and its NoC coordinates are walked alongside the expert, since a division takes
    // this core up to 33 cycles.
    const uint32_t num_assemblers = num_experts < num_cores ? num_experts : num_cores;
    uint32_t assembler = 0;
    uint32_t assembler_x = first_core_x;
    uint32_t assembler_y = first_core_y;
    const auto next_assembler = [&]() {
        if (++assembler == num_cores) {
            assembler = 0;
            assembler_x = first_core_x;
            assembler_y = first_core_y;
        } else if (++assembler_x > last_core_x) {
            assembler_x = first_core_x;
            ++assembler_y;
        }
    };
    for (uint32_t expert = 0; expert < num_experts; ++expert) {
        if (counts[expert] > 0 && assembler != core_index) {
            const uint32_t run_address = gather_scratch + first_row[expert] * kWordBytes;
            noc.async_write(
                CoreLocalMem<uint32_t>(run_address),
                UnicastEndpoint{},
                counts[expert] * kWordBytes,
                {},
                {.noc_x = assembler_x, .noc_y = assembler_y, .addr = run_address});
        }
        next_assembler();
    }
    // The rows must be in the assemblers' L1 before their counts of arrivals say so. Every other
    // core signals every assembler once, rows sent or not, so an assembler waits for a known count.
    noc.async_write_barrier();
    assembler = 0;
    assembler_x = first_core_x;
    assembler_y = first_core_y;
    for (uint32_t index = 0; index < num_assemblers; ++index) {
        if (assembler != core_index) {
            rows_ready.up(noc, assembler_x, assembler_y, 1);
        }
        next_assembler();
    }

    for (uint32_t k = 0; k < top_k; ++k) {
        noc.async_write(
            CoreLocalMem<uint32_t>(combine_scratch + k * combine_stride * kWordBytes),
            combine_index,
            core_slice_tokens * kWordBytes,
            {},
            {.page_id = k, .offset_bytes = first_token * kWordBytes});
    }

    // Assemble this core's regions: rows that hold no slot copy the window's first token, a real
    // row, so a gather by gather_index stays in range and finite. So do the rows after the last
    // region, which core 0 writes.
    for (uint32_t expert = core_index; expert < num_experts; expert += num_cores) {
        const uint32_t used_end = region_starts[expert] + totals[expert];
        const uint32_t region_end = region_starts[expert] + round_up_to_block(totals[expert]);
        fill_rows(gather, used_end, region_end, token_start);
    }
    if (is_coordinator) {
        fill_rows(gather, regions_end, gathered_rows, token_start);
    }
    drain_stores(drain_word);

    if (core_index < num_assemblers && num_cores > 1) {
        rows_ready.wait(num_cores - 1);
    }
    for (uint32_t expert = core_index; expert < num_experts; expert += num_cores) {
        const uint32_t region_rows = round_up_to_block(totals[expert]);
        if (region_rows > 0) {
            noc.async_write(
                CoreLocalMem<uint32_t>(gather_scratch + region_starts[expert] * kWordBytes),
                gather_index,
                region_rows * kWordBytes,
                {},
                {.page_id = 0, .offset_bytes = region_starts[expert] * kWordBytes});
        }
    }
    if (is_coordinator && gathered_rows > regions_end) {
        noc.async_write(
            CoreLocalMem<uint32_t>(gather_scratch + regions_end * kWordBytes),
            gather_index,
            (gathered_rows - regions_end) * kWordBytes,
            {},
            {.page_id = 0, .offset_bytes = regions_end * kWordBytes});
    }
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
