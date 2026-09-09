// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Pipelined direct TILE reshape reader (TTDM_RESHAPE_MAPPED_PIPELINED_READS,
// pattern backlog C08). Identical map walk, adjacent-duplicate dedup, and CB
// PUSH ORDER to reader_reshape_tile_mapped.cpp; only the NOC issue discipline
// changes: up to read_batch outstanding async reads, ONE barrier per batch
// (charter idiom "batched NOC issue, ONE barrier per batch"), then a batched
// push. The writer (writer_reshape_tile_mapped.cpp) is untouched: it still
// consumes tiles one at a time (wait_front(1)/pop_front(1)) in exactly the
// order this kernel pushes them.
//
// FIFO-wrap safety: pushes are RAGGED here (the number of distinct source
// tiles per output page varies), so `input_cb_depth % read_batch == 0` alone
// does not keep a batch contiguous in L1. `slot` tracks the absolute FIFO
// position and every batch's capacity is clamped to the distance to the wrap
// point, so `.offset_bytes = pending * tile_bytes` addressing never crosses
// the FIFO boundary.
//
// Deadlock margin: the writer holds ONE un-popped input slot across output
// pages (it pops lazily on the next distinct page / at end). reserve_back of
// more than input_cb_depth - 1 slots could therefore wait on a slot only this
// kernel can free -> read_batch < input_cb_depth is a static requirement.
#include <cstdint>
#include <limits>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"

struct SegmentMapData {
    uint32_t input_page_index;
    uint32_t input_page_offset;
    uint32_t output_page_offset;
    uint32_t num_elements;
};

void kernel_main() {
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t map_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_output_page = get_arg_val<uint32_t>(2);
    const uint32_t end_output_page = get_arg_val<uint32_t>(3);

    constexpr uint32_t map_page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t map_cb = get_compile_time_arg_val(2);
    constexpr uint32_t input_cb = get_compile_time_arg_val(3);
    constexpr uint32_t read_batch = get_compile_time_arg_val(4);
    constexpr uint32_t input_cb_depth = get_compile_time_arg_val(5);
    constexpr auto map_args = TensorAccessorArgs<6>();
    constexpr auto input_args = TensorAccessorArgs<map_args.next_compile_time_args_offset()>();
    constexpr uint32_t max_entries = map_page_bytes / sizeof(SegmentMapData);
    static_assert(read_batch >= 1, "pipelined reader needs a positive batch");
    static_assert(read_batch < input_cb_depth, "writer holds one un-popped slot; a full-depth reserve deadlocks");
    static_assert(input_cb_depth % read_batch == 0, "CB FIFO-wrap discipline (CHARTER.md contract 2)");

    const auto input_accessor = TensorAccessor(input_args, input_addr);
    const auto map_accessor = TensorAccessor(map_args, map_addr);
    Noc noc;
    CircularBuffer map_buffer(map_cb);
    CircularBuffer input_buffer(input_cb);

    // Absolute slot cursor into the input FIFO (CB pointers start at slot 0
    // on every program launch). Batches never cross the wrap point.
    uint32_t slot = 0;
    // Reads issued into the current reservation, not yet barriered/pushed.
    uint32_t pending = 0;
    uint32_t batch_capacity = 0;

    for (uint32_t output_page = start_output_page; output_page < end_output_page; ++output_page) {
        // `pending == 0` here by construction (every page flushes its tail),
        // so this barrier never publishes a half-batched input reservation.
        map_buffer.reserve_back(1);
        const uint32_t map_l1 = map_buffer.get_write_ptr();
        noc.async_read<NocOptions::DEFAULT, map_page_bytes>(
            map_accessor, map_buffer, map_page_bytes, {.page_id = output_page, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        map_buffer.push_back(1);

        CoreLocalMem<volatile SegmentMapData> segments(map_l1);
        // Reset per output page, exactly like the serial reader AND the
        // writer: a source tile shared by two consecutive output pages is
        // deliberately re-read (the writer re-waits for it).
        uint32_t previous_page = std::numeric_limits<uint32_t>::max();
        for (uint32_t index = 0; index < max_entries; ++index) {
            if (segments[index].num_elements == 0) {
                continue;
            }
            const uint32_t input_page = segments[index].input_page_index;
            if (input_page == previous_page) {
                continue;
            }
            if (pending == 0) {
                const uint32_t to_wrap = input_cb_depth - slot;
                batch_capacity = (read_batch < to_wrap) ? read_batch : to_wrap;
                input_buffer.reserve_back(batch_capacity);
            }
            noc.async_read<NocOptions::DEFAULT, tile_bytes>(
                input_accessor,
                input_buffer,
                tile_bytes,
                {.page_id = input_page, .offset_bytes = 0},
                {.offset_bytes = pending * tile_bytes});
            ++pending;
            if (pending == batch_capacity) {
                noc.async_read_barrier();
                input_buffer.push_back(pending);
                slot += pending;
                if (slot == input_cb_depth) {
                    slot = 0;
                }
                pending = 0;
            }
            previous_page = input_page;
        }
        if (pending != 0) {
            // Ragged per-page tail: same barrier+push, smaller batch. Keeping
            // the flush at page end preserves the `pending == 0` invariant at
            // the next map read above.
            noc.async_read_barrier();
            input_buffer.push_back(pending);
            slot += pending;
            if (slot == input_cb_depth) {
                slot = 0;
            }
            pending = 0;
        }
    }
}
