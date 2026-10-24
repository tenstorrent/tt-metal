// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Segment-mapped TILE reshape reader. Each map row lists the segments of one output tile (or one byte range of it);
// the reader loads one input tile per change of input page, in the order the writer consumes them, keeping up to
// read_batch reads in flight.
//
// Pushes are ragged (source tiles per row vary), so a batch is clamped to the distance to the FIFO wrap: `slot` tracks
// the absolute FIFO position and `pending * tile_bytes` addressing never crosses the wrap.
//
// The writer holds one input tile un-popped across rows, so reserving the full depth could wait on a slot only this
// kernel's progress frees; hence read_batch < input_cb_depth. Depth 1 reads one tile at a time: the writer pops its
// held tile before it waits for the next one.
#include <cstdint>
#include <limits>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/operations/data_movement/reshape_view/device/hostdevcommon/common.hpp"

using ttnn::prim::detail::SegmentMapData;

void kernel_main() {
    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t map_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t start_row = get_arg_val<uint32_t>(0);
    const uint32_t end_row = get_arg_val<uint32_t>(1);

    constexpr uint32_t map_page_bytes = get_named_compile_time_arg_val("map_page_bytes");
    constexpr uint32_t tile_bytes = get_named_compile_time_arg_val("tile_bytes");
    constexpr uint32_t map_cb = get_named_compile_time_arg_val("map_cb");
    constexpr uint32_t input_cb = get_named_compile_time_arg_val("input_cb");
    constexpr uint32_t read_batch = get_named_compile_time_arg_val("read_batch");
    constexpr uint32_t input_cb_depth = get_named_compile_time_arg_val("input_cb_depth");
    constexpr auto map_args = TensorAccessorArgs<0>();
    constexpr auto input_args = TensorAccessorArgs<map_args.next_compile_time_args_offset()>();
    constexpr uint32_t max_entries = map_page_bytes / sizeof(SegmentMapData);
    static_assert(read_batch >= 1, "pipelined reader needs a positive batch");
    static_assert(
        read_batch < input_cb_depth || input_cb_depth == 1,
        "writer holds one un-popped slot; a full-depth reserve deadlocks");
    static_assert(input_cb_depth % read_batch == 0, "full batches must not straddle the FIFO wrap");

    const auto input_accessor = TensorAccessor(input_args, input_addr);
    const auto map_accessor = TensorAccessor(map_args, map_addr);
    Noc noc;
    CircularBuffer map_buffer(map_cb);
    CircularBuffer input_buffer(input_cb);

    // CB pointers start at slot 0 on every launch.
    uint32_t slot = 0;
    uint32_t pending = 0;
    uint32_t batch_capacity = 0;

    auto flush = [&]() {
        noc.async_read_barrier();
        input_buffer.push_back(pending);
        slot += pending;
        if (slot == input_cb_depth) {
            slot = 0;
        }
        pending = 0;
    };

    for (uint32_t row = start_row; row < end_row; ++row) {
        // pending is 0 here (every row flushes its tail), so this barrier never publishes half a batch.
        map_buffer.reserve_back(1);
        const uint32_t map_l1 = map_buffer.get_write_ptr();
        noc.async_read<NocOptions::DEFAULT, map_page_bytes>(
            map_accessor, map_buffer, map_page_bytes, {.page_id = row, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        map_buffer.push_back(1);

        CoreLocalMem<volatile SegmentMapData> segments(map_l1);
        // Reset per row like the writer: a tile shared by consecutive rows is read again.
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
                batch_capacity = read_batch < to_wrap ? read_batch : to_wrap;
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
                flush();
            }
            previous_page = input_page;
        }
        if (pending != 0) {
            flush();
        }
    }
}
