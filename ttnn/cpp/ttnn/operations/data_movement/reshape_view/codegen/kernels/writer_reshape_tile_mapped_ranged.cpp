// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Segment-mapped TILE reshape writer for sliced maps. Each map row covers one byte range of an output tile and leads
// with a header entry {output_page, range_start_bytes, range_bytes, 0} that the readers skip (num_elements == 0).
// The core assembles its segments at their natural offsets in a zeroed scratch tile and writes only its range.
//
// The ranges of one output page are disjoint, 16-byte aligned and cover the whole physical tile, so the cores sharing
// a page never write the same byte and every padding byte is zeroed by exactly one of them. The scratch offset equals
// the page offset, so source and destination stay congruent for the NoC write.
#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/operations/data_movement/reshape_view/codegen/kernels/segment_map.h"

using namespace codegen_kernels;

void kernel_main() {
    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t start_row = get_arg_val<uint32_t>(0);
    const uint32_t end_row = get_arg_val<uint32_t>(1);

    constexpr uint32_t tile_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t max_entries = get_compile_time_arg_val(1);
    constexpr uint32_t element_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t map_cb = get_compile_time_arg_val(3);
    constexpr uint32_t input_cb = get_compile_time_arg_val(4);
    constexpr uint32_t output_cb = get_compile_time_arg_val(5);
    constexpr auto output_args = TensorAccessorArgs<6>();
    static_assert(max_entries >= 1, "sliced map rows lead with a header entry");

    const auto output_accessor = TensorAccessor(output_args, output_addr);
    Noc noc;
    CircularBuffer output_scratch(output_cb);
    CircularBuffer map_buffer(map_cb);
    CircularBuffer input_buffer(input_cb);
    output_scratch.reserve_back(1);
    const uint32_t output_l1 = output_scratch.get_write_ptr();
    const CoreLocalMem<uint8_t> output_mem(output_l1);

    bool have_input = false;
    for (uint32_t row = start_row; row < end_row; ++row) {
        // Segments cover only logical datums; the range's padding must not inherit the previous row's bytes.
        noc.async_write_zeros(output_scratch, tile_bytes);
        noc.write_zeros_l1_barrier();

        map_buffer.wait_front(1);
        invalidate_l1_cache();
        CoreLocalMem<volatile SegmentMapData> segments(map_buffer.get_read_ptr());
        const uint32_t output_page = segments[0].input_page_index;
        const uint32_t range_start = segments[0].input_page_offset;
        const uint32_t range_bytes = segments[0].output_page_offset;
        assemble_segments<tile_bytes, max_entries, element_bytes>(
            noc, input_buffer, segments, 1, output_l1, have_input);
        noc.async_write<NocOptions::DEFAULT, tile_bytes>(
            output_mem,
            output_accessor,
            range_bytes,
            {.offset_bytes = range_start},
            {.page_id = output_page, .offset_bytes = range_start});
        noc.async_writes_flushed();
        map_buffer.pop_front(1);
    }
    noc.async_write_barrier();
    if (have_input) {
        input_buffer.pop_front(1);
    }
    output_scratch.push_back(1);
}
