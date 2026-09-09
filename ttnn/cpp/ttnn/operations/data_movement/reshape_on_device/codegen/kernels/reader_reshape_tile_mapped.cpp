// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Direct TILE reshape reader. One mapping page describes the logical segments
// needed for one physical output tile; each distinct input tile is loaded once.
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
    constexpr auto map_args = TensorAccessorArgs<4>();
    constexpr auto input_args = TensorAccessorArgs<map_args.next_compile_time_args_offset()>();
    constexpr uint32_t max_entries = map_page_bytes / sizeof(SegmentMapData);

    const auto input_accessor = TensorAccessor(input_args, input_addr);
    const auto map_accessor = TensorAccessor(map_args, map_addr);
    Noc noc;
    CircularBuffer map_buffer(map_cb);
    CircularBuffer input_buffer(input_cb);

    for (uint32_t output_page = start_output_page; output_page < end_output_page; ++output_page) {
        map_buffer.reserve_back(1);
        const uint32_t map_l1 = map_buffer.get_write_ptr();
        noc.async_read<NocOptions::DEFAULT, map_page_bytes>(
            map_accessor, map_buffer, map_page_bytes, {.page_id = output_page, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        map_buffer.push_back(1);

        CoreLocalMem<volatile SegmentMapData> segments(map_l1);
        uint32_t previous_page = std::numeric_limits<uint32_t>::max();
        for (uint32_t index = 0; index < max_entries; ++index) {
            if (segments[index].num_elements == 0) {
                continue;
            }
            const uint32_t input_page = segments[index].input_page_index;
            if (input_page == previous_page) {
                continue;
            }
            input_buffer.reserve_back(1);
            noc.async_read<NocOptions::DEFAULT, tile_bytes>(
                input_accessor,
                input_buffer,
                tile_bytes,
                {.page_id = input_page, .offset_bytes = 0},
                {.offset_bytes = 0});
            noc.async_read_barrier();
            input_buffer.push_back(1);
            previous_page = input_page;
        }
    }
}
