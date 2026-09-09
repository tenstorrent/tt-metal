// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Direct TILE reshape writer. It assembles each output tile in one scratch CB
// from the host-computed logical segment map, then writes the physical tile.
#include <cstdint>
#include <limits>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

using namespace tt::data_movement::common;

struct SegmentMapData {
    uint32_t input_page_index;
    uint32_t input_page_offset;
    uint32_t output_page_offset;
    uint32_t num_elements;
};

void kernel_main() {
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_output_page = get_arg_val<uint32_t>(1);
    const uint32_t end_output_page = get_arg_val<uint32_t>(2);

    constexpr uint32_t tile_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t max_entries = get_compile_time_arg_val(1);
    constexpr uint32_t element_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t map_cb = get_compile_time_arg_val(3);
    constexpr uint32_t input_cb = get_compile_time_arg_val(4);
    constexpr uint32_t output_cb = get_compile_time_arg_val(5);
    constexpr auto output_args = TensorAccessorArgs<6>();

    const auto output_accessor = TensorAccessor(output_args, output_addr);
    Noc noc;
    CircularBuffer output_scratch(output_cb);
    CircularBuffer map_buffer(map_cb);
    CircularBuffer input_buffer(input_cb);
    output_scratch.reserve_back(1);
    // Raw L1 base retained: tt_memmove assembles the tile via raw L1 addresses.
    const uint32_t output_l1 = output_scratch.get_write_ptr();
    // Same reserved slot, as a Device 2.0 local-L1 handle for the tile write out.
    const CoreLocalMem<uint8_t> output_mem(output_l1);

    bool have_input = false;
    uint32_t input_l1 = 0;
    uint32_t previous_page = std::numeric_limits<uint32_t>::max();
    for (uint32_t output_page = start_output_page; output_page < end_output_page; ++output_page) {
        // Only logical elements have mapping segments. Clear the complete
        // physical tile before assembly so partial H/W padding never inherits
        // stale L1 data from the preceding output page.
        noc.async_write_zeros(output_scratch, tile_bytes);
        noc.write_zeros_l1_barrier();

        map_buffer.wait_front(1);
        const uint32_t map_l1 = map_buffer.get_read_ptr();
        CoreLocalMem<volatile SegmentMapData> segments(map_l1);
        previous_page = std::numeric_limits<uint32_t>::max();

        for (uint32_t index = 0; index < max_entries; ++index) {
            if (segments[index].num_elements == 0) {
                continue;
            }
            if (segments[index].input_page_index != previous_page) {
                if (have_input) {
                    noc.async_write_barrier();
                    input_buffer.pop_front(1);
                }
                input_buffer.wait_front(1);
                input_l1 = input_buffer.get_read_ptr();
                previous_page = segments[index].input_page_index;
                have_input = true;
            }
            const uint32_t dst = output_l1 + segments[index].output_page_offset * element_bytes;
            const uint32_t src = input_l1 + segments[index].input_page_offset * element_bytes;
            const uint32_t bytes = segments[index].num_elements * element_bytes;
            tt_memmove<false, true, false, tile_bytes>(noc, dst, src, bytes);
        }
        noc.async_write_barrier();
        noc.async_write<NocOptions::DEFAULT, tile_bytes>(
            output_mem, output_accessor, tile_bytes,
            {.offset_bytes = 0}, {.page_id = output_page, .offset_bytes = 0});
        noc.async_write_barrier();
        map_buffer.pop_front(1);
    }
    if (have_input) {
        noc.async_write_barrier();
        input_buffer.pop_front(1);
    }
    output_scratch.push_back(1);
}
