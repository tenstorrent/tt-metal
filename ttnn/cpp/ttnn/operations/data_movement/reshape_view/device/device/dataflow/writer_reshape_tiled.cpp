// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/operations/data_movement/reshape_view/device/hostdevcommon/common.hpp"

using namespace tt::data_movement::common;
using ttnn::prim::detail::SegmentMapData;

void kernel_main() {
    const uint32_t output_base_addr = get_arg_val<uint32_t>(0);

    const uint32_t start_output_page = get_arg_val<uint32_t>(1);
    const uint32_t end_output_page = get_arg_val<uint32_t>(2);

    constexpr uint32_t Tile_size_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t Max_Map_Entries = get_compile_time_arg_val(1);
    constexpr uint8_t element_sz_bytes = get_compile_time_arg_val(2);

    constexpr uint32_t cb_id_mapping = get_compile_time_arg_val(3);
    constexpr uint32_t cb_id_input = get_compile_time_arg_val(4);
    constexpr uint32_t cb_id_working = get_compile_time_arg_val(5);  // scratch
    constexpr auto output_args = TensorAccessorArgs<6>();

    const auto output_addrgen = TensorAccessor(output_args, output_base_addr, Tile_size_bytes);

    // loop over output (reshaped) pages this core is responsible for
    bool first = true;
    cb_reserve_back(cb_id_working, 1);
    const uint32_t working_write_addr = get_write_ptr(cb_id_working);
    for (uint32_t output_page_idx = start_output_page; output_page_idx < end_output_page; ++output_page_idx) {
        cb_wait_front(cb_id_mapping, 1);
        const uint32_t map_addr = get_read_ptr(cb_id_mapping);
        auto map_ptr = reinterpret_cast<volatile tt_l1_ptr SegmentMapData*>(map_addr);
        uint32_t input_base_addr, previous_input_page_idx = std::numeric_limits<uint32_t>::max();
        for (uint32_t seg_idx = 0; seg_idx < Max_Map_Entries; ++seg_idx) {
            if (map_ptr[seg_idx].num_elements == 0) {
                if (output_page_idx == end_output_page - 1 && seg_idx == Max_Map_Entries - 1) {
                    noc_async_write_barrier();
                    cb_pop_front(cb_id_input, 1);
                }
                continue;
            }

            if (first) {
                cb_wait_front(cb_id_input, 1);
                input_base_addr = get_read_ptr(cb_id_input);
                previous_input_page_idx = map_ptr[seg_idx].input_page_index;
                first = false;

            } else if (map_ptr[seg_idx].input_page_index != previous_input_page_idx) {
                noc_async_write_barrier();
                cb_pop_front(cb_id_input, 1);
                cb_wait_front(cb_id_input, 1);
                input_base_addr = get_read_ptr(cb_id_input);
                previous_input_page_idx = map_ptr[seg_idx].input_page_index;
            }
            // TODO (maybe) pre calculate size and offsets in bytes on host
            const uint32_t output_addr = working_write_addr + map_ptr[seg_idx].output_page_offset * element_sz_bytes;
            const uint32_t input_addr = input_base_addr + map_ptr[seg_idx].input_page_offset * element_sz_bytes;
            const uint32_t szbytes = map_ptr[seg_idx].num_elements * element_sz_bytes;
            tt_memmove<false, true, false, Tile_size_bytes>(output_addr, input_addr, szbytes);
        }
        noc_async_write_barrier();

        const uint64_t output_noc_addr = get_noc_addr(output_page_idx, output_addrgen);
        enhanced_noc_async_write<Tile_size_bytes, true>(working_write_addr, output_noc_addr, Tile_size_bytes);
        noc_async_write_barrier();

        cb_pop_front(cb_id_mapping, 1);
    }
    cb_push_back(cb_id_working, 1);
}
