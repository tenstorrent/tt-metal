// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/operations/data_movement/reshape_view/device/hostdevcommon/common.hpp"

using tt::data_movement::common::enhanced_noc_async_read;
using ttnn::prim::detail::SegmentMapData;
constexpr uint32_t One_Tile_Reserve = 1;

void kernel_main() {
    uint32_t input_addr = get_arg_val<uint32_t>(0);
    uint32_t map_addr = get_arg_val<uint32_t>(1);
    uint32_t start_output_page_idx = get_arg_val<uint32_t>(2);
    uint32_t end_output_page_idx = get_arg_val<uint32_t>(3);

    constexpr uint32_t Max_Map_Size_Bytes = get_compile_time_arg_val(0);
    constexpr uint32_t Tile_Size_Bytes = get_compile_time_arg_val(1);

    constexpr uint32_t mapping_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t input_cb_id = get_compile_time_arg_val(3);

    constexpr auto map_args = TensorAccessorArgs<4>();
    constexpr auto input_args = TensorAccessorArgs<map_args.next_compile_time_args_offset()>();

    constexpr uint32_t Max_Map_Entries = Max_Map_Size_Bytes / sizeof(SegmentMapData);
    constexpr uint32_t Max_Map_Elements = Max_Map_Entries * SegmentMapData::size;

    const auto input_addr_gen = TensorAccessor(input_args, input_addr, Tile_Size_Bytes);
    const auto map_addr_gen = TensorAccessor(map_args, map_addr, Max_Map_Size_Bytes);

    bool first = true;
    for (uint32_t out_page_idx = start_output_page_idx; out_page_idx < end_output_page_idx; ++out_page_idx) {
        cb_reserve_back(mapping_cb_id, One_Tile_Reserve);
        const uint64_t map_noc_addr = get_noc_addr(out_page_idx, map_addr_gen);
        const uint32_t map_addr = get_read_ptr(mapping_cb_id);
        enhanced_noc_async_read<Max_Map_Size_Bytes, true>(map_noc_addr, map_addr, Max_Map_Size_Bytes);
        noc_async_read_barrier();
        cb_push_back(mapping_cb_id, 1);

        auto map_ptr = reinterpret_cast<volatile tt_l1_ptr SegmentMapData*>(map_addr);
        uint32_t previous_input_page_idx = std::numeric_limits<uint32_t>::max();
        for (uint32_t map_idx = 0; map_idx < Max_Map_Entries; ++map_idx) {
            if (map_ptr[map_idx].num_elements == 0) {
                continue;
            }

            const uint32_t input_page_idx = map_ptr[map_idx].input_page_index;
            if (first) {
                first = false;
            } else {
                // this segment is also in a tile we've already loaded
                if (input_page_idx == previous_input_page_idx) {
                    continue;
                }
            }

            cb_reserve_back(input_cb_id, One_Tile_Reserve);
            const uint32_t input_write_addr = get_read_ptr(input_cb_id);
            const uint64_t input_page_noc_addr = get_noc_addr(input_page_idx, input_addr_gen);
            enhanced_noc_async_read<Tile_Size_Bytes, true>(input_page_noc_addr, input_write_addr, Tile_Size_Bytes);
            previous_input_page_idx = input_page_idx;
            noc_async_read_barrier();
            cb_push_back(input_cb_id, 1);
        }
    }
}
