// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "experimental/kernel_args.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"

#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/operations/data_movement/reshape_view/device/hostdevcommon/common.hpp"

using tt::data_movement::common::enhanced_noc_async_read;
using ttnn::prim::detail::SegmentMapData;
constexpr uint32_t One_Tile_Reserve = 1;

void kernel_main() {
    uint32_t start_output_page_idx = get_arg(args::start_output_page_idx);
    uint32_t end_output_page_idx = get_arg(args::end_output_page_idx);

    constexpr uint32_t Max_Map_Size_Bytes = get_arg(args::Max_Map_Size_Bytes);
    constexpr uint32_t Tile_size_bytes = get_arg(args::Tile_size_bytes);

    constexpr uint32_t Max_Map_Entries = Max_Map_Size_Bytes / sizeof(SegmentMapData);
    constexpr uint32_t Max_Map_Elements = Max_Map_Entries * SegmentMapData::size;

    const auto input_addr_gen = TensorAccessor(tensor::input);
    const auto map_addr_gen = TensorAccessor(tensor::map);

    Noc noc;
    DataflowBuffer dfb_mapping(dfb::mapping);
    DataflowBuffer dfb_in_tiles(dfb::in_tiles);
    bool first = true;
    for (uint32_t out_page_idx = start_output_page_idx; out_page_idx < end_output_page_idx; ++out_page_idx) {
        dfb_mapping.reserve_back(One_Tile_Reserve);
        const uint64_t map_noc_addr = map_addr_gen.get_noc_addr(out_page_idx);
        const uint32_t map_addr = dfb_mapping.get_write_ptr();
        enhanced_noc_async_read<Max_Map_Size_Bytes, true>(noc, map_noc_addr, map_addr, Max_Map_Size_Bytes);
        noc.async_read_barrier();
        dfb_mapping.push_back(1);

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

            dfb_in_tiles.reserve_back(One_Tile_Reserve);
            const uint32_t input_write_addr = dfb_in_tiles.get_write_ptr();
            const uint64_t input_page_noc_addr = input_addr_gen.get_noc_addr(input_page_idx);
            enhanced_noc_async_read<Tile_size_bytes, true>(noc, input_page_noc_addr, input_write_addr, Tile_size_bytes);
            previous_input_page_idx = input_page_idx;
            noc.async_read_barrier();
            dfb_in_tiles.push_back(1);
        }
    }
}
