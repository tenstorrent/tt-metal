// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal-2.0 (Quasar) variant of reader_reshape_tiled.cpp. Identical page-mapping data movement, but
// wired for the Metal-2 ProgramSpec path:
//   - input / mapping buffers come from bound tensor parameters (tensor::input / tensor::map), NOT from
//     RTA addresses + TensorAccessorArgs;
//   - the mapping / input CBs are bound DataflowBuffers (dfb::mapping / dfb::input) shared with the writer;
//   - args are named via get_arg(args::...).
//
// Compile args (named): max_map_size_bytes, tile_size_bytes.
// Runtime args (named): start_output_page_idx, end_output_page_idx.

#include <stdint.h>
#include <limits>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"  // [#48552] CoreLocalMem for the TRID read
#include "experimental/kernel_args.h"

#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/operations/experimental/quasar/reshape_view/device/hostdevcommon/common.hpp"

using tt::data_movement::common::enhanced_noc_async_read;
using ttnn::prim::qsr::detail::SegmentMapData;
constexpr uint32_t One_Tile_Reserve = 1;

void kernel_main() {
    const uint32_t start_output_page_idx = get_arg(args::start_output_page_idx);
    const uint32_t end_output_page_idx = get_arg(args::end_output_page_idx);

    constexpr uint32_t Max_Map_Size_Bytes = get_arg(args::max_map_size_bytes);
    constexpr uint32_t Tile_Size_Bytes = get_arg(args::tile_size_bytes);

    constexpr uint32_t Max_Map_Entries = Max_Map_Size_Bytes / sizeof(SegmentMapData);

    const auto input_addr_gen = TensorAccessor(tensor::input);
    const auto map_addr_gen = TensorAccessor(tensor::map);

    Noc noc;
    DataflowBuffer mapping_cb(dfb::mapping);
    DataflowBuffer input_cb(dfb::input);
    bool first = true;
    for (uint32_t out_page_idx = start_output_page_idx; out_page_idx < end_output_page_idx; ++out_page_idx) {
        mapping_cb.reserve_back(One_Tile_Reserve);
        uint32_t map_l1_addr;
        {
            // Hold the write lock across the NOC fill of this entry. Taken right after reserve_back, so
            // get_ptr() is the pre-push write pointer. The lock is what makes the NOC write into this DFB
            // legal -- every NOC write into a DFB region must be fully covered by a lock the writing
            // processor holds. It does NOT cover the parse below: that runs after the release (and after
            // push_back), and is coherent only because map_l1_addr is the uncached L1 alias on Quasar DM,
            // so the loads re-fetch from TL1.
            const auto map_lock = mapping_cb.scoped_write_lock(One_Tile_Reserve);
            const auto map_mem = map_lock.get_ptr();  // CoreLocalMem: the NOC dst, no rewrap needed
            map_l1_addr = static_cast<uint32_t>(map_mem.get_address());
            // [#48552] async_read_barrier is a no-op on Quasar (scmdbuf_tr_ack stubbed) and a repeated read
            // into the reused num_entries=1 slot doesn't re-deliver. Use the TRID read +
            // is_read_trid_flushed poll (the mechanism padded_slice relies on) so the transaction actually
            // completes before we parse.
            constexpr uint8_t map_trid = 1;
            noc.async_read<NocOptions::TXN_ID>(
                map_addr_gen,
                map_mem,
                Max_Map_Size_Bytes,
                {.page_id = out_page_idx, .offset_bytes = 0},
                {.offset_bytes = 0},
                NocOptVals{.trid = map_trid});
            while (!noc.is_read_trid_flushed(map_trid)) {
            }
        }
        mapping_cb.push_back(1);

        auto map_ptr = reinterpret_cast<volatile tt_l1_ptr SegmentMapData*>(map_l1_addr);
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

            input_cb.reserve_back(One_Tile_Reserve);
            {
                // Same pattern as the map read: the write lock covers the NOC fill of this entry.
                const auto input_lock = input_cb.scoped_write_lock(One_Tile_Reserve);
                // [#48552] TRID read (see map read above) — the input_cb slot is also reused
                // (num_entries=1), so the same no-op-barrier / no-re-deliver problem applies. Poll
                // is_read_trid_flushed for completion.
                constexpr uint8_t input_trid = 2;
                noc.async_read<NocOptions::TXN_ID>(
                    input_addr_gen,
                    input_lock.get_ptr(),
                    Tile_Size_Bytes,
                    {.page_id = input_page_idx, .offset_bytes = 0},
                    {.offset_bytes = 0},
                    NocOptVals{.trid = input_trid});
                previous_input_page_idx = input_page_idx;
                while (!noc.is_read_trid_flushed(input_trid)) {
                }
            }
            input_cb.push_back(1);
        }
    }
}
