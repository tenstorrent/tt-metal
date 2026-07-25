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
#include "api/debug/dprint.h"  // [#48552 DEBUG] reshape_tiled DM->DM root-cause
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
    uint32_t dbg_pushes = 0;                                                       // [#48552 DEBUG]
    DPRINT("RRD start op=[{},{})\n", start_output_page_idx, end_output_page_idx);  // [#48552 DEBUG]
    // [#48552 DEBUG] One-time probe: the FIRST NOC read of a kernel is known to deliver (startup latency
    // covers it). Read output page (start+1)'s map as this kernel's first read, into a scratch L1 addr. If
    // seg0_in_pg is page-(start+1)'s CORRECT value -> DRAM map is right per-page and the bug is device-side
    // (later reads into the reused slot don't deliver). If it's 0 (== page-start's value) -> the host map
    // itself is wrong. Uses input_cb's slot as raw scratch (no DFB state touched; overwritten later).
    if (start_output_page_idx + 1 < end_output_page_idx) {
        const uint32_t probe_l1 = input_cb.get_write_ptr();
        const uint64_t probe_noc = map_addr_gen.get_noc_addr(start_output_page_idx + 1);
        enhanced_noc_async_read<Max_Map_Size_Bytes, true>(noc, probe_noc, probe_l1, Max_Map_Size_Bytes);
        noc.async_read_barrier();
        for (volatile uint32_t d = 0; d < 50000; ++d) {
            asm volatile("nop");
        }
        invalidate_l1_cache();
        auto pm = reinterpret_cast<volatile tt_l1_ptr SegmentMapData*>(probe_l1);
        DPRINT(
            "RRDprobe page={} seg0_in_pg={} n={}\n",
            start_output_page_idx + 1,
            pm[0].input_page_index,
            pm[0].num_elements);
    }
    for (uint32_t out_page_idx = start_output_page_idx; out_page_idx < end_output_page_idx; ++out_page_idx) {
        mapping_cb.reserve_back(One_Tile_Reserve);
        const uint64_t map_noc_addr = map_addr_gen.get_noc_addr(out_page_idx);
        const uint32_t map_l1_addr = mapping_cb.get_write_ptr();
        enhanced_noc_async_read<Max_Map_Size_Bytes, true>(noc, map_noc_addr, map_l1_addr, Max_Map_Size_Bytes);
        noc.async_read_barrier();
        // [#48552 DEBUG] The 300k-nop delay ALONE did NOT de-stale the map -> not pure DMA latency. Next
        // hypothesis: the no-op barrier's invalidate_l1_cache() runs BEFORE the DMA lands, so the RISC keeps a
        // stale cached L1 line. Small delay to let the DMA land, THEN invalidate the cache right before we read.
        // If seg0_in_pg now VARIES per page -> stale-cache-after-late-landing is the cause. If still stale ->
        // the repeated read to the same L1 slot (281088) genuinely isn't delivering.
        for (volatile uint32_t d = 0; d < 50000; ++d) {
            asm volatile("nop");
        }
        invalidate_l1_cache();
        // [#48552 DEBUG] map_noc lo/hi + l1 slot + first segment's input page as read from L1 -> is the map
        // read advancing per output page, and does the L1 slot actually hold this page's map?
        {
            auto dbg_map = reinterpret_cast<volatile tt_l1_ptr SegmentMapData*>(map_l1_addr);
            DPRINT(
                "RRDmap op={} noc_lo={} noc_hi={} l1={} seg0_in_pg={} seg0_n={}\n",
                out_page_idx,
                (uint32_t)map_noc_addr,
                (uint32_t)(map_noc_addr >> 32),
                map_l1_addr,
                dbg_map[0].input_page_index,
                dbg_map[0].num_elements);
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
            const uint32_t input_write_addr = input_cb.get_write_ptr();
            const uint64_t input_page_noc_addr = input_addr_gen.get_noc_addr(input_page_idx);
            enhanced_noc_async_read<Tile_Size_Bytes, true>(noc, input_page_noc_addr, input_write_addr, Tile_Size_Bytes);
            previous_input_page_idx = input_page_idx;
            noc.async_read_barrier();
            input_cb.push_back(1);
            DPRINT(
                "RRD op={} push#{} in_pg={} waddr={}\n",
                out_page_idx,
                dbg_pushes,
                input_page_idx,
                input_write_addr);  // [#48552 DEBUG]
            dbg_pushes++;
        }
    }
    DPRINT("RRD END total_pushes={}\n", dbg_pushes);  // [#48552 DEBUG]
}
