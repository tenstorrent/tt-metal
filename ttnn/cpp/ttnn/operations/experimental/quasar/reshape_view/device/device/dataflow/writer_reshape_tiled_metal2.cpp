// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal-2.0 (Quasar) variant of writer_reshape_tiled.cpp. Identical segment-assembly data movement, but
// wired for the Metal-2 ProgramSpec path:
//   - output buffer comes from a bound tensor parameter (tensor::dst), NOT from an RTA address +
//     TensorAccessorArgs;
//   - the mapping / input CBs are bound DataflowBuffers (dfb::mapping / dfb::input) shared with the reader;
//   - the working (scratch) page is a private node-local Scratchpad (scratch::working), NOT a DFB -- a DM
//     kernel that both fills and drains a DFB is an unsupported producer+consumer self-loop on Gen2/Quasar;
//   - args are named via get_arg(args::...).
//
// Compile args (named): tile_size_bytes, max_map_entries, element_sz_bytes.
// Runtime args (named): start_output_page, end_output_page.

#include <stdint.h>
#include <limits>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "experimental/kernel_args.h"

#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/operations/experimental/quasar/reshape_view/device/hostdevcommon/common.hpp"

using namespace tt::data_movement::common;
using ttnn::prim::qsr::detail::SegmentMapData;

void kernel_main() {
    constexpr uint32_t Tile_size_bytes = get_arg(args::tile_size_bytes);
    constexpr uint32_t Max_Map_Entries = get_arg(args::max_map_entries);
    constexpr uint8_t element_sz_bytes = get_arg(args::element_sz_bytes);

    const uint32_t start_output_page = get_arg(args::start_output_page);
    const uint32_t end_output_page = get_arg(args::end_output_page);

    const auto output_addrgen = TensorAccessor(tensor::dst);

    Noc noc;
    DataflowBuffer mapping_cb(dfb::mapping);
    DataflowBuffer input_cb(dfb::input);
    // Private node-local L1 scratch page (raw memory, no producer/consumer credit semantics). Replaces the
    // legacy cb_id_working single-kernel scratch CB (which would be a Gen2 self-loop).
    Scratchpad<uint8_t> working(scratch::working);
    const uint32_t working_write_addr = working.get_base_address();

    // loop over output (reshaped) pages this core is responsible for
    bool first = true;
    for (uint32_t output_page_idx = start_output_page; output_page_idx < end_output_page; ++output_page_idx) {
        mapping_cb.wait_front(1);
        const uint32_t map_addr = mapping_cb.get_read_ptr();
        auto map_ptr = reinterpret_cast<volatile tt_l1_ptr SegmentMapData*>(map_addr);
        uint32_t input_base_addr, previous_input_page_idx = std::numeric_limits<uint32_t>::max();
        for (uint32_t seg_idx = 0; seg_idx < Max_Map_Entries; ++seg_idx) {
            if (map_ptr[seg_idx].num_elements == 0) {
                continue;
            }

            if (first) {
                input_cb.wait_front(1);
                input_base_addr = input_cb.get_read_ptr();
                previous_input_page_idx = map_ptr[seg_idx].input_page_index;
                first = false;

            } else if (map_ptr[seg_idx].input_page_index != previous_input_page_idx) {
                noc.async_write_barrier();
                input_cb.pop_front(1);
                input_cb.wait_front(1);
                input_base_addr = input_cb.get_read_ptr();
                previous_input_page_idx = map_ptr[seg_idx].input_page_index;
            }
            // TODO (maybe) pre calculate size and offsets in bytes on host
            const uint32_t output_addr = working_write_addr + map_ptr[seg_idx].output_page_offset * element_sz_bytes;
            const uint32_t input_addr = input_base_addr + map_ptr[seg_idx].input_page_offset * element_sz_bytes;
            const uint32_t szbytes = map_ptr[seg_idx].num_elements * element_sz_bytes;
            tt_memmove<false, true, false, Tile_size_bytes>(noc, output_addr, input_addr, szbytes);
        }
        noc.async_write_barrier();

        const uint64_t output_noc_addr = output_addrgen.get_noc_addr(output_page_idx);
        enhanced_noc_async_write<Tile_size_bytes, true>(noc, working_write_addr, output_noc_addr, Tile_size_bytes);
        noc.async_write_barrier();

        mapping_cb.pop_front(1);
    }
    // The per-transition pop only releases the previous input page's tile, so the final input
    // tile waited inside the loop is still held here. Pop it once to leave the input CB balanced.
    // Gated on `first` so a core that processed no segments does not pop a tile it never waited.
    if (!first) {
        noc.async_write_barrier();
        input_cb.pop_front(1);
    }
}
