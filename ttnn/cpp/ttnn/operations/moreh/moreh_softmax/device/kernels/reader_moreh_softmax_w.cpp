// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

#include <cstdint>

void kernel_main() {
    // Runtime args
    const std::uint32_t N = get_arg(args::num_rows);
    const std::uint32_t tile_offset = get_arg(args::tile_offset);
    const std::uint32_t Wt = get_arg(args::Wt);
    const std::uint32_t mask_w = get_arg(args::mask_w);

    // Constants
    constexpr auto dfb_in = dfb::in;
    constexpr auto dfb_mask = dfb::mask;

    // Ublocks size defined in tiles
    constexpr std::uint32_t onetile = 1;

    // Input tensor
    constexpr bool is_fp32 = get_arg(args::is_fp32) == 1;
    const auto src_in = TensorAccessor(tensor::src);

    // Generate mask tile
    const DataflowBuffer dfb_mask_obj(dfb_mask);
    if (is_fp32) {
        generate_mask_w<std::uint32_t>(dfb_mask_obj, mask_w);
    } else {
        generate_mask_w<std::uint16_t>(dfb_mask_obj, mask_w);
    }

    const Noc noc;
    DataflowBuffer dfb_in_obj(dfb_in);
    const std::uint32_t src_in_tile_bytes = dfb_in_obj.get_entry_size();

    std::uint32_t curr_tile = tile_offset;
    for (std::uint32_t i = 0; i < N; i += onetile) {
        dfb_in_obj.reserve_back(static_cast<uint16_t>(Wt));
        for (std::uint32_t w = 0; w < Wt; w++) {
            noc.async_read(
                src_in, dfb_in_obj, src_in_tile_bytes, {.page_id = curr_tile}, {.offset_bytes = w * src_in_tile_bytes});
            curr_tile++;
        }
        noc.async_read_barrier();
        dfb_in_obj.push_back(static_cast<uint16_t>(Wt));
    }
}
