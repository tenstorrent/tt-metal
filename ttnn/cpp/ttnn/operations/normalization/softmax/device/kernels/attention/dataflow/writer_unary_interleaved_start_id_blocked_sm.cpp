// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "../../../../../../kernel_helper_functions/pad_tile.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    using namespace tt::constants;

    const std::uint32_t num_tiles = get_arg(args::num_tiles);
    const std::uint32_t tile_offset = get_arg(args::tile_offset);
    const std::uint32_t blk = get_arg(args::blk);

    constexpr std::uint32_t num_datum_padded = get_arg(args::num_datum_padded);
    constexpr std::uint32_t tile_hw = get_arg(args::tile_hw);

    constexpr auto dfb_id_out0 = dfb::out0;
    constexpr std::uint32_t onetile = 1;

    constexpr auto dfb_id_mask = dfb::mask_padded;
    const std::uint32_t mask_padded_data = get_arg(args::mask_padded_data);

    Noc noc;
    DataflowBuffer dfb_id_out0_obj(dfb_id_out0);
    DataflowBuffer dfb_id_mask_obj(dfb_id_mask);
    const std::uint32_t tile_bytes = dfb_id_out0_obj.get_entry_size();

    // Adds -inf padding. Note: the value is the uint16 representation of bfloat16's -inf
    constexpr std::uint16_t mask_val = 0xFF80;
    constexpr std::uint32_t mask_val_32 = ((std::uint32_t)mask_val << 16) + mask_val;
    if (mask_padded_data) {
        // generate_bcast_row_mask(dfb_id_mask, num_datum_padded, mask_val);
        std::uint32_t ptr = (dfb_id_mask_obj.get_write_ptr());
        // same pointer, but for zeroing out the tile
        volatile tt_l1_ptr std::uint16_t* zero_ptr =
            reinterpret_cast<volatile tt_l1_ptr std::uint16_t*>(dfb_id_mask_obj.get_write_ptr());
        for (std::uint32_t i = 0; i < tile_hw; i++) {
            zero_ptr[i] = 0.0f;
        }
        constexpr std::uint32_t num_datum_unpadded = 32 - num_datum_padded;
        fill_pad_tile<std::uint16_t, num_datum_unpadded, 32>(ptr, mask_val);
        dfb_id_mask_obj.push_back(1);
    }

    const auto s = TensorAccessor(tensor::dst);

    std::uint32_t tile_id = tile_offset;
    for (std::uint32_t i = 0; i < num_tiles; i += blk) {
        dfb_id_out0_obj.wait_front(blk);

        std::uint32_t read_offset = 0;
        for (std::uint32_t j = 0; j < blk; j++) {
            noc.async_write(dfb_id_out0_obj, s, tile_bytes, {.offset_bytes = read_offset}, {.page_id = tile_id});
            tile_id++;
            read_offset += tile_bytes;
        }
        noc.async_write_barrier();
        dfb_id_out0_obj.pop_front(blk);
    }
}
