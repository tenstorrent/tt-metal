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
    // Wt is required so rem matches compute/reader (per-row), not flat num_tiles.
    // When Wt % blk != 0 but (num_rows*Wt) % blk == 0, a flat rem clamp deadlocks.
    const std::uint32_t Wt = get_arg(args::Wt);

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
    // num_rows below divides by Wt; guard against a zero-work core rather than trapping on 0/0.
    if (num_tiles > 0 && Wt > 0) {
        // Uniform blocks tile the CB capacity, so a row that blk divides needs no realignment.
        const std::uint32_t out0_pad = (Wt % blk == 0) ? 0 : (((blk * 2) - (Wt % (blk * 2))) % (blk * 2));
        const std::uint32_t num_rows = num_tiles / Wt;
        for (std::uint32_t row = 0; row < num_rows; ++row) {
            for (std::uint32_t wt = 0; wt < Wt; wt += blk) {
                const std::uint32_t rem = (wt + blk > Wt) ? (Wt - wt) : blk;  // clamped final block of each row
                dfb_id_out0_obj.wait_front(rem);

                std::uint32_t read_offset = 0;
                for (std::uint32_t j = 0; j < rem; j++) {
                    noc.async_write(
                        dfb_id_out0_obj, s, tile_bytes, {.offset_bytes = read_offset}, {.page_id = tile_id});
                    tile_id++;
                    read_offset += tile_bytes;
                }
                noc.async_write_barrier();
                dfb_id_out0_obj.pop_front(rem);
            }
            if (out0_pad > 0) {
                dfb_id_out0_obj.wait_front(out0_pad);
                dfb_id_out0_obj.pop_front(out0_pad);
            }
        }
    }
}
