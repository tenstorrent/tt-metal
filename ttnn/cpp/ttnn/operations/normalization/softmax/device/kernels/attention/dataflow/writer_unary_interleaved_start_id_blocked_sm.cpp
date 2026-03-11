// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "../../../../../../kernel_helper_functions/pad_tile.hpp"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    using namespace tt::constants;

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const uint32_t tile_offset = get_arg_val<uint32_t>(2);
    const uint32_t blk = get_arg_val<uint32_t>(3);

    constexpr uint32_t num_datum_padded = get_compile_time_arg_val(0);
    constexpr uint32_t tile_hw = get_compile_time_arg_val(1);
    constexpr auto dst_args = TensorAccessorArgs<2>();

    constexpr uint32_t cb_id_out0 = tt::CBIndex::c_11;
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_out0);

    constexpr uint32_t cb_id_mask = tt::CBIndex::c_5;
    const uint32_t mask_padded_data = get_arg_val<uint32_t>(4);
    // const uint32_t num_datum_padded = get_arg_val<uint32_t>(5);

    experimental::Noc noc;
    experimental::CircularBuffer cb_id_out0_obj(cb_id_out0);
    experimental::CircularBuffer cb_id_mask_obj(cb_id_mask);

    // Adds -inf padding. Note: the value is the uint16 representation of bfloat16's -inf
    constexpr uint16_t mask_val = 0xFF80;
    constexpr uint32_t mask_val_32 = ((uint32_t)mask_val << 16) + mask_val;
    if (mask_padded_data) {
        // generate_bcast_row_mask(cb_id_mask, num_datum_padded, mask_val);
        uint32_t ptr = (cb_id_mask_obj.get_write_ptr());
        // same pointer, but for zeroing out the tile
        volatile tt_l1_ptr uint16_t* zero_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_id_mask_obj.get_write_ptr());
        for (uint32_t i = 0; i < tile_hw; i++) {
            zero_ptr[i] = 0.0f;
        }
        constexpr uint32_t num_datum_unpadded = 32 - num_datum_padded;
        fill_pad_tile<uint16_t, num_datum_unpadded, 32>(ptr, mask_val);
        cb_id_mask_obj.push_back(1);
    }

    const auto s = TensorAccessor(dst_args, dst_addr, tile_bytes);

    uint32_t tile_id = tile_offset;
    for (uint32_t i = 0; i < num_tiles; i += blk) {
        cb_id_out0_obj.wait_front(blk);

        uint32_t read_offset = 0;
        for (uint32_t j = 0; j < blk; j++) {
            noc.async_write(cb_id_out0_obj, s, tile_bytes, {.offset_bytes = read_offset}, {.page_id = tile_id});
            tile_id++;
            read_offset += tile_bytes;
        }
        noc.async_write_barrier();
        cb_id_out0_obj.pop_front(blk);
    }
}
