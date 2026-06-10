// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
/**
 * add a cb full of indices for the tile
 * each row is identical in the index tensor, so we just need to add an offset based on which row tile it is
 * first 32 elements are {0,..31}, then next 32 are {32,..64}
 * wt is which tile it is along the row [0, Wt) so j + 32*wt is the value in the tile at each element
 */
FORCE_INLINE void generate_index_tile(const uint32_t cb_id, const uint32_t wt) {
    // TODO: investigate moving to compile time (binary size is at risk)
    CircularBuffer cb(cb_id);
    cb.reserve_back(1);
    CoreLocalMem<volatile uint32_t> ptr(cb.get_write_ptr());
    uint16_t wt_offset = wt << 5;

    uint32_t count = 0;
    for (uint32_t i = 0; i < 2; ++i) {
        for (uint32_t j = 0; j < 2; ++j) {
            for (uint32_t k = 0; k < 16; ++k) {
                for (uint32_t l = 0; l < 16; l += 2) {
                    uint16_t value = l + 16 * j + wt_offset;
                    ptr[count] = (value + 1) << 16 | value;
                    count++;
                }
            }
        }
    }
    cb.push_back(1);
}

void kernel_main() {
    // Metal 2.0: tensor addresses come from the TensorAccessor bindings (ta::), CB ids from the DFB
    // tokens (dfb::), and the shape/work-split scalars from named compile-time args (args::).
    constexpr uint32_t input_values_cb_index = dfb::input_values;
    constexpr uint32_t input_indices_cb_index = dfb::final_indices;
    constexpr uint32_t cb_intermed_index = dfb::index;

    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t input_indices_page_size = get_arg(args::final_indices_stick_size);
    constexpr uint32_t tile_height = get_arg(args::tile_height);

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;

    const auto s0 = TensorAccessor(ta::input_values);

    const auto s1 = TensorAccessor(ta::input_indices);

    Noc noc;
    CircularBuffer input_values_cb(input_values_cb_index);
    CircularBuffer input_indices_cb(input_indices_cb_index);
    const uint32_t tile_bytes_input_values = input_values_cb.get_tile_size();

    uint32_t tile_id_input_values = 0;
    uint32_t tile_id_input_indices = 0;
    for (uint32_t i = 0; i < Ht; ++i) {
        // input values TILE
        for (uint32_t j = 0; j < Wt; ++j) {
            input_values_cb.reserve_back(onetile);
            noc.async_read(
                s0, input_values_cb, tile_bytes_input_values, {.page_id = tile_id_input_values}, {.offset_bytes = 0});
            tile_id_input_values++;
            generate_index_tile(cb_intermed_index, j);
            noc.async_read_barrier();
            input_values_cb.push_back(onetile);
        }
    }

    // input indices RM
    for (uint32_t j = 0; j < Ht * tile_height; ++j) {
        input_indices_cb.reserve_back(onetile);
        noc.async_read(s1, input_indices_cb, input_indices_page_size, {.page_id = j}, {.offset_bytes = 0});
        noc.async_read_barrier();
        input_indices_cb.push_back(onetile);
    }
}
