// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

template <typename CB>
FORCE_INLINE void generate_index_tile(CB& cb, const uint32_t wt) {
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
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t input_indices_page_size = get_arg(args::input_indices_page_size);
    constexpr uint32_t tile_height = get_arg(args::tile_height);

    constexpr uint32_t onetile = 1;
    constexpr uint32_t tile_bytes_input_values = get_tile_size(dfb::cb_input_values);

    const auto s0 = TensorAccessor(ta::values);
    const auto s1 = TensorAccessor(ta::indices);

    Noc noc;
    DataflowBuffer input_values_cb(dfb::cb_input_values);
    DataflowBuffer input_indices_cb(dfb::cb_final_indices_rm);
    DataflowBuffer cb_intermed(dfb::cb_index);

    uint32_t tile_id_input_values = 0;
    for (uint32_t i = 0; i < Ht; ++i) {
        for (uint32_t j = 0; j < Wt; ++j) {
            input_values_cb.reserve_back(onetile);
            noc.async_read(
                s0, input_values_cb, tile_bytes_input_values, {.page_id = tile_id_input_values}, {.offset_bytes = 0});
            tile_id_input_values++;
            generate_index_tile(cb_intermed, j);
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
