// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/index_tile_dataflow.hpp"

#include "experimental/kernel_args.h"
void kernel_main() {
    constexpr auto Ht = get_arg(args::Ht);
    constexpr auto Wt = get_arg(args::Wt);
    constexpr auto input_indices_page_size = get_arg(args::input_indices_page_size);
    constexpr auto tile_height = get_arg(args::tile_height);
    constexpr bool use_32bit_index = get_arg(args::use_32bit_index) == 1;
    // Number of logical users (== number of running cores). Only this many input-index rows exist
    // and are streamed in, even though the values tile is padded to a full tile_height.
    constexpr auto num_users = get_arg(args::num_users);

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;

    const auto s0 = TensorAccessor(tensor::input_values);

    const auto s1 = TensorAccessor(tensor::input_indices);

    Noc noc;
    DataflowBuffer input_values_dfb(dfb::input_values);
    DataflowBuffer input_indices_dfb(dfb::input_indices);
    const uint32_t tile_bytes_input_values = input_values_dfb.get_entry_size();

    uint32_t tile_id_input_values = 0;
    uint32_t tile_id_input_indices = 0;
    for (uint32_t i = 0; i < Ht; ++i) {
        // input values TILE
        for (uint32_t j = 0; j < Wt; ++j) {
            input_values_dfb.reserve_back(onetile);
            noc.async_read(
                s0, input_values_dfb, tile_bytes_input_values, {.page_id = tile_id_input_values}, {.offset_bytes = 0});
            tile_id_input_values++;
            if constexpr (use_32bit_index) {
                dataflow_kernel_lib::generate_index_tile<uint32_t>(dfb::index, j);
            } else {
                dataflow_kernel_lib::generate_index_tile<uint16_t>(dfb::index, j);
            }
            noc.async_read_barrier();
            input_values_dfb.push_back(onetile);
        }
    }

    // input indices RM — push one stick per running core/user. Previously hard-coded to
    // Ht * tile_height (== 32); now `num_users` so fewer-than-32-user configs don't over-read.
    for (uint32_t j = 0; j < num_users; ++j) {
        input_indices_dfb.reserve_back(onetile);
        noc.async_read(s1, input_indices_dfb, input_indices_page_size, {.page_id = j}, {.offset_bytes = 0});
        noc.async_read_barrier();
        input_indices_dfb.push_back(onetile);
    }
}
