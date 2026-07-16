// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t dfb_id_in0 = get_named_compile_time_arg_val("dfb_id_in");
    constexpr uint32_t num_dims = get_compile_time_arg_val(0);
    constexpr auto src_args = TensorAccessorArgs<1>();
    const uint32_t src_addr = get_common_arg_val<uint32_t>(0);

    volatile tt_l1_ptr uint32_t* num_unpadded_tiles = (volatile tt_l1_ptr uint32_t*)(get_common_arg_addr(1));
    volatile tt_l1_ptr uint32_t* num_padded_tiles = num_unpadded_tiles + num_dims;

    const uint32_t start_id = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);

    tt_l1_ptr uint32_t* id_per_dim = (tt_l1_ptr uint32_t*)(get_arg_addr(2));

    // In and out are assumed to be same dataformat
    const auto s0 = TensorAccessor(src_args, src_addr);

    // Create objects for Device 2.0 API
    DataflowBuffer dfb_in0(dfb_id_in0);
    Noc noc;

    // Get tile size from CB interface
    const uint32_t tile_size = dfb_in0.get_entry_size();

    uint32_t src_tile_id = start_id;

    for (uint32_t i = 0; i < num_tiles; ++i) {
        // Copy Input
        dfb_in0.reserve_back(1);
        noc.async_read(s0, dfb_in0, tile_size, {.page_id = src_tile_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_in0.push_back(1);
        src_tile_id++;
        for (uint32_t j = 0; j < num_dims; ++j) {
            id_per_dim[j]++;
            if (id_per_dim[j] == num_unpadded_tiles[j]) {
                id_per_dim[j] = 0;
                src_tile_id += num_padded_tiles[j];
            } else {
                break;
            }
        }
    }
}
