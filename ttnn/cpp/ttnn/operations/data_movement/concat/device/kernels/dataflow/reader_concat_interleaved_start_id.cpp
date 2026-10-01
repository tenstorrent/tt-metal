// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// Make n reads defined by num_reads
// Writes to the bound Dataflow Buffer in L1
// Expects n input tensor bindings, reached positionally through the `inputs` binding sequence
void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t start_tensor = get_arg(args::start_tensor);
    const uint32_t start_tensor_id = get_arg(args::start_tensor_id);

    // The tensor binding sequence carries its own length, so the host passes no tensor count.
    constexpr uint32_t num_tensors = std::tuple_size_v<decltype(tensor::inputs)>;

    // ublocks size defined in tiles
    constexpr uint32_t ublock_size_tiles = 1;

    uint32_t num_tiles_per_block[num_tensors];
    uint32_t tile_id_per_tensor[num_tensors];

    auto tensor_accessors_tuple = make_tensor_accessors(tensor::inputs);
    auto abstract_tensor_accessor_wrappers = make_abstract_tensor_accessor_wrappers(tensor_accessors_tuple);

    // Two num_tensors-element runtime vararg blocks, in the order the host supplies them:
    // num_tiles_per_block first, then tile_id_per_tensor.
    constexpr uint32_t tile_id_per_tensor_offset = num_tensors;
    for (uint32_t i = 0; i < num_tensors; ++i) {
        num_tiles_per_block[i] = get_vararg(i);
        tile_id_per_tensor[i] = get_vararg(tile_id_per_tensor_offset + i);
    }

    DataflowBuffer dfb_in(dfb::in);
    // The tile size comes off the buffer object; the legacy free function took a circular-buffer
    // index, which no longer exists. Read after the buffer is constructed, since the value is not a
    // constant expression and so takes the member-getter form.
    const uint32_t tile_size_bytes = dfb_in.get_tile_size();
    Noc noc;

    uint32_t curr_tensor = start_tensor;
    uint32_t curr_tensor_id = start_tensor_id;
    for (uint32_t i = 0; i < num_tiles; ++i) {
        dfb_in.reserve_back(ublock_size_tiles);
        uint32_t l1_write_addr = dfb_in.get_write_ptr();
        noc.async_read(
            abstract_tensor_accessor_wrappers[curr_tensor],
            CoreLocalMem<uint8_t>(l1_write_addr),
            tile_size_bytes,
            {.page_id = tile_id_per_tensor[curr_tensor]},
            {});
        noc.async_read_barrier();
        dfb_in.push_back(ublock_size_tiles);

        tile_id_per_tensor[curr_tensor]++;
        curr_tensor_id++;

        if (curr_tensor_id == num_tiles_per_block[curr_tensor]) {
            curr_tensor_id = 0;
            curr_tensor++;
            if (curr_tensor == num_tensors) {
                curr_tensor = 0;
            }
        }
    }
}
