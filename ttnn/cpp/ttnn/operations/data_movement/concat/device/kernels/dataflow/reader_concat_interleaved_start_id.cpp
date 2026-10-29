// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <utility>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/data_movement/concat/device/kernels/dataflow/concat_reader_common.hpp"

// Reads num_tiles tiles into the bound Dataflow Buffer in L1, taking them round-robin from the
// inputs: num_tiles_per_block[t] tiles from input t, then the next input.
// Expects n input tensor bindings, reached positionally through the `inputs` binding sequence.
void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t start_tensor = get_arg(args::start_tensor);
    const uint32_t start_tensor_id = get_arg(args::start_tensor_id);

    // The tensor binding sequence carries its own length, so the host passes no tensor count.
    constexpr uint32_t num_tensors = std::tuple_size_v<decltype(tensor::inputs)>;
    constexpr auto tensor_indices = std::make_integer_sequence<uint32_t, num_tensors>();

    // Two num_tensors-element runtime vararg blocks, in the order the host supplies them:
    // num_tiles_per_block first, then each input's first tile id on this core. They are read when
    // needed instead of being copied to per-input stack arrays: with many inputs, per-input stack
    // objects overflow the small kernel stack (8 KB shared with TLS per DM core on Quasar).
    constexpr uint32_t tile_id_per_tensor_offset = num_tensors;

    DataflowBuffer dfb_in(dfb::in);
    // The tile size comes off the buffer object; the legacy free function took a circular-buffer
    // index, which no longer exists. Read after the buffer is constructed, since the value is not a
    // constant expression and so takes the member-getter form.
    const uint32_t tile_size_bytes = dfb_in.get_tile_size();
    Noc noc;

    constexpr uint32_t ublock_size_tiles = 1;
    concat_reader::RoundRobinCursor cursor(start_tensor, start_tensor_id);
    uint32_t tiles_left = num_tiles;
    while (tiles_left > 0) {
        const uint32_t num_tiles_per_block = get_vararg(cursor.tensor);
        const uint32_t page_id =
            get_vararg(tile_id_per_tensor_offset + cursor.tensor) + cursor.pages_already_read(num_tiles_per_block);

        uint32_t tiles_to_read = num_tiles_per_block - cursor.id_in_block;
        if (tiles_to_read > tiles_left) {
            tiles_to_read = tiles_left;
        }
        concat_reader::with_input(cursor.tensor, tensor_indices, [&](auto I) {
            const auto accessor = TensorAccessor(std::get<decltype(I)::value>(tensor::inputs));
            for (uint32_t t = 0; t < tiles_to_read; ++t) {
                dfb_in.reserve_back(ublock_size_tiles);
                noc.async_read(
                    accessor,
                    CoreLocalMem<uint8_t>(dfb_in.get_write_ptr()),
                    tile_size_bytes,
                    {.page_id = page_id + t},
                    {});
                noc.async_read_barrier();
                dfb_in.push_back(ublock_size_tiles);
            }
        });
        tiles_left -= tiles_to_read;
        cursor.advance(tiles_to_read, num_tiles_per_block, num_tensors);
    }
}
