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

// Reads num_tiles consecutive tiles of input I, starting at tile page_id, into the DFB.
// Only input I's accessor is built, so stack use does not grow with the number of inputs.
template <uint32_t I>
void read_input_tiles(
    Noc& noc, DataflowBuffer& dfb_in, uint32_t tile_size_bytes, uint32_t page_id, uint32_t num_tiles) {
    constexpr uint32_t ublock_size_tiles = 1;
    const auto accessor = TensorAccessor(std::get<I>(tensor::inputs));
    for (uint32_t t = 0; t < num_tiles; ++t) {
        dfb_in.reserve_back(ublock_size_tiles);
        noc.async_read(
            accessor, CoreLocalMem<uint8_t>(dfb_in.get_write_ptr()), tile_size_bytes, {.page_id = page_id + t}, {});
        noc.async_read_barrier();
        dfb_in.push_back(ublock_size_tiles);
    }
}

// Calls read_input_tiles<tensor_idx>() for an input index known only at run time.
template <uint32_t... Is>
void read_input_tiles(
    uint32_t tensor_idx,
    Noc& noc,
    DataflowBuffer& dfb_in,
    uint32_t tile_size_bytes,
    uint32_t page_id,
    uint32_t num_tiles,
    std::integer_sequence<uint32_t, Is...>) {
    (void)((tensor_idx == Is && (read_input_tiles<Is>(noc, dfb_in, tile_size_bytes, page_id, num_tiles), true)) || ...);
}

// Reads num_tiles tiles into the bound Dataflow Buffer in L1, taking them round-robin from the
// inputs: num_tiles_per_block[t] tiles from input t, then the next input.
// Expects n input tensor bindings, reached positionally through the `inputs` binding sequence.
void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t start_tensor = get_arg(args::start_tensor);
    const uint32_t start_tensor_id = get_arg(args::start_tensor_id);

    // The tensor binding sequence carries its own length, so the host passes no tensor count.
    constexpr uint32_t num_tensors = std::tuple_size_v<decltype(tensor::inputs)>;

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

    uint32_t curr_tensor = start_tensor;
    uint32_t curr_tensor_id = start_tensor_id;
    uint32_t num_wraps = 0;  // times curr_tensor wrapped from the last input back to input 0
    uint32_t tiles_left = num_tiles;
    while (tiles_left > 0) {
        const uint32_t num_tiles_per_block = get_vararg(curr_tensor);

        // Tiles of curr_tensor this core has already read: one block per earlier visit, less the part
        // of the first block skipped by starting mid-block. Inputs before start_tensor are first
        // visited after the first wrap.
        uint32_t tiles_already_read = num_wraps * num_tiles_per_block + curr_tensor_id;
        if (curr_tensor < start_tensor) {
            tiles_already_read -= num_tiles_per_block;
        } else if (curr_tensor == start_tensor) {
            tiles_already_read -= start_tensor_id;
        }
        const uint32_t page_id = get_vararg(tile_id_per_tensor_offset + curr_tensor) + tiles_already_read;

        uint32_t tiles_to_read = num_tiles_per_block - curr_tensor_id;
        if (tiles_to_read > tiles_left) {
            tiles_to_read = tiles_left;
        }
        read_input_tiles(
            curr_tensor,
            noc,
            dfb_in,
            tile_size_bytes,
            page_id,
            tiles_to_read,
            std::make_integer_sequence<uint32_t, num_tensors>());
        tiles_left -= tiles_to_read;
        curr_tensor_id += tiles_to_read;

        if (curr_tensor_id == num_tiles_per_block) {
            curr_tensor_id = 0;
            curr_tensor++;
            if (curr_tensor == num_tensors) {
                curr_tensor = 0;
                num_wraps++;
            }
        }
    }
}
