// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 port of the HC tiled-interleaved padding-aware reader (op-private copy). The shared HC
// padding-aware reader is still consumed positionally by the legacy permute_tiled program factory and
// must not be touched, so the transpose HC interleaved path carries its own copy here. Only the binding
// mechanism changed: the src0 CB id and the padding CB id come from the DFB tokens (dfb::), the input
// address from the TensorAccessor binding (ta::), the named compile-time scalars from named compile-time
// args (args::, constexpr), and the per-core work descriptors from named runtime args (args::). The read
// loop and the padding-fill epilogue are preserved unchanged.
//
// The padding DFB is always bound (the host always declares it) so the dfb::padding token always exists;
// the `if constexpr (needs_padding)` guard still elides the fill at compile time when there is no
// padding, leaving the always-present 1-entry DFB unused.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t start_id = get_arg(args::start_id);

    constexpr uint32_t num_writes = get_arg(args::num_writes);
    constexpr uint32_t padding_val_packed = get_arg(args::padding_val_packed);
    constexpr uint32_t needs_padding = get_arg(args::needs_padding) == 1;
    constexpr uint32_t swap_hw = get_arg(args::swap_hw) == 1;
    constexpr uint32_t H = get_arg(args::H);
    constexpr uint32_t W = get_arg(args::W);
    constexpr uint32_t accumulated_outer_dims = get_arg(args::accumulated_outer_dims);
    constexpr uint32_t TILE_HEIGHT = get_arg(args::tile_height);
    constexpr uint32_t TILE_WIDTH = get_arg(args::tile_width);

    constexpr uint32_t H_p = tt::data_movement::common::round_up<H, TILE_HEIGHT>();
    constexpr uint32_t W_p = tt::data_movement::common::round_up<W, TILE_WIDTH>();

    constexpr uint32_t Wt = W_p / TILE_WIDTH;
    constexpr uint32_t Ht = H_p / TILE_HEIGHT;

    constexpr uint32_t HtWt = Ht * Wt;

    constexpr uint32_t cb_id_in0 = dfb::src0;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    const auto s = TensorAccessor(ta::src_args);

    Noc noc;
    CircularBuffer cb(cb_id_in0);
    CircularBuffer cb_padding(dfb::padding);
    const uint32_t tile_bytes = cb.get_tile_size();

// read a ublock of tiles from src to CB, and then push the ublock to unpacker
#ifdef BACKWARDS
    uint32_t end_id = start_id - num_tiles;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        uint32_t linear_tile_index = 0;
        if constexpr (swap_hw) {
            uint32_t rem = i;
            uint32_t ht = rem % Ht;
            rem /= Ht;
            uint32_t wt = rem % Wt;
            rem /= Wt;
            uint32_t offset = rem % accumulated_outer_dims;
            linear_tile_index = offset * HtWt + ht * Wt + wt;
        } else {
            linear_tile_index = i;
        }
        cb.reserve_back(onetile);
        noc.async_read(s, cb, tile_bytes, {.page_id = linear_tile_index}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb.push_back(onetile);
    }
    if constexpr (needs_padding) {
        // Add padding
        cb_padding.reserve_back(1);
        uint32_t l1_write_addr = cb_padding.get_write_ptr();
        // Fill with padding value
        // if bfloat16 num_writes = FACE_WIDTH / (sizeof(uint32_t))/(element_size)
        tt::data_movement::common::fill_with_val(l1_write_addr, num_writes, padding_val_packed);
        cb_padding.push_back(1);
    }
}
