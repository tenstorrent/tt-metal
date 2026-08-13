// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of reader_unary_transpose_hc_interleaved_tiled_padding_aware.cpp (legacy copy alongside).
// Device-side NoC logic is unchanged; resource access uses Metal 2.0 named handles:
//   - src buffer address (legacy RTA 0)     -> tensor::input (TensorBinding)
//   - num_tiles / start_id (legacy RTA 1/2)  -> named RTAs
//   - c_0 (legacy magic CB index)            -> dfb::cb_in0 (DFBBinding)
//   - c_1 padding CB (conditional)           -> dfb::cb_pad, gated by NEEDS_PADDING define
//   - needs_padding CTA -> NEEDS_PADDING preprocessor define (gates the conditional cb_pad binding)
// Forked (not modified in place) because the legacy copy is still used by the transpose op.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t start_id = get_arg(args::start_id);

    constexpr uint32_t num_writes = get_arg(args::num_writes);
    constexpr uint32_t padding_val_packed = get_arg(args::padding_val_packed);
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

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    const auto s = TensorAccessor(tensor::input);

    Noc noc;
    DataflowBuffer dfb(dfb::cb_in0);
    const uint32_t tile_bytes = dfb.get_entry_size();

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
        dfb.reserve_back(onetile);
        noc.async_read(s, dfb, tile_bytes, {.page_id = linear_tile_index}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb.push_back(onetile);
    }
#ifdef NEEDS_PADDING
    {
        // Add padding (cb_pad, legacy c_1)
        DataflowBuffer dfb_padding(dfb::cb_pad);
        dfb_padding.reserve_back(1);
        uint32_t l1_write_addr = dfb_padding.get_write_ptr();
        // Fill with padding value
        // if bfloat16 num_writes = FACE_WIDTH / (sizeof(uint32_t))/(element_size)
        tt::data_movement::common::fill_with_val(l1_write_addr, num_writes, padding_val_packed);
        dfb_padding.push_back(1);
    }
#endif
}
