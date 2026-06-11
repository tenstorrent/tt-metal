// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 op-private copy of writer_unary_pad_dims_interleaved.cpp (single-core tiled pad writer).
// Binding changes only (data-movement logic is unchanged):
//   - the output CB id comes from dfb::cb_id_out0; the pad scratchpad CB from dfb::cb_id_out1
//   - the output tensor base address comes from the TensorAccessor binding (ta::dst_args)
//   - the per-core counts and the packed pad value are named runtime args (args::)

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t num_unpadded_W = get_arg(args::num_unpadded_W);
    const uint32_t num_padded_Wt = get_arg(args::num_padded_Wt);
    const uint32_t num_unpadded_Z = get_arg(args::num_unpadded_Z);
    const uint32_t num_padded_Zt = get_arg(args::num_padded_Zt);
    const uint32_t num_unpadded_Yt = get_arg(args::num_unpadded_Yt);
    const uint32_t num_padded_Yt = get_arg(args::num_padded_Yt);
    const uint32_t num_unpadded_Xt = get_arg(args::num_unpadded_Xt);
    const uint32_t num_padded_Xt = get_arg(args::num_padded_Xt);
    const uint32_t pad_value = get_arg(args::pad_value);

    constexpr uint32_t cb_id_out0 = dfb::cb_id_out0;
    constexpr uint32_t cb_id_out1 = dfb::cb_id_out1;

    const uint32_t tile_size = get_tile_size(cb_id_out0);

    const auto s1 = TensorAccessor(ta::dst_args);
    Noc noc;
    CircularBuffer cb_out0(cb_id_out0);
    CircularBuffer cb_out1(cb_id_out1);

    cb_out1.reserve_back(1);  // in this kernel we are not pushing anything into CBs, just using the space

    uint32_t pad_buffer_l1_addr = cb_out1.get_write_ptr();

    // Fill pad tile with pad value
    volatile tt_l1_ptr uint32_t* pad_buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pad_buffer_l1_addr);
    const uint32_t num_elems = tile_size / sizeof(uint32_t);
    for (uint32_t z = 0; z < num_elems; z++) {
        pad_buffer[z] = pad_value;
    }

    uint32_t dst_tile_id = 0;

    CoreLocalMem<uint32_t> pad_src(pad_buffer_l1_addr);
    auto pad_tiles = [&](uint32_t num_tiles) {
        for (uint32_t pad_tile = 0; pad_tile < num_tiles; pad_tile++) {
            noc.async_write(pad_src, s1, tile_size, {.offset_bytes = 0}, {.page_id = dst_tile_id, .offset_bytes = 0});
            dst_tile_id++;
        }
        noc.async_write_barrier();
    };

    for (uint32_t w = 0; w < num_unpadded_W; w++) {
        for (uint32_t z = 0; z < num_unpadded_Z; z++) {
            for (uint32_t yt = 0; yt < num_unpadded_Yt; yt++) {
                for (uint32_t xt = 0; xt < num_unpadded_Xt; xt++) {
                    cb_out0.wait_front(1);
                    noc.async_write(
                        cb_out0, s1, tile_size, {.offset_bytes = 0}, {.page_id = dst_tile_id, .offset_bytes = 0});
                    noc.async_write_barrier();
                    cb_out0.pop_front(1);
                    dst_tile_id++;
                }
                pad_tiles(num_padded_Xt);
            }
            pad_tiles(num_padded_Yt);
        }
        pad_tiles(num_padded_Zt);
    }
    pad_tiles(num_padded_Wt);
}
