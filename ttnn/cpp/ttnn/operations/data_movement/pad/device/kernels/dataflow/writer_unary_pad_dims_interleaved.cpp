// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 port. Used only by the PadTileCore factory, so ported in place.
// Logic unchanged from the legacy writer; only the access mechanism moves to named bindings:
//   - the reader->writer input DFB id  -> dfb::in0   (consumer endpoint)
//   - the pad-scratch DFB id           -> dfb::pad   (self-loop: producer + consumer)
//   - the output tensor address        -> ta::out
//   - positional runtime args          -> get_arg(args::...)
// The pad DFB has no FIFO peer (the writer is its only user): the writer reserves one entry of L1
// scratch in it, fills it with the pad value, and reuses it as the NoC source for every padded
// tile. It is bound as a self-loop (producer + consumer on this writer) to satisfy the
// producer/consumer invariant.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

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

    constexpr uint32_t cb_id_out0 = dfb::in0;
    constexpr uint32_t cb_id_out1 = dfb::pad;

    const uint32_t tile_size = get_tile_size(cb_id_out0);

    const auto s1 = TensorAccessor(ta::out);
    Noc noc;
    DataflowBuffer cb_out0(dfb::in0);
    DataflowBuffer cb_out1(dfb::pad);

    cb_out1.reserve_back(1);  // in this kernel we are not pushing anything into CBs, just using the space

    uint32_t pad_buffer_l1_addr = cb_out1.get_write_ptr();

    // Fill pad tile with pad value
    volatile tt_l1_ptr uint32_t* pad_buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pad_buffer_l1_addr);
    const uint32_t num_elems = tile_size / sizeof(uint32_t);
    for (uint32_t z = 0; z < num_elems; z++) {
        pad_buffer[z] = pad_value;
    }

    uint32_t src_tile_id = 0;
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
