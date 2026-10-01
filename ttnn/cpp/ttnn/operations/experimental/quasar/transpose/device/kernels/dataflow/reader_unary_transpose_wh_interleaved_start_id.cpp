// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t start_id = get_arg(args::start_id);
    uint32_t start_ht = get_arg(args::start_ht);
    uint32_t start_wt = get_arg(args::start_wt);

    uint32_t Ht = get_arg(args::Ht);
    uint32_t Wt = get_arg(args::Wt);
    uint32_t HtWt = get_arg(args::HtWt);

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    DataflowBuffer cb(dfb::cb_in0);
    const uint32_t tile_bytes = cb.get_entry_size();
    const auto s = TensorAccessor(tensor::src);

    Noc noc;

    uint32_t ht = start_ht;
    uint32_t wt = start_wt;
    uint32_t i_tile = start_id;

    // this reader will read a NHW tensor in NWH order
    for (uint32_t i = 0; i < num_tiles; i++) {
        cb.reserve_back(onetile);
        noc.async_read(s, cb, tile_bytes, {.page_id = i_tile}, {.offset_bytes = 0});
        noc.async_read_barrier();

        cb.push_back(onetile);
        i_tile += Wt;  // stride in H
        ht += 1;
        if (ht == Ht) {
            ht = 0;
            i_tile += 1;
            wt += 1;
            if (wt == Wt) {
                wt = 0;
                i_tile -= Wt;  // Start of next batch
            } else {
                i_tile -= HtWt;  // Start of next col
            }
        }
    }
}
