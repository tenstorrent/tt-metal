// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// One sender per grid row reads that row's tile from DRAM and broadcasts it across the row.
// The mcast wire is decoded by MCAST_ARGS(row): every name it reads -- the two sem:: bindings and
// the named metadata and vararg base -- was written by native attachment on the host,
// so this kernel chains no CT or RT offsets.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args_metal2.hpp"

void kernel_main() {
    const uint32_t row_page = get_arg(args::row_page);

    Noc noc;
    DataflowBuffer dfb_tile(dfb::tile);
    constexpr auto mc = MCAST_ARGS(row);

    const uint32_t tile_bytes = dfb_tile.get_tile_size();

    // num_entries == 1 and one push/pop per program, so the entry address is the same on every
    // core of the row -- which is what lets the sender mcast into the receivers' entry.
    dfb_tile.reserve_back(1);
    const uint32_t entry = dfb_tile.get_write_ptr();

    if (mc.can_send()) {
        const auto acc_in = TensorAccessor(tensor::in);
        noc.async_read(acc_in, dfb_tile, tile_bytes, {.page_id = row_page}, {.offset_bytes = 0});
        noc.async_read_barrier();

        // A sender with no receivers (one-column row, single-core rectangle) still sends: that is
        // the degenerate local copy, which is a no-op here because src == dst.
        if (auto pipe = mc.optional_sender(noc)) {
            pipe->send(entry, entry, tile_bytes);
        }
    } else if (auto pipe = mc.optional_receiver(noc)) {
        // The optional faces compile on every placement, including those with no receiver at all.
        pipe->receive();
    }

    dfb_tile.push_back(1);
}
