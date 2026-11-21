// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// typecast reader (load_block): this cluster's tiles start_id .. start_id + num_tiles - 1 of the input, dealt
// round-robin over the R reader harts (hart p reads tiles p, p+R, ...), into cb_input_tiles.
//
// Implicit sync: one tagged read per ring entry; the DM0 interrupt handler posts the credits, so there is no
// reserve / barrier / push. That per-entry call is the transport's unit; the block structure lives in the
// compute kernel's per-block wait/pop on the ring.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t start_id = get_arg(args::start_id);
    constexpr uint32_t R = get_arg(args::reader_threads);

    DataflowBuffer cb_input_tiles(dfb::cb_input_tiles);
    Noc noc;
    const auto input_accessor = TensorAccessor(tensor::input);

    for (uint32_t i = get_my_thread_id(); i < num_tiles; i += R) {
        noc.async_read<NocOptions::TXN_ID>(input_accessor, cb_input_tiles, {.page_id = start_id + i}, {});
    }
    // Hand the hart back with the user id in its (sticky) tag register.
    noc_async_read_set_trid(0, noc.get_noc_id());
}
