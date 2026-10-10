// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Multi-thread, implicit-sync DFB writer for binary_ng's interleaved output. Thread t of N writes the
// core's output tiles t, t + N, ...; the strided DFB hands it exactly those, in order. Each TXN_ID write
// drains one entry and acks it when it lands.

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t start_tile_id = get_arg(args::start_tile_id);
    const uint32_t num_tiles = get_arg(args::num_tiles);

    Noc noc;
    DataflowBuffer dfb_out(dfb::out);
    const auto dst = TensorAccessor(tensor::out);

    for (uint32_t k = get_my_thread_id(); k < num_tiles; k += get_num_threads()) {
        noc.async_write<NocOptions::TXN_ID>(dfb_out, dst, {}, {.page_id = start_tile_id + k});
    }
}
