// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    Noc noc;

    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t start_id = get_arg(args::start_id);

    DataflowBuffer dfb_output(dfb::out);

    const uint32_t output_tile_bytes = dfb_output.get_tile_size();
    const auto s = TensorAccessor(tensor::dst);

    uint32_t output_curr_id = start_id;

#ifdef OUT_SHARDED
    dfb_output.wait_front(num_tiles);
#else
    for (uint32_t i = 0; i < num_tiles; ++i) {
        dfb_output.wait_front(1);
        uint32_t l1_read_addr = dfb_output.get_read_ptr();
        noc.async_write(CoreLocalMem<uint32_t>(l1_read_addr), s, output_tile_bytes, {}, {.page_id = output_curr_id});
        noc.async_write_barrier();
        dfb_output.pop_front(1);
        output_curr_id++;
    }
#endif
}
