// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Grad-only reader for backward ops whose compute consumes just grad_output.
// Pairs with the ternary writer; mirrors ternary_reader_nobcast_ttt.cpp but
// without input/other reads so cb_input/cb_other backpressure can't deadlock.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);

    constexpr auto src0_args = TensorAccessorArgs<1, 0>();

    Noc noc;
    DataflowBuffer dfb0(cb_id_in0);

    const uint32_t tile_bytes_0 = dfb0.get_entry_size();
    const auto s0 = TensorAccessor(src0_args, src0_addr);
    constexpr uint32_t onetile = 1;

    for (uint32_t tile_id = start_id; tile_id < start_id + num_tiles; tile_id++) {
        dfb0.reserve_back(onetile);
        noc.async_read(s0, dfb0, tile_bytes_0, {.page_id = tile_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb0.push_back(onetile);
    }
}
