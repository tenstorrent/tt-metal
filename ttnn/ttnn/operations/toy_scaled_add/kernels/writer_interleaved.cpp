// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// toy_scaled_add writer for an interleaved output (BRISC, NoC1).
//
// Writes this core's tile-rows in the order compute packs them. A flush is enough to hand a CB slot
// back (the bytes have left L1); the one write barrier at the end guarantees they landed before the
// program completes.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/operations/toy_scaled_add/device/kernels/toy_scaled_add_args.hpp"

using namespace toy_scaled_add;

void kernel_main() {
    constexpr uint32_t Wt = get_named_compile_time_arg_val("Wt");
    constexpr auto out_args = TensorAccessorArgs<0>();

    const uint32_t row_start = get_arg_val<uint32_t>(core_arg::ROW_START);
    const uint32_t num_rows = get_arg_val<uint32_t>(core_arg::NUM_ROWS);

    Noc noc;
    CircularBuffer cb_out(cb::OUT);
    const uint32_t out_tile_bytes = get_tile_size(cb::OUT);
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(writer_arg::OUT_ADDR), out_tile_bytes);

    const uint32_t tile_end = (row_start + num_rows) * Wt;
    for (uint32_t tile = row_start * Wt; tile < tile_end; ++tile) {
        cb_out.wait_front(1);
        noc.async_write(cb_out, out, out_tile_bytes, {}, {.page_id = tile});
        noc.async_writes_flushed();
        cb_out.pop_front(1);
    }
    noc.async_write_barrier();
}
