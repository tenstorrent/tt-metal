// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 reader for the multi-core reduction primitive.
//
// Migration notes:
//   - Compile-time arguments are bound by name (`args::*`) and resolved through the
//     auto-generated `kernel_args_generated.h`.
//   - Runtime arguments are bound by name (`args::src_addr`, `args::num_tiles`,
//     `args::start_id`); per-node values come from the host's ProgramRunParams.
//   - The dataflow buffers are bound by name (`dfb::input`, `dfb::scaler`); placement
//     is derived from kernel bindings, not specified per-CB on the host.
//   - Address generation goes through `InterleavedAddrGenFast` rather than
//     `TensorAccessor`, because Metal 2.0 ProgramSpec does not currently support the
//     positional compile-time arguments that `TensorAccessorArgs<N>()` reads from.
//     Sharded buffers are therefore not supported by this Metal 2.0 reader yet; the
//     host factory enforces this.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "experimental/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    // Per-node runtime arguments (named).
    const uint32_t src_addr = get_arg(args::src_addr);
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t start_id = get_arg(args::start_id);

    // Compile-time arguments (named, baked into kernel_args_generated.h).
    constexpr uint32_t scaler_bits = get_arg(args::scaler_bits);
    constexpr uint32_t aligned_page_size = get_arg(args::aligned_page_size);
    constexpr bool is_dram = get_arg(args::is_dram) != 0;

    // Dataflow buffers (named bindings).
    experimental::DataflowBuffer cb_input(dfb::input);

    // Fill the scaler tile via the helper. The helper template is parameterized on the
    // scaler CB id; on Gen1 the DFB accessor id is the underlying CB id, so we can
    // forward `dfb::scaler.id` directly.
    constexpr uint32_t scaler_cb_id = dfb::scaler.id;
    const float scaler_f = __builtin_bit_cast(float, scaler_bits);
    dataflow_kernel_lib::prepare_reduce_scaler<scaler_cb_id, REDUCE_OP, REDUCE_DIM>(scaler_f);

    // Page-size and data-format come from the input DFB metadata (host-side data_format
    // is set on the DataflowBufferSpec; on Gen1 this round-trips through the CB).
    const InterleavedAddrGenFast<is_dram> s = {
        .bank_base_address = src_addr,
        .page_size = aligned_page_size,
        .data_format = get_dataformat(cb_input.get_id()),
    };

    constexpr uint32_t onetile = 1;
    for (uint32_t i = start_id; i < start_id + num_tiles; ++i) {
        cb_input.reserve_back(onetile);
        noc_async_read_tile(i, s, cb_input.get_write_ptr());
        noc_async_read_barrier();
        cb_input.push_back(onetile);
    }
}
