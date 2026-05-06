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
//   - Buffer sync (reserve/push) goes through `experimental::DataflowBuffer`, which is
//     arch-agnostic (Gen1 forwards to circular_buffer_interface, Gen2 drives real DFB
//     hardware). `prepare_reduce_scaler` is parameterized on the buffer id, which on
//     Gen1 is the CB id and on Gen2 is the DFB id — both resolve at compile time from
//     `dfb::scaler.id`.
//
// Arch coverage caveat:
//   Address generation goes through `InterleavedAddrGenFast` and `noc_async_read_tile`,
//   which are part of the Gen1 dataflow API. Quasar support for the reader/writer
//   data path requires either:
//     (a) the Metal 2.0 framework supporting the positional CTAs that `TensorAccessor`
//         needs (so we could use `experimental::Noc::async_read(tensor_accessor, dfb,
//         ...)` which has noc_traits specializations on both archs), or
//     (b) a `noc_traits_t<InterleavedAddrGenFast<...>>` specialization upstream so the
//         arch-agnostic `experimental::Noc` API can drive interleaved address gen.
//   Neither is in place today; the host factory still routes to the Gen1 DM config.
//   See METAL2_MIGRATION_NOTES.md (#1) for the framework-level discussion.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
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

    // Dataflow-buffer wrapper for the input. Arch-agnostic on the sync side (Gen1
    // forwards to circular_buffer_interface, Gen2 drives real DFB hardware).
    experimental::DataflowBuffer input_buf(dfb::input);

    // Fill the scaler tile. The helper is templated on the scaler buffer id (CB id
    // on Gen1, DFB id on Gen2 — both constexpr through `dfb::scaler.id`) and uses
    // the arch-agnostic DataflowBuffer wrapper internally.
    constexpr uint32_t scaler_buf_id = dfb::scaler.id;
    const float scaler_f = __builtin_bit_cast(float, scaler_bits);
    dataflow_kernel_lib::prepare_reduce_scaler<scaler_buf_id, REDUCE_OP, REDUCE_DIM>(scaler_f);

    // Page-size and data-format come from the input buffer metadata (host-side
    // data_format is set on the DataflowBufferSpec; on Gen1 this round-trips through
    // the underlying CB).
    const InterleavedAddrGenFast<is_dram> s = {
        .bank_base_address = src_addr,
        .page_size = aligned_page_size,
        .data_format = get_dataformat(input_buf.get_id()),
    };

    constexpr uint32_t onetile = 1;
    for (uint32_t i = start_id; i < start_id + num_tiles; ++i) {
        input_buf.reserve_back(onetile);
        const uint32_t wptr = input_buf.get_write_ptr();
        noc_async_read_tile(i, s, wptr);
        noc_async_read_barrier();
        input_buf.push_back(onetile);
    }
}
