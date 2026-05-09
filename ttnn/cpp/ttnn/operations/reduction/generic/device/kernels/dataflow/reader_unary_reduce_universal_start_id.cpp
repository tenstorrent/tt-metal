// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 reader for the multi-core / single-core reduction primitive (also
// used by Welford W).
//
// Migration notes:
//   - Compile-time arguments are bound by name (args::scaler_bits) and resolved
//     through the auto-generated kernel_args_generated.h.
//   - Runtime arguments are bound by name (args::num_tiles, args::start_id);
//     per-node values come from the host's ProgramRunParams.
//   - The dataflow buffers are bound by name (dfb::input, dfb::scaler).
//   - The input tensor is bound by name (ta::input_tensor) — the host declares
//     a TensorParameter in the ProgramSpec and supplies a MeshTensor reference
//     via ProgramRunParams::TensorArg, so address generation no longer needs
//     is_dram or aligned_page_size as kernel-side arguments.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/dataflow_buffer.h"
#include "experimental/noc.h"
#include "experimental/tensor.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    // Per-node runtime arguments (named).
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t start_id = get_arg(args::start_id);

    // Compile-time arguments (named).
    constexpr uint32_t scaler_bits = get_arg(args::scaler_bits);

    experimental::DataflowBuffer dfb_input(dfb::input);

    // Fill the scaler tile. The helper is templated on the scaler buffer id (CB id
    // on Gen1, DFB id on Gen2 — both constexpr through dfb::scaler.id) and uses
    // the arch-agnostic DataflowBuffer wrapper internally.
    constexpr uint32_t scaler_buf_id = dfb::scaler.id;
    const float scaler_f = __builtin_bit_cast(float, scaler_bits);
    dataflow_kernel_lib::prepare_reduce_scaler<scaler_buf_id, REDUCE_OP, REDUCE_DIM>(scaler_f);

    // TensorAccessor built from the Metal 2.0 tensor binding (ta::input_tensor).
    // The host-side TensorParameter / TensorArg pair supplies the underlying buffer
    // address through the kernel's TensorBinding common-runtime-arg slot.
    TensorAccessor input_accessor(ta::input_tensor);
    const uint32_t tile_bytes = dfb_input.get_tile_size();

    experimental::Noc noc;

    constexpr uint32_t onetile = 1;
    for (uint32_t i = start_id; i < start_id + num_tiles; ++i) {
        dfb_input.reserve_back(onetile);
        noc.async_read(input_accessor, dfb_input, tile_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_input.push_back(onetile);
    }
}
