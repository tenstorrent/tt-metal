// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Clone tilized-interleaved reader, ported to Metal 2.0.
//
// Host bindings expected (per CloneOperation::ProgramFactory's KernelSpec):
//   runtime_arguments_schema.named_runtime_args: { "num_tiles", "start_id" }
//   dfb_bindings: { INPUT_DFB (PRODUCER, name="src_dfb") }
//   tensor_bindings: { INPUT_TENSOR (name="input") }

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    auto num_tiles = get_arg(args::num_tiles);
    auto start_id = get_arg(args::start_id);

    DataflowBuffer src_dfb(dfb::src_dfb);
    Noc noc;
    const auto s = TensorAccessor(ta::input);
    const uint32_t tile_bytes = src_dfb.get_tile_size();

    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
        src_dfb.reserve_back(1);
        noc.async_read(s, src_dfb, tile_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        src_dfb.push_back(1);
    }
}
