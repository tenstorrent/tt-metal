// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of writer_unary_interleaved_start_id.cpp.
//
// Forked because the legacy file is consumed by many ops; modifying it in
// place would break every consumer still on the legacy positional-CTA path.
// During the bulk-port window, the legacy copy and this fork coexist; the
// legacy copy is deleted once the last unmigrated consumer ports.
// See the shared-dataflow-kernel Caution in
//   docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md
//
// Differences from the legacy writer:
//   - Named DFB binding (`dfb::output`) replaces the magic CB index CTA.
//   - Named TensorBinding (`ta::output`) replaces `TensorAccessorArgs` and the
//     buffer-address RTA.
//   - Named RTAs (`num_pages`, `start_id`) replace positional `get_arg_val<>`.
//   - `get_local_cb_interface(cb_id).fifo_page_size` -> `cb.get_tile_size()`.
//
// Host bindings expected (per the porting factories' KernelSpecs):
//   compile_time_arg_bindings: none required
//   runtime_arguments_schema.named_runtime_args: { "num_pages", "start_id" }
//   dfb_bindings: { OUTPUT (CONSUMER, name="output") }
//   tensor_bindings: { OUTPUT_TENSOR (name="output") }
//
// The legacy writer's BACKWARDS and OUT_SHARDED ifdef branches are dropped from
// this fork because no current consumer needs them. If a future consumer needs
// them, re-introduce with named-CTA gates.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    auto num_pages = get_arg(args::num_pages);
    auto start_id = get_arg(args::start_id);

    DataflowBuffer cb(dfb::output);

    // Page size for both TILE and ROW_MAJOR layouts.
    const uint32_t page_bytes = cb.get_tile_size();

    Noc noc;
    const auto s = TensorAccessor(ta::output);

    // Single-page ublocks (works for both TILE and ROW_MAJOR layouts).
    constexpr uint32_t onepage = 1;

    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb.wait_front(onepage);
        noc.async_write(cb, s, page_bytes, {}, {.page_id = i});
        noc.async_writes_flushed();
        cb.pop_front(onepage);
    }
    noc.async_write_barrier();
}
