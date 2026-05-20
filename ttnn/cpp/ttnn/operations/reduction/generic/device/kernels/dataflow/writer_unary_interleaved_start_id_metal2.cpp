// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of
// ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp.
//
// Forked because the legacy file is consumed by many ops; modifying it in place
// would break every other consumer on the legacy positional-CTA path. This fork
// is used only by the Metal 2.0 reduction-op ports.
//
// Differences from the legacy writer:
//   - Named arg retrieval via get_arg(args::name)
//   - Named DFB binding (dfb::output) replaces magic CB index
//   - TensorBinding (ta::output) replaces TensorAccessorArgs + buffer-address RTA
//   - get_local_cb_interface(cb_id).fifo_page_size -> derive from DFB wrapper
//
// Note: this fork drops the BACKWARDS and OUT_SHARDED ifdef paths, which the
// reduction ports do not use. If those paths are needed in the future, they can
// be reintroduced with named-CTA gates.
//
// Host bindings expected:
//   runtime_arguments_schema.named_runtime_args: { "num_pages", "start_id" }
//   dfb_bindings: { OUTPUT (CONSUMER, name="output") }
//   tensor_bindings: { OUTPUT_TENSOR (name="output") }

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    auto num_pages = get_arg(args::num_pages);
    auto start_id = get_arg(args::start_id);

    DataflowBuffer cb(dfb::output);

    // Page size: DFB wrapper provides get_tile_size() on WH/BH (DATA_FORMATS_DEFINED).
    // For the reduction outputs, the DFB's entry size is the tile size.
    const uint32_t page_bytes = cb.get_tile_size();

    Noc noc;
    const auto s = TensorAccessor(ta::output);

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
