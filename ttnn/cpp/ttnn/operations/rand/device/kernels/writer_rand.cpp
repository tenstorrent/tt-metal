// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 op-local writer for rand. rand's output is always FLOAT32 (rand.cpp forces FLOAT32 then
// typecasts), so this only implements the direct FLOAT32 path: stream each generated tile out of the
// intermed DataflowBuffer straight to the output tensor. The legacy uniform writer's bf16 conversion
// staging CB (dst_cb / c_0) is dropped — it was unused on the FLOAT32 path — which also removes the
// single-ended-FIFO shape the Metal 2.0 lowering rejects on a data-movement kernel.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t start_id = get_arg(args::start_id);
    const uint32_t num_tiles = get_arg(args::num_tiles);
    constexpr uint32_t page_size = get_arg(args::page_size);

    Noc noc;
    DataflowBuffer cb_intermed(dfb::cb_intermed);
    const auto out = TensorAccessor(tensor::output);

    const uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_intermed.wait_front(1);
        noc.async_write(cb_intermed, out, page_size, {.offset_bytes = 0}, {.page_id = i});
        noc.async_writes_flushed();
        cb_intermed.pop_front(1);
    }
    noc.async_write_barrier();
}
