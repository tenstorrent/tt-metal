// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    // The planned recipe arrives as compile-time varargs; prepare it once, before compute needs it.
    using Auxiliary =
        ttnn::kernel_lib::BoundReduceAuxiliaryArgs<ttnn::kernel_lib::ReduceAuxiliaryArgs<0>, dfb::auxiliary>;
    dataflow_kernel_lib::prepare_reduce_auxiliary_tiles<Auxiliary>();

    const auto dst = TensorAccessor(tensor::dst);
    Noc noc;
    DataflowBuffer out(dfb::out);
    for (uint32_t tile = 0; tile < get_arg(args::num_output_tiles); ++tile) {
        out.wait_front(1);
        noc.async_write(out, dst, out.get_tile_size(), {.offset_bytes = 0}, {.page_id = tile});
        noc.async_write_barrier();
        out.pop_front(1);
    }
}
