// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 slice rm-stride nd writer. Slice-local copy of writer_multicore_slice_nd.cpp;
// bindings are Metal 2.0 (dfb::cb_in consumer, ta::dst, named CTAs/RTAs, output_dims vararg).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t cb_id_in = dfb::cb_in;
    constexpr uint32_t tensor_rank = get_named_compile_time_arg_val("rank");
    constexpr uint32_t element_size = get_named_compile_time_arg_val("element_size");

    const uint32_t num_rows_for_this_core = get_arg(args::num_rows);
    const uint32_t start_row_for_this_core = get_arg(args::start_row);

    // output_dims[tensor_rank] as runtime varargs.
    uint32_t output_dims[tensor_rank];
    for (uint32_t i = 0; i < tensor_rank; ++i) {
        output_dims[i] = get_vararg(i);
    }

    const uint32_t output_bytes_per_row = output_dims[tensor_rank - 1] * element_size;

    const auto s0 = TensorAccessor(ta::dst);
    Noc noc;
    CircularBuffer cb_in(cb_id_in);

    for (uint32_t local_row = 0; local_row < num_rows_for_this_core; ++local_row) {
        cb_in.wait_front(1);
        uint32_t global_output_row = start_row_for_this_core + local_row;
        noc.async_write(
            cb_in, s0, output_bytes_per_row, {.offset_bytes = 0}, {.page_id = global_output_row, .offset_bytes = 0});
        noc.async_writes_flushed();
        cb_in.pop_front(1);
    }
    noc.async_write_barrier();
}
