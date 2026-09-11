// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "bluestein_streaming_common.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t base_chunk = get_arg(args::base_chunk);
    const uint32_t num_chunks = get_arg(args::num_chunks);
    constexpr uint32_t OUTPUT_SIZE = get_arg(args::output_size);
    const auto out_real = TensorAccessor(tensor::out_r);
    const auto out_imag = TensorAccessor(tensor::out_i);
    DataflowBuffer b_r(dfb::b_r);
    DataflowBuffer b_i(dfb::b_i);
    Noc noc;

    for (uint32_t i = 0; i < num_chunks; ++i) {
        const uint32_t chunk = base_chunk + i;
        const uint32_t valid = bluestein_streaming::valid_elements(OUTPUT_SIZE, chunk);
        const uint32_t bytes = valid * bluestein_streaming::element_bytes;
        const uint32_t offset = chunk * bluestein_streaming::chunk_elements * bluestein_streaming::element_bytes;
        b_r.wait_front(1);
        b_i.wait_front(1);
        // The work partition contains only nonempty output chunks. In POST,
        // write only the logical tail rather than the remainder of the tile.
        noc.async_write(b_r, out_real, bytes, {}, {.page_id = 0, .offset_bytes = offset});
        noc.async_write(b_i, out_imag, bytes, {}, {.page_id = 0, .offset_bytes = offset});
        noc.async_write_barrier();
        b_r.pop_front(1);
        b_i.pop_front(1);
    }
}
