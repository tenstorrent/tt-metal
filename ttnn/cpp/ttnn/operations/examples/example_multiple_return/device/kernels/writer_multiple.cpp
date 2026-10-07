// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/debug/dprint.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t start_id = get_arg(args::start_id);

    // single-tile ublocks
    constexpr uint32_t onetile = 1;

    Noc noc;
    DataflowBuffer dfb_out(dfb::out);

    // Each output is optional: the host binds its tensor, and defines the matching RETURN_OUTPUT*
    // flag, only when the operation returns that output.
#ifdef RETURN_OUTPUT1
    const auto s1 = TensorAccessor(tensor::dst1);
#endif
#ifdef RETURN_OUTPUT2
    const auto s2 = TensorAccessor(tensor::dst2);
#endif

    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
        dfb_out.wait_front(onetile);

#ifdef RETURN_OUTPUT1
        noc.async_write(dfb_out, s1, s1.get_aligned_page_size(), {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
#endif

#ifdef RETURN_OUTPUT2
        noc.async_write(dfb_out, s2, s2.get_aligned_page_size(), {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
#endif

        dfb_out.pop_front(onetile);
    }
}
