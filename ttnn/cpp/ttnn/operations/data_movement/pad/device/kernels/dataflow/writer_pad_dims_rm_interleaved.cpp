// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const auto num_total_Z = get_arg(args::num_total_Z);
    const auto padded_X_nbytes = get_arg(args::padded_X_nbytes);
    const auto start_dst_stick_id = get_arg(args::start_dst_stick_id);
    const auto num_local_Y = get_arg(args::num_local_Y);
    const auto dst_stick_offset = get_arg(args::dst_stick_offset);  // == start_dst_stick_wi * elem_size
    const auto num_local_W = get_arg(args::num_local_W);

    DataflowBuffer dfb(dfb::in0);

    const auto s1 = TensorAccessor(tensor::dst);
    Noc noc;

    uint32_t dst_stick_id = start_dst_stick_id;
    for (uint32_t w = 0; w < num_local_W; ++w) {
        for (uint32_t z = 0; z < num_total_Z; ++z) {
            for (uint32_t y = 0; y < num_local_Y; ++y) {
                // DPRINT("WR: w={} z={} y={}\n", w, z, y);
                dfb.wait_front(1);
                noc.async_write(
                    dfb,
                    s1,
                    padded_X_nbytes,
                    {.offset_bytes = 0},
                    {.page_id = dst_stick_id, .offset_bytes = dst_stick_offset});
                noc.async_write_barrier();
                ++dst_stick_id;
                dfb.pop_front(1);
            }
        }
    }
}
