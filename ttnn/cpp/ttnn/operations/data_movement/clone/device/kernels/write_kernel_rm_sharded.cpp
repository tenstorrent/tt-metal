// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    auto stick_size = get_arg(args::stick_size);
    auto num_sticks = get_arg(args::num_sticks);

    Noc noc;
    DataflowBuffer dst_dfb(dfb::dst);
    // Case 2 (raw pointer): the TensorBinding supplies the per-enqueue base address;
    // the raw local-L1 walk over the resident shard is unchanged from the legacy kernel.
    const auto s = TensorAccessor(tensor::output);
    uint32_t local_l1_write_addr = s.get_bank_base_address();

    for (uint32_t i = 0; i < num_sticks; ++i) {
        dst_dfb.wait_front(1);
        noc.async_write(
            dst_dfb,
            UnicastEndpoint{},
            stick_size,
            {.offset_bytes = 0},
            {.noc_x = (uint32_t)my_x[noc.get_noc_id()],
             .noc_y = (uint32_t)my_y[noc.get_noc_id()],
             .addr = local_l1_write_addr});
        noc.async_write_barrier();

        dst_dfb.pop_front(1);
        local_l1_write_addr += stick_size;
    }
}
