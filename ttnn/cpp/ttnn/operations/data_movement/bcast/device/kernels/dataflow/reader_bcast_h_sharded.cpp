// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t src1_addr = get_arg_val<uint32_t>(0);
    uint32_t Ht = get_arg_val<uint32_t>(1);
    uint32_t Wt = get_arg_val<uint32_t>(2);
    uint32_t offset = get_arg_val<uint32_t>(3);
    uint32_t NC = get_arg_val<uint32_t>(4);
    uint32_t batch_offset = get_arg_val<uint32_t>(5);

    constexpr uint32_t dfb_id_in0 = get_compile_time_arg_val(0);
    constexpr auto src1_args = TensorAccessorArgs<1>();

    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t onetile = 1;

    const auto s1 = TensorAccessor(src1_args, src1_addr);

    Noc noc;
    DataflowBuffer dfb_in0(dfb_id_in0);
    DataflowBuffer dfb_in1(cb_id_in1);
    const uint32_t tile_bytes_1 = get_tile_size(cb_id_in1);

    dfb_in0.push_back(Ht * Wt);
    for (uint32_t ht = 0; ht < Ht; ht++) {
        for (uint32_t wt = 0; wt < Wt; wt++) {
            dfb_in1.reserve_back(onetile);
            noc.async_read(s1, dfb_in1, tile_bytes_1, {.page_id = offset, .offset_bytes = 0}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_in1.push_back(onetile);
            offset++;
        }

        offset -= Wt;
        if (ht % NC == (NC - 1)) {
            offset += batch_offset;
        }
    }
}
