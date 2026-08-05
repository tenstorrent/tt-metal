// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src0_num_tiles = get_arg_val<uint32_t>(3);
    uint32_t src1_addr = get_arg_val<uint32_t>(4);
    uint32_t NCHtWt = get_arg_val<uint32_t>(8);
    uint32_t NC = get_arg_val<uint32_t>(9);
    uint32_t Ht = get_arg_val<uint32_t>(10);
    uint32_t Wt = get_arg_val<uint32_t>(11);
    uint32_t nc1 = get_arg_val<uint32_t>(12);
    uint32_t start_id = get_arg_val<uint32_t>(13);
    uint32_t HtWt = get_arg_val<uint32_t>(14);

    constexpr auto src0_args = TensorAccessorArgs<0>();
    constexpr auto src1_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t onetile = 1;

    const auto s0 = TensorAccessor(src0_args, src0_addr);
    const auto s1 = TensorAccessor(src1_args, src1_addr);

    Noc noc;
    DataflowBuffer dfb_in0(cb_id_in0);
    DataflowBuffer dfb_in1(cb_id_in1);
    const uint32_t tile_bytes_0 = get_tile_size(cb_id_in0);
    const uint32_t tile_bytes_1 = get_tile_size(cb_id_in1);

    uint32_t num_tiles = src0_num_tiles;
    uint32_t i = 0;
    uint32_t i1 = 0;
    uint32_t i_nc = 0;
    for (uint32_t nc = 0; nc < NC; nc++) {
        i = i_nc + start_id;
        for (uint32_t ht = 0; ht < Ht; ht++) {
            for (uint32_t wt = 0; wt < Wt; wt++) {
                dfb_in0.reserve_back(onetile);
                noc.async_read(s0, dfb_in0, tile_bytes_0, {.page_id = i, .offset_bytes = 0}, {.offset_bytes = 0});
                noc.async_read_barrier();
                dfb_in0.push_back(onetile);

                dfb_in1.reserve_back(onetile);
                noc.async_read(s1, dfb_in1, tile_bytes_1, {.page_id = i1, .offset_bytes = 0}, {.offset_bytes = 0});
                noc.async_read_barrier();
                dfb_in1.push_back(onetile);
                i1++;
                i++;
            }

            i1 -= Wt;
        }
        if (nc1 == 0) {
            i1 += Wt;
        }
        i_nc += HtWt;
    }
}
