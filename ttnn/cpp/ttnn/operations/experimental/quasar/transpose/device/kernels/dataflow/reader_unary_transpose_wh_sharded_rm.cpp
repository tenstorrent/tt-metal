// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t num_hw_blocks_per_core = get_compile_time_arg_val(0);
    constexpr uint32_t Ht = get_compile_time_arg_val(1);
    constexpr uint32_t H_per_tile = get_compile_time_arg_val(2);
    constexpr uint32_t H_per_tile_last = get_compile_time_arg_val(3);
    constexpr uint32_t Wt = get_compile_time_arg_val(4);
    constexpr uint32_t W_size_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t l1_write_offset_bytes = get_compile_time_arg_val(6);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in = tt::CBIndex::c_24;

    const uint32_t stick_size_bytes = W_size_bytes;

    Noc noc;
    CircularBuffer cb_src(cb_in0);
    CircularBuffer cb_dst(cb_in);

    uint32_t src_addr = cb_src.get_read_ptr();

    noc.set_async_read_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
        UnicastEndpoint{},
        stick_size_bytes,
        {.noc_x = (uint32_t)my_x[noc.get_noc_id()], .noc_y = (uint32_t)my_y[noc.get_noc_id()], .addr = src_addr});

    for (uint32_t n = 0; n < num_hw_blocks_per_core; n++) {
        for (uint32_t h = 0; h < Ht; ++h) {
            cb_dst.reserve_back(Wt);
            uint32_t l1_write_addr = cb_dst.get_write_ptr();
            uint32_t H_curr = h == Ht - 1 ? H_per_tile_last : H_per_tile;
            for (uint32_t h_datum = 0; h_datum < H_curr; ++h_datum) {
                CoreLocalMem<uint32_t> dst(l1_write_addr);
                noc.async_read_with_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                    UnicastEndpoint{},
                    dst,
                    stick_size_bytes,
                    {.noc_x = (uint32_t)my_x[noc.get_noc_id()],
                     .noc_y = (uint32_t)my_y[noc.get_noc_id()],
                     .addr = src_addr},
                    {.offset_bytes = 0});
                l1_write_addr += l1_write_offset_bytes;
                src_addr += stick_size_bytes;
            }
            noc.async_read_barrier();
            cb_dst.push_back(Wt);
        }
    }
}
