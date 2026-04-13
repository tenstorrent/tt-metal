// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#ifdef ARCH_QUASAR
#include "experimental/dataflow_buffer.h"
#else
#include "experimental/circular_buffer.h"
#endif
#include "experimental/endpoints.h"
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t Wt = get_arg_val<uint32_t>(1);
    uint32_t Ht = get_arg_val<uint32_t>(2);
    uint32_t batch = get_arg_val<uint32_t>(3);
    uint32_t row_size_bytes = get_arg_val<uint32_t>(4);
    uint32_t batch_size_bytes = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(1);

#ifdef REDUCE_SCALER
    constexpr uint32_t cb_id_in2 = get_compile_time_arg_val(2);
#endif

    constexpr uint32_t onetile = 1;

#ifdef ARCH_QUASAR
    experimental::DataflowBuffer dfb_in0(cb_id_in0);
    experimental::DataflowBuffer dfb_in1(cb_id_in1);
    const uint32_t tile_bytes = dfb_in0.get_entry_size();
#ifdef REDUCE_SCALER
    experimental::DataflowBuffer dfb_scaler(cb_id_in2);
    uint32_t scalar = get_arg_val<uint32_t>(6);
    generate_reduce_scaler(dfb_scaler, scalar);
#endif
#else
    const uint32_t tile_bytes = get_tile_size(cb_id_in0);
    experimental::CircularBuffer cb_in0(cb_id_in0);
    experimental::CircularBuffer cb_in1(cb_id_in1);
#ifdef REDUCE_SCALER
    uint32_t scalar = get_arg_val<uint32_t>(6);
    generate_reduce_scaler(cb_id_in2, scalar);
#endif
#endif

    experimental::Noc noc;

#ifdef ARCH_QUASAR
    dfb_in1.reserve_back(num_tiles);
    uint32_t base_l1_addr = dfb_in1.get_write_ptr();
#else
    cb_in1.reserve_back(num_tiles);
    uint32_t base_l1_addr = cb_in1.get_write_ptr();
#endif

    experimental::UnicastEndpoint src;
    uint32_t src_noc_x = my_x[noc_index];
    uint32_t src_noc_y = my_y[noc_index];

    for (uint32_t b = 0; b < batch; ++b) {
        uint32_t col_l1_addr = base_l1_addr;
        for (uint32_t i = 0; i < Wt; ++i) {
            uint32_t curr_l1_addr = col_l1_addr;
            for (uint32_t j = 0; j < Ht; ++j) {
#ifdef ARCH_QUASAR
                dfb_in0.reserve_back(onetile);
                noc.async_read(
                    src,
                    dfb_in0,
                    tile_bytes,
                    {.noc_x = src_noc_x, .noc_y = src_noc_y, .addr = curr_l1_addr},
                    {.offset_bytes = 0});
                curr_l1_addr += row_size_bytes;
                noc.async_read_barrier();
                dfb_in0.push_back(onetile);
#else
                cb_in0.reserve_back(onetile);
                noc.async_read(
                    src,
                    cb_in0,
                    tile_bytes,
                    {.noc_x = src_noc_x, .noc_y = src_noc_y, .addr = curr_l1_addr},
                    {.offset_bytes = 0});
                curr_l1_addr += row_size_bytes;
                noc.async_read_barrier();
                cb_in0.push_back(onetile);
#endif
            }
            col_l1_addr += tile_bytes;
        }
        base_l1_addr += batch_size_bytes;
    }
}
