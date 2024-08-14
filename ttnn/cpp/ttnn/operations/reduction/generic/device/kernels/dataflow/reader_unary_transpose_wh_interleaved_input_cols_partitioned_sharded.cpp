// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"

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
    uint32_t scalar = get_arg_val<uint32_t>(6);
    generate_reduce_scaler(cb_id_in2, scalar);
    #endif

    constexpr uint32_t onetile = 1;
    uint32_t tile_bytes = get_tile_size(cb_id_in0);

    cb_reserve_back(cb_id_in1, num_tiles);
    uint64_t base_noc_addr = get_noc_addr(get_write_ptr(cb_id_in1));

    for (uint32_t b = 0; b < batch; ++b)  {
        uint64_t col_noc_addr = base_noc_addr;
        for (uint32_t i = 0; i < Wt; ++i) {
            uint64_t curr_noc_addr = col_noc_addr;
            for (uint32_t j = 0; j < Ht; ++j) {
                cb_reserve_back(cb_id_in0, onetile);
                uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
                noc_async_read(curr_noc_addr, l1_write_addr, tile_bytes);
                curr_noc_addr += row_size_bytes;
                noc_async_read_barrier();
                cb_push_back(cb_id_in0, onetile);
            }
            col_noc_addr += tile_bytes;
        }
        base_noc_addr += batch_size_bytes;
    }

}
