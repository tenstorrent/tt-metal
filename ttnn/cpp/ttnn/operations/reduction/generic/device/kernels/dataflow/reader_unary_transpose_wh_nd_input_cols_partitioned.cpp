// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "accessor/sharded_accessor.h"
#include "dataflow_api.h"
#include "cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t Wt = get_arg_val<uint32_t>(1);
    uint32_t Ht = get_arg_val<uint32_t>(2);
    uint32_t batch = get_arg_val<uint32_t>(3);
    uint32_t tile_row_elements = get_arg_val<uint32_t>(4);
    uint32_t batch_elements = get_arg_val<uint32_t>(5);
    const uint32_t bank_base_address = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(1);

    // Always have reduce scalar for nd sharded
    constexpr uint32_t cb_id_in2 = get_compile_time_arg_val(2);
    uint32_t scalar = get_arg_val<uint32_t>(7);
    generate_reduce_scaler(cb_id_in2, scalar);

    constexpr uint32_t tile_elements = get_compile_time_arg_val(3);
    constexpr DataFormat data_format = static_cast<DataFormat>(get_compile_time_arg_val(4));
    constexpr uint32_t page_size = get_compile_time_arg_val(5);
    constexpr uint32_t rank = get_compile_time_arg_val(6);
    constexpr uint32_t num_banks = get_compile_time_arg_val(7);

    constexpr uint32_t arg_index = 8;
    using input_dspec = distribution_spec_t<arg_index, rank, num_banks>;

    auto sharded_accessor = ShardedAccessor<input_dspec, page_size>{.bank_base_address = bank_base_address};

    constexpr uint32_t onetile = 1;
    uint32_t tile_bytes = get_tile_size(cb_id_in0);

    cb_reserve_back(cb_id_in1, num_tiles);
    uint64_t base_index = 0;

    for (uint32_t b = 0; b < batch; ++b) {
        uint64_t col_index = base_index;
        for (uint32_t i = 0; i < Wt; ++i) {  // process each column of tiles across the entire width of the batch
            uint64_t curr_index = col_index;
            for (uint32_t j = 0; j < Ht; ++j) {  // read each tile in the column
                cb_reserve_back(cb_id_in0, onetile);
                uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
                uint64_t curr_noc_addr = sharded_accessor.get_noc_addr(curr_index);
                noc_async_read(curr_noc_addr, l1_write_addr, tile_bytes);
                curr_index += tile_row_elements;  // go to the next tile lower down
                noc_async_read_barrier();
                cb_push_back(cb_id_in0, onetile);
            }
            col_index += tile_elements;  // go to the next tile to the right, for the next column to be processed
        }
        base_index += batch_elements;  // go past the processed Ht*Wt tiles
    }
}
