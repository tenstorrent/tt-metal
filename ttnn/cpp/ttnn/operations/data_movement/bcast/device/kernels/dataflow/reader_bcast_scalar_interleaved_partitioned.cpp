// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"

void kernel_main() {
    auto src0_addr = get_arg_val<uint32_t>(0);
    auto packed_scalar = get_arg_val<uint32_t>(1);
    auto num_tiles = get_arg_val<uint32_t>(2);
    auto HtWt = get_arg_val<uint32_t>(3);
    auto base_start_id_HtWt = get_arg_val<uint32_t>(4);
    auto curr_id_from_base = get_arg_val<uint32_t>(5);
    auto bcast_id = get_arg_val<uint32_t>(6);

#ifndef IN0_SHARDED
    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;
#endif

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t onetile = 1;

    // single-tile ublocks
    const uint32_t in0_tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat in0_data_format = get_dataformat(cb_id_in0);
    const DataFormat in1_data_format = DataFormat::Float16_b;

    uint32_t l1_write_addr_in0;
    uint32_t l1_write_addr_in1;

#ifndef IN0_SHARDED
    const InterleavedAddrGenFast<src0_is_dram> s0 = {
        .bank_base_address = src0_addr, .page_size = in0_tile_bytes, .data_format = in0_data_format};
#else
    cb_reserve_back(cb_id_in0, num_tiles);
    cb_push_back(cb_id_in0, num_tiles);
#endif

    generate_bcast_unary_scalar(cb_id_in1, packed_scalar);

    for (uint32_t i = 0; i < num_tiles; i++) {
        uint32_t curr_id = base_start_id_HtWt + curr_id_from_base;

#ifndef IN0_SHARDED
        cb_reserve_back(cb_id_in0, onetile);
        l1_write_addr_in0 = get_write_ptr(cb_id_in0);
        noc_async_read_tile(curr_id, s0, l1_write_addr_in0);
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, onetile);
#endif

        curr_id_from_base++;

        if (curr_id_from_base == HtWt) {
            base_start_id_HtWt += HtWt;
            curr_id_from_base = 0;
        }
    }
}
