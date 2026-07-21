// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr auto src_args = TensorAccessorArgs<2>();

    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t col_start_tile_id =
        get_arg_val<uint32_t>(1);  // Start id in column major order. This should be the start of a column
    uint32_t curr_col_in_batch = get_arg_val<uint32_t>(2);
    uint32_t num_cols = get_arg_val<uint32_t>(3);  // number of cols to read
    uint32_t mask_h = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_in0 = 0;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;

#ifdef DO_MASK_H
    constexpr uint32_t cb_id_mask_h = 1;
    DataflowBuffer dfb_mask_h_obj(cb_id_mask_h);
    generate_mask_h<int32_t>(dfb_mask_h_obj, mask_h);
#endif

    const auto s = TensorAccessor(src_args, src_addr);

    Noc noc;
    DataflowBuffer dfb_in0_obj(cb_id_in0);
    const auto in0_tile_bytes = get_tile_size(cb_id_in0);

    uint32_t w = curr_col_in_batch;

    for (uint32_t i = 0; i < num_cols; i++) {
        uint32_t curr_id = col_start_tile_id;
        for (uint32_t j = 0; j < Ht; j++) {
            dfb_in0_obj.reserve_back(onetile);
            noc.async_read(s, dfb_in0_obj, in0_tile_bytes, {.page_id = curr_id}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_in0_obj.push_back(onetile);
            curr_id += Wt;
        }
        w++;
        if (w == Wt) {
            col_start_tile_id = curr_id - Wt + 1;
            w = 0;
        } else {
            col_start_tile_id++;
        }
    }
}
