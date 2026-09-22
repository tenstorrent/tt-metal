// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime args
    const uint32_t in0_tensor_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_work_units = get_arg_val<uint32_t>(1);
    const uint32_t work_unit_start = get_arg_val<uint32_t>(2);

    // Compile-time args
    constexpr uint32_t in0_h_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t in0_w_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t in0_c = get_compile_time_arg_val(2);
    constexpr uint32_t in0_HtWt = get_compile_time_arg_val(3);
    constexpr uint32_t head_groups = get_compile_time_arg_val(4);
    constexpr uint32_t heads_per_group = get_compile_time_arg_val(5);
    constexpr auto in0_args = TensorAccessorArgs<6>();

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t onetile = 1;
    const uint32_t single_tile_size_bytes = get_tile_size(cb_id_in0);
    const auto s0 = TensorAccessor(in0_args, in0_tensor_addr, single_tile_size_bytes);

    constexpr uint32_t group_tiles = heads_per_group * in0_w_tiles;
    constexpr uint32_t in0_CHtWt = in0_c * in0_HtWt;

    for (uint32_t work = 0; work < num_work_units; ++work) {
        const uint32_t work_unit = work_unit_start + work;
        const uint32_t block = work_unit / head_groups;
        const uint32_t group = work_unit - block * head_groups;
        const uint32_t batch = block / in0_h_tiles;
        const uint32_t h_tile = block - batch * in0_h_tiles;
        const uint32_t head_start = group * heads_per_group;

        const uint32_t base_tile = batch * in0_CHtWt + head_start * in0_HtWt + h_tile * in0_w_tiles;
        for (uint32_t i = 0; i < group_tiles; ++i) {
            const uint32_t head_offset = i / in0_w_tiles;
            const uint32_t w = i - head_offset * in0_w_tiles;
            cb_reserve_back(cb_id_in0, onetile);
            const uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
            noc_async_read_tile(base_tile + head_offset * in0_HtWt + w, s0, l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(cb_id_in0, onetile);
        }
    }
}
