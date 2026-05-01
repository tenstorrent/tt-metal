// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime args
    const uint32_t out_tensor_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_work_units = get_arg_val<uint32_t>(1);
    const uint32_t work_unit_start = get_arg_val<uint32_t>(2);

    // Compile-time args
    constexpr uint32_t head_groups = get_compile_time_arg_val(0);
    constexpr uint32_t heads_per_group = get_compile_time_arg_val(1);
    constexpr uint32_t in0_w_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t per_tensor_tiles = get_compile_time_arg_val(3);
    constexpr auto out_args = TensorAccessorArgs<4>();

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t onetile = 1;
    const uint32_t single_tile_size_bytes = get_tile_size(cb_id_in0);
    const auto s0 = TensorAccessor(out_args, out_tensor_addr, single_tile_size_bytes);

    constexpr uint32_t group_tiles = heads_per_group * in0_w_tiles;

    for (uint32_t work = 0; work < num_work_units; ++work) {
        const uint32_t work_unit = work_unit_start + work;
        const uint32_t block = work_unit / head_groups;
        const uint32_t group = work_unit - block * head_groups;
        const uint32_t out_tile_base = block * per_tensor_tiles + group * group_tiles;

        for (uint32_t i = 0; i < group_tiles; ++i) {
            cb_wait_front(cb_id_in0, onetile);
            const uint32_t l1_read_addr = get_read_ptr(cb_id_in0);
            noc_async_write_tile(out_tile_base + i, s0, l1_read_addr);
            noc_async_write_barrier();
            cb_pop_front(cb_id_in0, onetile);
        }
    }
}
