// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime args
    const uint32_t in0_tensor_addr = get_arg_val<uint32_t>(0);
    [[maybe_unused]] const uint32_t unused_in1_tensor_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_work_units = get_arg_val<uint32_t>(2);
    [[maybe_unused]] const uint32_t unused_in0_tile_id = get_arg_val<uint32_t>(3);
    const uint32_t work_unit_start = get_arg_val<uint32_t>(4);

    // Compile-time args
    constexpr uint32_t q_heads_per_kv = get_compile_time_arg_val(0);
    constexpr uint32_t num_kv_heads = get_compile_time_arg_val(1);
    constexpr uint32_t head_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t in0_w_tiles = get_compile_time_arg_val(3);
    constexpr auto in0_args = TensorAccessorArgs<4>();

    constexpr uint32_t cb_id_qkv = 1;
    constexpr uint32_t onetile = 1;
    const uint32_t single_tile_size_bytes = get_tile_size(cb_id_qkv);
    const auto s0 = TensorAccessor(in0_args, in0_tensor_addr, single_tile_size_bytes);

    constexpr uint32_t q_tiles_per_group = q_heads_per_kv * head_tiles;
    constexpr uint32_t q_tiles_total = q_heads_per_kv * num_kv_heads * head_tiles;
    constexpr uint32_t kv_tiles_total = num_kv_heads * head_tiles;

    for (uint32_t work = 0; work < num_work_units; ++work) {
        const uint32_t work_unit = work_unit_start + work;
        const uint32_t block = work_unit / num_kv_heads;
        const uint32_t kv_group = work_unit - block * num_kv_heads;
        const uint32_t block_base = block * in0_w_tiles;

        uint32_t q_tile = block_base + kv_group * q_tiles_per_group;
        for (uint32_t i = 0; i < q_tiles_per_group; ++i) {
            cb_reserve_back(cb_id_qkv, onetile);
            const uint32_t l1_write_addr = get_write_ptr(cb_id_qkv);
            noc_async_read_tile(q_tile + i, s0, l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(cb_id_qkv, onetile);
        }

        uint32_t k_tile = block_base + q_tiles_total + kv_group * head_tiles;
        for (uint32_t i = 0; i < head_tiles; ++i) {
            cb_reserve_back(cb_id_qkv, onetile);
            const uint32_t l1_write_addr = get_write_ptr(cb_id_qkv);
            noc_async_read_tile(k_tile + i, s0, l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(cb_id_qkv, onetile);
        }

        uint32_t v_tile = block_base + q_tiles_total + kv_tiles_total + kv_group * head_tiles;
        for (uint32_t i = 0; i < head_tiles; ++i) {
            cb_reserve_back(cb_id_qkv, onetile);
            const uint32_t l1_write_addr = get_write_ptr(cb_id_qkv);
            noc_async_read_tile(v_tile + i, s0, l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(cb_id_qkv, onetile);
        }
    }
}
