// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime args
    const uint32_t q_tensor_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_tensor_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_tensor_addr = get_arg_val<uint32_t>(2);
    const uint32_t num_work_units = get_arg_val<uint32_t>(3);
    const uint32_t work_unit_start = get_arg_val<uint32_t>(4);

    // Compile-time args
    constexpr uint32_t q_out_h_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t q_out_w_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t q_out_HtWt = get_compile_time_arg_val(2);
    constexpr uint32_t num_q_heads = get_compile_time_arg_val(3);
    constexpr uint32_t num_kv_heads = get_compile_time_arg_val(4);
    constexpr uint32_t q_heads_per_kv = get_compile_time_arg_val(5);
    constexpr auto q_args = TensorAccessorArgs<6>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_id_qkv = 1;
    constexpr uint32_t onetile = 1;
    const uint32_t single_tile_size_bytes = get_tile_size(cb_id_qkv);
    const auto sq = TensorAccessor(q_args, q_tensor_addr, single_tile_size_bytes);
    const auto sk = TensorAccessor(k_args, k_tensor_addr, single_tile_size_bytes);
    const auto sv = TensorAccessor(v_args, v_tensor_addr, single_tile_size_bytes);

    constexpr uint32_t q_tiles_per_group = q_heads_per_kv * q_out_w_tiles;
    constexpr uint32_t q_out_CHtWt = num_q_heads * q_out_HtWt;
    constexpr uint32_t kv_out_CHtWt = num_kv_heads * q_out_HtWt;

    for (uint32_t work = 0; work < num_work_units; ++work) {
        const uint32_t work_unit = work_unit_start + work;
        const uint32_t block = work_unit / num_kv_heads;
        const uint32_t kv_group = work_unit - block * num_kv_heads;
        const uint32_t batch = block / q_out_h_tiles;
        const uint32_t h_tile = block - batch * q_out_h_tiles;
        const uint32_t q_head_start = kv_group * q_heads_per_kv;

        uint32_t q_tile_base = batch * q_out_CHtWt + q_head_start * q_out_HtWt + h_tile * q_out_w_tiles;
        for (uint32_t i = 0; i < q_tiles_per_group; ++i) {
            cb_wait_front(cb_id_qkv, onetile);
            const uint32_t l1_read_addr = get_read_ptr(cb_id_qkv);
            const uint32_t q_head_offset = i / q_out_w_tiles;
            const uint32_t w = i - q_head_offset * q_out_w_tiles;
            noc_async_write_tile(q_tile_base + q_head_offset * q_out_HtWt + w, sq, l1_read_addr);
            noc_async_write_barrier();
            cb_pop_front(cb_id_qkv, onetile);
        }

        uint32_t k_tile_base = batch * kv_out_CHtWt + kv_group * q_out_HtWt + h_tile * q_out_w_tiles;
        for (uint32_t i = 0; i < q_out_w_tiles; ++i) {
            cb_wait_front(cb_id_qkv, onetile);
            const uint32_t l1_read_addr = get_read_ptr(cb_id_qkv);
            noc_async_write_tile(k_tile_base + i, sk, l1_read_addr);
            noc_async_write_barrier();
            cb_pop_front(cb_id_qkv, onetile);
        }

        uint32_t v_tile_base = batch * kv_out_CHtWt + kv_group * q_out_HtWt + h_tile * q_out_w_tiles;
        for (uint32_t i = 0; i < q_out_w_tiles; ++i) {
            cb_wait_front(cb_id_qkv, onetile);
            const uint32_t l1_read_addr = get_read_ptr(cb_id_qkv);
            noc_async_write_tile(v_tile_base + i, sv, l1_read_addr);
            noc_async_write_barrier();
            cb_pop_front(cb_id_qkv, onetile);
        }
    }
}
