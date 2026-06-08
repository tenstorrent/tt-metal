// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// MQA/GQA-aware Q-head-parallel reader. Work-unit decomposition:
//   work_unit = (block * num_q_heads) + q_head_idx
//   kv_group  = q_head_idx / q_heads_per_kv
//   is_first_in_kv = (q_head_idx % q_heads_per_kv == 0)
//
// Each work unit reads `head_tiles` Q tiles for ONE Q-head. K and V are
// shared across q_heads_per_kv Q-heads in a KV-group, so they are read
// (and later written) by ONLY the "first" Q-head in each group. This
// keeps total reads/writes minimal while making num_blocks scale with
// num_q_heads (instead of num_kv_heads).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime args
    const uint32_t in0_tensor_addr = get_arg_val<uint32_t>(0);
    [[maybe_unused]] const uint32_t unused_in1_tensor_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_work_units = get_arg_val<uint32_t>(2);
    [[maybe_unused]] const uint32_t unused_in0_tile_id = get_arg_val<uint32_t>(3);
    const uint32_t work_unit_start = get_arg_val<uint32_t>(4);

    // Compile-time args — matches the head_split reader's layout so the
    // program_factory can pass the same reader_compile_time_args vector to
    // either kernel without per-kernel branching.
    constexpr uint32_t q_heads_per_kv = get_compile_time_arg_val(0);
    constexpr uint32_t num_kv_heads = get_compile_time_arg_val(1);
    constexpr uint32_t head_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t in0_w_tiles = get_compile_time_arg_val(3);
    constexpr auto in0_args = TensorAccessorArgs<4>();

    constexpr uint32_t num_q_heads = q_heads_per_kv * num_kv_heads;

    constexpr uint32_t cb_id_qkv = 1;
    constexpr uint32_t onetile = 1;
    const uint32_t single_tile_size_bytes = get_tile_size(cb_id_qkv);
    const auto s0 = TensorAccessor(in0_args, in0_tensor_addr, single_tile_size_bytes);

    // Input tile layout per block: [Q0..Q_{num_q_heads-1} K_0..K_{num_kv_heads-1} V_0..V_{num_kv_heads-1}],
    // each "head" occupying head_tiles tiles.
    constexpr uint32_t q_tiles_total = num_q_heads * head_tiles;
    constexpr uint32_t kv_tiles_total = num_kv_heads * head_tiles;

    for (uint32_t work = 0; work < num_work_units; ++work) {
        const uint32_t work_unit = work_unit_start + work;
        const uint32_t block = work_unit / num_q_heads;
        const uint32_t q_head_idx = work_unit - block * num_q_heads;
        const uint32_t kv_group = q_head_idx / q_heads_per_kv;
        const bool is_first_in_kv = (q_head_idx - kv_group * q_heads_per_kv) == 0;
        const uint32_t block_base = block * in0_w_tiles;

        // Read this Q-head's tiles.
        const uint32_t q_tile = block_base + q_head_idx * head_tiles;
        for (uint32_t i = 0; i < head_tiles; ++i) {
            cb_reserve_back(cb_id_qkv, onetile);
            const uint32_t l1_write_addr = get_write_ptr(cb_id_qkv);
            noc_async_read_tile(q_tile + i, s0, l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(cb_id_qkv, onetile);
        }

        if (is_first_in_kv) {
            // Read K (shared across q_heads_per_kv Q-heads in this kv_group).
            const uint32_t k_tile = block_base + q_tiles_total + kv_group * head_tiles;
            for (uint32_t i = 0; i < head_tiles; ++i) {
                cb_reserve_back(cb_id_qkv, onetile);
                const uint32_t l1_write_addr = get_write_ptr(cb_id_qkv);
                noc_async_read_tile(k_tile + i, s0, l1_write_addr);
                noc_async_read_barrier();
                cb_push_back(cb_id_qkv, onetile);
            }

            // Read V.
            const uint32_t v_tile = block_base + q_tiles_total + kv_tiles_total + kv_group * head_tiles;
            for (uint32_t i = 0; i < head_tiles; ++i) {
                cb_reserve_back(cb_id_qkv, onetile);
                const uint32_t l1_write_addr = get_write_ptr(cb_id_qkv);
                noc_async_read_tile(v_tile + i, s0, l1_write_addr);
                noc_async_read_barrier();
                cb_push_back(cb_id_qkv, onetile);
            }
        }
    }
}
