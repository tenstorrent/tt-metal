// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0
//
// Mamba2 SSD decode-step writer (BRISC). Drains cb_state_out and cb_y to
// interleaved DRAM (decision D9).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "experimental/tensor.h"

namespace {

template <typename Accessor>
FORCE_INLINE void write_n_tiles_from_cb(
    const Accessor& accessor, uint32_t base_tile_id, uint32_t n_tiles, uint32_t cb_id) {
    for (uint32_t t = 0; t < n_tiles; ++t) {
        cb_wait_front(cb_id, 1);
        noc_async_write_tile(base_tile_id + t, accessor, get_read_ptr(cb_id));
        noc_async_write_barrier();
        cb_pop_front(cb_id, 1);
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_state_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_y = get_compile_time_arg_val(1);

    constexpr auto state_args = TensorAccessorArgs<2>();
    constexpr auto y_args = TensorAccessorArgs<state_args.next_compile_time_args_offset()>();

    const uint32_t state_addr = get_arg_val<uint32_t>(0);
    const uint32_t y_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_block = get_arg_val<uint32_t>(2);
    const uint32_t num_blocks = get_arg_val<uint32_t>(3);
    const uint32_t head_dim_tiles = get_arg_val<uint32_t>(4);
    const uint32_t ssm_state_tiles = get_arg_val<uint32_t>(5);

    const uint32_t bf16_tile_size = get_tile_size(cb_y);
    const uint32_t fp32_tile_size = get_tile_size(cb_state_out);

    const auto state_tensor = TensorAccessor(state_args, state_addr, fp32_tile_size);
    const auto y_tensor = TensorAccessor(y_args, y_addr, bf16_tile_size);

    const uint32_t state_tiles_per_block = head_dim_tiles * ssm_state_tiles;

    for (uint32_t block = 0; block < num_blocks; ++block) {
        const uint32_t global_block = start_block + block;

        // Drain ssm_state_out (head_dim_tiles * ssm_state_tiles fp32 tiles).
        write_n_tiles_from_cb(state_tensor, global_block * state_tiles_per_block, state_tiles_per_block, cb_state_out);

        // Drain y (head_dim_tiles bf16 tiles).
        write_n_tiles_from_cb(y_tensor, global_block * head_dim_tiles, head_dim_tiles, cb_y);
    }
}
