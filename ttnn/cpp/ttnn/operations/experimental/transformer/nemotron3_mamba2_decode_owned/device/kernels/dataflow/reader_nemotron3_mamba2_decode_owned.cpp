// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0
//
// Mamba2 SSD decode-step reader (NCRISC). Loads 9 input tensors from
// interleaved DRAM (decision D9) into 9 circular buffers consumed by the
// compute kernel.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "experimental/tensor.h"

namespace {

template <typename Accessor>
FORCE_INLINE void read_n_tiles_to_cb(
    const Accessor& accessor, uint32_t base_tile_id, uint32_t n_tiles, uint32_t cb_id) {
    for (uint32_t t = 0; t < n_tiles; ++t) {
        cb_reserve_back(cb_id, 1);
        noc_async_read_tile(base_tile_id + t, accessor, get_write_ptr(cb_id));
        noc_async_read_barrier();
        cb_push_back(cb_id, 1);
    }
}

}  // namespace

void kernel_main() {
    // Compile-time args: 9 CB indices, then 9 TensorAccessorArgs blocks.
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_z = get_compile_time_arg_val(1);
    constexpr uint32_t cb_dt = get_compile_time_arg_val(2);
    constexpr uint32_t cb_dt_bias = get_compile_time_arg_val(3);
    constexpr uint32_t cb_A_log = get_compile_time_arg_val(4);
    constexpr uint32_t cb_D = get_compile_time_arg_val(5);
    constexpr uint32_t cb_B = get_compile_time_arg_val(6);
    constexpr uint32_t cb_C = get_compile_time_arg_val(7);
    constexpr uint32_t cb_state = get_compile_time_arg_val(8);

    constexpr auto x_args = TensorAccessorArgs<9>();
    constexpr auto z_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto dt_args = TensorAccessorArgs<z_args.next_compile_time_args_offset()>();
    constexpr auto dt_bias_args = TensorAccessorArgs<dt_args.next_compile_time_args_offset()>();
    constexpr auto A_log_args = TensorAccessorArgs<dt_bias_args.next_compile_time_args_offset()>();
    constexpr auto D_args = TensorAccessorArgs<A_log_args.next_compile_time_args_offset()>();
    constexpr auto B_args = TensorAccessorArgs<D_args.next_compile_time_args_offset()>();
    constexpr auto C_args = TensorAccessorArgs<B_args.next_compile_time_args_offset()>();
    constexpr auto state_args = TensorAccessorArgs<C_args.next_compile_time_args_offset()>();

    // Runtime args: 9 DRAM addresses + start_block + blocks_per_core + dims.
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t z_addr = get_arg_val<uint32_t>(1);
    const uint32_t dt_addr = get_arg_val<uint32_t>(2);
    const uint32_t dt_bias_addr = get_arg_val<uint32_t>(3);
    const uint32_t A_log_addr = get_arg_val<uint32_t>(4);
    const uint32_t D_addr = get_arg_val<uint32_t>(5);
    const uint32_t B_addr = get_arg_val<uint32_t>(6);
    const uint32_t C_addr = get_arg_val<uint32_t>(7);
    const uint32_t state_addr = get_arg_val<uint32_t>(8);
    const uint32_t start_block = get_arg_val<uint32_t>(9);
    const uint32_t num_blocks = get_arg_val<uint32_t>(10);
    const uint32_t head_dim_tiles = get_arg_val<uint32_t>(11);
    const uint32_t ssm_state_tiles = get_arg_val<uint32_t>(12);

    const uint32_t bf16_tile_size = get_tile_size(cb_x);
    const uint32_t fp32_tile_size = get_tile_size(cb_state);

    const auto x_tensor = TensorAccessor(x_args, x_addr, bf16_tile_size);
    const auto z_tensor = TensorAccessor(z_args, z_addr, bf16_tile_size);
    const auto dt_tensor = TensorAccessor(dt_args, dt_addr, bf16_tile_size);
    const auto dt_bias_tensor = TensorAccessor(dt_bias_args, dt_bias_addr, bf16_tile_size);
    const auto A_log_tensor = TensorAccessor(A_log_args, A_log_addr, bf16_tile_size);
    const auto D_tensor = TensorAccessor(D_args, D_addr, bf16_tile_size);
    const auto B_tensor = TensorAccessor(B_args, B_addr, bf16_tile_size);
    const auto C_tensor = TensorAccessor(C_args, C_addr, bf16_tile_size);
    const auto state_tensor = TensorAccessor(state_args, state_addr, fp32_tile_size);

    const uint32_t state_tiles_per_block = head_dim_tiles * ssm_state_tiles;

    for (uint32_t block = 0; block < num_blocks; ++block) {
        const uint32_t global_block = start_block + block;

        // ── scalars (1 tile each)
        read_n_tiles_to_cb(dt_tensor, global_block, 1, cb_dt);
        // Weights are per-head; v0 single-head reads tile 0. G2/G3 will map
        // block → head_idx for the per-head slice.
        read_n_tiles_to_cb(dt_bias_tensor, global_block, 1, cb_dt_bias);
        read_n_tiles_to_cb(A_log_tensor, global_block, 1, cb_A_log);
        read_n_tiles_to_cb(D_tensor, global_block, 1, cb_D);

        // ── x, z (head_dim_tiles each)
        read_n_tiles_to_cb(x_tensor, global_block * head_dim_tiles, head_dim_tiles, cb_x);
        read_n_tiles_to_cb(z_tensor, global_block * head_dim_tiles, head_dim_tiles, cb_z);

        // ── B, C (ssm_state_tiles each, per-group broadcast handled by host
        //         pre-replication for v0; G2+ uses group_idx = head // 8)
        read_n_tiles_to_cb(B_tensor, global_block * ssm_state_tiles, ssm_state_tiles, cb_B);
        read_n_tiles_to_cb(C_tensor, global_block * ssm_state_tiles, ssm_state_tiles, cb_C);

        // ── ssm_state (head_dim_tiles * ssm_state_tiles fp32 tiles)
        read_n_tiles_to_cb(state_tensor, global_block * state_tiles_per_block, state_tiles_per_block, cb_state);
    }
}
