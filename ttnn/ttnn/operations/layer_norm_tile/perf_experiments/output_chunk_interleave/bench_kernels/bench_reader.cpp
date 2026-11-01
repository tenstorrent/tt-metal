// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Isolated normalize/affine bench reader: loads c, inv_std, gamma and beta tiles once.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_c_src = get_compile_time_arg_val(0);
    constexpr uint32_t cb_inv_src = get_compile_time_arg_val(1);
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(2);
    constexpr uint32_t cb_beta = get_compile_time_arg_val(3);
    constexpr uint32_t rows = get_compile_time_arg_val(4);
    constexpr uint32_t tensor_w_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t work_tile_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t affine_tile_bytes = get_compile_time_arg_val(7);
    constexpr auto c_args = TensorAccessorArgs<8>();
    constexpr auto inv_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr auto gamma_args = TensorAccessorArgs<inv_args.next_compile_time_args_offset()>();
    constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

    const auto c_acc = TensorAccessor(c_args, get_arg_val<uint32_t>(0), work_tile_bytes);
    const auto inv_acc = TensorAccessor(inv_args, get_arg_val<uint32_t>(1), work_tile_bytes);
    const auto gamma_acc = TensorAccessor(gamma_args, get_arg_val<uint32_t>(2), affine_tile_bytes);
    const auto beta_acc = TensorAccessor(beta_args, get_arg_val<uint32_t>(3), affine_tile_bytes);

    auto load = [](uint32_t cb, const auto& acc, uint32_t num_tiles, uint32_t tile_bytes) {
        DataflowBuffer buf(cb);
        buf.reserve_back(num_tiles);
        uint32_t l1 = buf.get_write_ptr();
        for (uint32_t tile = 0; tile < num_tiles; ++tile) {
            noc_async_read(acc.get_noc_addr(tile), l1 + tile * tile_bytes, tile_bytes);
        }
        noc_async_read_barrier();
        buf.push_back(num_tiles);
    };
    load(cb_gamma, gamma_acc, tensor_w_tiles, affine_tile_bytes);
    load(cb_beta, beta_acc, tensor_w_tiles, affine_tile_bytes);
    load(cb_inv_src, inv_acc, rows, work_tile_bytes);
    load(cb_c_src, c_acc, rows * tensor_w_tiles, work_tile_bytes);
}
