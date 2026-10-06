// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

// Streams every tile in exactly the order the compute kernel consumes it, one tile at a time.
// Nothing is kept between uses: x is re-read for every H tile, and the weights for every row.
void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_gate_addr = get_arg_val<uint32_t>(1);
    const uint32_t w_up_addr = get_arg_val<uint32_t>(2);
    const uint32_t w_down_addr = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w_gate = get_compile_time_arg_val(1);
    constexpr uint32_t cb_w_up = get_compile_time_arg_val(2);
    constexpr uint32_t cb_w_down = get_compile_time_arg_val(3);
    constexpr uint32_t m_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(6);

    constexpr auto x_args = TensorAccessorArgs<7>();
    constexpr auto w_gate_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto w_up_args = TensorAccessorArgs<w_gate_args.next_compile_time_args_offset()>();
    constexpr auto w_down_args = TensorAccessorArgs<w_up_args.next_compile_time_args_offset()>();

    Noc noc;
    CircularBuffer x_cb(cb_x);
    CircularBuffer w_gate_cb(cb_w_gate);
    CircularBuffer w_up_cb(cb_w_up);
    CircularBuffer w_down_cb(cb_w_down);

    const auto x_accessor = TensorAccessor(x_args, x_addr, x_cb.get_tile_size());
    const auto w_gate_accessor = TensorAccessor(w_gate_args, w_gate_addr, w_gate_cb.get_tile_size());
    const auto w_up_accessor = TensorAccessor(w_up_args, w_up_addr, w_up_cb.get_tile_size());
    const auto w_down_accessor = TensorAccessor(w_down_args, w_down_addr, w_down_cb.get_tile_size());

    auto read_tile = [&](const auto& accessor, CircularBuffer& cb, uint32_t tile_id) {
        cb.reserve_back(1);
        noc.async_read(accessor, cb, cb.get_tile_size(), {.page_id = tile_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb.push_back(1);
    };

    // Tiles are numbered row-major: x (T, K) tile (m, k) is m * k_tiles + k, a (K, N) weight tile
    // (k, n) is k * n_tiles + n, and w_down (N, K) tile (n, j) is n * k_tiles + j.
    for (uint32_t m = 0; m < m_tiles; ++m) {
        for (uint32_t n = 0; n < n_tiles; ++n) {
            for (uint32_t k = 0; k < k_tiles; ++k) {
                read_tile(x_accessor, x_cb, m * k_tiles + k);
                read_tile(w_gate_accessor, w_gate_cb, k * n_tiles + n);
                read_tile(w_up_accessor, w_up_cb, k * n_tiles + n);
            }
        }
        for (uint32_t j = 0; j < k_tiles; ++j) {
            for (uint32_t n = 0; n < n_tiles; ++n) {
                read_tile(w_down_accessor, w_down_cb, n * k_tiles + j);
            }
        }
    }
}
