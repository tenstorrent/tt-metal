// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t pre_w_addr = get_arg_val<uint32_t>(0);
    const uint32_t post_w_addr = get_arg_val<uint32_t>(1);
    const uint32_t pre_bias_addr = get_arg_val<uint32_t>(2);
    const uint32_t post_bias_addr = get_arg_val<uint32_t>(3);
    const uint32_t hidden_addr = get_arg_val<uint32_t>(4);
    const uint32_t d_tiles = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_pre_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_post_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_pre_bias = get_compile_time_arg_val(2);
    constexpr uint32_t cb_post_bias = get_compile_time_arg_val(3);
    constexpr uint32_t cb_hidden = get_compile_time_arg_val(4);

    constexpr auto pre_w_args = TensorAccessorArgs<5>();
    constexpr auto post_w_args = TensorAccessorArgs<pre_w_args.next_compile_time_args_offset()>();
    constexpr auto pre_bias_args = TensorAccessorArgs<post_w_args.next_compile_time_args_offset()>();
    constexpr auto post_bias_args = TensorAccessorArgs<pre_bias_args.next_compile_time_args_offset()>();
    constexpr auto hidden_args = TensorAccessorArgs<post_bias_args.next_compile_time_args_offset()>();

    const auto pre_w = TensorAccessor(pre_w_args, pre_w_addr);
    const auto post_w = TensorAccessor(post_w_args, post_w_addr);
    const auto pre_bias = TensorAccessor(pre_bias_args, pre_bias_addr);
    const auto post_bias = TensorAccessor(post_bias_args, post_bias_addr);
    const auto hidden = TensorAccessor(hidden_args, hidden_addr);

    Noc noc;
    CircularBuffer cb_pw(cb_pre_w);
    CircularBuffer cb_ppw(cb_post_w);
    CircularBuffer cb_pb(cb_pre_bias);
    CircularBuffer cb_ppb(cb_post_bias);
    CircularBuffer cb_h(cb_hidden);

    constexpr uint32_t one_tile = 1;

    // Decode (T == 1): a single projection/bias tile each.
    cb_pw.reserve_back(one_tile);
    cb_ppw.reserve_back(one_tile);
    cb_pb.reserve_back(one_tile);
    cb_ppb.reserve_back(one_tile);

    noc.async_read(pre_w, cb_pw, cb_pw.get_tile_size(), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read(post_w, cb_ppw, cb_ppw.get_tile_size(), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read(pre_bias, cb_pb, cb_pb.get_tile_size(), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read(post_bias, cb_ppb, cb_ppb.get_tile_size(), {.page_id = 0}, {.offset_bytes = 0});

    // hidden_streams [1,1,H,D] -> d_tiles tiles laid out along the width.
    const uint32_t hidden_tile_size = cb_h.get_tile_size();
    cb_h.reserve_back(d_tiles);
    for (uint32_t n = 0; n < d_tiles; ++n) {
        noc.async_read(hidden, cb_h, hidden_tile_size, {.page_id = n}, {.offset_bytes = n * hidden_tile_size});
    }

    noc.async_read_barrier();

    cb_pw.push_back(one_tile);
    cb_ppw.push_back(one_tile);
    cb_pb.push_back(one_tile);
    cb_ppb.push_back(one_tile);
    cb_h.push_back(d_tiles);
}
