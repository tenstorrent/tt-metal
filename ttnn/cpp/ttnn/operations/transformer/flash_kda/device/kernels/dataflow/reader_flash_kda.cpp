// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Reader: loads the six per-core Flash KDA inputs into CBs (single-shot, no chunk loop).
//
// Runtime args:
//   0  item_idx
//   1  S_prev_addr
//   2  g_addr
//   3  k_addr
//   4  v_addr
//   5  beta_addr
//   6  q_addr
//
// Compile-time args: Kt, Vt

#include <stdint.h>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr std::uint32_t Kt = get_compile_time_arg_val(0);
    constexpr std::uint32_t Vt = get_compile_time_arg_val(1);

    const std::uint32_t item_idx = get_arg_val<std::uint32_t>(0);
    const std::uint32_t s_prev_addr = get_arg_val<std::uint32_t>(1);
    const std::uint32_t g_addr = get_arg_val<std::uint32_t>(2);
    const std::uint32_t k_addr = get_arg_val<std::uint32_t>(3);
    const std::uint32_t v_addr = get_arg_val<std::uint32_t>(4);
    const std::uint32_t beta_addr = get_arg_val<std::uint32_t>(5);
    const std::uint32_t q_addr = get_arg_val<std::uint32_t>(6);

    constexpr std::uint32_t state_tiles = Kt * Vt;

    constexpr std::uint32_t cb_S_prev = tt::CBIndex::c_0;
    constexpr std::uint32_t cb_g = tt::CBIndex::c_1;
    constexpr std::uint32_t cb_k = tt::CBIndex::c_2;
    constexpr std::uint32_t cb_v = tt::CBIndex::c_3;
    constexpr std::uint32_t cb_beta = tt::CBIndex::c_4;
    constexpr std::uint32_t cb_q = tt::CBIndex::c_5;

    constexpr std::uint32_t f32_tile = get_tile_size(cb_S_prev);

    // TensorAccessors for the interleaved fp32 DRAM inputs. The per-tensor
    // TensorAccessorArgs compile-time blocks are appended (in this order) by the
    // program factory right after the {Kt, Vt} compile-time args, so the first block
    // starts at compile-time-arg offset 2 and each chains off the previous.
    constexpr auto sp_args = TensorAccessorArgs<2>();
    const auto sp_gen = TensorAccessor(sp_args, s_prev_addr, f32_tile);
    constexpr auto g_args = TensorAccessorArgs<sp_args.next_compile_time_args_offset()>();
    const auto g_gen = TensorAccessor(g_args, g_addr, f32_tile);
    constexpr auto k_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    const auto k_gen = TensorAccessor(k_args, k_addr, f32_tile);
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    const auto v_gen = TensorAccessor(v_args, v_addr, f32_tile);
    constexpr auto beta_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    const auto beta_gen = TensorAccessor(beta_args, beta_addr, f32_tile);
    constexpr auto q_args = TensorAccessorArgs<beta_args.next_compile_time_args_offset()>();
    const auto q_gen = TensorAccessor(q_args, q_addr, f32_tile);

    Noc noc;
    CircularBuffer cb_S_prev_o(cb_S_prev);
    CircularBuffer cb_g_o(cb_g);
    CircularBuffer cb_k_o(cb_k);
    CircularBuffer cb_v_o(cb_v);
    CircularBuffer cb_beta_o(cb_beta);
    CircularBuffer cb_q_o(cb_q);

    // Per-item tile offsets in the flat [N, ...] tensors.
    const std::uint32_t s_prev_off = item_idx * state_tiles;
    const std::uint32_t g_off = item_idx * Kt;
    const std::uint32_t k_off = item_idx * Kt;
    const std::uint32_t v_off = item_idx * Vt;
    const std::uint32_t beta_off = item_idx;
    const std::uint32_t q_off = item_idx * Kt;

    // S_prev [Dk,Dv]
    cb_S_prev_o.reserve_back(state_tiles);
    for (std::uint32_t t = 0; t < state_tiles; t++) {
        noc.async_read(sp_gen, cb_S_prev_o, f32_tile, {.page_id = s_prev_off + t}, {.offset_bytes = t * f32_tile});
    }
    noc.async_read_barrier();
    cb_S_prev_o.push_back(state_tiles);

    // g [Dk,1] (COL layout: 1 tile per Kt row-block, only column 0 meaningful)
    cb_g_o.reserve_back(Kt);
    for (std::uint32_t t = 0; t < Kt; t++) {
        noc.async_read(g_gen, cb_g_o, f32_tile, {.page_id = g_off + t}, {.offset_bytes = t * f32_tile});
    }
    noc.async_read_barrier();
    cb_g_o.push_back(Kt);

    // k [1,Dk] (ROW layout: 1 tile per Kt col-block, only row 0 meaningful)
    cb_k_o.reserve_back(Kt);
    for (std::uint32_t t = 0; t < Kt; t++) {
        noc.async_read(k_gen, cb_k_o, f32_tile, {.page_id = k_off + t}, {.offset_bytes = t * f32_tile});
    }
    noc.async_read_barrier();
    cb_k_o.push_back(Kt);

    // v [1,Dv]
    cb_v_o.reserve_back(Vt);
    for (std::uint32_t t = 0; t < Vt; t++) {
        noc.async_read(v_gen, cb_v_o, f32_tile, {.page_id = v_off + t}, {.offset_bytes = t * f32_tile});
    }
    noc.async_read_barrier();
    cb_v_o.push_back(Vt);

    // beta [1,1] (single scalar tile)
    cb_beta_o.reserve_back(1);
    noc.async_read(beta_gen, cb_beta_o, f32_tile, {.page_id = beta_off}, {.offset_bytes = 0});
    noc.async_read_barrier();
    cb_beta_o.push_back(1);

    // q [1,Dk] (ROW layout, same convention as k)
    cb_q_o.reserve_back(Kt);
    for (std::uint32_t t = 0; t < Kt; t++) {
        noc.async_read(q_gen, cb_q_o, f32_tile, {.page_id = q_off + t}, {.offset_bytes = t * f32_tile});
    }
    noc.async_read_barrier();
    cb_q_o.push_back(Kt);
}
