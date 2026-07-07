// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t Ct = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Vt = get_compile_time_arg_val(2);

    const uint32_t core_work_start = get_arg_val<uint32_t>(0);
    const uint32_t num_work = get_arg_val<uint32_t>(1);
    const uint32_t NC = get_arg_val<uint32_t>(2);
    const uint32_t work_stride = get_arg_val<uint32_t>(3);
    const uint32_t q_addr = get_arg_val<uint32_t>(4);
    const uint32_t k_addr = get_arg_val<uint32_t>(5);
    const uint32_t v_addr = get_arg_val<uint32_t>(6);
    const uint32_t beta_addr = get_arg_val<uint32_t>(7);
    const uint32_t g_addr = get_arg_val<uint32_t>(8);
    const uint32_t triu_addr = get_arg_val<uint32_t>(9);
    const uint32_t tril_addr = get_arg_val<uint32_t>(10);
    const uint32_t eye_addr = get_arg_val<uint32_t>(11);
    const uint32_t lower_addr = get_arg_val<uint32_t>(12);
    const uint32_t eye32_addr = get_arg_val<uint32_t>(13);

    constexpr uint32_t cb_q = tt::CBIndex::c_0;
    constexpr uint32_t cb_k = tt::CBIndex::c_1;
    constexpr uint32_t cb_v = tt::CBIndex::c_2;
    constexpr uint32_t cb_beta = tt::CBIndex::c_3;
    constexpr uint32_t cb_g = tt::CBIndex::c_4;
    constexpr uint32_t cb_triu = tt::CBIndex::c_5;
    constexpr uint32_t cb_tril = tt::CBIndex::c_6;
    constexpr uint32_t cb_eye = tt::CBIndex::c_7;
    constexpr uint32_t cb_lower = tt::CBIndex::c_8;
    constexpr uint32_t cb_eye32 = tt::CBIndex::c_9;

    constexpr uint32_t in_kv_tiles = Ct * Kt;
    constexpr uint32_t out_tiles = Ct * Vt;
    constexpr uint32_t attn_tiles = Ct * Ct;
    constexpr uint32_t beta_tiles = Ct;
    constexpr uint32_t g_tiles = Ct;
    constexpr uint32_t f32_tile = get_tile_size(cb_q);

    constexpr auto q_args = TensorAccessorArgs<3>();
    const auto q_gen = TensorAccessor(q_args, q_addr, f32_tile);
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    const auto k_gen = TensorAccessor(k_args, k_addr, f32_tile);
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    const auto v_gen = TensorAccessor(v_args, v_addr, f32_tile);
    constexpr auto beta_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    const auto beta_gen = TensorAccessor(beta_args, beta_addr, f32_tile);
    constexpr auto g_args = TensorAccessorArgs<beta_args.next_compile_time_args_offset()>();
    const auto g_gen = TensorAccessor(g_args, g_addr, f32_tile);
    constexpr auto triu_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    const auto triu_gen = TensorAccessor(triu_args, triu_addr, f32_tile);
    constexpr auto tril_args = TensorAccessorArgs<triu_args.next_compile_time_args_offset()>();
    const auto tril_gen = TensorAccessor(tril_args, tril_addr, f32_tile);
    constexpr auto eye_args = TensorAccessorArgs<tril_args.next_compile_time_args_offset()>();
    const auto eye_gen = TensorAccessor(eye_args, eye_addr, f32_tile);
    constexpr auto lower_args = TensorAccessorArgs<eye_args.next_compile_time_args_offset()>();
    const auto lower_gen = TensorAccessor(lower_args, lower_addr, f32_tile);
    constexpr auto eye32_args = TensorAccessorArgs<lower_args.next_compile_time_args_offset()>();
    const auto eye32_gen = TensorAccessor(eye32_args, eye32_addr, f32_tile);

    Noc noc;
    CircularBuffer q(cb_q);
    CircularBuffer k(cb_k);
    CircularBuffer v(cb_v);
    CircularBuffer beta(cb_beta);
    CircularBuffer g(cb_g);
    CircularBuffer triu(cb_triu);
    CircularBuffer tril(cb_tril);
    CircularBuffer eye(cb_eye);
    CircularBuffer lower(cb_lower);
    CircularBuffer eye32(cb_eye32);

    for (uint32_t work = core_work_start; work < num_work; work += work_stride) {
        const uint32_t h = work / NC;
        const uint32_t c = work - h * NC;
        const uint32_t off_kv = h * NC * in_kv_tiles + c * in_kv_tiles;
        const uint32_t off_v = h * NC * out_tiles + c * out_tiles;
        const uint32_t off_beta = h * NC * beta_tiles + c * beta_tiles;
        const uint32_t off_g = h * NC * g_tiles + c * g_tiles;

        q.reserve_back(in_kv_tiles);
        k.reserve_back(in_kv_tiles);
        for (uint32_t t = 0; t < in_kv_tiles; t++) {
            noc.async_read(q_gen, q, f32_tile, {.page_id = off_kv + t}, {.offset_bytes = t * f32_tile});
            noc.async_read(k_gen, k, f32_tile, {.page_id = off_kv + t}, {.offset_bytes = t * f32_tile});
        }
        noc.async_read_barrier();
        q.push_back(in_kv_tiles);
        k.push_back(in_kv_tiles);

        v.reserve_back(out_tiles);
        for (uint32_t t = 0; t < out_tiles; t++) {
            noc.async_read(v_gen, v, f32_tile, {.page_id = off_v + t}, {.offset_bytes = t * f32_tile});
        }
        noc.async_read_barrier();
        v.push_back(out_tiles);

        beta.reserve_back(beta_tiles);
        g.reserve_back(g_tiles);
        for (uint32_t t = 0; t < beta_tiles; t++) {
            noc.async_read(beta_gen, beta, f32_tile, {.page_id = off_beta + t}, {.offset_bytes = t * f32_tile});
            noc.async_read(g_gen, g, f32_tile, {.page_id = off_g + t}, {.offset_bytes = t * f32_tile});
        }
        noc.async_read_barrier();
        beta.push_back(beta_tiles);
        g.push_back(g_tiles);

        triu.reserve_back(attn_tiles);
        tril.reserve_back(attn_tiles);
        eye.reserve_back(attn_tiles);
        lower.reserve_back(attn_tiles);
        for (uint32_t t = 0; t < attn_tiles; t++) {
            noc.async_read(triu_gen, triu, f32_tile, {.page_id = t}, {.offset_bytes = t * f32_tile});
            noc.async_read(tril_gen, tril, f32_tile, {.page_id = t}, {.offset_bytes = t * f32_tile});
            noc.async_read(eye_gen, eye, f32_tile, {.page_id = t}, {.offset_bytes = t * f32_tile});
            noc.async_read(lower_gen, lower, f32_tile, {.page_id = t}, {.offset_bytes = t * f32_tile});
        }
        noc.async_read_barrier();
        triu.push_back(attn_tiles);
        tril.push_back(attn_tiles);
        eye.push_back(attn_tiles);
        lower.push_back(attn_tiles);

        eye32.reserve_back(1);
        noc.async_read(eye32_gen, eye32, f32_tile, {.page_id = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        eye32.push_back(1);
    }
}
