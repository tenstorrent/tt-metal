// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// rotary_embedding_indexed chain mode reader. A block is (batch b, head h, tile row t) of the [B, H, S, W] heads.
// Per block: the head's W / 32 input tiles (IN_MODE 0: [B, H, S, W] TILE; 1: [B, 1, S, H W] TILE, head h = tile
// columns h WT .. h WT + WT - 1, the nlp_create_q_heads split read in place) or its 32 row pages (IN_MODE 2:
// [B, H, S, W] ROW_MAJOR, tilized by compute), then the RT cos and RT sin tiles of tile row t (per-chip pre-sliced
// [1, 1, >= S, R] tables, row 0 = the chip's first row). The trans-mat once.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "chain_share.hpp"

void kernel_main() {
    const uint32_t in_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t cos_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t sin_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t trans_addr = get_common_arg_val<uint32_t>(3);
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_cos = get_compile_time_arg_val(1);
    constexpr uint32_t cb_sin = get_compile_time_arg_val(2);
    constexpr uint32_t cb_trans = get_compile_time_arg_val(3);
    constexpr uint32_t IN_MODE = get_compile_time_arg_val(4);
    constexpr uint32_t H = get_compile_time_arg_val(5);
    constexpr uint32_t Ht = get_compile_time_arg_val(6);
    constexpr uint32_t WT = get_compile_time_arg_val(7);
    constexpr uint32_t RT = get_compile_time_arg_val(8);
    constexpr uint32_t GX = get_compile_time_arg_val(9);
    constexpr uint32_t in_page = get_compile_time_arg_val(10);
    constexpr auto in_args = TensorAccessorArgs<11>();
    constexpr auto cos_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
    constexpr auto tr_args = TensorAccessorArgs<sin_args.next_compile_time_args_offset()>();

    uint32_t first, count;
    chain_share(get_common_arg_val<uint32_t>(4), get_common_arg_val<uint32_t>(5), GX, first, count);
    if (count == 0) {
        return;
    }
    const uint32_t cos_tile = get_tile_size(cb_cos);
    const uint32_t sin_tile = get_tile_size(cb_sin);
    const uint32_t tr_tile = get_tile_size(cb_trans);
    const auto in_acc = TensorAccessor(in_args, in_addr, in_page);
    const auto cos_acc = TensorAccessor(cos_args, cos_addr, cos_tile);
    const auto sin_acc = TensorAccessor(sin_args, sin_addr, sin_tile);
    const auto tr_acc = TensorAccessor(tr_args, trans_addr, tr_tile);

    cb_reserve_back(cb_trans, 1);
    noc_async_read(tr_acc.get_noc_addr(0), get_write_ptr(cb_trans), tr_tile);
    noc_async_read_barrier();
    cb_push_back(cb_trans, 1);

    for (uint32_t k = first; k < first + count; ++k) {
        const uint32_t t = k % Ht;
        const uint32_t bh = k / Ht;  // b H + h
        if constexpr (IN_MODE == 2) {
            cb_reserve_back(cb_in, 32);
            uint32_t l1 = get_write_ptr(cb_in);
            const uint32_t row0 = bh * Ht * 32 + t * 32;
            for (uint32_t r = 0; r < 32; ++r, l1 += in_page) {
                noc_async_read(in_acc.get_noc_addr(row0 + r), l1, in_page);
            }
        } else {
            cb_reserve_back(cb_in, WT);
            uint32_t l1 = get_write_ptr(cb_in);
            uint32_t page;
            if constexpr (IN_MODE == 1) {
                const uint32_t h = bh % H;
                const uint32_t b = bh / H;
                page = (b * Ht + t) * (H * WT) + h * WT;
            } else {
                page = (bh * Ht + t) * WT;
            }
            for (uint32_t j = 0; j < WT; ++j, l1 += in_page) {
                noc_async_read(in_acc.get_noc_addr(page + j), l1, in_page);
            }
        }
        cb_reserve_back(cb_cos, RT);
        cb_reserve_back(cb_sin, RT);
        uint32_t lc = get_write_ptr(cb_cos);
        uint32_t ls = get_write_ptr(cb_sin);
        for (uint32_t j = 0; j < RT; ++j, lc += cos_tile, ls += sin_tile) {
            noc_async_read(cos_acc.get_noc_addr(t * RT + j), lc, cos_tile);
            noc_async_read(sin_acc.get_noc_addr(t * RT + j), ls, sin_tile);
        }
        noc_async_read_barrier();
        cb_push_back(cb_in, IN_MODE == 2 ? 32 : WT);
        cb_push_back(cb_cos, RT);
        cb_push_back(cb_sin, RT);
    }
}
