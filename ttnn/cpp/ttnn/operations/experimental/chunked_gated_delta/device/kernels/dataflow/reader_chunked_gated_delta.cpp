// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/debug/device_print.h"

void kernel_main() {
    const uint32_t g_exp_addr = get_arg_val<uint32_t>(0);
    const uint32_t factor_addr = get_arg_val<uint32_t>(1);
    const uint32_t bktv_addr = get_arg_val<uint32_t>(2);
    const uint32_t state_addr = get_arg_val<uint32_t>(3);
    // Per-core slice of the global head dimension. ``num_heads`` is how many heads
    // this core is responsible for; ``head_offset`` is the global index of the
    // first head, used to compute absolute DRAM page IDs.
    const uint32_t num_heads = get_arg_val<uint32_t>(4);
    const uint32_t head_offset = get_arg_val<uint32_t>(5);

    constexpr uint32_t g_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t factor_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t bktv_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t src0_cb_index = get_compile_time_arg_val(3);
    constexpr uint32_t seq_len = get_compile_time_arg_val(4);
    constexpr uint32_t dim_k = get_compile_time_arg_val(5);
    constexpr uint32_t dim_v = get_compile_time_arg_val(6);

    constexpr auto g_exp_args = TensorAccessorArgs<7>();
    constexpr auto factor_args = TensorAccessorArgs<g_exp_args.next_compile_time_args_offset()>();
    constexpr auto bktv_args = TensorAccessorArgs<factor_args.next_compile_time_args_offset()>();
    constexpr auto state_args = TensorAccessorArgs<bktv_args.next_compile_time_args_offset()>();

    constexpr uint32_t tile_hw = 32;
    constexpr uint32_t factor_ht = dim_k / tile_hw;
    constexpr uint32_t factor_wt = dim_k / tile_hw;
    constexpr uint32_t factor_tiles_per_step = factor_ht * factor_wt;
    constexpr uint32_t bktv_ht = dim_k / tile_hw;
    constexpr uint32_t bktv_wt = dim_v / tile_hw;
    constexpr uint32_t bktv_tiles_per_step = bktv_ht * bktv_wt;
    constexpr uint32_t g_tiles_per_step = 1;

    const auto g_exp_accessor = TensorAccessor(g_exp_args, g_exp_addr);
    const auto factor_accessor = TensorAccessor(factor_args, factor_addr);
    const auto bktv_accessor = TensorAccessor(bktv_args, bktv_addr);
    const auto state_accessor = TensorAccessor(state_args, state_addr);

    Noc noc;
    CircularBuffer g_cb(g_cb_index);
    CircularBuffer factor_cb(factor_cb_index);
    CircularBuffer bktv_cb(bktv_cb_index);
    CircularBuffer state_cb(src0_cb_index);

    const uint32_t tile_bytes = get_tile_size(g_cb_index);

    constexpr uint32_t onetile = 1;
    uint32_t pushed_tiles = 0;

    DEVICE_PRINT("g_exp_addr = {}\n", g_exp_addr);
    DEVICE_PRINT("factor_addr = {}, {}\n", factor_addr, g_exp_args.next_compile_time_args_offset());
    DEVICE_PRINT("bktv_addr   = {}, {}\n", bktv_addr, factor_args.next_compile_time_args_offset());
    DEVICE_PRINT("state_addr  = {}, {}\n", state_addr, bktv_args.next_compile_time_args_offset());

    DEVICE_PRINT("num_heads: {}, head_offset: {}\n", num_heads, head_offset);
    DEVICE_PRINT("seq_len: {}\n", seq_len);
    DEVICE_PRINT("dim_k: {}\n", dim_k);
    DEVICE_PRINT("dim_v: {}\n", dim_v);
    DEVICE_PRINT("bktv_tiles_per_step: {}\n", bktv_tiles_per_step);
    DEVICE_PRINT("bktv_ht: {}\n", bktv_ht);
    DEVICE_PRINT("bktv_wt: {}\n", bktv_wt);
    DEVICE_PRINT("tile_bytes: {}\n", tile_bytes);

    for (uint32_t head_local = 0; head_local < num_heads; ++head_local) {
        const uint32_t head = head_local + head_offset;
        state_cb.reserve_back(bktv_tiles_per_step);
        for (uint32_t tile = 0; tile < bktv_tiles_per_step; ++tile) {
            noc.async_read(
                state_accessor,
                state_cb,
                tile_bytes,
                {.page_id = head * bktv_tiles_per_step + tile},
                {.offset_bytes = tile * tile_bytes});
        }
        noc.async_read_barrier();
        state_cb.push_back(bktv_tiles_per_step);
        pushed_tiles += bktv_tiles_per_step;

        for (uint32_t seq_idx = 0; seq_idx < seq_len; ++seq_idx) {
            const uint32_t g_page_base = head * seq_len * g_tiles_per_step + seq_idx * g_tiles_per_step;
            g_cb.reserve_back(onetile);
            noc.async_read(g_exp_accessor, g_cb, tile_bytes, {.page_id = g_page_base}, {.offset_bytes = 0});
            noc.async_read_barrier();

            g_cb.push_back(onetile);

            const uint32_t factor_page_base = head * seq_len * factor_tiles_per_step + seq_idx * factor_tiles_per_step;
            factor_cb.reserve_back(factor_tiles_per_step);
            for (uint32_t tile = 0; tile < factor_tiles_per_step; ++tile) {
                noc.async_read(
                    factor_accessor,
                    factor_cb,
                    tile_bytes,
                    {.page_id = factor_page_base + tile},
                    {.offset_bytes = tile * tile_bytes});
            }
            noc.async_read_barrier();
            factor_cb.push_back(factor_tiles_per_step);

            const uint32_t bktv_page_base = head * seq_len * bktv_tiles_per_step + seq_idx * bktv_tiles_per_step;
            bktv_cb.reserve_back(bktv_tiles_per_step);
            for (uint32_t tile = 0; tile < bktv_tiles_per_step; ++tile) {
                noc.async_read(
                    bktv_accessor,
                    bktv_cb,
                    tile_bytes,
                    {.page_id = bktv_page_base + tile},
                    {.offset_bytes = tile * tile_bytes});
            }
            noc.async_read_barrier();
            bktv_cb.push_back(bktv_tiles_per_step);
        }
        SliceRange sr = SliceRange{.h0 = 0, .h1 = 4, .hs = 1, .w0 = 0, .w1 = 4, .ws = 1};
        // DEVICE_PRINT("reader_head: state reserve {}\n", head);
        // DEVICE_PRINT("state_cb: {}\n", TileSlice(src0_cb_index, 0, sr, TSLICE_OUTPUT_CB, TSLICE_WR_PTR, true, true));
        // DEVICE_PRINT("g_cb: {}\n", TileSlice(g_cb_index, 0, sr, TSLICE_OUTPUT_CB, TSLICE_WR_PTR, true, true));
    }
}
