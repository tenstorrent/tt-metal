// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Reader for fused distributed GroupNorm.
 *
 * Two passes over the input (streaming; input_cb is block-sized):
 *   PRE  — push every input tile so compute can accumulate per-group stats
 *   POST — re-push every input tile (+ optional weight/bias) for normalize
 *
 * Weight/bias: TILE (face-row broadcast) or ROW_MAJOR last-dim C stick.
 * No RoPE / input_mask in v1.
 */

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include <tt-metalium/constants.hpp>
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t weight_cb = get_compile_time_arg_val(1);
    constexpr uint32_t bias_cb = get_compile_time_arg_val(2);
    constexpr uint32_t num_tile_cols = get_compile_time_arg_val(3);
    constexpr uint32_t block_size = get_compile_time_arg_val(4);
    constexpr uint32_t has_weight = get_compile_time_arg_val(5);
    constexpr uint32_t has_bias = get_compile_time_arg_val(6);
    constexpr uint32_t weight_is_tile = get_compile_time_arg_val(7);
    constexpr uint32_t bias_is_tile = get_compile_time_arg_val(8);
    constexpr auto input_args = TensorAccessorArgs<9>();
    constexpr auto weight_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto bias_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();

    uint32_t arg_idx = 0;
    const uint32_t input_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t weight_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t bias_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t tile_row_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t tile_row_end = get_arg_val<uint32_t>(arg_idx++);

    const uint32_t input_tile_bytes = get_tile_size(input_cb);
    const uint32_t weight_tile_bytes = get_tile_size(weight_cb);
    const uint32_t bias_tile_bytes = get_tile_size(bias_cb);

    const auto input_accessor = TensorAccessor(input_args, input_addr);
    const auto weight_accessor = TensorAccessor(weight_args, weight_addr);
    const auto bias_accessor = TensorAccessor(bias_args, bias_addr);

    constexpr uint32_t kTileHW = tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH;

    auto read_input_pass = [&]() {
        for (uint32_t tile_row = tile_row_start; tile_row < tile_row_end; tile_row++) {
            const uint32_t input_tile_idx = tile_row * num_tile_cols;
            for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                const uint32_t tiles_in_block =
                    ((num_tile_cols - col_tile) >= block_size) ? block_size : (num_tile_cols - col_tile);
                cb_reserve_back(input_cb, tiles_in_block);
                uint32_t wr = get_write_ptr(input_cb);
                for (uint32_t i = 0; i < tiles_in_block; i++) {
                    noc_async_read_tile(input_tile_idx + col_tile + i, input_accessor, wr);
                    wr += input_tile_bytes;
                }
                noc_async_read_barrier();
                cb_push_back(input_cb, tiles_in_block);
            }
        }
    };

    auto read_affine_tile_face_rows = [&](uint32_t cb, uint32_t tile_bytes, const auto& accessor) {
        const uint32_t datum_bytes = tile_bytes / kTileHW;
        const uint32_t face_row_bytes = tt::constants::FACE_WIDTH * datum_bytes;
        const uint32_t face_bytes = tt::constants::FACE_HW * datum_bytes;
        cb_reserve_back(cb, num_tile_cols);
        uint32_t wr = get_write_ptr(cb);
        for (uint32_t c = 0; c < num_tile_cols; c++) {
            const uint64_t noc = get_noc_addr(c, accessor);
            volatile tt_l1_ptr uint8_t* dst = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(wr);
            for (uint32_t b = 0; b < tile_bytes; b++) {
                dst[b] = 0;
            }
            noc_async_read(noc, wr, face_row_bytes);
            noc_async_read(noc + face_bytes, wr + face_bytes, face_row_bytes);
            wr += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb, num_tile_cols);
    };

    // -------- PRE pass --------
    {
        DeviceZoneScopedN("GN_R_PRE");
        read_input_pass();
    }

    // -------- Side inputs for POST (once, before POST re-read) --------
    if constexpr (has_weight) {
        if constexpr (weight_is_tile) {
            read_affine_tile_face_rows(weight_cb, weight_tile_bytes, weight_accessor);
        } else {
            cb_reserve_back(weight_cb, 1);
            noc_async_read(get_noc_addr(0, weight_accessor), get_write_ptr(weight_cb), get_tile_size(weight_cb));
            noc_async_read_barrier();
            cb_push_back(weight_cb, 1);
        }
    }
    if constexpr (has_bias) {
        if constexpr (bias_is_tile) {
            read_affine_tile_face_rows(bias_cb, bias_tile_bytes, bias_accessor);
        } else {
            cb_reserve_back(bias_cb, 1);
            noc_async_read(get_noc_addr(0, bias_accessor), get_write_ptr(bias_cb), get_tile_size(bias_cb));
            noc_async_read_barrier();
            cb_push_back(bias_cb, 1);
        }
    }

    // -------- POST pass --------
    {
        DeviceZoneScopedN("GN_R_POST");
        read_input_pass();
    }
}
