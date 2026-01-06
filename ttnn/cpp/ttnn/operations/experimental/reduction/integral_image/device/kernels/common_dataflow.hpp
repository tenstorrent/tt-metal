// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common.hpp"

template <typename addr_gen_t>
FORCE_INLINE void write_to_dram(
    uint32_t cb, const addr_gen_t& addr_gtor, uint32_t write_tile_id, uint32_t num_tiles = ONE_TILE) {
    ReadCBGuard read_guard{cb, num_tiles};

    uint32_t l1_read_addr{get_read_ptr(cb)};
    noc_async_write_tile(write_tile_id, addr_gtor, l1_read_addr);
    noc_async_write_barrier();
}

template <typename addr_gen_type>
FORCE_INLINE void load_from_dram(
    uint32_t cb, const addr_gen_type& addr_gtor, uint32_t read_tile_id, uint32_t num_tiles = ONE_TILE) {
    WriteCBGuard write_guard{cb, num_tiles};

    uint32_t l1_write_addr{get_write_ptr(cb)};
    noc_async_read_tile(read_tile_id, addr_gtor, l1_write_addr);
    noc_async_read_barrier();
}

template <typename InputAccessorArgs, typename OutputAccessorArgs>
struct IntImgCTAs {
    const uint32_t start_cb;
    const uint32_t input_cb;
    const uint32_t acc_cb;
    const uint32_t cumsum_stage_0_cb;
    const uint32_t cumsum_stage_1_cb;
    const uint32_t cumsum_stage_2_cb;
    const uint32_t output_cb;
    const uint32_t axis_2_buffer_cb;    // covers entire propagation
    const uint32_t axis_3_buffer_cb;    // each tile is spawned from broadcasting the last row of
                                        // upper block across all rows of a given tile
    const uint32_t tile_height;
    const uint32_t tile_width;
    const uint32_t block_depth;
    const uint32_t num_channels;  // axis 4/4
    const uint32_t input_height;  // axis 3/4
    const uint32_t input_depth;   // axis 2/4
    const uint32_t num_batches;   // axis 1/4
    const uint32_t cores_x;
    const uint32_t cores_y;
    const InputAccessorArgs input_args;
    const OutputAccessorArgs output_args;  // reused for reading upper block for propagation.
};

FORCE_INLINE constexpr auto get_ctas() {
    constexpr auto input_args = TensorAccessorArgs<18>();
    constexpr auto output_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    return IntImgCTAs<decltype(input_args), decltype(output_args)>{
        get_compile_time_arg_val(0),
        get_compile_time_arg_val(1),
        get_compile_time_arg_val(2),
        get_compile_time_arg_val(3),
        get_compile_time_arg_val(4),
        get_compile_time_arg_val(5),
        get_compile_time_arg_val(6),
        get_compile_time_arg_val(7),
        get_compile_time_arg_val(8),
        get_compile_time_arg_val(9),
        get_compile_time_arg_val(10),
        get_compile_time_arg_val(11),
        get_compile_time_arg_val(12),
        get_compile_time_arg_val(13),
        get_compile_time_arg_val(14),
        get_compile_time_arg_val(15),
        get_compile_time_arg_val(16),
        get_compile_time_arg_val(17),
        input_args,
        output_args,
    };
}
