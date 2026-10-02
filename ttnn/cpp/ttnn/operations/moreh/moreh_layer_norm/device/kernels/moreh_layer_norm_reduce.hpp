// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/transpose.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_plan_args.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"

ALWI void copy_moreh_statistic(DataflowBuffer source, uint32_t dst) {
    constexpr bool last_dim = get_compile_time_arg_val(9) == 1;
    if constexpr (last_dim) {
        // The writer consumes a row of statistics; a W reduction produces a
        // column. Transpose both the faces and the elements within each face.
        reconfig_data_format_srca(source.get_id());
        transpose_init(source.get_id());
        transpose_tile(source.get_id(), 0, dst);
    } else {
        copy_tile_init_with_dt(source);
        copy_tile(source.get_id(), 0, dst);
    }
}

// Both moments use the same shape, scaling and auxiliary recipe. Their
// lifetimes are disjoint, so they share the resident block and accumulator.
template <bool Variance, uint32_t I>
using MorehMomentCall =
    ttnn::kernel_lib::BoundReduceCallArgs<ttnn::kernel_lib::ReduceCallAtT<15, I>, 31, 1, Variance ? 28 : 24, 27>;

template <bool Variance, bool Streaming>
ALWI void reduce_moreh_moment() {
    constexpr uint32_t origin_h = get_compile_time_arg_val(1);
    constexpr uint32_t origin_w = get_compile_time_arg_val(2);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(3);
    constexpr bool last_dim = get_compile_time_arg_val(9) == 1;
    constexpr uint32_t block_tiles = get_compile_time_arg_val(11);
    constexpr uint32_t buffer_tiles = get_compile_time_arg_val(12);
    constexpr uint32_t call_count = get_compile_time_arg_val(14);
    constexpr uint32_t num_blocks = num_tiles < block_tiles ? 1 : num_tiles / block_tiles;
    constexpr uint32_t input_cb = Variance && !Streaming ? 25 : 0;
    constexpr uint32_t ht = (origin_h + 31) / 32;
    constexpr uint32_t wt = (origin_w + 31) / 32;
    DataflowBuffer input(input_cb);
    DataflowBuffer values(31);
    DataflowBuffer mean(24);
    DataflowBuffer centered(25);
    DataflowBuffer mask_h(5);
    DataflowBuffer mask_w(6);
    if constexpr (!Streaming) {
        input.wait_front(num_tiles);
    }
    if constexpr (Variance && Streaming) {
        mean.wait_front(1);
    }
    const auto mask_hw = [&](uint32_t tile_index) {
        if constexpr (!last_dim && origin_h % 32 != 0) {
            if (((tile_index / wt) + 1) % ht == 0) {
                copy_tile_init_with_dt(mask_h);
                copy_tile(5, 0, 1);
                mask_tile_init();
                mask_tile(0, 1);
            }
        }
        if constexpr (!last_dim && origin_w % 32 != 0) {
            if ((tile_index + 1) % wt == 0) {
                copy_tile_init_with_dt(mask_w);
                copy_tile(6, 0, 1);
                mask_tile_init();
                mask_tile(0, 1);
            }
        }
    };
    for (uint32_t block = 0; block < num_blocks; ++block) {
        const uint32_t current_tiles = block + 1 == num_blocks ? num_tiles - block * block_tiles : block_tiles;
        values.reserve_back(buffer_tiles);
        for (uint32_t tile = 0; tile < current_tiles; ++tile) {
            const uint32_t index = block * block_tiles + tile;
            const uint32_t input_index = Streaming ? 0 : index;
            if constexpr (Streaming) {
                input.wait_front(1);
            }
            if constexpr (Variance && Streaming) {
                // Preserve the intermediate format of x - mean before squaring.
                centered.reserve_back(1);
                tile_regs_acquire();
                if constexpr (last_dim) {
                    sub_bcast_cols_init_with_dt(input, mean);
                    sub_tiles_bcast_cols(input_cb, 24, 0, 0, 0);
                } else {
                    sub_bcast_scalar_init_with_dt(input, mean);
                    sub_tiles_bcast_scalar(input_cb, 24, 0, 0, 0);
                }
                mask_hw(index);
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(25);
                pack_tile(0, 25);
                tile_regs_release();
                centered.push_back(1);
                centered.wait_front(1);
            }
            tile_regs_acquire();
            if constexpr (Variance) {
                auto source = Streaming ? centered : input;
                mul_tiles_init_with_dt(source, source);
                mul_tiles(source.get_id(), source.get_id(), input_index, input_index, 0);
            } else {
                copy_tile_init_with_dt(input);
                copy_tile(input_cb, input_index, 0);
                mask_hw(index);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(31);
            pack_tile<true>(0, 31, tile);
            tile_regs_release();
            if constexpr (Variance && Streaming) {
                centered.pop_front(1);
            }
            if constexpr (Streaming) {
                input.pop_front(1);
            }
        }
        values.push_back(buffer_tiles);
        if (block == 0) {
            compute_kernel_lib::reduce<MorehMomentCall<Variance, 0>>();
        } else if constexpr (call_count > 1) {
            if (block + 1 == num_blocks) {
                compute_kernel_lib::reduce<MorehMomentCall<Variance, call_count - 1>>();
            } else {
                compute_kernel_lib::reduce<MorehMomentCall<Variance, 1>>();
            }
        }
        values.pop_front(buffer_tiles);
    }
}
