// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ttnn.bringup.rms_norm, the split program's reader (CHANGELOG 6; rms_norm_split_compute.cpp has the modes).
// Unit u = (row tile r = u / B, column block b = u % B); the unit's input tiles are the row's window tiles
// [C0T + b WB, C0T + (b + 1) WB) of an input whose rows are WT_IN tiles wide.

#include <stdint.h>

#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

constexpr uint32_t MODE_LOCAL = 0;
constexpr uint32_t MODE_STATS_OUT = 1;
constexpr uint32_t MODE_STATS_IN = 2;

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(1);
    constexpr uint32_t cb_stats = get_compile_time_arg_val(2);
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(3);
    constexpr uint32_t WT_IN = get_compile_time_arg_val(4);
    constexpr uint32_t C0T = get_compile_time_arg_val(5);
    constexpr uint32_t WB = get_compile_time_arg_val(6);
    constexpr uint32_t BLK = get_compile_time_arg_val(7);
    constexpr uint32_t B = get_compile_time_arg_val(8);
    constexpr uint32_t G = get_compile_time_arg_val(9);
    constexpr uint32_t MODE = get_compile_time_arg_val(10);
    constexpr uint32_t HAS_GAMMA = get_compile_time_arg_val(11);
    constexpr auto x_args = TensorAccessorArgs<12>();
    constexpr auto stats_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto gamma_args = TensorAccessorArgs<stats_args.next_compile_time_args_offset()>();

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t stats_addr = get_arg_val<uint32_t>(1);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(2);
    const uint32_t unit_start = get_arg_val<uint32_t>(3);
    const uint32_t unit_end = get_arg_val<uint32_t>(4);

    const uint32_t x_tile_bytes = get_tile_size(cb_x);
    const auto x_acc = TensorAccessor(x_args, x_addr);

    Noc noc;
    CircularBuffer cbx(cb_x);

    if constexpr (MODE != MODE_STATS_OUT) {
        dataflow_kernel_lib::
            calculate_and_prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
    }

    auto read_x = [&](uint32_t first_page) {
        for (uint32_t col = 0; col < WB; col += BLK) {
            cbx.reserve_back(BLK);
            uint32_t off = 0;
            for (uint32_t i = 0; i < BLK; ++i) {
                noc.async_read(x_acc, cbx, x_tile_bytes, {.page_id = first_page + col + i}, {.offset_bytes = off});
                off += x_tile_bytes;
            }
            noc.async_read_barrier();
            cbx.push_back(BLK);
        }
    };

    for (uint32_t u = unit_start; u < unit_end; ++u) {
        const uint32_t r = u / B;
        const uint32_t b = u % B;
        const uint32_t x_first = r * WT_IN + C0T + b * WB;
        if constexpr (MODE == MODE_STATS_IN) {
            const auto s_acc = TensorAccessor(stats_args, stats_addr);
            CircularBuffer cbs(cb_stats);
            const uint32_t s_tile_bytes = get_tile_size(cb_stats);
            cbs.reserve_back(G);
            uint32_t off = 0;
            for (uint32_t g = 0; g < G; ++g) {
                noc.async_read(s_acc, cbs, s_tile_bytes, {.page_id = r * G + g}, {.offset_bytes = off});
                off += s_tile_bytes;
            }
            noc.async_read_barrier();
            cbs.push_back(G);
        } else {
            read_x(x_first);  // pass A
        }
        if constexpr (MODE == MODE_STATS_OUT) {
            continue;
        }
        // pass B: x again, block by block, with the block's gamma (row 0 of each tile: the first row of faces 0, 1)
        if constexpr (HAS_GAMMA) {
            const auto g_acc = TensorAccessor(gamma_args, gamma_addr);
            CircularBuffer cbg(cb_gamma);
            const uint32_t g_tile_bytes = get_tile_size(cb_gamma);
            const uint32_t datum = g_tile_bytes / tt::constants::TILE_HW;
            const uint32_t face_row_bytes = tt::constants::FACE_WIDTH * datum;
            const uint32_t face_bytes = tt::constants::FACE_HW * datum;
            for (uint32_t col = 0; col < WB; col += BLK) {
                cbx.reserve_back(BLK);
                uint32_t off = 0;
                for (uint32_t i = 0; i < BLK; ++i) {
                    noc.async_read(x_acc, cbx, x_tile_bytes, {.page_id = x_first + col + i}, {.offset_bytes = off});
                    off += x_tile_bytes;
                }
                cbg.reserve_back(BLK);
                uint32_t goff = 0;
                for (uint32_t i = 0; i < BLK; ++i) {
                    const uint32_t page = b * WB + col + i;
                    noc.async_read(
                        g_acc, cbg, face_row_bytes, {.page_id = page, .offset_bytes = 0}, {.offset_bytes = goff});
                    noc.async_read(
                        g_acc,
                        cbg,
                        face_row_bytes,
                        {.page_id = page, .offset_bytes = face_bytes},
                        {.offset_bytes = goff + face_bytes});
                    goff += g_tile_bytes;
                }
                noc.async_read_barrier();
                cbx.push_back(BLK);
                cbg.push_back(BLK);
            }
        } else {
            read_x(x_first);
        }
    }
}
