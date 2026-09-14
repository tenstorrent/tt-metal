// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// reader_x_first bake-off: the consumer (BRISC). Stands in for the compute kernel with the same
// consumption ORDER (x first, block by block; gamma; scaler) and dumps every CB page it consumes
// to DRAM so the host can gate each reader variant on byte-identical CB contents:
//   x      -> out_x      (same shape / layout / tile ids as x)
//   gamma  -> out_gamma  (TILE: same tile ids as gamma; RM: a (32, W) stick block, row k = chunk k)
//   scaler -> out_scaler (one bf16 tile per active core, tile id = core index)
// The dump is identical work for every variant, so a kernel-duration delta is the reader's.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_x_tiles = get_named_compile_time_arg_val("CB_X_TILES");
    constexpr uint32_t cb_x_sticks = get_named_compile_time_arg_val("CB_X_STICKS");
    constexpr uint32_t cb_scaler = get_named_compile_time_arg_val("CB_SCALER");
    constexpr uint32_t cb_gamma_tiles = get_named_compile_time_arg_val("CB_GAMMA_TILES");
    constexpr uint32_t cb_gamma_sticks = get_named_compile_time_arg_val("CB_GAMMA_STICKS");
    constexpr bool input_rm = get_named_compile_time_arg_val("INPUT_RM") != 0;
    constexpr uint32_t gamma_mode = get_named_compile_time_arg_val("GAMMA_MODE");
    constexpr uint32_t in_page_bytes = get_named_compile_time_arg_val("IN_PAGE_BYTES");
    constexpr uint32_t in_tile_bytes = get_named_compile_time_arg_val("IN_TILE_BYTES");
    constexpr uint32_t in_elem_bytes = get_named_compile_time_arg_val("IN_ELEM_BYTES");
    constexpr uint32_t gamma_tile_bytes = get_named_compile_time_arg_val("GAMMA_TILE_BYTES");
    constexpr uint32_t gamma_elem_bytes = get_named_compile_time_arg_val("GAMMA_ELEM_BYTES");
    constexpr uint32_t out_gamma_page_bytes = get_named_compile_time_arg_val("OUT_GAMMA_PAGE_BYTES");
    constexpr uint32_t scaler_tile_bytes = get_named_compile_time_arg_val("SCALER_TILE_BYTES");
    constexpr uint32_t tile_rows = 32;

    constexpr auto out_x_args = TensorAccessorArgs<0>();
    constexpr auto out_gamma_args = TensorAccessorArgs<out_x_args.next_compile_time_args_offset()>();
    constexpr auto out_scaler_args = TensorAccessorArgs<out_gamma_args.next_compile_time_args_offset()>();

    const uint32_t out_x_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_scaler_addr = get_arg_val<uint32_t>(2);
    const uint32_t row_tile_start = get_arg_val<uint32_t>(3);
    const uint32_t num_blocks_this_core = get_arg_val<uint32_t>(4);
    const uint32_t block_rows = get_arg_val<uint32_t>(5);
    const uint32_t last_block_rows = get_arg_val<uint32_t>(6);
    const uint32_t w_tile_start = get_arg_val<uint32_t>(7);
    const uint32_t core_w_tiles = get_arg_val<uint32_t>(8);
    const uint32_t tensor_w_tiles = get_arg_val<uint32_t>(9);
    const uint32_t core_index = get_arg_val<uint32_t>(10);

    const auto out_x_acc = TensorAccessor(out_x_args, out_x_addr, in_page_bytes);
    [[maybe_unused]] const auto out_gamma_acc = TensorAccessor(out_gamma_args, out_gamma_addr, out_gamma_page_bytes);
    const auto out_scaler_acc = TensorAccessor(out_scaler_args, out_scaler_addr, scaler_tile_bytes);

    // ---- x, block by block (the compute kernel's first wait) ----
    for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
        const uint32_t rows = (block_idx + 1 < num_blocks_this_core) ? block_rows : last_block_rows;
        const uint32_t block_row_tile_start = row_tile_start + block_idx * block_rows;
        if constexpr (input_rm) {
            MaybeDeviceZoneScope("wr_x_sticks");
            dataflow_kernel_lib::write_sticks_after_untilize<cb_x_sticks>(
                out_x_acc,
                tile_rows * rows,
                core_w_tiles * tile_rows * in_elem_bytes,
                tile_rows * block_row_tile_start,
                w_tile_start * tile_rows * in_elem_bytes);
        } else {
            const uint32_t block_tiles = rows * core_w_tiles;
            {
                MaybeDeviceZoneScope("wr_x_wait");
                cb_wait_front(cb_x_tiles, block_tiles);
            }
            if (block_idx == 0) {
                MaybeDeviceZoneScope("mark_wr_x0_seen");
            }
            {
                MaybeDeviceZoneScope("wr_x_write");
                uint32_t src = get_read_ptr(cb_x_tiles);
                for (uint32_t r = 0; r < rows; ++r) {
                    const uint32_t row_base = (block_row_tile_start + r) * tensor_w_tiles + w_tile_start;
                    for (uint32_t c = 0; c < core_w_tiles; ++c) {
                        noc_async_write(src, out_x_acc.get_noc_addr(row_base + c), in_tile_bytes);
                        src += in_tile_bytes;
                    }
                }
                noc_async_write_barrier();
            }
            cb_pop_front(cb_x_tiles, block_tiles);
        }
    }

    // ---- gamma ----
    if constexpr (gamma_mode == 1) {
        {
            MaybeDeviceZoneScope("wr_gamma_wait");
            cb_wait_front(cb_gamma_tiles, core_w_tiles);
        }
        const uint32_t src = get_read_ptr(cb_gamma_tiles);
        for (uint32_t c = 0; c < core_w_tiles; ++c) {
            noc_async_write(src + c * gamma_tile_bytes, out_gamma_acc.get_noc_addr(w_tile_start + c), gamma_tile_bytes);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_gamma_tiles, core_w_tiles);
    } else if constexpr (gamma_mode == 2) {
        const uint32_t chunk_bytes = core_w_tiles * tile_rows * gamma_elem_bytes;
        const uint32_t w_byte_offset = w_tile_start * tile_rows * gamma_elem_bytes;
        {
            MaybeDeviceZoneScope("wr_gamma_wait");
            cb_wait_front(cb_gamma_sticks, core_w_tiles);
        }
        const uint32_t src = get_read_ptr(cb_gamma_sticks);
        for (uint32_t k = 0; k < tile_rows; ++k) {
            noc_async_write(src + k * chunk_bytes, out_gamma_acc.get_noc_addr(k, w_byte_offset), chunk_bytes);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_gamma_sticks, core_w_tiles);
    }

    // ---- scaler ----
    {
        MaybeDeviceZoneScope("wr_scaler_wait");
        cb_wait_front(cb_scaler, 1);
    }
    noc_async_write(get_read_ptr(cb_scaler), out_scaler_acc.get_noc_addr(core_index), scaler_tile_bytes);
    noc_async_write_barrier();
    cb_pop_front(cb_scaler, 1);
}
