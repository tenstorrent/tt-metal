// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// reader_x_first bake-off: the rms_norm reader's start-up ordering, isolated (NCRISC).
//
// Same CT/RT contract, same tile ids, same byte counts and same CB contents as the op's
// rms_norm_reader.cpp; only the ORDER of the three start-up stages differs, selected by the
// named CT arg READER_ORDER:
//
//   0  baseline (the op today):  scaler fill -> gamma issue -> gamma barrier -> gamma push ->
//                                per block { x issue -> x barrier -> x push }
//   1  x-first, ONE barrier:     x(block 0) issue -> gamma issue -> scaler fill ->
//                                barrier -> x push -> gamma push -> blocks 1..
//   2  x-first, split barriers:  x(block 0) issue -> x barrier -> x push ->
//                                gamma issue -> scaler fill -> barrier -> gamma push -> blocks 1..
//   3  x-first, trid barriers:   set_trid(1) -> x(block 0) issue -> set_trid(2) -> gamma issue ->
//                                scaler fill -> barrier_with_trid(1) -> x push ->
//                                barrier_with_trid(2) -> gamma push -> blocks 1..   (TILE x, TILE gamma)
//   4  scaler first, x, gamma:   scaler fill -> x(block 0) issue -> barrier -> x push ->
//                                gamma issue -> barrier -> gamma push -> blocks 1..
//   5  scaler first, one barrier: scaler fill -> x issue -> gamma issue -> barrier -> x push -> gamma push
//   6  scaler first, trid split: scaler fill -> x issue[1] -> gamma issue[2] -> bar(1) -> x push -> bar(2) -> gamma
//   push 7  scaler last:              x issue -> barrier -> x push -> gamma issue -> barrier -> gamma push -> scaler
//   fill 8  x, scaler, gamma:         x issue -> barrier -> x push -> scaler fill -> gamma issue -> barrier -> gamma
//   push
//
// Measured (focus geometry): a RISC scaler fill placed while NoC read data is landing in the same L1
// slows ~6x (305 ns -> ~1.7 us; L1 port arbitration favours the NoC), so "RISC work under the DRAM
// latency" is NOT free when that work is L1 stores. Orders 4..7 keep the fill out of that window.
//
// ROW_MAJOR x goes through read_sticks_for_tilize<TILE granularity>, which owns its own barrier
// and push per tile-row, so orders 1 and 2 coincide there (block 0 of x lands and is published
// before gamma / scaler are even issued); order 3 is not expressible on that path.
//
// Order 3 relies on a mechanism fact, not a helper: the plain read path (ncrisc_noc_fast_read)
// never writes NOC_PACKET_TAG, so noc_async_read_set_trid() on the read command buffer tags every
// following noc_async_read() until the next set_trid(). This is API-level (dataflow_api.h), not
// raw LLK, but it is a stateful-register contract worth stating.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_x_tiles = get_named_compile_time_arg_val("CB_X_TILES");
    constexpr uint32_t cb_x_sticks = get_named_compile_time_arg_val("CB_X_STICKS");
    constexpr uint32_t cb_scaler = get_named_compile_time_arg_val("CB_SCALER");
    constexpr uint32_t cb_gamma_tiles = get_named_compile_time_arg_val("CB_GAMMA_TILES");
    constexpr uint32_t cb_gamma_sticks = get_named_compile_time_arg_val("CB_GAMMA_STICKS");
    constexpr bool input_rm = get_named_compile_time_arg_val("INPUT_RM") != 0;
    constexpr uint32_t gamma_mode = get_named_compile_time_arg_val("GAMMA_MODE");  // 0 none, 1 TILE, 2 RM
    constexpr uint32_t in_page_bytes = get_named_compile_time_arg_val("IN_PAGE_BYTES");
    constexpr uint32_t in_tile_bytes = get_named_compile_time_arg_val("IN_TILE_BYTES");
    constexpr uint32_t in_elem_bytes = get_named_compile_time_arg_val("IN_ELEM_BYTES");
    constexpr uint32_t gamma_page_bytes = get_named_compile_time_arg_val("GAMMA_PAGE_BYTES");
    constexpr uint32_t gamma_tile_bytes = get_named_compile_time_arg_val("GAMMA_TILE_BYTES");
    constexpr uint32_t gamma_elem_bytes = get_named_compile_time_arg_val("GAMMA_ELEM_BYTES");
    constexpr uint32_t order = get_named_compile_time_arg_val("READER_ORDER");
    constexpr uint32_t tile_rows = 32;

    constexpr auto input_args = TensorAccessorArgs<0>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t row_tile_start = get_arg_val<uint32_t>(2);
    const uint32_t num_blocks_this_core = get_arg_val<uint32_t>(3);
    const uint32_t block_rows = get_arg_val<uint32_t>(4);
    const uint32_t last_block_rows = get_arg_val<uint32_t>(5);
    const uint32_t w_tile_start = get_arg_val<uint32_t>(6);
    const uint32_t core_w_tiles = get_arg_val<uint32_t>(7);
    const uint32_t tensor_w_tiles = get_arg_val<uint32_t>(8);

    const auto input_acc = TensorAccessor(input_args, input_addr, in_page_bytes);
    [[maybe_unused]] const auto gamma_acc = TensorAccessor(gamma_args, gamma_addr, gamma_page_bytes);

    // ---------------- stages (identical bodies in every order) ----------------
    // Empty zones are readiness MARKERS: their ZONE_START timestamp is "this CB just became visible to
    // the consumer" (avail_report.py measures them from the NCRISC kernel start).
    auto fill_scaler = [&]() {
        {
            MaybeDeviceZoneScope("rd_scaler");
            dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
                cb_scaler,
                dataflow_kernel_lib::PoolType::SUM,
                dataflow_kernel_lib::ReduceDim::REDUCE_ROW>();
        }
        {
            MaybeDeviceZoneScope("mark_scaler_ready");
        }
    };

    auto gamma_reserve_issue = [&]() {
        if constexpr (gamma_mode == 1) {
            cb_reserve_back(cb_gamma_tiles, core_w_tiles);
            const uint32_t dst = get_write_ptr(cb_gamma_tiles);
            MaybeDeviceZoneScope("rd_gamma_issue");
            for (uint32_t c = 0; c < core_w_tiles; ++c) {
                noc_async_read(gamma_acc.get_noc_addr(w_tile_start + c), dst + c * gamma_tile_bytes, gamma_tile_bytes);
            }
        } else if constexpr (gamma_mode == 2) {
            const uint32_t chunk_bytes = core_w_tiles * tile_rows * gamma_elem_bytes;
            const uint32_t w_byte_offset = w_tile_start * tile_rows * gamma_elem_bytes;
            cb_reserve_back(cb_gamma_sticks, core_w_tiles);
            {
                MaybeDeviceZoneScope("rd_gamma_zero_fill");
                Noc noc;
                CircularBuffer gamma_sticks(cb_gamma_sticks);
                noc.async_write_zeros(gamma_sticks, (tile_rows - 1) * chunk_bytes, {.offset_bytes = chunk_bytes});
                noc.write_zeros_l1_barrier();
            }
            MaybeDeviceZoneScope("rd_gamma_issue");
            noc_async_read(gamma_acc.get_noc_addr(0, w_byte_offset), get_write_ptr(cb_gamma_sticks), chunk_bytes);
        }
    };

    auto gamma_push = [&]() {
        if constexpr (gamma_mode == 1) {
            cb_push_back(cb_gamma_tiles, core_w_tiles);
        } else if constexpr (gamma_mode == 2) {
            cb_push_back(cb_gamma_sticks, core_w_tiles);
        }
        {
            MaybeDeviceZoneScope("mark_gamma_ready");
        }
    };

    auto rows_of = [&](uint32_t block_idx) {
        return (block_idx + 1 < num_blocks_this_core) ? block_rows : last_block_rows;
    };

    // TILE x: reserve + issue every tile read of one block (no barrier, no push).
    auto x_reserve_issue = [&](uint32_t block_idx) {
        const uint32_t rows = rows_of(block_idx);
        const uint32_t block_row_tile_start = row_tile_start + block_idx * block_rows;
        cb_reserve_back(cb_x_tiles, rows * core_w_tiles);
        uint32_t dst = get_write_ptr(cb_x_tiles);
        MaybeDeviceZoneScope("rd_x_issue");
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t row_base = (block_row_tile_start + r) * tensor_w_tiles + w_tile_start;
            for (uint32_t c = 0; c < core_w_tiles; ++c) {
                noc_async_read(input_acc.get_noc_addr(row_base + c), dst, in_tile_bytes);
                dst += in_tile_bytes;
            }
        }
    };
    auto x_push = [&](uint32_t block_idx) {
        cb_push_back(cb_x_tiles, rows_of(block_idx) * core_w_tiles);
        if (block_idx == 0) {
            MaybeDeviceZoneScope("mark_x0_ready");
        }
    };

    // ROW_MAJOR x: the helper owns reserve / issue / barrier / push per tile-row.
    auto x_rm_block = [&](uint32_t block_idx) {
        const uint32_t rows = rows_of(block_idx);
        const uint32_t block_row_tile_start = row_tile_start + block_idx * block_rows;
        {
            MaybeDeviceZoneScope("rd_x_sticks");
            dataflow_kernel_lib::read_sticks_for_tilize<cb_x_sticks, dataflow_kernel_lib::TilizeGranularity::TILE>(
                input_acc,
                tile_rows * rows,
                core_w_tiles * tile_rows * in_elem_bytes,
                tile_rows * block_row_tile_start,
                w_tile_start * tile_rows * in_elem_bytes);
        }
        if (block_idx == 0) {
            MaybeDeviceZoneScope("mark_x0_ready");
        }
    };

    // The op's per-block x load (issue -> barrier -> push), used for every block not covered by the reorder.
    auto x_block_plain = [&](uint32_t block_idx) {
        if constexpr (input_rm) {
            x_rm_block(block_idx);
        } else {
            x_reserve_issue(block_idx);
            {
                MaybeDeviceZoneScope("rd_x_barrier");
                noc_async_read_barrier();
            }
            x_push(block_idx);
        }
    };

    // ---------------- the orderings under test ----------------
    // S = scaler fill, X = x block 0 (issue / barrier / push), G = gamma (issue / barrier / push).
    //   0  S, G(issue,barrier,push), X(issue,barrier,push)                 -- baseline (the op today)
    //   1  X issue, G issue, S, barrier, X push, G push                     -- x-first, one barrier
    //   2  X issue, barrier, X push, G issue, S, barrier, G push            -- x-first, split barriers
    //   3  X issue[trid1], G issue[trid2], S, bar(1), X push, bar(2), G push
    //   4  S, X issue, barrier, X push, G issue, barrier, G push            -- scaler first, then x, then gamma
    //   5  S, X issue, G issue, barrier, X push, G push                     -- scaler first, one barrier
    //   6  S, X issue[trid1], G issue[trid2], bar(1), X push, bar(2), G push
    //   7  X issue, barrier, X push, G issue, barrier, G push, S            -- scaler last
    //   8  X issue, barrier, X push, S, G issue, barrier, G push            -- x, then scaler, then gamma
    // RM x: X is the stick helper (owns barrier + push), so 1 == 2, 5 == 4; 3 and 6 are not expressible.
    static_assert(order <= 8, "READER_ORDER in 0..8");
    constexpr bool trid_split = (order == 3 || order == 6);
    constexpr bool one_barrier = (order == 1 || order == 5);
    constexpr bool scaler_first = (order == 0 || order == 4 || order == 5 || order == 6);
    constexpr bool scaler_last = (order == 7);
    constexpr bool scaler_after_x = (order == 8);
    static_assert(!(trid_split && input_rm), "trid split not expressible on the stick-helper x path");
    static_assert(!(trid_split && gamma_mode == 2), "trid split bench covers TILE gamma only");

    uint32_t first_plain_block = 0;
    if constexpr (order == 0) {
        fill_scaler();
        gamma_reserve_issue();
        {
            MaybeDeviceZoneScope("rd_gamma_barrier");
            noc_async_read_barrier();
        }
        gamma_push();
    } else {
        if constexpr (scaler_first) {
            fill_scaler();
        }
        // x block 0
        if constexpr (input_rm) {
            x_rm_block(0);
        } else if constexpr (one_barrier) {
            x_reserve_issue(0);
        } else if constexpr (trid_split) {
            noc_async_read_set_trid(1);
            x_reserve_issue(0);
            noc_async_read_set_trid(2);
        } else {
            x_reserve_issue(0);
            {
                MaybeDeviceZoneScope("rd_x_barrier");
                noc_async_read_barrier();
            }
            x_push(0);
        }
        if constexpr (scaler_after_x) {
            fill_scaler();
        }
        // gamma (+ scaler in the middle for orders 1..3)
        gamma_reserve_issue();
        if constexpr (!scaler_first && !scaler_last && !scaler_after_x) {
            fill_scaler();
        }
        if constexpr (one_barrier && !input_rm) {
            {
                MaybeDeviceZoneScope("rd_barrier_x_gamma");
                noc_async_read_barrier();
            }
            x_push(0);
            gamma_push();
        } else if constexpr (trid_split) {
            {
                MaybeDeviceZoneScope("rd_x_barrier");
                noc_async_read_barrier_with_trid(1);
            }
            x_push(0);
            {
                MaybeDeviceZoneScope("rd_gamma_barrier");
                noc_async_read_barrier_with_trid(2);
            }
            gamma_push();
            noc_async_read_set_trid(0);
        } else {
            {
                MaybeDeviceZoneScope("rd_gamma_barrier");
                noc_async_read_barrier();
            }
            gamma_push();
        }
        if constexpr (scaler_last) {
            fill_scaler();
        }
        first_plain_block = 1;
    }

    for (uint32_t block_idx = first_plain_block; block_idx < num_blocks_this_core; ++block_idx) {
        x_block_plain(block_idx);
    }
}
