// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// per_token_cast_back, tile output: write each unit's G tiles to the TILE output at (tile-row, unit column).
// Recomputes the reader's count-aware unit range from the same routing tables.

#include <cstdint>

#ifndef TILES_PER_SCALE
#define TILES_PER_SCALE 1
#endif

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_to_fp8/device/kernels/dataflow/count_aware_range.hpp"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_region_scratch = get_compile_time_arg_val(1);
    constexpr uint32_t cb_counts_scratch = get_compile_time_arg_val(2);
    constexpr uint32_t cb_table_scratch = get_compile_time_arg_val(3);
    constexpr uint32_t G = get_compile_time_arg_val(4);
    constexpr uint32_t Wt = get_compile_time_arg_val(5);
    constexpr uint32_t experts_per_chip = get_compile_time_arg_val(6);
    constexpr uint32_t max_rows = get_compile_time_arg_val(7);
    constexpr uint32_t num_cores = get_compile_time_arg_val(8);
    constexpr auto output_args = TensorAccessorArgs<9>();
    constexpr auto region_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr auto counts_args = TensorAccessorArgs<region_args.next_compile_time_args_offset()>();
    constexpr auto table_args = TensorAccessorArgs<counts_args.next_compile_time_args_offset()>();
    constexpr uint32_t units_per_row = Wt / G;
    // The scale side of each unit (32 64-B-aligned tail windows -> G bf16 column-0 scale tiles) runs here, on the
    // other NoC, so the reader only issues the e4m3 row reads.
    constexpr uint32_t cb_scale = get_compile_time_arg_val(table_args.next_compile_time_args_offset());
    constexpr uint32_t cb_scale_scratch = get_compile_time_arg_val(table_args.next_compile_time_args_offset() + 1);
    constexpr uint32_t seg_bytes = get_compile_time_arg_val(table_args.next_compile_time_args_offset() + 2);
    constexpr uint32_t seg_tiles = get_compile_time_arg_val(table_args.next_compile_time_args_offset() + 3);
    constexpr uint32_t scale_slot_bytes = get_compile_time_arg_val(table_args.next_compile_time_args_offset() + 4);
    constexpr auto input_args = TensorAccessorArgs<table_args.next_compile_time_args_offset() + 5>();
    constexpr uint32_t DRAM_ALIGN = 64;

    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t region_addr = get_arg_val<uint32_t>(1);
    const uint32_t counts_addr = get_arg_val<uint32_t>(2);
    const uint32_t table_addr = get_arg_val<uint32_t>(3);
    const uint32_t core_id = get_arg_val<uint32_t>(4);
    const uint32_t input_addr = get_arg_val<uint32_t>(5);

    Noc noc;
    CircularBuffer cb_out_obj(cb_out);
    const auto output = TensorAccessor(output_args, output_addr, cb_out_obj.get_tile_size());
    const auto region = TensorAccessor(region_args, region_addr);
    const auto counts = TensorAccessor(counts_args, counts_addr);
    const auto table = TensorAccessor(table_args, table_addr);
    CircularBuffer region_scratch(cb_region_scratch);
    CircularBuffer counts_scratch(cb_counts_scratch);
    CircularBuffer table_scratch(cb_table_scratch);

    // The valid tile-row count arrives in the semaphore the reader of core 0 multicasts (arg 6).
    volatile tt_l1_ptr uint32_t* rows_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(6)));
    noc_semaphore_wait_min(rows_sem, 1);
    const CountAwareRange range = count_aware_block_range_rows(*rows_sem - 1, units_per_row, core_id, num_cores);

    const auto input = TensorAccessor(input_args, input_addr);
    CircularBuffer cb_scale_obj(cb_scale);
    CircularBuffer scale_scratch_obj(cb_scale_scratch);
    scale_scratch_obj.reserve_back(1);
    const uint32_t scratch = scale_scratch_obj.get_write_ptr();
    auto produce_scales = [&](uint32_t unit) {
        const uint32_t row0 = (unit / units_per_row) * 32;
        const uint32_t tile0 = (unit % units_per_row) * G;
        const uint32_t seg = tile0 / seg_tiles;
        const uint32_t scale_first = seg * seg_bytes + seg_tiles * 32 + (tile0 - seg * seg_tiles) / TILES_PER_SCALE * 4;
        const uint32_t scale_window = scale_first & ~(DRAM_ALIGN - 1);
        const uint32_t skip_words = (scale_first - scale_window) / 4;
        const uint32_t read_bytes =
            (scale_first - scale_window + G / TILES_PER_SCALE * 4 + DRAM_ALIGN - 1) & ~(DRAM_ALIGN - 1);
        for (uint32_t r = 0; r < 32; ++r) {
            noc.async_read(
                input,
                use<CircularBuffer::AddrSelector::WRITE_PTR>(scale_scratch_obj),
                read_bytes,
                {.page_id = row0 + r, .offset_bytes = scale_window},
                {.offset_bytes = r * scale_slot_bytes});
        }
        cb_scale_obj.reserve_back(G);
        noc.async_read_barrier();
        // Plain (cached) loads after one invalidate, a row's G scales gathered before their strided stores: the
        // volatile load-shift-store per element cost ~10 us per unit (G = 28).
        invalidate_l1_cache();
        volatile tt_l1_ptr uint16_t* tiles =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_scale_obj.get_write_ptr());
        for (uint32_t r = 0; r < 32; ++r) {
            const uint32_t col0 = (r < 16) ? r * 16 : 512 + (r - 16) * 16;
            const uint32_t* row_scales = reinterpret_cast<const uint32_t*>(scratch + r * scale_slot_bytes) + skip_words;
            // One scale per TILES_PER_SCALE tiles (scale_block_w / 32).
            uint32_t v[G / TILES_PER_SCALE];
#pragma GCC unroll 32
            for (uint32_t j = 0; j < G / TILES_PER_SCALE; ++j) {
                v[j] = row_scales[j];
            }
            volatile tt_l1_ptr uint16_t* dst = tiles + col0;
#pragma GCC unroll 32
            for (uint32_t j = 0; j < G; ++j) {
                dst[j * 1024] = v[j / TILES_PER_SCALE] >> 16;  // pow2 fp32 -> bf16: the high half, exact
            }
        }
        cb_scale_obj.push_back(G);
    };

    const uint32_t tile_bytes = cb_out_obj.get_tile_size();
    uint32_t tile_row = range.first_block / units_per_row;
    uint32_t unit_col = range.first_block % units_per_row;
    if (range.num_blocks > 0) {
        produce_scales(range.first_block);
    }
    for (uint32_t u = 0; u < range.num_blocks; ++u) {
        if (u + 1 < range.num_blocks) {
            produce_scales(range.first_block + u + 1);  // one unit ahead, so compute never waits on scales
        }
        cb_out_obj.wait_front(G);
        const uint32_t first_tile = tile_row * Wt + unit_col * G;
        for (uint32_t j = 0; j < G; ++j) {
            noc.async_write(
                cb_out_obj, output, tile_bytes, {.offset_bytes = j * tile_bytes}, {.page_id = first_tile + j});
        }
        noc.async_writes_flushed();
        cb_out_obj.pop_front(G);
        if (++unit_col == units_per_row) {
            unit_col = 0;
            ++tile_row;
        }
    }
    noc.async_write_barrier();
}
