// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// per_token_cast_back, tile output (token-count-aware, 1x32 scales in the row tail): dequantize the valid prefix
// of a MoE dispatch buffer of packed rows [H e4m3 | H/32 fp32] straight into TILE output, so the routed expert
// takes its tiled-x path and nothing is untilized here.
//
// Work unit = one tile-row (32 tokens) x G tiles (G * 32 columns). Per unit the reader stacks the 32 rows' e4m3
// slice in c_in (G tiles wide, row-major) and builds G bf16 scale tiles whose column 0 row r is token r's scale of
// that tile's 1x32 block (pow2 fp32 -> bf16 is its high half, exact). The compute tilizes and column-broadcast
// multiplies. Rows past an expert's count inside a tile-row are dequantized garbage, as on the row-major path; the
// expert never emits them.

#ifndef TILES_PER_SCALE
#define TILES_PER_SCALE 1
#endif
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_to_fp8/device/kernels/dataflow/count_aware_range.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scale = get_compile_time_arg_val(1);
    constexpr uint32_t cb_loop_count = get_compile_time_arg_val(2);
    constexpr uint32_t cb_region_scratch = get_compile_time_arg_val(3);
    constexpr uint32_t cb_counts_scratch = get_compile_time_arg_val(4);
    constexpr uint32_t cb_table_scratch = get_compile_time_arg_val(5);
    constexpr uint32_t cb_scale_scratch = get_compile_time_arg_val(6);
    constexpr uint32_t G = get_compile_time_arg_val(7);  // tiles per unit
    constexpr uint32_t Wt = get_compile_time_arg_val(8);
    constexpr uint32_t H = get_compile_time_arg_val(9);  // e4m3 bytes per row (all segments)
    constexpr uint32_t experts_per_chip = get_compile_time_arg_val(10);
    constexpr uint32_t max_rows = get_compile_time_arg_val(11);
    constexpr uint32_t num_cores = get_compile_time_arg_val(12);
    constexpr uint32_t scale_slot_bytes = get_compile_time_arg_val(13);
    // Row = segments of [seg_tiles * 32 e4m3 | seg_tiles fp32 scales (| pad)], seg_bytes each (one segment: the
    // plain [H e4m3 | H/32 scales] row). A unit never straddles a segment.
    constexpr uint32_t seg_bytes = get_compile_time_arg_val(14);
    constexpr uint32_t seg_tiles = get_compile_time_arg_val(15);
    constexpr auto input_args = TensorAccessorArgs<16>();
    constexpr auto region_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto counts_args = TensorAccessorArgs<region_args.next_compile_time_args_offset()>();
    constexpr auto table_args = TensorAccessorArgs<counts_args.next_compile_time_args_offset()>();

    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t TILE_HW = 1024;
    constexpr uint32_t unit_row_bytes = G * 32;  // e4m3
    constexpr uint32_t units_per_row = Wt / G;
    constexpr uint32_t DRAM_ALIGN = 64;  // Blackhole DRAM reads: source and destination congruent mod 64 B

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t region_addr = get_arg_val<uint32_t>(2);
    const uint32_t counts_addr = get_arg_val<uint32_t>(3);
    const uint32_t table_addr = get_arg_val<uint32_t>(4);
    const uint32_t core_id = get_arg_val<uint32_t>(5);

    Noc noc;
    const auto input = TensorAccessor(input_args, input_addr);
    const auto region = TensorAccessor(region_args, region_addr);
    const auto counts = TensorAccessor(counts_args, counts_addr);
    const auto table = TensorAccessor(table_args, table_addr);
    CircularBuffer cb_in_obj(cb_in);
    CircularBuffer cb_scale_obj(cb_scale);
    CircularBuffer cb_loop_count_obj(cb_loop_count);
    CircularBuffer region_scratch(cb_region_scratch);
    CircularBuffer counts_scratch(cb_counts_scratch);
    CircularBuffer table_scratch(cb_table_scratch);
    CircularBuffer scale_scratch_obj(cb_scale_scratch);

    // Core 0 reads the routing tables and multicasts the valid tile-row count (args 6..11: semaphore id, NoC 0
    // multicast rectangle, destination count) into every core's semaphore; the writer reads it from there too.
    const uint32_t rows_sem_addr = get_semaphore(get_arg_val<uint32_t>(6));
    const uint32_t valid_tile_rows = count_aware_valid_tile_rows_mcast(
        noc,
        region,
        counts,
        table,
        region_scratch,
        counts_scratch,
        table_scratch,
        experts_per_chip,
        max_rows,
        TILE_H,
        core_id == 0,
        rows_sem_addr,
        // NoC 1 multicasts take the rectangle corners in reverse order.
        noc_index == 0 ? get_noc_multicast_addr(
                             get_arg_val<uint32_t>(7),
                             get_arg_val<uint32_t>(8),
                             get_arg_val<uint32_t>(9),
                             get_arg_val<uint32_t>(10),
                             rows_sem_addr)
                       : get_noc_multicast_addr(
                             get_arg_val<uint32_t>(9),
                             get_arg_val<uint32_t>(10),
                             get_arg_val<uint32_t>(7),
                             get_arg_val<uint32_t>(8),
                             rows_sem_addr),
        get_arg_val<uint32_t>(11));
    const CountAwareRange range = count_aware_block_range_rows(valid_tile_rows, units_per_row, core_id, num_cores);

    cb_loop_count_obj.reserve_back(1);
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_loop_count_obj.get_write_ptr())[0] = range.num_blocks;
    cb_loop_count_obj.push_back(1);

    scale_scratch_obj.reserve_back(1);
    const uint32_t scratch = scale_scratch_obj.get_write_ptr();

    uint32_t tile_row = range.first_block / units_per_row;
    uint32_t unit_col = range.first_block % units_per_row;
    for (uint32_t u = 0; u < range.num_blocks; ++u) {
        const uint32_t row0 = tile_row * TILE_H;
        const uint32_t tile0 = unit_col * G;
        const uint32_t seg = tile0 / seg_tiles;
        const uint32_t in_seg = tile0 - seg * seg_tiles;
        const uint32_t col_bytes = seg * seg_bytes + in_seg * 32;
        // The unit's G scales follow its segment's e4m3 bytes: read the 64-B aligned window holding them.
        const uint32_t scale_first = seg * seg_bytes + seg_tiles * 32 + in_seg / TILES_PER_SCALE * 4;
        const uint32_t scale_window = scale_first & ~(DRAM_ALIGN - 1);
        const uint32_t scale_skip_words = (scale_first - scale_window) / 4;
        const uint32_t scale_read_bytes =
            (scale_first - scale_window + G / TILES_PER_SCALE * 4 + DRAM_ALIGN - 1) & ~(DRAM_ALIGN - 1);

        cb_in_obj.reserve_back(G);
        for (uint32_t r = 0; r < TILE_H; ++r) {
            noc.async_read(
                input,
                cb_in_obj,
                unit_row_bytes,
                {.page_id = row0 + r, .offset_bytes = col_bytes},
                {.offset_bytes = r * unit_row_bytes});
#ifndef SCALES_ON_WRITER
            noc.async_read(
                input,
                use<CircularBuffer::AddrSelector::WRITE_PTR>(scale_scratch_obj),
                scale_read_bytes,
                {.page_id = row0 + r, .offset_bytes = scale_window},
                {.offset_bytes = r * scale_slot_bytes});
#endif
        }
        noc.async_read_barrier();
        cb_in_obj.push_back(G);
#ifndef SCALES_ON_WRITER

        cb_scale_obj.reserve_back(G);
        volatile tt_l1_ptr uint16_t* tiles =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_scale_obj.get_write_ptr());
        for (uint32_t r = 0; r < TILE_H; ++r) {
            // column 0 of row r: face 0 for rows 0-15, face 2 for rows 16-31 (16 elements per face row)
            const uint32_t col0 = (r < 16) ? r * 16 : 512 + (r - 16) * 16;
            const uint32_t* row_scales =
                reinterpret_cast<const uint32_t*>(scratch + r * scale_slot_bytes) + scale_skip_words;
            uint32_t v[G / TILES_PER_SCALE];
#pragma GCC unroll 32
            for (uint32_t j = 0; j < G / TILES_PER_SCALE; ++j) {
                v[j] = row_scales[j];
            }
            volatile tt_l1_ptr uint16_t* dst = tiles + col0;
#pragma GCC unroll 32
            for (uint32_t j = 0; j < G; ++j) {
                dst[j * TILE_HW] = v[j / TILES_PER_SCALE] >> 16;  // pow2 fp32 -> bf16: the high half, exact
            }
        }
        cb_scale_obj.push_back(G);
#else
        (void)scale_read_bytes;
        (void)scale_skip_words;
#endif

        if (++unit_col == units_per_row) {
            unit_col = 0;
            ++tile_row;
        }
    }
}
