// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer for per_token_cast_to_fp8. Outputs are ROW_MAJOR for both input layouts. Mirror of the reader:
// the compute produces, per block, tile_h output_e4m3 block-rows plus one fp32 scale tile whose column 0
// holds the tile_h per-block scales (pulled out by extract_first_column).
//
// ROW_MAJOR input: the writer walks the same flat block stream as the reader with a (current_row,
// current_col) cursor and: writes each bank-contiguous run of output_e4m3 blocks back to its row at the
// current block column offset (one NoC async write per run, mirroring the reader's contiguous reads);
// accumulates per-block scales into a persistent per-token scale-row scratch and flushes the full row
// (page-aligned source -> aligned DRAM page) when the token's last block is emitted. A block may straddle
// tokens and a token may straddle blocks, so the scratch and cursor persist across blocks.
//
// INPUT_TILE_LAYOUT: each block maps to (row-tile, column-block); e4m3 rows go to their folded RM output
// rows (padding rows skipped) and each row-tile's scales are staged and flushed as one contiguous run per row.

#include <cstdint>

// Chunks (128 columns each) per work unit on the count-aware TILE FAST_1X32 path; 1 everywhere else.
#ifndef Q_UNIT
#define Q_UNIT 1
#endif

#include "api/dataflow/dataflow_api.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#ifdef TOKEN_COUNT_AWARE
#include "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_to_fp8/device/kernels/dataflow/count_aware_range.hpp"
#endif

#ifdef FAST_1X32
// FAST_1X32: the compute hands over bf16 tiles of 1/scale (exact powers of two); column 0 -> the fp32 scale
// bits, (254 << 23) - (1/scale bits), one value per block row.
template <uint32_t face_h, uint32_t face_w, uint32_t FACE_ROWS, uint32_t FACE_ROW_STRIDE>
static inline void extract_first_column(volatile tt_l1_ptr uint32_t* tile_words, uint32_t* out) {
    volatile tt_l1_ptr uint16_t* tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile_words);
    uint32_t s = 0;
    uint32_t face_base = 0;
    for (uint32_t fr = 0; fr < FACE_ROWS; ++fr) {
        uint32_t col0_idx = face_base;
        for (uint32_t r = 0; r < face_h; ++r) {
            out[s] = (254u << 23) - (static_cast<uint32_t>(tile[col0_idx]) << 16);
            col0_idx += face_w;
            ++s;
        }
        face_base += FACE_ROW_STRIDE;
    }
}
#else
// Extract column 0 of one fp32 tile into out[0..tile_h) (one value per block row); face-aware walk.
template <uint32_t face_h, uint32_t face_w, uint32_t FACE_ROWS, uint32_t FACE_ROW_STRIDE>
static inline void extract_first_column(volatile tt_l1_ptr uint32_t* tile, uint32_t* out) {
    uint32_t s = 0;
    uint32_t face_base = 0;  // = face_row * FACE_ROW_STRIDE
    for (uint32_t fr = 0; fr < FACE_ROWS; ++fr) {
        uint32_t col0_idx = face_base;  // in-face row 0, col 0
        for (uint32_t r = 0; r < face_h; ++r) {
            out[s] = tile[col0_idx];
            col0_idx += face_w;
            ++s;
        }
        face_base += FACE_ROW_STRIDE;
    }
}
#endif

void kernel_main() {
#ifdef INPUT_TILE_LAYOUT
    uint32_t output_e4m3_addr = get_arg_val<uint32_t>(0);
    uint32_t scale_addr = get_arg_val<uint32_t>(1);
#ifndef TOKEN_COUNT_AWARE
    uint32_t block_offset = get_arg_val<uint32_t>(2);         // first global block of this core
    uint32_t num_blocks = get_arg_val<uint32_t>(3);           // blocks owned by this core
#endif
    uint32_t rows_per_batch = get_arg_val<uint32_t>(4);       // R = logical rows per batch (dim -2)
    uint32_t row_tiles_per_batch = get_arg_val<uint32_t>(5);  // ceil(R / tile_h)
#else
    uint32_t output_e4m3_addr = get_arg_val<uint32_t>(0);
    uint32_t scale_addr = get_arg_val<uint32_t>(1);
    uint32_t num_blocks = get_arg_val<uint32_t>(2);
    uint32_t start_row = get_arg_val<uint32_t>(3);  // absolute first row of this core's stream
    uint32_t num_rows = get_arg_val<uint32_t>(4);   // rows owned by this core
    uint32_t width = get_arg_val<uint32_t>(5);      // H (elements per row)
#endif

    constexpr uint32_t cb_output_e4m3 = get_compile_time_arg_val(0);
    constexpr uint32_t output_e4m3_block_bytes = get_compile_time_arg_val(1);  // 128 (1 byte/elem)
    constexpr uint32_t cb_scale_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scale_scratch = get_compile_time_arg_val(3);
    constexpr uint32_t scale_blocks_per_row = get_compile_time_arg_val(4);      // H / scale_block_w (scales per row)
    constexpr uint32_t scale_aligned_page_bytes = get_compile_time_arg_val(5);  // per-row scratch footprint
    constexpr uint32_t tile_h = get_compile_time_arg_val(6);
    constexpr uint32_t tile_w = get_compile_time_arg_val(7);
    constexpr uint32_t face_h = get_compile_time_arg_val(8);
    constexpr uint32_t face_w = get_compile_time_arg_val(9);
    constexpr uint32_t chunks_per_row = get_compile_time_arg_val(10);      // H / 128 (I/O chunks per row)
    constexpr uint32_t chunk_scale_blocks = get_compile_time_arg_val(11);  // scales per chunk row (128/scale_block_w)
    // Packed output: each e4m3 row is [data_row_bytes e4m3 | scale_blocks_per_row fp32]; the scales go into
    // the same row (scale tensor unused). Used for the scaled fp8 combine wire.
    constexpr bool pack_scales_in_row = get_compile_time_arg_val(12) != 0;
    constexpr uint32_t data_row_bytes = get_compile_time_arg_val(13);
    constexpr uint32_t block_w = 128;
    constexpr uint32_t tiles_per_block = block_w / tile_w;
    constexpr uint32_t scale_bytes = sizeof(uint32_t);                // fp32 scale = one 32-bit word = 4 bytes
    constexpr uint32_t face_elems = face_h * face_w;                  // fp32 per face
    constexpr uint32_t faces_per_row = tile_w / face_w;               // face columns per tile
    constexpr uint32_t FACE_ROWS = tile_h / face_h;                   // face rows per tile
    constexpr uint32_t FACE_ROW_STRIDE = faces_per_row * face_elems;  // fp32 stride per face row

    constexpr auto output_e4m3_accessor_args = TensorAccessorArgs<14>();
    constexpr auto scale_args = TensorAccessorArgs<output_e4m3_accessor_args.next_compile_time_args_offset()>();
    const auto output_e4m3 = TensorAccessor(output_e4m3_accessor_args, output_e4m3_addr);
    const auto scale = TensorAccessor(scale_args, scale_addr);
    Noc noc;
#ifdef TOKEN_COUNT_AWARE
    constexpr uint32_t ca_base = scale_args.next_compile_time_args_offset();
    constexpr uint32_t cb_region_scratch = get_compile_time_arg_val(ca_base + 0);
    constexpr uint32_t cb_counts_scratch = get_compile_time_arg_val(ca_base + 1);
    constexpr uint32_t cb_table_scratch = get_compile_time_arg_val(ca_base + 2);
    constexpr uint32_t ca_num_cores = get_compile_time_arg_val(ca_base + 3);
    constexpr uint32_t ca_experts_per_chip = get_compile_time_arg_val(ca_base + 4);
    constexpr uint32_t ca_max_rows = get_compile_time_arg_val(ca_base + 5);
    constexpr auto region_args = TensorAccessorArgs<ca_base + 6>();
    constexpr auto counts_args = TensorAccessorArgs<region_args.next_compile_time_args_offset()>();
    constexpr auto table_args = TensorAccessorArgs<counts_args.next_compile_time_args_offset()>();
    const auto region = TensorAccessor(region_args, get_arg_val<uint32_t>(6));
    const auto counts = TensorAccessor(counts_args, get_arg_val<uint32_t>(7));
    const auto table = TensorAccessor(table_args, get_arg_val<uint32_t>(8));
    const uint32_t core_id = get_arg_val<uint32_t>(9);
    CircularBuffer cb_region_scratch_obj(cb_region_scratch);
    CircularBuffer cb_counts_scratch_obj(cb_counts_scratch);
    CircularBuffer cb_table_scratch_obj(cb_table_scratch);
    // The valid tile-row count arrives in the semaphore the reader of core 0 multicasts (arg 10).
    volatile tt_l1_ptr uint32_t* rows_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(10)));
    noc_semaphore_wait_min(rows_sem, 1);
    const CountAwareRange range = count_aware_block_range_frac_rows(
        *rows_sem - 1,
        chunks_per_row / Q_UNIT,   // units per row
        get_arg_val<uint32_t>(2),  // host-weighted share [start, end) in 1/65536 units (matches the reader)
        get_arg_val<uint32_t>(3));
    (void)core_id;
    (void)ca_num_cores;
    const uint32_t block_offset = range.first_block;
    const uint32_t num_blocks = range.num_blocks;
#endif
    CircularBuffer cb_output_e4m3_obj(cb_output_e4m3);
    CircularBuffer cb_scale_tiles_obj(cb_scale_tiles);
    CircularBuffer cb_scale_scratch_obj(cb_scale_scratch);

    // Persistent scale scratch (writer-private), reused across blocks / row-tiles.
    CoreLocalMem<volatile uint32_t> tok(cb_scale_scratch_obj.get_write_ptr());

    uint32_t block_scales[chunk_scale_blocks][tile_h];  // [scale block in chunk][row]

#ifdef INPUT_TILE_LAYOUT
    constexpr uint32_t scale_page_words = scale_aligned_page_bytes / scale_bytes;
    constexpr uint32_t NO_ROW_TILE_OPEN = 0xFFFFFFFFu;  // sentinel: no row-tile open yet

    // Flush the accumulated scale rows for a row-tile over its owned column range [first_col, last_col].
    auto flush_scale = [&](uint32_t row_base, uint32_t n_rows, uint32_t first_col, uint32_t last_col) {
        // first_col/last_col are chunk columns; each chunk owns chunk_scale_blocks scale columns.
        const uint32_t first_scale = first_col * chunk_scale_blocks;
        const uint32_t run_bytes = (last_col - first_col + 1) * chunk_scale_blocks * scale_bytes;
        for (uint32_t row_in_tile = 0; row_in_tile < n_rows; ++row_in_tile) {
            if constexpr (pack_scales_in_row) {
                noc.async_write(
                    use<CircularBuffer::AddrSelector::WRITE_PTR>(cb_scale_scratch_obj),
                    output_e4m3,
                    run_bytes,
                    {.offset_bytes = row_in_tile * scale_aligned_page_bytes + first_scale * scale_bytes},
                    {.page_id = row_base + row_in_tile, .offset_bytes = data_row_bytes + first_scale * scale_bytes});
            } else {
                noc.async_write(
                    use<CircularBuffer::AddrSelector::WRITE_PTR>(cb_scale_scratch_obj),
                    scale,
                    run_bytes,
                    {.offset_bytes = row_in_tile * scale_aligned_page_bytes + first_scale * scale_bytes},
                    {.page_id = row_base + row_in_tile, .offset_bytes = first_scale * scale_bytes});
            }
        }
        noc.async_write_barrier();  // scratch is reused by the next row-tile
    };

    const uint32_t global_block_end = block_offset + num_blocks;
    uint32_t current_row_tile = NO_ROW_TILE_OPEN;
    uint32_t output_row_base = 0, real_rows = 0, owned_first_col = 0, owned_last_col = 0;

    // A unit is Q_UNIT consecutive chunks of one tile-row: one write of Q_UNIT * 128 bytes per row.
    constexpr uint32_t units_per_row = chunks_per_row / Q_UNIT;
    constexpr uint32_t unit_scale_tiles = Q_UNIT * chunk_scale_blocks;
    constexpr uint32_t unit_row_bytes = Q_UNIT * output_e4m3_block_bytes;
#ifdef FAST_1X32
    // NoC address of each output row of the open row-tile, resolved once per row-tile (not once per unit).
    uint64_t row_noc_addr[tile_h];
    const uint32_t scale_tile_elems = get_tile_size(cb_scale_tiles) / sizeof(uint16_t);
#ifndef Q_WGROUP
#define Q_WGROUP 1
#endif
    constexpr uint32_t group_tiles = Q_WGROUP * Q_UNIT * tiles_per_block;
    constexpr uint32_t group_row_bytes = Q_WGROUP * unit_row_bytes;
    uint32_t group_pos = 0, group_col_byte = 0;
#endif
    for (uint32_t global_block = block_offset; global_block < global_block_end; ++global_block) {
        const uint32_t row_tile = global_block / units_per_row;
        const uint32_t col_block = (global_block % units_per_row) * Q_UNIT;  // first chunk of the unit

        if (row_tile != current_row_tile) {
            if (current_row_tile != NO_ROW_TILE_OPEN) {
                flush_scale(output_row_base, real_rows, owned_first_col, owned_last_col);
            }
            // Map the row-tile to its batch and folded RM output rows (tile-padding rows skipped).
            const uint32_t batch = row_tile / row_tiles_per_batch;
            const uint32_t local_row_base = (row_tile % row_tiles_per_batch) * tile_h;
            output_row_base = batch * rows_per_batch + local_row_base;
            const uint32_t rows_left = rows_per_batch - local_row_base;
            real_rows = rows_left < tile_h ? rows_left : tile_h;
            current_row_tile = row_tile;
            owned_first_col = col_block;
#ifdef FAST_1X32
            for (uint32_t r = 0; r < real_rows; ++r) {
                row_noc_addr[r] = output_e4m3.get_noc_addr(output_row_base + r);
            }
#endif
        }
        owned_last_col = col_block + Q_UNIT - 1;

#ifdef FAST_1X32
        cb_scale_tiles_obj.wait_front(unit_scale_tiles);
#else
        cb_output_e4m3_obj.wait_front(Q_UNIT * tiles_per_block);
        cb_scale_tiles_obj.wait_front(unit_scale_tiles);
#endif
        const uint32_t col_byte_offset = col_block * block_w;
#ifdef FAST_1X32
        // Stage this unit's scales, then write the e4m3 block once the write group (Q_WGROUP units) is complete. The
        // 1/scale tiles are bf16 powers of two: each column-0 element becomes its fp32 scale bits in one pass,
        // straight into the row-tile's scale staging (no per-tile temporary).
        {
            // Plain (cached) loads after one invalidate, gathered per row before the stores, so the 8 L1 reads of a
            // row issue back to back instead of one volatile load-convert-store at a time (2.9 -> 1.3 us per unit).
            invalidate_l1_cache();
            const uint16_t* inv = reinterpret_cast<const uint16_t*>(cb_scale_tiles_obj.get_read_ptr());
            volatile tt_l1_ptr uint32_t* stage =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_scale_scratch_obj.get_write_ptr()) +
                col_block * chunk_scale_blocks;
            for (uint32_t row_in_tile = 0; row_in_tile < real_rows; ++row_in_tile) {
                const uint16_t* src = inv + (row_in_tile / face_h) * FACE_ROW_STRIDE + (row_in_tile % face_h) * face_w;
                uint32_t v[unit_scale_tiles];
#pragma GCC unroll 16
                for (uint32_t st = 0; st < unit_scale_tiles; ++st) {
                    v[st] = src[st * scale_tile_elems];
                }
                volatile tt_l1_ptr uint32_t* dst = stage + row_in_tile * scale_page_words;
#pragma GCC unroll 16
                for (uint32_t st = 0; st < unit_scale_tiles; ++st) {
                    dst[st] = (254u << 23) - (v[st] << 16);
                }
            }
        }
        cb_scale_tiles_obj.pop_front(unit_scale_tiles);
        // The group's e4m3 block arrives once its last unit is packed: one run of the group's units per row.
        if (group_pos == 0) {
            group_col_byte = col_byte_offset;
        }
        if (group_pos + 1 == Q_WGROUP || (global_block + 1) % units_per_row == 0 ||
            global_block + 1 == global_block_end) {
            const uint32_t run_bytes = (group_pos + 1) * unit_row_bytes;
            cb_output_e4m3_obj.wait_front(group_tiles);
            const uint32_t e4m3_l1 = cb_output_e4m3_obj.get_read_ptr();
            for (uint32_t row_in_tile = 0; row_in_tile < real_rows; ++row_in_tile) {
                noc_async_write(
                    e4m3_l1 + row_in_tile * group_row_bytes, row_noc_addr[row_in_tile] + group_col_byte, run_bytes);
            }
            noc.async_writes_flushed();  // the block only needs the writes to have left L1
            cb_output_e4m3_obj.pop_front(group_tiles);
            group_pos = 0;
        } else {
            ++group_pos;
        }
#else
        for (uint32_t st = 0; st < unit_scale_tiles; ++st) {
            extract_first_column<face_h, face_w, FACE_ROWS, FACE_ROW_STRIDE>(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    cb_scale_tiles_obj.get_read_ptr() + st * get_tile_size(cb_scale_tiles)),
                block_scales[0]);
            for (uint32_t row_in_tile = 0; row_in_tile < real_rows; ++row_in_tile) {
                tok[row_in_tile * scale_page_words + col_block * chunk_scale_blocks + st] =
                    block_scales[0][row_in_tile];
            }
        }

        for (uint32_t row_in_tile = 0; row_in_tile < real_rows; ++row_in_tile) {
            noc.async_write(
                cb_output_e4m3_obj,
                output_e4m3,
                unit_row_bytes,
                {.offset_bytes = row_in_tile * unit_row_bytes},
                {.page_id = output_row_base + row_in_tile, .offset_bytes = col_byte_offset});
        }
#endif
        // The CB page only needs the e4m3 writes to have left L1; flush_scale's barrier (and the final one) waits for
        // their completion.
#ifndef FAST_1X32
        noc.async_writes_flushed();
        cb_output_e4m3_obj.pop_front(Q_UNIT * tiles_per_block);
        cb_scale_tiles_obj.pop_front(unit_scale_tiles);
#endif
    }

    if (current_row_tile != NO_ROW_TILE_OPEN) {
        flush_scale(output_row_base, real_rows, owned_first_col, owned_last_col);
    }
#else
    constexpr uint32_t scale_row_bytes = scale_blocks_per_row * scale_bytes;

    const uint32_t blocks_per_row = width / block_w;  // H / 128 chunks
    const uint32_t total_blocks = num_rows * blocks_per_row;
    const uint32_t end_row = start_row + num_rows;
    uint32_t current_row = start_row;
    uint32_t current_col = 0;  // element offset within the current row

    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        const uint32_t base = blk * tile_h;
        const uint32_t remaining = total_blocks - base;
        const uint32_t real_in_block = remaining < tile_h ? remaining : tile_h;

        cb_output_e4m3_obj.wait_front(tiles_per_block);
        cb_scale_tiles_obj.wait_front(chunk_scale_blocks);
        for (uint32_t sb = 0; sb < chunk_scale_blocks; ++sb) {
            extract_first_column<face_h, face_w, FACE_ROWS, FACE_ROW_STRIDE>(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    cb_scale_tiles_obj.get_read_ptr() + sb * get_tile_size(cb_scale_tiles)),
                block_scales[sb]);
        }

        uint32_t br = 0;  // block-local row
        while (br < real_in_block && current_row < end_row) {
            uint32_t block_idx_in_row = current_col / block_w;  // block index within the row
            uint32_t blocks_left_in_row = blocks_per_row - block_idx_in_row;
            uint32_t blocks_left_in_batch = real_in_block - br;
            uint32_t run = blocks_left_in_row < blocks_left_in_batch ? blocks_left_in_row : blocks_left_in_batch;

            // output_e4m3: write `run` contiguous 128-element blocks back to (current_row, current_col).
            noc.async_write(
                cb_output_e4m3_obj,
                output_e4m3,
                run * output_e4m3_block_bytes,
                {.offset_bytes = br * output_e4m3_block_bytes},
                {.page_id = current_row, .offset_bytes = current_col});
            // scale: stage this run's per-block scales into the token's scratch row.
            for (uint32_t g = 0; g < run; ++g) {
                for (uint32_t sb = 0; sb < chunk_scale_blocks; ++sb) {
                    tok[(block_idx_in_row + g) * chunk_scale_blocks + sb] = block_scales[sb][br + g];
                }
            }
            br += run;
            current_col += run * block_w;
            if (current_col >= width) {  // token complete -> flush its full scale row (aligned)
                noc.async_write(
                    use<CircularBuffer::AddrSelector::WRITE_PTR>(cb_scale_scratch_obj),
                    scale,
                    scale_row_bytes,
                    {.offset_bytes = 0},
                    {.page_id = current_row});
                noc.async_write_barrier();  // scratch is reused by the next token
                current_col = 0;
                ++current_row;
            }
        }
        noc.async_write_barrier();  // drain this block's output_e4m3 writes before the CB page is reused
        cb_output_e4m3_obj.pop_front(tiles_per_block);
        cb_scale_tiles_obj.pop_front(chunk_scale_blocks);
    }
#endif
}
