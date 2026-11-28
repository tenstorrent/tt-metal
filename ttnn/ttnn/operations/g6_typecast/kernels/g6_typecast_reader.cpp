// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_typecast reader (NCRISC / NoC0) — block operation `load_block` (op_design.md, Block schedule).
// Streams this core's contiguous range of input tiles [start_tile, start_tile + core_tiles) into
// cb_input_tiles in blocks of `block_tiles`: block_tiles page reads in flight, ONE read barrier and
// ONE reserve/push handshake per block. A ragged last block reserves/pushes its actual extent
// (compute and writer synchronize the same per-block extent).
// Perf 1: each tile moves only its LIVE faces (g6_typecast_tile_faces.hpp): faces 1/3 of a last-column tile
// when W % 32 is 1..16, faces 2/3 of a last-row tile when H % 32 is 1..16 are pure padding and are never
// read (an unread input face only feeds an output face the writer never writes; Q2185 35.1 -> 16.9 us).
// Reads are single-packet (a tile <= 4 KB < NOC_MAX_BURST_SIZE; ~30 ns per core on 1-tile cells).
// No dataflow helper owns an interleaved page-range stream, so this is the raw TensorAccessor loop.
//
// rm_mode (regime R2 `rm_pseudo_tile_stream`): the core owns rows [start_row, start_row + core_rows) of an
// interleaved ROW_MAJOR tensor. Per block of `rows_per_block` rows: ONE NoC read per bank run (rows r, r+nb,
// r+2nb, ... sit adjacent in one bank at the aligned-page stride, nb = accessor.contiguous_page_stride())
// into the 64B-aligned staging zone, ONE read barrier, then a RISC-V compaction of the rows into the dense
// pseudo-tile block of cb_input_tiles (ceil(rows * W / 1024) pages).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "g6_typecast_zones.hpp"
#include "g6_typecast_rm_rows.hpp"
#include "g6_typecast_tile_faces.hpp"

void kernel_main() {
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t in_page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t mask_pad_columns = get_compile_time_arg_val(3);
    constexpr uint32_t tensor_tile_cols = get_compile_time_arg_val(4);
    constexpr uint32_t w_tail = get_compile_time_arg_val(5);
    constexpr uint32_t rm_mode = get_compile_time_arg_val(6);
    constexpr uint32_t cb_input_staging = get_compile_time_arg_val(7);
    constexpr uint32_t rows_per_block = get_compile_time_arg_val(8);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t values_per_row = get_compile_time_arg_val(10);
    constexpr uint32_t staging_align = get_compile_time_arg_val(11);
    constexpr uint32_t staging_depth = get_compile_time_arg_val(12);
    // TILE only: tile rows per batch slice, H % 32, and the tile's byte geometry (header, face) for face skip.
    constexpr uint32_t tensor_tile_rows = get_compile_time_arg_val(13);
    constexpr uint32_t h_tail = get_compile_time_arg_val(14);
    constexpr uint32_t tile_hdr_bytes = get_compile_time_arg_val(15);
    constexpr uint32_t face_bytes = get_compile_time_arg_val(16);
    constexpr auto input_args = TensorAccessorArgs<17>();

    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(0);
    const uint32_t core_tiles = get_arg_val<uint32_t>(1);

    // in_page_bytes is the aligned page size (TILE: the tile; RM: the aligned row slot).
    const auto input_accessor = TensorAccessor(input_args, input_addr, in_page_bytes);

    if constexpr (rm_mode) {
        // start_tile / core_tiles carry start_row / core_rows here.
        // staging_depth >= 2: the bank-run reads of block b+1 are issued into the next staging slot BEFORE
        // block b is compacted, so the DMA overlaps the RISC-V compaction. Only one block is ever in flight,
        // so the plain read barrier retires exactly block b (no transaction ids needed).
        constexpr uint32_t pseudo_tile_values = 1024;
        const uint32_t nb = input_accessor.contiguous_page_stride();
        const uint32_t staging0 = (get_write_ptr(cb_input_staging) + staging_align - 1) & ~(staging_align - 1);
        const uint32_t staging_bytes = rows_per_block * in_page_bytes;
        const uint32_t end_row = start_tile + core_tiles;
        auto block_rows = [&](uint32_t blk) {
            return (end_row - blk < rows_per_block) ? (end_row - blk) : rows_per_block;
        };
        // One read per bank run: rows blk + j + k*nb, k = 0..cnt-1, contiguous in their bank.
        auto issue_block = [&](uint32_t blk, uint32_t staging) {
            const uint32_t rows = block_rows(blk);
            const uint32_t runs = rows < nb ? rows : nb;
            uint32_t slot = 0;
            for (uint32_t j = 0; j < runs; ++j) {
                const uint32_t cnt = (rows - j - 1) / nb + 1;
                noc_async_read(
                    input_accessor.get_noc_addr(blk + j), staging + slot * in_page_bytes, cnt * in_page_bytes);
                slot += cnt;
            }
        };
        uint32_t stage = 0;
        if (start_tile < end_row) {
            issue_block(start_tile, staging0);
        }
        for (uint32_t blk = start_tile; blk < end_row; blk += rows_per_block) {
            G6_ZONE("READ_BLOCK");
            const uint32_t rows = block_rows(blk);
            const uint32_t block_pages = (rows * values_per_row + pseudo_tile_values - 1) / pseudo_tile_values;
            const uint32_t staging = staging0 + stage * staging_bytes;
            const uint32_t next_stage = (stage + 1 == staging_depth) ? 0 : stage + 1;
            const uint32_t next_blk = blk + rows_per_block;
            noc_async_read_barrier();  // block blk landed (the only block in flight)
            if constexpr (staging_depth > 1) {
                if (next_blk < end_row) {
                    issue_block(next_blk, staging0 + next_stage * staging_bytes);
                }
            }
            {
                G6_ZONE("READ_WAIT_SPACE");
                cb_reserve_back(cb_input_tiles, block_pages);
            }
            g6_rm::compact_block<row_bytes>(staging, in_page_bytes, nb, rows, get_write_ptr(cb_input_tiles));
            cb_push_back(cb_input_tiles, block_pages);
            if constexpr (staging_depth == 1) {
                if (next_blk < end_row) {
                    issue_block(next_blk, staging0);
                }
            }
            stage = next_stage;
        }
    } else {
        constexpr uint32_t Wt = tensor_tile_cols;
        constexpr uint32_t Ht = tensor_tile_rows;
        constexpr bool skip_r = w_tail >= 1 && w_tail <= 16;  // faces 1/3 of a last-column tile are padding
        constexpr bool skip_b = h_tail >= 1 && h_tail <= 16;  // faces 2/3 of a last-row tile are padding
        // bf8b output only (bf16 input, 2 B values): zero the padding columns (col >= w_tail % 16) of the face pair
        // that straddles W, so the packer's shared exponent of a straddling 16-value face row never sees padding.
        // Whole padding faces need nothing: they are never read nor written, and bfp8 shares an exponent only
        // within one 16-value face row.
        constexpr uint32_t col0 = w_tail % 16;
        constexpr uint32_t face_a = (w_tail != 0 && w_tail < 16) ? 0 : 1;

        const uint32_t end_tile = start_tile + core_tiles;
        g6_faces::TilePos tp = g6_faces::tile_pos<Wt, Ht>(start_tile);
        for (uint32_t block_start = start_tile; block_start < end_tile; block_start += block_tiles) {
            const uint32_t extent = (end_tile - block_start < block_tiles) ? (end_tile - block_start) : block_tiles;
            {
                G6_ZONE("READ_WAIT_SPACE");
                cb_reserve_back(cb_input_tiles, extent);
            }
            G6_ZONE("READ_BLOCK");
            const uint32_t base = get_write_ptr(cb_input_tiles);
            [[maybe_unused]] const g6_faces::TilePos tp_block = tp;
            uint32_t l1 = base;
            for (uint32_t t = 0; t < extent; ++t) {
                const uint32_t page = block_start + t;
                g6_faces::for_live_segments<in_page_bytes, tile_hdr_bytes, face_bytes, skip_r, skip_b, Wt, Ht>(
                    tp, [&](uint32_t off, uint32_t bytes) {
                        noc_async_read_one_packet(input_accessor.get_noc_addr(page, off), l1 + off, bytes);
                    });
                g6_faces::advance<Wt, Ht>(tp);
                l1 += in_page_bytes;
            }
            noc_async_read_barrier();
            if constexpr (mask_pad_columns && col0 != 0) {
                g6_faces::TilePos tpm = tp_block;
                for (uint32_t t = 0; t < extent; ++t) {
                    if (tpm.col == Wt - 1) {
                        const uint32_t tile = base + t * in_page_bytes;
                        for (uint32_t f : {face_a, face_a + 2}) {
                            volatile tt_l1_ptr uint16_t* p =
                                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile + f * face_bytes);
                            for (uint32_t r = 0; r < 16; ++r) {
                                for (uint32_t c = col0; c < 16; ++c) {
                                    p[r * 16 + c] = 0;
                                }
                            }
                        }
                    }
                    g6_faces::advance<Wt, Ht>(tpm);
                }
            }
            cb_push_back(cb_input_tiles, extent);
        }
    }
}
