// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_typecast reader (NCRISC / NoC0) — block operation `load_block` (op_design.md, Block schedule).
// Streams this core's contiguous range of input tiles [start_tile, start_tile + core_tiles) into
// cb_input_tiles in blocks of `block_tiles`: block_tiles page reads in flight, ONE read barrier and
// ONE reserve/push handshake per block. A ragged last block reserves/pushes its actual extent
// (compute and writer synchronize the same per-block extent).
// No dataflow helper owns an interleaved page-range stream, so this is the raw TensorAccessor loop.
//
// rm_mode (regime R2 `rm_pseudo_tile_stream`): the core owns rows [start_row, start_row + core_rows) of an
// interleaved ROW_MAJOR tensor. Per block of `rows_per_block` rows: ONE NoC read per bank run (rows r, r+nb,
// r+2nb, ... sit adjacent in one bank at the aligned-page stride, nb = accessor.contiguous_page_stride())
// into the 64B-aligned staging zone, ONE read barrier, then a RISC-V compaction of the rows into the dense
// pseudo-tile block of cb_input_tiles (ceil(rows * W / 1024) pages).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "g6_typecast_rm_rows.hpp"

namespace {

// bfloat8_b output only: zero the padding columns (col >= w_tail) of a last-tile-column tile so the
// packer's shared exponent of a straddling 16-value group never sees padding values.
// Tile = 4 faces of 16x16 bf16 (face0 TL, face1 TR, face2 BL, face3 BR), row-major inside a face.
template <uint32_t w_tail>
FORCE_INLINE void zero_pad_columns(const Noc& noc, const CircularBuffer& cb, uint32_t tile_offset_bytes) {
    constexpr uint32_t face_dim = 16;
    constexpr uint32_t elem_bytes = 2;
    constexpr uint32_t face_bytes = face_dim * face_dim * elem_bytes;
    if constexpr (w_tail < face_dim) {
        // Right faces (1, 3) are entirely padding: zero them with the DM engine.
        noc.async_write_zeros(cb, face_bytes, {.offset_bytes = tile_offset_bytes + 1 * face_bytes});
        noc.async_write_zeros(cb, face_bytes, {.offset_bytes = tile_offset_bytes + 3 * face_bytes});
    }
    // Partial face rows: only the padding lanes of the face that straddles W.
    constexpr uint32_t col0 = w_tail % face_dim;  // first padding column inside the straddling face
    constexpr uint32_t face_a = (w_tail < face_dim) ? 0 : 1;
    constexpr uint32_t face_b = face_a + 2;
    if constexpr (col0 != 0) {
        uint32_t base = cb.get_write_ptr() + tile_offset_bytes;
        for (uint32_t face : {face_a, face_b}) {
            volatile tt_l1_ptr uint16_t* f = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + face * face_bytes);
            for (uint32_t r = 0; r < face_dim; ++r) {
                for (uint32_t c = col0; c < face_dim; ++c) {
                    f[r * face_dim + c] = 0;
                }
            }
        }
    }
}

}  // namespace

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
    constexpr auto input_args = TensorAccessorArgs<13>();

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
            DeviceZoneScopedN("READ_BLOCK");
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
                DeviceZoneScopedN("READ_WAIT_SPACE");
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
        return;
    }
    [[maybe_unused]] Noc noc;
    CircularBuffer cb_in(cb_input_tiles);

    const uint32_t end_tile = start_tile + core_tiles;
    for (uint32_t block_start = start_tile; block_start < end_tile; block_start += block_tiles) {
        DeviceZoneScopedN("READ_BLOCK");
        const uint32_t block_extent = (end_tile - block_start < block_tiles) ? (end_tile - block_start) : block_tiles;
        {
            DeviceZoneScopedN("READ_WAIT_SPACE");
            cb_reserve_back(cb_input_tiles, block_extent);
        }
        uint32_t l1_addr = get_write_ptr(cb_input_tiles);
        for (uint32_t t = 0; t < block_extent; ++t) {
            noc_async_read_page(block_start + t, input_accessor, l1_addr);
            l1_addr += in_page_bytes;
        }
        noc_async_read_barrier();

        if constexpr (mask_pad_columns) {
            // Last-tile-column tiles of this block: page id % tensor_tile_cols == tensor_tile_cols - 1.
            for (uint32_t t = 0; t < block_extent; ++t) {
                if ((block_start + t) % tensor_tile_cols == tensor_tile_cols - 1) {
                    zero_pad_columns<w_tail>(noc, cb_in, t * in_page_bytes);
                }
            }
            noc.write_zeros_l1_barrier();
        }

        cb_push_back(cb_input_tiles, block_extent);
    }
}
