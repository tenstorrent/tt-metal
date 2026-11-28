// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_unsqueeze face-row move — shared body of both data-movement kernels.
//
// One logical input row of fp32 tiles becomes one logical output column:
//   input tile t, (row 0, col c)  ->  output tile t, (row c, col 0)
// Tile half h (c in [16h, 16h+16)) lives in input face h row 0 (byte offset 1024*h, 64 B)
// and goes to output face 2h column 0 (byte offset 2048*h, a 1024 B face).
//
// Per block (block_w_tiles tiles x halves_per_risc halves):
//   load_face_rows_block        64 B DRAM reads -> row 0 of each staging entry, ONE read barrier
//   scatter_row_to_column_block in place, entry[row_words*r] = entry[r], r = 15..1 (32-bit RISC stores)
//   store_column_faces_block    DRAM writes in the WRITE_SHAPE transfer shape, no per-block barrier
// One write barrier at kernel end.
//
// WRITE_SHAPE (host knob, Refinement 2) picks the output transfer shape; only the staging stride and
// the write loop change:
//   WRITE_SHAPE_SEGMENT  entry = 16 rows x 16 B (256 B); 16 x 16 B column-0 writes per half, issued
//                        with the stateful one-packet write (state set once per half)
//   WRITE_SHAPE_FACE     entry = 16 rows x 64 B (1024 B face); one 1024 B write per half
//   WRITE_SHAPE_TILE     entry = 2048 B (face 2h sits at 2048 h of a 4096 B tile image); needs
//                        halves_per_risc == 2; one 4096 B whole-tile write per tile
// Only column 0 of output faces 0/2 is logical; whatever else a wider write carries is padding.
//
// Helper deviation (raw NoC + raw RISC stores), as recorded in op_design.md "Helpers considered
// and rejected": no kernel_lib helper moves a tile ROW into a tile COLUMN
// (dataflow_kernel_lib::copy_tile_column reads a source column, tile_column_dataflow.hpp:91-101),
// and local_copy_helpers' read_with_state is self-aimed (L1->L1) only. No compute kernel exists:
// any FPU/SFPU path narrows fp32 operands and would break bit-exactness.
//
// The staging CB is private scratch of this kernel instance: no push/pop hand-off exists.

#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"

namespace g6_unsqueeze {

constexpr uint32_t TILE_W = 32;
constexpr uint32_t FACE_W = 16;

constexpr uint32_t WRITE_SHAPE_SEGMENT = 0;
constexpr uint32_t WRITE_SHAPE_FACE = 1;
constexpr uint32_t WRITE_SHAPE_TILE = 2;

template <
    uint32_t half_begin,
    uint32_t halves_per_risc,
    uint32_t block_w_tiles,
    uint32_t buffer_depth,
    uint32_t logical_w,
    uint32_t elem_bytes,
    uint32_t read_align_bytes,
    uint32_t write_shape,
    typename InAccessor,
    typename OutAccessor>
struct FaceMover {
    static constexpr uint32_t face_row_bytes = FACE_W * elem_bytes;          // 64 B (fp32)
    static constexpr uint32_t face_bytes = FACE_W * face_row_bytes;          // 1024 B (fp32)
    static constexpr uint32_t in_half_stride = face_bytes;                   // input face h at 1024*h
    static constexpr uint32_t out_half_stride = 2 * face_bytes;              // output face 2h at 2048*h
    static constexpr uint32_t write_align = 16;                              // BH L1/DRAM NoC write alignment
    // Staging row stride: one aligned column-0 segment (SEGMENT) or a whole face row (FACE/TILE).
    static constexpr uint32_t row_bytes = write_shape == WRITE_SHAPE_SEGMENT ? write_align : face_row_bytes;
    static constexpr uint32_t row_words = row_bytes / elem_bytes;
    // Bytes per staged (tile, half) entry. TILE stages the output-tile image: face 2h at out_half_stride*h.
    static constexpr uint32_t entry_bytes = write_shape == WRITE_SHAPE_TILE ? out_half_stride : FACE_W * row_bytes;
    static constexpr uint32_t slot_entries = block_w_tiles * halves_per_risc;  // entries per staging slot
    static constexpr uint32_t slot_bytes = slot_entries * entry_bytes;
    static constexpr uint32_t read_align = read_align_bytes;  // BH DRAM read alignment (host READ_ALIGN_SLACK_BYTES)
    static_assert(write_shape <= WRITE_SHAPE_TILE, "unknown write shape");
    static_assert(
        write_shape != WRITE_SHAPE_TILE || (halves_per_risc == 2 && half_begin == 0),
        "whole-tile writes need both halves on one RISC");
    static_assert(entry_bytes % read_align == 0, "every entry's row 0 must be a 64 B-aligned read landing");
    static_assert((read_align & (read_align - 1)) == 0, "read alignment must be a power of two");
    static_assert(elem_bytes == 4, "scatter moves 32-bit words (float32 only)");

    const InAccessor& in;
    const OutAccessor& out;
    uint32_t stage_base;  // 64 B-aligned base of staging slot 0
    uint32_t start_tile;
    uint32_t core_w_tiles;

    static bool half_is_logical(uint32_t tile, uint32_t half) { return tile * TILE_W + half * FACE_W < logical_w; }

    uint32_t slot_addr(uint32_t block_idx) const { return stage_base + (block_idx % buffer_depth) * slot_bytes; }

    uint32_t block_extent(uint32_t block_idx) const {
        const uint32_t first = block_idx * block_w_tiles;
        const uint32_t left = core_w_tiles - first;
        return left < block_w_tiles ? left : block_w_tiles;
    }

    // Face j of a slot holds tile j / halves_per_risc, half half_begin + j % halves_per_risc.
    void load_face_rows_block(uint32_t block_idx) const {
        const uint32_t base = slot_addr(block_idx);
        const uint32_t tile0 = start_tile + block_idx * block_w_tiles;
        const uint32_t extent = block_extent(block_idx);
        for (uint32_t j = 0; j < extent; ++j) {
            const uint32_t tile = tile0 + j;
            for (uint32_t k = 0; k < halves_per_risc; ++k) {
                const uint32_t half = half_begin + k;
                if (!half_is_logical(tile, half)) {
                    continue;
                }
                noc_async_read(
                    in.get_noc_addr(tile, half * in_half_stride),
                    base + (j * halves_per_risc + k) * entry_bytes,
                    face_row_bytes);
            }
        }
    }

    void scatter_row_to_column_block(uint32_t block_idx) const {
        const uint32_t base = slot_addr(block_idx);
        const uint32_t extent = block_extent(block_idx);
        for (uint32_t f = 0; f < extent * halves_per_risc; ++f) {
            volatile tt_l1_ptr uint32_t* entry = dataflow_kernel_lib::addr_to_l1_ptr(base + f * entry_bytes);
            // Row-0 word r -> word row_words*r. Descending r: when row_words < FACE_W the targets overlap
            // row 0 (word 4 <- word 1), and word r is overwritten only at step r/row_words, after it was read.
            for (uint32_t r = FACE_W - 1; r >= 1; --r) {
                entry[r * row_words] = entry[r];
            }
        }
    }

    void store_column_faces_block(uint32_t block_idx) const {
        const uint32_t base = slot_addr(block_idx);
        const uint32_t tile0 = start_tile + block_idx * block_w_tiles;
        const uint32_t extent = block_extent(block_idx);
        for (uint32_t j = 0; j < extent; ++j) {
            const uint32_t tile = tile0 + j;
            if constexpr (write_shape == WRITE_SHAPE_TILE) {
                // Half 0 is always logical for an assigned tile; half 1 (if padding) rides along.
                noc_async_write(base + j * halves_per_risc * entry_bytes, out.get_noc_addr(tile), 2 * out_half_stride);
                continue;
            }
            for (uint32_t k = 0; k < halves_per_risc; ++k) {
                const uint32_t half = half_begin + k;
                if (!half_is_logical(tile, half)) {
                    continue;
                }
                const uint32_t src = base + (j * halves_per_risc + k) * entry_bytes;
                const uint64_t dst = out.get_noc_addr(tile, half * out_half_stride);
                if constexpr (write_shape == WRITE_SHAPE_FACE) {
                    noc_async_write(src, dst, face_bytes);
                } else {
                    // One state per half (all 16 segments go to the same bank), then 16 issues.
                    noc_async_write_one_packet_set_state(dst, row_bytes);
                    for (uint32_t r = 0; r < FACE_W; ++r) {
                        noc_async_write_one_packet_with_state(
                            src + r * row_bytes, static_cast<uint32_t>(dst) + r * face_row_bytes);
                    }
                }
            }
        }
    }

    void run() const {
        if (core_w_tiles == 0) {
            return;
        }
        const uint32_t num_blocks = (core_w_tiles + block_w_tiles - 1) / block_w_tiles;
        {
            DeviceZoneScopedN("G6U_READ_ISSUE");
            load_face_rows_block(0);
        }
        for (uint32_t b = 0; b < num_blocks; ++b) {
            {
                DeviceZoneScopedN("G6U_READ_WAIT");
                noc_async_read_barrier();
            }
            const bool has_next = b + 1 < num_blocks;
            if constexpr (buffer_depth > 1) {
                // Prefetch block b+1 into the other slot; that slot's writes (block b-1) must have left L1.
                if (has_next) {
                    noc_async_writes_flushed();
                    load_face_rows_block(b + 1);
                }
            }
            {
                DeviceZoneScopedN("G6U_SCATTER");
                scatter_row_to_column_block(b);
            }
            {
                DeviceZoneScopedN("G6U_WRITE_ISSUE");
                store_column_faces_block(b);
            }
            if constexpr (buffer_depth == 1) {
                // Single slot: block b+1 may land only after block b's writes have left it.
                if (has_next) {
                    noc_async_writes_flushed();
                    load_face_rows_block(b + 1);
                }
            }
        }
        DeviceZoneScopedN("G6U_WRITE_WAIT");
        noc_async_write_barrier();
    }
};

// Shared kernel_main body. CT args (in order):
//   0 half_begin, 1 halves_per_risc, 2 block_w_tiles, 3 buffer_depth, 4 logical_w, 5 cb_face_stage,
//   6 elem_bytes, 7 tile_bytes, 8 read_align_bytes, 9 write_shape, then TensorAccessorArgs(input),
//   TensorAccessorArgs(output).
// RT args: 0 input addr, 1 output addr, 2 start_tile, 3 core_w_tiles.
inline void kernel_body() {
    constexpr uint32_t half_begin = get_compile_time_arg_val(0);
    constexpr uint32_t halves_per_risc = get_compile_time_arg_val(1);
    constexpr uint32_t block_w_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t buffer_depth = get_compile_time_arg_val(3);
    constexpr uint32_t logical_w = get_compile_time_arg_val(4);
    constexpr uint32_t cb_face_stage = get_compile_time_arg_val(5);
    constexpr uint32_t elem_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t read_align_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t write_shape = get_compile_time_arg_val(9);
    constexpr auto in_args = TensorAccessorArgs<10>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();

    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_tile = get_arg_val<uint32_t>(2);
    const uint32_t core_w_tiles = get_arg_val<uint32_t>(3);

    const auto in = TensorAccessor(in_args, in_addr, tile_bytes);
    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);

    using Mover = FaceMover<
        half_begin,
        halves_per_risc,
        block_w_tiles,
        buffer_depth,
        logical_w,
        elem_bytes,
        read_align_bytes,
        write_shape,
        decltype(in),
        decltype(out)>;
    // The CB carries read_align bytes of slack so the 64 B-aligned landing fits.
    const uint32_t stage_base = (get_write_ptr(cb_face_stage) + Mover::read_align - 1) & ~(Mover::read_align - 1);
    const Mover mover{in, out, stage_base, start_tile, core_w_tiles};
    mover.run();
}

}  // namespace g6_unsqueeze
