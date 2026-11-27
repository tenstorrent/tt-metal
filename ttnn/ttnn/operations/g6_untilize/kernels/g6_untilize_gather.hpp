// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// gather_row_block: face-row pieces of the TILE input land directly at their row-major positions in
// one cb_row_block slot. Raw NoC reads: no kernel-lib helper gathers face rows of TILE pages into
// sticks (read_sticks_for_tilize reads whole sticks; the local-copy helpers are L1->L1 only). A
// gather_sticks_from_tiles helper, the producer twin of write_sticks_after_untilize, would close the gap.
//
// Nothing touches the values, so all 32 bits survive by construction. The caller owns the barrier.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace g6_untilize {

// Compile-time geometry shared by every kernel that gathers; CT-arg indices 0..10 of both kernels.
struct GatherArgs {
    static constexpr uint32_t cb_row_block = get_compile_time_arg_val(0);
    static constexpr uint32_t tensor_rows = get_compile_time_arg_val(1);
    static constexpr uint32_t num_cores = get_compile_time_arg_val(2);
    static constexpr uint32_t rect_w = get_compile_time_arg_val(3);
    static constexpr uint32_t rows_per_image = get_compile_time_arg_val(4);
    static constexpr uint32_t row_elems = get_compile_time_arg_val(5);
    static constexpr uint32_t tile_rows_per_image = get_compile_time_arg_val(6);
    static constexpr uint32_t tensor_tile_cols = get_compile_time_arg_val(7);
    static constexpr uint32_t block_tile_cols = get_compile_time_arg_val(8);
    static constexpr bool bank_stagger = get_compile_time_arg_val(9) != 0;
    // Number of data-movement RISCs issuing the gather: 1 (reader only) or 2 (reader + writer).
    static constexpr uint32_t gather_riscs = get_compile_time_arg_val(10);
    static constexpr uint32_t next_ct_arg = 11;

    static constexpr uint32_t num_col_blocks = tensor_tile_cols / block_tile_cols;
    static_assert(num_col_blocks * block_tile_cols == tensor_tile_cols, "block_tile_cols must divide the tile cols");
    static_assert(gather_riscs == 1 || gather_riscs == 2, "gather_riscs must be 1 or 2");

    // Tile/face geometry of the CB, which the host formats as one tile of the input.
    static constexpr uint32_t tile_h = unpack_tile_r_dim[cb_row_block];
    static constexpr uint32_t tile_w = unpack_tile_c_dim[cb_row_block];
    static constexpr uint32_t face_h = unpack_tile_face_r_dim[cb_row_block];
    static constexpr uint32_t faces_per_tile_row = unpack_num_faces_c_dim[cb_row_block];
    static constexpr uint32_t face_w = tile_w / faces_per_tile_row;
    static constexpr uint32_t elem_bytes = get_tile_size(cb_row_block) / get_tile_hw(cb_row_block);
    static constexpr uint32_t face_row_bytes = face_w * elem_bytes;
    static constexpr uint32_t face_bytes = face_h * face_row_bytes;
    static constexpr uint32_t face_band_bytes = faces_per_tile_row * face_bytes;
    static constexpr uint32_t tile_row_bytes = tile_w * elem_bytes;
    // Row stride inside a slot: the writer helper's padded row of block_tile_cols tile rows.
    static constexpr uint32_t padded_chunk_row_bytes = block_tile_cols * tile_row_bytes;
    static_assert(face_row_bytes % 64 == 0, "face-row pieces must keep the BH 64-B DRAM read alignment");
};

struct CoreRows {
    uint32_t core_idx;
    uint32_t start;
    uint32_t count;
};

// Contiguous balanced row range of this core in the rect_w-wide rectangle at logical (0,0).
// Must match _core_rows() in g6_untilize_program_descriptor.py (describe_plan reports it).
FORCE_INLINE CoreRows core_rows() {
    using A = GatherArgs;
    constexpr uint32_t base = A::tensor_rows / A::num_cores;
    constexpr uint32_t extra = A::tensor_rows % A::num_cores;
    const uint32_t core_idx = get_relative_logical_y() * A::rect_w + get_relative_logical_x();
    const uint32_t start = core_idx * base + (core_idx < extra ? core_idx : extra);
    return {core_idx, start, base + (core_idx < extra ? 1u : 0u)};
}

// First tile column of each core's walk. With bank_stagger the first requests of all cores spread
// over the banks of the block's tiles instead of queueing on one.
FORCE_INLINE uint32_t first_tile_col(const CoreRows& mine) {
    return GatherArgs::bank_stagger ? mine.core_idx % GatherArgs::block_tile_cols : 0;
}

// Issue the reads of one block's walk steps risc_idx, risc_idx + gather_riscs, ... on this RISC's NoC.
// Walk step s covers tile column (first_c + s) mod block_tile_cols, all block_rows rows of it.
// Tile-column outer, rows inner: consecutive pieces come from the same source tile, so the bank
// coordinate and piece size are programmed once per source tile and each piece only rewrites addresses.
template <uint32_t risc_idx, typename Accessor>
FORCE_INLINE void gather_row_block(
    const Accessor& input,
    uint32_t l1_block,
    uint32_t g0,
    uint32_t block_rows,
    uint32_t col_block_idx,
    uint32_t first_c) {
    using A = GatherArgs;
    const uint32_t tile_col0 = col_block_idx * A::block_tile_cols;
    const uint32_t b0 = g0 / A::rows_per_image;
    const uint32_t r0 = g0 - b0 * A::rows_per_image;

    uint32_t c = first_c + risc_idx;
    for (uint32_t step = risc_idx; step < A::block_tile_cols; step += A::gather_riscs) {
        if (c >= A::block_tile_cols) {
            c -= A::block_tile_cols;
        }
        const uint32_t tile_col = tile_col0 + c;
        uint32_t valid_faces = A::faces_per_tile_row;
        while (valid_faces > 1 && tile_col * A::tile_w + (valid_faces - 1) * A::face_w >= A::row_elems) {
            --valid_faces;  // faces of pure tile padding are never read
        }

        uint32_t b = b0, r = r0;
        uint32_t cur_tile_row = 0xFFFFFFFF;
        uint32_t tile_local_addr = 0;
        uint32_t l1_dst = l1_block + c * A::tile_row_bytes;
        for (uint32_t j = 0; j < block_rows; ++j) {
            const uint32_t tile_row = b * A::tile_rows_per_image + r / A::tile_h;
            if (tile_row != cur_tile_row) {
                cur_tile_row = tile_row;
                const uint64_t tile_noc_addr = input.get_noc_addr(tile_row * A::tensor_tile_cols + tile_col);
                noc_async_read_one_packet_set_state(tile_noc_addr, A::face_row_bytes);
                tile_local_addr = static_cast<uint32_t>(tile_noc_addr);
            }
            const uint32_t in_tile_r = r % A::tile_h;
            uint32_t src = tile_local_addr + (in_tile_r / A::face_h) * A::face_band_bytes +
                           (in_tile_r % A::face_h) * A::face_row_bytes;
            uint32_t dst = l1_dst;
            for (uint32_t f = 0; f < valid_faces; ++f) {
                noc_async_read_one_packet_with_state(src, dst);
                src += A::face_bytes;
                dst += A::face_row_bytes;
            }
            l1_dst += A::padded_chunk_row_bytes;
            if (++r == A::rows_per_image) {
                r = 0;
                ++b;
            }
        }
        c += A::gather_riscs;
    }
}

}  // namespace g6_untilize
