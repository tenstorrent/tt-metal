// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Copy the six rows the SP convolution exchanges out of a TILE-layout projection into a ROW_MAJOR [6, width]
// table: rows [0, 3) precede the next physical rank's segment, rows [3, 6) end this rank's valid interval.
// Each worker covers a range of channel tiles; per row it reads the two faces holding that row, then packs the
// row's 32-byte face halves into one contiguous row segment.

#include "ttnn/cpp/ttnn/operations/experimental/kda/chronological_selections/device/kernels/chronology.hpp"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/scratchpad.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

template <
    uint32_t input_wt,
    uint32_t input_ct_offset,
    uint32_t max_tiles,
    uint32_t sp_rank,
    uint32_t sp_size,
    uint32_t local_rows,
    uint32_t has_actual_end>
TT_KERNEL void reader(uint32_t ct_start, uint32_t ct_count) {
    constexpr uint32_t history_rows = 3;
    constexpr uint32_t face_row_bytes = 32;
    constexpr uint32_t face_bytes = 16 * face_row_bytes;
    constexpr uint32_t row_pair_bytes = 2 * face_bytes;
    if (ct_count == 0) {
        return;
    }
    Noc noc;
    Scratchpad<uint32_t> scratch(scratch::scratch);
    const auto input = TensorAccessor(tensor::input);
    const auto output = TensorAccessor(tensor::output);
    const uint32_t scratch_base = scratch.get_base_address();
    const uint32_t base = (scratch_base + 63) / 64 * 64;
    const uint32_t faces = base;                              // ct_count x two faces
    const uint32_t row = faces + max_tiles * row_pair_bytes;  // one packed row segment
    const uint32_t words = row + max_tiles * 2 * face_row_bytes;
    auto* scalar = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words);

    const auto actual_start = TensorAccessor(tensor::actual_start);
    noc.async_read(actual_start, CoreLocalMem<uint32_t>(words), sizeof(uint32_t), {.page_id = 0}, {});
    noc.async_read_barrier();
    const uint32_t start = scalar[0];
    auto topology = kda_chronology::derive(start, sp_rank, sp_size, local_rows);
    if constexpr (has_actual_end) {
        const auto actual_end = TensorAccessor(*tensor::get_token_if_present<"actual_end">());
        noc.async_read(actual_end, CoreLocalMem<uint32_t>(words), sizeof(uint32_t), {.page_id = 0}, {});
        noc.async_read_barrier();
        topology = kda_chronology::derive_interval(start, scalar[0], sp_rank, sp_size, local_rows);
    }

    uint32_t source_rows[2 * history_rows];
    // Outgoing: the rows before the next physical rank's segment, i.e. before this rank's head end.
    const uint32_t outgoing = (topology.local_split ? topology.head_rows : local_rows) - history_rows;
    // End: the rows ending at the valid end of this rank's end segment, as many as it holds (rows past a short
    // segment are unspecified; the convolution reads the preceding tokens for them instead).
    const bool tail = topology.has_valid_tail();
    const uint32_t segment_rows = topology.valid_rows - (tail ? topology.head_rows : 0);
    const uint32_t carried = segment_rows < history_rows ? segment_rows : history_rows;
    const uint32_t end_begin = topology.valid_rows - carried;
    for (uint32_t i = 0; i < history_rows; ++i) {
        source_rows[i] = outgoing + i;
        const uint32_t end_row = end_begin + i;
        source_rows[history_rows + i] = end_row < local_rows ? end_row : local_rows - 1;
    }

    const uint32_t row_offset = ct_start * 2 * face_row_bytes;
    for (uint32_t r = 0; r < 2 * history_rows; ++r) {
        const uint32_t source = source_rows[r];
        const uint32_t tile_row = source / 32;
        const uint32_t in_tile = source % 32;
        // Rows 0-15 live in faces 0 and 1, rows 16-31 in faces 2 and 3.
        const uint32_t pair_offset = (in_tile / 16) * row_pair_bytes;
        const uint32_t face_row = in_tile % 16;
        for (uint32_t ct = 0; ct < ct_count; ++ct) {
            noc.async_read(
                input,
                scratch,
                row_pair_bytes,
                {.page_id = tile_row * input_wt + input_ct_offset + ct_start + ct, .offset_bytes = pair_offset},
                {.offset_bytes = faces - scratch_base + ct * row_pair_bytes});
        }
        noc.async_read_barrier();
        auto* packed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(row);
        for (uint32_t ct = 0; ct < ct_count; ++ct) {
            const auto* left =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(faces + ct * row_pair_bytes + face_row * face_row_bytes);
            const auto* right = left + face_bytes / sizeof(uint32_t);
            for (uint32_t w = 0; w < face_row_bytes / sizeof(uint32_t); ++w) {
                packed[ct * 16 + w] = left[w];
                packed[ct * 16 + 8 + w] = right[w];
            }
        }
        noc.async_write(
            scratch,
            output,
            ct_count * 2 * face_row_bytes,
            {.offset_bytes = row - scratch_base},
            {.page_id = r, .offset_bytes = row_offset});
        noc.async_write_barrier();
    }
}
