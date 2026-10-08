// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/experimental/kda/chronological_selections/device/kernels/chronology.hpp"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/scratchpad.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

namespace {

constexpr uint32_t tap_count = 4;
constexpr uint32_t history_rows = tap_count - 1;
// A BF16 tile holds four 16x16 faces; each face row is 16 contiguous values.
constexpr uint32_t face_rows = 16;
constexpr uint32_t face_row_bytes = 16 * sizeof(uint16_t);
constexpr uint32_t face_bytes = face_rows * face_row_bytes;

FORCE_INLINE void copy_local(uint32_t source, uint32_t destination, uint32_t bytes) {
    noc_async_read(get_noc_addr(source), destination, bytes);
}

// Row r of the shifted tile is row r - shift of the current tile; its first `shift` rows are the
// last rows of the previous tile. Within one face column the moved rows stay contiguous.
FORCE_INLINE void shift_rows(uint32_t current, uint32_t previous, uint32_t destination, uint32_t shift) {
    const uint32_t kept = (face_rows - shift) * face_row_bytes;
    const uint32_t carried = shift * face_row_bytes;
    for (uint32_t column = 0; column < 2; ++column) {
        const uint32_t top = column * face_bytes;
        const uint32_t bottom = (2 + column) * face_bytes;
        copy_local(previous + bottom + kept, destination + top, carried);
        copy_local(current + top, destination + top + carried, kept);
        copy_local(current + top + kept, destination + bottom, carried);
        copy_local(current + bottom, destination + bottom + carried, kept);
    }
}

}  // namespace

// Reads the tiled projection directly: each work item stages one channel block's tile row and the
// rows before it, then builds the four shifted tap views as tiles with local copies.
template <
    uint32_t block_ct,
    uint32_t Mt,
    uint32_t input_row_tiles,
    uint32_t sp_rank,
    uint32_t sp_size,
    uint32_t local_rows>
TT_KERNEL void reader(uint32_t wi_start, uint32_t wi_count) {
    const auto input = TensorAccessor(tensor::input);
    const auto history = TensorAccessor(tensor::history);
    const auto predecessor_carry = TensorAccessor(tensor::predecessor_carry);
    const auto tap0 = TensorAccessor(tensor::tap0);
    const auto tap1 = TensorAccessor(tensor::tap1);
    const auto tap2 = TensorAccessor(tensor::tap2);
    const auto tap3 = TensorAccessor(tensor::tap3);
    DataflowBuffer weights(dfb::weights);
    DataflowBuffer taps(dfb::act_tile);
    // Private reader scratch: [current tile row | previous tile row | carry rows].
    Scratchpad<uint16_t> window(scratch::act_window);
    Noc noc;

    const uint32_t window_base = window.get_base_address();

    uint32_t local_split_row = 0;
    bool initial_from_predecessor = false;
    {
        // The window is free until the first work item, so it doubles as landing space for actual_start.
        const auto actual_start = TensorAccessor(tensor::actual_start);
        noc.async_read(actual_start, CoreLocalMem<uint32_t>(window_base), sizeof(uint32_t), {.page_id = 0}, {});
        noc.async_read_barrier();
        const auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(window_base);
        const auto topology = kda_chronology::derive(words[0], sp_rank, sp_size, local_rows);
        local_split_row = topology.local_split ? topology.head_rows : 0;
        initial_from_predecessor = topology.rank != topology.first_rank;
    }

    constexpr uint32_t tile_width = tt::constants::TILE_WIDTH;
    constexpr uint32_t tile_height = tt::constants::TILE_HEIGHT;
    constexpr uint32_t block_row_bytes = block_ct * tile_width * sizeof(uint16_t);
    constexpr uint32_t block_offset_scale = tile_width * sizeof(uint16_t);
    const uint32_t tile_bytes = weights.get_entry_size();
    const uint32_t block_bytes = block_ct * tile_bytes;
    uint32_t current = window_base;
    uint32_t previous = window_base + block_bytes;
    const uint32_t carry_rows = window_base + 2 * block_bytes;

    const auto read_tile_row = [&](uint32_t destination, uint32_t mt, uint32_t ct_start) {
        for (uint32_t ct = 0; ct < block_ct; ++ct) {
            noc.async_read(
                input,
                CoreLocalMem<uint32_t>(destination + ct * tile_bytes),
                tile_bytes,
                {.page_id = mt * input_row_tiles + ct_start + ct},
                {});
        }
    };

    // Work items are channel-block-major, so consecutive items on a core share tap weights and
    // the next item's previous tile row is the one already staged.
    constexpr uint32_t no_block = ~0U;
    uint32_t loaded_block = no_block;
    uint32_t staged_work = wi_start + wi_count;
    for (uint32_t item = 0; item < wi_count; ++item) {
        const uint32_t work = wi_start + item;
        const uint32_t block = work / Mt;
        const uint32_t mt = work % Mt;
        const uint32_t ct_start = block * block_ct;

        if (block != loaded_block) {
            weights.reserve_back(tap_count * block_ct);
            // The row broadcast reads only each tap tile's row 0, which spans the first rows of faces 0 and 1.
            const auto read_tap_row = [&](const auto& tap, uint32_t source_ct, uint32_t slot) {
                for (uint32_t column = 0; column < 2; ++column) {
                    noc.async_read(
                        tap,
                        weights,
                        face_row_bytes,
                        {.page_id = source_ct, .offset_bytes = column * face_bytes},
                        {.offset_bytes = slot * tile_bytes + column * face_bytes});
                }
            };
            for (uint32_t ct = 0; ct < block_ct; ++ct) {
                const uint32_t source_ct = ct_start + ct;
                // The weight DFB is laid out as [tap][channel tile].
                read_tap_row(tap0, source_ct, ct);
                read_tap_row(tap1, source_ct, block_ct + ct);
                read_tap_row(tap2, source_ct, 2 * block_ct + ct);
                read_tap_row(tap3, source_ct, 3 * block_ct + ct);
            }
            noc.async_read_barrier();
            weights.push_back(tap_count * block_ct);
            loaded_block = block;
        }

        // Tile-aligned actual_start and local_rows make the split tile-aligned, so the rows before
        // this tile are either the previous tile row or, at a segment start, the three carry rows.
        const uint32_t row_floor = local_split_row != 0 && mt * tile_height >= local_split_row ? local_split_row : 0;
        const bool from_carry = mt * tile_height == row_floor;
        if (!from_carry && staged_work + 1 == work && mt != 0) {
            const uint32_t staged = current;
            current = previous;
            previous = staged;
        } else if (!from_carry) {
            read_tile_row(previous, mt - 1, ct_start);
        }
        read_tile_row(current, mt, ct_start);
        if (from_carry) {
            const auto read_carry = [&](const auto& carry) {
                for (uint32_t row = 0; row < history_rows; ++row) {
                    noc.async_read(
                        carry,
                        CoreLocalMem<uint32_t>(carry_rows + row * block_row_bytes),
                        block_row_bytes,
                        {.page_id = row, .offset_bytes = ct_start * block_offset_scale},
                        {});
                }
            };
            if (initial_from_predecessor || row_floor != 0) {
                read_carry(predecessor_carry);
            } else {
                read_carry(history);
            }
            noc.async_read_barrier();
            // Only the previous tile's last three rows are read by the shifted views.
            for (uint32_t ct = 0; ct < block_ct; ++ct) {
                for (uint32_t row = 0; row < history_rows; ++row) {
                    for (uint32_t column = 0; column < 2; ++column) {
                        copy_local(
                            carry_rows + row * block_row_bytes + ct * block_offset_scale + column * face_row_bytes,
                            previous + ct * tile_bytes + (2 + column) * face_bytes +
                                (face_rows - history_rows + row) * face_row_bytes,
                            face_row_bytes);
                    }
                }
            }
        }
        noc.async_read_barrier();
        staged_work = work;

        // Tap t reads rows shifted down by 3 - t; the last tap is the current tile row itself.
        for (uint32_t tap = 0; tap < tap_count; ++tap) {
            const uint32_t shift = history_rows - tap;
            taps.reserve_back(block_ct);
            const uint32_t destination = taps.get_write_ptr();
            if (shift == 0) {
                copy_local(current, destination, block_bytes);
            } else {
                for (uint32_t ct = 0; ct < block_ct; ++ct) {
                    shift_rows(
                        current + ct * tile_bytes, previous + ct * tile_bytes, destination + ct * tile_bytes, shift);
                }
            }
            noc.async_read_barrier();
            taps.push_back(block_ct);
        }
    }
}
