// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Tiled-input reader for qkv_causal_conv1d_silu.
//
// The input is a TILE-layout tensor read in place (optionally a column window of a wider tensor). For every channel
// tile the reader stages a 35-row "strip" per face column: rows [0, 3) hold the three tokens preceding the tile and
// rows [3, 35) the tile's own 32 tokens. A strip is the left (columns 0-15) or right (columns 16-31) half of the
// tile, stored as 32-byte face rows, so the tile's faces land in it with four contiguous face reads. Every tap's
// shifted tile is then four contiguous 512-byte local copies out of the two strips: tap t reads tokens r - 3 + t,
// i.e. strip rows [t, t + 16) and [t + 16, t + 32). Compute applies the taps as matmuls with diagonal weight tiles,
// which the reader builds once per channel block.

#include "ttnn/cpp/ttnn/operations/experimental/kda/chronological_selections/device/kernels/chronology.hpp"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/scratchpad.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

namespace {

constexpr uint32_t tap_count = 4;
constexpr uint32_t history_rows = tap_count - 1;
constexpr uint32_t face_rows = 16;
constexpr uint32_t face_row_bytes = 16 * sizeof(uint16_t);
constexpr uint32_t face_bytes = face_rows * face_row_bytes;
constexpr uint32_t tile_bytes = 4 * face_bytes;
constexpr uint32_t strip_rows = history_rows + 2 * face_rows;
// Strips start 32 bytes past a 64-byte boundary, so the face reads into rows 3 and 19 are 64-byte aligned.
constexpr uint32_t strip_stride = 1152;
static_assert(strip_stride >= strip_rows * face_row_bytes && strip_stride % 64 == 0);
constexpr uint32_t strip_phase = 32;
static_assert((strip_phase + history_rows * face_row_bytes) % 64 == 0);
// Exchanged history table: each SP rank contributes its outgoing rows, then its end-of-interval rows.
constexpr uint32_t table_rows_per_rank = 2 * history_rows;

FORCE_INLINE uint32_t align_up(uint32_t value, uint32_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

}  // namespace

template <
    uint32_t block_ct,
    uint32_t Mt,
    uint32_t group,
    uint32_t input_wt,
    uint32_t input_ct_offset,
    uint32_t has_table,
    uint32_t sp_rank,
    uint32_t sp_size,
    uint32_t local_rows,
    uint32_t has_actual_end>
TT_KERNEL void reader(uint32_t wi_start, uint32_t wi_count) {
    const auto input = TensorAccessor(tensor::input);
    const auto history = TensorAccessor(tensor::history);
    const auto predecessor = TensorAccessor(tensor::predecessor_carry);
    const auto tap0 = TensorAccessor(tensor::tap0);
    const auto tap1 = TensorAccessor(tensor::tap1);
    const auto tap2 = TensorAccessor(tensor::tap2);
    const auto tap3 = TensorAccessor(tensor::tap3);
    DataflowBuffer weights(dfb::weights);
    DataflowBuffer activation(dfb::act_tile);
    Scratchpad<uint16_t> scratch(scratch::act_window);
    Noc noc;
    UnicastEndpoint self;
    const uint32_t self_x = my_x[noc.get_noc_id()];
    const uint32_t self_y = my_y[noc.get_noc_id()];
    const auto local = [&](uint32_t address) {
        return noc_traits_t<UnicastEndpoint>::src_args_type{.noc_x = self_x, .noc_y = self_y, .addr = address};
    };

    constexpr uint32_t block_row_bytes = block_ct * 2 * face_row_bytes;
    // Scratch layout (offsets from a 64-byte aligned base):
    //   strips   [2 parities][block_ct][left, right] x strip_stride, shifted by strip_phase
    //   staging  block_ct tiles: the previous tile's bottom faces, or tap rows while building weights
    //   rows     history_rows x block_row_bytes: three carried rows of the channel block
    const uint32_t scratch_base = scratch.get_base_address();
    const uint32_t base = align_up(scratch_base, 64);
    const uint32_t strips = base;
    constexpr uint32_t strips_bytes = 2 * block_ct * 2 * strip_stride + 64;
    const uint32_t staging = base + strips_bytes;
    constexpr uint32_t staging_bytes = block_ct * tile_bytes;
    const uint32_t rows = staging + staging_bytes;
    const auto strip = [&](uint32_t parity, uint32_t ct, uint32_t side) {
        return strips + strip_phase + ((parity * block_ct + ct) * 2 + side) * strip_stride;
    };
    const auto offset_of = [&](uint32_t address) { return address - scratch_base; };

    uint32_t start = 0;
    kda_chronology::Topology topology;
    {
        const auto actual_start = TensorAccessor(tensor::actual_start);
        noc.async_read(actual_start, CoreLocalMem<uint32_t>(rows), sizeof(uint32_t), {.page_id = 0}, {});
        noc.async_read_barrier();
        start = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows)[0];
        topology = kda_chronology::derive(start, sp_rank, sp_size, local_rows);
    }
    uint32_t end = 0;
    if constexpr (has_actual_end) {
        const auto actual_end = TensorAccessor(*tensor::get_token_if_present<"actual_end">());
        noc.async_read(actual_end, CoreLocalMem<uint32_t>(rows), sizeof(uint32_t), {.page_id = 0}, {});
        noc.async_read_barrier();
        end = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows)[0];
    }
    const uint32_t local_split_row = topology.local_split ? topology.head_rows : 0;
    const bool initial_from_predecessor = topology.rank != topology.first_rank;
    // With an exchanged table the predecessor's outgoing rows are the first rows of its table entry.
    const uint32_t predecessor_base = has_table ? ((sp_rank + sp_size - 1) % sp_size) * table_rows_per_rank : 0;

    // Read history_rows rows of this channel block's columns into the rows staging area.
    const auto read_rows = [&](const auto& source, uint32_t first_row, uint32_t ct_start) {
        for (uint32_t row = 0; row < history_rows; ++row) {
            noc.async_read(
                source,
                scratch,
                block_row_bytes,
                {.page_id = first_row + row, .offset_bytes = ct_start * 2 * face_row_bytes},
                {.offset_bytes = offset_of(rows) + row * block_row_bytes});
        }
    };

    constexpr uint32_t no_block = ~0U;
    uint32_t loaded_block = no_block;
    uint32_t previous_work = no_block;
    uint32_t parity = 0;
    for (uint32_t item = 0; item < wi_count; ++item) {
        const uint32_t work = wi_start + item;
        const uint32_t block = work / Mt;
        const uint32_t mt = work % Mt;
        const uint32_t ct_start = block * block_ct;

        if (block != loaded_block) {
            // Diagonal weight tiles, laid out [tap][channel tile]: diag(tap)[c][c] = tap[c].
            // The weight buffer holds exactly one block, so every block lands at the same address: zero it once,
            // after which each block only rewrites the diagonals.
            weights.reserve_back(tap_count * block_ct);
            if (loaded_block == no_block) {
                noc.async_write_zeros(weights, tap_count * block_ct * tile_bytes);
            }
            // Each (channel tile, tap) gets a 128-byte slot: the face 0 row at +0 and the face 1 row at +64, so every
            // 32-byte DRAM read lands 64-byte aligned.
            constexpr uint32_t slot_bytes = 128;
            static_assert(block_ct * tap_count * slot_bytes <= block_ct * tile_bytes);
            for (uint32_t ct = 0; ct < block_ct; ++ct) {
                const uint32_t source_ct = ct_start + ct;
                for (uint32_t face = 0; face < 2; ++face) {
                    // Row 0 of the tap tile holds the weights: face 0 has channels 0-15, face 1 channels 16-31.
                    const uint32_t src = face * face_bytes;
                    const uint32_t d = offset_of(staging) + ct * tap_count * slot_bytes + face * 64;
                    noc.async_read(
                        tap0,
                        scratch,
                        face_row_bytes,
                        {.page_id = source_ct, .offset_bytes = src},
                        {.offset_bytes = d});
                    noc.async_read(
                        tap1,
                        scratch,
                        face_row_bytes,
                        {.page_id = source_ct, .offset_bytes = src},
                        {.offset_bytes = d + slot_bytes});
                    noc.async_read(
                        tap2,
                        scratch,
                        face_row_bytes,
                        {.page_id = source_ct, .offset_bytes = src},
                        {.offset_bytes = d + 2 * slot_bytes});
                    noc.async_read(
                        tap3,
                        scratch,
                        face_row_bytes,
                        {.page_id = source_ct, .offset_bytes = src},
                        {.offset_bytes = d + 3 * slot_bytes});
                }
            }
            noc.write_zeros_l1_barrier();
            noc.async_read_barrier();
            auto* diag = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(weights.get_write_ptr());
            for (uint32_t ct = 0; ct < block_ct; ++ct) {
                for (uint32_t tap = 0; tap < tap_count; ++tap) {
                    const auto* values =
                        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(staging + (ct * tap_count + tap) * slot_bytes);
                    auto* tile = diag + (tap * block_ct + ct) * (tile_bytes / sizeof(uint16_t));
                    for (uint32_t i = 0; i < face_rows; ++i) {
                        tile[i * 16 + i] = values[i];                 // face 0
                        tile[3 * 256 + i * 16 + i] = values[32 + i];  // face 3
                    }
                }
            }
            weights.push_back(tap_count * block_ct);
            loaded_block = block;
        }

        // The three tokens preceding this tile: the chronological carry at a segment start (the logical start reads
        // the caller's history, any other segment start the predecessor's rows), else the previous tile's last rows.
        // Tile-aligned actual_start makes every segment start tile-aligned.
        const bool segment_start = mt == 0 || (local_split_row != 0 && mt * 32 == local_split_row);
        const bool from_predecessor = segment_start && (initial_from_predecessor || mt != 0);
        const bool from_previous_strip = !segment_start && previous_work + 1 == work;
        if (segment_start) {
            if (from_predecessor) {
                read_rows(predecessor, predecessor_base, ct_start);
            } else {
                read_rows(history, 0, ct_start);
            }
        } else if (!from_previous_strip) {
            for (uint32_t ct = 0; ct < block_ct; ++ct) {
                noc.async_read(
                    input,
                    scratch,
                    2 * face_bytes,
                    {.page_id = (mt - 1) * input_wt + input_ct_offset + ct_start + ct, .offset_bytes = 2 * face_bytes},
                    {.offset_bytes = offset_of(staging) + ct * tile_bytes});
            }
        }
        for (uint32_t ct = 0; ct < block_ct; ++ct) {
            const uint32_t page = mt * input_wt + input_ct_offset + ct_start + ct;
            for (uint32_t face = 0; face < 4; ++face) {
                const uint32_t side = face % 2;
                const uint32_t row = history_rows + (face / 2) * face_rows;
                noc.async_read(
                    input,
                    scratch,
                    face_bytes,
                    {.page_id = page, .offset_bytes = face * face_bytes},
                    {.offset_bytes = offset_of(strip(parity, ct, side)) + row * face_row_bytes});
            }
        }
        noc.async_read_barrier();

        // Fill strip rows [0, 3).
        constexpr uint32_t carried_bytes = history_rows * face_row_bytes;
        for (uint32_t ct = 0; ct < block_ct; ++ct) {
            for (uint32_t side = 0; side < 2; ++side) {
                const uint32_t dst = offset_of(strip(parity, ct, side));
                if (segment_start) {
                    for (uint32_t row = 0; row < history_rows; ++row) {
                        noc.async_read(
                            self,
                            scratch,
                            face_row_bytes,
                            local(rows + row * block_row_bytes + (ct * 2 + side) * face_row_bytes),
                            {.offset_bytes = dst + row * face_row_bytes});
                    }
                } else if (from_previous_strip) {
                    noc.async_read(
                        self,
                        scratch,
                        carried_bytes,
                        local(strip(parity ^ 1, ct, side) + (strip_rows - history_rows) * face_row_bytes),
                        {.offset_bytes = dst});
                } else {
                    // Bottom faces 2 (left) and 3 (right) of the previous tile; their last three rows.
                    noc.async_read(
                        self,
                        scratch,
                        carried_bytes,
                        local(
                            staging + ct * tile_bytes + side * face_bytes +
                            (face_rows - history_rows) * face_row_bytes),
                        {.offset_bytes = dst});
                }
            }
        }
        noc.async_read_barrier();

        // Shifted tap tiles, [channel tile][tap] per group.
        for (uint32_t ct0 = 0; ct0 < block_ct; ct0 += group) {
            activation.reserve_back(group * tap_count);
            for (uint32_t i = 0; i < group; ++i) {
                const uint32_t left = strip(parity, ct0 + i, 0);
                const uint32_t right = strip(parity, ct0 + i, 1);
                for (uint32_t tap = 0; tap < tap_count; ++tap) {
                    const uint32_t tile = (i * tap_count + tap) * tile_bytes;
                    noc.async_read(
                        self, activation, face_bytes, local(left + tap * face_row_bytes), {.offset_bytes = tile});
                    noc.async_read(
                        self,
                        activation,
                        face_bytes,
                        local(right + tap * face_row_bytes),
                        {.offset_bytes = tile + face_bytes});
                    noc.async_read(
                        self,
                        activation,
                        face_bytes,
                        local(left + (tap + face_rows) * face_row_bytes),
                        {.offset_bytes = tile + 2 * face_bytes});
                    noc.async_read(
                        self,
                        activation,
                        face_bytes,
                        local(right + (tap + face_rows) * face_row_bytes),
                        {.offset_bytes = tile + 3 * face_bytes});
                }
            }
            noc.async_read_barrier();
            activation.push_back(group * tap_count);
        }

        if constexpr (has_table) {
            if (mt == 0) {
                // The replacement carry: the three tokens ending at the logical end of the interval, which the
                // owner of the final segment exchanged; with fewer than three valid rows in that segment, the
                // tokens before it come from the caller's history or the preceding rank's outgoing rows.
                const auto final_history = TensorAccessor(*tensor::get_token_if_present<"final_history">());
                const uint32_t owner =
                    has_actual_end
                        ? kda_chronology::derive_interval(start, end, sp_rank, sp_size, local_rows).final_owner
                        : topology.final_owner;
                const auto owner_topology =
                    has_actual_end ? kda_chronology::derive_interval(start, end, owner, sp_size, local_rows)
                                   : kda_chronology::derive(start, owner, sp_size, local_rows);
                const bool tail = owner_topology.has_valid_tail();
                const uint32_t segment_rows = owner_topology.valid_rows - (tail ? owner_topology.head_rows : 0);
                const uint32_t carried = segment_rows < history_rows ? segment_rows : history_rows;
                const bool from_layer_history = owner == owner_topology.first_rank && !tail;
                for (uint32_t row = 0; row < history_rows; ++row) {
                    const uint32_t position = carried + row;
                    const uint32_t dst = offset_of(rows) + row * block_row_bytes;
                    const uint32_t offset = ct_start * 2 * face_row_bytes;
                    if (position >= history_rows) {
                        noc.async_read(
                            predecessor,
                            scratch,
                            block_row_bytes,
                            {.page_id = owner * table_rows_per_rank + position, .offset_bytes = offset},
                            {.offset_bytes = dst});
                    } else if (from_layer_history) {
                        noc.async_read(
                            history,
                            scratch,
                            block_row_bytes,
                            {.page_id = position, .offset_bytes = offset},
                            {.offset_bytes = dst});
                    } else {
                        noc.async_read(
                            predecessor,
                            scratch,
                            block_row_bytes,
                            {.page_id = ((owner + sp_size - 1) % sp_size) * table_rows_per_rank + position,
                             .offset_bytes = offset},
                            {.offset_bytes = dst});
                    }
                }
                noc.async_read_barrier();
                for (uint32_t row = 0; row < history_rows; ++row) {
                    noc.async_write(
                        scratch,
                        final_history,
                        block_row_bytes,
                        {.offset_bytes = offset_of(rows) + row * block_row_bytes},
                        {.page_id = row, .offset_bytes = ct_start * 2 * face_row_bytes});
                }
                noc.async_write_barrier();
            }
        }

        previous_work = work;
        parity ^= 1;
    }
}
