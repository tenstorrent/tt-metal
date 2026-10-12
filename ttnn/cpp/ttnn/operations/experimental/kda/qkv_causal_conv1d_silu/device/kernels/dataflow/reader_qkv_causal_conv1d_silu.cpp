// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Reader for qkv_causal_conv1d_silu.
//
// The input is a TILE-layout tensor read in place (optionally a column window of a wider tensor). For every channel
// tile the reader stages a 35-row "strip" per face column: rows [0, 3) hold the three tokens preceding the tile and
// rows [3, 35) the tile's own 32 tokens. A strip is the left (columns 0-15) or right (columns 16-31) half of the
// tile, stored as 32-byte face rows, so the tile's faces land in it with four contiguous face reads. Every tap's
// shifted tile is then four contiguous 512-byte local copies out of the two strips: tap t reads tokens r - 3 + t,
// i.e. strip rows [t, t + 16) and [t + 16, t + 32). Compute applies the taps as matmuls with diagonal weight tiles,
// which it builds once per channel block from the tap tiles and an identity tile the reader supplies.

#include "ttnn/cpp/ttnn/operations/experimental/kda/chronological_selections/device/kernels/chronology.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/qkv_causal_conv1d_silu_layout.hpp"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/scratchpad.h"
#include "api/semaphore.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

namespace {

using namespace qkv_conv;

constexpr uint32_t tap_count = history_rows + 1;
constexpr uint32_t face_rows = 16;
constexpr uint32_t tile_bytes = 4 * face_bytes;
constexpr uint32_t strip_rows = history_rows + 2 * face_rows;
static_assert(strip_stride >= strip_rows * face_row_bytes && strip_stride % 64 == 0);
// Strips start 32 bytes past a 64-byte boundary, so the face reads into rows 3 and 19 are 64-byte aligned.
constexpr uint32_t strip_phase = 32;
static_assert((strip_phase + history_rows * face_row_bytes) % 64 == 0);

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
    uint32_t exchange,
    uint32_t sp_rank,
    uint32_t sp_size,
    uint32_t local_rows>
TT_KERNEL void reader(uint32_t wi_start, uint32_t wi_count) {
    const auto input = TensorAccessor(tensor::input);
    const auto history = TensorAccessor(tensor::history);
    const auto predecessor = TensorAccessor(tensor::predecessor_rows);
    const auto tap0 = TensorAccessor(tensor::tap0);
    const auto tap1 = TensorAccessor(tensor::tap1);
    const auto tap2 = TensorAccessor(tensor::tap2);
    const auto tap3 = TensorAccessor(tensor::tap3);
    DataflowBuffer taps(dfb::taps);
    DataflowBuffer eye(dfb::eye);
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
    //   staging  block_ct tiles: the previous tile's bottom faces
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

    const auto read_scalar = [&](const auto& accessor) {
        noc.async_read(accessor, CoreLocalMem<uint32_t>(rows), sizeof(uint32_t), {.page_id = 0}, {});
        noc.async_read_barrier();
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows)[0];
    };
    const auto topology =
        kda_chronology::derive(read_scalar(TensorAccessor(tensor::actual_start)), sp_rank, sp_size, local_rows);
    const uint32_t local_split_row = topology.local_split ? topology.head_rows : 0;
    const bool initial_from_predecessor = topology.rank != topology.first_rank;

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

    // With the fabric exchange, the exchange core releases this core once the predecessor's rows have landed.
    Semaphore predecessor_ready(sem::predecessor_ready);
    bool predecessor_arrived = !exchange;

    constexpr uint32_t no_block = ~0U;
    uint32_t loaded_block = no_block;
    uint32_t previous_work = no_block;
    uint32_t parity = 0;
    const auto process = [&](uint32_t work) {
        const uint32_t block = work / Mt;
        const uint32_t mt = work % Mt;
        const uint32_t ct_start = block * block_ct;

        if (block != loaded_block) {
            // Compute turns each tap into a diagonal tile, identity times the tap's row: it needs the identity once
            // and, per channel block, the tap tiles laid out [tap][channel tile]. Row 0 of a tap tile holds the
            // weights, so only its top faces (0 and 1) are read.
            if (loaded_block == no_block) {
                eye.reserve_back(1);
                noc.async_write_zeros(eye, tile_bytes);
                noc.write_zeros_l1_barrier();
                constexpr uint16_t bf16_one = 0x3F80;
                auto* tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(eye.get_write_ptr());
                for (uint32_t i = 0; i < face_rows; ++i) {
                    tile[i * 16 + i] = bf16_one;            // face 0
                    tile[3 * 256 + i * 16 + i] = bf16_one;  // face 3
                }
                eye.push_back(1);
            }
            taps.reserve_back(tap_count * block_ct);
            for (uint32_t ct = 0; ct < block_ct; ++ct) {
                const uint32_t source_ct = ct_start + ct;
                const auto read_tap = [&](const auto& tap, uint32_t index) {
                    noc.async_read(
                        tap, taps, 2 * face_bytes, {.page_id = source_ct}, {.offset_bytes = index * tile_bytes});
                };
                read_tap(tap0, ct);
                read_tap(tap1, block_ct + ct);
                read_tap(tap2, 2 * block_ct + ct);
                read_tap(tap3, 3 * block_ct + ct);
            }
            noc.async_read_barrier();
            taps.push_back(tap_count * block_ct);
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
                if (!predecessor_arrived) {
                    predecessor_ready.wait_min(1);
                    predecessor_arrived = true;
                }
                read_rows(predecessor, predecessor_page, ct_start);
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

        previous_work = work;
        parity ^= 1;
    };
    for_each_work_item(wi_start, wi_count, Mt, exchange, process);
}
