// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/experimental/kda/chronological_selections/device/kernels/chronology.hpp"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/scratchpad.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

constexpr uint32_t tap_count = 4;

template <uint32_t block_ct, typename Tap0Accessor, typename Tap1Accessor, typename Tap2Accessor, typename Tap3Accessor>
FORCE_INLINE void load_weight_block(
    Noc& noc,
    DataflowBuffer& weights,
    const Tap0Accessor& tap0,
    const Tap1Accessor& tap1,
    const Tap2Accessor& tap2,
    const Tap3Accessor& tap3,
    uint32_t tile_bytes,
    uint32_t ct_start) {
    weights.reserve_back(tap_count * block_ct);
    for (uint32_t ct = 0; ct < block_ct; ++ct) {
        const uint32_t source_ct = ct_start + ct;
        // The weight DFB is laid out as [tap][channel tile].
        noc.async_read(tap0, weights, tile_bytes, {.page_id = source_ct}, {.offset_bytes = ct * tile_bytes});
        noc.async_read(
            tap1, weights, tile_bytes, {.page_id = source_ct}, {.offset_bytes = (block_ct + ct) * tile_bytes});
        noc.async_read(
            tap2, weights, tile_bytes, {.page_id = source_ct}, {.offset_bytes = (2 * block_ct + ct) * tile_bytes});
        noc.async_read(
            tap3, weights, tile_bytes, {.page_id = source_ct}, {.offset_bytes = (3 * block_ct + ct) * tile_bytes});
    }
    noc.async_read_barrier();
    weights.push_back(tap_count * block_ct);
}

template <uint32_t block_ct, uint32_t Mt, uint32_t sp_rank, uint32_t sp_size, uint32_t local_rows>
TT_KERNEL void reader(uint32_t wi_start, uint32_t wi_count) {
    const auto input = TensorAccessor(tensor::input);
    const auto history = TensorAccessor(tensor::history);
    const auto predecessor_carry = TensorAccessor(tensor::predecessor_carry);
    const auto tap0 = TensorAccessor(tensor::tap0);
    const auto tap1 = TensorAccessor(tensor::tap1);
    const auto tap2 = TensorAccessor(tensor::tap2);
    const auto tap3 = TensorAccessor(tensor::tap3);
    DataflowBuffer weights(dfb::weights);
    DataflowBuffer activation(dfb::act_rm);
    // Private reader scratch: rows [mt * 32 - 3, mt * 32 + 32) of one channel block, read from DRAM once and
    // then copied locally into the four shifted tap views.
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

    constexpr uint32_t history_rows = tap_count - 1;
    constexpr uint32_t tile_width = tt::constants::TILE_WIDTH;
    constexpr uint32_t tile_height = tt::constants::TILE_HEIGHT;
    constexpr uint32_t block_row_bytes = block_ct * tile_width * sizeof(uint16_t);
    constexpr uint32_t block_offset_scale = tile_width * sizeof(uint16_t);
    // Every row and tap offset into DRAM and the L1 window is a multiple of block_row_bytes.
    static_assert(block_row_bytes % L1_ALIGNMENT == 0 && block_row_bytes % DRAM_ALIGNMENT == 0);
    const uint32_t tile_bytes = weights.get_entry_size();

    UnicastEndpoint self;
    const uint32_t self_x = my_x[noc.get_noc_id()];
    const uint32_t self_y = my_y[noc.get_noc_id()];

    // Work items are channel-block-major, so consecutive items on a core share tap weights.
    constexpr uint32_t no_block = ~0U;
    uint32_t loaded_block = no_block;
    for (uint32_t item = 0; item < wi_count; ++item) {
        const uint32_t work = wi_start + item;
        const uint32_t block = work / Mt;
        const uint32_t mt = work % Mt;
        const uint32_t ct_start = block * block_ct;

        if (block != loaded_block) {
            load_weight_block<block_ct>(noc, weights, tap0, tap1, tap2, tap3, tile_bytes, ct_start);
            loaded_block = block;
        }

        // Tile-aligned actual_start and local_rows make the split tile-aligned,
        // so every row in this tile uses the same segment boundary.
        int32_t row_floor = 0;
        if (local_split_row != 0 && mt * tile_height >= local_split_row) {
            row_floor = static_cast<int32_t>(local_split_row);
        }

        for (uint32_t row = 0; row < tile_height + history_rows; ++row) {
            const int32_t source_row =
                static_cast<int32_t>(mt * tile_height + row) - static_cast<int32_t>(history_rows);
            if (source_row < row_floor) {
                const auto read_history = [&](const auto& carry) {
                    noc.async_read(
                        carry,
                        window,
                        block_row_bytes,
                        {.page_id = static_cast<uint32_t>(source_row - row_floor + static_cast<int32_t>(history_rows)),
                         .offset_bytes = ct_start * block_offset_scale},
                        {.offset_bytes = row * block_row_bytes});
                };
                if (initial_from_predecessor || row_floor != 0) {
                    read_history(predecessor_carry);
                } else {
                    read_history(history);
                }
            } else {
                noc.async_read(
                    input,
                    window,
                    block_row_bytes,
                    {.page_id = static_cast<uint32_t>(source_row), .offset_bytes = ct_start * block_offset_scale},
                    {.offset_bytes = row * block_row_bytes});
            }
        }
        noc.async_read_barrier();

        for (uint32_t tap = 0; tap < tap_count; ++tap) {
            activation.reserve_back(block_ct);
            noc.async_read(
                self,
                activation,
                tile_height * block_row_bytes,
                {.noc_x = self_x, .noc_y = self_y, .addr = window_base + tap * block_row_bytes},
                {});
            noc.async_read_barrier();
            activation.push_back(block_ct);
        }
    }
}
