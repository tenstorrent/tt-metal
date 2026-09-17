// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/experimental/kda/chronological_topology/chronology.hpp"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

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
    weights.reserve_back(4 * block_ct);
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
    weights.push_back(4 * block_ct);
}

// Rows of causal history per fragment: one fewer than the four learned taps.
constexpr uint32_t history_rows_per_plane = 3;

template <
    uint32_t block_ct,
    uint32_t num_blocks,
    uint32_t has_wrap_indicator,
    uint32_t dynamic_chronology,
    uint32_t sp_rank,
    uint32_t sp_size,
    uint32_t local_rows>
TT_KERNEL void reader(uint32_t wi_start, uint32_t wi_count, uint32_t wrap_row) {
    const auto input = TensorAccessor(tensor::input);
    const auto history = TensorAccessor(tensor::history);
    const auto tap0 = TensorAccessor(tensor::tap0);
    const auto tap1 = TensorAccessor(tensor::tap1);
    const auto tap2 = TensorAccessor(tensor::tap2);
    const auto tap3 = TensorAccessor(tensor::tap3);
    DataflowBuffer weights(dfb::weights);
    DataflowBuffer activation(dfb::act_rm);
    Noc noc;

    uint32_t device_wrap_row = wrap_row;
    bool initial_from_predecessor = false;
    if constexpr (dynamic_chronology) {
        const auto actual_start = TensorAccessor(tensor::actual_start);
        noc.async_read(
            actual_start, CoreLocalMem<uint32_t>(activation.get_write_ptr()), sizeof(uint32_t), {.page_id = 0}, {});
        noc.async_read_barrier();
        const auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(activation.get_write_ptr());
        const auto topology = kda_chronology::derive(words[0], sp_rank, sp_size, local_rows);
        device_wrap_row = topology.local_split ? topology.head_rows : 0;
        initial_from_predecessor = topology.rank != topology.boundary;
    } else if constexpr (has_wrap_indicator) {
        const auto wrap_indicator = TensorAccessor(tensor::wrap_indicator);
        // The activation buffer has not been queued yet, so its first word is
        // available as temporary reader-local storage for the scalar control.
        noc.async_read(
            wrap_indicator, CoreLocalMem<uint32_t>(activation.get_write_ptr()), sizeof(uint32_t), {.page_id = 0}, {});
        noc.async_read_barrier();
        const auto control = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(activation.get_write_ptr());
        if (control[0] == 0) {
            device_wrap_row = 0;
        }
    }

    const uint32_t tile_bytes = weights.get_entry_size();
    if constexpr (num_blocks == 1) {
        load_weight_block<block_ct>(noc, weights, tap0, tap1, tap2, tap3, tile_bytes, 0);
    }

    constexpr uint32_t tile_width = 32;
    constexpr uint32_t tile_height = 32;
    constexpr uint32_t block_row_bytes = block_ct * tile_width * sizeof(uint16_t);
    constexpr uint32_t block_offset_scale = tile_width * sizeof(uint16_t);
    for (uint32_t item = 0; item < wi_count; ++item) {
        const uint32_t work = wi_start + item;
        const uint32_t mt = work / num_blocks;
        const uint32_t ct_start = (work % num_blocks) * block_ct;

        if constexpr (num_blocks > 1) {
            load_weight_block<block_ct>(noc, weights, tap0, tap1, tap2, tap3, tile_bytes, ct_start);
        }

        // A wrap restarts the causal stream mid-buffer. wrap_row is tile aligned, so
        // this whole 32-row output tile lies on one side of it: rows below the wrap
        // reach back into history plane 0, rows at or above it into plane 1. With
        // wrap_row == 0 the whole branch is skipped.
        int32_t row_floor = 0;
        uint32_t history_plane = 0;
        if (device_wrap_row != 0 && mt * tile_height >= device_wrap_row) {
            row_floor = static_cast<int32_t>(device_wrap_row);
            history_plane = history_rows_per_plane;
        }

        for (uint32_t tap = 0; tap < 4; ++tap) {
            activation.reserve_back(block_ct);
            for (uint32_t row = 0; row < tile_height; ++row) {
                const int32_t source_row = static_cast<int32_t>(mt * tile_height + row + tap) - 3;
                if (source_row < row_floor) {
                    if constexpr (dynamic_chronology) {
                        const auto predecessor = TensorAccessor(tensor::predecessor_carry);
                        if (initial_from_predecessor || row_floor != 0) {
                            noc.async_read(
                                predecessor,
                                activation,
                                block_row_bytes,
                                {.page_id = static_cast<uint32_t>(source_row - row_floor + 3),
                                 .offset_bytes = ct_start * block_offset_scale},
                                {.offset_bytes = row * block_row_bytes});
                        } else {
                            noc.async_read(
                                history,
                                activation,
                                block_row_bytes,
                                {.page_id = static_cast<uint32_t>(source_row + 3),
                                 .offset_bytes = ct_start * block_offset_scale},
                                {.offset_bytes = row * block_row_bytes});
                        }
                    } else {
                        noc.async_read(
                            history,
                            activation,
                            block_row_bytes,
                            {.page_id = static_cast<uint32_t>(source_row - row_floor + 3) + history_plane,
                             .offset_bytes = ct_start * block_offset_scale},
                            {.offset_bytes = row * block_row_bytes});
                    }
                } else {
                    noc.async_read(
                        input,
                        activation,
                        block_row_bytes,
                        {.page_id = static_cast<uint32_t>(source_row), .offset_bytes = ct_start * block_offset_scale},
                        {.offset_bytes = row * block_row_bytes});
                }
            }
            noc.async_read_barrier();
            activation.push_back(block_ct);
        }
    }
}
