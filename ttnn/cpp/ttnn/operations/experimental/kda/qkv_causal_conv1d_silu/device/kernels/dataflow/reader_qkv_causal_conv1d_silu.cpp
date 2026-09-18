// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

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

template <uint32_t block_ct, uint32_t num_blocks>
TT_KERNEL void reader(uint32_t wi_start, uint32_t wi_count) {
    const auto input = TensorAccessor(tensor::input);
    const auto history = TensorAccessor(tensor::history);
    const auto tap0 = TensorAccessor(tensor::tap0);
    const auto tap1 = TensorAccessor(tensor::tap1);
    const auto tap2 = TensorAccessor(tensor::tap2);
    const auto tap3 = TensorAccessor(tensor::tap3);
    DataflowBuffer weights(dfb::weights);
    DataflowBuffer activation(dfb::act_rm);
    Noc noc;

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

        for (uint32_t tap = 0; tap < 4; ++tap) {
            activation.reserve_back(block_ct);
            for (uint32_t row = 0; row < tile_height; ++row) {
                const int32_t source_row = static_cast<int32_t>(mt * tile_height + row + tap) - 3;
                if (source_row < 0) {
                    noc.async_read(
                        history,
                        activation,
                        block_row_bytes,
                        {.page_id = static_cast<uint32_t>(source_row + 3),
                         .offset_bytes = ct_start * block_offset_scale},
                        {.offset_bytes = row * block_row_bytes});
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
