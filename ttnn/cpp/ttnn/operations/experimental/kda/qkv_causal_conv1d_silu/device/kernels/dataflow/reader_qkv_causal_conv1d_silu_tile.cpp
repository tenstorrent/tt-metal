// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// TILE-layout reader for qkv_causal_conv1d_silu.
//
// The ROW_MAJOR reader gathers, per (tile-row, channel block, tap), 32 shifted row-major sticks so the
// compute kernel can tilize them. That costs one full re-read of the activation per tap (4x the input
// bytes) and forces the caller to untilize the projection output first. This variant instead reads the
// activation in its native TILE layout -- two whole tile pages per output tile (the current tile-row and
// the one before it, or the history tile-row for the first) -- and leaves the row shift to the compute
// kernel, which applies it as a matmul against constant 0/1 shift matrices built here.
//
// Shift matrices, laid out as 9 tiles in the `shift` DFB (d = 1..3 rows of shift):
//   tiles 0..2   S_cur[d] : S[r][r - d]      = 1 for r in [d, 31]   -- rows taken from the current tile
//   tiles 3..5   S_prev[d]: S[r][32 - d + r] = 1 for r in [0, d)    -- rows taken from the previous tile
//   tiles 6..8   S_hist[d]: S[r][r + 3 - d]  = 1 for r in [0, d)    -- rows taken from the history tile
// The history tensor is [1, 3, C] in TILE layout, so its three carry rows sit at tile rows 0, 1, 2 --
// hence the separate S_hist set for the first tile-row instead of reusing S_prev.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

namespace {

// bf16 bit pattern for 1.0.
constexpr uint16_t bf16_one = 0x3F80;
// A 32x32 bf16 tile is 4 faces of 16x16 in face order (0,0), (0,1), (1,0), (1,1); each face is 512 bytes,
// row-major, 32 bytes per face row. Face height/width from tt_metal/api/tt-metalium/constants.hpp:16-18;
// the 512-byte face stride matches ttnn/cpp/ttnn/kernel/dataflow/generate_reduce_scaler.hpp:32.
constexpr uint32_t face_bytes = 512;
constexpr uint32_t face_row_bytes = 32;

FORCE_INLINE uint32_t tile_element_offset(uint32_t row, uint32_t col) {
    const uint32_t face = ((row >> 4) << 1) | (col >> 4);
    return face * face_bytes + (row & 15u) * face_row_bytes + (col & 15u) * sizeof(uint16_t);
}

// Zero the 9 shift tiles, then set the ones. Must run before any other NoC traffic on this kernel:
// async_write_zeros drives the read state machine (a loopback read from MEM_ZEROS_BASE) and its barrier
// must not have unrelated transfers interleaved.
FORCE_INLINE void build_shift_tiles(Noc& noc, DataflowBuffer& shift, uint32_t tile_bytes) {
    constexpr uint32_t shift_tile_count = 9;
    shift.reserve_back(shift_tile_count);
    const uint32_t base = shift.get_write_ptr();
    noc.async_write_zeros(shift, shift_tile_count * tile_bytes);
    noc.write_zeros_l1_barrier();

    for (uint32_t d = 1; d <= 3; ++d) {
        const uint32_t cur_base = (d - 1) * tile_bytes;
        const uint32_t prev_base = (3 + d - 1) * tile_bytes;
        const uint32_t hist_base = (6 + d - 1) * tile_bytes;
        for (uint32_t row = d; row < 32; ++row) {
            *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + cur_base + tile_element_offset(row, row - d)) =
                bf16_one;
        }
        for (uint32_t row = 0; row < d; ++row) {
            *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + prev_base + tile_element_offset(row, 32 - d + row)) =
                bf16_one;
            *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + hist_base + tile_element_offset(row, row + 3 - d)) =
                bf16_one;
        }
    }
    shift.push_back(shift_tile_count);
}

}  // namespace

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
    DataflowBuffer activation(dfb::act_tile);
    DataflowBuffer shift(dfb::shift);
    Noc noc;

    const uint32_t tile_bytes = weights.get_entry_size();
    // Total channel tiles per tile-row of the activation; the input's tile page stride between tile-rows.
    constexpr uint32_t channel_tiles = block_ct * num_blocks;

    build_shift_tiles(noc, shift, tile_bytes);

    if constexpr (num_blocks == 1) {
        load_weight_block<block_ct>(noc, weights, tap0, tap1, tap2, tap3, tile_bytes, 0);
    }

    for (uint32_t item = 0; item < wi_count; ++item) {
        const uint32_t work = wi_start + item;
        const uint32_t mt = work / num_blocks;
        const uint32_t ct_start = (work % num_blocks) * block_ct;

        if constexpr (num_blocks > 1) {
            load_weight_block<block_ct>(noc, weights, tap0, tap1, tap2, tap3, tile_bytes, ct_start);
        }

        // Entries [0, block_ct) hold the previous tile-row, [block_ct, 2 * block_ct) the current one.
        activation.reserve_back(2 * block_ct);
        if (mt == 0) {
            // history is [1, 3, C] TILE: one tile-row of `channel_tiles` pages, carry rows at tile rows 0-2.
            for (uint32_t ct = 0; ct < block_ct; ++ct) {
                noc.async_read(
                    history, activation, tile_bytes, {.page_id = ct_start + ct}, {.offset_bytes = ct * tile_bytes});
            }
        } else {
            for (uint32_t ct = 0; ct < block_ct; ++ct) {
                noc.async_read(
                    input,
                    activation,
                    tile_bytes,
                    {.page_id = (mt - 1) * channel_tiles + ct_start + ct},
                    {.offset_bytes = ct * tile_bytes});
            }
        }
        for (uint32_t ct = 0; ct < block_ct; ++ct) {
            noc.async_read(
                input,
                activation,
                tile_bytes,
                {.page_id = mt * channel_tiles + ct_start + ct},
                {.offset_bytes = (block_ct + ct) * tile_bytes});
        }
        noc.async_read_barrier();
        activation.push_back(2 * block_ct);
    }
}
