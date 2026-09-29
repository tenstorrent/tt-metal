// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/experimental/kda/chronological_selections/device/kernels/chronology.hpp"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
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

// UnicastEndpoint source arguments for a read of this core's own L1 (a NoC loopback read). NOC 0 and NOC 1
// have different coordinate spaces, so the coordinates come from the NoC the read is issued on.
FORCE_INLINE auto self_l1(uint32_t addr, uint8_t noc_id) {
    return noc_traits_t<UnicastEndpoint>::src_args_type{.noc_x = my_x[noc_id], .noc_y = my_y[noc_id], .addr = addr};
}

// Work item `work` is (channel block blk = work / Mt, tile row mt = work % Mt). distribute_prep hands every core a
// contiguous range, so a core walks consecutive tile rows of ONE block and crosses a block boundary at most once
// per Mt items. That ordering is what lets this reader:
//   - load a block's four tap tiles once per block per core (not once per item), and
//   - read each tile row's 32 activation rows from DRAM once, staging them with their three look-back rows and
//     handing the compute its four tap-shifted 32-row windows as local L1 copies (window t = stage rows t..t+31);
//     when the previous item was the previous tile row of the same block, the look-back rows are the previous
//     stage's last three rows, another local copy.
// The compute sees, per tap, exactly the 32 rows it saw before (tap t <-> source rows mt*32 + row + t - 3), so
// its tilize and arithmetic are unchanged and the outputs are bit-identical.
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
    // Reader-private staging of one tile row plus its look-back: 35 rows of block_row_bytes, addressed directly
    // (never pushed or popped).
    DataflowBuffer stage(dfb::act_stage);
    Noc noc;
    UnicastEndpoint self_ep;

    uint32_t local_split_row = 0;
    bool initial_from_predecessor = false;
    {
        const auto actual_start = TensorAccessor(tensor::actual_start);
        noc.async_read(
            actual_start, CoreLocalMem<uint32_t>(activation.get_write_ptr()), sizeof(uint32_t), {.page_id = 0}, {});
        noc.async_read_barrier();
        const auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(activation.get_write_ptr());
        const auto topology = kda_chronology::derive(words[0], sp_rank, sp_size, local_rows);
        local_split_row = topology.local_split ? topology.head_rows : 0;
        initial_from_predecessor = topology.rank != topology.first_rank;
    }

    constexpr uint32_t tile_width = tt::constants::TILE_WIDTH;
    constexpr uint32_t tile_height = tt::constants::TILE_HEIGHT;
    constexpr uint32_t history_rows = 3;
    constexpr uint32_t block_row_bytes = block_ct * tile_width * sizeof(uint16_t);
    constexpr uint32_t block_offset_scale = tile_width * sizeof(uint16_t);
    constexpr uint32_t window_bytes = tile_height * block_row_bytes;
    constexpr uint32_t no_block = 0xFFFFFFFFu;
    const uint32_t tile_bytes = weights.get_entry_size();
    const uint32_t stage_addr = stage.get_write_ptr();
    const uint8_t noc_id = noc.get_noc_id();

    uint32_t cur_blk = no_block;
    uint32_t prev_mt = no_block;  // tile row of the previous item when it belonged to the current block
    for (uint32_t item = 0; item < wi_count; ++item) {
        const uint32_t work = wi_start + item;
        const uint32_t blk = work / Mt;
        const uint32_t mt = work % Mt;
        const uint32_t ct_start = blk * block_ct;

        if (blk != cur_blk) {
            // The weights DFB holds one block, so this reserve waits until the compute has popped the previous
            // block's taps, which it does at the start of its first item of the new block. Every activation
            // window of the previous block was pushed before this point, so nothing the compute needs waits on us.
            load_weight_block<block_ct>(noc, weights, tap0, tap1, tap2, tap3, tile_bytes, ct_start);
            cur_blk = blk;
            prev_mt = no_block;
        }

        // Tile-aligned actual_start and local_rows make the split tile-aligned,
        // so every row in this tile uses the same segment boundary.
        int32_t row_floor = 0;
        if (local_split_row != 0 && mt * tile_height >= local_split_row) {
            row_floor = static_cast<int32_t>(local_split_row);
        }
        const int32_t first_row = static_cast<int32_t>(mt * tile_height) - static_cast<int32_t>(history_rows);

        // Stage rows 0..2: the three tokens before this tile. row_floor is tile-aligned and never above mt*32, so
        // the three rows agree on their source: a segment start (mt == 0, or the local split at row_floor) reads
        // the incoming history, or the predecessor's carry on later ranks and at the split, exactly as before.
        if (first_row < row_floor) {
            const auto read_history = [&](const auto& carry) {
                for (uint32_t j = 0; j < history_rows; ++j) {
                    noc.async_read(
                        carry,
                        CoreLocalMem<uint16_t>(stage_addr + j * block_row_bytes),
                        block_row_bytes,
                        {.page_id = static_cast<uint32_t>(first_row + static_cast<int32_t>(j) - row_floor + 3),
                         .offset_bytes = ct_start * block_offset_scale},
                        {});
                }
            };
            if (initial_from_predecessor || row_floor != 0) {
                read_history(predecessor_carry);
            } else {
                read_history(history);
            }
        } else if (prev_mt != no_block && prev_mt + 1 == mt) {
            // The previous item was tile row mt-1 of this block: its rows 29..31 sit in stage rows 32..34. Copy
            // them down before the DRAM reads below overwrite them, hence the barrier.
            noc.async_read(
                self_ep,
                CoreLocalMem<uint16_t>(stage_addr),
                history_rows * block_row_bytes,
                self_l1(stage_addr + tile_height * block_row_bytes, noc_id),
                {});
            noc.async_read_barrier();
        } else {
            for (uint32_t j = 0; j < history_rows; ++j) {
                noc.async_read(
                    input,
                    CoreLocalMem<uint16_t>(stage_addr + j * block_row_bytes),
                    block_row_bytes,
                    {.page_id = static_cast<uint32_t>(first_row) + j, .offset_bytes = ct_start * block_offset_scale},
                    {});
            }
        }
        // Stage rows 3..34: this tile row's 32 activation rows, read from DRAM once.
        for (uint32_t row = 0; row < tile_height; ++row) {
            noc.async_read(
                input,
                CoreLocalMem<uint16_t>(stage_addr + (history_rows + row) * block_row_bytes),
                block_row_bytes,
                {.page_id = mt * tile_height + row, .offset_bytes = ct_start * block_offset_scale},
                {});
        }
        noc.async_read_barrier();

        // The four tap-shifted windows the compute tilizes, one contiguous local copy each.
        for (uint32_t tap = 0; tap < 4; ++tap) {
            activation.reserve_back(block_ct);
            noc.async_read(
                self_ep,
                activation,
                window_bytes,
                self_l1(stage_addr + tap * block_row_bytes, noc_id),
                {.offset_bytes = 0});
            noc.async_read_barrier();
            activation.push_back(block_ct);
        }
        prev_mt = mt;
    }
}
