// SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// moe_down_proj store_wave, run by whichever dataflow RISC the host assigns (store_on_reader): per tile row m of
// this core's rows, one untilized block of core_n_tiles tiles -> 32 row segments of y, row 32 * m + i at byte offset
// line * seg_bytes; one flush per block before the pop. A call stores up to max_blocks of the wave's blocks starting
// at block `first` and returns the next unstored block, so a caller can pace one wave's store over several calls.
// The last wave goes through moe_store_tail_owner / moe_store_tail_helper (below) instead.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "moe_down_proj_plan.h"

// One untilized block (tile row) -> its 32 y row segments.
template <uint32_t core_n_tiles, typename Accessor>
FORCE_INLINE void moe_store_block(const Accessor& out_accessor, uint32_t block_l1, uint32_t first_row, uint32_t line) {
    constexpr uint32_t tile_rows = 32;
    constexpr uint32_t seg_bytes = core_n_tiles * tile_rows * sizeof(uint16_t);
    const uint32_t seg_offset = line * seg_bytes;
    for (uint32_t i = 0; i < tile_rows; ++i) {
#ifndef ABL_NO_STORE
        noc_async_write(block_l1 + i * seg_bytes, out_accessor.get_noc_addr(first_row + i, seg_offset), seg_bytes);
#endif
    }
}

template <uint32_t cb_output_rows, uint32_t core_n_tiles, typename Accessor>
uint32_t moe_store_wave(
    const MoePlan& plan,
    const Accessor& out_accessor,
    uint32_t wave,
    uint32_t group,
    uint32_t line,
    uint32_t first = 0,
    uint32_t max_blocks = UINT32_MAX) {
    const uint32_t row_lo = plan.group_first(wave, group);
    const uint32_t my_rows = plan.group_rows(wave, group);
    const uint32_t last = my_rows - first < max_blocks ? my_rows : first + max_blocks;
    if (last <= first) {
        return last;
    }
    MaybeDeviceZoneScope("y_store");
    for (uint32_t r = first; r < last; ++r) {
        cb_wait_front(cb_output_rows, core_n_tiles);
        moe_store_block<core_n_tiles>(out_accessor, get_read_ptr(cb_output_rows), (row_lo + r) * 32, line);
        noc_async_writes_flushed();
        cb_pop_front(cb_output_rows, core_n_tiles);
    }
    return last;
}

// The last wave's store has nothing left to overlap, so the storing RISC (owner) hands every odd block to the other
// dataflow RISC (helper), whose writes go out on the other NoC: the tail runs on both NoCs. cb_tail (owner -> helper)
// carries one {block L1 address, first y row} page per handed block; the helper pops it once the block's writes have
// left L1, so the owner, still the only consumer of cb_output_rows, pops the blocks in order. Handed blocks stay
// un-popped meanwhile, so block r sits at front + r blocks (CB wrap applied by hand; get_read_ptr only gives front).
template <uint32_t cb_output_rows, uint32_t cb_tail, uint32_t core_n_tiles, typename Accessor>
void moe_store_tail_owner(
    const MoePlan& plan, const Accessor& out_accessor, uint32_t wave, uint32_t group, uint32_t line) {
    const uint32_t row_lo = plan.group_first(wave, group);
    const uint32_t rows = plan.group_rows(wave, group);
    if (rows == 0) {
        return;
    }
    MaybeDeviceZoneScope("y_store_tail");
    const auto& cb = get_local_cb_interface(cb_output_rows);
    const uint32_t block_bytes = core_n_tiles * cb.fifo_page_size;
    const uint32_t depth_blocks = cb.fifo_num_pages / core_n_tiles;
    const uint32_t tail_pages = get_local_cb_interface(cb_tail).fifo_num_pages;
    uint32_t front = 0;
    uint32_t handed = 0;
    uint32_t handed_retired = 0;
    auto retire_front = [&]() {
        if (front % 2 == 1) {
            // The helper has popped the first handed_retired pages once this much of cb_tail is free.
            ++handed_retired;
            cb_reserve_back(cb_tail, tail_pages - (handed - handed_retired));
        } else {
            noc_async_writes_flushed();
        }
        cb_pop_front(cb_output_rows, core_n_tiles);
        ++front;
    };
    for (uint32_t r = 0; r < rows; ++r) {
        if (r - front == depth_blocks) {
            retire_front();
        }
        cb_wait_front(cb_output_rows, (r - front + 1) * core_n_tiles);
        uint32_t block_l1 = cb.fifo_rd_ptr + (r - front) * block_bytes;
        if (block_l1 >= cb.fifo_limit) {
            block_l1 -= cb.fifo_size;
        }
        const uint32_t first_row = (row_lo + r) * 32;
        if (r % 2 == 1) {
            cb_reserve_back(cb_tail, 1);
            volatile tt_l1_ptr uint32_t* page = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_tail));
            page[0] = block_l1;
            page[1] = first_row;
            cb_push_back(cb_tail, 1);
            ++handed;
        } else {
            moe_store_block<core_n_tiles>(out_accessor, block_l1, first_row, line);
        }
    }
    while (front < rows) {
        retire_front();
    }
}

template <uint32_t cb_tail, uint32_t core_n_tiles, typename Accessor>
void moe_store_tail_helper(
    const MoePlan& plan, const Accessor& out_accessor, uint32_t wave, uint32_t group, uint32_t line) {
    const uint32_t blocks = plan.group_rows(wave, group) / 2;
    if (blocks == 0) {
        return;
    }
    MaybeDeviceZoneScope("y_store_tail_helper");
    for (uint32_t b = 0; b < blocks; ++b) {
        cb_wait_front(cb_tail, 1);
        volatile tt_l1_ptr uint32_t* page = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_tail));
        moe_store_block<core_n_tiles>(out_accessor, page[0], page[1], line);
        noc_async_writes_flushed();
        cb_pop_front(cb_tail, 1);
    }
}
