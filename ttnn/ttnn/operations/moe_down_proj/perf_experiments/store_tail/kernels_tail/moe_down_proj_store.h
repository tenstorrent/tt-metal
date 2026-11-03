// SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// moe_down_proj store_wave, run by whichever dataflow RISC the host assigns (store_on_reader): per tile row m of
// this core's rows, one untilized block of core_n_tiles tiles -> 32 row segments of y, row 32 * m + i at byte offset
// line * seg_bytes; one flush per block before the pop. A call stores up to max_blocks of the wave's blocks starting
// at block `first` and returns the next unstored block, so a caller can pace one wave's store over several calls.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "moe_down_proj_plan.h"

// The n-lines of an m-group store the same 32 y rows per block, each row a different bank; starting each line's
// segment walk at a different row keeps the group's cores from hitting the same bank at the same time.
#ifndef MOE_STORE_ROT
#define MOE_STORE_ROT 0
#endif
FORCE_INLINE uint32_t moe_store_seg(uint32_t i, uint32_t line) { return (i + line * MOE_STORE_ROT) % 32; }

template <uint32_t cb_output_rows, uint32_t core_n_tiles, typename Accessor>
uint32_t moe_store_wave(
    const MoePlan& plan,
    const Accessor& out_accessor,
    uint32_t wave,
    uint32_t group,
    uint32_t line,
    uint32_t first = 0,
    uint32_t max_blocks = UINT32_MAX) {
    constexpr uint32_t tile_rows = 32;
    constexpr uint32_t seg_bytes = core_n_tiles * tile_rows * sizeof(uint16_t);
    const uint32_t seg_offset = line * seg_bytes;
    const uint32_t row_lo = plan.group_first(wave, group);
    const uint32_t my_rows = plan.group_rows(wave, group);
    const uint32_t last = my_rows - first < max_blocks ? my_rows : first + max_blocks;
    if (last <= first) {
        return last;
    }
    MaybeDeviceZoneScope("y_store");
    for (uint32_t r = first; r < last; ++r) {
        cb_wait_front(cb_output_rows, core_n_tiles);
        const uint32_t block_l1 = get_read_ptr(cb_output_rows);
        const uint32_t first_row = (row_lo + r) * tile_rows;
        for (uint32_t j = 0; j < tile_rows; ++j) {
            const uint32_t i = moe_store_seg(j, line);
            noc_async_write(block_l1 + i * seg_bytes, out_accessor.get_noc_addr(first_row + i, seg_offset), seg_bytes);
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_output_rows, core_n_tiles);
    }
    return last;
}

// Last-wave tail split. Nothing overlaps the last wave's store, so the RISC that owns the store (owner) hands every
// TAIL_SPLIT-th block to the other dataflow RISC (helper), whose writes go out on the other NoC. cb_tail (owner ->
// helper, one 16 B page per handed block: {block L1 address, first y row}); the helper pops a page once that block's
// writes have left L1, so the owner still pops cb_output_rows in order, as its single consumer.
// Raw CB pointer math (fifo_rd_ptr / fifo_limit) instead of get_read_ptr: blocks stay un-popped while the helper
// stores them, so the owner addresses block r at front + r blocks with the CB wrap.
#ifndef MOE_STORE_CMDBUFS
#define MOE_STORE_CMDBUFS 1
#endif
#ifndef MOE_TAIL_SPLIT
#define MOE_TAIL_SPLIT 2
#endif

inline bool moe_tail_is_helper_block(uint32_t r) { return r % MOE_TAIL_SPLIT == MOE_TAIL_SPLIT - 1; }

template <uint32_t core_n_tiles, typename Accessor>
FORCE_INLINE void moe_store_block(const Accessor& out_accessor, uint32_t block_l1, uint32_t first_row, uint32_t line) {
    constexpr uint32_t tile_rows = 32;
    constexpr uint32_t seg_bytes = core_n_tiles * tile_rows * sizeof(uint16_t);
    const uint32_t seg_offset = line * seg_bytes;
#if MOE_STORE_CMDBUFS == 2
    // Raw: alternate the write and write-reg command buffers of this RISC's NoC (nothing else issues on them during
    // the tail), so one segment's request setup overlaps the previous segment's injection. noc_async_write has no
    // command-buffer choice.
    for (uint32_t j = 0; j < tile_rows; ++j) {
        const uint32_t i = moe_store_seg(j, line);
        const uint64_t dst = out_accessor.get_noc_addr(first_row + i, seg_offset);
        const uint32_t buf = (j & 1) ? write_reg_cmd_buf : write_cmd_buf;
        while (!noc_cmd_buf_ready(noc_index, buf));
        ncrisc_noc_fast_write<noc_mode>(
            noc_index, buf, block_l1 + i * seg_bytes, dst, seg_bytes, NOC_UNICAST_WRITE_VC, false, false, 1, true);
    }
#else
    for (uint32_t j = 0; j < tile_rows; ++j) {
        const uint32_t i = moe_store_seg(j, line);
        noc_async_write(block_l1 + i * seg_bytes, out_accessor.get_noc_addr(first_row + i, seg_offset), seg_bytes);
    }
#endif
}

template <uint32_t cb_output_rows, uint32_t cb_tail, uint32_t core_n_tiles, typename Accessor>
void moe_store_tail_owner(
    const MoePlan& plan, const Accessor& out_accessor, uint32_t wave, uint32_t group, uint32_t line) {
    const uint32_t row_lo = plan.group_first(wave, group);
    const uint32_t rows = plan.group_rows(wave, group);
    if (rows == 0) {
        return;
    }
    MaybeDeviceZoneScope("y_store_tail");
    auto& cb = get_local_cb_interface(cb_output_rows);
    const uint32_t block_bytes = core_n_tiles * cb.fifo_page_size;
    const uint32_t depth_blocks = cb.fifo_num_pages / core_n_tiles;
    const uint32_t tail_pages = get_local_cb_interface(cb_tail).fifo_num_pages;
    uint32_t front = 0;
    uint32_t handed = 0;
    uint32_t retired_handed = 0;
    auto retire = [&]() {
        if (moe_tail_is_helper_block(front)) {
            ++retired_handed;
            // Helper popped the first retired_handed pages <=> free space >= tail_pages - (handed - retired_handed).
            cb_reserve_back(cb_tail, tail_pages - (handed - retired_handed));
        } else {
            noc_async_writes_flushed();
        }
        cb_pop_front(cb_output_rows, core_n_tiles);
        ++front;
    };
    for (uint32_t r = 0; r < rows; ++r) {
        if (r - front == depth_blocks) {
            retire();
        }
        cb_wait_front(cb_output_rows, (r - front + 1) * core_n_tiles);
        uint32_t block_l1 = cb.fifo_rd_ptr + (r - front) * block_bytes;
        if (block_l1 >= cb.fifo_limit) {
            block_l1 -= cb.fifo_size;
        }
        const uint32_t first_row = (row_lo + r) * 32;
        if (moe_tail_is_helper_block(r)) {
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
        retire();
    }
}

template <uint32_t cb_tail, uint32_t core_n_tiles, typename Accessor>
void moe_store_tail_helper(
    const MoePlan& plan, const Accessor& out_accessor, uint32_t wave, uint32_t group, uint32_t line) {
    const uint32_t rows = plan.group_rows(wave, group);
    const uint32_t blocks = rows / MOE_TAIL_SPLIT;
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
