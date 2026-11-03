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
// store_issue variants (bench switches): STORE_INC steps (bank, page-in-bank) across the 32 consecutive y pages
// instead of a get_noc_addr per row; STORE_STATE also programs the write command buffer's NOC_CTRL / length once per
// block (noc_write_init_state) and issues each row with noc_write_with_state<SNDl> (src, coord, dst, send). Both are
// raw: no helper / TensorAccessor API steps an interleaved page walk, and no public stateful write varies the
// destination coordinate per issue (noc_async_write_one_packet_with_state fixes it).
#if defined(STORE_INC) || defined(STORE_STATE)
template <uint32_t core_n_tiles, typename Accessor>
FORCE_INLINE void moe_store_block(const Accessor& out_accessor, uint32_t block_l1, uint32_t first_row, uint32_t line) {
    constexpr uint32_t tile_rows = 32;
    constexpr uint32_t seg_bytes = core_n_tiles * tile_rows * sizeof(uint16_t);
    constexpr bool dram = Accessor::is_dram;
    constexpr uint32_t num_banks = dram ? NUM_DRAM_BANKS : NUM_L1_BANKS;
    const auto& gen = static_cast<const InterleavedAddrGen<dram>&>(out_accessor);
    uint32_t page_in_bank = interleaved_addr_gen::get_bank_offset_index<dram>(first_row);
    uint32_t bank = interleaved_addr_gen::get_bank_index<dram>(first_row, page_in_bank);
    uint32_t local = gen.bank_base_address + page_in_bank * gen.aligned_page_size + line * seg_bytes;
    const uint8_t noc = noc_index;
#ifdef STORE_STATE
    while (!noc_cmd_buf_ready(noc, write_cmd_buf));
    noc_write_init_state<write_cmd_buf>(noc, NOC_UNICAST_WRITE_VC);
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_AT_LEN_BE, seg_bytes);
#ifdef STORE_DUAL
    while (!noc_cmd_buf_ready(noc, write_reg_cmd_buf));
    noc_write_init_state<write_reg_cmd_buf>(noc, NOC_UNICAST_WRITE_VC);
    NOC_CMD_BUF_WRITE_REG(noc, write_reg_cmd_buf, NOC_AT_LEN_BE, seg_bytes);
#endif
#ifdef STORE_BULK
    constexpr bool per_write_count = false;
#else
    constexpr bool per_write_count = true;
#endif
#endif
#ifdef STORE_SKIPDST
    // NOC_RET_ADDR_LO only changes at a bank wrap or a per-bank offset step, so rewrite it only then.
    uint32_t last_lo = 0xFFFFFFFF;
#endif
#ifdef STORE_UNROLL
#pragma GCC unroll 4
#endif
    for (uint32_t i = 0; i < tile_rows; ++i) {
        const uint64_t dst = noc_address_backend::bank_address<dram>(
            bank, local + interleaved_addr_gen::get_bank_offset<dram>(bank), noc);
#if defined(STORE_STATE) && defined(STORE_SKIPDST)
        noc_write_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNdl, CQ_NOC_send, CQ_NOC_WAIT, per_write_count>(
            noc, block_l1 + i * seg_bytes, dst);
        if (static_cast<uint32_t>(dst) != last_lo) {
            last_lo = static_cast<uint32_t>(dst);
            NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_LO, last_lo);
        }
        NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
#elif defined(STORE_STATE)
#ifdef STORE_DUAL
        if (i % 2 == 1) {
            noc_write_with_state<noc_mode, write_reg_cmd_buf, CQ_NOC_SNDl, CQ_NOC_SEND, CQ_NOC_WAIT, per_write_count>(
                noc, block_l1 + i * seg_bytes, dst);
        } else
#endif
            noc_write_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDl, CQ_NOC_SEND, CQ_NOC_WAIT, per_write_count>(
                noc, block_l1 + i * seg_bytes, dst);
#else
        noc_async_write<seg_bytes>(block_l1 + i * seg_bytes, dst, seg_bytes);
#endif
        if (++bank == num_banks) {
            bank = 0;
            local += gen.aligned_page_size;
        }
    }
#if defined(STORE_STATE) && defined(STORE_BULK)
    static_assert(noc_mode == DM_DEDICATED_NOC);
    noc_nonposted_writes_num_issued[noc] += tile_rows;
    noc_nonposted_writes_acked[noc] += tile_rows;
#endif
}
#else
template <uint32_t core_n_tiles, typename Accessor>
FORCE_INLINE void moe_store_block(const Accessor& out_accessor, uint32_t block_l1, uint32_t first_row, uint32_t line) {
    constexpr uint32_t tile_rows = 32;
    constexpr uint32_t seg_bytes = core_n_tiles * tile_rows * sizeof(uint16_t);
    const uint32_t seg_offset = line * seg_bytes;
    for (uint32_t i = 0; i < tile_rows; ++i) {
#ifdef STORE_ONEPKT
        noc_async_write<seg_bytes>(
            block_l1 + i * seg_bytes, out_accessor.get_noc_addr(first_row + i, seg_offset), seg_bytes);
#else
        noc_async_write(block_l1 + i * seg_bytes, out_accessor.get_noc_addr(first_row + i, seg_offset), seg_bytes);
#endif
    }
}
#endif

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
#ifdef STORE_LAZY
    // One flush per batch of blocks (up to the CB depth) instead of one per block; blocks stay un-popped meanwhile.
    const auto& cb = get_local_cb_interface(cb_output_rows);
    const uint32_t block_bytes = core_n_tiles * cb.fifo_page_size;
    const uint32_t depth_blocks = cb.fifo_num_pages / core_n_tiles;
    uint32_t r = first;
    while (r < last) {
        const uint32_t batch = last - r < depth_blocks ? last - r : depth_blocks;
        for (uint32_t b = 0; b < batch; ++b) {
            cb_wait_front(cb_output_rows, (b + 1) * core_n_tiles);
            uint32_t block_l1 = cb.fifo_rd_ptr + b * block_bytes;
            if (block_l1 >= cb.fifo_limit) {
                block_l1 -= cb.fifo_size;
            }
            moe_store_block<core_n_tiles>(out_accessor, block_l1, (row_lo + r + b) * 32, line);
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_output_rows, batch * core_n_tiles);
        r += batch;
    }
#else
    for (uint32_t r = first; r < last; ++r) {
        cb_wait_front(cb_output_rows, core_n_tiles);
        moe_store_block<core_n_tiles>(out_accessor, get_read_ptr(cb_output_rows), (row_lo + r) * 32, line);
        noc_async_writes_flushed();
        cb_pop_front(cb_output_rows, core_n_tiles);
    }
#endif
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
