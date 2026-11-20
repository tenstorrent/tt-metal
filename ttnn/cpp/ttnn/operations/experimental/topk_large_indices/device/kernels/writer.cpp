// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "topk_large_indices_runtime_args.hpp"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#ifdef TOPK_TWO_LEVEL
#include "topk_large_indices_two_level.hpp"
#endif

namespace {

template <uint32_t source_slices_per_row, uint32_t output_slices_per_row, uint32_t slice_bytes>
FORCE_INLINE void copy_row_to_scratch(CircularBuffer& src_cb, CircularBuffer& scratch_cb, const Noc& noc) {
    static_assert(source_slices_per_row == 64 || source_slices_per_row == 128);
    static_assert(output_slices_per_row >= 1 && output_slices_per_row <= source_slices_per_row);
    constexpr uint32_t transfer_bytes = slice_bytes;
    static_assert(transfer_bytes <= NOC_MAX_BURST_SIZE);

    const uint32_t src_base = src_cb.get_read_ptr();
    const uint32_t dst_base = scratch_cb.get_write_ptr();
    const uint32_t noc_id = noc.get_noc_id();
    const auto local_src = [noc_id](uint32_t addr) {
        return noc_traits_t<UnicastEndpoint>::src_args_type{
            .noc_x = static_cast<uint32_t>(my_x[noc_id]), .noc_y = static_cast<uint32_t>(my_y[noc_id]), .addr = addr};
    };

    noc.set_async_read_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
        UnicastEndpoint{}, transfer_bytes, local_src(src_base));

    for (uint32_t dst_slice = 0; dst_slice < output_slices_per_row; ++dst_slice) {
        // pack_untilize emits 16-element slices in face-pair order:
        // [top-left, bottom-left, top-right, bottom-right] for each tile column.
        const uint32_t tile_col = dst_slice >> 2;
        const uint32_t face_col = dst_slice & 0x1;
        const uint32_t face_row_offset = (dst_slice & 0x2) ? source_slices_per_row / 2 : 0;
        const uint32_t src_slice = (2 * tile_col) + face_col + face_row_offset;
        const uint32_t src_addr = src_base + src_slice * slice_bytes;
        const uint32_t dst_addr = dst_base + dst_slice * slice_bytes;
        noc.async_read_with_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
            UnicastEndpoint{},
            CoreLocalMem<uint32_t>(dst_addr),
            transfer_bytes,
            local_src(src_addr),
            {.offset_bytes = 0});
    }
    noc.async_read_barrier();
}

template <
    uint32_t source_slices_per_row,
    uint32_t output_slices_per_row,
    uint32_t slice_bytes,
    typename TensorAccessorT>
FORCE_INLINE void issue_reordered_row_write(
    CircularBuffer& src_cb,
    CircularBuffer& scratch_cb,
    const Noc& noc,
    const TensorAccessorT& tensor,
    uint32_t row,
    uint32_t row_bytes) {
    src_cb.wait_front(1);
    scratch_cb.reserve_back(1);
    copy_row_to_scratch<source_slices_per_row, output_slices_per_row, slice_bytes>(src_cb, scratch_cb, noc);
    src_cb.pop_front(1);

    scratch_cb.push_back(1);
    scratch_cb.wait_front(1);
    noc.async_write(scratch_cb, tensor, row_bytes, {.offset_bytes = 0}, {.page_id = row, .offset_bytes = 0});
}

template <typename TensorAccessorT>
FORCE_INLINE void issue_contiguous_row_write(
    CircularBuffer& src_cb, const Noc& noc, const TensorAccessorT& tensor, uint32_t row, uint32_t row_bytes) {
    src_cb.wait_front(1);
    noc.async_write(src_cb, tensor, row_bytes, {.offset_bytes = 0}, {.page_id = row, .offset_bytes = 0});
}

template <
    uint32_t source_slices_per_row,
    uint32_t output_slices_per_row,
    uint32_t slice_bytes,
    typename TensorAccessorT>
FORCE_INLINE void write_row_indices(
    CircularBuffer& indices_cb,
    CircularBuffer& indices_scratch_cb,
    const Noc& noc,
    const TensorAccessorT& indices,
    uint32_t row,
    uint32_t row_bytes) {
    if constexpr (source_slices_per_row == 32) {
        issue_contiguous_row_write(indices_cb, noc, indices, row, row_bytes);
        noc.async_writes_flushed();
        indices_cb.pop_front(1);
    } else {
        issue_reordered_row_write<source_slices_per_row, output_slices_per_row, slice_bytes>(
            indices_cb, indices_scratch_cb, noc, indices, row, row_bytes);
        noc.async_writes_flushed();
        indices_scratch_cb.pop_front(1);
    }
}

#ifdef TOPK_TWO_LEVEL
// Phase 1 of a two-level program (see topk_large_indices_two_level.hpp), one unit: the max reduce leaves each tile's 32
// row maxima in column 0 (rows 0-15 in face 0, 16-31 in face 2). The unit's 32 tiles were tilized as one 32-row
// block, so row r of tile t holds block r * 32 + t of the unit; its 1024 maxima go to the maxima tensor in block
// order. Once they are written, the unit's done word on the row's owner goes up. `cleared`: whether every core has
// cleared its done words for this run, which the first unit waits for.
template <typename MaxOutT>
FORCE_INLINE void two_level_unit(
    const Noc& noc,
    const MaxOutT& max_out,
    uint32_t search_len,
    const topk_two_level::Units& units,
    uint32_t unit,
    bool& cleared) {
    using namespace topk_two_level;
    constexpr uint32_t stage1_batch = get_named_compile_time_arg_val("tl_stage1_batch");
    constexpr uint32_t unit_blocks = stage1_batch * (tile_keys / block_keys);
    CircularBuffer red_cb(get_named_compile_time_arg_val("tl_cb_red"));
    const uint32_t staging = CircularBuffer(get_named_compile_time_arg_val("tl_cb_max")).get_write_ptr();
    const uint32_t num_blocks = div_up(search_len, block_keys);
    const uint32_t row = units.row(unit);
    const uint32_t first_block = units.part(unit) * unit_blocks;
    const uint32_t live = num_blocks - first_block < unit_blocks ? num_blocks - first_block : unit_blocks;
    auto* dense = reinterpret_cast<uint16_t*>(staging);
    for (uint32_t t = 0; t < stage1_batch; ++t) {
        red_cb.wait_front(1);
        // Plain loads, fenced on both sides: after the compute's pack, before the slot is released.
        asm volatile("" ::: "memory");
        const auto* reduced = reinterpret_cast<const uint16_t*>(red_cb.get_read_ptr());
#pragma GCC unroll 16
        for (uint32_t r = 0; r < 16; ++r) {
            dense[r * stage1_batch + t] = reduced[r * 16];
            dense[(r + 16) * stage1_batch + t] = reduced[512 + r * 16];
        }
        asm volatile("" ::: "memory");
        red_cb.pop_front(1);
    }
    // The last maxima stored are those of the last tile's row 31.
    wait_l1_store(reinterpret_cast<uint32_t>(&dense[31 * stage1_batch + stage1_batch - 1]));
    noc.async_write(
        CoreLocalMem<uint16_t>(staging),
        max_out,
        live * sizeof(uint16_t),
        {},
        {.page_id = row, .offset_bytes = first_block * sizeof(uint16_t)});
    noc.async_write_barrier();
    const uint32_t num_cores = get_common_arg_val<uint32_t>(topk_common_args::writer_two_level_num_cores);
    const RowOwner owner = row_owner(
        row, units.num_rows, get_common_arg_val<uint32_t>(topk_common_args::writer_two_level_split_rows), num_cores);
    if (!cleared) {
        // The coordinator counts the cores that have cleared; the staging word is free once the maxima are written.
        const uint32_t coordinator = get_common_arg_val<uint32_t>(topk_common_args::writer_two_level_core_coords);
        const uint64_t cleared_count = get_noc_addr(
            coordinator & 0xFFFFu,
            coordinator >> 16,
            get_semaphore(get_named_compile_time_arg_val("tl_cleared_sem")),
            noc.get_noc_id());
        const auto* landed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(staging);
        do {
            noc_async_read(cleared_count, staging, sizeof(uint32_t), noc.get_noc_id());
            noc_async_read_barrier(noc.get_noc_id());
            invalidate_l1_cache();
        } while (*landed < num_cores);
        cleared = true;
    }
    const uint32_t coords = get_common_arg_val<uint32_t>(topk_common_args::writer_two_level_core_coords + owner.core);
    noc_semaphore_inc(
        get_noc_addr(
            coords & 0xFFFFu,
            coords >> 16,
            CircularBuffer(get_named_compile_time_arg_val("tl_cb_ready_flags")).get_write_ptr() +
                ready_flag_offset(owner.local_row, units.part(unit)),
            noc.get_noc_id()),
        1,
        noc.get_noc_id());
}

// Stage 2 of one of this core's rows: the selected blocks, in any order, become the ascending block table the reader
// gathers by, sentinels last; stage 3's remap reads it too. The table goes out a chunk's page at a
// time, so the reader gathers each chunk as soon as its blocks are known. For a split row the pages of the table's
// second half also go to the partner, which gathers by them.
FORCE_INLINE void two_level_table(const Noc& noc, CircularBuffer& indices_cb, uint32_t search_len, bool split) {
    using namespace topk_two_level;
    CircularBuffer table_cb(get_named_compile_time_arg_val("tl_cb_table"));
    const uint32_t num_blocks = div_up(search_len, block_keys);
    // The partner's remote-table CB sits at the same address as this core's.
    const uint32_t remote_table_addr =
        CircularBuffer(get_named_compile_time_arg_val("tl_cb_remote_table")).get_write_ptr();
    // A byte per block: setting one is a single store, with no read-modify-write to wait on.
    auto* flags =
        reinterpret_cast<uint8_t*>(CircularBuffer(get_named_compile_time_arg_val("tl_cb_block_flags")).get_write_ptr());
    indices_cb.wait_front(1);
    invalidate_l1_cache();
    asm volatile("" ::: "memory");
    const auto* selected = reinterpret_cast<const uint32_t*>(indices_cb.get_read_ptr());
    constexpr uint32_t batch = 8;
    for (uint32_t i = 0; i < selected_blocks; i += batch) {
        // The loads go first, so their latencies overlap.
        uint32_t blocks[batch];
#pragma GCC unroll 8
        for (uint32_t j = 0; j < batch; ++j) {
            blocks[j] = selected[i + j];
        }
#pragma GCC unroll 8
        for (uint32_t j = 0; j < batch; ++j) {
            if (blocks[j] < num_blocks) {
                flags[blocks[j]] = 1;
            }
        }
    }
    asm volatile("" ::: "memory");
    indices_cb.pop_front(1);

    table_cb.reserve_back(table_pages);
    const uint32_t table_addr = table_cb.get_write_ptr();
    auto* table = reinterpret_cast<uint32_t*>(table_addr);
    const uint32_t partner_x = split ? get_arg_val<uint32_t>(topk_core_args::writer_partner_noc_x) : 0;
    const uint32_t partner_y = split ? get_arg_val<uint32_t>(topk_core_args::writer_partner_noc_y) : 0;
    uint32_t pushed = 0;
    const auto publish = [&](uint32_t entries) {
        if ((pushed + 1) * table_page_entries > entries) {
            return;
        }
        // A store to L1 may still be in flight when the NoC or the reader sees the page. The last store was
        // table[entries - 1]: reading it back and using the value waits for the stores to land.
        wait_l1_store(reinterpret_cast<uint32_t>(&table[entries - 1]));
        for (; (pushed + 1) * table_page_entries <= entries; ++pushed) {
            if (split && pushed >= table_pages / 2) {
                const uint32_t offset = pushed * table_page_bytes;
                noc.async_write(
                    CoreLocalMem<uint32_t>(table_addr + offset),
                    UnicastEndpoint{},
                    table_page_bytes,
                    {},
                    {.noc_x = partner_x,
                     .noc_y = partner_y,
                     .addr = remote_table_addr + offset - table_pages / 2 * table_page_bytes});
                // The partner reads the page once half_sem counts it, so the page must have landed.
                noc.async_write_barrier();
                Semaphore<>(get_named_compile_time_arg_val("tl_half_sem")).up(noc, partner_x, partner_y, 1);
            }
            table_cb.push_back(1);
        }
    };
    // Four flag words at a time; each nonzero one is cleared for the next table.
    auto* packed = reinterpret_cast<uint32_t*>(flags);
    uint32_t count = 0;
    const uint32_t words = flag_words(search_len);
    for (uint32_t word = 0; word < words; word += 4) {
        uint32_t set[4];
#pragma GCC unroll 4
        for (uint32_t j = 0; j < 4; ++j) {
            set[j] = packed[word + j];
        }
#pragma GCC unroll 4
        for (uint32_t j = 0; j < 4; ++j) {
            if (set[j] != 0) {
                packed[word + j] = 0;
                do {
                    table[count++] = ((word + j) << 2) + (__builtin_ctz(set[j]) >> 3);
                    set[j] &= set[j] - 1;
                } while (set[j] != 0);
            }
        }
        publish(count);
    }
    for (; count < selected_blocks; ++count) {
        table[count] = sentinel;
    }
    publish(selected_blocks);
}

// Stage 3's output for one of this core's rows: candidate positions in rank order, mapped through the table to key
// indices. Out of line: it has a call site per schedule.
template <uint32_t source_slices_per_row, uint32_t output_slices_per_row, uint32_t slice_bytes, typename IndicesT>
__attribute__((noinline)) void two_level_output(
    const Noc& noc,
    CircularBuffer& indices_cb,
    CircularBuffer& indices_scratch_cb,
    const IndicesT& indices,
    uint32_t row_bytes,
    uint32_t row,
    uint32_t table_addr) {
    using namespace topk_two_level;
    const auto* table = reinterpret_cast<const uint32_t*>(table_addr);
    indices_cb.wait_front(1);
    indices_scratch_cb.reserve_back(1);
    copy_row_to_scratch<source_slices_per_row, output_slices_per_row, slice_bytes>(indices_cb, indices_scratch_cb, noc);
    indices_cb.pop_front(1);
    invalidate_l1_cache();
    asm volatile("" ::: "memory");
    auto* ranked = reinterpret_cast<uint32_t*>(indices_scratch_cb.get_write_ptr());
    constexpr uint32_t ranks = output_slices_per_row * slice_bytes / sizeof(uint32_t);
    constexpr uint32_t batch = 8;
    static_assert(ranks % batch == 0);
    uint32_t last_stored = 0;
    for (uint32_t i = 0; i < ranks; i += batch) {
        // Loads first, so their latencies overlap: the positions, then their blocks.
        uint32_t position[batch];
        uint32_t block[batch];
#pragma GCC unroll 8
        for (uint32_t j = 0; j < batch; ++j) {
            position[j] = ranked[i + j];
        }
#pragma GCC unroll 8
        for (uint32_t j = 0; j < batch; ++j) {
            block[j] = position[j] != sentinel ? table[position[j] / block_keys] : 0;
        }
#pragma GCC unroll 8
        for (uint32_t j = 0; j < batch; ++j) {
            if (position[j] != sentinel) {
                ranked[i + j] = block[j] * block_keys + position[j] % block_keys;
                last_stored = i + j;
            }
        }
    }
    // The stores land before the NoC reads the row.
    wait_l1_store(reinterpret_cast<uint32_t>(&ranked[last_stored]));
    indices_scratch_cb.push_back(1);
    indices_scratch_cb.wait_front(1);
    noc.async_write(indices_scratch_cb, indices, row_bytes, {.offset_bytes = 0}, {.page_id = row, .offset_bytes = 0});
    noc.async_writes_flushed();
    indices_scratch_cb.pop_front(1);
}
#endif

}  // namespace

void kernel_main() {
    const uint32_t indices_addr = get_common_arg_val<uint32_t>(topk_common_args::output_address);
    const uint32_t start_row = get_arg_val<uint32_t>(topk_core_args::writer_start_row);
    const uint32_t num_rows = get_arg_val<uint32_t>(topk_core_args::writer_num_rows);
    constexpr bool row_split = get_named_compile_time_arg_val("row_split") != 0;
    const bool has_split = row_split && get_arg_val<uint32_t>(topk_core_args::writer_has_split) != 0;
    const bool second_half = row_split && get_arg_val<uint32_t>(topk_core_args::writer_second_half) != 0;
    // The second half ships its survivor to the first, which merges it and writes the row.
    const bool receives_survivor = has_split && !second_half;
    const bool sends_survivor = has_split && second_half;
    const uint32_t partner_noc_x = row_split ? get_arg_val<uint32_t>(topk_core_args::writer_partner_noc_x) : 0;
    const uint32_t partner_noc_y = row_split ? get_arg_val<uint32_t>(topk_core_args::writer_partner_noc_y) : 0;
    const uint32_t split_row = row_split ? get_arg_val<uint32_t>(topk_core_args::writer_split_row) : 0;

    constexpr uint32_t cb_indices = get_compile_time_arg_val(0);
    constexpr uint32_t cb_indices_scratch = get_compile_time_arg_val(1);
    constexpr uint32_t indices_page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t source_slices_per_row = get_compile_time_arg_val(3);
    constexpr uint32_t output_slices_per_row = get_compile_time_arg_val(4);
    constexpr uint32_t indices_slice_bytes = get_compile_time_arg_val(5);
    constexpr auto indices_args = TensorAccessorArgs<6>();
    constexpr uint32_t split_args_base = indices_args.next_compile_time_args_offset();
    constexpr uint32_t cb_landing = get_compile_time_arg_val(split_args_base);
    constexpr uint32_t cb_send = get_compile_time_arg_val(split_args_base + 1);
    constexpr uint32_t credit_sem_id = get_compile_time_arg_val(split_args_base + 2);
    constexpr uint32_t data_sem_id = get_compile_time_arg_val(split_args_base + 3);
    constexpr uint32_t survivor_tiles = get_compile_time_arg_val(split_args_base + 4);
    constexpr uint32_t survivor_bytes = get_compile_time_arg_val(split_args_base + 5);

    const auto indices = TensorAccessor(indices_args, indices_addr, indices_page_bytes);
    CircularBuffer indices_cb(cb_indices);
    CircularBuffer indices_scratch_cb(cb_indices_scratch);
    CircularBuffer landing_cb(cb_landing);
    CircularBuffer send_cb(cb_send);
    Semaphore<> credit_sem(credit_sem_id);
    Semaphore<> data_sem(data_sem_id);
    Noc noc;

    // The landing CB is one survivor slot cycled whole, so its base is the same address on every core -- every CB
    // here is created over the whole grid, so the allocator gives each core the same layout.
    const uint32_t landing_base = row_split ? landing_cb.get_write_ptr() : 0;

    const uint32_t num_whole_rows = has_split ? num_rows - 1 : num_rows;
#ifdef TOPK_TWO_LEVEL
    {
        using namespace topk_two_level;
        // The second core of a split pair sends its half's survivor into the first core's landing CB once the first
        // core has credited the slot, then counts it on the first core's data_sem. Out of line: it has a call site per
        // schedule.
        const auto send_survivor = [&]() __attribute__((noinline)) {
            credit_sem.wait_min(1);
            send_cb.wait_front(survivor_tiles);
            noc.async_write(
                send_cb,
                UnicastEndpoint{},
                survivor_bytes,
                {.offset_bytes = 0},
                {.noc_x = partner_noc_x, .noc_y = partner_noc_y, .addr = landing_base});
            noc.async_write_barrier();
            send_cb.pop_front(survivor_tiles);
            data_sem.up(noc, partner_noc_x, partner_noc_y, 1);
        };
        CircularBuffer mailbox_cb(get_named_compile_time_arg_val("tl_cb_writer_mailbox"));
        mailbox_cb.wait_front(1);
        const auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mailbox_cb.get_read_ptr());
        const bool two_level_mode = words[writer_mailbox_mode_word] != 0;
        const uint32_t search_len = words[writer_mailbox_search_length_word];
        mailbox_cb.pop_front(1);
        // Test output: the mode this program ran in. Every core whose rows start at 0 writes it (with split pairs and
        // fewer rows than cores that is several cores); they all write the same value.
        if constexpr (get_named_compile_time_arg_val("tl_mode_output") != 0) {
            const uint32_t mode_addr = get_common_arg_val<uint32_t>(topk_common_args::mode_output_address);
            if (start_row == 0 && num_rows > 0) {
                constexpr auto mode_args = TensorAccessorArgs<get_named_compile_time_arg_val("tl_mode_args_offset")>();
                const auto mode_out = TensorAccessor(mode_args, mode_addr, sizeof(uint32_t));
                CircularBuffer scratch(get_named_compile_time_arg_val("tl_cb_block_flags"));
                const uint32_t word_addr = scratch.get_write_ptr();
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(word_addr) = two_level_mode ? 1u : 0u;
                wait_l1_store(word_addr);
                noc.async_write(CoreLocalMem<uint32_t>(word_addr), mode_out, sizeof(uint32_t), {}, {.page_id = 0});
                noc.async_write_barrier();
            }
        }
        constexpr bool early_halves = get_named_compile_time_arg_val("tl_early_halves") != 0;
        if (two_level_mode && early_halves && receives_survivor) {
            // The landing slot is this pair's only use of it: free from the start, so the partner may send its half's
            // survivor whenever it has it.
            landing_cb.reserve_back(survivor_tiles);
            credit_sem.up(noc, partner_noc_x, partner_noc_y, 1);
        }
        if (two_level_mode) {
            // The block flags start clear; each table's scan clears the flags it finds.
            auto* packed = reinterpret_cast<uint32_t*>(
                CircularBuffer(get_named_compile_time_arg_val("tl_cb_block_flags")).get_write_ptr());
            for (uint32_t word = 0; word < flag_words(search_len); ++word) {
                packed[word] = 0;
            }
            // The reader announces each item: a phase-1 unit, one of this core's rows, its half of a split row, or the
            // end.
            const auto units = Units::of(
                search_len,
                get_named_compile_time_arg_val("tl_stage1_batch"),
                get_common_arg_val<uint32_t>(topk_common_args::writer_two_level_num_input_rows),
                get_common_arg_val<uint32_t>(topk_common_args::writer_two_level_split_rows),
                split_rows_first(early_halves, search_len));
            constexpr auto max_out_args =
                TensorAccessorArgs<get_named_compile_time_arg_val("tl_max_out_args_offset")>();
            const auto max_out = TensorAccessor(
                max_out_args,
                get_common_arg_val<uint32_t>(topk_common_args::two_level_max_address),
                get_common_arg_val<uint32_t>(topk_common_args::writer_two_level_max_row_bytes));
            bool cleared = false;
            while (true) {
                mailbox_cb.wait_front(1);
                const auto* header = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mailbox_cb.get_read_ptr());
                const uint32_t code = header[writer_header_code_word];
                uint32_t row = header[writer_header_row_word];
                mailbox_cb.pop_front(1);
                const bool end = code == header_end;
                if (early_halves && code == header_half) {
                    if (sends_survivor) {
                        send_survivor();
                    }
                    continue;
                }
                if (early_halves && end) {
                    if (!receives_survivor) {
                        break;
                    }
                    data_sem.wait_min(1);
                    landing_cb.push_back(survivor_tiles);
                    // The split row's table, kept from its stage 2 while later rows reused the table CB.
                    two_level_output<source_slices_per_row, output_slices_per_row, indices_slice_bytes>(
                        noc,
                        indices_cb,
                        indices_scratch_cb,
                        indices,
                        indices_page_bytes,
                        split_row,
                        CircularBuffer(get_named_compile_time_arg_val("tl_cb_split_table")).get_write_ptr());
                    break;
                }
                if (end) {
                    // The split row: the second core sends its half's survivor into the first core's landing CB,
                    // as the single-level split does; the first core merges it and writes the row below.
                    if (sends_survivor) {
                        send_survivor();
                    }
                    if (!receives_survivor) {
                        break;
                    }
                    landing_cb.reserve_back(survivor_tiles);
                    credit_sem.up(noc, partner_noc_x, partner_noc_y, 1);
                    data_sem.wait_min(1);
                    landing_cb.push_back(survivor_tiles);
                    row = split_row;
                } else if (code == header_row || code == header_split) {
                    two_level_table(noc, indices_cb, search_len, code == header_split);
                    if (code == header_split) {
                        if constexpr (early_halves) {
                            // Kept for the row's output at the end: the table CB holds the next row's table by then.
                            const uint32_t from =
                                CircularBuffer(get_named_compile_time_arg_val("tl_cb_table")).get_write_ptr();
                            const uint32_t to =
                                CircularBuffer(get_named_compile_time_arg_val("tl_cb_split_table")).get_write_ptr();
                            noc_async_read(
                                get_noc_addr(from, noc.get_noc_id()),
                                to,
                                table_pages * table_page_bytes,
                                noc.get_noc_id());
                            noc_async_read_barrier(noc.get_noc_id());
                        }
                        continue;
                    }
                } else {
                    two_level_unit(noc, max_out, search_len, units, code, cleared);
                    continue;
                }
                // The row's block table: the table CB cycles a whole table at a time, so it starts at the CB's base,
                // and the next table is built only after this output.
                two_level_output<source_slices_per_row, output_slices_per_row, indices_slice_bytes>(
                    noc,
                    indices_cb,
                    indices_scratch_cb,
                    indices,
                    indices_page_bytes,
                    row,
                    CircularBuffer(get_named_compile_time_arg_val("tl_cb_table")).get_write_ptr());
                if (end) {
                    break;
                }
            }
            noc.async_write_barrier();
            noc.async_atomic_barrier();
            return;
        }
    }
#endif
    for (uint32_t local_row = 0; local_row < num_rows; ++local_row) {
        const bool is_split = has_split && local_row == num_whole_rows;
        const uint32_t row = is_split ? split_row : start_row + local_row;

        if (is_split && receives_survivor) {
            // The slot is free once the compute has consumed it: tell the partner to write.
            landing_cb.reserve_back(survivor_tiles);
            credit_sem.up(noc, partner_noc_x, partner_noc_y, 1);
            // The partner increments data_sem only after its write is drained, so the slot is complete here.
            data_sem.wait_min(1);
            landing_cb.push_back(survivor_tiles);
        }

        if (is_split && sends_survivor) {
            credit_sem.wait_min(1);
            send_cb.wait_front(survivor_tiles);
            noc.async_write(
                send_cb,
                UnicastEndpoint{},
                survivor_bytes,
                {.offset_bytes = 0},
                {.noc_x = partner_noc_x, .noc_y = partner_noc_y, .addr = landing_base});
            noc.async_write_barrier();
            send_cb.pop_front(survivor_tiles);
            data_sem.up(noc, partner_noc_x, partner_noc_y, 1);
            noc.async_atomic_barrier();
        } else {
            write_row_indices<source_slices_per_row, output_slices_per_row, indices_slice_bytes>(
                indices_cb, indices_scratch_cb, noc, indices, row, indices_page_bytes);
        }
    }

    noc.async_write_barrier();
    if constexpr (row_split) {
        noc.async_atomic_barrier();
    }
}
