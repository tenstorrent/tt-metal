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
// Two-level rows (see topk_large_indices_two_level.hpp): gather each row's block maxima into the stage-2 chunks,
// turn the stage-2 selection into the ascending block table the reader gathers by, and map the stage-3 candidate
// positions back to key indices.
template <
    uint32_t source_slices_per_row,
    uint32_t output_slices_per_row,
    uint32_t slice_bytes,
    typename IndicesT,
    typename MaxOutT>
FORCE_INLINE void two_level_rows(
    const Noc& noc,
    CircularBuffer& indices_cb,
    CircularBuffer& indices_scratch_cb,
    const IndicesT& indices,
    const MaxOutT& max_out,
    uint32_t row_bytes,
    uint32_t search_len,
    uint32_t start_row,
    uint32_t num_whole_rows,
    bool takes_split_row,
    uint32_t split_row) {
    using namespace topk_two_level;
    constexpr bool has_block_max = get_named_compile_time_arg_val("tl_block_max_input") != 0;
    constexpr bool has_max_out = get_named_compile_time_arg_val("tl_block_max_output") != 0;
    constexpr uint32_t page_bytes = get_named_compile_time_arg_val("tl_page_bytes");
    constexpr uint32_t chunk_keys = get_named_compile_time_arg_val("tl_chunk_keys");
    CircularBuffer red_cb(get_named_compile_time_arg_val("tl_cb_red"));
    Semaphore<> max_ready_sem(get_named_compile_time_arg_val("tl_max_ready_sem"));
    // The maxima scratch: never pushed, so its write pointer is its base on every RISC.
    const uint32_t max_scratch = CircularBuffer(get_named_compile_time_arg_val("tl_cb_max")).get_write_ptr();
    CircularBuffer table_cb(get_named_compile_time_arg_val("tl_cb_table"));
    CircularBuffer bitmap_cb(get_named_compile_time_arg_val("tl_cb_bitmap"));
    CircularBuffer own_table_cb(get_named_compile_time_arg_val("tl_cb_own_table"));

    constexpr uint32_t stage1_batch = get_named_compile_time_arg_val("tl_stage1_batch");
    const uint32_t num_blocks = div_up(search_len, block_keys);
    const uint32_t stage1_tiles = div_up(div_up(search_len, tile_keys), stage1_batch) * stage1_batch;
    const uint32_t bitmap_words = div_up(num_blocks, 32);
    auto* bitmap = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bitmap_cb.get_write_ptr());
    auto* own_table = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(own_table_cb.get_write_ptr());
    const uint32_t num_rows = num_whole_rows + (takes_split_row ? 1 : 0);

    for (uint32_t local_row = 0; local_row < num_rows; ++local_row) {
        const uint32_t row = local_row < num_whole_rows ? start_row + local_row : split_row;

        if constexpr (!has_block_max) {
            // Stage 1: the max reduce leaves each tile's 32 row maxima in column 0 (rows 0-15 in face 0, 16-31 in
            // face 2). They go to the maxima scratch in block order; the reader streams them to the compute.
            auto* dense = reinterpret_cast<uint16_t*>(max_scratch);
            for (uint32_t tile = 0; tile < stage1_tiles; ++tile) {
                red_cb.wait_front(1);
                // Plain loads: the wait above orders them after the compute's pack.
                asm volatile("" ::: "memory");
                const auto* reduced = reinterpret_cast<const uint16_t*>(red_cb.get_read_ptr());
                // A batch is tilized as one 32-row block, stage1_batch tiles wide: row r of tile t holds the
                // 32 keys of block (batch start) + r * stage1_batch + t.
                const uint32_t first =
                    (tile / stage1_batch) * stage1_batch * (tile_keys / block_keys) + tile % stage1_batch;
                if (first + 31 * stage1_batch < num_blocks) {
#pragma GCC unroll 16
                    for (uint32_t r = 0; r < 16; ++r) {
                        dense[first + r * stage1_batch] = reduced[r * 16];
                        dense[first + (r + 16) * stage1_batch] = reduced[512 + r * 16];
                    }
                } else {
                    for (uint32_t r = 0; r < 32; ++r) {
                        const uint32_t block = first + r * stage1_batch;
                        if (block < num_blocks) {
                            dense[block] = reduced[r < 16 ? r * 16 : 512 + (r - 16) * 16];
                        }
                    }
                }
                red_cb.pop_front(1);
            }
            // The maxima are read over the NoC next (test output, reader): keep the stores ahead of it.
            asm volatile("" ::: "memory");
            if constexpr (has_max_out) {
                if (num_blocks > 0) {
                    noc.async_write(
                        CoreLocalMem<uint16_t>(max_scratch),
                        max_out,
                        num_blocks * sizeof(uint16_t),
                        {},
                        {.page_id = row});
                    noc.async_write_barrier();
                }
            }
            // The reader copies the maxima out of the scratch before this row's stage 3, so the next row cannot
            // overwrite them early.
            max_ready_sem.up(noc, my_x[noc.get_noc_id()], my_y[noc.get_noc_id()], 1);
            noc.async_atomic_barrier();
        }

        // Stage 2: the selected blocks, in any order, become an ascending table with sentinels last.
        for (uint32_t word = 0; word < bitmap_words; ++word) {
            bitmap[word] = 0;
        }
        indices_cb.wait_front(1);
        const auto* selected = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(indices_cb.get_read_ptr());
        for (uint32_t i = 0; i < selected_blocks; ++i) {
            const uint32_t block = selected[i];
            if (block < num_blocks) {
                bitmap[block >> 5] |= 1u << (block & 31);
            }
        }
        indices_cb.pop_front(1);
        table_cb.reserve_back(1);
        auto* table = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(table_cb.get_write_ptr());
        uint32_t count = 0;
        for (uint32_t word = 0; word < bitmap_words; ++word) {
            uint32_t bits = bitmap[word];
            while (bits != 0) {
                const uint32_t bit = __builtin_ctz(bits);
                bits &= bits - 1;
                const uint32_t block = (word << 5) + bit;
                table[count] = block;
                own_table[count] = block;
                ++count;
            }
        }
        for (; count < selected_blocks; ++count) {
            table[count] = sentinel;
            own_table[count] = sentinel;
        }
        table_cb.push_back(1);

        // Stage 3: candidate positions in rank order, mapped through the table to key indices.
        indices_cb.wait_front(1);
        indices_scratch_cb.reserve_back(1);
        copy_row_to_scratch<source_slices_per_row, output_slices_per_row, slice_bytes>(
            indices_cb, indices_scratch_cb, noc);
        indices_cb.pop_front(1);
        auto* ranked = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(indices_scratch_cb.get_write_ptr());
        for (uint32_t i = 0; i < output_slices_per_row * slice_bytes / sizeof(uint32_t); ++i) {
            const uint32_t position = ranked[i];
            if (position != sentinel) {
                ranked[i] = own_table[position / block_keys] * block_keys + position % block_keys;
            }
        }
        indices_scratch_cb.push_back(1);
        indices_scratch_cb.wait_front(1);
        noc.async_write(
            indices_scratch_cb, indices, row_bytes, {.offset_bytes = 0}, {.page_id = row, .offset_bytes = 0});
        noc.async_writes_flushed();
        indices_scratch_cb.pop_front(1);
    }
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
                CircularBuffer scratch(get_named_compile_time_arg_val("tl_cb_bitmap"));
                const uint32_t word_addr = scratch.get_write_ptr();
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(word_addr) = two_level_mode ? 1u : 0u;
                noc.async_write(CoreLocalMem<uint32_t>(word_addr), mode_out, sizeof(uint32_t), {}, {.page_id = 0});
                noc.async_write_barrier();
            }
        }
        if (two_level_mode) {
            constexpr auto max_out_args =
                TensorAccessorArgs<get_named_compile_time_arg_val("tl_max_out_args_offset")>();
            const uint32_t max_out_addr = get_common_arg_val<uint32_t>(topk_common_args::block_max_output_address);
            const auto max_out =
                TensorAccessor(max_out_args, max_out_addr, get_named_compile_time_arg_val("tl_block_max_row_bytes"));
            two_level_rows<source_slices_per_row, output_slices_per_row, indices_slice_bytes>(
                noc,
                indices_cb,
                indices_scratch_cb,
                indices,
                max_out,
                indices_page_bytes,
                search_len,
                start_row,
                num_whole_rows,
                has_split && !second_half,
                split_row);
            noc.async_write_barrier();
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
