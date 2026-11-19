// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "api/core_local_mem.h"
// Shared metadata read invalidates the reused L1 address before loading the next value.
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/metadata_scalar_read.hpp"
#include "topk_large_indices_metadata.hpp"
#include "topk_large_indices_runtime_args.hpp"
#ifdef TOPK_TWO_LEVEL
#include "api/dataflow/noc_semaphore.h"
#include "topk_large_indices_two_level.hpp"

namespace {

constexpr uint16_t neginf_bf16 = 0xFF80;

FORCE_INLINE void fill_neginf(uint32_t l1_addr, uint32_t num_elements) {
    auto* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr);
    for (uint32_t i = 0; i < num_elements; ++i) {
        p[i] = neginf_bf16;
    }
}

// What the reader hands the compute and the writer next (see topk_two_level::header_row): a header page then 32
// tile pages in the stage-1 CB (the tiles carry a unit's keys and are unused otherwise), and a mailbox page.
FORCE_INLINE uint32_t two_level_announce(uint32_t code, uint32_t row) {
    using namespace topk_two_level;
    constexpr uint32_t stage1_batch = get_named_compile_time_arg_val("tl_stage1_batch");
    CircularBuffer writer_mailbox(get_named_compile_time_arg_val("tl_cb_writer_mailbox"));
    writer_mailbox.reserve_back(1);
    auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(writer_mailbox.get_write_ptr());
    words[writer_header_code_word] = code;
    words[writer_header_row_word] = row;
    clobber_all_memory();
    writer_mailbox.push_back(1);
    CircularBuffer stage1_cb(get_named_compile_time_arg_val("tl_cb_stage1"));
    stage1_cb.reserve_back(1 + stage1_batch);
    const uint32_t base = stage1_cb.get_write_ptr();
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base) = code;
    return base;
}

FORCE_INLINE void two_level_push_stage1() {
    clobber_all_memory();
    CircularBuffer(get_named_compile_time_arg_val("tl_cb_stage1"))
        .push_back(1 + get_named_compile_time_arg_val("tl_stage1_batch"));
}

// One phase-1 unit: its 32 tiles of 1024 keys, keys past the search length -inf.
template <typename InputAccessor>
FORCE_INLINE void two_level_unit(
    const Noc& noc,
    const InputAccessor& input,
    uint32_t search_len,
    const topk_two_level::Units& units,
    uint32_t unit) {
    using namespace topk_two_level;
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t stage1_batch = get_named_compile_time_arg_val("tl_stage1_batch");
    static_assert(tile_bytes == tile_keys * sizeof(uint16_t));
    CircularBuffer stage1_cb(get_named_compile_time_arg_val("tl_cb_stage1"));
    const uint32_t base = two_level_announce(unit, 0);
    // The unit's tiles are contiguous in the row and in the CB: its live keys, in whole 64 B granules (the row is a
    // multiple of 32 keys long, so this never leaves it), read one piece at a time. Phase 1 saturates the DRAM from
    // the other cores too; keeping little in flight per core keeps the queues short for the rows' small reads.
    constexpr uint32_t piece_bytes = 4096;
    const uint32_t first_key = units.part(unit) * stage1_batch * tile_keys;
    constexpr uint32_t unit_keys = stage1_batch * tile_keys;
    const uint32_t live = search_len - first_key < unit_keys ? search_len - first_key : unit_keys;
    const uint32_t live_bytes = div_up(live, block_keys) * block_bytes;
    for (uint32_t offset = 0; offset < live_bytes; offset += piece_bytes) {
        noc.async_read(
            input,
            stage1_cb,
            live_bytes - offset < piece_bytes ? live_bytes - offset : piece_bytes,
            {.page_id = units.row(unit), .offset_bytes = first_key * static_cast<uint32_t>(sizeof(uint16_t)) + offset},
            {.offset_bytes = tile_bytes + offset});
        noc.async_read_barrier();
    }
    if (live < unit_keys) {
        fill_neginf(base + tile_bytes + live * sizeof(uint16_t), unit_keys - live);
    }
    two_level_push_stage1();
}

// Stage 3's gather: the selected blocks table[0 .. 64 * num_chunks) of `row` into num_chunks 2048-key candidate
// chunks. A row is one page in one bank, so its NoC address is resolved once and each block read is an offset from
// it. Sentinel slots and keys past the search length are -inf.
// Out of line: three call sites, and the kernel config buffer is tight.
template <typename InputAccessor>
__attribute__((noinline)) void two_level_gather(
    const Noc& noc,
    const InputAccessor& input,
    uint32_t search_len,
    uint32_t row,
    const volatile tt_l1_ptr uint32_t* table,
    uint32_t num_chunks) {
    using namespace topk_two_level;
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t chunk_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t tiles_per_chunk = get_compile_time_arg_val(3);
    constexpr uint32_t blocks_per_chunk = chunk_bytes / block_bytes;
    CircularBuffer input_cb(cb_in);
    const uint32_t num_blocks = div_up(search_len, block_keys);
    const uint32_t last_block = num_blocks - 1;
    const uint32_t last_block_live = search_len - last_block * block_keys;
    const uint64_t row_noc_addr = input.get_noc_addr(row, 0, noc.get_noc_id());
    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        input_cb.reserve_back(tiles_per_chunk);
        const uint32_t chunk_base = input_cb.get_write_ptr();
        ASSERT((chunk_base & (block_bytes - 1)) == 0);
        uint32_t partial_slot = blocks_per_chunk;
        uint32_t slot = 0;
        while (slot < blocks_per_chunk) {
            const uint32_t block = table[chunk * blocks_per_chunk + slot];
            if (block == sentinel) {
                fill_neginf(chunk_base + slot * block_bytes, block_keys);
                ++slot;
                continue;
            }
            // One read for a run of consecutive blocks within this chunk.
            uint32_t run = 1;
            while (slot + run < blocks_per_chunk && table[chunk * blocks_per_chunk + slot + run] == block + run) {
                ++run;
            }
            if (block <= last_block && last_block < block + run) {
                partial_slot = slot + (last_block - block);
            }
            noc_async_read(
                row_noc_addr + block * block_bytes,
                chunk_base + slot * block_bytes,
                run * block_bytes,
                noc.get_noc_id());
            slot += run;
        }
        noc.async_read_barrier();
        if (partial_slot < blocks_per_chunk && last_block_live < block_keys) {
            // The last block is read whole; its keys past the search length are stale.
            fill_neginf(
                chunk_base + partial_slot * block_bytes + last_block_live * sizeof(uint16_t),
                block_keys - last_block_live);
        }
        clobber_all_memory();
        input_cb.push_back(tiles_per_chunk);
    }
}

// Stage 2 of one row: its block maxima from the maxima tensor through the input CB, one chunk per reserve so a chunk
// never wraps. Past the row's blocks a chunk is -inf.
template <typename MaxAccessor>
FORCE_INLINE void two_level_stage2(const Noc& noc, const MaxAccessor& block_max, uint32_t search_len, uint32_t row) {
    using namespace topk_two_level;
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t chunk_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t tiles_per_chunk = get_compile_time_arg_val(3);
    CircularBuffer input_cb(cb_in);
    const uint32_t max_chunks = stage2_chunks(search_len);
    const uint32_t max_bytes = div_up(search_len, block_keys) * sizeof(uint16_t);
    for (uint32_t chunk = 0; chunk < max_chunks; ++chunk) {
        input_cb.reserve_back(tiles_per_chunk);
        const uint32_t dst = input_cb.get_write_ptr();
        const uint32_t first = chunk * chunk_bytes;
        const uint32_t bytes =
            max_bytes > first ? (max_bytes - first < chunk_bytes ? max_bytes - first : chunk_bytes) : 0;
        for (uint32_t offset = 0; offset < bytes; offset += tile_bytes) {
            const uint32_t piece = bytes - offset < tile_bytes ? bytes - offset : tile_bytes;
            noc.async_read(
                block_max, input_cb, piece, {.page_id = row, .offset_bytes = first + offset}, {.offset_bytes = offset});
        }
        fill_neginf(dst + bytes, (chunk_bytes - bytes) / sizeof(uint16_t));
        noc.async_read_barrier();
        clobber_all_memory();
        input_cb.push_back(tiles_per_chunk);
    }
}

// A two-level program's schedule on this core. Its whole rows (see topk_two_level::row_owner) run as soon as their
// maxima are complete; until then the core claims phase-1 units, so the cores without a ready row keep the DRAM
// streaming. With the caller's maxima there is no phase 1 and every row is ready at once. After the end header a
// core of a split pair gathers its half of the pair's split row: the first core once its own writer has the table,
// the second once the first core's writer has sent it the second half.
template <typename InputAccessor, typename MaxAccessor>
FORCE_INLINE void two_level_run(
    const Noc& noc,
    const InputAccessor& input,
    const MaxAccessor& block_max,
    uint32_t search_len,
    const topk_two_level::Units& units,
    uint32_t core_index,
    uint32_t num_cores,
    uint32_t split_rows,
    bool has_split,
    bool second_half,
    uint32_t split_row) {
    using namespace topk_two_level;
    constexpr uint32_t chunk_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t half_chunks = selected_blocks / (chunk_bytes / block_bytes) / 2;
    constexpr bool has_block_max = get_named_compile_time_arg_val("tl_block_max_input") != 0;
    const uint32_t whole_rows = units.num_rows - split_rows;
    const uint32_t local_whole_rows = core_index < whole_rows ? div_up(whole_rows - core_index, num_cores) : 0;
    // The first core of a split pair also owns the split row's stage 2, as its last local row.
    const bool split_owner = has_split && !second_half;
    const uint32_t local_rows = local_whole_rows + (split_owner ? 1 : 0);
    const auto* ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_named_compile_time_arg_val("tl_ready_sem")));
    const uint64_t counter = get_noc_addr(
        get_common_arg_val<uint32_t>(topk_common_args::reader_coordinator_noc_x),
        get_common_arg_val<uint32_t>(topk_common_args::reader_coordinator_noc_y),
        get_semaphore(get_named_compile_time_arg_val("tl_unit_sem")),
        noc.get_noc_id());
    // A non-posted atomic increment returns the counter's old value to an L1 word of the issuing core. The claims
    // return to a word of their own: the default one is shared with the writer's atomics.
    const uint32_t claim_word = get_semaphore(get_named_compile_time_arg_val("tl_claim_sem"));
    const auto* claimed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(claim_word);
    const auto claim = [&](uint32_t increment, uint32_t return_word) {
        noc_fast_atomic_increment<noc_mode, true>(
            noc.get_noc_id(),
            write_at_cmd_buf,
            counter,
            NOC_UNICAST_WRITE_VC,
            increment,
            31 /*wrap*/,
            false /*linked*/,
            false /*posted*/,
            return_word);
        noc.async_atomic_barrier();
    };
    CircularBuffer table_cb(get_named_compile_time_arg_val("tl_cb_table"));
    bool units_left = !has_block_max;
    uint32_t local_row = 0;
    while (true) {
        // The split row is the last: its units are the last claimed, so it waits for the claims to run out.
        const bool is_split_row = split_owner && local_row == local_whole_rows;
        if (local_row < local_rows && !(is_split_row && units_left)) {
            invalidate_l1_cache();
            if (has_block_max || ready_count(*ready, local_row) >= units.per_row) {
                const uint32_t row = is_split_row ? split_row : core_index + local_row * num_cores;
                two_level_announce(is_split_row ? header_split : header_row, row);
                two_level_push_stage1();
                two_level_stage2(noc, block_max, search_len, row);
                if (!is_split_row) {
                    table_cb.wait_front(1);
                    two_level_gather(
                        noc,
                        input,
                        search_len,
                        row,
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(table_cb.get_read_ptr()),
                        2 * half_chunks);
                    table_cb.pop_front(1);
                }
                ++local_row;
                continue;
            }
        }
        if (units_left) {
            claim(1, claim_word);
            invalidate_l1_cache();
            const uint32_t unit = *claimed;
            if (unit < units.total) {
                two_level_unit(noc, input, search_len, units, unit);
            } else {
                units_left = false;
            }
            continue;
        }
        if (local_row == local_rows) {
            break;
        }
    }
    if constexpr (!has_block_max) {
        // Point this command buffer's atomic returns back at the default word, for the kernels that run after.
        claim(0, MEM_NOC_ATOMIC_RET_VAL_ADDR);
    }
    two_level_announce(header_end, 0);
    two_level_push_stage1();
    if (has_split) {
        // This core's half of the split row's candidates: the first core's by its own writer's table, the second
        // core's by the table half the first core's writer sent.
        uint32_t table_addr;
        if (split_owner) {
            table_cb.wait_front(1);
            table_addr = table_cb.get_read_ptr();
        } else {
            Semaphore<>(get_named_compile_time_arg_val("tl_half_sem")).wait_min(1);
            table_addr =
                CircularBuffer(get_named_compile_time_arg_val("tl_cb_own_table")).get_write_ptr() + remote_table_offset;
        }
        two_level_gather(
            noc, input, search_len, split_row, reinterpret_cast<volatile tt_l1_ptr uint32_t*>(table_addr), half_chunks);
        if (split_owner) {
            table_cb.pop_front(1);
        }
    }
}

}  // namespace
#endif

void kernel_main() {
    const uint32_t src_addr = get_common_arg_val<uint32_t>(topk_common_args::input_address);
    const uint32_t start_row = get_arg_val<uint32_t>(topk_core_args::reader_start_row);
    const uint32_t num_rows = get_arg_val<uint32_t>(topk_core_args::reader_num_rows);
    constexpr bool row_split = get_named_compile_time_arg_val("row_split") != 0;
    const uint32_t split_row = row_split ? get_arg_val<uint32_t>(topk_core_args::reader_split_row) : 0;
    const bool has_split = row_split && get_arg_val<uint32_t>(topk_core_args::reader_has_split) != 0;
    const bool second_half = row_split && get_arg_val<uint32_t>(topk_core_args::reader_second_half) != 0;
    uint32_t search_len = get_common_arg_val<uint32_t>(topk_common_args::search_length);
    const uint32_t input_page_bytes = get_common_arg_val<uint32_t>(topk_common_args::input_row_bytes);

    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t chunk_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t tiles_per_chunk = get_compile_time_arg_val(3);
    constexpr uint32_t element_bytes = sizeof(uint16_t);  // input is validated as BFLOAT16
    constexpr uint32_t llk_k = chunk_bytes / element_bytes;
    constexpr auto input_args = TensorAccessorArgs<4>();
    constexpr uint32_t metadata_args_base = input_args.next_compile_time_args_offset();
    // THE metadata-mode selector for this kernel. One flag, not one per metadata argument: the host
    // validates the metadata tensors as a group, so inside this mode every metadata read is legal.
    constexpr bool metadata_mode = get_compile_time_arg_val(metadata_args_base) != 0;
    constexpr uint32_t meta_cb = get_compile_time_arg_val(metadata_args_base + 1);
    constexpr uint32_t meta_offset = get_compile_time_arg_val(metadata_args_base + 2);
    constexpr auto meta_args = TensorAccessorArgs<metadata_args_base + 3>();
    // Real-token-end block, appended after the accessor above so those offsets are untouched.
    constexpr uint32_t vend_args_base = meta_args.next_compile_time_args_offset();
    constexpr auto vend_args = TensorAccessorArgs<vend_args_base>();
    constexpr uint32_t neginf_bf16_pair = 0xFF80FF80u;
#ifdef TOPK_TWO_LEVEL
    const uint32_t two_level_cutover = get_common_arg_val<uint32_t>(topk_common_args::two_level_cutover);
#endif

    const auto input = TensorAccessor(input_args, src_addr, input_page_bytes);
    const uint32_t input_width = input_page_bytes / element_bytes;
    CircularBuffer input_cb(cb_in);
    Noc noc;

    TopkMetadataBounds bounds;
    TopkSegmentBounds whole;
    TopkSegmentBounds split;
    if constexpr (metadata_mode) {
        CircularBuffer meta_cb_obj(meta_cb);
        meta_cb_obj.reserve_back(1);
        const uint32_t scratch = meta_cb_obj.get_write_ptr();
        const uint32_t metadata_length = trace_metadata::read_metadata_scalar_u32(
            noc, meta_args, get_common_arg_val<uint32_t>(topk_common_args::metadata_address), scratch);
        // Validate before addition so the offset cannot wrap and no malformed metadata can produce a NoC read
        // outside the input row. ASSERT is watcher-gated, so it compiles to nothing in a normal build; pair it
        // with a clamp, as bounded_kv_actual_isl / bounded_cache_batch_idx do, or the bound would hold only
        // under the watcher. Clamping to input_width also matches the score kernel's coupled derivation, which
        // saturates its own extent (kv_len_tiles is clamped to k_len_tiles) rather than reading past the row.
        const bool metadata_valid = meta_offset <= input_width && metadata_length <= input_width - meta_offset &&
                                    (metadata_length != 0 || meta_offset != 0);
        ASSERT(metadata_valid);
        search_len = metadata_valid ? metadata_length + meta_offset : input_width;
        // 0 = no bound supplied (full chunk): the host zeroes the slot rather than compiling a second
        // variant, so presence costs a compare here instead of a program hash split.
        const uint32_t valid_end_addr = get_common_arg_val<uint32_t>(topk_common_args::valid_end_address);
        if (valid_end_addr != 0) {
            // Same scratch page as the read above: the helper invalidates the L1 line before loading, which
            // is exactly what it documents for a reused address.
            const uint32_t valid_end =
                trace_metadata::read_metadata_scalar_u32(noc, vend_args, valid_end_addr, scratch);
            const uint32_t capped = ((valid_end + 31) / 32) * 32;
            if (capped < search_len) {
                search_len = capped;
            }
        }
        bounds = calculate_topk_bounds(search_len, llk_k);
        whole = whole_row_bounds(bounds);
        CoreLocalMem<TopkMetadataBounds> mailbox(scratch);
#ifdef TOPK_TWO_LEVEL
        auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
        words[topk_two_level::mailbox_mode_word] =
            topk_two_level::two_level_mode_for(search_len, two_level_cutover) ? 1u : 0u;
        words[topk_two_level::mailbox_search_length_word] = search_len;
#endif
        mailbox->num_chunks = whole.num_chunks;
        mailbox->tail_elements = whole.tail_elements;
        if constexpr (row_split) {
            // The compute also needs this core's half of the split row.
            split = half_bounds(bounds, llk_k, second_half);
            CoreLocalMem<TopkMetadataBounds> split_mailbox(scratch + sizeof(TopkMetadataBounds));
            split_mailbox->num_chunks = split.num_chunks;
            split_mailbox->tail_elements = split.tail_elements;
        }
        clobber_all_memory();
        meta_cb_obj.push_back(1);
    } else {
        bounds = calculate_topk_bounds(search_len, llk_k);
        whole = whole_row_bounds(bounds);
        if constexpr (row_split) {
            split = half_bounds(bounds, llk_k, second_half);
        }
#ifdef TOPK_TWO_LEVEL
        // A two-level program's compute always reads its bounds and mode from the mailbox.
        CircularBuffer meta_cb_obj(meta_cb);
        meta_cb_obj.reserve_back(1);
        auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_cb_obj.get_write_ptr());
        words[topk_metadata_num_chunks_word] = whole.num_chunks;
        words[topk_metadata_tail_elements_word] = whole.tail_elements;
        if constexpr (row_split) {
            words[topk_metadata_split_word_offset + topk_metadata_num_chunks_word] = split.num_chunks;
            words[topk_metadata_split_word_offset + topk_metadata_tail_elements_word] = split.tail_elements;
        }
        words[topk_two_level::mailbox_mode_word] =
            topk_two_level::two_level_mode_for(search_len, two_level_cutover) ? 1u : 0u;
        words[topk_two_level::mailbox_search_length_word] = search_len;
        clobber_all_memory();
        meta_cb_obj.push_back(1);
#endif
    }

#ifdef TOPK_TWO_LEVEL
    const bool two_level_mode = topk_two_level::two_level_mode_for(search_len, two_level_cutover);
    // The writer learns the mode and the search length here; it has no bound of its own.
    CircularBuffer writer_mailbox(get_named_compile_time_arg_val("tl_cb_writer_mailbox"));
    writer_mailbox.reserve_back(1);
    auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(writer_mailbox.get_write_ptr());
    words[topk_two_level::writer_mailbox_mode_word] = two_level_mode ? 1u : 0u;
    words[topk_two_level::writer_mailbox_search_length_word] = search_len;
    clobber_all_memory();
    writer_mailbox.push_back(1);
    if (two_level_mode) {
        constexpr bool has_block_max = get_named_compile_time_arg_val("tl_block_max_input") != 0;
        if constexpr (!has_block_max) {
            // The max reduce multiplies by this scaler tile: 1.0 everywhere keeps every value exact.
            CircularBuffer scaler_cb(get_named_compile_time_arg_val("tl_cb_scaler"));
            scaler_cb.reserve_back(1);
            auto* scaler = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scaler_cb.get_write_ptr());
            for (uint32_t word = 0; word < tile_bytes / sizeof(uint32_t); ++word) {
                scaler[word] = 0x3F803F80u;
            }
            clobber_all_memory();
            scaler_cb.push_back(1);
        }
        constexpr auto block_max_args = TensorAccessorArgs<vend_args.next_compile_time_args_offset()>();
        const auto block_max = TensorAccessor(
            block_max_args,
            get_common_arg_val<uint32_t>(topk_common_args::block_max_address),
            get_common_arg_val<uint32_t>(topk_common_args::two_level_max_row_bytes));
        two_level_run(
            noc,
            input,
            block_max,
            search_len,
            topk_two_level::Units::of(
                search_len,
                get_named_compile_time_arg_val("tl_stage1_batch"),
                get_common_arg_val<uint32_t>(topk_common_args::two_level_num_input_rows)),
            get_arg_val<uint32_t>(topk_core_args::reader_two_level_core_index),
            get_common_arg_val<uint32_t>(topk_common_args::two_level_num_cores),
            get_common_arg_val<uint32_t>(topk_common_args::two_level_split_rows),
            has_split,
            second_half,
            split_row);
        return;
    }
#endif

    // One loop for both: without a split row, has_split is false and every row takes the whole-row bounds.
    const uint32_t num_whole_rows = has_split ? num_rows - 1 : num_rows;
    for (uint32_t local_row = 0; local_row < num_rows; ++local_row) {
        const bool is_split = has_split && local_row == num_whole_rows;
        const uint32_t row = is_split ? split_row : start_row + local_row;
        const TopkSegmentBounds& segment = is_split ? split : whole;
        const uint32_t tail_chunk_bytes = segment.tail_elements * element_bytes;
        if (segment.empty) {
            // Nothing valid in this half: hand the compute one chunk of -inf so its survivors lose the merge.
            input_cb.reserve_back(tiles_per_chunk);
            auto* chunk_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(input_cb.get_write_ptr());
            for (uint32_t word = 0; word < tiles_per_chunk * tile_bytes / sizeof(uint32_t); ++word) {
                chunk_words[word] = neginf_bf16_pair;
            }
            // Order the fill ahead of the push, as the metadata mailbox above does.
            clobber_all_memory();
            input_cb.push_back(tiles_per_chunk);
            continue;
        }
        for (uint32_t chunk = 0; chunk < segment.num_chunks; ++chunk) {
            const uint32_t row_chunk = segment.first_chunk + chunk;
            const uint32_t active_chunk_bytes = (chunk + 1 == segment.num_chunks) ? tail_chunk_bytes : chunk_bytes;
            input_cb.reserve_back(tiles_per_chunk);
            for (uint32_t tile = 0; tile < tiles_per_chunk; ++tile) {
                const uint32_t tile_offset = tile * tile_bytes;
                const uint32_t read_bytes =
                    tile_offset < active_chunk_bytes
                        ? (active_chunk_bytes - tile_offset < tile_bytes ? active_chunk_bytes - tile_offset
                                                                         : tile_bytes)
                        : 0;
                if (read_bytes != 0) {
                    noc.async_read(
                        input,
                        input_cb,
                        read_bytes,
                        {.page_id = row, .offset_bytes = row_chunk * chunk_bytes + tile_offset},
                        {.offset_bytes = tile_offset});
                }
            }
            noc.async_read_barrier();
            input_cb.push_back(tiles_per_chunk);
        }
    }
}
