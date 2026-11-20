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

// Returns the address of the last key stored. Its caller waits for the stores (topk_two_level::wait_l1_store) before
// the CB push that hands them on.
FORCE_INLINE uint32_t fill_neginf(uint32_t l1_addr, uint32_t num_elements) {
    auto* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr);
    for (uint32_t i = 0; i < num_elements; ++i) {
        p[i] = neginf_bf16;
    }
    return l1_addr + (num_elements - 1) * sizeof(uint16_t);
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
    wait_l1_store(reinterpret_cast<uint32_t>(&words[writer_header_row_word]));
    writer_mailbox.push_back(1);
    CircularBuffer stage1_cb(get_named_compile_time_arg_val("tl_cb_stage1"));
    stage1_cb.reserve_back(1 + stage1_batch);
    const uint32_t base = stage1_cb.get_write_ptr();
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base) = code;
    return base;
}

// last_store: the address of the reader's last L1 store into the pushed pages (the header, or a -inf fill after it).
FORCE_INLINE void two_level_push_stage1(uint32_t last_store) {
    topk_two_level::wait_l1_store(last_store);
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
    two_level_push_stage1(
        live < unit_keys ? fill_neginf(base + tile_bytes + live * sizeof(uint16_t), unit_keys - live) : base);
}

// One read of the gather from the source set by noc_async_read_set_state (the row's DRAM bank, or this core's L1 for
// the staged copies). Out of line: each inlined NoC read is a few hundred bytes of a tight kernel config buffer.
__attribute__((noinline)) void gather_read(uint32_t src, uint32_t dst, uint32_t bytes, uint8_t noc_id) {
    noc_async_read_with_state(src, dst, bytes, noc_id);
}

// Stage 3's gather: the selected blocks table[0 .. 64 * num_chunks) of `row` into num_chunks 2048-key candidate
// chunks. A row is one page in one bank, so its NoC address is resolved once and each block read is an offset from
// it. Sentinel slots and keys past the search length are -inf.
//
// The gather is bound by the number of DRAM reads more than their size, so reads are coalesced: a run of consecutive
// blocks is one read into its slots, and other selected blocks sharing an aligned 1-2 KB span are one read of their
// range into a staging area, then copied to their slots by local L1 reads.
// The staging area is the stage-1 CB, idle while a row gathers: reserving all of it waits until the compute has
// consumed every page there, and nothing is pushed.
//
// Out of line: two call sites, and the kernel config buffer is tight. Each chunk waits for its page of the table: from
// this core's writer through the table CB, or (from_partner) from the split pair's first core, counted by half_sem.
template <typename InputAccessor>
__attribute__((noinline)) void two_level_gather(
    const Noc& noc,
    const InputAccessor& input,
    uint32_t search_len,
    uint32_t row,
    const volatile tt_l1_ptr uint32_t* table,
    uint32_t num_chunks,
    bool from_partner) {
    using namespace topk_two_level;
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t chunk_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t tiles_per_chunk = get_compile_time_arg_val(3);
    constexpr uint32_t blocks_per_chunk = chunk_bytes / block_bytes;
    static_assert(blocks_per_chunk == table_page_entries);
    constexpr uint32_t stage1_batch = get_named_compile_time_arg_val("tl_stage1_batch");
    constexpr uint32_t span_blocks = 32;  // the largest span, 2 KB
    CircularBuffer input_cb(cb_in);
    CircularBuffer stage1_cb(get_named_compile_time_arg_val("tl_cb_stage1"));
    // A staged range holds at least two of the chunk's 64 slots, so a chunk stages at most 32 spans: 64 KB, within the
    // CB's contiguous half from its write pointer.
    static_assert(blocks_per_chunk / 2 * span_blocks * block_bytes <= (1 + stage1_batch) * get_compile_time_arg_val(2));
    stage1_cb.reserve_back(2 * (1 + stage1_batch));
    const uint32_t staging = stage1_cb.get_write_ptr();
    const uint32_t num_blocks = div_up(search_len, block_keys);
    const uint32_t last_block = num_blocks - 1;
    const uint32_t last_block_live = search_len - last_block * block_keys;
    // 2 KB spans for rows of at most 8 blocks per selected slot (up to about 512K keys), 1 KB for longer rows, where
    // the wider range reads measured slower than the reads they save.
    const uint32_t span_shift = num_blocks > 8 * selected_blocks ? 4 : 5;
    const uint8_t noc_id = noc.get_noc_id();
    const uint64_t row_noc_addr = input.get_noc_addr(row, 0, noc_id);
    // The row is one page in one DRAM bank: its reads differ only in the low address word.
    const uint32_t row_addr = static_cast<uint32_t>(row_noc_addr);
    ASSERT(num_blocks == 0 || row_addr + num_blocks * block_bytes - 1 >= row_addr);
    // A slot's staged source address, with bit 31 set; 0 for a slot read in place. Zeroed by a loop, not an
    // initialiser, which would pull in memset.
    uint32_t copy_from[blocks_per_chunk];
    for (uint32_t slot = 0; slot < blocks_per_chunk; ++slot) {
        copy_from[slot] = 0;
    }
    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        if (from_partner) {
            Semaphore<>(get_named_compile_time_arg_val("tl_half_sem")).wait_min(chunk + 1);
        } else {
            CircularBuffer(get_named_compile_time_arg_val("tl_cb_table")).wait_front(chunk + 1);
        }
        invalidate_l1_cache();
        asm volatile("" ::: "memory");
        input_cb.reserve_back(tiles_per_chunk);
        const uint32_t chunk_base = input_cb.get_write_ptr();
        ASSERT((chunk_base & (block_bytes - 1)) == 0);
        // The chunk's blocks, read from L1 once.
        uint32_t chunk_table[blocks_per_chunk];
        const auto* page = const_cast<const uint32_t*>(table) + chunk * blocks_per_chunk;
#pragma GCC unroll 8
        for (uint32_t i = 0; i < blocks_per_chunk; ++i) {
            chunk_table[i] = page[i];
        }
        noc_async_read_set_state(row_noc_addr, noc_id);
        uint32_t partial_slot = blocks_per_chunk;
        // The last -inf stored into the chunk, if any.
        uint32_t last_fill = 0;
        uint32_t staged = staging;
        uint32_t copies = 0;
        uint32_t slot = 0;
        while (slot < blocks_per_chunk) {
            const uint32_t block = chunk_table[slot];
            if (block == sentinel) {
                last_fill = fill_neginf(chunk_base + slot * block_bytes, block_keys);
                ++slot;
                continue;
            }
            uint32_t run = 1;
            while (slot + run < blocks_per_chunk && chunk_table[slot + run] == block + run) {
                ++run;
            }
            // A lone block: the other selected blocks of its span, if any, join it in one staged read.
            uint32_t group = 1;
            if (run == 1) {
                while (slot + group < blocks_per_chunk && chunk_table[slot + group] != sentinel &&
                       chunk_table[slot + group] >> span_shift == block >> span_shift) {
                    ++group;
                }
            }
            if (group == 1) {
                if (block <= last_block && last_block < block + run) {
                    partial_slot = slot + (last_block - block);
                }
                gather_read(row_addr + block * block_bytes, chunk_base + slot * block_bytes, run * block_bytes, noc_id);
                slot += run;
                continue;
            }
            const uint32_t span_bytes = (chunk_table[slot + group - 1] - block + 1) * block_bytes;
            gather_read(row_addr + block * block_bytes, staged, span_bytes, noc_id);
            for (uint32_t i = 0; i < group; ++i) {
                const uint32_t member = chunk_table[slot + i];
                if (member == last_block) {
                    partial_slot = slot + i;
                }
                copy_from[slot + i] = (staged + (member - block) * block_bytes) | (1u << 31);
            }
            copies = 1;
            staged += span_bytes;
            slot += group;
        }
        noc.async_read_barrier();
        if (copies) {
            // This core's own L1.
            noc_async_read_set_state(get_noc_addr(0, noc_id), noc_id);
            for (slot = 0; slot < blocks_per_chunk;) {
                const uint32_t from = copy_from[slot];
                if (!(from >> 31)) {
                    ++slot;
                    continue;
                }
                // Slots whose staged sources are contiguous (consecutive members, or adjacent groups) move in one copy.
                uint32_t count = 1;
                while (slot + count < blocks_per_chunk && copy_from[slot + count] == from + count * block_bytes) {
                    copy_from[slot + count] = 0;
                    ++count;
                }
                copy_from[slot] = 0;
                gather_read(from & 0x7FFFFFFFu, chunk_base + slot * block_bytes, count * block_bytes, noc_id);
                slot += count;
            }
            noc.async_read_barrier();
        }
        if (partial_slot < blocks_per_chunk && last_block_live < block_keys) {
            // The last block is read whole; its keys past the search length are stale.
            last_fill = fill_neginf(
                chunk_base + partial_slot * block_bytes + last_block_live * sizeof(uint16_t),
                block_keys - last_block_live);
        }
        if (last_fill != 0) {
            wait_l1_store(last_fill);
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
        if (bytes < chunk_bytes) {
            wait_l1_store(fill_neginf(dst + bytes, (chunk_bytes - bytes) / sizeof(uint16_t)));
        }
        noc.async_read_barrier();
        clobber_all_memory();
        input_cb.push_back(tiles_per_chunk);
    }
}

// A two-level program's schedule on this core. Its rows (see topk_two_level::row_owner) run as soon as their maxima are
// complete; until then the core claims phase-1 units, so the cores without a ready row keep the DRAM streaming. With
// the caller's maxima there is no phase 1 and every row is ready at once. A core of a split pair gathers its half of
// the pair's split row, a chunk as soon as its page of the table is there: the first core's from its own writer, the
// second core's from the first core's writer; with tl_early_halves as soon as the halves can run, else after the end
// header.
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
    auto* flags = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        CircularBuffer(get_named_compile_time_arg_val("tl_cb_ready_flags")).get_write_ptr());
    const uint64_t counter = get_noc_addr(
        get_common_arg_val<uint32_t>(topk_common_args::reader_coordinator_noc_x),
        get_common_arg_val<uint32_t>(topk_common_args::reader_coordinator_noc_y),
        get_semaphore(get_named_compile_time_arg_val("tl_unit_sem")),
        noc.get_noc_id());
    const uint32_t claim_word = get_semaphore(get_named_compile_time_arg_val("tl_claim_sem"));
    const auto* claimed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(claim_word);
    // With tl_early_halves a split row's halves of stage 3 run as soon as they can, rather than after the end: the
    // first core's right after the row's stage 2, the second core's once the first table page has arrived from its
    // partner.
    constexpr bool early_halves = get_named_compile_time_arg_val("tl_early_halves") != 0;
    // The cores in row mode while units remain, counted on the coordinator: with split_rows_first about a cap of them
    // take rows (the count each core gets back is an estimate, see below), so the rest keep claiming units and phase 1
    // does not stall behind the rows.
    const uint64_t rows_counter = get_noc_addr(
        get_common_arg_val<uint32_t>(topk_common_args::reader_coordinator_noc_x),
        get_common_arg_val<uint32_t>(topk_common_args::reader_coordinator_noc_y),
        get_semaphore(get_named_compile_time_arg_val("tl_rows_sem")),
        noc.get_noc_id());
    const uint32_t row_cap = split_rows_first(early_halves, search_len) ? num_cores * row_cap_fifths / 5 : num_cores;
    // A non-posted atomic add returns the counter's old value to the return address of its command buffer. That value
    // is not reliable: the atomic barrier can complete before it lands, it goes to whatever return address the buffer
    // holds when it does, and under load a core can get another claim's value. The increments themselves and NoC reads
    // are reliable, so correctness rests on those:
    // - A claim waits for claim_word to change from the core's last value (the word starts at 0xFFFFFFFF, which no
    //   claim returns), for at most claim_timeout. A wrong or missing value is a unit done twice, which its done word
    //   absorbs, or a unit nobody does, which its row's owner repairs (see below).
    // - The other atomics (the cleared count and the row counter) use the register-write buffer, unused by this kernel
    //   otherwise, with rows_word as its return address, so their values never land in claim_word. The row counter's
    //   value is only the cap's estimate.
    static_assert(noc_mode == DM_DEDICATED_NOC);
    constexpr uint32_t claim_timeout = 200000;  // wall-clock ticks, about 150 us
    constexpr uint32_t repair_after = 4000000;  // wall-clock ticks idle (about 3 ms) once every unit is claimed
    uint32_t last_claimed = 0xFFFFFFFFu;
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
        if (return_word != claim_word) {
            return true;
        }
        const uint32_t start = get_timestamp_32b();
        do {
            invalidate_l1_cache();
            if (*claimed != last_claimed) {
                last_claimed = *claimed;
                return true;
            }
        } while (get_timestamp_32b() - start < claim_timeout);
        return false;
    };
    const uint32_t rows_word = get_semaphore(get_named_compile_time_arg_val("tl_rows_return_sem"));
    const auto* rows_returned = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows_word);
    const uint32_t local_coordinate =
        static_cast<uint32_t>(NOC_XY_ADDR(my_x[noc.get_noc_id()], my_y[noc.get_noc_id()], 0) >> NOC_ADDR_COORD_SHIFT) &
        NOC_COORDINATE_MASK;
    const auto aux_add = [&](uint64_t target, uint32_t increment) {
        noc_fast_atomic_increment<noc_mode, true>(
            noc.get_noc_id(),
            write_reg_cmd_buf,
            target,
            NOC_UNICAST_WRITE_VC,
            increment,
            31 /*wrap*/,
            false /*linked*/,
            false /*posted*/,
            rows_word);
        noc.async_atomic_barrier();
        invalidate_l1_cache();
        return *rows_returned;
    };
    const auto rows_add = [&](uint32_t increment) { return aux_add(rows_counter, increment); };
    const bool capped = !has_block_max && row_cap < num_cores;
    if constexpr (!has_block_max) {
        // The atomic sets the return address's low word; the rest points it at this core.
        NOC_CMD_BUF_WRITE_REG(noc.get_noc_id(), write_reg_cmd_buf, NOC_RET_ADDR_MID, 0);
        NOC_CMD_BUF_WRITE_REG(noc.get_noc_id(), write_reg_cmd_buf, NOC_RET_ADDR_COORDINATE, local_coordinate);
        // Clear this run's done words, then count this core on the coordinator; no writer sends a done word before
        // every core is counted.
        for (uint32_t word = 0; word < ready_flags_bytes / sizeof(uint32_t); ++word) {
            flags[word] = 0;
        }
        wait_l1_store(reinterpret_cast<uint32_t>(&flags[ready_flags_bytes / sizeof(uint32_t) - 1]));
        aux_add(
            get_noc_addr(
                get_common_arg_val<uint32_t>(topk_common_args::reader_coordinator_noc_x),
                get_common_arg_val<uint32_t>(topk_common_args::reader_coordinator_noc_y),
                get_semaphore(get_named_compile_time_arg_val("tl_cleared_sem")),
                noc.get_noc_id()),
            1);
    }
    // A local row is ready when its first per_row done words are nonzero; complete_parts[r] counts those seen so far.
    uint8_t complete_parts[max_local_rows] = {};
    const auto row_ready = [&](uint32_t local_row) {
        uint32_t part = complete_parts[local_row];
        while (part < units.per_row && flags[local_row * max_row_units + part] != 0) {
            ++part;
        }
        complete_parts[local_row] = part;
        return part == units.per_row;
    };
    const auto local_row_index = [&](uint32_t local_row) {
        return split_owner && local_row == local_whole_rows ? split_row : core_index + local_row * num_cores;
    };
    // The coordinator's counter, read over the NoC: the number of units claimed so far, or more.
    const auto read_counter = [&]() {
        auto* scratch = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(get_named_compile_time_arg_val("tl_scratch_sem")));
        noc_async_read(counter, reinterpret_cast<uint32_t>(scratch), sizeof(uint32_t), noc.get_noc_id());
        noc_async_read_barrier(noc.get_noc_id());
        invalidate_l1_cache();
        return *scratch;
    };
    // The idle time since this core last had work, for the repair.
    bool idle = false;
    uint32_t idle_since = 0;
    bool repairing = false;
    uint32_t repair_next = 0;
    CircularBuffer table_cb(get_named_compile_time_arg_val("tl_cb_table"));
    const auto* half_pages =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_named_compile_time_arg_val("tl_half_sem")));
    const uint32_t remote_table = CircularBuffer(get_named_compile_time_arg_val("tl_cb_remote_table")).get_write_ptr();
    bool half_pending = early_halves && has_split;
    bool units_left = !has_block_max;
    // This core's local rows run in the order their maxima complete; done_rows has a bit per finished one.
    uint32_t done_rows = 0;
    uint32_t rows_done = 0;
    while (true) {
        if (half_pending && !split_owner) {
            invalidate_l1_cache();
            bool counted = false;
            bool allowed = *half_pages >= 1;
            if (allowed && units_left && capped) {
                counted = true;
                if (rows_add(1) >= row_cap) {
                    rows_add(0xFFFFFFFFu);
                    counted = false;
                    allowed = false;
                }
            }
            if (allowed) {
                two_level_push_stage1(two_level_announce(header_half, split_row));
                two_level_gather(
                    noc,
                    input,
                    search_len,
                    split_row,
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(remote_table),
                    half_chunks,
                    true);
                if (counted) {
                    rows_add(0xFFFFFFFFu);
                }
                half_pending = false;
                idle = false;
                continue;
            }
        }
        // The first of this core's rows whose maxima are complete. Without tl_early_halves the split row is the last:
        // its table stays in the table CB until its half gathers after the end, so it waits for the claims to run out
        // and for the whole rows.
        uint32_t local_row = local_rows;
        if (rows_done < local_rows) {
            invalidate_l1_cache();
            for (uint32_t candidate = 0; candidate < local_rows; ++candidate) {
                const bool split = split_owner && candidate == local_whole_rows;
                if ((done_rows >> candidate & 1) ||
                    (split && !early_halves && (units_left || rows_done < local_whole_rows))) {
                    continue;
                }
                if (has_block_max || row_ready(candidate)) {
                    local_row = candidate;
                    break;
                }
            }
        }
        // Over the cap, the core claims a unit instead.
        bool counted = false;
        if (local_row < local_rows && units_left && capped) {
            counted = true;
            if (rows_add(1) >= row_cap) {
                rows_add(0xFFFFFFFFu);
                counted = false;
                local_row = local_rows;
            }
        }
        const bool is_split_row = split_owner && local_row == local_whole_rows;
        if (local_row < local_rows) {
            const uint32_t row = local_row_index(local_row);
            two_level_push_stage1(two_level_announce(is_split_row ? header_split : header_row, row));
            two_level_stage2(noc, block_max, search_len, row);
            if (is_split_row && early_halves) {
                two_level_push_stage1(two_level_announce(header_half, row));
                two_level_gather(
                    noc,
                    input,
                    search_len,
                    row,
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(table_cb.get_read_ptr()),
                    half_chunks,
                    false);
                table_cb.wait_front(table_pages);
                table_cb.pop_front(table_pages);
                half_pending = false;
            }
            if (!is_split_row) {
                two_level_gather(
                    noc,
                    input,
                    search_len,
                    row,
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(table_cb.get_read_ptr()),
                    2 * half_chunks,
                    false);
                table_cb.pop_front(table_pages);
            }
            if (counted) {
                rows_add(0xFFFFFFFFu);
            }
            done_rows |= 1u << local_row;
            ++rows_done;
            idle = false;
            continue;
        }
        // The next phase-1 unit: a claimed one, or with `repairing` one of this core's rows' missing ones.
        uint32_t unit = units.total;
        if (units_left) {
            if (claim(1, claim_word) && last_claimed < units.total) {
                unit = last_claimed;
            } else {
                // Past the last unit, or no value came (its unit is then missing): the counter itself says whether
                // units remain.
                units_left = read_counter() < units.total;
                continue;
            }
        } else if (repairing) {
            // The repair walks this core's done words once, from repair_next: each missing unit is done once.
            for (; repair_next < local_rows * max_row_units && unit == units.total; ++repair_next) {
                const uint32_t local_row = repair_next / max_row_units;
                const uint32_t part = repair_next % max_row_units;
                if (part < units.per_row && !(done_rows >> local_row & 1) && flags[repair_next] == 0) {
                    unit = units.unit_of(local_row_index(local_row), part);
                }
            }
            repairing = unit < units.total;
        }
        if (unit < units.total) {
            two_level_unit(noc, input, search_len, units, unit);
            idle = false;
            continue;
        }
        if (rows_done == local_rows && !half_pending) {
            break;
        }
        // Nothing to do: every unit is claimed, and this core waits for its rows' units. If it has waited repair_after,
        // one went missing (see the claims above), and the core does its rows' missing units itself.
        if constexpr (!has_block_max) {
            const uint32_t now = get_timestamp_32b();
            if (!idle) {
                idle = true;
                idle_since = now;
            } else if (!units_left && now - idle_since > repair_after) {
                repairing = read_counter() >= units.total;
                repair_next = 0;
                idle = false;
            }
        }
    }
    if constexpr (!has_block_max) {
        // Point the atomic buffer's returns back at the default word, and the register-write buffer's target back at
        // this core, as the kernels that run after expect.
        claim(0, MEM_NOC_ATOMIC_RET_VAL_ADDR);
        NOC_CMD_BUF_WRITE_REG(noc.get_noc_id(), write_reg_cmd_buf, NOC_TARG_ADDR_MID, 0);
        NOC_CMD_BUF_WRITE_REG(noc.get_noc_id(), write_reg_cmd_buf, NOC_TARG_ADDR_COORDINATE, local_coordinate);
    }
    two_level_push_stage1(two_level_announce(header_end, 0));
    if (has_split && !early_halves) {
        // This core's half of the split row's candidates: the first core's by its own writer's table, the second
        // core's by the table half the first core's writer sends into its remote-table CB.
        const uint32_t table_addr =
            split_owner ? table_cb.get_read_ptr()
                        : CircularBuffer(get_named_compile_time_arg_val("tl_cb_remote_table")).get_write_ptr();
        two_level_gather(
            noc,
            input,
            search_len,
            split_row,
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(table_addr),
            half_chunks,
            !split_owner);
        if (split_owner) {
            table_cb.wait_front(table_pages);
            table_cb.pop_front(table_pages);
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
        uint32_t last_store = reinterpret_cast<uint32_t>(&mailbox->tail_elements);
        if constexpr (row_split) {
            // The compute also needs this core's half of the split row.
            split = half_bounds(bounds, llk_k, second_half);
            CoreLocalMem<TopkMetadataBounds> split_mailbox(scratch + sizeof(TopkMetadataBounds));
            split_mailbox->num_chunks = split.num_chunks;
            split_mailbox->tail_elements = split.tail_elements;
            last_store = reinterpret_cast<uint32_t>(&split_mailbox->tail_elements);
        }
#ifdef TOPK_TWO_LEVEL
        topk_two_level::wait_l1_store(last_store);
#else
        (void)last_store;
        clobber_all_memory();
#endif
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
        topk_two_level::wait_l1_store(reinterpret_cast<uint32_t>(&words[topk_two_level::mailbox_search_length_word]));
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
    topk_two_level::wait_l1_store(
        reinterpret_cast<uint32_t>(&words[topk_two_level::writer_mailbox_search_length_word]));
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
            topk_two_level::wait_l1_store(reinterpret_cast<uint32_t>(&scaler[tile_bytes / sizeof(uint32_t) - 1]));
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
                get_common_arg_val<uint32_t>(topk_common_args::two_level_num_input_rows),
                get_common_arg_val<uint32_t>(topk_common_args::two_level_split_rows),
                topk_two_level::split_rows_first(get_named_compile_time_arg_val("tl_early_halves") != 0, search_len)),
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
#ifdef TOPK_TWO_LEVEL
            topk_two_level::wait_l1_store(
                reinterpret_cast<uint32_t>(&chunk_words[tiles_per_chunk * tile_bytes / sizeof(uint32_t) - 1]));
#else
            clobber_all_memory();
#endif
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
