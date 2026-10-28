// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writes this core's index scores to the scores output, selects every query row's global top-k with
// the other cores, then gathers the second half of this core's selected kv_cache rows into the
// output. With PARTIAL_OUTPUT the top-k is this device's part only and is written out as candidates
// instead, and in CANDIDATE_SOURCE mode so are the part's best candidate blocks.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "ckernel.h"
#include "tt_metal/tools/profiler/kernel_profiler.hpp"
#include "experimental/kernel_args.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/operations/experimental/deepseek/fused_lightning_select_kv/device/kernels/dataflow/gather_kv_rows.hpp"

// Maps fp32 bits to a uint32 whose unsigned order matches the float order.
FORCE_INLINE uint32_t topk_order_key(uint32_t bits) { return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u); }

// One core's share of one query row's cross-core top-k: its keys whose top (32 - final_shift) order-key
// bits exceed final_prefix are selected, plus the first num_from_bin of those equal to it; together
// they are num_contribute keys starting at output_offset of the row's global selection.
struct SelectResult {
    uint32_t final_prefix;
    uint32_t final_shift;
    uint32_t num_contribute;
    uint32_t num_from_bin;
    uint32_t output_offset;
};

void kernel_main() {
    // ---- Compile-time args ----
    [[maybe_unused]] constexpr uint32_t k = get_arg(args::k);
    constexpr uint32_t num_queries = get_arg(args::num_queries);  // Sq query rows, each with its own top-k
    static_assert(num_queries >= 1 && num_queries <= 32, "query rows are tracked in a 32-bit mask");
    constexpr uint32_t kAllRows = num_queries == 32 ? 0xFFFFFFFFu : (1u << num_queries) - 1;
    constexpr uint32_t page_block_size = get_arg(args::page_block_size);
    constexpr uint32_t chunks_per_block = page_block_size / 32;
    constexpr uint32_t local_key_capacity = get_arg(args::local_key_capacity);  // keys per row in cand_list
    constexpr uint32_t num_hist_bins = get_arg(args::num_hist_bins);
    static_assert(num_hist_bins == 256, "the radix passes take 8 bits of the order key at a time");
    constexpr uint32_t kHistBits = 8;
    constexpr uint32_t bins_per_slice = get_arg(args::hist_bins_per_slice);
    constexpr uint32_t kRowSlices = num_hist_bins / bins_per_slice;
    static_assert(bins_per_slice == 4, "the threshold walk reads one 16 B slice of bins at a time");
    // Each core's histograms are [query row][bin]; owner o sums words [o * owner_words, ...) of them,
    // which may span query rows, and writes them to the coordinator followed by its slices' totals.
    constexpr uint32_t kHistWords = num_queries * num_hist_bins;
    constexpr uint32_t num_owners = get_arg(args::hist_num_owners);
    constexpr uint32_t slices_per_owner = get_arg(args::hist_slices_per_owner);
    constexpr uint32_t owner_words = slices_per_owner * bins_per_slice;
    constexpr uint32_t owner_chunk_words =
        owner_words + (slices_per_owner + bins_per_slice - 1) / bins_per_slice * bins_per_slice;
    // Per query row {keys above its threshold bin, keys in it}, padded to 16 B for NoC alignment.
    constexpr uint32_t kCountsWords = (2 * num_queries + 3) / 4 * 4;
    constexpr uint32_t num_select_words = get_arg(args::num_select_words);
    // select: {query rows still selecting, pad x7}, then per query row a header {prefix, prefix bits,
    // keys above it, keys still needed from it, pad, total keys, pad, pad}, then the allocation.
    constexpr uint32_t kSelectLiveWord = 0;
    constexpr uint32_t kSelectRowsWord = 8;
    constexpr uint32_t kSelectHeaderWords = 8;
    constexpr uint32_t kHdrPrefix = 0;
    constexpr uint32_t kHdrPrefixBits = 1;
    constexpr uint32_t kHdrAbove = 2;
    constexpr uint32_t kHdrNeeded = 3;
    constexpr uint32_t kHdrTotal = 5;
    constexpr uint32_t kSelectAllocWord = kSelectRowsWord + num_queries * kSelectHeaderWords;
    // Allocation entry of (core c, query row j) at kSelectAllocWord + (c * num_queries + j) * 3:
    // {keys to contribute, of which equal to the threshold, output offset}.
    constexpr uint32_t kSelectWordsPerCore = 3;
    static_assert(32 % kHistBits == 0, "radix passes must tile the 32-bit order key");
    // select_ready value of a select's final allocation, above its ready_base; pass p's header uses
    // ready_base + p + 1 <= ready_base + 32 / kHistBits.
    constexpr uint32_t kAllocReady = 32 / kHistBits + 1;
    constexpr uint32_t kv_stage_rows = get_arg(args::kv_stage_rows);
    // sel: {num_keys, pad, pad, pad, reader's first output page per query row..., rows..., masks...}
    [[maybe_unused]] constexpr uint32_t kSelPagesWord = 4;
    [[maybe_unused]] constexpr uint32_t sel_rows_word = get_arg(args::sel_rows_word);
    [[maybe_unused]] constexpr uint32_t sel_max_rows = get_arg(args::sel_max_rows);
    // Query row j's top-k goes to output rows [output_row_offset + j * output_row_stride, ... + k).
    [[maybe_unused]] constexpr uint32_t output_row_offset = get_arg(args::output_row_offset);
    [[maybe_unused]] constexpr uint32_t output_row_stride = get_arg(args::output_row_stride);
    [[maybe_unused]] constexpr uint32_t num_new_kv_rows = get_arg(args::num_new_kv_rows);
    constexpr uint32_t num_parts = get_arg(args::num_parts);
    [[maybe_unused]] constexpr uint32_t candidate_topk_blocks = get_arg(args::candidate_topk_blocks);
    [[maybe_unused]] constexpr uint32_t candidate_block_size = get_arg(args::candidate_block_size);
    // PARTIAL_OUTPUT candidates page: {k rows, k order keys, header}, then in CANDIDATE_SOURCE mode
    // {candidate_topk_blocks blocks, their order keys}. Header: {keys selected, blocks selected,
    // selectable keys, pad...}.
    [[maybe_unused]] constexpr uint32_t kCandHeaderWord = 2 * k;
    [[maybe_unused]] constexpr uint32_t kCandBlocksWord = 2 * k + 16;
    // cur_pos words from the reader: the most keys of any query row, the part, each row's keys.
    constexpr uint32_t kMaxKeysWord = 48;
    constexpr uint32_t kPartWord = 49;
    constexpr uint32_t kRowKeysWord = 64;

    // ---- Runtime args ----
    const uint32_t core_index = get_arg(args::core_index);
    const uint32_t num_cores = get_arg(args::num_cores);
    const uint32_t coord_x = get_arg(args::coord_x);
    const uint32_t coord_y = get_arg(args::coord_y);
    const uint32_t num_mcast_dests = get_arg(args::num_mcast_dests);
    // The host passes the rectangle as top-left -> bottom-right, which is NOC_0's direction. NOC_1
    // routes the other way, so its multicast start is the bottom-right corner.
    const bool mcast_reversed = noc_index == 1;
    const uint32_t mcast_x_start = mcast_reversed ? get_arg(args::mcast_x_end) : get_arg(args::mcast_x_start);
    const uint32_t mcast_y_start = mcast_reversed ? get_arg(args::mcast_y_end) : get_arg(args::mcast_y_start);
    const uint32_t mcast_x_end = mcast_reversed ? get_arg(args::mcast_x_start) : get_arg(args::mcast_x_end);
    const uint32_t mcast_y_end = mcast_reversed ? get_arg(args::mcast_y_start) : get_arg(args::mcast_y_end);

    // ---- Tensors ----
    // [rows, Dh] row-major, interleaved, one row per page; with PARTIAL_OUTPUT the one-page candidates.
    const auto output = TensorAccessor(tensor::output);
    [[maybe_unused]] const auto kv_cache =
        TensorAccessor(tensor::kv_cache);                // [num_blocks, 1, block_size, Dh] row-major
    const auto scores = TensorAccessor(tensor::scores);  // [1, 1, Sq, T] fp32, one row-major page per query row

    // ---- Dataflow buffers ----
    DataflowBuffer kv_wr_dfb(dfb::kv_wr);      // writer-local staging for gathered kv rows
    DataflowBuffer blocks_dfb(dfb::blocks);    // consumer <- reader (physical block of each local block)
    DataflowBuffer sel_dfb(dfb::sel);          // producer -> reader (selected kv_cache rows)
    DataflowBuffer cur_pos_dfb(dfb::cur_pos);  // consumer <- reader (selectable keys per query row)
    DataflowBuffer scores_dfb(dfb::scores);    // consumer <- compute
    DataflowBuffer hist_dfb(dfb::hist);        // writer-local scratch
    DataflowBuffer hist_gather_dfb(dfb::hist_gather);    // per-core counts around the thresholds, used on core 0
    DataflowBuffer select_dfb(dfb::select);              // top-k thresholds, multicast from core 0
    DataflowBuffer hist_scatter_dfb(dfb::hist_scatter);  // every core's copy of the bins this core owns
    DataflowBuffer hist_sum_dfb(dfb::hist_sum);          // global histograms, assembled on core 0
    DataflowBuffer cand_list_dfb(dfb::cand_list);        // writer-local: keys still in each row's running

    Semaphore hist_arrived(sem::hist_arrived);  // on core 0: summed bins, then per-core counts received
    // On every core: ready_base + pass + 1 = that pass's header arrived, ready_base + kAllocReady = the
    // allocation arrived. Never lowered between selects, so a slow core cannot miss a later select's header.
    Semaphore select_ready(sem::select_ready);
    Semaphore slice_arrived(sem::slice_arrived);  // on each owner: slices received

    Noc noc;

#ifdef HAS_NEW_KV_ROW
    // Write this step's closing entries into kv_cache before anything is selected. Every gather, on
    // any core, starts only after core 0's final allocation multicast, which follows these writes.
    // hist and kv_wr are free until the top-k and the gather, so they stage the lookups and the rows.
    if (core_index == 0) {
        DeviceZoneScopedN("SEL-NEW-ROW");
        [[maybe_unused]] const auto page_table = TensorAccessor(tensor::page_table);
        const auto new_kv_row = TensorAccessor(tensor::new_kv_row);
        const auto new_kv_row_index = TensorAccessor(tensor::new_kv_row_index);
        // scratch: the new rows' indices at word 0, page_table_row at word 32, a page-table chunk at 48.
        constexpr uint32_t kIndexReadBytes = (num_new_kv_rows * sizeof(uint32_t) + 63) / 64 * 64;
        static_assert(kIndexReadBytes <= 128, "new kv row indices must fit below page_table_row");
        hist_dfb.reserve_back(1);
        volatile tt_l1_ptr uint32_t* scratch = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hist_dfb.get_write_ptr());
        noc.async_read(new_kv_row_index, hist_dfb, kIndexReadBytes, {.page_id = 0}, {.offset_bytes = 0});
#ifdef PAGE_TABLE_ROW
        const auto page_table_row_gen = TensorAccessor(tensor::page_table_row);
        noc.async_read(page_table_row_gen, hist_dfb, 64, {.page_id = 0}, {.offset_bytes = 128});
#endif
        noc.async_read_barrier();
        invalidate_l1_cache();
#ifdef PAGE_TABLE_ROW
        [[maybe_unused]] const uint32_t page_table_row = scratch[32];
#else
        [[maybe_unused]] const uint32_t page_table_row = 0;
#endif
        uint32_t physical_rows[num_new_kv_rows];
#if defined(ROW_TABLE) && !defined(KV_PAGE_TABLE)
        // The page table maps the candidate pool, not kv_cache, so the indices are already kv_cache rows.
        for (uint32_t r = 0; r < num_new_kv_rows; ++r) {
            physical_rows[r] = scratch[r];
        }
#else
#ifdef ROW_TABLE
        // The page table maps the candidate pool; kv_page_table maps kv_cache.
        const auto row_page_table = TensorAccessor(tensor::kv_page_table);
#else
        const auto& row_page_table = page_table;
#endif
        // Each 64 B DRAM-aligned chunk of the page table holds 16 uint32_t entries.
        uint32_t loaded_chunk = UINT32_MAX;
        for (uint32_t r = 0; r < num_new_kv_rows; ++r) {
            const uint32_t logical_row = scratch[r];
            const uint32_t block = logical_row / page_block_size;
            if (block / 16 != loaded_chunk) {
                loaded_chunk = block / 16;
                noc.async_read(
                    row_page_table,
                    hist_dfb,
                    64,
                    {.page_id = page_table_row, .offset_bytes = 64 * loaded_chunk},
                    {.offset_bytes = 192});
                noc.async_read_barrier();
                invalidate_l1_cache();
            }
            physical_rows[r] = scratch[48 + block % 16] * page_block_size + logical_row % page_block_size;
        }
#endif

        kv_wr_dfb.reserve_back(kv_stage_rows);
        const uint32_t row_bytes = kv_wr_dfb.get_entry_size();
        for (uint32_t first = 0; first < num_new_kv_rows; first += kv_stage_rows) {
            const uint32_t n = std::min(kv_stage_rows, num_new_kv_rows - first);
            for (uint32_t r = 0; r < n; ++r) {
                noc.async_read(
                    new_kv_row, kv_wr_dfb, row_bytes, {.page_id = first + r}, {.offset_bytes = r * row_bytes});
            }
            noc.async_read_barrier();
            for (uint32_t r = 0; r < n; ++r) {
                noc.async_write(
                    kv_wr_dfb,
                    kv_cache,
                    row_bytes,
                    {.offset_bytes = r * row_bytes},
                    {.page_id = physical_rows[first + r]});
            }
            noc.async_write_barrier();
        }
    }
#endif

    // Same block split as the reader, which sends the selectable keys of every query row (the split
    // follows the row with the most) and this device's part.
    cur_pos_dfb.wait_front(1);
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* key_count = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cur_pos_dfb.get_read_ptr());
    const uint32_t num_keys = key_count[kMaxKeysWord];
    const uint32_t part = key_count[kPartWord];
    uint32_t row_keys[num_queries];
    for (uint32_t row = 0; row < num_queries; ++row) {
        row_keys[row] = key_count[kRowKeysWord + row];
    }
    cur_pos_dfb.pop_front(1);
    const uint32_t total_num_blocks = tt::data_movement::common::div_up(num_keys, page_block_size);
    const uint32_t blocks_per_part = tt::data_movement::common::div_up(total_num_blocks, num_parts);
    const uint32_t part_start = std::min(blocks_per_part * part, total_num_blocks);
    const uint32_t part_blocks = std::min(blocks_per_part, total_num_blocks - part_start);
    const uint32_t work_per_core = tt::data_movement::common::div_up(part_blocks, num_cores);
    const uint32_t start_block = part_start + std::min(work_per_core * core_index, part_blocks);
    const uint32_t end_block = std::min(start_block + work_per_core, part_start + part_blocks);

    // scores_dfb is sized to hold every tile this core produces and is never popped until the end, so
    // the tiles stay resident and contiguous from the read pointer. Compute emits one 1x32 tile per
    // (32-key chunk, query row), rows fastest: row j's local key i is score (i / 32 * Sq + j) * 32 + i % 32.
    const uint32_t score_tile_bytes = scores_dfb.get_entry_size();
    const uint32_t num_local_chunks = (end_block - start_block) * chunks_per_block;
    const uint32_t first_local_key = start_block * page_block_size;
    volatile tt_l1_ptr uint32_t* scores_l1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scores_dfb.get_read_ptr());
    // Keys of each query row on this core; a row's keys past its own count are left out (same as -inf).
    uint32_t num_local_keys[num_queries];
    for (uint32_t row = 0; row < num_queries; ++row) {
        num_local_keys[row] =
            row_keys[row] > first_local_key ? std::min(row_keys[row] - first_local_key, num_local_chunks * 32) : 0;
    }

#ifdef ROW_TABLE
    // kv_cache row of each local key, waited on before the selected keys are listed.
    const auto row_table = TensorAccessor(tensor::row_table);
    DataflowBuffer row_tbl_dfb(dfb::row_tbl);
    row_tbl_dfb.reserve_back(1);
    volatile tt_l1_ptr uint32_t* row_tbl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(row_tbl_dfb.get_write_ptr());
    if (num_local_chunks != 0) {
        noc.async_read(
            row_table,
            row_tbl_dfb,
            num_local_chunks * 32 * sizeof(uint32_t),
            {.page_id = 0, .offset_bytes = first_local_key * sizeof(uint32_t)},
            {.offset_bytes = 0});
    }
#endif

    hist_dfb.reserve_back(1);
    const uint32_t hist_addr = hist_dfb.get_write_ptr();
    uint32_t* const hist = reinterpret_cast<uint32_t*>(hist_addr);
    for (uint32_t w = 0; w < kHistWords; ++w) {
        hist[w] = 0;
    }

    {
        DeviceZoneScopedN("SEL-HIST");
        uint32_t tile = 0;
        for (uint32_t chunk = 0; chunk < num_local_chunks; ++chunk) {
            const uint32_t first_key = first_local_key + chunk * 32;
            for (uint32_t row = 0; row < num_queries; ++row, ++tile) {
                scores_dfb.wait_front(tile + 1);
                noc.async_write(
                    scores_dfb,
                    scores,
                    score_tile_bytes,
                    {.offset_bytes = tile * score_tile_bytes},
                    {.page_id = row, .offset_bytes = first_key * sizeof(float)});

                const uint32_t valid_in_tile =
                    first_key >= row_keys[row] ? 0 : std::min<uint32_t>(row_keys[row] - first_key, 32);
                volatile tt_l1_ptr uint32_t* tile_scores = scores_l1 + tile * 32;
                uint32_t* row_hist = hist + row * num_hist_bins;
                uint32_t i = 0;
                for (; i + 4 <= valid_in_tile; i += 4) {
                    const uint32_t b0 = topk_order_key(tile_scores[i + 0]) >> (32 - kHistBits);
                    const uint32_t b1 = topk_order_key(tile_scores[i + 1]) >> (32 - kHistBits);
                    const uint32_t b2 = topk_order_key(tile_scores[i + 2]) >> (32 - kHistBits);
                    const uint32_t b3 = topk_order_key(tile_scores[i + 3]) >> (32 - kHistBits);
                    ++row_hist[b0];
                    ++row_hist[b1];
                    ++row_hist[b2];
                    ++row_hist[b3];
                }
                for (; i < valid_in_tile; ++i) {
                    ++row_hist[topk_order_key(tile_scores[i]) >> (32 - kHistBits)];
                }
            }
        }
        noc.async_write_barrier();
    }

    // ---- Cross-core radix select ----
    // Every query row selects its own top-k, and all rows go through the passes together, so a pass
    // costs one round of barriers and multicasts however many rows are still selecting. Pass 0
    // histograms the top kHistBits of every valid key's order key. Each later pass narrows a row to the
    // keys whose already-fixed prefix equals its threshold bin found so far, and histograms their next
    // kHistBits. A row stops once its threshold bin can be taken whole or all 32 bits are fixed, so only
    // keys with equal scores are left to the index-order tie-break.
    //
    // Blocks go to cores in order, so exactly cores [0, num_active) have keys; the others' histograms
    // are all zero and take no part. Per pass, each active core bumps hist_arrived on the coordinator
    // once its histograms are final; when all have, the coordinator releases every core. Each owner
    // reads its range of the histograms from every active core into entry c of hist_scatter, sums the
    // rows still selecting, writes the sums and their slice totals into hist_sum on the coordinator and
    // bumps hist_arrived again. The coordinator then walks each row's slice totals and bins down to its
    // threshold and multicasts the pass header with select_ready = ready_base + pass + 1.
    //
    // After each header, every active core rescans the keys it histogrammed for each row: it counts
    // those above the threshold bin, keeps those in it in cand_list for the next pass, and clears the
    // bins it filled, so the histograms never need a full clear.
    const uint32_t num_active = work_per_core == 0 ? 0 : tt::data_movement::common::div_up(part_blocks, work_per_core);
    const bool is_active = core_index < num_active;
    const bool is_owner = core_index < num_owners;
    const uint32_t gather_addr = hist_gather_dfb.get_write_ptr();
    const uint32_t scatter_addr = hist_scatter_dfb.get_write_ptr();
    const uint32_t sum_addr = hist_sum_dfb.get_write_ptr();
    const uint32_t select_addr = select_dfb.get_write_ptr();
    volatile tt_l1_ptr uint32_t* select = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(select_addr);
    cand_list_dfb.reserve_back(1);
    uint16_t* const cand_list = reinterpret_cast<uint16_t*>(cand_list_dfb.get_write_ptr());

    // Selects, for every query row j, the k_select largest order keys key_at(j, i), i < num_local[j],
    // over every active core. hist must already hold this core's pass-0 histograms of them. Every core
    // gets its share of each row in results, copied out of select, which the next select overwrites;
    // that select passes this one's ready_base + kAllocReady.
    auto radix_select =
        [&](uint32_t k_select, const uint32_t* num_local, uint32_t ready_base, auto key_at, SelectResult* results) {
            // This core's keys above each row's final threshold and equal to it, counted by the rescans.
            uint32_t core_above[num_queries] = {};
            uint32_t core_bin_count[num_queries] = {};
            uint32_t cand_count[num_queries] = {};

            if (core_index == 0) {
                for (uint32_t row = 0; row < num_queries; ++row) {
                    volatile tt_l1_ptr uint32_t* hdr = select + kSelectRowsWord + row * kSelectHeaderWords;
                    hdr[kHdrPrefix] = 0;
                    hdr[kHdrPrefixBits] = 0;
                    hdr[kHdrAbove] = 0;
                }
            }

            // Inactive non-owners have no part in the passes and only wait for the final allocation.
            const bool in_passes = is_active || is_owner || core_index == 0;
            uint32_t live = in_passes ? kAllRows : 0;
            for (uint32_t pass = 0; live != 0; ++pass) {
                // RISC-V stores can still be in flight when the NoC reads the buffer; a store followed by
                // a blocking load of the same word drains them all, since stores retire in order.
                if (is_active) {
                    DeviceZoneScopedN("SEL-SIGNAL");
                    asm volatile("" ::: "memory");
                    hist[kHistWords - 1] = hist[kHistWords - 1];
                    (void)ckernel::load_blocking(&hist[kHistWords - 1]);
                    hist_arrived.up(noc, coord_x, coord_y, 1);
                    noc.async_atomic_barrier();
                }

                // Concurrent multicast atomics from many cores serialize badly, so the coordinator collects
                // the active cores' signals and releases everyone with a single multicast of slice_arrived = 1.
                if (core_index == 0) {
                    {
                        DeviceZoneScopedN("SEL-READY-WAIT");
                        hist_arrived.wait(num_active);
                    }
                    hist_arrived.set(0);
                    if (num_mcast_dests > 0) {
                        DeviceZoneScopedN("SEL-RELEASE");
                        slice_arrived.set(1);
                        slice_arrived.set_multicast(
                            noc, mcast_x_start, mcast_y_start, mcast_x_end, mcast_y_end, num_mcast_dests);
                        noc.async_write_barrier();
                    }
                } else if (is_owner) {
                    DeviceZoneScopedN("SEL-OWNER-WAIT");
                    slice_arrived.wait(1);
                }
                slice_arrived.set(0);

                if (is_owner) {
                    DeviceZoneScopedN("SEL-OWNER-SUM");
                    const uint32_t first_word = core_index * owner_words;
                    const uint32_t my_words = std::min(owner_words, kHistWords - first_word);
                    for (uint32_t c = 0; c < num_active; ++c) {
                        noc.async_read(
                            UnicastEndpoint{},
                            UnicastEndpoint{},
                            my_words * sizeof(uint32_t),
                            {.noc_x = get_common_vararg(2 * c),
                             .noc_y = get_common_vararg(2 * c + 1),
                             .addr = hist_addr + first_word * sizeof(uint32_t)},
                            {.addr = scatter_addr + c * owner_chunk_words * sizeof(uint32_t)});
                    }
                    noc.async_read_barrier();
                    invalidate_l1_cache();
                    asm volatile("" ::: "memory");
                    // Entry 0 has been read for a slice before the slice's sums overwrite it, and its slice
                    // totals go past owner_words, which no read touches. Slices of rows that are done are
                    // skipped; the coordinator does not read them.
                    uint32_t* const scattered = reinterpret_cast<uint32_t*>(scatter_addr);
                    const uint32_t my_slices = my_words / bins_per_slice;
                    uint32_t slice = core_index * slices_per_owner;
                    for (uint32_t s = 0; s < my_slices; ++s, ++slice) {
                        if (((live >> (slice / kRowSlices)) & 1) == 0) {
                            continue;
                        }
                        uint32_t s0 = 0;
                        uint32_t s1 = 0;
                        uint32_t s2 = 0;
                        uint32_t s3 = 0;
                        const uint32_t* entry = scattered + s * bins_per_slice;
                        for (uint32_t c = 0; c < num_active; ++c, entry += owner_chunk_words) {
                            s0 += entry[0];
                            s1 += entry[1];
                            s2 += entry[2];
                            s3 += entry[3];
                        }
                        uint32_t* sums = scattered + s * bins_per_slice;
                        sums[0] = s0;
                        sums[1] = s1;
                        sums[2] = s2;
                        sums[3] = s3;
                        scattered[owner_words + s] = s0 + s1 + s2 + s3;
                    }
                    asm volatile("" ::: "memory");
                    scattered[owner_chunk_words - 1] = scattered[owner_chunk_words - 1];
                    (void)ckernel::load_blocking(&scattered[owner_chunk_words - 1]);
                    noc.async_write(
                        UnicastEndpoint{},
                        UnicastEndpoint{},
                        owner_chunk_words * sizeof(uint32_t),
                        {.addr = scatter_addr},
                        {.noc_x = coord_x,
                         .noc_y = coord_y,
                         .addr = sum_addr + core_index * owner_chunk_words * sizeof(uint32_t)});
                    noc.async_write_barrier();
                    hist_arrived.up(noc, coord_x, coord_y, 1);
                    noc.async_atomic_barrier();
                }

                if (core_index == 0) {
                    {
                        DeviceZoneScopedN("SEL-COORD-WAIT");
                        hist_arrived.wait(num_owners);
                    }
                    hist_arrived.set(0);
                    invalidate_l1_cache();
                    asm volatile("" ::: "memory");
                    DeviceZoneScopedN("SEL-COORD-WALK");
                    const uint32_t* const sums = reinterpret_cast<const uint32_t*>(sum_addr);
                    uint32_t next_live = 0;
                    for (uint32_t rows = live; rows != 0; rows &= rows - 1) {
                        const uint32_t row = __builtin_ctz(rows);
                        volatile tt_l1_ptr uint32_t* hdr = select + kSelectRowsWord + row * kSelectHeaderWords;
                        // Row slices are global slices [row * kRowSlices, (row + 1) * kRowSlices); slice g
                        // is local slice g % slices_per_owner of owner g / slices_per_owner.
                        const uint32_t top_slice = row * kRowSlices + kRowSlices - 1;
                        uint32_t target;
                        if (pass == 0) {
                            const uint32_t first_slice = row * kRowSlices;
                            uint32_t owner = first_slice / slices_per_owner;
                            uint32_t local = first_slice - owner * slices_per_owner;
                            uint32_t total = 0;
                            for (uint32_t s = 0; s < kRowSlices; ++s) {
                                total += sums[owner * owner_chunk_words + owner_words + local];
                                if (++local == slices_per_owner) {
                                    local = 0;
                                    ++owner;
                                }
                            }
                            hdr[kHdrTotal] = total;
                            target = std::min<uint32_t>(k_select, total);
                        } else {
                            target = hdr[kHdrNeeded];
                        }
                        // Walk slice totals, then the bins of the slice where the count is reached, from the
                        // largest scores down until the running count reaches the keys still needed. The bin
                        // where that happens holds the k-th largest score.
                        uint32_t owner = top_slice / slices_per_owner;
                        uint32_t local = top_slice - owner * slices_per_owner;
                        uint32_t above = 0;
                        uint32_t slice = kRowSlices - 1;
                        while (true) {
                            const uint32_t slice_total = sums[owner * owner_chunk_words + owner_words + local];
                            if (above + slice_total >= target || slice == 0) {
                                break;
                            }
                            above += slice_total;
                            --slice;
                            if (local == 0) {
                                local = slices_per_owner - 1;
                                --owner;
                            } else {
                                --local;
                            }
                        }
                        const uint32_t* bins = sums + owner * owner_chunk_words + local * bins_per_slice;
                        uint32_t bin = bins_per_slice - 1;
                        while (bin > 0 && above + bins[bin] < target) {
                            above += bins[bin];
                            --bin;
                        }
                        const uint32_t threshold_bin = slice * bins_per_slice + bin;
                        const uint32_t bin_count = bins[bin];
                        const uint32_t needed = target - above;
                        const uint32_t prefix_bits = hdr[kHdrPrefixBits] + kHistBits;
                        hdr[kHdrPrefix] = (hdr[kHdrPrefix] << kHistBits) | threshold_bin;
                        hdr[kHdrPrefixBits] = prefix_bits;
                        hdr[kHdrAbove] = hdr[kHdrAbove] + above;
                        hdr[kHdrNeeded] = needed;
                        if (needed != 0 && needed != bin_count && prefix_bits != 32) {
                            next_live |= 1u << row;
                        }
                    }
                    select[kSelectLiveWord] = next_live;
                    (void)ckernel::load_blocking(&select[kSelectLiveWord]);

                    if (num_mcast_dests > 0) {
                        noc.async_write_multicast(
                            UnicastEndpoint{},
                            MulticastEndpoint{},
                            kSelectAllocWord * sizeof(uint32_t),
                            num_mcast_dests,
                            {.addr = select_addr},
                            {.noc_x_start = mcast_x_start,
                             .noc_y_start = mcast_y_start,
                             .noc_x_end = mcast_x_end,
                             .noc_y_end = mcast_y_end,
                             .addr = select_addr});
                        noc.async_write_barrier();
                        select_ready.set(ready_base + pass + 1);
                        select_ready.set_multicast(
                            noc, mcast_x_start, mcast_y_start, mcast_x_end, mcast_y_end, num_mcast_dests);
                        noc.async_write_barrier();
                    }
                } else {
                    // The allocation keeps the header words, so a core that sees it instead of this pass's
                    // header reads the same final header.
                    select_ready.wait_min(ready_base + pass + 1);
                    invalidate_l1_cache();
                }
                const uint32_t next_live = select[kSelectLiveWord];

                if (is_active) {
                    DeviceZoneScopedN("SEL-REFINE-HIST");
                    for (uint32_t rows = live; rows != 0; rows &= rows - 1) {
                        const uint32_t row = __builtin_ctz(rows);
                        volatile tt_l1_ptr uint32_t* hdr = select + kSelectRowsWord + row * kSelectHeaderWords;
                        const uint32_t shift = 32 - hdr[kHdrPrefixBits];
                        const uint32_t threshold_bin = hdr[kHdrPrefix] & (num_hist_bins - 1);
                        uint32_t* row_hist = hist + row * num_hist_bins;
                        uint16_t* cand = cand_list + row * local_key_capacity;
                        uint32_t above = 0;
                        uint32_t kept = 0;
                        // The same keys and bits as this pass's histogram: pass 0 had every local key,
                        // later passes the keys kept by the previous one.
                        const uint32_t num_scanned = pass == 0 ? num_local[row] : cand_count[row];
                        for (uint32_t t = 0; t < num_scanned; ++t) {
                            const uint32_t i = pass == 0 ? t : cand[t];
                            const uint32_t bin = (key_at(row, i) >> shift) & (num_hist_bins - 1);
                            if (bin > threshold_bin) {
                                ++above;
                            } else if (bin == threshold_bin) {
                                cand[kept++] = i;
                            }
                            row_hist[bin] = 0;
                        }
                        core_above[row] += above;
                        core_bin_count[row] = kept;
                        cand_count[row] = kept;
                        if ((next_live >> row) & 1) {
                            const uint32_t next_shift = shift - kHistBits;
                            for (uint32_t t = 0; t < kept; ++t) {
                                ++row_hist[(key_at(row, cand[t]) >> next_shift) & (num_hist_bins - 1)];
                            }
                        }
                    }
                }
                live = next_live;
            }

            // Every active core reports its keys above each row's threshold and equal to it to slot
            // core_index of hist_gather on the coordinator. Inactive cores have neither. Every owner has
            // read the histograms by the last header, so they stage the message.
            if (is_active) {
                for (uint32_t row = 0; row < num_queries; ++row) {
                    hist[2 * row] = core_above[row];
                    hist[2 * row + 1] = core_bin_count[row];
                }
                asm volatile("" ::: "memory");
                (void)ckernel::load_blocking(&hist[2 * num_queries - 1]);
                noc.async_write(
                    UnicastEndpoint{},
                    UnicastEndpoint{},
                    kCountsWords * sizeof(uint32_t),
                    {.addr = hist_addr},
                    {.noc_x = coord_x,
                     .noc_y = coord_y,
                     .addr = gather_addr + core_index * kCountsWords * sizeof(uint32_t)});
                noc.async_write_barrier();
                hist_arrived.up(noc, coord_x, coord_y, 1);
                noc.async_atomic_barrier();
            }

            if (core_index == 0) {
                hist_arrived.wait(num_active);
                hist_arrived.set(0);
                invalidate_l1_cache();
                volatile tt_l1_ptr uint32_t* gathered = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gather_addr);

                // Per-core allocation of each row. A core contributes all of its keys above the threshold
                // plus a share of the keys equal to it; that share is handed out in core order, so ties go
                // to lower key indices. Output offsets are the running sum of contributions.
                for (uint32_t row = 0; row < num_queries; ++row) {
                    uint32_t remaining_from_bin = select[kSelectRowsWord + row * kSelectHeaderWords + kHdrNeeded];
                    uint32_t next_offset = 0;
                    for (uint32_t c = 0; c < num_active; ++c) {
                        volatile tt_l1_ptr uint32_t* counts = gathered + c * kCountsWords + 2 * row;
                        const uint32_t in_bin = counts[1];
                        const uint32_t core_from_bin = std::min(in_bin, remaining_from_bin);
                        remaining_from_bin -= core_from_bin;
                        const uint32_t contribute = counts[0] + core_from_bin;
                        volatile tt_l1_ptr uint32_t* entry =
                            select + kSelectAllocWord + (c * num_queries + row) * kSelectWordsPerCore;
                        entry[0] = contribute;
                        entry[1] = core_from_bin;
                        entry[2] = next_offset;
                        next_offset += contribute;
                    }
                }
                // Only active cores read their entries. The last entry stored is the last active core's
                // entry of the last row (a header word when no core is active, stored long before).
                const uint32_t alloc_end = kSelectAllocWord + num_active * num_queries * kSelectWordsPerCore;
                const uint32_t alloc_words = std::min(num_select_words, (alloc_end + 3) / 4 * 4);
                (void)ckernel::load_blocking(&select[alloc_end - 1]);

                if (num_mcast_dests > 0) {
                    noc.async_write_multicast(
                        UnicastEndpoint{},
                        MulticastEndpoint{},
                        alloc_words * sizeof(uint32_t),
                        num_mcast_dests,
                        {.addr = select_addr},
                        {.noc_x_start = mcast_x_start,
                         .noc_y_start = mcast_y_start,
                         .noc_x_end = mcast_x_end,
                         .noc_y_end = mcast_y_end,
                         .addr = select_addr});
                    noc.async_write_barrier();
                    select_ready.set(ready_base + kAllocReady);
                    select_ready.set_multicast(
                        noc, mcast_x_start, mcast_y_start, mcast_x_end, mcast_y_end, num_mcast_dests);
                    noc.async_write_barrier();
                }
            } else {
                select_ready.wait_min(ready_base + kAllocReady);
                invalidate_l1_cache();
            }

            // This core's assignment per row: num_contribute keys (all local keys above the threshold plus
            // num_from_bin keys equal to it) at [output_offset, output_offset + num_contribute).
            for (uint32_t row = 0; row < num_queries; ++row) {
                volatile tt_l1_ptr uint32_t* hdr = select + kSelectRowsWord + row * kSelectHeaderWords;
                results[row].final_prefix = hdr[kHdrPrefix];
                results[row].final_shift = 32 - hdr[kHdrPrefixBits];
                if (is_active) {
                    volatile tt_l1_ptr uint32_t* my_entry =
                        select + kSelectAllocWord + (core_index * num_queries + row) * kSelectWordsPerCore;
                    results[row].num_contribute = my_entry[0];
                    results[row].num_from_bin = my_entry[1];
                    results[row].output_offset = my_entry[2];
                } else {
                    results[row].num_contribute = 0;
                    results[row].num_from_bin = 0;
                    results[row].output_offset = 0;
                }
            }
        };

    // Every score tile has been waited on by now and stays resident, so the rescans read them through
    // a plain pointer.
    const uint32_t* local_scores = const_cast<const uint32_t*>(scores_l1);
    auto key_at = [local_scores](uint32_t row, uint32_t i) {
        return topk_order_key(local_scores[((i / 32 * num_queries) + row) * 32 + i % 32]);
    };
    SelectResult key_shares[num_queries];
    radix_select(k, num_local_keys, /*ready_base=*/0, key_at, key_shares);

    // ---- KV gather ----
    blocks_dfb.wait_front(1);
    invalidate_l1_cache();
    [[maybe_unused]] volatile tt_l1_ptr uint32_t* physical_blocks =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(blocks_dfb.get_read_ptr());
#ifdef ROW_TABLE
    noc.async_read_barrier();
    invalidate_l1_cache();
#endif
#ifdef PARTIAL_OUTPUT
    static_assert(num_queries == 1, "a partial top-k selects for one query row");
    // A later select overwrites the header.
    const uint32_t num_keys_selected = select[kSelectRowsWord + kHdrAbove] + select[kSelectRowsWord + kHdrNeeded];
    const SelectResult& key_share = key_shares[0];
    // List the selected keys in local key order as kv_cache row ids, taking the first num_from_bin
    // keys equal to the threshold. Each core lists its runs straight into the candidates page's layout
    // (see kCandHeaderWord), at the same kv_wr address it then writes to on core 0.
    kv_wr_dfb.reserve_back(kv_stage_rows);
    const uint32_t cand_addr = kv_wr_dfb.get_write_ptr();
    volatile tt_l1_ptr uint32_t* cand = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cand_addr);
    volatile tt_l1_ptr uint32_t* rows = cand + key_share.output_offset;
    volatile tt_l1_ptr uint32_t* keys = cand + k + key_share.output_offset;
    uint32_t num_rows = 0;
    {
        DeviceZoneScopedN("SEL-LIST");
        uint32_t from_bin_left = key_share.num_from_bin;
        for (uint32_t i = 0; i < num_local_keys[0] && num_rows < key_share.num_contribute; ++i) {
            const uint32_t key = key_at(0, i);
            const uint32_t top = key >> key_share.final_shift;
            bool take = top > key_share.final_prefix;
            if (top == key_share.final_prefix && from_bin_left != 0) {
                --from_bin_left;
                take = true;
            }
            if (take) {
#ifdef ROW_TABLE
                rows[num_rows] = row_tbl[i];
#else
                rows[num_rows] = physical_blocks[i / page_block_size] * page_block_size + i % page_block_size;
#endif
                keys[num_rows] = key;
                ++num_rows;
            }
        }
    }

    // Every active core copies its runs into core 0's page, at the offsets it listed them at locally;
    // core 0 then zeroes the unused entries, fills the header and writes the page. A run is `count`
    // ids from word `first`, with their order keys `keys_stride` words later.
    auto write_partial = [&](uint32_t first_block, uint32_t num_blocks_listed, uint32_t num_blocks_selected) {
        DeviceZoneScopedN("SEL-PARTIAL");
        // hist_arrived is back at zero: core 0 consumed the last counts before the allocation.
        if (core_index != 0 && is_active) {
            auto send_run = [&](uint32_t first, uint32_t count, uint32_t keys_stride) {
                if (count == 0) {
                    return;
                }
                (void)ckernel::load_blocking(&cand[first + keys_stride + count - 1]);
                const uint32_t ids_addr = cand_addr + first * sizeof(uint32_t);
                const uint32_t keys_addr = ids_addr + keys_stride * sizeof(uint32_t);
                noc.async_write(
                    UnicastEndpoint{},
                    UnicastEndpoint{},
                    count * sizeof(uint32_t),
                    {.addr = ids_addr},
                    {.noc_x = coord_x, .noc_y = coord_y, .addr = ids_addr});
                noc.async_write(
                    UnicastEndpoint{},
                    UnicastEndpoint{},
                    count * sizeof(uint32_t),
                    {.addr = keys_addr},
                    {.noc_x = coord_x, .noc_y = coord_y, .addr = keys_addr});
            };
            send_run(key_share.output_offset, num_rows, k);
            send_run(kCandBlocksWord + first_block, num_blocks_listed, candidate_topk_blocks);
            noc.async_write_barrier();
            hist_arrived.up(noc, coord_x, coord_y, 1);
            noc.async_atomic_barrier();
        }
        if (core_index == 0) {
            if (num_active > 1) {
                hist_arrived.wait(num_active - 1);
                hist_arrived.set(0);
            }
            invalidate_l1_cache();
            for (uint32_t j = num_keys_selected; j < k; ++j) {
                cand[j] = 0;
                cand[k + j] = 0;
            }
            for (uint32_t j = num_blocks_selected; j < candidate_topk_blocks; ++j) {
                cand[kCandBlocksWord + j] = 0;
                cand[kCandBlocksWord + candidate_topk_blocks + j] = 0;
            }
            for (uint32_t j = 0; j < 16; ++j) {
                cand[kCandHeaderWord + j] = 0;
            }
            cand[kCandHeaderWord] = num_keys_selected;
            cand[kCandHeaderWord + 1] = num_blocks_selected;
            cand[kCandHeaderWord + 2] = num_keys;
            (void)ckernel::load_blocking(&cand[kCandHeaderWord + 15]);
            noc.async_write(
                kv_wr_dfb,
                output,
                (kCandBlocksWord + 2 * candidate_topk_blocks) * sizeof(uint32_t),
                {.offset_bytes = 0},
                {.page_id = 0, .offset_bytes = 0});
            noc.async_write_barrier();
        }
    };
#ifndef CANDIDATE_SOURCE
    write_partial(0, 0, 0);
#endif
#else
    // List this core's selected keys once each, in local key order, as kv_cache row ids with the mask
    // of query rows that selected them; a row takes the first num_from_bin of its keys equal to its
    // threshold. The reader gathers the first half of the list on its NoC, this kernel the rest on the
    // other one, so a kv_cache row selected by several query rows is read once.
    sel_dfb.reserve_back(1);
    volatile tt_l1_ptr uint32_t* sel = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sel_dfb.get_write_ptr());
    volatile tt_l1_ptr uint32_t* rows = sel + sel_rows_word;
    volatile tt_l1_ptr uint32_t* masks = rows + sel_max_rows;
    uint32_t num_rows = 0;
    {
        DeviceZoneScopedN("SEL-LIST");
        uint32_t left[num_queries];
        uint32_t from_bin_left[num_queries];
        uint32_t listing = 0;  // rows with keys left to list
        uint32_t scan_end = 0;
        for (uint32_t row = 0; row < num_queries; ++row) {
            left[row] = key_shares[row].num_contribute;
            from_bin_left[row] = key_shares[row].num_from_bin;
            if (left[row] != 0) {
                listing |= 1u << row;
                scan_end = std::max(scan_end, num_local_keys[row]);
            }
        }
        for (uint32_t i = 0; i < scan_end && listing != 0; ++i) {
            uint32_t mask = 0;
            for (uint32_t pending = listing; pending != 0; pending &= pending - 1) {
                const uint32_t row = __builtin_ctz(pending);
                if (i >= num_local_keys[row]) {
                    continue;
                }
                const uint32_t top = key_at(row, i) >> key_shares[row].final_shift;
                bool take = top > key_shares[row].final_prefix;
                if (top == key_shares[row].final_prefix && from_bin_left[row] != 0) {
                    --from_bin_left[row];
                    take = true;
                }
                if (take) {
                    mask |= 1u << row;
                    if (--left[row] == 0) {
                        listing &= ~(1u << row);
                    }
                }
            }
            if (mask != 0) {
#ifdef ROW_TABLE
                rows[num_rows] = row_tbl[i];
#else
                rows[num_rows] = physical_blocks[i / page_block_size] * page_block_size + i % page_block_size;
#endif
                masks[num_rows] = mask;
                ++num_rows;
            }
        }
    }

    // The reader's first output page for each row, then this kernel's, past the reader's half.
    uint32_t out_pages[num_queries];
    for (uint32_t row = 0; row < num_queries; ++row) {
        out_pages[row] = output_row_offset + row * output_row_stride + key_shares[row].output_offset;
        sel[kSelPagesWord + row] = out_pages[row];
    }
    for (uint32_t e = 0; e < num_rows / 2; ++e) {
        for (uint32_t mask = masks[e]; mask != 0; mask &= mask - 1) {
            ++out_pages[__builtin_ctz(mask)];
        }
    }
    sel[0] = num_rows;
    (void)ckernel::load_blocking(&sel[0]);
    sel_dfb.push_back(1);

    {
        DeviceZoneScopedN("SEL-GATHER-WR");
        gather_selected_kv_rows<kv_stage_rows, num_queries>(
            noc, kv_cache, output, kv_wr_dfb, rows, masks, num_rows / 2, num_rows, out_pages);
    }
#endif

#ifdef CANDIDATE_SOURCE
    static_assert(num_queries == 1, "candidate blocks are selected for one query row");
    // ---- Candidate blocks ----
    // Keys split into blocks of candidate_block_size, each scored by its best key; the block holding
    // the newest key is only partly filled and always kept. The candidate_topk_blocks best blocks go
    // to candidate_blocks in ascending order (zero past the last), and pool_count is the number of
    // valid keys in them. Page blocks are whole multiples of a candidate block, so no candidate block
    // spans two cores (or two parts). With PARTIAL_OUTPUT the best blocks of this part go into the
    // candidates page with their order keys instead, for merge_select_kv to pick the global best.
    {
        DeviceZoneScopedN("SEL-CAND");
        const uint32_t num_blocks = tt::data_movement::common::div_up(num_keys, candidate_block_size);
        const uint32_t pinned_block = num_blocks - 1;
        const uint32_t first_local_block = first_local_key / candidate_block_size;
        const uint32_t num_local_blocks = tt::data_movement::common::div_up(num_local_keys[0], candidate_block_size);

        // Every owner has read this core's histogram of the key select, so it can be refilled. The
        // scores are no longer needed either: block b's order key overwrites score b, which only
        // blocks <= b have read.
        for (uint32_t bin = 0; bin < num_hist_bins; ++bin) {
            hist[bin] = 0;
        }
        for (uint32_t b = 0; b < num_local_blocks; ++b) {
            uint32_t best = 0xFFFFFFFFu;
            if (first_local_block + b != pinned_block) {
                const uint32_t end = std::min((b + 1) * candidate_block_size, num_local_keys[0]);
                best = 0;
                for (uint32_t i = b * candidate_block_size; i < end; ++i) {
                    best = std::max(best, topk_order_key(scores_l1[i]));
                }
            }
            scores_l1[b] = best;
            ++hist[best >> (32 - kHistBits)];
        }
        const uint32_t* block_keys = local_scores;
        auto block_key = [block_keys](uint32_t, uint32_t b) { return block_keys[b]; };
        SelectResult block_share;
        radix_select(candidate_topk_blocks, &num_local_blocks, /*ready_base=*/kAllocReady, block_key, &block_share);
        [[maybe_unused]] const uint32_t num_blocks_selected =
            select[kSelectRowsWord + kHdrAbove] + select[kSelectRowsWord + kHdrNeeded];

#ifdef PARTIAL_OUTPUT
        // The blocks go after this core's key run in the same staging page.
        volatile tt_l1_ptr uint32_t* list = cand + kCandBlocksWord;
#else
        // kv_wr is free once this core's gather is done, and core 0's is done before the allocation,
        // so every core lists its blocks at the same L1 offsets it then writes to on core 0.
        const uint32_t list_addr = kv_wr_dfb.get_write_ptr();
        volatile tt_l1_ptr uint32_t* list = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(list_addr);
#endif
        uint32_t num_listed = 0;
        uint32_t from_bin_left = block_share.num_from_bin;
        for (uint32_t b = 0; b < num_local_blocks && num_listed < block_share.num_contribute; ++b) {
            const uint32_t key = block_key(0, b);
            const uint32_t top = key >> block_share.final_shift;
            bool take = top > block_share.final_prefix;
            if (top == block_share.final_prefix && from_bin_left != 0) {
                --from_bin_left;
                take = true;
            }
            if (take) {
#ifdef PARTIAL_OUTPUT
                list[candidate_topk_blocks + block_share.output_offset + num_listed] = key;
#endif
                list[block_share.output_offset + num_listed++] = first_local_block + b;
            }
        }

#ifdef PARTIAL_OUTPUT
        write_partial(block_share.output_offset, num_listed, num_blocks_selected);
#else
        constexpr uint32_t kCountOffsetWords = (candidate_topk_blocks * sizeof(uint32_t) + 63) / 64 * 16;
        const auto candidate_blocks = TensorAccessor(tensor::candidate_blocks);
        const auto pool_count = TensorAccessor(tensor::pool_count);
        if (core_index != 0 && is_active) {
            if (num_listed != 0) {
                const uint32_t addr = list_addr + block_share.output_offset * sizeof(uint32_t);
                (void)ckernel::load_blocking(&list[block_share.output_offset + num_listed - 1]);
                noc.async_write(
                    UnicastEndpoint{},
                    UnicastEndpoint{},
                    num_listed * sizeof(uint32_t),
                    {.addr = addr},
                    {.noc_x = coord_x, .noc_y = coord_y, .addr = addr});
                noc.async_write_barrier();
            }
            hist_arrived.up(noc, coord_x, coord_y, 1);
            noc.async_atomic_barrier();
        }

        if (core_index == 0) {
            if (num_active > 1) {
                hist_arrived.wait(num_active - 1);
                hist_arrived.set(0);
            }
            const uint32_t num_chosen = std::min(candidate_topk_blocks, num_blocks);
            for (uint32_t j = num_chosen; j < candidate_topk_blocks; ++j) {
                list[j] = 0;
            }
            // Every chosen block but the pinned one, which is last, is full.
            const uint32_t pinned_keys = num_keys - pinned_block * candidate_block_size;
            list[kCountOffsetWords] = num_chosen == 0 ? 0 : (num_chosen - 1) * candidate_block_size + pinned_keys;
            (void)ckernel::load_blocking(&list[kCountOffsetWords]);
            noc.async_write(
                kv_wr_dfb,
                candidate_blocks,
                candidate_topk_blocks * sizeof(uint32_t),
                {.offset_bytes = 0},
                {.page_id = 0, .offset_bytes = 0});
            noc.async_write(
                kv_wr_dfb,
                pool_count,
                sizeof(uint32_t),
                {.offset_bytes = kCountOffsetWords * sizeof(uint32_t)},
                {.page_id = 0, .offset_bytes = 0});
            noc.async_write_barrier();
        }
#endif
    }
#endif

    blocks_dfb.pop_front(1);
    scores_dfb.pop_front(num_local_chunks * num_queries);
    select_ready.set(0);
}
