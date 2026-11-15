// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writes this core's index scores to the scores output, selects every query row's global top-k with
// the other cores, then gathers the second half of this core's selected kv_cache rows into the
// output. With PARTIAL_OUTPUT the top-k is this device's part only and is written out as candidates
// instead, and in CANDIDATE_SOURCE mode so are the part's best candidate blocks.

#include <algorithm>
#include <cstdint>
#include <functional>
#include <type_traits>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "ckernel.h"
#include "api/debug/kernel_profiler.hpp"
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
    // Once at most max_ties keys of a row are left in its threshold bin, the cores send them to the
    // coordinator, which finds the exact threshold, instead of running more passes.
    constexpr uint32_t kMaxTies = get_arg(args::max_ties);
    // Per query row {keys above its threshold bin, keys in it, then those keys if the row ends on ties},
    // padded to 16 B for NoC alignment.
    constexpr uint32_t kRowCountsWords = 2 + kMaxTies;
    constexpr uint32_t kCountsWords = (kRowCountsWords + 3) / 4 * 4;
    constexpr uint32_t num_select_words = get_arg(args::num_select_words);
    // select: {query rows still selecting, pad x7}, then per query row a header {prefix, prefix bits,
    // keys above it, keys still needed from it, resolved from ties, pad, pad, pad}, then the allocation.
    constexpr uint32_t kSelectLiveWord = 0;
    constexpr uint32_t kSelectRowsWord = 8;
    constexpr uint32_t kSelectHeaderWords = 8;
    constexpr uint32_t kHdrPrefix = 0;
    constexpr uint32_t kHdrPrefixBits = 1;
    constexpr uint32_t kHdrAbove = 2;
    constexpr uint32_t kHdrNeeded = 3;
    constexpr uint32_t kHdrTies = 4;  // the row's last bin holds at most kMaxTies keys: resolved from them
    constexpr uint32_t kSelectAllocWord = kSelectRowsWord + num_queries * kSelectHeaderWords;
    // Allocation entry of (core c, query row j) at kSelectAllocWord + j * alloc_row_words + c * 3:
    // {keys to contribute, of which equal to the threshold, output offset}.
    constexpr uint32_t kSelectWordsPerCore = 3;
    // Query row j's allocation is computed by core j, which writes it and row j's header to core 0 and
    // bumps alloc_done there; core 0 multicasts them all with select_ready = ready_base + kAllocReady.
    constexpr uint32_t alloc_row_words = get_arg(args::alloc_row_words);
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
    // Query row j's candidates page starts j * cand_page_words words into the staging buffer.
    [[maybe_unused]] constexpr uint32_t cand_page_words = get_arg(args::cand_page_words);
    // ROW_MASK: keys per row_mask entry (a candidate block of the pool).
    [[maybe_unused]] constexpr uint32_t row_mask_group = get_arg(args::row_mask_group);
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
    Semaphore slice_arrived(sem::slice_arrived);
    Semaphore alloc_done(sem::alloc_done);  // on core 0: rows other than row 0 allocated by their cores  // on each
                                            // owner: slices received

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

#ifdef ROW_MASK
    // Rows that may select each local key, one bit per query row, per row_mask_group keys: a pool shared
    // by several rows, each of which sees only its own candidate blocks. Keys a row may not select are
    // left out of its histograms and get order key 0 (score bits 0xFFFFFFFF), below every real score.
    const auto row_mask = TensorAccessor(tensor::row_mask);
    DataflowBuffer row_mask_dfb(dfb::row_mask_l1);
    row_mask_dfb.reserve_back(1);
    const uint32_t mask_first = first_local_key / row_mask_group;
    const uint32_t mask_read_first = mask_first * sizeof(uint32_t) / 64 * 64;
    volatile tt_l1_ptr uint32_t* local_row_mask = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        row_mask_dfb.get_write_ptr() + mask_first * sizeof(uint32_t) - mask_read_first);
    if (num_local_chunks != 0) {
        const uint32_t mask_end =
            tt::data_movement::common::div_up(first_local_key + num_local_chunks * 32, row_mask_group);
        const uint32_t mask_read_end = tt::data_movement::common::div_up(mask_end * sizeof(uint32_t), 64) * 64;
        noc.async_read(
            row_mask,
            row_mask_dfb,
            mask_read_end - mask_read_first,
            {.page_id = 0, .offset_bytes = mask_read_first},
            {.offset_bytes = 0});
        noc.async_read_barrier();
        invalidate_l1_cache();
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
#ifdef ROW_MASK
                {
                    volatile tt_l1_ptr uint32_t* masked_scores = scores_l1 + tile * 32;
                    const uint32_t local_first = chunk * 32;
                    for (uint32_t i = 0; i < 32; ++i) {
                        if (((local_row_mask[(local_first + i) / row_mask_group] >> row) & 1) == 0) {
                            masked_scores[i] = 0xFFFFFFFFu;
                        }
                    }
                    // Drain the stores before the NoC reads the tile.
                    masked_scores[31] = masked_scores[31];
                    (void)ckernel::load_blocking(&masked_scores[31]);
                }
#endif
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
#ifdef ROW_MASK
                for (; i < valid_in_tile; ++i) {
                    const uint32_t key = topk_order_key(tile_scores[i]);
                    if (key != 0) {
                        ++row_hist[key >> (32 - kHistBits)];
                    }
                }
#endif
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
    // that select passes this one's ready_base + kAllocReady. With marks, bit j of marks[i] is set for
    // every local key i query row j selects (marks must be zero for the keys on entry).
    // The keys are words of base in the scores' layout (row j's key i at ((i / 32 * Sq) + j) * 32 + i % 32),
    // fp32 scores mapped to order keys when transform is set, else order keys already.
    auto radix_select =
        [&](uint32_t k_select,
            const uint32_t* num_local,
            uint32_t ready_base,
            const uint32_t* base,
            auto transform_tag,
            SelectResult* results,
            uint32_t* marks) {
            constexpr bool transform = decltype(transform_tag)::value;
            auto key_at = [base](uint32_t row, uint32_t i) {
                const uint32_t word = base[((i / 32 * num_queries) + row) * 32 + i % 32];
                if constexpr (transform) {
                    return topk_order_key(word);
                } else {
                    return word;
                }
            };
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
                    hdr[kHdrTies] = 0;
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
                            // min(k_select, keys): with fewer keys the walk below reaches slice 0 first.
                            target = k_select;
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
                            if (above + slice_total >= target) {
                                break;
                            }
                            if (slice == 0) {
                                // Fewer keys than k_select: take them all.
                                target = above + slice_total;
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
                            if (bin_count <= kMaxTies) {
                                // Few keys left in the bin: the final counts carry them instead.
                                hdr[kHdrTies] = 1;
                            } else {
                                next_live |= 1u << row;
                            }
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
                        if (pass == 0) {
                            // Every local key, a 32-key chunk at a time; the bins are cleared whole after.
                            const uint32_t n = num_local[row];
                            const uint32_t row_bit = 1u << row;
                            uint16_t* __restrict cand_out = cand;
                            for (uint32_t chunk = 0; chunk * 32 < n; ++chunk) {
                                const uint32_t* __restrict words = base + (chunk * num_queries + row) * 32;
                                const uint32_t first = chunk * 32;
                                const uint32_t count = std::min<uint32_t>(32, n - first);
                                for (uint32_t c = 0; c < count; ++c) {
                                    uint32_t key = words[c];
                                    if constexpr (transform) {
                                        key = topk_order_key(key);
                                    }
                                    const uint32_t bin = key >> (32 - kHistBits);
                                    if (bin >= threshold_bin) {
                                        if (bin == threshold_bin) {
                                            cand_out[kept++] = first + c;
                                        } else {
                                            ++above;
                                            if (marks != nullptr) {
                                                marks[first + c] |= row_bit;
                                            }
                                        }
                                    }
                                }
                            }
                            for (uint32_t b = 0; b < num_hist_bins; ++b) {
                                row_hist[b] = 0;
                            }
                        } else {
                            for (uint32_t t = 0; t < cand_count[row]; ++t) {
                                const uint32_t i = cand[t];
                                const uint32_t bin = (key_at(row, i) >> shift) & (num_hist_bins - 1);
                                if (bin > threshold_bin) {
                                    ++above;
                                    if (marks != nullptr) {
                                        marks[i] |= 1u << row;
                                    }
                                } else if (bin == threshold_bin) {
                                    cand[kept++] = i;
                                }
                                row_hist[bin] = 0;
                            }
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

            // Every active core reports, per query row, its keys above the row's threshold and equal to it
            // (and on ties those keys) to slot core_index of hist_gather on the row's coordinator, core
            // row. Inactive cores have neither. Every owner has read the histograms by the last header, so
            // they stage the messages.
            if (is_active) {
                DeviceZoneScopedN("SEL-COUNTS");
                for (uint32_t row = 0; row < num_queries; ++row) {
                    uint32_t* counts = hist + row * kCountsWords;
                    counts[0] = core_above[row];
                    counts[1] = core_bin_count[row];
                    if (select[kSelectRowsWord + row * kSelectHeaderWords + kHdrTies] != 0) {
                        const uint16_t* cand = cand_list + row * local_key_capacity;
                        for (uint32_t t = 0; t < cand_count[row]; ++t) {
                            counts[2 + t] = key_at(row, cand[t]);
                        }
                    }
                }
                asm volatile("" ::: "memory");
                hist[num_queries * kCountsWords - 1] = hist[num_queries * kCountsWords - 1];
                (void)ckernel::load_blocking(&hist[num_queries * kCountsWords - 1]);
                for (uint32_t row = 0; row < num_queries; ++row) {
                    noc.async_write(
                        UnicastEndpoint{},
                        UnicastEndpoint{},
                        kCountsWords * sizeof(uint32_t),
                        {.addr = hist_addr + row * kCountsWords * sizeof(uint32_t)},
                        {.noc_x = get_common_vararg(2 * row),
                         .noc_y = get_common_vararg(2 * row + 1),
                         .addr = gather_addr + core_index * kCountsWords * sizeof(uint32_t)});
                }
                noc.async_write_barrier();
                for (uint32_t row = 0; row < num_queries; ++row) {
                    hist_arrived.up(noc, get_common_vararg(2 * row), get_common_vararg(2 * row + 1), 1);
                }
                noc.async_atomic_barrier();
            }

            // Core r allocates query row r. A core contributes all of its keys above the threshold plus a
            // share of the keys equal to it; that share is handed out in core order, so ties go to lower
            // key indices. Output offsets are the running sum of contributions.
            if (core_index < num_queries) {
                const uint32_t row = core_index;
                {
                    DeviceZoneScopedN("SEL-ALLOC-GATHER");
                    hist_arrived.wait(num_active);
                }
                DeviceZoneScopedN("SEL-ALLOC");
                hist_arrived.set(0);
                invalidate_l1_cache();
                asm volatile("" ::: "memory");
                // Plain pointers: every count has landed (the semaphore) and the cache is invalidated.
                uint32_t* const gathered = reinterpret_cast<uint32_t*>(gather_addr);
                uint32_t* const select_words = const_cast<uint32_t*>(select);
                uint32_t* const hdr = select_words + kSelectRowsWord + row * kSelectHeaderWords;
                if (hdr[kHdrTies] != 0) {
                    // The row's last bin holds at most kMaxTies keys over all cores: find the needed-th
                    // largest of them exactly. It becomes the row's full 32-bit threshold, and every core's
                    // keys above it join its keys above the bin.
                    uint32_t all[kMaxTies];
                    uint32_t num_ties = 0;
                    for (uint32_t c = 0; c < num_active; ++c) {
                        const uint32_t* counts = gathered + c * kCountsWords;
                        for (uint32_t t = 0; t < counts[1]; ++t) {
                            all[num_ties++] = counts[2 + t];
                        }
                    }
                    const uint32_t needed = hdr[kHdrNeeded];
                    std::nth_element(all, all + needed - 1, all + num_ties, std::greater<uint32_t>());
                    const uint32_t threshold = all[needed - 1];
                    uint32_t greater = 0;
                    for (uint32_t c = 0; c < num_active; ++c) {
                        uint32_t* counts = gathered + c * kCountsWords;
                        uint32_t core_greater = 0;
                        uint32_t core_equal = 0;
                        for (uint32_t t = 0; t < counts[1]; ++t) {
                            core_greater += counts[2 + t] > threshold ? 1 : 0;
                            core_equal += counts[2 + t] == threshold ? 1 : 0;
                        }
                        counts[0] += core_greater;
                        counts[1] = core_equal;
                        greater += core_greater;
                    }
                    hdr[kHdrPrefix] = threshold;
                    hdr[kHdrPrefixBits] = 32;
                    hdr[kHdrAbove] += greater;
                    hdr[kHdrNeeded] = needed - greater;
                }
                uint32_t remaining_from_bin = hdr[kHdrNeeded];
                uint32_t next_offset = 0;
                uint32_t* const row_entries = select_words + kSelectAllocWord + row * alloc_row_words;
                uint32_t* entry = row_entries;
                const uint32_t* counts = gathered;
                for (uint32_t c = 0; c < num_active; ++c, counts += kCountsWords, entry += kSelectWordsPerCore) {
                    const uint32_t core_from_bin = std::min(counts[1], remaining_from_bin);
                    remaining_from_bin -= core_from_bin;
                    const uint32_t contribute = counts[0] + core_from_bin;
                    entry[0] = contribute;
                    entry[1] = core_from_bin;
                    entry[2] = next_offset;
                    next_offset += contribute;
                }
                asm volatile("" ::: "memory");
                // Only active cores read their entries.
                const uint32_t entry_words = (num_active * kSelectWordsPerCore + 3) / 4 * 4;
                // Stores retire in order: storing the header's last (pad) word and loading it back drains them.
                hdr[kSelectHeaderWords - 1] = 0;
                (void)ckernel::load_blocking(&hdr[kSelectHeaderWords - 1]);
                if (row != 0) {
                    // Hand row r's header and allocation to the coordinator (same addresses), which
                    // multicasts every row's once all have landed.
                    auto to_coordinator = [&](uint32_t word, uint32_t words) {
                        noc.async_write(
                            UnicastEndpoint{},
                            UnicastEndpoint{},
                            words * sizeof(uint32_t),
                            {.addr = select_addr + word * sizeof(uint32_t)},
                            {.noc_x = coord_x, .noc_y = coord_y, .addr = select_addr + word * sizeof(uint32_t)});
                    };
                    to_coordinator(kSelectRowsWord + row * kSelectHeaderWords, kSelectHeaderWords);
                    if (entry_words != 0) {
                        to_coordinator(kSelectAllocWord + row * alloc_row_words, entry_words);
                    }
                    noc.async_write_barrier();
                    alloc_done.up(noc, coord_x, coord_y, 1);
                    noc.async_atomic_barrier();
                }
            }
            if (core_index == 0) {
                if constexpr (num_queries > 1) {
                    DeviceZoneScopedN("SEL-ALLOC-ROWS");
                    alloc_done.wait(num_queries - 1);
                    alloc_done.set(0);
                    invalidate_l1_cache();
                }
                if (num_mcast_dests > 0) {
                    DeviceZoneScopedN("SEL-ALLOC-MCAST");
                    // Every header and every row's run of allocation (only active cores read theirs).
                    const uint32_t alloc_end =
                        kSelectAllocWord + (num_queries - 1) * alloc_row_words + num_active * kSelectWordsPerCore;
                    const uint32_t alloc_words = std::min(num_select_words, (alloc_end + 3) / 4 * 4);
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
                DeviceZoneScopedN("SEL-ALLOC-WAIT");
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
                        select + kSelectAllocWord + row * alloc_row_words + core_index * kSelectWordsPerCore;
                    results[row].num_contribute = my_entry[0];
                    results[row].num_from_bin = my_entry[1];
                    results[row].output_offset = my_entry[2];
                } else {
                    results[row].num_contribute = 0;
                    results[row].num_from_bin = 0;
                    results[row].output_offset = 0;
                }
            }

            // Marks for the keys of each row's final bin (the threshold's keys and, on ties, those above it):
            // all those above the threshold and the first num_from_bin equal to it.
            if (marks != nullptr && is_active) {
                DeviceZoneScopedN("SEL-MARK");
                for (uint32_t row = 0; row < num_queries; ++row) {
                    const uint16_t* cand = cand_list + row * local_key_capacity;
                    const uint32_t shift = results[row].final_shift;
                    const uint32_t prefix = results[row].final_prefix;
                    uint32_t from_bin_left = results[row].num_from_bin;
                    for (uint32_t t = 0; t < cand_count[row]; ++t) {
                        const uint32_t i = cand[t];
                        const uint32_t top = key_at(row, i) >> shift;
                        bool take = top > prefix;
                        if (top == prefix && from_bin_left != 0) {
                            --from_bin_left;
                            take = true;
                        }
                        if (take) {
                            marks[i] |= 1u << row;
                        }
                    }
                }
            }
        };

    // Every score tile has been waited on by now and stays resident, so the rescans read them through
    // a plain pointer.
    const uint32_t* local_scores = const_cast<const uint32_t*>(scores_l1);
    auto key_at = [local_scores](uint32_t row, uint32_t i) {
        return topk_order_key(local_scores[((i / 32 * num_queries) + row) * 32 + i % 32]);
    };
    // Bit j of key_marks[i]: query row j selects local key i.
    DataflowBuffer marks_dfb(dfb::marks);
    marks_dfb.reserve_back(1);
    uint32_t* const key_marks = reinterpret_cast<uint32_t*>(marks_dfb.get_write_ptr());
    {
        uint32_t scan_end = 0;
        for (uint32_t row = 0; row < num_queries; ++row) {
            scan_end = std::max(scan_end, num_local_keys[row]);
        }
        for (uint32_t i = 0; i < scan_end; ++i) {
            key_marks[i] = 0;
        }
    }
    SelectResult key_shares[num_queries];
    radix_select(k, num_local_keys, /*ready_base=*/0, local_scores, std::true_type{}, key_shares, key_marks);

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
    // Query row j's candidates page sits cand_page_words words after row j - 1's in kv_wr, and is page j
    // of the output. A later select overwrites the headers.
    uint32_t num_keys_selected[num_queries];
    for (uint32_t row = 0; row < num_queries; ++row) {
        volatile tt_l1_ptr uint32_t* hdr = select + kSelectRowsWord + row * kSelectHeaderWords;
        num_keys_selected[row] = hdr[kHdrAbove] + hdr[kHdrNeeded];
    }
    // List each row's selected keys in local key order as kv_cache row ids, taking the first num_from_bin
    // keys equal to the threshold. Each core lists its runs straight into the candidates page's layout
    // (see kCandHeaderWord), at the same kv_wr address it then writes to on core 0.
    kv_wr_dfb.reserve_back(kv_stage_rows);
    const uint32_t cand_addr = kv_wr_dfb.get_write_ptr();
    volatile tt_l1_ptr uint32_t* cand = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cand_addr);
    uint32_t num_rows[num_queries];
    {
        DeviceZoneScopedN("SEL-LIST");
        for (uint32_t row = 0; row < num_queries; ++row) {
            const SelectResult& key_share = key_shares[row];
            volatile tt_l1_ptr uint32_t* rows = cand + row * cand_page_words + key_share.output_offset;
            volatile tt_l1_ptr uint32_t* keys = rows + k;
            uint32_t listed = 0;
            for (uint32_t i = 0; i < num_local_keys[row] && listed < key_share.num_contribute; ++i) {
                if ((key_marks[i] >> row) & 1) {
                    const uint32_t key = key_at(row, i);
#ifdef ROW_TABLE
                    rows[listed] = row_tbl[i];
#else
                    rows[listed] = physical_blocks[i / page_block_size] * page_block_size + i % page_block_size;
#endif
                    keys[listed] = key;
                    ++listed;
                }
            }
            num_rows[row] = listed;
        }
    }

    // Every active core copies its runs into core 0's pages, at the offsets it listed them at locally;
    // core 0 then zeroes the unused entries, fills the headers and writes the pages. A run is `count`
    // ids from word `first`, with their order keys `keys_stride` words later. Per query row: its first
    // listed block, the blocks this core listed and the blocks selected over the part.
    auto write_partial =
        [&](const uint32_t* first_block, const uint32_t* num_blocks_listed, const uint32_t* num_blocks_selected) {
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
                for (uint32_t row = 0; row < num_queries; ++row) {
                    const uint32_t page = row * cand_page_words;
                    send_run(page + key_shares[row].output_offset, num_rows[row], k);
                    send_run(page + kCandBlocksWord + first_block[row], num_blocks_listed[row], candidate_topk_blocks);
                }
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
                for (uint32_t row = 0; row < num_queries; ++row) {
                    volatile tt_l1_ptr uint32_t* page = cand + row * cand_page_words;
                    for (uint32_t j = num_keys_selected[row]; j < k; ++j) {
                        page[j] = 0;
                        page[k + j] = 0;
                    }
                    for (uint32_t j = num_blocks_selected[row]; j < candidate_topk_blocks; ++j) {
                        page[kCandBlocksWord + j] = 0;
                        page[kCandBlocksWord + candidate_topk_blocks + j] = 0;
                    }
                    for (uint32_t j = 0; j < 16; ++j) {
                        page[kCandHeaderWord + j] = 0;
                    }
                    page[kCandHeaderWord] = num_keys_selected[row];
                    page[kCandHeaderWord + 1] = num_blocks_selected[row];
                    page[kCandHeaderWord + 2] = row_keys[row];
                    (void)ckernel::load_blocking(&page[kCandHeaderWord + 15]);
                    noc.async_write(
                        kv_wr_dfb,
                        output,
                        (kCandBlocksWord + 2 * candidate_topk_blocks) * sizeof(uint32_t),
                        {.offset_bytes = row * cand_page_words * static_cast<uint32_t>(sizeof(uint32_t))},
                        {.page_id = row, .offset_bytes = 0});
                }
                noc.async_write_barrier();
            }
        };
#ifndef CANDIDATE_SOURCE
    {
        const uint32_t zeros[num_queries] = {};
        write_partial(zeros, zeros, zeros);
    }
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
        uint32_t scan_end = 0;
        uint32_t to_list = 0;
        for (uint32_t row = 0; row < num_queries; ++row) {
            to_list += key_shares[row].num_contribute;
            if (key_shares[row].num_contribute != 0) {
                scan_end = std::max(scan_end, num_local_keys[row]);
            }
        }
        for (uint32_t i = 0; i < scan_end && to_list != 0; ++i) {
            const uint32_t mask = key_marks[i];
            if (mask != 0) {
                to_list -= __builtin_popcount(mask);
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
    // ---- Candidate blocks ----
    // Each query row's keys split into blocks of candidate_block_size, each scored by the row's best key
    // in it; the block holding the row's newest key is only partly filled and always kept. Every row's
    // candidate_topk_blocks best blocks go to its candidate_blocks row in ascending order (zero past the
    // last), and its pool_count entry is the number of valid keys in them. Page blocks are whole
    // multiples of a candidate block, so no candidate block spans two cores (or two parts). With
    // PARTIAL_OUTPUT each row's best blocks of this part go into its candidates page with their order
    // keys instead, for merge_select_kv to pick the global best.
    {
        DeviceZoneScopedN("SEL-CAND");
        const uint32_t first_local_block = first_local_key / candidate_block_size;
        uint32_t num_blocks[num_queries];
        uint32_t pinned_block[num_queries];
        uint32_t num_local_blocks[num_queries];
        uint32_t max_local_blocks = 0;
        for (uint32_t row = 0; row < num_queries; ++row) {
            num_blocks[row] = tt::data_movement::common::div_up(row_keys[row], candidate_block_size);
            pinned_block[row] = num_blocks[row] - 1;
            num_local_blocks[row] = tt::data_movement::common::div_up(num_local_keys[row], candidate_block_size);
            max_local_blocks = std::max(max_local_blocks, num_local_blocks[row]);
        }

        // Every owner has read this core's histograms of the key select, so they can be refilled. The
        // scores are no longer needed either: row j's block b's order key overwrites row j's score b,
        // which only row j's blocks <= b have read (key_at's layout with the block for the key).
        for (uint32_t w = 0; w < kHistWords; ++w) {
            hist[w] = 0;
        }
        auto block_slot = [](uint32_t row, uint32_t b) { return ((b / 32 * num_queries) + row) * 32 + b % 32; };
        for (uint32_t b = 0; b < max_local_blocks; ++b) {
            for (uint32_t row = 0; row < num_queries; ++row) {
                if (b >= num_local_blocks[row]) {
                    continue;
                }
                uint32_t best = 0xFFFFFFFFu;
                if (first_local_block + b != pinned_block[row]) {
                    const uint32_t end = std::min((b + 1) * candidate_block_size, num_local_keys[row]);
                    best = 0;
                    for (uint32_t i = b * candidate_block_size; i < end; ++i) {
                        best = std::max(best, key_at(row, i));
                    }
                }
                scores_l1[block_slot(row, b)] = best;
                ++hist[row * num_hist_bins + (best >> (32 - kHistBits))];
            }
        }
        const uint32_t* block_keys = local_scores;
        auto block_key = [block_keys, block_slot](uint32_t row, uint32_t b) { return block_keys[block_slot(row, b)]; };
        SelectResult block_shares[num_queries];
        radix_select(
            candidate_topk_blocks,
            num_local_blocks,
            /*ready_base=*/kAllocReady,
            block_keys,
            std::false_type{},
            block_shares,
            nullptr);
        uint32_t num_blocks_selected[num_queries];
        for (uint32_t row = 0; row < num_queries; ++row) {
            volatile tt_l1_ptr uint32_t* hdr = select + kSelectRowsWord + row * kSelectHeaderWords;
            num_blocks_selected[row] = hdr[kHdrAbove] + hdr[kHdrNeeded];
        }

#ifdef PARTIAL_OUTPUT
        // The blocks go after this core's key run in the same staging page.
        constexpr uint32_t kListRowWords = cand_page_words;
        volatile tt_l1_ptr uint32_t* list = cand + kCandBlocksWord;
#else
        // kv_wr is free once this core's gather is done, and core 0's is done before the allocation,
        // so every core lists its blocks at the same L1 offsets it then writes to on core 0. Row j's
        // list starts kListRowWords words after row j - 1's.
        constexpr uint32_t kListRowWords = (candidate_topk_blocks + 15) / 16 * 16;
        const uint32_t list_addr = kv_wr_dfb.get_write_ptr();
        volatile tt_l1_ptr uint32_t* list = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(list_addr);
#endif
        uint32_t num_listed[num_queries];
        uint32_t first_listed[num_queries];
        for (uint32_t row = 0; row < num_queries; ++row) {
            const SelectResult& block_share = block_shares[row];
            volatile tt_l1_ptr uint32_t* row_list = list + row * kListRowWords;
            uint32_t listed = 0;
            uint32_t from_bin_left = block_share.num_from_bin;
            for (uint32_t b = 0; b < num_local_blocks[row] && listed < block_share.num_contribute; ++b) {
                const uint32_t key = block_key(row, b);
                const uint32_t top = key >> block_share.final_shift;
                bool take = top > block_share.final_prefix;
                if (top == block_share.final_prefix && from_bin_left != 0) {
                    --from_bin_left;
                    take = true;
                }
                if (take) {
#ifdef PARTIAL_OUTPUT
                    row_list[candidate_topk_blocks + block_share.output_offset + listed] = key;
#endif
                    row_list[block_share.output_offset + listed++] = first_local_block + b;
                }
            }
            num_listed[row] = listed;
            first_listed[row] = block_share.output_offset;
        }

#ifdef PARTIAL_OUTPUT
        write_partial(first_listed, num_listed, num_blocks_selected);
#else
        // pool_count's Sq words follow the last row's list, 64 B aligned.
        constexpr uint32_t kCountOffsetWords = num_queries * kListRowWords;
        const auto candidate_blocks = TensorAccessor(tensor::candidate_blocks);
        const auto pool_count = TensorAccessor(tensor::pool_count);
        if (core_index != 0 && is_active) {
            for (uint32_t row = 0; row < num_queries; ++row) {
                if (num_listed[row] != 0) {
                    const uint32_t first = row * kListRowWords + first_listed[row];
                    const uint32_t addr = list_addr + first * sizeof(uint32_t);
                    (void)ckernel::load_blocking(&list[first + num_listed[row] - 1]);
                    noc.async_write(
                        UnicastEndpoint{},
                        UnicastEndpoint{},
                        num_listed[row] * sizeof(uint32_t),
                        {.addr = addr},
                        {.noc_x = coord_x, .noc_y = coord_y, .addr = addr});
                }
            }
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
            for (uint32_t row = 0; row < num_queries; ++row) {
                volatile tt_l1_ptr uint32_t* row_list = list + row * kListRowWords;
                const uint32_t num_chosen = std::min(candidate_topk_blocks, num_blocks[row]);
                for (uint32_t j = num_chosen; j < candidate_topk_blocks; ++j) {
                    row_list[j] = 0;
                }
                // Every chosen block but the pinned one, which is last, is full.
                const uint32_t pinned_keys = row_keys[row] - pinned_block[row] * candidate_block_size;
                list[kCountOffsetWords + row] =
                    num_chosen == 0 ? 0 : (num_chosen - 1) * candidate_block_size + pinned_keys;
            }
            (void)ckernel::load_blocking(&list[kCountOffsetWords + num_queries - 1]);
            for (uint32_t row = 0; row < num_queries; ++row) {
                noc.async_write(
                    kv_wr_dfb,
                    candidate_blocks,
                    candidate_topk_blocks * sizeof(uint32_t),
                    {.offset_bytes = row * kListRowWords * static_cast<uint32_t>(sizeof(uint32_t))},
                    {.page_id = row, .offset_bytes = 0});
            }
            noc.async_write(
                kv_wr_dfb,
                pool_count,
                num_queries * sizeof(uint32_t),
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
    alloc_done.set(0);
}
