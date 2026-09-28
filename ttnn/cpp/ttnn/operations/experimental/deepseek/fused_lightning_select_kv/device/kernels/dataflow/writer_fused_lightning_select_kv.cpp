// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writes this core's index scores to the scores output, selects the global top-k with the other
// cores, then gathers the second half of this core's selected kv_cache rows into the output.

#define COMPRESS_RATE 4
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

void kernel_main() {
    // ---- Compile-time args ----
    [[maybe_unused]] constexpr uint32_t k = get_arg(args::k);
    constexpr uint32_t page_block_size = get_arg(args::page_block_size);
    constexpr uint32_t chunks_per_block = page_block_size / 32;
    constexpr uint32_t num_hist_bins = get_arg(args::num_hist_bins);
    static_assert((num_hist_bins & (num_hist_bins - 1)) == 0, "num_hist_bins must be a power of two");
    constexpr uint32_t kHistBits = __builtin_ctz(num_hist_bins);
    constexpr uint32_t bins_per_slice = get_arg(args::hist_bins_per_slice);
    constexpr uint32_t num_owners = get_arg(args::hist_num_owners);
    constexpr uint32_t slices_per_owner = get_arg(args::hist_slices_per_owner);
    constexpr uint32_t owner_bins = slices_per_owner * bins_per_slice;  // owner o holds bins [o * owner_bins, ...)
    constexpr uint32_t owner_bytes = owner_bins * sizeof(uint32_t);
    constexpr uint32_t kCountsWords = 4;  // {keys above threshold bin, keys in it, pad, pad}: 16 B for NoC alignment
    constexpr uint32_t num_select_words = get_arg(args::num_select_words);
    // select header: {prefix, prefix bits, keys above it, keys still needed from it, last pass, total keys, pad, pad}
    constexpr uint32_t kSelectHeaderWords = 8;
    constexpr uint32_t kHdrPrefix = 0;
    constexpr uint32_t kHdrPrefixBits = 1;
    constexpr uint32_t kHdrAbove = 2;
    constexpr uint32_t kHdrNeeded = 3;
    constexpr uint32_t kHdrDone = 4;
    constexpr uint32_t kHdrTotal = 5;
    static_assert(32 % kHistBits == 0, "radix passes must tile the 32-bit order key");
    // select_ready value of the final allocation; pass p's header uses p + 1 <= 32 / kHistBits.
    constexpr uint32_t kAllocReady = 32 / kHistBits + 1;
    constexpr uint32_t kSelectWordsPerCore = 3;
    constexpr uint32_t kv_stage_rows = get_arg(args::kv_stage_rows);
    constexpr uint32_t kSelRowsOffset = 4;  // sel: {num_rows, output_offset, pad, pad, rows...}
    constexpr uint32_t output_row_offset = get_arg(args::output_row_offset);  // output row of selected row 0

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
    const auto output = TensorAccessor(tensor::output);  // [1, 1, rows, Dh] row-major, interleaved, one row per page
    const auto kv_cache = TensorAccessor(tensor::kv_cache);  // [num_blocks, 1, block_size, Dh] row-major
    const auto scores = TensorAccessor(tensor::scores);  // [1, 1, 1, T] fp32, one row-major page

    // ---- Dataflow buffers ----
    DataflowBuffer kv_wr_dfb(dfb::kv_wr);      // writer-local staging for gathered kv rows
    DataflowBuffer blocks_dfb(dfb::blocks);    // consumer <- reader (physical block of each local block)
    DataflowBuffer sel_dfb(dfb::sel);          // producer -> reader (selected kv_cache rows)
    DataflowBuffer cur_pos_dfb(dfb::cur_pos);  // consumer <- reader
    DataflowBuffer scores_dfb(dfb::scores);    // consumer <- compute
    DataflowBuffer hist_dfb(dfb::hist);        // writer-local scratch
    DataflowBuffer hist_gather_dfb(dfb::hist_gather);    // per-core counts around the threshold, used on core 0
    DataflowBuffer select_dfb(dfb::select);              // top-k threshold, multicast from core 0
    DataflowBuffer hist_scatter_dfb(dfb::hist_scatter);  // every core's copy of the bins this core owns
    DataflowBuffer hist_sum_dfb(dfb::hist_sum);          // global histogram, assembled on core 0

    Semaphore hist_arrived(sem::hist_arrived);    // on core 0: summed bins, then per-core counts received
    Semaphore select_ready(sem::select_ready);    // on every core: 1 = threshold header, 2 = allocation arrived
    Semaphore slice_arrived(sem::slice_arrived);  // on each owner: slices received

    Noc noc;

#ifdef HAS_NEW_KV_ROW
    // Write this step's closing entry into kv_cache before anything is selected. Every gather, on
    // any core, starts only after core 0's final allocation multicast, which follows this barrier.
    // hist and kv_wr are free until the top-k and the gather, so they stage the lookups and the row.
    if (core_index == 0) {
        DeviceZoneScopedN("SEL-NEW-ROW");
        const auto page_table = TensorAccessor(tensor::page_table);
        const auto new_kv_row = TensorAccessor(tensor::new_kv_row);
        const auto new_kv_row_index = TensorAccessor(tensor::new_kv_row_index);
        hist_dfb.reserve_back(1);
        volatile tt_l1_ptr uint32_t* scratch = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hist_dfb.get_write_ptr());
        noc.async_read(new_kv_row_index, hist_dfb, 64, {.page_id = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        invalidate_l1_cache();
        const uint32_t logical_row = scratch[0];
        const uint32_t block = logical_row / page_block_size;
        // Each 64 B DRAM-aligned chunk of the page table holds 16 uint32_t entries.
        noc.async_read(
            page_table, hist_dfb, 64, {.page_id = 0, .offset_bytes = 64 * (block / 16)}, {.offset_bytes = 64});
        noc.async_read_barrier();
        invalidate_l1_cache();
        const uint32_t physical_row = scratch[16 + block % 16] * page_block_size + logical_row % page_block_size;

        kv_wr_dfb.reserve_back(kv_stage_rows);
        const uint32_t row_bytes = kv_wr_dfb.get_entry_size();
        noc.async_read(new_kv_row, kv_wr_dfb, row_bytes, {.page_id = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        noc.async_write(kv_wr_dfb, kv_cache, row_bytes, {.offset_bytes = 0}, {.page_id = physical_row});
        noc.async_write_barrier();
    }
#endif

    // Same block split as the reader.
    cur_pos_dfb.wait_front(1);
    const uint32_t cur_pos_value = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cur_pos_dfb.get_read_ptr())[0];
    cur_pos_dfb.pop_front(1);
    const uint32_t num_keys = (cur_pos_value + 1) / COMPRESS_RATE;
    const uint32_t total_num_blocks = tt::data_movement::common::div_up(num_keys, page_block_size);
    const uint32_t work_per_core = tt::data_movement::common::div_up(total_num_blocks, num_cores);
    const uint32_t start_block = std::min(work_per_core * core_index, total_num_blocks);
    const uint32_t end_block = std::min(start_block + work_per_core, total_num_blocks);

    // scores_dfb is sized to hold every tile this core produces and is never popped until the end, so
    // the tiles stay resident and contiguous from the read pointer: local key i is scores_l1[i].
    // A 1x32 tile is 32 consecutive values, i.e. already the row-major layout of 32 scores.
    const uint32_t score_tile_bytes = scores_dfb.get_entry_size();
    const uint32_t num_local_tiles = (end_block - start_block) * chunks_per_block;
    const uint32_t first_local_key = start_block * page_block_size;
    volatile tt_l1_ptr uint32_t* scores_l1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scores_dfb.get_read_ptr());

    hist_dfb.reserve_back(1);
    volatile tt_l1_ptr uint32_t* hist = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hist_dfb.get_write_ptr());
    for (uint32_t bin = 0; bin < num_hist_bins; ++bin) {
        hist[bin] = 0;
    }

    {
        DeviceZoneScopedN("SEL-HIST");
        for (uint32_t tile = 0; tile < num_local_tiles; ++tile) {
            const uint32_t first_key = first_local_key + tile * 32;
            scores_dfb.wait_front(tile + 1);
            noc.async_write(
                scores_dfb,
                scores,
                score_tile_bytes,
                {.offset_bytes = tile * score_tile_bytes},
                {.page_id = 0, .offset_bytes = first_key * sizeof(float)});

            // Keys at or past num_keys (tail of the last block) are left out, which is the same as -inf.
            const uint32_t valid_in_tile = first_key >= num_keys ? 0 : std::min<uint32_t>(num_keys - first_key, 32);
            volatile tt_l1_ptr uint32_t* tile_scores = scores_l1 + tile * 32;
            uint32_t i = 0;
            for (; i + 4 <= valid_in_tile; i += 4) {
                const uint32_t b0 = topk_order_key(tile_scores[i + 0]) >> (32 - kHistBits);
                const uint32_t b1 = topk_order_key(tile_scores[i + 1]) >> (32 - kHistBits);
                const uint32_t b2 = topk_order_key(tile_scores[i + 2]) >> (32 - kHistBits);
                const uint32_t b3 = topk_order_key(tile_scores[i + 3]) >> (32 - kHistBits);
                ++hist[b0];
                ++hist[b1];
                ++hist[b2];
                ++hist[b3];
            }
            for (; i < valid_in_tile; ++i) {
                ++hist[topk_order_key(tile_scores[i]) >> (32 - kHistBits)];
            }
        }
        noc.async_write_barrier();
    }

    // ---- Cross-core radix select ----
    // Pass 0 histograms the top kHistBits of every valid key's order key. Each later pass narrows to
    // the keys whose already-fixed prefix equals the threshold bin found so far, and histograms their
    // next kHistBits. It stops once the threshold bin can be taken whole or all 32 bits are fixed, so
    // only keys with equal scores are left to the index-order tie-break.
    //
    // Blocks go to cores in order, so exactly cores [0, num_active) have keys; the others' histograms
    // are all zero and take no part. Per pass, each active core bumps hist_arrived on the coordinator
    // once its histogram is final; when all have, the coordinator releases every core. Owner o holds
    // bins [o * owner_bins, (o + 1) * owner_bins): it reads that range from every active core's
    // histogram into entry c of hist_scatter, sums them, writes the result into hist_sum on the
    // coordinator and bumps hist_arrived again, so the coordinator then waits for num_owners arrivals
    // before multicasting the pass header with select_ready = pass + 1.
    const uint32_t num_active =
        work_per_core == 0 ? 0 : tt::data_movement::common::div_up(total_num_blocks, work_per_core);
    const bool is_active = core_index < num_active;
    const bool is_owner = core_index < num_owners;
    const uint32_t num_local_keys =
        num_local_tiles == 0 ? 0 : std::min<uint32_t>(num_keys - first_local_key, num_local_tiles * 32);
    const uint32_t hist_addr = hist_dfb.get_write_ptr();
    const uint32_t gather_addr = hist_gather_dfb.get_write_ptr();
    const uint32_t scatter_addr = hist_scatter_dfb.get_write_ptr();
    const uint32_t sum_addr = hist_sum_dfb.get_write_ptr();
    const uint32_t select_addr = select_dfb.get_write_ptr();
    volatile tt_l1_ptr uint32_t* select = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(select_addr);

    // Coordinator-side state, published in the select header after every pass.
    uint32_t coord_prefix = 0;
    uint32_t coord_prefix_bits = 0;
    uint32_t coord_above = 0;
    uint32_t coord_needed = 0;
    uint32_t coord_total = 0;

    // Every score tile has been waited on by now and stays resident, so the rescans below read them
    // through a plain pointer.
    const uint32_t* local_scores = const_cast<const uint32_t*>(scores_l1);
    // This core's keys above the final threshold and equal to it, accumulated from its own
    // histogram after each pass so no extra scan is needed.
    uint32_t core_above = 0;
    uint32_t core_bin_count = 0;

    // Inactive non-owners have no part in the passes and only wait for the final allocation.
    const bool in_passes = is_active || is_owner || core_index == 0;
    bool done = !in_passes;
    for (uint32_t pass = 0; !done; ++pass) {
        if (pass > 0 && is_active) {
            DeviceZoneScopedN("SEL-REFINE-HIST");
            const uint32_t prefix = select[kHdrPrefix];
            const uint32_t prefix_bits = select[kHdrPrefixBits];
            for (uint32_t bin = 0; bin < num_hist_bins; ++bin) {
                hist[bin] = 0;
            }
            const uint32_t match_shift = 32 - prefix_bits;
            const uint32_t bin_shift = match_shift - kHistBits;
#pragma GCC unroll 4
            for (uint32_t i = 0; i < num_local_keys; ++i) {
                const uint32_t key = topk_order_key(local_scores[i]);
                if ((key >> match_shift) == prefix) {
                    ++hist[(key >> bin_shift) & (num_hist_bins - 1)];
                }
            }
        }

        // RISC-V stores can still be in flight when the NoC reads the buffer; a store followed by a
        // blocking load of the same word drains them all, since stores retire in order.
        if (is_active) {
            DeviceZoneScopedN("SEL-SIGNAL");
            hist[num_hist_bins - 1] = hist[num_hist_bins - 1];
            (void)ckernel::load_blocking(&hist[num_hist_bins - 1]);
            hist_arrived.up(noc, coord_x, coord_y, 1);
            noc.async_atomic_barrier();
        }

        // Concurrent multicast atomics from many cores serialize badly, so the coordinator collects the
        // active cores' signals and releases everyone with a single multicast of slice_arrived = 1.
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
            const uint32_t first_bin = core_index * owner_bins;
            const uint32_t my_bins = std::min(owner_bins, num_hist_bins - first_bin);
            for (uint32_t c = 0; c < num_active; ++c) {
                noc.async_read(
                    UnicastEndpoint{},
                    UnicastEndpoint{},
                    my_bins * sizeof(uint32_t),
                    {.noc_x = get_common_vararg(2 * c),
                     .noc_y = get_common_vararg(2 * c + 1),
                     .addr = hist_addr + first_bin * sizeof(uint32_t)},
                    {.addr = scatter_addr + c * owner_bytes});
            }
            noc.async_read_barrier();
            invalidate_l1_cache();
            volatile tt_l1_ptr uint32_t* scattered = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scatter_addr);
            // One independent accumulator per bin keeps consecutive loads free of dependencies. The last
            // owner may hold fewer than owner_bins bins; the extra words it sums lie inside each entry and
            // are discarded.
            uint32_t sums[owner_bins] = {};
            volatile tt_l1_ptr uint32_t* entry = scattered;
            for (uint32_t c = 0; c < num_active; ++c, entry += owner_bins) {
#pragma GCC unroll 8
                for (uint32_t b = 0; b < owner_bins; ++b) {
                    sums[b] += entry[b];
                }
            }
            // Entry 0 has been consumed, so it stages the sums.
            for (uint32_t b = 0; b < my_bins; ++b) {
                scattered[b] = sums[b];
            }
            (void)ckernel::load_blocking(&scattered[my_bins - 1]);
            noc.async_write(
                UnicastEndpoint{},
                UnicastEndpoint{},
                my_bins * sizeof(uint32_t),
                {.addr = scatter_addr},
                {.noc_x = coord_x, .noc_y = coord_y, .addr = sum_addr + first_bin * sizeof(uint32_t)});
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
            volatile tt_l1_ptr uint32_t* global_hist = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sum_addr);
            if (pass == 0) {
                for (uint32_t bin = 0; bin < num_hist_bins; ++bin) {
                    coord_total += global_hist[bin];
                }
                coord_needed = std::min<uint32_t>(k, coord_total);
            }
            // Walk bins from the largest scores down until the running count reaches the keys still
            // needed. The bin where that happens holds the k-th largest score.
            const uint32_t target = coord_needed;
            uint32_t above = 0;
            uint32_t threshold_bin = num_hist_bins;
            while (threshold_bin > 0) {
                --threshold_bin;
                if (above + global_hist[threshold_bin] >= target) {
                    break;
                }
                above += global_hist[threshold_bin];
            }
            const uint32_t bin_count = global_hist[threshold_bin];
            coord_needed = target - above;
            coord_above += above;
            coord_prefix = (coord_prefix << kHistBits) | threshold_bin;
            coord_prefix_bits += kHistBits;
            const bool last = coord_needed == 0 || coord_needed == bin_count || coord_prefix_bits == 32;

            DPRINT(
                "top{} pass {} threshold_bin {} above {} needed_from_bin {} bin_count {} total {}\n",
                k,
                pass,
                threshold_bin,
                coord_above,
                coord_needed,
                bin_count,
                coord_total);

            select[kHdrPrefix] = coord_prefix;
            select[kHdrPrefixBits] = coord_prefix_bits;
            select[kHdrAbove] = coord_above;
            select[kHdrNeeded] = coord_needed;
            select[kHdrDone] = last ? 1 : 0;
            select[kHdrTotal] = coord_total;
            (void)ckernel::load_blocking(&select[kSelectHeaderWords - 1]);

            if (num_mcast_dests > 0) {
                noc.async_write_multicast(
                    UnicastEndpoint{},
                    MulticastEndpoint{},
                    kSelectHeaderWords * sizeof(uint32_t),
                    num_mcast_dests,
                    {.addr = select_addr},
                    {.noc_x_start = mcast_x_start,
                     .noc_y_start = mcast_y_start,
                     .noc_x_end = mcast_x_end,
                     .noc_y_end = mcast_y_end,
                     .addr = select_addr});
                noc.async_write_barrier();
                select_ready.set(pass + 1);
                select_ready.set_multicast(
                    noc, mcast_x_start, mcast_y_start, mcast_x_end, mcast_y_end, num_mcast_dests);
                noc.async_write_barrier();
            }
        } else {
            // The allocation keeps the header words, so a core that sees it instead of this pass's
            // header reads the same final header.
            select_ready.wait_min(pass + 1);
            invalidate_l1_cache();
        }
        done = select[kHdrDone] != 0;
        if (is_active) {
            const uint32_t threshold_bin = select[kHdrPrefix] & (num_hist_bins - 1);
            for (uint32_t bin = threshold_bin + 1; bin < num_hist_bins; ++bin) {
                core_above += hist[bin];
            }
            core_bin_count = hist[threshold_bin];
        }
    }

    // Final threshold: keys whose top prefix_bits exceed prefix are in the top-k; num_needed of the
    // keys equal to it are also taken.
    const uint32_t final_prefix = select[kHdrPrefix];
    const uint32_t final_shift = 32 - select[kHdrPrefixBits];

    // Every active core reports its keys above the threshold and equal to it to slot core_index of
    // hist_gather on the coordinator. Inactive cores have neither. The local histogram is not needed
    // afterwards, so it stages the message.
    if (is_active) {
        hist[0] = core_above;
        hist[1] = core_bin_count;
        (void)ckernel::load_blocking(&hist[1]);
        noc.async_write(
            UnicastEndpoint{},
            UnicastEndpoint{},
            kCountsWords * sizeof(uint32_t),
            {.addr = hist_addr},
            {.noc_x = coord_x, .noc_y = coord_y, .addr = gather_addr + core_index * kCountsWords * sizeof(uint32_t)});
        noc.async_write_barrier();
        hist_arrived.up(noc, coord_x, coord_y, 1);
        noc.async_atomic_barrier();
    }

    if (core_index == 0) {
        hist_arrived.wait(num_active);
        hist_arrived.set(0);
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* gathered = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gather_addr);

        // Per-core allocation. A core contributes all of its keys above the threshold plus a share of
        // the keys equal to it; that share is handed out in core order, so ties go to lower key
        // indices. Output offsets are the running sum of contributions.
        uint32_t remaining_from_bin = select[kHdrNeeded];
        uint32_t next_offset = 0;
        for (uint32_t c = 0; c < num_cores; ++c) {
            uint32_t core_above = 0;
            uint32_t core_from_bin = 0;
            if (c < num_active) {
                volatile tt_l1_ptr uint32_t* counts = gathered + c * kCountsWords;
                core_above = counts[0];
                const uint32_t core_bin_count = counts[1];
                core_from_bin = std::min<uint32_t>(core_bin_count, remaining_from_bin);
                remaining_from_bin -= core_from_bin;
            }

            volatile tt_l1_ptr uint32_t* entry = select + kSelectHeaderWords + c * kSelectWordsPerCore;
            entry[0] = core_above + core_from_bin;
            entry[1] = core_from_bin;
            entry[2] = next_offset;
            next_offset += core_above + core_from_bin;
        }
        DPRINT("top{} allocated {} of {}\n", k, next_offset, select[kHdrAbove] + select[kHdrNeeded]);
        (void)ckernel::load_blocking(&select[kSelectHeaderWords + (num_cores - 1) * kSelectWordsPerCore + 2]);

        if (num_mcast_dests > 0) {
            noc.async_write_multicast(
                UnicastEndpoint{},
                MulticastEndpoint{},
                num_select_words * sizeof(uint32_t),
                num_mcast_dests,
                {.addr = select_addr},
                {.noc_x_start = mcast_x_start,
                 .noc_y_start = mcast_y_start,
                 .noc_x_end = mcast_x_end,
                 .noc_y_end = mcast_y_end,
                 .addr = select_addr});
            noc.async_write_barrier();
            select_ready.set(kAllocReady);
            select_ready.set_multicast(noc, mcast_x_start, mcast_y_start, mcast_x_end, mcast_y_end, num_mcast_dests);
            noc.async_write_barrier();
        }
    } else {
        select_ready.wait_min(kAllocReady);
        invalidate_l1_cache();
    }
    select_ready.set(0);

    // This core's assignment: emit num_contribute keys (all local keys above the threshold plus
    // num_from_bin keys equal to it) to output rows [output_offset, output_offset + num_contribute).
    volatile tt_l1_ptr uint32_t* my_entry = select + kSelectHeaderWords + core_index * kSelectWordsPerCore;
    const uint32_t num_contribute = my_entry[0];
    const uint32_t num_from_bin = my_entry[1];
    const uint32_t output_offset = output_row_offset + my_entry[2];
    if (num_contribute != 0) {
        DPRINT(
            "core {} contributes {} ({} from threshold bin) at output offset {}\n",
            core_index,
            num_contribute,
            num_from_bin,
            output_offset);
    }

    // ---- KV gather ----
    // List the selected keys in local key order as kv_cache row ids, taking the first num_from_bin
    // keys equal to the threshold. The reader gathers the first half of the list on its NoC, this
    // kernel the rest on the other one.
    blocks_dfb.wait_front(1);
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* physical_blocks =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(blocks_dfb.get_read_ptr());
    sel_dfb.reserve_back(1);
    volatile tt_l1_ptr uint32_t* sel = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sel_dfb.get_write_ptr());
    volatile tt_l1_ptr uint32_t* rows = sel + kSelRowsOffset;
    uint32_t num_rows = 0;
    {
        DeviceZoneScopedN("SEL-LIST");
        uint32_t from_bin_left = num_from_bin;
        for (uint32_t i = 0; i < num_local_keys && num_rows < num_contribute; ++i) {
            const uint32_t top = topk_order_key(local_scores[i]) >> final_shift;
            bool take = top > final_prefix;
            if (top == final_prefix && from_bin_left != 0) {
                --from_bin_left;
                take = true;
            }
            if (take) {
                rows[num_rows++] = physical_blocks[i / page_block_size] * page_block_size + i % page_block_size;
            }
        }
    }
    sel[0] = num_rows;
    sel[1] = output_offset;
    (void)ckernel::load_blocking(num_rows == 0 ? &sel[1] : &rows[num_rows - 1]);
    sel_dfb.push_back(1);

    {
        DeviceZoneScopedN("SEL-GATHER-WR");
        gather_kv_rows<kv_stage_rows>(noc, kv_cache, output, kv_wr_dfb, rows, num_rows / 2, num_rows, output_offset);
    }
    blocks_dfb.pop_front(1);
    scores_dfb.pop_front(num_local_tiles);
}
