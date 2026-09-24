// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writes this core's index scores to the scores output, then drains one gathered kv entry
// (boilerplate) to the kv output.

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
    constexpr uint32_t kSelectHeaderWords = 4;
    constexpr uint32_t kSelectWordsPerCore = 3;

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
    const auto output = TensorAccessor(tensor::output);
    const auto scores = TensorAccessor(tensor::scores);  // [1, 1, 1, T] fp32, one row-major page

    // ---- Dataflow buffers ----
    DataflowBuffer kv_dfb(dfb::kv);            // consumer <- reader
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

    // ---- Cross-core reduction ----
    // Blocks go to cores in order, so exactly cores [0, num_active) have keys; the others' histograms
    // are all zero and take no part. Each active core bumps hist_arrived on the coordinator once its
    // histogram is final; when all have, the coordinator releases every core. Owner o holds bins
    // [o * owner_bins, (o + 1) * owner_bins): it reads that range from every active core's histogram
    // into entry c of hist_scatter, sums them, writes the result into hist_sum on the coordinator and
    // bumps hist_arrived again, so the coordinator then waits for num_owners arrivals.
    const uint32_t num_active =
        work_per_core == 0 ? 0 : tt::data_movement::common::div_up(total_num_blocks, work_per_core);
    const bool is_active = core_index < num_active;
    const uint32_t hist_addr = hist_dfb.get_write_ptr();
    [[maybe_unused]] const uint32_t gather_addr = hist_gather_dfb.get_write_ptr();
    const uint32_t scatter_addr = hist_scatter_dfb.get_write_ptr();
    const uint32_t sum_addr = hist_sum_dfb.get_write_ptr();
    const uint32_t select_addr = select_dfb.get_write_ptr();

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
            slice_arrived.set_multicast(noc, mcast_x_start, mcast_y_start, mcast_x_end, mcast_y_end, num_mcast_dests);
            noc.async_write_barrier();
        }
    } else {
        DeviceZoneScopedN("SEL-OWNER-WAIT");
        slice_arrived.wait(1);
    }
    slice_arrived.set(0);

    if (core_index < num_owners) {
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

    volatile tt_l1_ptr uint32_t* select = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(select_addr);
    if (core_index == 0) {
        {
            DeviceZoneScopedN("SEL-COORD-WAIT");
            hist_arrived.wait(num_owners);
        }
        hist_arrived.set(0);
        invalidate_l1_cache();
        [[maybe_unused]] volatile tt_l1_ptr uint32_t* global_hist =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sum_addr);
        //     // Walk bins from the largest scores down until the running count reaches min(k, total). The bin
        //     // where that happens holds the k-th largest score.
        //     uint32_t global_total = 0;
        //     for (uint32_t bin = 0; bin < num_hist_bins; ++bin) {
        //         global_total += global_hist[bin];
        //     }
        //     const uint32_t target = std::min<uint32_t>(k, global_total);
        //     uint32_t above = 0;
        //     uint32_t threshold_bin = num_hist_bins;
        //     while (threshold_bin > 0) {
        //         --threshold_bin;
        //         if (above + global_hist[threshold_bin] >= target) {
        //             break;
        //         }
        //         above += global_hist[threshold_bin];
        //     }

        //     DPRINT(
        //         "top{} threshold_bin {} above {} needed_from_bin {} bin_count {} total {}\n",
        //         k,
        //         threshold_bin,
        //         above,
        //         target - above,
        //         global_hist[threshold_bin],
        //         global_total);

        //     select[0] = threshold_bin;
        //     select[1] = above;
        //     select[2] = target - above;
        //     select[3] = global_total;
        //     (void)ckernel::load_blocking(&select[kSelectHeaderWords - 1]);

        //     // Phase 1: send the header to every core.
        //     if (num_mcast_dests > 0) {
        //         noc.async_write_multicast(
        //             UnicastEndpoint{},
        //             MulticastEndpoint{},
        //             kSelectHeaderWords * sizeof(uint32_t),
        //             num_mcast_dests,
        //             {.addr = select_addr},
        //             {.noc_x_start = mcast_x_start,
        //              .noc_y_start = mcast_y_start,
        //              .noc_x_end = mcast_x_end,
        //              .noc_y_end = mcast_y_end,
        //              .addr = select_addr});
        //         noc.async_write_barrier();
        //         select_ready.set(1);
        //         select_ready.set_multicast(noc, mcast_x_start, mcast_y_start, mcast_x_end, mcast_y_end,
        //         num_mcast_dests); noc.async_write_barrier();
        //     }
    }
    // if (core_index != 0) {
    //     select_ready.wait(1);
    //     invalidate_l1_cache();
    // }

    // // Every core reports its keys above the threshold bin and inside it to slot core_index of
    // // hist_gather on the coordinator. The local histogram is not needed afterwards, so it stages the
    // // message.
    // const uint32_t threshold_bin = select[0];
    // uint32_t core_above = 0;
    // for (uint32_t bin = threshold_bin + 1; bin < num_hist_bins; ++bin) {
    //     core_above += hist[bin];
    // }
    // const uint32_t core_bin_count = hist[threshold_bin];
    // hist[0] = core_above;
    // hist[1] = core_bin_count;
    // (void)ckernel::load_blocking(&hist[1]);
    // noc.async_write(
    //     UnicastEndpoint{},
    //     UnicastEndpoint{},
    //     kCountsWords * sizeof(uint32_t),
    //     {.addr = hist_addr},
    //     {.noc_x = coord_x, .noc_y = coord_y, .addr = gather_addr + core_index * kCountsWords * sizeof(uint32_t)});
    // noc.async_write_barrier();
    // hist_arrived.up(noc, coord_x, coord_y, 1);
    // noc.async_atomic_barrier();

    // if (core_index == 0) {
    //     hist_arrived.wait(num_cores);
    //     hist_arrived.set(0);
    //     invalidate_l1_cache();
    //     volatile tt_l1_ptr uint32_t* gathered = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gather_addr);

    //     // Phase 2: per-core allocation. A core contributes all of its keys above the threshold bin plus
    //     // a share of the threshold bin's keys; that share is handed out in core order, so ties go to
    //     // lower key indices. Output offsets are the running sum of contributions.
    //     uint32_t remaining_from_bin = select[2];
    //     uint32_t next_offset = 0;
    //     for (uint32_t c = 0; c < num_cores; ++c) {
    //         volatile tt_l1_ptr uint32_t* counts = gathered + c * kCountsWords;
    //         const uint32_t core_from_bin = std::min<uint32_t>(counts[1], remaining_from_bin);
    //         remaining_from_bin -= core_from_bin;

    //         volatile tt_l1_ptr uint32_t* entry = select + kSelectHeaderWords + c * kSelectWordsPerCore;
    //         entry[0] = counts[0] + core_from_bin;
    //         entry[1] = core_from_bin;
    //         entry[2] = next_offset;
    //         next_offset += counts[0] + core_from_bin;
    //     }
    //     DPRINT("top{} allocated {} of {}\n", k, next_offset, select[1] + select[2]);
    //     (void)ckernel::load_blocking(&select[kSelectHeaderWords + (num_cores - 1) * kSelectWordsPerCore + 2]);

    //     if (num_mcast_dests > 0) {
    //         noc.async_write_multicast(
    //             UnicastEndpoint{},
    //             MulticastEndpoint{},
    //             num_select_words * sizeof(uint32_t),
    //             num_mcast_dests,
    //             {.addr = select_addr},
    //             {.noc_x_start = mcast_x_start,
    //              .noc_y_start = mcast_y_start,
    //              .noc_x_end = mcast_x_end,
    //              .noc_y_end = mcast_y_end,
    //              .addr = select_addr});
    //         noc.async_write_barrier();
    //         select_ready.set(2);
    //         select_ready.set_multicast(noc, mcast_x_start, mcast_y_start, mcast_x_end, mcast_y_end, num_mcast_dests);
    //         noc.async_write_barrier();
    //     }
    // } else {
    //     select_ready.wait(2);
    //     invalidate_l1_cache();
    // }
    // select_ready.set(0);

    // // This core's assignment: emit num_contribute keys (all local keys above the threshold bin plus
    // // num_from_bin keys from inside it) to output rows [output_offset, output_offset + num_contribute).
    // volatile tt_l1_ptr uint32_t* my_entry = select + kSelectHeaderWords + core_index * kSelectWordsPerCore;
    // const uint32_t num_contribute = my_entry[0];
    // const uint32_t num_from_bin = my_entry[1];
    // const uint32_t output_offset = my_entry[2];
    // if (num_contribute != 0) {
    //     DPRINT(
    //         "core {} contributes {} ({} from threshold bin) at output offset {}\n",
    //         core_index,
    //         num_contribute,
    //         num_from_bin,
    //         output_offset);
    // }

    // scores_dfb.pop_front(num_local_tiles);

    // kv_dfb.wait_front(1);
    // if (core_index == 0) {
    //     noc.async_write(kv_dfb, output, kv_dfb.get_entry_size(), {}, {.page_id = 0});
    //     noc.async_write_barrier();
    // }
    // kv_dfb.pop_front(1);
}
