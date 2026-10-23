// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Merges every part's top-k candidates into the global top-k and gathers the selected kv_cache rows.
// A group of cores runs the merge as a radix select. The candidates, in (part, entry) order, are split
// into one slice per core. Each pass every core histograms its slice; core o sums bins
// [o * owner_bins, (o + 1) * owner_bins) of every core's histogram and sends the sums to every core,
// so all cores find the same threshold. The cores then exchange how many of their keys are above it
// and equal to it, and each core lists and gathers its range of the output, reading the candidates it
// needs from the other cores' slices. In CANDIDATE_SOURCE mode a second group of cores merges the
// parts' candidate blocks the same way and writes the chosen blocks and their key count.
//
// Part p's candidates are one page {k rows, k order keys, 16 header words, then in candidate source
// mode candidate_topk_blocks blocks and their order keys}; header {keys selected, blocks selected,
// selectable keys, 0...}. Entries are in key (block) order and parts cover ascending key ranges, so
// (part, entry) order is key (block) order, and ties go to the lower key (block) as in the unsplit op.

#include <algorithm>
#include <cstdint>
#include <functional>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "ckernel.h"
#include "experimental/kernel_args.h"
#include "tt_metal/tools/profiler/kernel_profiler.hpp"
#include "ttnn/operations/experimental/deepseek/fused_lightning_select_kv/device/kernels/dataflow/gather_kv_rows.hpp"

// Adds the 8-bit bins (key >> shift) of keys[0, n) to hist. The keys of a top-k mostly share their
// leading bits, and back-to-back increments of one L1 word stall on each other, so runs of one bin
// are counted in a register.
inline void histogram_runs(uint32_t* hist, const uint32_t* keys, uint32_t n, uint32_t shift) {
    if (n == 0) {
        return;
    }
    uint32_t bin = (keys[0] >> shift) & 0xFF;
    uint32_t run = 0;
    for (uint32_t j = 0; j < n; ++j) {
        const uint32_t b = (keys[j] >> shift) & 0xFF;
        if (b == bin) {
            ++run;
        } else {
            hist[bin] += run;
            bin = b;
            run = 1;
        }
    }
    hist[bin] += run;
}

// sums[b] = sum over c < num_entries of scatter[c * num_bins + b], accumulated in registers.
template <uint32_t num_bins>
inline void sum_owner_bins(uint32_t* sums, const uint32_t* scatter, uint32_t num_entries) {
    uint32_t acc[num_bins] = {};
    for (uint32_t c = 0; c < num_entries; ++c) {
        const uint32_t* entry = scatter + c * num_bins;
#pragma GCC unroll 64
        for (uint32_t b = 0; b < num_bins; ++b) {
            acc[b] += entry[b];
        }
    }
#pragma GCC unroll 64
    for (uint32_t b = 0; b < num_bins; ++b) {
        sums[b] = acc[b];
    }
}

void kernel_main() {
    // ---- Compile-time args ----
    constexpr uint32_t k = get_arg(args::k);
    constexpr uint32_t num_parts = get_arg(args::num_parts);
    constexpr uint32_t key_section_bytes = get_arg(args::key_section_bytes);
    constexpr uint32_t num_hist_bins = get_arg(args::num_hist_bins);
    constexpr uint32_t num_key_cores = get_arg(args::num_key_cores);
    constexpr uint32_t num_block_cores = get_arg(args::num_block_cores);
    constexpr uint32_t key_slice_size = get_arg(args::key_slice_size);
    constexpr uint32_t block_slice_size = get_arg(args::block_slice_size);
    constexpr uint32_t rows_per_core = get_arg(args::rows_per_core);
    constexpr uint32_t blocks_per_core = get_arg(args::blocks_per_core);
    constexpr uint32_t kv_stage_rows = get_arg(args::kv_stage_rows);
    constexpr uint32_t output_row_offset = get_arg(args::output_row_offset);
    constexpr uint32_t candidate_topk_blocks = get_arg(args::candidate_topk_blocks);
    [[maybe_unused]] constexpr uint32_t candidate_block_size = get_arg(args::candidate_block_size);
    static_assert(num_hist_bins == 256, "the merge histograms 8 order-key bits per pass");
    constexpr uint32_t kHistBits = 8;
    constexpr uint32_t kHeaderWords = 16;
    constexpr uint32_t kCountWords = 4;  // {keys above the threshold, keys equal to it, pad, pad}: 16 B for the NoC
    // Once at most kMaxTies keys are tied with the prefix, they are exchanged instead of running more
    // passes; each core's message is {count, pad, pad, pad, keys...}.
    constexpr uint32_t kMaxTies = get_arg(args::max_ties);
    constexpr uint32_t kTieHeaderWords = 4;
    constexpr uint32_t kTieSlotBytes = (kTieHeaderWords + kMaxTies) * sizeof(uint32_t);

    // ---- Runtime args ----
    const uint32_t core_index = get_arg(args::core_index);

    // Cores [0, num_key_cores) merge the keys, the next num_block_cores the candidate blocks. A group's
    // cores only ever talk to each other.
    const bool is_block_group = core_index >= num_key_cores;
    const uint32_t group_base = is_block_group ? num_key_cores : 0;
    const uint32_t group_size = is_block_group ? num_block_cores : num_key_cores;
    const uint32_t member = core_index - group_base;
    const uint32_t entries_per_part = is_block_group ? candidate_topk_blocks : k;
    const uint32_t num_entries = num_parts * entries_per_part;
    const uint32_t slice_size = is_block_group ? block_slice_size : key_slice_size;
    // Byte offset of the entries (rows or blocks) in a part's page; their order keys follow them.
    const uint32_t entries_offset = is_block_group ? key_section_bytes : 0;
    const uint32_t count_word = is_block_group ? 1 : 0;
    const uint32_t num_wanted = is_block_group ? candidate_topk_blocks : k;
    const uint32_t out_per_core = is_block_group ? blocks_per_core : rows_per_core;
    const uint32_t owner_bins = num_hist_bins / group_size;
    const uint32_t owner_bytes = owner_bins * sizeof(uint32_t);
    auto noc_x_of = [group_base](uint32_t c) { return get_common_vararg(2 * (group_base + c)); };
    auto noc_y_of = [group_base](uint32_t c) { return get_common_vararg(2 * (group_base + c) + 1); };

    // ---- Tensors ----
    const auto candidates = TensorAccessor(tensor::candidates);  // [1, 1, num_parts, width], one page per part
    const auto kv_cache = TensorAccessor(tensor::kv_cache);      // one row per page
    const auto output = TensorAccessor(tensor::output);          // one row per page

    // ---- Dataflow buffers ----
    DataflowBuffer header_dfb(dfb::header);    // every part's header
    DataflowBuffer slice_dfb(dfb::slice);      // this core's slice: slice_size entries, then their order keys
    DataflowBuffer hist_dfb(dfb::hist);        // this core's histogram, then its counts message
    DataflowBuffer scatter_dfb(dfb::scatter);  // entry c: core c's copy of the bins this core owns
    DataflowBuffer sums_dfb(dfb::sums);        // the summed bins this core owns
    DataflowBuffer global_dfb(dfb::global);    // the group's histogram
    DataflowBuffer counts_dfb(dfb::counts);    // entry c: core c's keys above and equal to the threshold
    DataflowBuffer compact_dfb(dfb::compact);  // this core's keys still tied with the threshold
    DataflowBuffer ties_dfb(dfb::ties);        // entry c: core c's tied keys, once few are left
    DataflowBuffer remote_dfb(dfb::remote);    // entry c: core c's slice, when it feeds this core's output
    DataflowBuffer list_dfb(dfb::list);        // this core's output entries (then pool_count)
    DataflowBuffer stage_dfb(dfb::stage);      // kv rows in flight

    Semaphore hist_in(sem::hist_in);      // pass p: p * group_size histogram slices received
    Semaphore sum_in(sem::sum_in);        // pass p: p * group_size summed slices received
    Semaphore counts_in(sem::counts_in);  // group_size counts received
    Semaphore ties_in(sem::ties_in);      // group_size tied-key messages received

    Noc noc;

    header_dfb.reserve_back(1);
    slice_dfb.reserve_back(1);
    hist_dfb.reserve_back(1);
    scatter_dfb.reserve_back(1);
    sums_dfb.reserve_back(1);
    global_dfb.reserve_back(1);
    counts_dfb.reserve_back(1);
    compact_dfb.reserve_back(1);
    ties_dfb.reserve_back(1);
    remote_dfb.reserve_back(1);
    list_dfb.reserve_back(1);
    const uint32_t slice_addr = slice_dfb.get_write_ptr();
    const uint32_t hist_addr = hist_dfb.get_write_ptr();
    const uint32_t scatter_addr = scatter_dfb.get_write_ptr();
    const uint32_t sums_addr = sums_dfb.get_write_ptr();
    const uint32_t global_addr = global_dfb.get_write_ptr();
    const uint32_t counts_addr = counts_dfb.get_write_ptr();
    const uint32_t remote_addr = remote_dfb.get_write_ptr();
    const uint32_t ties_addr = ties_dfb.get_write_ptr();
    const uint32_t* ties = reinterpret_cast<const uint32_t*>(ties_addr);
    const uint32_t* header = reinterpret_cast<const uint32_t*>(header_dfb.get_write_ptr());
    const uint32_t* slice = reinterpret_cast<const uint32_t*>(slice_addr);
    uint32_t* hist = reinterpret_cast<uint32_t*>(hist_addr);
    const uint32_t* scatter = reinterpret_cast<const uint32_t*>(scatter_addr);
    uint32_t* sums = reinterpret_cast<uint32_t*>(sums_addr);
    const uint32_t* global = reinterpret_cast<const uint32_t*>(global_addr);
    uint32_t* counts = reinterpret_cast<uint32_t*>(counts_addr);
    uint32_t* compact = reinterpret_cast<uint32_t*>(compact_dfb.get_write_ptr());
    const uint32_t* remote = reinterpret_cast<const uint32_t*>(remote_addr);
    volatile tt_l1_ptr uint32_t* list = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(list_dfb.get_write_ptr());

    // ---- Inputs ----
    // Slices and parts start on multiples of 16 entries, so every read is 64 B aligned.
    for (uint32_t p = 0; p < num_parts; ++p) {
        noc.async_read(
            candidates,
            header_dfb,
            kHeaderWords * sizeof(uint32_t),
            {.page_id = p, .offset_bytes = 2 * k * sizeof(uint32_t)},
            {.offset_bytes = p * kHeaderWords * sizeof(uint32_t)});
    }
    const uint32_t slice_first = member * slice_size;
    const uint32_t slice_end = std::min(slice_first + slice_size, num_entries);
    for (uint32_t s = slice_first; s < slice_end;) {
        const uint32_t p = s / entries_per_part;
        const uint32_t j = s - p * entries_per_part;
        const uint32_t n = std::min(slice_end - s, entries_per_part - j);
        noc.async_read(
            candidates,
            slice_dfb,
            n * sizeof(uint32_t),
            {.page_id = p, .offset_bytes = entries_offset + j * sizeof(uint32_t)},
            {.offset_bytes = (s - slice_first) * sizeof(uint32_t)});
        noc.async_read(
            candidates,
            slice_dfb,
            n * sizeof(uint32_t),
            {.page_id = p, .offset_bytes = entries_offset + (entries_per_part + j) * sizeof(uint32_t)},
            {.offset_bytes = (slice_size + s - slice_first) * sizeof(uint32_t)});
        s += n;
    }
    noc.async_read_barrier();
    invalidate_l1_cache();
    auto count_of = [header, count_word](uint32_t p) { return header[p * kHeaderWords + count_word]; };

    // Calls fn(first, n) for each run [first, first + n) of slice c's valid entries, in key order, with
    // first counted from the slice's start; stops once fn returns false.
    auto for_each_run = [&](uint32_t c, auto&& fn) {
        const uint32_t first = c * slice_size;
        const uint32_t end = std::min(first + slice_size, num_entries);
        for (uint32_t s = first; s < end;) {
            const uint32_t p = s / entries_per_part;
            const uint32_t part_end = std::min(end, (p + 1) * entries_per_part);
            const uint32_t valid_end = std::min(part_end, p * entries_per_part + count_of(p));
            if (valid_end > s && !fn(s - first, valid_end - s)) {
                return;
            }
            s = part_end;
        }
    };

    // ---- Radix select over the group ----
    // Fixes the prefix of the num_wanted-th largest order key, 8 bits per pass, until its bin can be
    // taken whole or all 32 bits are fixed. After the first pass each core keeps only its keys still
    // matching the prefix, in compact. Every buffer a pass writes remotely is next written only after
    // all of the group's cores have finished reading it, since each core's next send waits on this
    // pass's sums from every core.
    const uint32_t* slice_keys = slice + slice_size;
    uint32_t prefix = 0;
    uint32_t prefix_bits = 0;
    uint32_t needed = 0;
    uint32_t num_compact = 0;
    uint32_t core_above = 0;
    uint32_t core_equal = 0;
    for (uint32_t p = 0; p < num_parts; ++p) {
        needed += count_of(p);
    }
    needed = std::min(num_wanted, needed);
    {
        DeviceZoneScopedN("MERGE-SELECT");
        for (uint32_t pass = 1;; ++pass) {
            {
                DeviceZoneScopedN("MERGE-PASS-HIST");
                for (uint32_t bin = 0; bin < num_hist_bins; ++bin) {
                    hist[bin] = 0;
                }
                const uint32_t bin_shift = 32 - prefix_bits - kHistBits;
                if (prefix_bits == 0) {
                    for_each_run(member, [&](uint32_t first, uint32_t n) {
                        histogram_runs(hist, slice_keys + first, n, bin_shift);
                        return true;
                    });
                } else {
                    histogram_runs(hist, compact, num_compact, bin_shift);
                }
            }
            // RISC-V stores can still be in flight when the NoC reads the buffer; a store followed by a
            // blocking load of the same word drains them all, since stores retire in order.
            {
                DeviceZoneScopedN("MERGE-PASS-SEND-HIST");
                hist[num_hist_bins - 1] = hist[num_hist_bins - 1];
                (void)ckernel::load_blocking(&hist[num_hist_bins - 1]);
                for (uint32_t o = 0; o < group_size; ++o) {
                    noc.async_write(
                        UnicastEndpoint{},
                        UnicastEndpoint{},
                        owner_bytes,
                        {.addr = hist_addr + o * owner_bytes},
                        {.noc_x = noc_x_of(o), .noc_y = noc_y_of(o), .addr = scatter_addr + member * owner_bytes});
                }
                noc.async_write_barrier();
                for (uint32_t o = 0; o < group_size; ++o) {
                    hist_in.up(noc, noc_x_of(o), noc_y_of(o), 1);
                }
                noc.async_atomic_barrier();
            }

            {
                DeviceZoneScopedN("MERGE-PASS-WAIT-HIST");
                hist_in.wait_min(pass * group_size);
            }
            {
                DeviceZoneScopedN("MERGE-PASS-SUM");
                invalidate_l1_cache();
#ifdef CANDIDATE_SOURCE
                if (is_block_group) {
                    sum_owner_bins<num_hist_bins / num_block_cores>(sums, scatter, group_size);
                } else
#endif
                {
                    sum_owner_bins<num_hist_bins / num_key_cores>(sums, scatter, group_size);
                }
                sums[owner_bins - 1] = sums[owner_bins - 1];
                (void)ckernel::load_blocking(&sums[owner_bins - 1]);
            }
            {
                DeviceZoneScopedN("MERGE-PASS-SEND-SUM");
                for (uint32_t c = 0; c < group_size; ++c) {
                    noc.async_write(
                        UnicastEndpoint{},
                        UnicastEndpoint{},
                        owner_bytes,
                        {.addr = sums_addr},
                        {.noc_x = noc_x_of(c), .noc_y = noc_y_of(c), .addr = global_addr + member * owner_bytes});
                }
                noc.async_write_barrier();
                for (uint32_t c = 0; c < group_size; ++c) {
                    sum_in.up(noc, noc_x_of(c), noc_y_of(c), 1);
                }
                noc.async_atomic_barrier();
            }

            {
                DeviceZoneScopedN("MERGE-PASS-WAIT-SUM");
                sum_in.wait_min(pass * group_size);
            }
            DeviceZoneScopedN("MERGE-PASS-SCAN");
            invalidate_l1_cache();
            uint32_t above = 0;
            uint32_t local_above = 0;
            uint32_t bin = num_hist_bins;
            while (bin > 0) {
                --bin;
                if (above + global[bin] >= needed) {
                    break;
                }
                above += global[bin];
                local_above += hist[bin];
            }
            needed -= above;
            prefix = (prefix << kHistBits) | bin;
            prefix_bits += kHistBits;
            core_above += local_above;
            core_equal = hist[bin];
            if (needed == 0 || needed == global[bin] || prefix_bits == 32) {
                break;
            }
            const bool gather_ties = global[bin] <= kMaxTies;
            const uint32_t match_shift = 32 - prefix_bits;
            if (prefix_bits == kHistBits) {
                for_each_run(member, [&](uint32_t first, uint32_t n) {
                    for (uint32_t i = first; i < first + n; ++i) {
                        if ((slice_keys[i] >> match_shift) == prefix) {
                            compact[num_compact++] = slice_keys[i];
                        }
                    }
                    return true;
                });
            } else {
                uint32_t kept = 0;
                for (uint32_t i = 0; i < num_compact; ++i) {
                    if ((compact[i] >> match_shift) == prefix) {
                        compact[kept++] = compact[i];
                    }
                }
                num_compact = kept;
            }
            if (gather_ties) {
                // Few keys are left tied with the prefix, so every core collects all of them and finds
                // the needed-th largest exactly, instead of running the remaining passes.
                DeviceZoneScopedN("MERGE-TIES");
                hist[0] = num_compact;
                for (uint32_t i = 0; i < num_compact; ++i) {
                    hist[kTieHeaderWords + i] = compact[i];
                }
                const uint32_t message_words = (kTieHeaderWords + num_compact + 3) / 4 * 4;
                hist[message_words - 1] = hist[message_words - 1];
                (void)ckernel::load_blocking(&hist[message_words - 1]);
                for (uint32_t c = 0; c < group_size; ++c) {
                    noc.async_write(
                        UnicastEndpoint{},
                        UnicastEndpoint{},
                        message_words * sizeof(uint32_t),
                        {.addr = hist_addr},
                        {.noc_x = noc_x_of(c), .noc_y = noc_y_of(c), .addr = ties_addr + member * kTieSlotBytes});
                }
                noc.async_write_barrier();
                for (uint32_t c = 0; c < group_size; ++c) {
                    ties_in.up(noc, noc_x_of(c), noc_y_of(c), 1);
                }
                noc.async_atomic_barrier();
                ties_in.wait_min(group_size);
                invalidate_l1_cache();

                uint32_t all[kMaxTies];
                uint32_t num_ties = 0;
                for (uint32_t c = 0; c < group_size; ++c) {
                    const uint32_t* slot = ties + c * (kTieSlotBytes / sizeof(uint32_t));
                    for (uint32_t i = 0; i < slot[0]; ++i) {
                        all[num_ties++] = slot[kTieHeaderWords + i];
                    }
                }
                std::nth_element(all, all + needed - 1, all + num_ties, std::greater<uint32_t>());
                const uint32_t threshold = all[needed - 1];
                uint32_t greater = 0;
                for (uint32_t i = 0; i < num_ties; ++i) {
                    greater += all[i] > threshold ? 1 : 0;
                }
                uint32_t own_greater = 0;
                uint32_t own_equal = 0;
                for (uint32_t i = 0; i < num_compact; ++i) {
                    own_greater += compact[i] > threshold ? 1 : 0;
                    own_equal += compact[i] == threshold ? 1 : 0;
                }
                core_above += own_greater;
                core_equal = own_equal;
                prefix = threshold;
                prefix_bits = 32;
                needed -= greater;
                break;
            }
        }
    }

    // ---- Allocation ----
    // Every core takes all of its keys above the threshold and, in core order, up to the keys still
    // needed of those equal to it, so ties go to lower keys; its output starts after the lower cores'.
    hist[0] = core_above;
    hist[1] = core_equal;
    hist[2] = 0;
    hist[3] = 0;
    (void)ckernel::load_blocking(&hist[kCountWords - 1]);
    for (uint32_t c = 0; c < group_size; ++c) {
        noc.async_write(
            UnicastEndpoint{},
            UnicastEndpoint{},
            kCountWords * sizeof(uint32_t),
            {.addr = hist_addr},
            {.noc_x = noc_x_of(c),
             .noc_y = noc_y_of(c),
             .addr = counts_addr + member * kCountWords * sizeof(uint32_t)});
    }
    noc.async_write_barrier();
    for (uint32_t c = 0; c < group_size; ++c) {
        counts_in.up(noc, noc_x_of(c), noc_y_of(c), 1);
    }
    noc.async_atomic_barrier();
    counts_in.wait_min(group_size);
    invalidate_l1_cache();

    // counts entry c becomes {keys above, keys equal, keys equal taken, output offset}.
    uint32_t num_selected = 0;
    {
        uint32_t equal_left = needed;
        for (uint32_t c = 0; c < group_size; ++c) {
            uint32_t* entry = counts + c * kCountWords;
            const uint32_t from_bin = std::min(entry[1], equal_left);
            equal_left -= from_bin;
            entry[2] = from_bin;
            entry[3] = num_selected;
            num_selected += entry[0] + from_bin;
        }
    }

    // ---- This core's output range ----
    // Walks every slice whose output overlaps [out_first, out_first + list_count) in key order.
    const uint32_t out_first = member * out_per_core;
    const uint32_t list_count = num_selected > out_first ? std::min(out_per_core, num_selected - out_first) : 0;
    {
        DeviceZoneScopedN("MERGE-LIST");
        const uint32_t list_end = out_first + list_count;
        const uint32_t slice_bytes = 2 * slice_size * sizeof(uint32_t);
        auto overlaps = [&](uint32_t c) {
            const uint32_t* entry = counts + c * kCountWords;
            const uint32_t first = entry[3];
            const uint32_t last = first + entry[0] + entry[2];
            return first < list_end && last > out_first;
        };
        for (uint32_t c = 0; c < group_size && list_count != 0; ++c) {
            if (c != member && overlaps(c)) {
                noc.async_read(
                    UnicastEndpoint{},
                    UnicastEndpoint{},
                    slice_bytes,
                    {.noc_x = noc_x_of(c), .noc_y = noc_y_of(c), .addr = slice_addr},
                    {.addr = remote_addr + c * slice_bytes});
            }
        }
        noc.async_read_barrier();
        invalidate_l1_cache();

        const uint32_t final_shift = 32 - prefix_bits;
        for (uint32_t c = 0; c < group_size && list_count != 0; ++c) {
            if (!overlaps(c)) {
                continue;
            }
            const uint32_t* source = c == member ? slice : remote + c * 2 * slice_size;
            const uint32_t* keys = source + slice_size;
            uint32_t position = counts[c * kCountWords + 3];
            uint32_t equal_left = counts[c * kCountWords + 2];
            for_each_run(c, [&](uint32_t first, uint32_t n) {
                for (uint32_t i = first; i < first + n; ++i) {
                    const uint32_t top = keys[i] >> final_shift;
                    bool take = top > prefix;
                    if (top == prefix && equal_left != 0) {
                        --equal_left;
                        take = true;
                    }
                    if (take) {
                        if (position >= out_first) {
                            list[position - out_first] = source[i];
                        }
                        if (++position == list_end) {
                            return false;
                        }
                    }
                }
                return true;
            });
        }
    }

#ifdef CANDIDATE_SOURCE
    if (is_block_group) {
        // The chosen blocks, ascending, with pool_count 64 B past the longest list.
        const auto candidate_blocks = TensorAccessor(tensor::candidate_blocks);
        const auto pool_count = TensorAccessor(tensor::pool_count);
        constexpr uint32_t kCountOffsetWords = (blocks_per_core * sizeof(uint32_t) + 63) / 64 * 16;
        if (out_first < candidate_topk_blocks) {
            const uint32_t out_count = std::min(out_per_core, candidate_topk_blocks - out_first);
            for (uint32_t j = list_count; j < out_count; ++j) {
                list[j] = 0;
            }
            (void)ckernel::load_blocking(&list[out_count - 1]);
            noc.async_write(
                list_dfb,
                candidate_blocks,
                out_count * sizeof(uint32_t),
                {.offset_bytes = 0},
                {.page_id = 0, .offset_bytes = out_first * sizeof(uint32_t)});
        }
        if (member == 0) {
            // The newest block always scores highest, so it is chosen, and last; every other one is full.
            const uint32_t num_keys = header[2];
            const uint32_t pinned_block = (num_keys + candidate_block_size - 1) / candidate_block_size - 1;
            const uint32_t pinned_keys = num_keys - pinned_block * candidate_block_size;
            list[kCountOffsetWords] = num_selected == 0 ? 0 : (num_selected - 1) * candidate_block_size + pinned_keys;
            (void)ckernel::load_blocking(&list[kCountOffsetWords]);
            noc.async_write(
                list_dfb,
                pool_count,
                sizeof(uint32_t),
                {.offset_bytes = kCountOffsetWords * sizeof(uint32_t)},
                {.page_id = 0, .offset_bytes = 0});
        }
        noc.async_write_barrier();
        return;
    }
#endif

    if (list_count != 0) {
        DeviceZoneScopedN("MERGE-GATHER");
        gather_kv_rows<kv_stage_rows>(
            noc, kv_cache, output, stage_dfb, list, 0, list_count, output_row_offset + out_first);
    }
}
