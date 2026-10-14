// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Merges every part's top-k candidates into the global top-k and gathers this core's slice of the
// selected kv_cache rows. Every core runs the whole merge on its own copy of the candidates, so the
// cores need no synchronization.
//
// Part p's candidates are one page {k rows, k order keys, number selected, pad...}; entries are in
// key order and parts cover ascending key ranges, so (part, entry) order is key order.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "tt_metal/tools/profiler/kernel_profiler.hpp"
#include "ttnn/operations/experimental/deepseek/fused_lightning_select_kv/device/kernels/dataflow/gather_kv_rows.hpp"

void kernel_main() {
    // ---- Compile-time args ----
    constexpr uint32_t k = get_arg(args::k);
    constexpr uint32_t num_parts = get_arg(args::num_parts);
    constexpr uint32_t cand_page_bytes = get_arg(args::cand_page_bytes);
    constexpr uint32_t num_hist_bins = get_arg(args::num_hist_bins);
    static_assert(num_hist_bins == 256, "the merge histograms 8 order-key bits per pass");
    constexpr uint32_t kHistBits = 8;
    constexpr uint32_t rows_per_core = get_arg(args::rows_per_core);
    constexpr uint32_t kv_stage_rows = get_arg(args::kv_stage_rows);
    constexpr uint32_t output_row_offset = get_arg(args::output_row_offset);
    constexpr uint32_t cand_words = cand_page_bytes / sizeof(uint32_t);

    // ---- Runtime args ----
    const uint32_t core_index = get_arg(args::core_index);

    // ---- Tensors ----
    const auto candidates = TensorAccessor(tensor::candidates);  // [1, 1, num_parts, 2k + 16], one page per part
    const auto kv_cache = TensorAccessor(tensor::kv_cache);      // one row per page
    const auto output = TensorAccessor(tensor::output);          // one row per page

    // ---- Dataflow buffers ----
    DataflowBuffer cand_dfb(dfb::cand);    // every part's candidates
    DataflowBuffer hist_dfb(dfb::hist);    // radix histogram
    DataflowBuffer rows_dfb(dfb::rows);    // kv_cache rows of this core's output slice
    DataflowBuffer stage_dfb(dfb::stage);  // kv rows in flight

    Noc noc;

    cand_dfb.reserve_back(1);
    for (uint32_t p = 0; p < num_parts; ++p) {
        noc.async_read(candidates, cand_dfb, cand_page_bytes, {.page_id = p}, {.offset_bytes = p * cand_page_bytes});
    }
    noc.async_read_barrier();
    invalidate_l1_cache();
    const uint32_t* cand = reinterpret_cast<const uint32_t*>(cand_dfb.get_write_ptr());

    uint32_t total = 0;
    for (uint32_t p = 0; p < num_parts; ++p) {
        total += cand[p * cand_words + 2 * k];
    }
    const uint32_t num_selected = std::min(k, total);

    // Radix select, 8 bits per pass: fix the prefix of the k-th largest order key until its bin can
    // be taken whole or all 32 bits are fixed; then needed of the keys equal to it are taken, in order.
    hist_dfb.reserve_back(1);
    uint32_t* hist = reinterpret_cast<uint32_t*>(hist_dfb.get_write_ptr());
    uint32_t prefix = 0;
    uint32_t prefix_bits = 0;
    uint32_t needed = num_selected;
    {
        DeviceZoneScopedN("MERGE-SELECT");
        while (true) {
            for (uint32_t bin = 0; bin < num_hist_bins; ++bin) {
                hist[bin] = 0;
            }
            const uint32_t bin_shift = 32 - prefix_bits - kHistBits;
            for (uint32_t p = 0; p < num_parts; ++p) {
                const uint32_t* keys = cand + p * cand_words + k;
                const uint32_t n = cand[p * cand_words + 2 * k];
                for (uint32_t j = 0; j < n; ++j) {
                    const uint32_t key = keys[j];
                    if (prefix_bits == 0 || (key >> (32 - prefix_bits)) == prefix) {
                        ++hist[(key >> bin_shift) & (num_hist_bins - 1)];
                    }
                }
            }
            uint32_t above = 0;
            uint32_t bin = num_hist_bins;
            while (bin > 0) {
                --bin;
                if (above + hist[bin] >= needed) {
                    break;
                }
                above += hist[bin];
            }
            needed -= above;
            prefix = (prefix << kHistBits) | bin;
            prefix_bits += kHistBits;
            if (needed == 0 || needed == hist[bin] || prefix_bits == 32) {
                break;
            }
        }
    }

    // Walk the candidates in key order and keep the rows whose output position is in this core's slice.
    const uint32_t slice_start = std::min(core_index * rows_per_core, num_selected);
    const uint32_t slice_end = std::min(slice_start + rows_per_core, num_selected);
    rows_dfb.reserve_back(1);
    volatile tt_l1_ptr uint32_t* rows = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows_dfb.get_write_ptr());
    const uint32_t final_shift = 32 - prefix_bits;
    uint32_t from_bin_left = needed;
    uint32_t position = 0;
    {
        DeviceZoneScopedN("MERGE-LIST");
        for (uint32_t p = 0; p < num_parts && position < slice_end; ++p) {
            const uint32_t* part = cand + p * cand_words;
            const uint32_t n = part[2 * k];
            for (uint32_t j = 0; j < n && position < slice_end; ++j) {
                const uint32_t top = part[k + j] >> final_shift;
                bool take = top > prefix;
                if (top == prefix && from_bin_left != 0) {
                    --from_bin_left;
                    take = true;
                }
                if (take) {
                    if (position >= slice_start) {
                        rows[position - slice_start] = part[j];
                    }
                    ++position;
                }
            }
        }
    }

    {
        DeviceZoneScopedN("MERGE-GATHER");
        gather_kv_rows<kv_stage_rows>(
            noc, kv_cache, output, stage_dfb, rows, 0, slice_end - slice_start, output_row_offset + slice_start);
    }
}
