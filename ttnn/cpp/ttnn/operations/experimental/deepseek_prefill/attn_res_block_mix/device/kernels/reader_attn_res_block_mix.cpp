// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Brings the row's statistics and the run's candidate tiles in, then builds the per-site
// weights from the exponentials compute leaves behind: site r's weight on candidate c is
// column 1 + r of exponential tile c, moved to column 0 of a tile of its own. The weights
// go out one site group at a time, so compute mixes a group while the next is built, and
// they never exist outside L1.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/experimental/deepseek_prefill/attn_res_block_mix/device/kernels/attn_res_block_mix_items.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/attn_res_block_mix/device/kernels/attn_res_tile_columns.hpp"

void kernel_main() {
    constexpr uint32_t num_candidates = get_compile_time_arg_val(0);
    constexpr uint32_t num_partials = get_compile_time_arg_val(1);
    constexpr uint32_t Ht = get_compile_time_arg_val(2);
    constexpr uint32_t num_sites = get_compile_time_arg_val(3);
    constexpr uint32_t sites_per_group = get_compile_time_arg_val(4);
    constexpr uint32_t num_groups = get_compile_time_arg_val(5);
    constexpr uint32_t kSplitArgs = 6;
    constexpr uint32_t Wt = get_compile_time_arg_val(kSplitArgs + 4);
    constexpr auto stats_args = TensorAccessorArgs<kSplitArgs + 5>();
    constexpr auto sealed_args = TensorAccessorArgs<stats_args.next_compile_time_args_offset()>();

    // Common args: the statistics, then one (address, first page) pair per candidate.
    constexpr uint32_t kCandidateArgs = 1;

    constexpr uint32_t cb_stats = 0;
    constexpr uint32_t cb_exp = 5;
    constexpr uint32_t cb_values = 6;
    constexpr uint32_t cb_weights = 7;
    constexpr uint32_t stats_tile_bytes = get_tile_size(cb_stats);
    constexpr uint32_t value_tile_bytes = get_tile_size(cb_values);

    Noc noc;
    CircularBuffer cb_stats_obj(cb_stats);
    CircularBuffer cb_exp_obj(cb_exp);
    CircularBuffer cb_values_obj(cb_values);
    CircularBuffer cb_weights_obj(cb_weights);
    auto stats_accessor = TensorAccessor(stats_args, get_common_arg_val<uint32_t>(0));

    for_each_attn_res_block_mix_item<kSplitArgs>([&](const AttnResBlockMixItem& item) {
        // The row's statistics, candidate-major with the partials innermost: a candidate's
        // partials are adjacent pages.
        cb_stats_obj.reserve_back(num_candidates * num_partials);
        for (uint32_t c = 0; c < num_candidates; ++c) {
            for (uint32_t p = 0; p < num_partials; ++p) {
                noc.async_read(
                    stats_accessor,
                    cb_stats_obj,
                    stats_tile_bytes,
                    {.page_id = (c * Ht + item.row) * num_partials + p},
                    {.offset_bytes = (c * num_partials + p) * stats_tile_bytes});
            }
        }
        noc.async_read_barrier();
        cb_stats_obj.push_back(num_candidates * num_partials);

        // The run's candidate tiles, position-major, read once and held for every group.
        // They land while compute scores the row.
        const uint32_t num_tiles = (item.end - item.begin) * num_candidates;
        cb_values_obj.reserve_back(num_tiles);
        uint32_t offset = 0;
        for (uint32_t w = item.begin; w < item.end; ++w) {
            for (uint32_t c = 0; c < num_candidates; ++c) {
                const uint32_t address = get_common_arg_val<uint32_t>(kCandidateArgs + 2 * c);
                const uint32_t first_page = get_common_arg_val<uint32_t>(kCandidateArgs + 2 * c + 1);
                noc.async_read(
                    TensorAccessor(sealed_args, address),
                    cb_values_obj,
                    value_tile_bytes,
                    {.page_id = first_page + item.row * Wt + w},
                    {.offset_bytes = offset});
                offset += value_tile_bytes;
            }
        }

        // The weights, a group at a time: site s of the group on candidate c at s * C + c.
        cb_exp_obj.wait_front(num_candidates);
        const uint32_t exp_base = cb_exp_obj.get_read_ptr();
        for (uint32_t group = 0; group < num_groups; ++group) {
            const uint32_t first_site = group * sites_per_group;
            const uint32_t sites = first_site + sites_per_group <= num_sites ? sites_per_group : num_sites - first_site;
            cb_weights_obj.reserve_back(sites * num_candidates);
            const uint32_t weights_base = cb_weights_obj.get_write_ptr();
            for (uint32_t s = 0; s < sites; ++s) {
                for (uint32_t c = 0; c < num_candidates; ++c) {
                    copy_column_to_column0<uint16_t>(
                        exp_base + c * value_tile_bytes,
                        weights_base + (s * num_candidates + c) * value_tile_bytes,
                        1 + first_site + s);
                }
            }
            if (group == 0) {
                // Compute takes the candidate tiles with the first group's weights.
                noc.async_read_barrier();
                cb_values_obj.push_back(num_tiles);
            }
            cb_weights_obj.push_back(sites * num_candidates);
        }
        cb_exp_obj.pop_front(num_candidates);
    });
}
