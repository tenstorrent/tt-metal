// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Brings the row's statistics and the run's candidate tiles in, then builds the per-site
// weights from the exponentials compute leaves behind: site r's weight on candidate c is
// column 1 + r of exponential tile c, moved to column 0 of a tile of its own. The weights
// go out one site group at a time, so compute mixes a group while the next is built, and
// they never exist outside L1. Between groups it writes out the odd groups' partials, on its
// own NoC.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/experimental/deepseek_prefill/attn_res_block_mix/device/kernels/attn_res_block_mix_items.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/attn_res_block_mix/device/kernels/attn_res_block_mix_partials.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t num_candidates = get_named_compile_time_arg_val("num_candidates");
    constexpr uint32_t num_partials = get_named_compile_time_arg_val("num_partials");
    constexpr uint32_t Ht = get_named_compile_time_arg_val("Ht");
    constexpr uint32_t num_sites = get_named_compile_time_arg_val("num_sites");
    constexpr uint32_t sites_per_group = get_named_compile_time_arg_val("sites_per_group");
    constexpr uint32_t num_groups = get_named_compile_time_arg_val("num_groups");
    constexpr uint32_t num_closed_sites = get_named_compile_time_arg_val("num_closed_sites");
    constexpr uint32_t Wt = get_named_compile_time_arg_val("Wt");
    constexpr auto stats_args = TensorAccessorArgs<0>();
    constexpr auto sealed_args = TensorAccessorArgs<stats_args.next_compile_time_args_offset()>();
    constexpr auto partials_args = TensorAccessorArgs<sealed_args.next_compile_time_args_offset()>();
    constexpr auto closed_args = TensorAccessorArgs<partials_args.next_compile_time_args_offset()>();

    // Common args: the statistics, the partials, the closed sites' reads, then one
    // (address, first page) pair per candidate.
    constexpr uint32_t kCandidateArgs = 3;

    constexpr uint32_t cb_stats = get_named_compile_time_arg_val("cb_stats");
    constexpr uint32_t cb_exp = get_named_compile_time_arg_val("cb_exp");
    constexpr uint32_t cb_values = get_named_compile_time_arg_val("cb_values");
    constexpr uint32_t cb_weights = get_named_compile_time_arg_val("cb_weights");
    constexpr uint32_t cb_closed_mask = get_named_compile_time_arg_val("cb_closed_mask");
    constexpr uint32_t cb_partials_odd = get_named_compile_time_arg_val("cb_partials_odd");
    constexpr uint32_t stats_tile_bytes = get_tile_size(cb_stats);
    constexpr uint32_t value_tile_bytes = get_tile_size(cb_values);

    Noc noc;
    CircularBuffer cb_stats_obj(cb_stats);
    CircularBuffer cb_exp_obj(cb_exp);
    CircularBuffer cb_values_obj(cb_values);
    CircularBuffer cb_weights_obj(cb_weights);
    CircularBuffer cb_partials_odd_obj(cb_partials_odd);
    auto stats_accessor = TensorAccessor(stats_args, get_common_arg_val<uint32_t>(0));
    auto partials_accessor = TensorAccessor(partials_args, get_common_arg_val<uint32_t>(1));
    auto closed_accessor = TensorAccessor(closed_args, get_common_arg_val<uint32_t>(2));

    if constexpr (num_closed_sites > 0) {
        // 1.0f in every row of a closed site's column (1 + r for site r), 0 elsewhere. Compute
        // holds it for every row.
        CircularBuffer cb_mask_obj(cb_closed_mask);
        cb_mask_obj.reserve_back(1);
        dataflow_kernel_lib::zero_tile(cb_mask_obj);
        for (uint32_t col = 1; col <= num_closed_sites; ++col) {
            dataflow_kernel_lib::fill_tile_column<cb_closed_mask, uint32_t>(
                cb_mask_obj.get_write_ptr(), col, 0x3F800000);
        }
        cb_mask_obj.push_back(1);
    }

    for_each_attn_res_block_mix_item([&](const AttnResBlockMixItem& item) {
        // The row's statistics, partial-major: one tensor-parallel rank's statistics for every
        // candidate are adjacent, which is the window compute's partial sum reads at a time.
        cb_stats_obj.reserve_back(num_candidates * num_partials);
        for (uint32_t c = 0; c < num_candidates; ++c) {
            for (uint32_t p = 0; p < num_partials; ++p) {
                noc.async_read(
                    stats_accessor,
                    cb_stats_obj,
                    stats_tile_bytes,
                    {.page_id = (c * Ht + item.row) * num_partials + p},
                    {.offset_bytes = (p * num_candidates + c) * stats_tile_bytes});
            }
        }
        noc.async_read_barrier();
        cb_stats_obj.push_back(num_candidates * num_partials);

        // The run's candidate tiles, position-major, read once and held for every group.
        // They land while compute scores the row.
        const uint32_t num_tiles = item.num_tiles() * num_candidates;
        cb_values_obj.reserve_back(num_tiles);
        uint32_t offset = 0;
        for (uint32_t i = 0; i < item.num_tiles(); ++i) {
            const uint32_t w = item.begin + i;
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
        // Building group g needs compute done with group g - 2, which by then has also
        // finished group g - 1 into a buffer that holds a whole group: so after each group's
        // weights, the previous group's partials are ready to write if they are this core's.
        cb_exp_obj.wait_front(num_candidates);
        const uint32_t exp_base = cb_exp_obj.get_read_ptr();
        auto write_group = [&](uint32_t group) {
            write_partials_group<Ht, Wt, num_closed_sites, sites_per_group>(
                noc,
                cb_partials_odd_obj,
                partials_accessor,
                closed_accessor,
                value_tile_bytes,
                item,
                group * sites_per_group,
                sites_in_group(group, sites_per_group, num_sites));
        };
        for (uint32_t group = 0; group < num_groups; ++group) {
            const uint32_t first_site = group * sites_per_group;
            const uint32_t sites = sites_in_group(group, sites_per_group, num_sites);
            cb_weights_obj.reserve_back(sites_per_group * num_candidates);
            const uint32_t weights_base = cb_weights_obj.get_write_ptr();
            for (uint32_t s = 0; s < sites; ++s) {
                for (uint32_t c = 0; c < num_candidates; ++c) {
                    dataflow_kernel_lib::copy_tile_column<cb_weights, uint16_t>(
                        exp_base + c * value_tile_bytes,
                        1 + first_site + s,
                        weights_base + (s * num_candidates + c) * value_tile_bytes,
                        0);
                }
            }
            if (group == 0) {
                // Compute takes the candidate tiles with the first group's weights.
                noc.async_read_barrier();
                cb_values_obj.push_back(num_tiles);
            }
            cb_weights_obj.push_back(sites_per_group * num_candidates);
            if (group > 0 && reader_writes_group(group - 1)) {
                write_group(group - 1);
            }
        }
        if (reader_writes_group(num_groups - 1)) {
            write_group(num_groups - 1);
        }
        noc.async_write_barrier();
        cb_exp_obj.pop_front(num_candidates);
    });
}
