// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    // compile-time args
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t num_partials = get_compile_time_arg_val(1);
    constexpr auto partial_args = TensorAccessorArgs<2>();
    constexpr auto prefix_sum_args = TensorAccessorArgs<partial_args.next_compile_time_args_offset()>();
    constexpr auto shift_args = TensorAccessorArgs<prefix_sum_args.next_compile_time_args_offset()>();
    constexpr auto mass_args = TensorAccessorArgs<shift_args.next_compile_time_args_offset()>();
    constexpr auto live_scores_args = TensorAccessorArgs<mass_args.next_compile_time_args_offset()>();

    // runtime args
    const auto partial_addr = get_arg_val<uint32_t>(0);
    const auto prefix_sum_addr = get_arg_val<uint32_t>(1);
    const auto shift_addr = get_arg_val<uint32_t>(2);
    const auto mass_addr = get_arg_val<uint32_t>(3);
    const auto live_scores_addr = get_arg_val<uint32_t>(4);
    const auto num_output_tiles = get_arg_val<uint32_t>(5);
    const auto start_id = get_arg_val<uint32_t>(6);

    // Common args: every core reads the same site, and the host re-patches the
    // four site offsets in place on a program-cache hit.
    //
    // Each scalar tensor's read site, already multiplied out to pages. Zero for a
    // scalar that carries a single plane, so the site is applied per operand.
    const auto shift_page_offset = get_common_arg_val<uint32_t>(0);
    const auto mass_page_offset = get_common_arg_val<uint32_t>(1);
    const auto live_scores_page_offset = get_common_arg_val<uint32_t>(2);
    // The partial's read site, in whole Ht*Wt planes rather than scalar rows.
    const auto partial_page_offset = get_common_arg_val<uint32_t>(3);
    // Unsummed statistics only: the distance from a rank's sum of squares to its
    // dots, and from one rank's pair to the next.
    const auto live_dots_page_offset = get_common_arg_val<uint32_t>(4);
    const auto live_partial_page_stride = get_common_arg_val<uint32_t>(5);

    constexpr uint32_t cb_id_wide = 0;
    constexpr uint32_t cb_id_scalars = 2;
    constexpr uint32_t wide_tile_bytes = get_tile_size(cb_id_wide);
    constexpr uint32_t scalar_tile_bytes = get_tile_size(cb_id_scalars);
    constexpr uint32_t kOperands = 2;
    constexpr uint32_t kFixedScalars = 2;
    constexpr uint32_t kStatsPerPartial = 2;
    constexpr uint32_t kScalars = kFixedScalars + (num_partials == 0 ? 1 : kStatsPerPartial * num_partials);

    Noc noc;
    CircularBuffer cb_wide(cb_id_wide);
    CircularBuffer cb_scalars(cb_id_scalars);

    auto partial_accessor = TensorAccessor(partial_args, partial_addr);
    auto prefix_sum_accessor = TensorAccessor(prefix_sum_args, prefix_sum_addr);
    auto shift_accessor = TensorAccessor(shift_args, shift_addr);
    auto mass_accessor = TensorAccessor(mass_args, mass_addr);
    auto live_scores_accessor = TensorAccessor(live_scores_args, live_scores_addr);

    // This core owns a contiguous run of output tiles, so the scalar traffic is
    // negligible: the Wt tiles of one token row share one scalar per tensor.
    for (uint32_t i = start_id; i < start_id + num_output_tiles; ++i) {
        // Fetch a scalar set on the first tile and whenever the token row turns
        // over. `i % Wt == 0` is exactly the row boundary, and the scalar
        // tensors are one tile column wide so the row index is the page id.
        //
        // They all go into one CB so the derivation reads them through a single
        // unpack configuration; the device operation rejects a mixed scalar dtype
        // for that reason.
        if (i == start_id || i % Wt == 0) {
            const uint32_t row = i / Wt;

            cb_scalars.reserve_back(kScalars);
            noc.async_read(
                shift_accessor,
                cb_scalars,
                scalar_tile_bytes,
                {.page_id = row + shift_page_offset},
                {.offset_bytes = 0});
            noc.async_read(
                mass_accessor,
                cb_scalars,
                scalar_tile_bytes,
                {.page_id = row + mass_page_offset},
                {.offset_bytes = scalar_tile_bytes});

            if constexpr (num_partials == 0) {
                noc.async_read(
                    live_scores_accessor,
                    cb_scalars,
                    scalar_tile_bytes,
                    {.page_id = row + live_scores_page_offset},
                    {.offset_bytes = kFixedScalars * scalar_tile_bytes});
            } else {
                // Rank-major, matching the layout a gathering collective leaves,
                // and compute sums across the pairs.
                uint32_t page = row + live_scores_page_offset;
                uint32_t offset_bytes = kFixedScalars * scalar_tile_bytes;
                for (uint32_t p = 0; p < num_partials; ++p) {
                    noc.async_read(
                        live_scores_accessor,
                        cb_scalars,
                        scalar_tile_bytes,
                        {.page_id = page},
                        {.offset_bytes = offset_bytes});
                    noc.async_read(
                        live_scores_accessor,
                        cb_scalars,
                        scalar_tile_bytes,
                        {.page_id = page + live_dots_page_offset},
                        {.offset_bytes = offset_bytes + scalar_tile_bytes});
                    page += live_partial_page_stride;
                    offset_bytes += kStatsPerPartial * scalar_tile_bytes;
                }
            }
            noc.async_read_barrier();
            cb_scalars.push_back(kScalars);
        }

        // Both full-width operands share the output's tile indexing, and they go
        // into one CB as a pair so compute drives both MACs off a single CB pair.
        cb_wide.reserve_back(kOperands);
        noc.async_read(
            partial_accessor, cb_wide, wide_tile_bytes, {.page_id = i + partial_page_offset}, {.offset_bytes = 0});
        noc.async_read(
            prefix_sum_accessor, cb_wide, wide_tile_bytes, {.page_id = i}, {.offset_bytes = wide_tile_bytes});
        noc.async_read_barrier();
        cb_wide.push_back(kOperands);
    }
}
