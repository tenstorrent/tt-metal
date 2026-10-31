// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writes the even site groups of a run's partials as compute finishes them, and the row's
// shift and mass for each open site, as column 0 of its own page: the layout the read takes
// per-site scalars in.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/experimental/deepseek_prefill/attn_res_block_mix/device/kernels/attn_res_block_mix_items.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/attn_res_block_mix/device/kernels/attn_res_block_mix_partials.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/attn_res_block_mix/device/kernels/attn_res_tile_columns.hpp"

void kernel_main() {
    constexpr uint32_t num_sites = get_compile_time_arg_val(0);
    constexpr uint32_t Ht = get_compile_time_arg_val(1);
    constexpr uint32_t sites_per_group = get_compile_time_arg_val(2);
    constexpr uint32_t num_groups = get_compile_time_arg_val(3);
    constexpr uint32_t num_closed_sites = get_compile_time_arg_val(4);
    constexpr uint32_t num_open_sites = num_sites - num_closed_sites;
    constexpr uint32_t kSplitArgs = 5;
    constexpr uint32_t Wt = get_compile_time_arg_val(kSplitArgs + 4);
    constexpr auto partials_args = TensorAccessorArgs<kSplitArgs + 5>();
    constexpr auto closed_args = TensorAccessorArgs<partials_args.next_compile_time_args_offset()>();
    constexpr auto shift_args = TensorAccessorArgs<closed_args.next_compile_time_args_offset()>();
    constexpr auto mass_args = TensorAccessorArgs<shift_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_shift = 16;
    constexpr uint32_t cb_mass = 17;
    constexpr uint32_t cb_partials = 18;
    constexpr uint32_t cb_staging = 24;
    constexpr uint32_t value_tile_bytes = get_tile_size(cb_partials);
    constexpr uint32_t scalar_tile_bytes = get_tile_size(cb_shift);

    Noc noc;
    CircularBuffer cb_shift_obj(cb_shift);
    CircularBuffer cb_mass_obj(cb_mass);
    CircularBuffer cb_partials_obj(cb_partials);
    // Never pushed, so both pointers stay at its base and it can be addressed directly.
    CircularBuffer cb_staging_obj(cb_staging);
    const uint32_t staging_base = cb_staging_obj.get_write_ptr();

    auto partials_accessor = TensorAccessor(partials_args, get_common_arg_val<uint32_t>(0));
    auto closed_accessor = TensorAccessor(closed_args, get_common_arg_val<uint32_t>(1));
    auto shift_accessor = TensorAccessor(shift_args, get_common_arg_val<uint32_t>(2));
    auto mass_accessor = TensorAccessor(mass_args, get_common_arg_val<uint32_t>(3));

    // Zeroed once: only column 0 of a staging tile is ever rewritten, so its padding reads
    // as zero for every page written from it.
    {
        auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(staging_base);
        for (uint32_t i = 0; i < num_open_sites * scalar_tile_bytes / sizeof(uint32_t); ++i) {
            words[i] = 0;
        }
    }

    // Every open site's column of a row's shift or mass tile, each as column 0 of page
    // r * Ht + row, r counted among the open sites. A closed site's read is already whole.
    auto scatter_scalar = [&](CircularBuffer& src_obj, const auto& accessor, uint32_t row) {
        src_obj.wait_front(1);
        for (uint32_t r = 0; r < num_open_sites; ++r) {
            copy_column_to_column0<uint32_t>(
                src_obj.get_read_ptr(), staging_base + r * scalar_tile_bytes, 1 + num_closed_sites + r);
            noc.async_write(
                cb_staging_obj,
                accessor,
                scalar_tile_bytes,
                {.offset_bytes = r * scalar_tile_bytes},
                {.page_id = r * Ht + row});
        }
        noc.async_write_barrier();
        src_obj.pop_front(1);
    };

    for_each_attn_res_block_mix_item<kSplitArgs>([&](const AttnResBlockMixItem& item) {
        // The row's shift and mass belong to the row, so its first run writes them.
        if (item.chunk == 0) {
            scatter_scalar(cb_shift_obj, shift_accessor, item.row);
            scatter_scalar(cb_mass_obj, mass_accessor, item.row);
        } else {
            cb_shift_obj.wait_front(1);
            cb_shift_obj.pop_front(1);
            cb_mass_obj.wait_front(1);
            cb_mass_obj.pop_front(1);
        }

        for (uint32_t group = 0; group < num_groups; ++group) {
            if (reader_writes_group(group)) {
                continue;
            }
            const uint32_t first_site = group * sites_per_group;
            const uint32_t sites = first_site + sites_per_group <= num_sites ? sites_per_group : num_sites - first_site;
            write_partials_group<Ht, Wt, num_closed_sites>(
                noc, cb_partials_obj, partials_accessor, closed_accessor, value_tile_bytes, item, first_site, sites);
        }
        noc.async_write_barrier();
    });
}
