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
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t num_sites = get_named_compile_time_arg_val("num_sites");
    constexpr uint32_t Ht = get_named_compile_time_arg_val("Ht");
    constexpr uint32_t sites_per_group = get_named_compile_time_arg_val("sites_per_group");
    constexpr uint32_t num_groups = get_named_compile_time_arg_val("num_groups");
    constexpr uint32_t num_closed_sites = get_named_compile_time_arg_val("num_closed_sites");
    constexpr uint32_t num_open_sites = num_sites - num_closed_sites;
    constexpr uint32_t Wt = get_named_compile_time_arg_val("Wt");
    constexpr auto partials_args = TensorAccessorArgs<0>();
    constexpr auto closed_args = TensorAccessorArgs<partials_args.next_compile_time_args_offset()>();
    constexpr auto shift_args = TensorAccessorArgs<closed_args.next_compile_time_args_offset()>();
    constexpr auto mass_args = TensorAccessorArgs<shift_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_shift = get_named_compile_time_arg_val("cb_shift_out");
    constexpr uint32_t cb_mass = get_named_compile_time_arg_val("cb_mass_out");
    constexpr uint32_t cb_partials = get_named_compile_time_arg_val("cb_partials");
    constexpr uint32_t cb_staging = get_named_compile_time_arg_val("cb_staging");
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
    // as zero for every page written from it. With RISC-V stores, not NoC zero reads: those
    // share this core's NoC port with the reader's first candidate reads, and at tens of KB
    // they hold up the first run.
    dataflow_kernel_lib::fill_l1_range<sizeof(uint32_t)>(staging_base, num_open_sites * scalar_tile_bytes, 0);

    // Every open site's column of a row's shift or mass tile, each as column 0 of page
    // r * Ht + row, r counted among the open sites. A closed site's read is already whole.
    auto scatter_scalar = [&](CircularBuffer& src_obj, const auto& accessor, uint32_t row) {
        src_obj.wait_front(1);
        for (uint32_t r = 0; r < num_open_sites; ++r) {
            dataflow_kernel_lib::copy_tile_column<cb_staging, uint32_t>(
                src_obj.get_read_ptr(), 1 + num_closed_sites + r, staging_base + r * scalar_tile_bytes, 0);
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

    for_each_attn_res_block_mix_item([&](const AttnResBlockMixItem& item) {
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
            write_partials_group<Ht, Wt, num_closed_sites, sites_per_group>(
                noc,
                cb_partials_obj,
                partials_accessor,
                closed_accessor,
                value_tile_bytes,
                item,
                group * sites_per_group,
                sites_in_group(group, sites_per_group, num_sites));
        }
        noc.async_write_barrier();
    });
}
