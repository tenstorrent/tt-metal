// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // compile-time args
    constexpr uint32_t inner_tile_size = get_arg(args::inner_tile_size);
    constexpr uint32_t num_sites = get_arg(args::num_sites);
    constexpr uint32_t sites_per_group = get_arg(args::sites_per_group);
    constexpr uint32_t num_groups = get_arg(args::num_groups);

    // runtime args
    const auto num_positions = get_arg(args::num_positions);
    const auto start_id = get_arg(args::start_id);

    Noc noc;
    DataflowBuffer dfb_out(dfb::out);

    const uint32_t output_tile_bytes = dfb_out.get_tile_size();
    auto tensor_accessor = TensorAccessor(tensor::output);

    // Compute emits a group's sites together for one tile position, so a
    // position's outputs are scattered across the site planes rather than
    // contiguous. The reduced axis collapses to 1, so within a plane the page
    // index is the position itself.
    for (uint32_t group = 0; group < num_groups; ++group) {
        const uint32_t first_site = group * sites_per_group;
        const uint32_t sites_in_group =
            (first_site + sites_per_group <= num_sites) ? sites_per_group : num_sites - first_site;

        for (uint32_t i = start_id; i < start_id + num_positions; ++i) {
            dfb_out.wait_front(sites_in_group);
            uint32_t read_offset = 0;
            uint32_t page_id = first_site * inner_tile_size + i;
            for (uint32_t s = 0; s < sites_in_group; ++s) {
                noc.async_write(
                    dfb_out, tensor_accessor, output_tile_bytes, {.offset_bytes = read_offset}, {.page_id = page_id});
                read_offset += output_tile_bytes;
                page_id += inner_tile_size;
            }
            noc.async_write_barrier();
            dfb_out.pop_front(sites_in_group);
        }
    }
}
