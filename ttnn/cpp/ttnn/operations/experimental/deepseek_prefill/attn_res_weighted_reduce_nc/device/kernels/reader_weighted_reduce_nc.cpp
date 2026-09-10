// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    // compile-time args
    constexpr uint32_t num_candidates = get_compile_time_arg_val(0);
    constexpr uint32_t inner_tile_size = get_compile_time_arg_val(1);
    constexpr uint32_t Wt = get_compile_time_arg_val(2);
    constexpr uint32_t weight_inner_tile_size = get_compile_time_arg_val(3);
    constexpr uint32_t weight_reduce_tile_size = get_compile_time_arg_val(4);
    constexpr uint32_t num_sites = get_compile_time_arg_val(5);
    constexpr uint32_t sites_per_group = get_compile_time_arg_val(6);
    constexpr uint32_t num_groups = get_compile_time_arg_val(7);
    constexpr auto input_args = TensorAccessorArgs<8>();
    constexpr auto weight_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

    // runtime args
    const auto input_addr = get_arg_val<uint32_t>(0);
    const auto weight_addr = get_arg_val<uint32_t>(1);
    const auto num_positions = get_arg_val<uint32_t>(2);
    const auto start_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t input_tile_bytes = get_tile_size(cb_id_in0);
    constexpr uint32_t weight_tile_bytes = get_tile_size(cb_id_in1);

    Noc noc;
    CircularBuffer cb_in0_obj(cb_id_in0);
    CircularBuffer cb_in1_obj(cb_id_in1);

    auto input_accessor = TensorAccessor(input_args, input_addr);
    auto weight_accessor = TensorAccessor(weight_args, weight_addr);

    // Sites outermost, positions innermost. The reverse order would hold one
    // position's candidates across every site and read the input exactly once,
    // but it would also re-fetch a group's whole weight set at every position —
    // Wt times a pass over the weight, against the num_groups passes over the
    // input this order costs. Wt is the larger number wherever the op is worth
    // using.
    for (uint32_t group = 0; group < num_groups; ++group) {
        const uint32_t first_site = group * sites_per_group;
        // The last group is short whenever the sites do not divide evenly.
        const uint32_t sites_in_group =
            (first_site + sites_per_group <= num_sites) ? sites_per_group : num_sites - first_site;

        // This core owns a contiguous run of tile positions, so the weight
        // traffic is negligible: the Wt positions of one token row share a
        // single set of sites_in_group * num_candidates tiles.
        for (uint32_t i = start_id; i < start_id + num_positions; ++i) {
            // Fetch a weight set on the first position and whenever the token
            // row turns over. `i % Wt == 0` is exactly the row boundary: the row
            // index only advances when the width index wraps.
            if (i == start_id || i % Wt == 0) {
                const uint32_t row = i / Wt;

                cb_in1_obj.reserve_back(sites_in_group * num_candidates);
                uint32_t weight_write_offset = 0;
                for (uint32_t s = 0; s < sites_in_group; ++s) {
                    uint32_t weight_tile_id = (first_site + s) * weight_reduce_tile_size + row;
                    for (uint32_t c = 0; c < num_candidates; ++c) {
                        noc.async_read(
                            weight_accessor,
                            cb_in1_obj,
                            weight_tile_bytes,
                            {.page_id = weight_tile_id},
                            {.offset_bytes = weight_write_offset});
                        weight_write_offset += weight_tile_bytes;
                        weight_tile_id += weight_inner_tile_size;
                    }
                }
                noc.async_read_barrier();
                cb_in1_obj.push_back(sites_in_group * num_candidates);
            }

            // A whole reduction as one CB unit, so a candidate's position in the
            // stream is its index and compute needs no separate mapping.
            cb_in0_obj.reserve_back(num_candidates);
            uint32_t read_tile_id = i;
            uint32_t write_offset = 0;
            for (uint32_t c = 0; c < num_candidates; ++c) {
                noc.async_read(
                    input_accessor,
                    cb_in0_obj,
                    input_tile_bytes,
                    {.page_id = read_tile_id},
                    {.offset_bytes = write_offset});
                write_offset += input_tile_bytes;
                read_tile_id += inner_tile_size;
            }
            noc.async_read_barrier();
            cb_in0_obj.push_back(num_candidates);
        }
    }
}
