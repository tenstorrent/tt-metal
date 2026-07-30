// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    // compile-time args
    constexpr uint32_t input_granularity = get_compile_time_arg_val(0);
    constexpr uint32_t num_candidates = get_compile_time_arg_val(1);
    constexpr uint32_t inner_tile_size = get_compile_time_arg_val(2);
    constexpr uint32_t reduce_tile_size = get_compile_time_arg_val(3);
    constexpr uint32_t Wt = get_compile_time_arg_val(4);
    constexpr uint32_t weight_inner_tile_size = get_compile_time_arg_val(5);
    constexpr uint32_t weight_reduce_tile_size = get_compile_time_arg_val(6);
    constexpr auto input_args = TensorAccessorArgs<7>();
    constexpr auto weight_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

    // runtime args
    const auto input_addr = get_arg_val<uint32_t>(0);
    const auto weight_addr = get_arg_val<uint32_t>(1);
    const auto num_output_tiles = get_arg_val<uint32_t>(2);
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

    uint32_t granularity_index = 0;
    uint32_t write_offset = 0;

    // This core owns a contiguous run of output tiles. Contiguous is what makes
    // the weight traffic negligible: the Wt tiles of one token row share a
    // single weight set, and a contiguous run walks whole rows.
    for (uint32_t i = start_id; i < start_id + num_output_tiles; ++i) {
        const uint32_t batch = i / inner_tile_size;
        const uint32_t rem = i - batch * inner_tile_size;

        // Fetch a weight set on the first tile and whenever the token row turns
        // over. `i % Wt == 0` is exactly the row boundary: the row index only
        // advances when the width index wraps, and inner_tile_size is a whole
        // number of rows.
        if (i == start_id || i % Wt == 0) {
            uint32_t weight_tile_id = batch * weight_reduce_tile_size + rem / Wt;
            cb_in1_obj.reserve_back(num_candidates);
            uint32_t weight_write_offset = 0;
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
            noc.async_read_barrier();
            cb_in1_obj.push_back(num_candidates);
        }

        // The candidate tiles for output tile i start at the same position
        // within the batch and step by the inner block, same as fast_reduce_nc.
        uint32_t read_tile_id = batch * reduce_tile_size + rem;
        for (uint32_t c = 0; c < num_candidates; ++c) {
            if (granularity_index == 0) {
                cb_in0_obj.reserve_back(input_granularity);
                write_offset = 0;
            }
            noc.async_read(
                input_accessor,
                cb_in0_obj,
                input_tile_bytes,
                {.page_id = read_tile_id},
                {.offset_bytes = write_offset});
            write_offset += input_tile_bytes;
            read_tile_id += inner_tile_size;
            ++granularity_index;
            if (granularity_index == input_granularity) {
                noc.async_read_barrier();
                cb_in0_obj.push_back(input_granularity);
                granularity_index = 0;
            }
        }
    }
}
