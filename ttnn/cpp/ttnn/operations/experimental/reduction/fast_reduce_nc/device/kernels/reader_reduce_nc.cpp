// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include "experimental/kernel_args.h"

inline uint32_t get_read_tile_id(uint32_t output_tile_id, uint32_t reduce_tile_size, uint32_t inner_tile_size) {
    return ((output_tile_id / inner_tile_size) * reduce_tile_size) + (output_tile_id % inner_tile_size);
}

void kernel_main() {
    // compile-time args
    constexpr auto input_granularity = get_arg(args::input_granularity);
    constexpr auto shard_factor = get_arg(args::shard_factor);
    constexpr auto num_cores_to_be_used = get_arg(args::num_cores_to_be_used);
    constexpr uint32_t outer_id_increment = shard_factor * num_cores_to_be_used;

    // runtime args
    const auto num_input_tiles = get_arg(args::num_input_tiles);
    const auto id_range_length = get_arg(args::id_range_length);
    const auto start_id = get_arg(args::start_id);
    const auto dim = get_arg(args::dim);
    const auto reduce_tile_size = get_arg(args::reduce_tile_size);
    const auto inner_tile_size = get_arg(args::inner_tile_size);

    constexpr uint32_t onetile = 1;

    Noc noc;
    DataflowBuffer cb_in0_obj(dfb::in0);
    dataflow_kernel_lib::prepare_zero_tile<dfb::in1>();

    const uint32_t input_tile_bytes = cb_in0_obj.get_entry_size();

    auto tensor_accessor = TensorAccessor(tensor::src);
    uint32_t input_granularity_index = 0;
    uint32_t write_offset = 0;

    // For each shard, start at the index of the first shard to be reduced (same
    // index as output), then increment by the appropriate increment (based on
    // the grid size), until the range length is reached. E.g. For 130 shards
    // on an 8x8 grid, the first core would have start_id equal 0,
    // outer_id_increment equal 64, and id_range_length 64*3. The outer_id
    // values would be 0, 64, and 128.
    for (uint32_t outer_id = start_id; outer_id < start_id + id_range_length; outer_id += outer_id_increment) {
        // Go through each tile of each shard.
        for (uint32_t id_offset = 0; id_offset < shard_factor; ++id_offset) {
            uint32_t i = outer_id + id_offset;
            auto read_tile_id = (dim == 0) ? (i) : (get_read_tile_id(i, reduce_tile_size, inner_tile_size));
            // Now reduce all tiles in the reduction dim. The first index is the
            // same as the output index. After that need to increment by the
            // size of the inner dimensions in tiles. E.g. for 130 tiles
            // (where shard factor equals 1), the increment is 130. If 4 tiles
            // need to be reduced, then the first core would access tiles at
            // indices 0, 130, 260, 390, 64, 64+130, 64+260, 64+390, 128,
            // 128+130, 128+260, and 128+390.
            for (uint32_t j = 0; j < num_input_tiles; ++j) {
                if (input_granularity_index == 0) {
                    cb_in0_obj.reserve_back(input_granularity);
                    write_offset = 0;
                }
                noc.async_read(
                    tensor_accessor,
                    cb_in0_obj,
                    input_tile_bytes,
                    {.page_id = read_tile_id},
                    {.offset_bytes = write_offset});
                write_offset += input_tile_bytes;
                read_tile_id += inner_tile_size;
                input_granularity_index++;
                if (input_granularity_index == input_granularity) {
                    noc.async_read_barrier();
                    cb_in0_obj.push_back(input_granularity);
                    input_granularity_index = 0;
                }
            }
        }
    }
}
