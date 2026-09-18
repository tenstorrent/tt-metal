// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // compile-time args
    constexpr auto shard_factor = get_arg(args::shard_factor);
    constexpr auto num_cores_to_be_used = get_arg(args::num_cores_to_be_used);
    constexpr uint32_t outer_id_increment = shard_factor * num_cores_to_be_used;

    // runtime args
    const auto id_range_length = get_arg(args::id_range_length);
    const auto start_id = get_arg(args::start_id);

    constexpr uint32_t onetile = 1;

    Noc noc;
    DataflowBuffer cb_out_obj(dfb::out0);

    uint32_t output_tile_bytes = cb_out_obj.get_entry_size();

    auto tensor_accessor = TensorAccessor(tensor::dst);

    // For each shard, start at the index of the first shard to be reduced (same
    // index as output), then increment by the appropriate increment (based on
    // the grid size), until the range length is reached. See reader and program
    // factory for examples.
    for (uint32_t outer_id = start_id; outer_id < start_id + id_range_length; outer_id += outer_id_increment) {
        for (uint32_t id_offset = 0; id_offset < shard_factor; id_offset++) {
            uint32_t i = outer_id + id_offset;
            uint32_t write_tile_id = i;
            cb_out_obj.wait_front(onetile);
            noc.async_write(
                cb_out_obj, tensor_accessor, output_tile_bytes, {.offset_bytes = 0}, {.page_id = write_tile_id});
            noc.async_write_barrier();
            cb_out_obj.pop_front(onetile);
        }
    }
}
