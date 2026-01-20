// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t page_size = get_compile_time_arg_val(0);
constexpr uint32_t num_cores_to_be_used = get_compile_time_arg_val(1);
constexpr uint32_t compute_output_cb_id = get_compile_time_arg_val(2);

constexpr uint32_t initial_ct_idx = 3;

void kernel_main() {
    uint32_t arg_idx = 0;

    uint32_t output_address = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t id_range_length = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_id = get_arg_val<uint32_t>(arg_idx++);

    constexpr auto tensor_args = TensorAccessorArgs<initial_ct_idx>();
    auto tensor_accessor = TensorAccessor(tensor_args, output_address, page_size);

    constexpr uint32_t one_tile = 1;

    // For each shard, start at the index of the first shard to be reduced (same
    // index as output), then increment by the appropriate increment (based on
    // the grid size), until the range length is reached. See reader and program
    // factory for examples.
    for (uint32_t outer_id = start_id; outer_id < start_id + id_range_length; outer_id += num_cores_to_be_used) {
        uint64_t noc_addr = get_noc_addr(outer_id, tensor_accessor);

        cb_wait_front(compute_output_cb_id, one_tile);
        uint32_t l1_read_addr = get_read_ptr(compute_output_cb_id);
        noc_async_write(l1_read_addr, noc_addr, page_size);
        noc_async_writes_flushed();
        cb_pop_front(compute_output_cb_id, one_tile);
    }
    noc_async_write_barrier();
}
