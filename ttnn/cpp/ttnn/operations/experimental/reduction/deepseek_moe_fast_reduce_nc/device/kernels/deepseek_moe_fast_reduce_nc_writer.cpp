// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t page_size = get_compile_time_arg_val(0);
constexpr uint32_t num_cores_to_be_used = get_compile_time_arg_val(1);
constexpr uint32_t compute_output_cb_id = get_compile_time_arg_val(2);
constexpr uint32_t input_tensor_Wt = get_compile_time_arg_val(3);

constexpr uint32_t initial_ct_idx = 4;

void kernel_main() {
    uint32_t arg_idx = 0;

    uint32_t output_slice_0_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_slice_1_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_slice_2_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_slice_3_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_slice_4_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_slice_5_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_slice_6_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_slice_7_address = get_arg_val<uint32_t>(arg_idx++);

    const uint32_t id_range_length = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_id = get_arg_val<uint32_t>(arg_idx++);

    // TensorAccessors
    constexpr uint32_t output_slice_0_ct_val = initial_ct_idx;
    constexpr auto output_slice_0_tensor_args = TensorAccessorArgs<output_slice_0_ct_val>();
    constexpr uint32_t output_slice_0_ct_offset = output_slice_0_tensor_args.num_compile_time_args();
    const auto output_slice_0_tensor_accesor =
        TensorAccessor(output_slice_0_tensor_args, output_slice_0_address, page_size);

    constexpr uint32_t output_slice_1_ct_val = output_slice_0_ct_val + output_slice_0_ct_offset;
    constexpr auto output_slice_1_tensor_args = TensorAccessorArgs<output_slice_1_ct_val>();
    constexpr uint32_t output_slice_1_ct_offset = output_slice_1_tensor_args.num_compile_time_args();
    const auto output_slice_1_tensor_accesor =
        TensorAccessor(output_slice_1_tensor_args, output_slice_1_address, page_size);

    constexpr uint32_t output_slice_2_ct_val = output_slice_1_ct_val + output_slice_1_ct_offset;
    constexpr auto output_slice_2_tensor_args = TensorAccessorArgs<output_slice_2_ct_val>();
    constexpr uint32_t output_slice_2_ct_offset = output_slice_2_tensor_args.num_compile_time_args();
    const auto output_slice_2_tensor_accesor =
        TensorAccessor(output_slice_2_tensor_args, output_slice_2_address, page_size);

    constexpr uint32_t output_slice_3_ct_val = output_slice_2_ct_val + output_slice_2_ct_offset;
    constexpr auto output_slice_3_tensor_args = TensorAccessorArgs<output_slice_3_ct_val>();
    constexpr uint32_t output_slice_3_ct_offset = output_slice_3_tensor_args.num_compile_time_args();
    const auto output_slice_3_tensor_accesor =
        TensorAccessor(output_slice_3_tensor_args, output_slice_3_address, page_size);

    constexpr uint32_t output_slice_4_ct_val = output_slice_3_ct_val + output_slice_3_ct_offset;
    constexpr auto output_slice_4_tensor_args = TensorAccessorArgs<output_slice_4_ct_val>();
    constexpr uint32_t output_slice_4_ct_offset = output_slice_4_tensor_args.num_compile_time_args();
    const auto output_slice_4_tensor_accesor =
        TensorAccessor(output_slice_4_tensor_args, output_slice_4_address, page_size);

    constexpr uint32_t output_slice_5_ct_val = output_slice_4_ct_val + output_slice_4_ct_offset;
    constexpr auto output_slice_5_tensor_args = TensorAccessorArgs<output_slice_5_ct_val>();
    constexpr uint32_t output_slice_5_ct_offset = output_slice_5_tensor_args.num_compile_time_args();
    const auto output_slice_5_tensor_accesor =
        TensorAccessor(output_slice_5_tensor_args, output_slice_5_address, page_size);

    constexpr uint32_t output_slice_6_ct_val = output_slice_5_ct_val + output_slice_5_ct_offset;
    constexpr auto output_slice_6_tensor_args = TensorAccessorArgs<output_slice_6_ct_val>();
    constexpr uint32_t output_slice_6_ct_offset = output_slice_6_tensor_args.num_compile_time_args();
    const auto output_slice_6_tensor_accesor =
        TensorAccessor(output_slice_6_tensor_args, output_slice_6_address, page_size);

    constexpr uint32_t output_slice_7_ct_val = output_slice_6_ct_val + output_slice_6_ct_offset;
    constexpr auto output_slice_7_tensor_args = TensorAccessorArgs<output_slice_7_ct_val>();
    constexpr uint32_t output_slice_7_ct_offset = output_slice_7_tensor_args.num_compile_time_args();
    const auto output_slice_7_tensor_accesor =
        TensorAccessor(output_slice_7_tensor_args, output_slice_7_address, page_size);

    constexpr uint32_t one_tile = 1;

    constexpr uint32_t num_split_tensors = 8;
    constexpr uint32_t slice_Wt = input_tensor_Wt / num_split_tensors;

    // For each shard, start at the index of the first shard to be reduced (same
    // index as output), then increment by the appropriate increment (based on
    // the grid size), until the range length is reached. See reader and program
    // factory for examples.
    for (uint32_t page_id = start_id; page_id < start_id + id_range_length; page_id += num_cores_to_be_used) {
        uint32_t num_rows_processed = page_id / input_tensor_Wt;
        uint32_t page_id_within_row = page_id - (num_rows_processed * input_tensor_Wt);
        uint32_t slice_id = page_id_within_row / slice_Wt;

        uint32_t normalized_page_id = (num_rows_processed * slice_Wt) + (page_id_within_row % slice_Wt);

        uint64_t noc_addr;
        switch (slice_id) {
            case 0: noc_addr = get_noc_addr(normalized_page_id, output_slice_0_tensor_accesor); break;
            case 1: noc_addr = get_noc_addr(normalized_page_id, output_slice_1_tensor_accesor); break;
            case 2: noc_addr = get_noc_addr(normalized_page_id, output_slice_2_tensor_accesor); break;
            case 3: noc_addr = get_noc_addr(normalized_page_id, output_slice_3_tensor_accesor); break;
            case 4: noc_addr = get_noc_addr(normalized_page_id, output_slice_4_tensor_accesor); break;
            case 5: noc_addr = get_noc_addr(normalized_page_id, output_slice_5_tensor_accesor); break;
            case 6: noc_addr = get_noc_addr(normalized_page_id, output_slice_6_tensor_accesor); break;
            case 7: noc_addr = get_noc_addr(normalized_page_id, output_slice_7_tensor_accesor); break;
        }

        cb_wait_front(compute_output_cb_id, one_tile);
        uint32_t l1_read_addr = get_read_ptr(compute_output_cb_id);
        noc_async_write(l1_read_addr, noc_addr, page_size);
        noc_async_writes_flushed();
        cb_pop_front(compute_output_cb_id, one_tile);
    }
    noc_async_write_barrier();
}
