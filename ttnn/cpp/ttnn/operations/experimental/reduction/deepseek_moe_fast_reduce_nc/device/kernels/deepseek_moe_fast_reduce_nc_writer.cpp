// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t compute_output_cb_id = get_compile_time_arg_val(0);
constexpr uint32_t page_size = get_compile_time_arg_val(1);
constexpr uint32_t num_cores_to_be_used = get_compile_time_arg_val(2);
constexpr uint32_t input_tensor_Wt = get_compile_time_arg_val(3);
constexpr uint32_t slice_Wt = get_compile_time_arg_val(4);
constexpr uint32_t num_output_tensors = get_compile_time_arg_val(5);

constexpr uint32_t initial_ct_idx = 6;

void kernel_main() {
    uint32_t arg_idx = 0;

    const uint32_t start_tiles_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_slice_row_offset = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_pages_read_in_row = get_arg_val<uint32_t>(arg_idx++);

    constexpr uint32_t one_tile = 1;

    // TensorAccessors
    constexpr auto output_slice_tensor_accessor_args_tuple =
        make_tensor_accessor_args_tuple<num_output_tensors, initial_ct_idx + num_output_tensors>();
    const auto output_slice_tensor_accessor_tuple =
        make_tensor_accessor_tuple(output_slice_tensor_accessor_args_tuple, arg_idx, initial_ct_idx);
    const auto output_slice_tensor_accessors =
        make_abstract_tensor_accessor_wrappers(output_slice_tensor_accessor_tuple);
    arg_idx += num_output_tensors;

    uint32_t slice_row_offset = start_slice_row_offset;
    uint32_t pages_read_in_row = start_pages_read_in_row;

    uint32_t slice_id = pages_read_in_row / slice_Wt;
    uint32_t intra_slice_offset = pages_read_in_row % slice_Wt;

    uint32_t tiles_read = start_tiles_read;
    uint32_t tiles_to_read = start_tiles_to_read;
    while (tiles_read < tiles_to_read) {
        uint32_t normalized_page_id = slice_row_offset + intra_slice_offset;
        uint64_t noc_addr = output_slice_tensor_accessors[slice_id].get_noc_addr(normalized_page_id);

        cb_wait_front(compute_output_cb_id, one_tile);
        uint32_t l1_read_addr = get_read_ptr(compute_output_cb_id);
        noc_async_write(l1_read_addr, noc_addr, page_size);
        noc_async_writes_flushed();
        cb_pop_front(compute_output_cb_id, one_tile);

        tiles_read += num_cores_to_be_used;

        intra_slice_offset += num_cores_to_be_used;
        while (intra_slice_offset >= slice_Wt) {
            intra_slice_offset -= slice_Wt;
            slice_id++;
            if (slice_id == num_output_tensors) {
                slice_id = 0;
            }
        }

        pages_read_in_row += num_cores_to_be_used;
        while (pages_read_in_row >= input_tensor_Wt) {
            pages_read_in_row -= input_tensor_Wt;
            slice_row_offset += slice_Wt;
        }
    }
    noc_async_write_barrier();
}
