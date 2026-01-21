// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t compute_output_cb_id = get_compile_time_arg_val(0);
constexpr uint32_t page_size = get_compile_time_arg_val(1);
constexpr uint32_t num_cores_to_be_used = get_compile_time_arg_val(2);
constexpr uint32_t input_tensor_Wt = get_compile_time_arg_val(3);
constexpr uint32_t slice_Wt = get_compile_time_arg_val(4);

constexpr uint32_t initial_ct_idx = 5;

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

    const uint32_t start_tiles_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_slice_row_offset = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_pages_read_in_row = get_arg_val<uint32_t>(arg_idx++);

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

    // hardcoded constants
    constexpr uint32_t num_split_tensors = 8;
    constexpr uint32_t one_tile = 1;

    uint32_t slice_row_offset = start_slice_row_offset;
    uint32_t pages_read_in_row = start_pages_read_in_row;

    uint32_t tiles_read = start_tiles_read;
    uint32_t tiles_to_read = start_tiles_to_read;
    while (tiles_read < tiles_to_read) {
        uint32_t slice_id = pages_read_in_row / slice_Wt;
        uint32_t slice_offset = pages_read_in_row - (slice_id * slice_Wt);
        uint32_t normalized_page_id = slice_row_offset + slice_offset;

        cb_wait_front(compute_output_cb_id, one_tile);
        uint32_t l1_read_addr = get_read_ptr(compute_output_cb_id);
        switch (slice_id) {
            case 0: noc_async_write_page(normalized_page_id, output_slice_0_tensor_accesor, l1_read_addr); break;
            case 1: noc_async_write_page(normalized_page_id, output_slice_1_tensor_accesor, l1_read_addr); break;
            case 2: noc_async_write_page(normalized_page_id, output_slice_2_tensor_accesor, l1_read_addr); break;
            case 3: noc_async_write_page(normalized_page_id, output_slice_3_tensor_accesor, l1_read_addr); break;
            case 4: noc_async_write_page(normalized_page_id, output_slice_4_tensor_accesor, l1_read_addr); break;
            case 5: noc_async_write_page(normalized_page_id, output_slice_5_tensor_accesor, l1_read_addr); break;
            case 6: noc_async_write_page(normalized_page_id, output_slice_6_tensor_accesor, l1_read_addr); break;
            case 7: noc_async_write_page(normalized_page_id, output_slice_7_tensor_accesor, l1_read_addr); break;
        }
        noc_async_writes_flushed();
        cb_pop_front(compute_output_cb_id, one_tile);

        tiles_read += num_cores_to_be_used;
        pages_read_in_row += num_cores_to_be_used;
        while (pages_read_in_row >= input_tensor_Wt) {
            pages_read_in_row -= input_tensor_Wt;
            slice_row_offset += slice_Wt;
        }
    }
    noc_async_write_barrier();
}
