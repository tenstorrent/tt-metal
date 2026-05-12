// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    uint32_t i = 0;
    // buffers
    uint32_t src_addr = get_arg_val<uint32_t>(i++);
    uint32_t index0_addr = get_arg_val<uint32_t>(i++);
    uint32_t index1_addr = get_arg_val<uint32_t>(i++);
    uint32_t index2_addr = get_arg_val<uint32_t>(i++);
    uint32_t index3_addr = get_arg_val<uint32_t>(i++);
    uint32_t index4_addr = get_arg_val<uint32_t>(i++);

    // input
    uint32_t input_stick_idx_stride_n = get_arg_val<uint32_t>(i++);
    uint32_t input_stick_idx_stride_c = get_arg_val<uint32_t>(i++);
    uint32_t input_stick_idx_stride_d = get_arg_val<uint32_t>(i++);
    uint32_t input_stick_idx_stride_h = get_arg_val<uint32_t>(i++);

    uint32_t input_size_n = get_arg_val<uint32_t>(i++);
    uint32_t input_size_c = get_arg_val<uint32_t>(i++);
    uint32_t input_size_d = get_arg_val<uint32_t>(i++);
    uint32_t input_size_h = get_arg_val<uint32_t>(i++);
    uint32_t input_size_w = get_arg_val<uint32_t>(i++);

    // index
    uint32_t index0_is_defined = get_arg_val<uint32_t>(i++);
    uint32_t index1_is_defined = get_arg_val<uint32_t>(i++);
    uint32_t index2_is_defined = get_arg_val<uint32_t>(i++);
    uint32_t index3_is_defined = get_arg_val<uint32_t>(i++);
    uint32_t index4_is_defined = get_arg_val<uint32_t>(i++);
    uint32_t index0_stick_size = get_arg_val<uint32_t>(i++);
    uint32_t index1_stick_size = get_arg_val<uint32_t>(i++);
    uint32_t index2_stick_size = get_arg_val<uint32_t>(i++);
    uint32_t index3_stick_size = get_arg_val<uint32_t>(i++);
    uint32_t index4_stick_size = get_arg_val<uint32_t>(i++);
    uint32_t index_size = get_arg_val<uint32_t>(i++);
    int32_t index_start_dim = static_cast<int32_t>(get_arg_val<uint32_t>(i++));
    int32_t index_end_dim = static_cast<int32_t>(get_arg_val<uint32_t>(i++));

    // output
    uint32_t output_size_n = get_arg_val<uint32_t>(i++);
    uint32_t output_size_c = get_arg_val<uint32_t>(i++);
    uint32_t output_size_d = get_arg_val<uint32_t>(i++);
    uint32_t output_size_h = get_arg_val<uint32_t>(i++);
    uint32_t output_size_w = get_arg_val<uint32_t>(i++);

    // etc
    uint32_t start_id = get_arg_val<uint32_t>(i++);
    uint32_t num_sticks = get_arg_val<uint32_t>(i++);
    uint32_t stick_size = get_arg_val<uint32_t>(i++);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_in2 = tt::CBIndex::c_2;
    constexpr auto cb_in3 = tt::CBIndex::c_3;
    constexpr auto cb_in4 = tt::CBIndex::c_4;
    constexpr auto cb_in5 = tt::CBIndex::c_5;

    constexpr auto in_args = TensorAccessorArgs<0>();
    constexpr auto index0_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr auto index1_args = TensorAccessorArgs<index0_args.next_compile_time_args_offset()>();
    constexpr auto index2_args = TensorAccessorArgs<index1_args.next_compile_time_args_offset()>();
    constexpr auto index3_args = TensorAccessorArgs<index2_args.next_compile_time_args_offset()>();
    constexpr auto index4_args = TensorAccessorArgs<index3_args.next_compile_time_args_offset()>();

    // Third argument page_size from runtime args overrides TensorAccessorArgs::AlignedPageSize, which may be stale on
    // program cache hits.
    const auto s0 = TensorAccessor(in_args, src_addr, stick_size);

    // Third argument page_size from runtime args overrides TensorAccessorArgs::AlignedPageSize, which may be stale on
    // program cache hits.
    const auto index0 = TensorAccessor(index0_args, index0_addr, index0_stick_size);
    const auto index1 = TensorAccessor(index1_args, index1_addr, index1_stick_size);
    const auto index2 = TensorAccessor(index2_args, index2_addr, index2_stick_size);
    const auto index3 = TensorAccessor(index3_args, index3_addr, index3_stick_size);
    const auto index4 = TensorAccessor(index4_args, index4_addr, index4_stick_size);

    uint32_t index_is_defined[5] = {
        index0_is_defined,
        index1_is_defined,
        index2_is_defined,
        index3_is_defined,
        index4_is_defined,
    };

    tt::CBIndex index_cbs[5] = {
        cb_in1,
        cb_in2,
        cb_in3,
        cb_in4,
        cb_in5,
    };

    uint32_t input_size_list[5] = {
        input_size_n,
        input_size_c,
        input_size_d,
        input_size_h,
        input_size_w,
    };

    uint32_t output_size_list[5] = {
        output_size_n,
        output_size_c,
        output_size_d,
        output_size_h,
        output_size_w,
    };

    uint32_t input_stick_idx_strides[4] = {
        input_stick_idx_stride_n,
        input_stick_idx_stride_c,
        input_stick_idx_stride_d,
        input_stick_idx_stride_h,
    };

    uint32_t index_stick_sizes[5] = {
        index0_stick_size,
        index1_stick_size,
        index2_stick_size,
        index3_stick_size,
        index4_stick_size,
    };

    experimental::Noc noc;
    experimental::CircularBuffer cb_in0_obj(cb_in0);
    experimental::CircularBuffer cb_in1_obj(cb_in1);
    experimental::CircularBuffer cb_in2_obj(cb_in2);
    experimental::CircularBuffer cb_in3_obj(cb_in3);
    experimental::CircularBuffer cb_in4_obj(cb_in4);

    uint32_t end_id = start_id + num_sticks;
    for (uint32_t i = start_id; i < end_id; ++i) {
        uint32_t noc_id = 0;
        uint32_t output_stick_idx = i;
        uint32_t index_index = 0;
        bool is_first_index = true;
        int32_t output_dim = 3;
        for (int32_t dim = 3; dim >= 0; dim--) {
            uint32_t input_stick_idx_stride = input_stick_idx_strides[dim];
            auto output_size = output_size_list[output_dim];

            if (index_is_defined[dim]) {
                tt::CBIndex idx_cb = index_cbs[dim];

                if (is_first_index) {
                    index_index = output_stick_idx % index_size;
                }

                uint32_t index_l1_addr = 0;
                if (dim == 0) {
                    cb_in1_obj.reserve_back(1);
                    index_l1_addr = cb_in1_obj.get_write_ptr();
                    noc.async_read(index0, cb_in1_obj, index_stick_sizes[dim], {.page_id = 0}, {.offset_bytes = 0});
                }
                if (dim == 1) {
                    cb_in2_obj.reserve_back(1);
                    index_l1_addr = cb_in2_obj.get_write_ptr();
                    noc.async_read(index1, cb_in2_obj, index_stick_sizes[dim], {.page_id = 0}, {.offset_bytes = 0});
                }
                if (dim == 2) {
                    cb_in3_obj.reserve_back(1);
                    index_l1_addr = cb_in3_obj.get_write_ptr();
                    noc.async_read(index2, cb_in3_obj, index_stick_sizes[dim], {.page_id = 0}, {.offset_bytes = 0});
                }
                if (dim == 3) {
                    cb_in4_obj.reserve_back(1);
                    index_l1_addr = cb_in4_obj.get_write_ptr();
                    noc.async_read(index3, cb_in4_obj, index_stick_sizes[dim], {.page_id = 0}, {.offset_bytes = 0});
                }
                noc.async_read_barrier();

                volatile tt_l1_ptr int32_t* index_l1_ptr = reinterpret_cast<volatile tt_l1_ptr int32_t*>(index_l1_addr);
                int32_t noc_idx = index_l1_ptr[index_index];

                if (noc_idx < 0) {
                    noc_idx += input_size_list[dim];
                }

                noc_id += noc_idx * input_stick_idx_stride;
                if (is_first_index) {
                    output_stick_idx /= output_size;
                }
                is_first_index = false;
            } else {
                uint32_t noc_idx = output_stick_idx % output_size;
                noc_id += noc_idx * input_stick_idx_stride;
                output_stick_idx /= output_size;
            }
            if (!(index_start_dim < dim && dim <= index_end_dim)) {
                output_dim--;
            }
        }

        cb_in0_obj.reserve_back(1);
        noc.async_read(s0, cb_in0_obj, stick_size, {.page_id = noc_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_in0_obj.push_back(1);
    }
}
