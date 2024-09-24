// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

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

    constexpr auto cb_in0 = tt::CB::c_in0;
    constexpr auto cb_in1 = tt::CB::c_in1;
    constexpr auto cb_in2 = tt::CB::c_in2;
    constexpr auto cb_in3 = tt::CB::c_in3;
    constexpr auto cb_in4 = tt::CB::c_in4;
    constexpr auto cb_in5 = tt::CB::c_in5;

    constexpr bool in_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool index0_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool index1_is_dram = get_compile_time_arg_val(2) == 1;
    constexpr bool index2_is_dram = get_compile_time_arg_val(3) == 1;
    constexpr bool index3_is_dram = get_compile_time_arg_val(4) == 1;
    constexpr bool index4_is_dram = get_compile_time_arg_val(5) == 1;

    const InterleavedAddrGen<in_is_dram> s0 = {.bank_base_address = src_addr, .page_size = stick_size};

    const InterleavedAddrGen<index0_is_dram> index0 = {
        .bank_base_address = index0_addr, .page_size = index0_stick_size};
    const InterleavedAddrGen<index1_is_dram> index1 = {
        .bank_base_address = index1_addr, .page_size = index1_stick_size};
    const InterleavedAddrGen<index2_is_dram> index2 = {
        .bank_base_address = index2_addr, .page_size = index2_stick_size};
    const InterleavedAddrGen<index3_is_dram> index3 = {
        .bank_base_address = index3_addr, .page_size = index3_stick_size};
    const InterleavedAddrGen<index4_is_dram> index4 = {
        .bank_base_address = index4_addr, .page_size = index4_stick_size};

    uint32_t index_is_defined[5] = {
        index0_is_defined,
        index1_is_defined,
        index2_is_defined,
        index3_is_defined,
        index4_is_defined,
    };

    tt::CB index_cbs[5] = {
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

    uint32_t end_id = start_id + num_sticks;
    for (uint32_t i = start_id; i < end_id; ++i) {
        // compute src noc id
        uint32_t noc_id = 0;
        uint32_t output_stick_idx = i;
        uint32_t index_index = 0;
        bool is_first_index = true;
        int32_t output_dim = 3;
        for (int32_t dim = 3; dim >= 0; dim--) {

            uint32_t input_stick_idx_stride = input_stick_idx_strides[dim];
            auto output_size = output_size_list[output_dim];

            if (index_is_defined[dim]) {
                // read index tensor
                tt::CB idx_cb = index_cbs[dim];

                cb_reserve_back(idx_cb, 1);
                uint32_t index_l1_addr = get_write_ptr(idx_cb);
                uint64_t index_noc_addr;

                if (is_first_index) {
                    index_index = output_stick_idx % index_size;
                }

                if (dim == 0) {
                    index_noc_addr = get_noc_addr(0, index0);
                }
                if (dim == 1) {
                    index_noc_addr = get_noc_addr(0, index1);
                }
                if (dim == 2) {
                    index_noc_addr = get_noc_addr(0, index2);
                }
                if (dim == 3) {
                    index_noc_addr = get_noc_addr(0, index3);
                }
                noc_async_read(index_noc_addr, index_l1_addr, index_stick_sizes[dim]);
                noc_async_read_barrier();

                volatile tt_l1_ptr int32_t* index_l1_ptr =
                    reinterpret_cast<volatile tt_l1_ptr int32_t*>(index_l1_addr);
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
                output_dim --;
            }
        }

        cb_reserve_back(cb_in0, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_in0);
        uint64_t src_noc_addr = get_noc_addr(noc_id, s0);
        noc_async_read(src_noc_addr, l1_write_addr, stick_size);
        noc_async_read_barrier();
        cb_push_back(cb_in0, 1);
    }
}
