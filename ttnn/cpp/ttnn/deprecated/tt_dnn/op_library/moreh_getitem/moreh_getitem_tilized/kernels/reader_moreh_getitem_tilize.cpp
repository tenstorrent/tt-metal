// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_getitem/moreh_getitem_tilized/kernels/common.hpp"

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
    uint32_t input_stick_idx_stride_w = get_arg_val<uint32_t>(i++);
    uint32_t input_size_c_without_padding = get_arg_val<uint32_t>(i++);
    uint32_t input_size_d_without_padding = get_arg_val<uint32_t>(i++);
    uint32_t input_size_h_without_padding = get_arg_val<uint32_t>(i++);
    uint32_t input_noc_id_stride_n = get_arg_val<uint32_t>(i++);
    uint32_t input_noc_id_stride_c = get_arg_val<uint32_t>(i++);
    uint32_t input_noc_id_stride_d = get_arg_val<uint32_t>(i++);
    uint32_t input_noc_id_stride_h = get_arg_val<uint32_t>(i++);
    uint32_t input_num_stick_width = get_arg_val<uint32_t>(i++);

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

    // output
    uint32_t output_size_n = get_arg_val<uint32_t>(i++);
    uint32_t output_size_c = get_arg_val<uint32_t>(i++);
    uint32_t output_size_d = get_arg_val<uint32_t>(i++);
    uint32_t output_size_h = get_arg_val<uint32_t>(i++);
    uint32_t output_size_w = get_arg_val<uint32_t>(i++);
    uint32_t output_num_stick_width = get_arg_val<uint32_t>(i++);

    // etc
    uint32_t start_id = get_arg_val<uint32_t>(i++);
    uint32_t num_sticks = get_arg_val<uint32_t>(i++);
    uint32_t stick_size = get_arg_val<uint32_t>(i++);
    uint32_t element_size = get_arg_val<uint32_t>(i++);

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

    const InterleavedAddrGen<in_is_dram> s0 = {
        .bank_base_address = src_addr,
        .page_size = 1024 * element_size,
    };

    const InterleavedAddrGen<index0_is_dram> index0 = {
        .bank_base_address = index0_addr, .page_size = INDEX_TILE_SIZE};
    const InterleavedAddrGen<index1_is_dram> index1 = {
        .bank_base_address = index1_addr, .page_size = INDEX_TILE_SIZE};
    const InterleavedAddrGen<index2_is_dram> index2 = {
        .bank_base_address = index2_addr, .page_size = INDEX_TILE_SIZE};
    const InterleavedAddrGen<index3_is_dram> index3 = {
        .bank_base_address = index3_addr, .page_size = INDEX_TILE_SIZE};
    const InterleavedAddrGen<index4_is_dram> index4 = {
        .bank_base_address = index4_addr, .page_size = INDEX_TILE_SIZE};

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

    uint32_t input_stick_idx_strides[5] = {
        input_stick_idx_stride_n,
        input_stick_idx_stride_c,
        input_stick_idx_stride_d,
        input_stick_idx_stride_h,
        input_stick_idx_stride_w,
    };

    #define NOC_MINIMUM_READ_SIZE (32)
    #define INDEX_SIZE (4)

    uint32_t end_id = start_id + num_sticks;

    for (uint32_t i = start_id; i < end_id; ++i) {
        // compute src noc id
        uint32_t output_stick_idx = i;
        uint32_t input_stick_idx = 0;
        uint32_t index_index = 0;
        bool is_first_index = true;

        for (int32_t dim = 4; dim >= 0; dim--) {
            uint32_t input_stick_idx_stride = input_stick_idx_strides[dim];

            if (index_is_defined[dim]) {
                tt::CB idx_cb = index_cbs[dim];

                cb_reserve_back(idx_cb, 1);
                uint32_t index_l1_addr = get_write_ptr(idx_cb);
                uint64_t index_noc_addr;

                if (is_first_index) {
                    index_index = output_stick_idx % index_size;
                    is_first_index = false;
                }
                #ifdef TILIZE_INDEX
                uint32_t index_noc_id = index_index / TILE_HEIGHT;
                if (dim == 0) {
                    index_noc_addr = get_noc_addr(index_noc_id, index0);
                }
                if (dim == 1) {
                    index_noc_addr = get_noc_addr(index_noc_id, index1);
                }
                if (dim == 2) {
                    index_noc_addr = get_noc_addr(index_noc_id, index2);
                }
                if (dim == 3) {
                    index_noc_addr = get_noc_addr(index_noc_id, index3);
                }
                noc_async_read(index_noc_addr, index_l1_addr, INDEX_TILE_SIZE);
                noc_async_read_barrier();

                volatile tt_l1_ptr int32_t* index_l1_ptr =
                    reinterpret_cast<volatile tt_l1_ptr int32_t*>(index_l1_addr);
                uint32_t index_dim_offset;
                uint32_t index_tile_idx = index_index % TILE_WIDTH;
                if (index_tile_idx < FACE_WIDTH) index_dim_offset = index_tile_idx;
                else index_dim_offset = index_tile_idx + 256 - 16;

                int32_t index_val = index_l1_ptr[index_dim_offset];
                #endif
                #ifdef ROW_MAJOR_INDEX
                uint32_t noc_offset = ((uint32_t)((index_index * INDEX_SIZE) / NOC_MINIMUM_READ_SIZE)) * NOC_MINIMUM_READ_SIZE;
                if (dim == 0) {
                    index_noc_addr = get_noc_addr(0, index0, noc_offset);
                }
                if (dim == 1) {
                    index_noc_addr = get_noc_addr(0, index1, noc_offset);
                }
                if (dim == 2) {
                    index_noc_addr = get_noc_addr(0, index2, noc_offset);
                }
                if (dim == 3) {
                    index_noc_addr = get_noc_addr(0, index3, noc_offset);
                }
                noc_async_read(index_noc_addr, index_l1_addr, NOC_MINIMUM_READ_SIZE);
                noc_async_read_barrier();

                volatile tt_l1_ptr int32_t* index_l1_ptr =
                    reinterpret_cast<volatile tt_l1_ptr int32_t*>(index_l1_addr);

                uint32_t index_dim_offset = (index_index * INDEX_SIZE - noc_offset) / INDEX_SIZE;
                int32_t index_val = index_l1_ptr[index_dim_offset];

                #endif

                if (index_val < 0) {
                    index_val += input_size_list[dim];
                }

                input_stick_idx += index_val * input_stick_idx_stride;
            } else {
                uint32_t index_val;

                if (dim == 4) {
                    index_val = output_stick_idx % input_num_stick_width;
                    input_stick_idx += index_val * input_stick_idx_stride;
                } else {
                    auto output_size = output_size_list[dim];
                    index_val = output_stick_idx % output_size;
                    input_stick_idx += index_val * input_stick_idx_stride;
                }
            }
            if (dim == 4) {
                output_stick_idx /= output_num_stick_width;
            } else {
                auto output_size = output_size_list[dim];
                output_stick_idx /= output_size;
            }
        }

        // input_stick_idx = 5;
        cb_reserve_back(cb_in0, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_in0);

        Idx5d stick_index_5d = get_stick_indices(input_stick_idx, input_size_c_without_padding, input_size_d_without_padding, input_size_h_without_padding, input_num_stick_width);
        Idx5d tile_index_5d = get_tile_indices(stick_index_5d);

        uint32_t noc_id =   tile_index_5d.n * input_noc_id_stride_n +
                            tile_index_5d.c * input_noc_id_stride_c +
                            tile_index_5d.d * input_noc_id_stride_d +
                            tile_index_5d.h * input_noc_id_stride_h +
                            tile_index_5d.w;

        uint32_t noc_offset = get_noc_offset_in_tile(stick_index_5d.h , stick_index_5d.w, tile_index_5d.h, element_size);

        uint64_t src_noc_addr = get_noc_addr(noc_id, s0, noc_offset);

        noc_async_read(src_noc_addr, l1_write_addr, stick_size);
        noc_async_read_barrier();
        cb_push_back(cb_in0, 1);
    }
}
