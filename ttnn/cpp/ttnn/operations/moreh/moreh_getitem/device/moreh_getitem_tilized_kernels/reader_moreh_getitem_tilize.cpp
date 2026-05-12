// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/moreh/moreh_getitem/device/moreh_getitem_tilized_kernels/common.hpp"
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

    const auto s0 = TensorAccessor(in_args, src_addr);

    const auto index0 = TensorAccessor(index0_args, index0_addr);
    const auto index1 = TensorAccessor(index1_args, index1_addr);
    const auto index2 = TensorAccessor(index2_args, index2_addr);
    const auto index3 = TensorAccessor(index3_args, index3_addr);
    const auto index4 = TensorAccessor(index4_args, index4_addr);

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

    uint32_t input_stick_idx_strides[5] = {
        input_stick_idx_stride_n,
        input_stick_idx_stride_c,
        input_stick_idx_stride_d,
        input_stick_idx_stride_h,
        input_stick_idx_stride_w,
    };

#define NOC_MINIMUM_READ_SIZE (32)
#define INDEX_SIZE (4)

    experimental::Noc noc;
    experimental::CircularBuffer cb_in0_obj(cb_in0);
    experimental::CircularBuffer cb_in1_obj(cb_in1);
    experimental::CircularBuffer cb_in2_obj(cb_in2);
    experimental::CircularBuffer cb_in3_obj(cb_in3);
    experimental::CircularBuffer cb_in4_obj(cb_in4);

    uint32_t end_id = start_id + num_sticks;

    for (uint32_t i = start_id; i < end_id; ++i) {
        uint32_t output_stick_idx = i;
        uint32_t input_stick_idx = 0;
        uint32_t index_index = 0;
        bool is_first_index = true;

        for (int32_t dim = 4; dim >= 0; dim--) {
            uint32_t input_stick_idx_stride = input_stick_idx_strides[dim];

            if (index_is_defined[dim]) {
                uint32_t index_l1_addr = 0;

                if (is_first_index) {
                    index_index = output_stick_idx % index_size;
                    is_first_index = false;
                }
#ifdef TILIZE_INDEX
                uint32_t index_noc_id = index_index / TILE_HEIGHT;
                if (dim == 0) {
                    cb_in1_obj.reserve_back(1);
                    index_l1_addr = cb_in1_obj.get_write_ptr();
                    noc.async_read(index0, cb_in1_obj, INDEX_TILE_SIZE, {.page_id = index_noc_id}, {.offset_bytes = 0});
                }
                if (dim == 1) {
                    cb_in2_obj.reserve_back(1);
                    index_l1_addr = cb_in2_obj.get_write_ptr();
                    noc.async_read(index1, cb_in2_obj, INDEX_TILE_SIZE, {.page_id = index_noc_id}, {.offset_bytes = 0});
                }
                if (dim == 2) {
                    cb_in3_obj.reserve_back(1);
                    index_l1_addr = cb_in3_obj.get_write_ptr();
                    noc.async_read(index2, cb_in3_obj, INDEX_TILE_SIZE, {.page_id = index_noc_id}, {.offset_bytes = 0});
                }
                if (dim == 3) {
                    cb_in4_obj.reserve_back(1);
                    index_l1_addr = cb_in4_obj.get_write_ptr();
                    noc.async_read(index3, cb_in4_obj, INDEX_TILE_SIZE, {.page_id = index_noc_id}, {.offset_bytes = 0});
                }
                noc.async_read_barrier();

                volatile tt_l1_ptr int32_t* index_l1_ptr = reinterpret_cast<volatile tt_l1_ptr int32_t*>(index_l1_addr);
                uint32_t index_dim_offset;
                uint32_t index_tile_idx = index_index % TILE_WIDTH;
                if (index_tile_idx < FACE_WIDTH) {
                    index_dim_offset = index_tile_idx;
                } else {
                    index_dim_offset = index_tile_idx + 256 - 16;
                }

                int32_t index_val = index_l1_ptr[index_dim_offset];
#endif
#ifdef ROW_MAJOR_INDEX
                uint32_t noc_offset =
                    ((uint32_t)((index_index * INDEX_SIZE) / NOC_MINIMUM_READ_SIZE)) * NOC_MINIMUM_READ_SIZE;
                if (dim == 0) {
                    cb_in1_obj.reserve_back(1);
                    index_l1_addr = cb_in1_obj.get_write_ptr();
                    noc.async_read(
                        index0,
                        cb_in1_obj,
                        NOC_MINIMUM_READ_SIZE,
                        {.page_id = 0, .offset_bytes = noc_offset},
                        {.offset_bytes = 0});
                }
                if (dim == 1) {
                    cb_in2_obj.reserve_back(1);
                    index_l1_addr = cb_in2_obj.get_write_ptr();
                    noc.async_read(
                        index1,
                        cb_in2_obj,
                        NOC_MINIMUM_READ_SIZE,
                        {.page_id = 0, .offset_bytes = noc_offset},
                        {.offset_bytes = 0});
                }
                if (dim == 2) {
                    cb_in3_obj.reserve_back(1);
                    index_l1_addr = cb_in3_obj.get_write_ptr();
                    noc.async_read(
                        index2,
                        cb_in3_obj,
                        NOC_MINIMUM_READ_SIZE,
                        {.page_id = 0, .offset_bytes = noc_offset},
                        {.offset_bytes = 0});
                }
                if (dim == 3) {
                    cb_in4_obj.reserve_back(1);
                    index_l1_addr = cb_in4_obj.get_write_ptr();
                    noc.async_read(
                        index3,
                        cb_in4_obj,
                        NOC_MINIMUM_READ_SIZE,
                        {.page_id = 0, .offset_bytes = noc_offset},
                        {.offset_bytes = 0});
                }
                noc.async_read_barrier();

                volatile tt_l1_ptr int32_t* index_l1_ptr = reinterpret_cast<volatile tt_l1_ptr int32_t*>(index_l1_addr);

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

        cb_in0_obj.reserve_back(1);

        Idx5d stick_index_5d = get_stick_indices(
            input_stick_idx,
            input_size_c_without_padding,
            input_size_d_without_padding,
            input_size_h_without_padding,
            input_num_stick_width);
        Idx5d tile_index_5d = get_tile_indices(stick_index_5d);

        uint32_t noc_id = tile_index_5d.n * input_noc_id_stride_n + tile_index_5d.c * input_noc_id_stride_c +
                          tile_index_5d.d * input_noc_id_stride_d + tile_index_5d.h * input_noc_id_stride_h +
                          tile_index_5d.w;

        uint32_t noc_offset = get_noc_offset_in_tile(stick_index_5d.h, stick_index_5d.w, tile_index_5d.h, element_size);

        noc.async_read(
            s0, cb_in0_obj, stick_size, {.page_id = noc_id, .offset_bytes = noc_offset}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_in0_obj.push_back(1);
    }
}
