// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/moreh/moreh_getitem/device/moreh_getitem_tilized_kernels/common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // input
    uint32_t input_stick_idx_stride_n = get_arg(args::input_stick_idx_stride_n);
    uint32_t input_stick_idx_stride_c = get_arg(args::input_stick_idx_stride_c);
    uint32_t input_stick_idx_stride_d = get_arg(args::input_stick_idx_stride_d);
    uint32_t input_stick_idx_stride_h = get_arg(args::input_stick_idx_stride_h);
    uint32_t input_stick_idx_stride_w = get_arg(args::input_stick_idx_stride_w);
    uint32_t input_size_c_without_padding = get_arg(args::input_size_c_without_padding);
    uint32_t input_size_d_without_padding = get_arg(args::input_size_d_without_padding);
    uint32_t input_size_h_without_padding = get_arg(args::input_size_h_without_padding);
    uint32_t input_num_stick_width = get_arg(args::input_num_stick_width);
    uint32_t input_noc_id_stride_n = get_arg(args::input_noc_id_stride_n);
    uint32_t input_noc_id_stride_c = get_arg(args::input_noc_id_stride_c);
    uint32_t input_noc_id_stride_d = get_arg(args::input_noc_id_stride_d);
    uint32_t input_noc_id_stride_h = get_arg(args::input_noc_id_stride_h);

    uint32_t input_size_n = get_arg(args::input_size_n);
    uint32_t input_size_c = get_arg(args::input_size_c);
    uint32_t input_size_d = get_arg(args::input_size_d);
    uint32_t input_size_h = get_arg(args::input_size_h);
    uint32_t input_size_w = get_arg(args::input_size_w);

    // index
    uint32_t index0_is_defined = get_arg(args::index0_is_defined);
    uint32_t index1_is_defined = get_arg(args::index1_is_defined);
    uint32_t index2_is_defined = get_arg(args::index2_is_defined);
    uint32_t index3_is_defined = get_arg(args::index3_is_defined);
    uint32_t index4_is_defined = get_arg(args::index4_is_defined);
    uint32_t index0_stick_size = get_arg(args::index0_stick_size);
    uint32_t index1_stick_size = get_arg(args::index1_stick_size);
    uint32_t index2_stick_size = get_arg(args::index2_stick_size);
    uint32_t index3_stick_size = get_arg(args::index3_stick_size);
    uint32_t index4_stick_size = get_arg(args::index4_stick_size);
    uint32_t index_size = get_arg(args::index_size);

    // output
    uint32_t output_size_n = get_arg(args::output_size_n);
    uint32_t output_size_c = get_arg(args::output_size_c);
    uint32_t output_size_d = get_arg(args::output_size_d);
    uint32_t output_size_h = get_arg(args::output_size_h);
    uint32_t output_size_w = get_arg(args::output_size_w);
    uint32_t output_num_stick_width = get_arg(args::output_num_stick_width);

    // etc
    uint32_t start_id = get_arg(args::start_id);
    uint32_t num_sticks = get_arg(args::num_sticks);
    uint32_t element_size = get_arg(args::element_size);
    uint32_t num_elements_per_alignment = get_arg(args::num_elements_per_alignment);
    uint32_t num_alignment_width = get_arg(args::num_alignment_width);

    const auto s0 = TensorAccessor(tensor::s0);

    // Only the index dimensions the caller supplied are bound, so tensor::index<N> and dfb::in<N+1>
    // exist only in a build whose HAS_INDEX<N> define the host emitted. Everything referencing them is
    // gated on that define; at runtime index_is_defined[dim] selects the same dimensions, so the gated
    // code is exactly the code that could have run.
#ifdef HAS_INDEX0
    const auto index0 = TensorAccessor(tensor::index0);
#endif
#ifdef HAS_INDEX1
    const auto index1 = TensorAccessor(tensor::index1);
#endif
#ifdef HAS_INDEX2
    const auto index2 = TensorAccessor(tensor::index2);
#endif
#ifdef HAS_INDEX3
    const auto index3 = TensorAccessor(tensor::index3);
#endif
#ifdef HAS_INDEX4
    const auto index4 = TensorAccessor(tensor::index4);
#endif

    uint32_t index_is_defined[5] = {
        index0_is_defined,
        index1_is_defined,
        index2_is_defined,
        index3_is_defined,
        index4_is_defined,
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

    uint32_t w_index;

#define NOC_MINIMUM_READ_SIZE (32)
#define INDEX_SIZE (4)

    Noc noc;
    DataflowBuffer dfb_in0_obj(dfb::in0);
#ifdef HAS_INDEX0
    DataflowBuffer dfb_in1_obj(dfb::in1);
#endif
#ifdef HAS_INDEX1
    DataflowBuffer dfb_in2_obj(dfb::in2);
#endif
#ifdef HAS_INDEX2
    DataflowBuffer dfb_in3_obj(dfb::in3);
#endif
#ifdef HAS_INDEX3
    DataflowBuffer dfb_in4_obj(dfb::in4);
#endif
#ifdef HAS_INDEX4
    DataflowBuffer dfb_in5_obj(dfb::in5);
#endif

    uint32_t end_id = start_id + num_sticks;
    uint32_t index_size_w = output_size_w;
    for (uint32_t i = start_id; i < end_id; ++i) {
        uint32_t index_w_index = i % num_alignment_width;
        uint32_t index_off = index_w_index * num_elements_per_alignment;
        uint32_t index_start = index_off;
        uint32_t index_end = std::min(index_off + num_elements_per_alignment, index_size_w);

        uint32_t j = 0;
        for (uint32_t index_index = index_start; index_index < index_end; index_index++, j++) {
            // compute src noc id
            uint32_t output_stick_h = (i / num_alignment_width);
            uint32_t output_stick_w = index_index / FACE_WIDTH;
            uint32_t output_stick_idx = output_stick_h * output_num_stick_width + output_stick_w;
            uint32_t input_stick_idx = 0;
            for (int32_t dim = 4; dim >= 0; dim--) {
                uint32_t input_stick_idx_stride = input_stick_idx_strides[dim];

                if (index_is_defined[dim]) {
                    uint32_t index_l1_addr = 0;

                    uint32_t index_noc_id;
                    if (dim == 4) {
                        index_noc_id = index_index / TILE_WIDTH;
                    } else {
                        index_noc_id = index_index / TILE_HEIGHT;
                    }

#ifdef TILIZE_INDEX
#ifdef HAS_INDEX0
                    if (dim == 0) {
                        dfb_in1_obj.reserve_back(1);
                        index_l1_addr = dfb_in1_obj.get_write_ptr();
                        noc.async_read(
                            index0, dfb_in1_obj, INDEX_TILE_SIZE, {.page_id = index_noc_id}, {.offset_bytes = 0});
                    }
#endif
#ifdef HAS_INDEX1
                    if (dim == 1) {
                        dfb_in2_obj.reserve_back(1);
                        index_l1_addr = dfb_in2_obj.get_write_ptr();
                        noc.async_read(
                            index1, dfb_in2_obj, INDEX_TILE_SIZE, {.page_id = index_noc_id}, {.offset_bytes = 0});
                    }
#endif
#ifdef HAS_INDEX2
                    if (dim == 2) {
                        dfb_in3_obj.reserve_back(1);
                        index_l1_addr = dfb_in3_obj.get_write_ptr();
                        noc.async_read(
                            index2, dfb_in3_obj, INDEX_TILE_SIZE, {.page_id = index_noc_id}, {.offset_bytes = 0});
                    }
#endif
#ifdef HAS_INDEX3
                    if (dim == 3) {
                        dfb_in4_obj.reserve_back(1);
                        index_l1_addr = dfb_in4_obj.get_write_ptr();
                        noc.async_read(
                            index3, dfb_in4_obj, INDEX_TILE_SIZE, {.page_id = index_noc_id}, {.offset_bytes = 0});
                    }
#endif
#ifdef HAS_INDEX4
                    if (dim == 4) {
                        dfb_in5_obj.reserve_back(1);
                        index_l1_addr = dfb_in5_obj.get_write_ptr();
                        noc.async_read(
                            index4, dfb_in5_obj, INDEX_TILE_SIZE, {.page_id = index_noc_id}, {.offset_bytes = 0});
                    }
#endif
                    noc.async_read_barrier();

                    if (dim == 4) {
                        volatile tt_l1_ptr int32_t* index_l1_ptr =
                            reinterpret_cast<volatile tt_l1_ptr int32_t*>(index_l1_addr);
                        uint32_t index_dim_offset = index_index % FACE_WIDTH;
                        if ((index_index % TILE_WIDTH) >= 16) {
                            index_dim_offset += 256;
                        }

                        int32_t index_val = index_l1_ptr[index_dim_offset];

                        if (index_val < 0) {
                            index_val += input_size_list[dim];
                        }

                        w_index = index_val;
                        input_stick_idx += index_val / FACE_WIDTH;
                    } else {
                        volatile tt_l1_ptr int32_t* index_l1_ptr =
                            reinterpret_cast<volatile tt_l1_ptr int32_t*>(index_l1_addr);
                        uint32_t index_dim_offset;
                        uint32_t index_tile_idx = index_index % TILE_WIDTH;
                        if (index_tile_idx < FACE_WIDTH) {
                            index_dim_offset = index_tile_idx;
                        } else {
                            index_dim_offset = index_tile_idx + 256 - 16;
                        }

                        int32_t index_val = index_l1_ptr[index_dim_offset];

                        if (index_val < 0) {
                            index_val += input_size_list[dim];
                        }

                        input_stick_idx += index_val * input_stick_idx_stride;
                    }
#endif
#ifdef ROW_MAJOR_INDEX
                    uint32_t noc_offset =
                        ((uint32_t)((index_index * INDEX_SIZE) / NOC_MINIMUM_READ_SIZE)) * NOC_MINIMUM_READ_SIZE;
#ifdef HAS_INDEX0
                    if (dim == 0) {
                        dfb_in1_obj.reserve_back(1);
                        index_l1_addr = dfb_in1_obj.get_write_ptr();
                        noc.async_read(
                            index0,
                            dfb_in1_obj,
                            NOC_MINIMUM_READ_SIZE,
                            {.page_id = 0, .offset_bytes = noc_offset},
                            {.offset_bytes = 0});
                    }
#endif
#ifdef HAS_INDEX1
                    if (dim == 1) {
                        dfb_in2_obj.reserve_back(1);
                        index_l1_addr = dfb_in2_obj.get_write_ptr();
                        noc.async_read(
                            index1,
                            dfb_in2_obj,
                            NOC_MINIMUM_READ_SIZE,
                            {.page_id = 0, .offset_bytes = noc_offset},
                            {.offset_bytes = 0});
                    }
#endif
#ifdef HAS_INDEX2
                    if (dim == 2) {
                        dfb_in3_obj.reserve_back(1);
                        index_l1_addr = dfb_in3_obj.get_write_ptr();
                        noc.async_read(
                            index2,
                            dfb_in3_obj,
                            NOC_MINIMUM_READ_SIZE,
                            {.page_id = 0, .offset_bytes = noc_offset},
                            {.offset_bytes = 0});
                    }
#endif
#ifdef HAS_INDEX3
                    if (dim == 3) {
                        dfb_in4_obj.reserve_back(1);
                        index_l1_addr = dfb_in4_obj.get_write_ptr();
                        noc.async_read(
                            index3,
                            dfb_in4_obj,
                            NOC_MINIMUM_READ_SIZE,
                            {.page_id = 0, .offset_bytes = noc_offset},
                            {.offset_bytes = 0});
                    }
#endif
#ifdef HAS_INDEX4
                    if (dim == 4) {
                        dfb_in5_obj.reserve_back(1);
                        index_l1_addr = dfb_in5_obj.get_write_ptr();
                        noc.async_read(
                            index4,
                            dfb_in5_obj,
                            NOC_MINIMUM_READ_SIZE,
                            {.page_id = 0, .offset_bytes = noc_offset},
                            {.offset_bytes = 0});
                    }
#endif
                    noc.async_read_barrier();

                    volatile tt_l1_ptr int32_t* index_l1_ptr =
                        reinterpret_cast<volatile tt_l1_ptr int32_t*>(index_l1_addr);

                    uint32_t index_dim_offset = (index_index * INDEX_SIZE - noc_offset) / INDEX_SIZE;
                    int32_t index_val = index_l1_ptr[index_dim_offset];

                    if (index_val < 0) {
                        index_val += input_size_list[dim];
                    }

                    if (dim == 4) {
                        w_index = index_val;
                        input_stick_idx += index_val / FACE_WIDTH;
                    } else {
                        input_stick_idx += index_val * input_stick_idx_stride;
                    }

#endif
                } else {
                    uint32_t index_val;

                    auto output_size = output_size_list[dim];
                    index_val = output_stick_idx % output_size;
                    input_stick_idx += index_val * input_stick_idx_stride;
                }
                if (dim == 4) {
                    output_stick_idx /= output_num_stick_width;
                } else {
                    auto output_size = output_size_list[dim];
                    output_stick_idx /= output_size;
                }
            }

            dfb_in0_obj.reserve_back(1);
            uint32_t l1_write_addr = dfb_in0_obj.get_write_ptr();

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

            uint32_t noc_offset =
                get_noc_offset_in_tile(stick_index_5d.h, stick_index_5d.w, tile_index_5d.h, element_size);

            if (num_elements_per_alignment == 8) {
                noc_offset += ((w_index / 8) % 2) * NOC_MINIMUM_READ_SIZE;
            }

            noc.async_read(
                s0,
                dfb_in0_obj,
                NOC_MINIMUM_READ_SIZE,
                {.page_id = noc_id, .offset_bytes = noc_offset},
                {.offset_bytes = 0});
            noc.async_read_barrier();

            if (element_size == 4) {
                volatile tt_l1_ptr uint32_t* index_l1_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_write_addr);
                index_l1_ptr[0] = index_l1_ptr[w_index % num_elements_per_alignment];
            } else if (element_size == 2) {
                volatile tt_l1_ptr uint16_t* index_l1_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_addr);
                index_l1_ptr[0] = index_l1_ptr[w_index % num_elements_per_alignment];
            }

            dfb_in0_obj.push_back(1);
        }
    }
}
