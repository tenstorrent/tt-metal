// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
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
    int32_t index_start_dim = static_cast<int32_t>(get_arg(args::index_start_dim));
    int32_t index_end_dim = static_cast<int32_t>(get_arg(args::index_end_dim));

    // output
    uint32_t output_size_n = get_arg(args::output_size_n);
    uint32_t output_size_c = get_arg(args::output_size_c);
    uint32_t output_size_d = get_arg(args::output_size_d);
    uint32_t output_size_h = get_arg(args::output_size_h);
    uint32_t output_size_w = get_arg(args::output_size_w);

    // etc
    uint32_t start_id = get_arg(args::start_id);
    uint32_t num_sticks = get_arg(args::num_sticks);
    uint32_t stick_size = get_arg(args::stick_size);

    // The aligned page size of a row-major stick rides the binding token, baked in when the program is
    // built. A row of a different width is a different program-cache key, so the program is rebuilt
    // rather than reused with a stale page size.
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
                if (is_first_index) {
                    index_index = output_stick_idx % index_size;
                }

                uint32_t index_l1_addr = 0;
                // Selected per dimension below; only a dimension whose index tensor is bound assigns
                // it, which is the same condition index_is_defined[dim] tests, so it is never null here.
                DataflowBuffer* index_dfb_obj = nullptr;
#ifdef HAS_INDEX0
                if (dim == 0) {
                    index_dfb_obj = &dfb_in1_obj;
                    dfb_in1_obj.reserve_back(1);
                    index_l1_addr = dfb_in1_obj.get_write_ptr();
                    noc.async_read(index0, dfb_in1_obj, index_stick_sizes[dim], {.page_id = 0}, {.offset_bytes = 0});
                }
#endif
#ifdef HAS_INDEX1
                if (dim == 1) {
                    index_dfb_obj = &dfb_in2_obj;
                    dfb_in2_obj.reserve_back(1);
                    index_l1_addr = dfb_in2_obj.get_write_ptr();
                    noc.async_read(index1, dfb_in2_obj, index_stick_sizes[dim], {.page_id = 0}, {.offset_bytes = 0});
                }
#endif
#ifdef HAS_INDEX2
                if (dim == 2) {
                    index_dfb_obj = &dfb_in3_obj;
                    dfb_in3_obj.reserve_back(1);
                    index_l1_addr = dfb_in3_obj.get_write_ptr();
                    noc.async_read(index2, dfb_in3_obj, index_stick_sizes[dim], {.page_id = 0}, {.offset_bytes = 0});
                }
#endif
#ifdef HAS_INDEX3
                if (dim == 3) {
                    index_dfb_obj = &dfb_in4_obj;
                    dfb_in4_obj.reserve_back(1);
                    index_l1_addr = dfb_in4_obj.get_write_ptr();
                    noc.async_read(index3, dfb_in4_obj, index_stick_sizes[dim], {.page_id = 0}, {.offset_bytes = 0});
                }
#endif
                noc.async_read_barrier();
                index_dfb_obj->push_back(1);

                volatile tt_l1_ptr int32_t* index_l1_ptr = reinterpret_cast<volatile tt_l1_ptr int32_t*>(index_l1_addr);
                int32_t noc_idx = index_l1_ptr[index_index];

                index_dfb_obj->wait_front(1);
                index_dfb_obj->pop_front(1);

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

        dfb_in0_obj.reserve_back(1);
        noc.async_read(s0, dfb_in0_obj, stick_size, {.page_id = noc_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_in0_obj.push_back(1);
    }
}
