// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

#include "api/debug/dprint.h"

void kernel_main() {
    uint32_t input_addr = get_arg_val<uint32_t>(0);
    uint32_t output_addr_0 = get_arg_val<uint32_t>(1);
    uint32_t output_addr_1 = get_arg_val<uint32_t>(2);
    uint32_t aligned_elements = get_arg_val<uint32_t>(3);
    uint32_t actual_elements = get_arg_val<uint32_t>(4);
    uint32_t element_size = get_arg_val<uint32_t>(5);

    constexpr uint32_t input_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t output_cb_index_0 = get_compile_time_arg_val(1);
    constexpr uint32_t output_cb_index_1 = get_compile_time_arg_val(2);
    constexpr auto src0_args = TensorAccessorArgs<3>();
    constexpr auto dst0_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();
    constexpr auto dst1_args = TensorAccessorArgs<dst0_args.next_compile_time_args_offset()>();

    // Third argument page_size from runtime args overrides TensorAccessorArgs::AlignedPageSize, which may be stale on
    // program cache hits.
    const auto s0 = TensorAccessor(src0_args, input_addr, aligned_elements * element_size);
    const auto out0 = TensorAccessor(dst0_args, output_addr_0);
    const auto out1 = TensorAccessor(dst1_args, output_addr_1, aligned_elements * element_size);

    experimental::Noc noc;
    experimental::CircularBuffer input_cb(input_cb_index);
    experimental::CircularBuffer output_cb_0(output_cb_index_0);
    experimental::CircularBuffer output_cb_1(output_cb_index_1);

    input_cb.reserve_back(1);
    uint32_t input_l1_addr = input_cb.get_write_ptr();
    noc.async_read(s0, input_cb, aligned_elements * element_size, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    input_cb.push_back(1);
    output_cb_1.reserve_back(1);
    uint32_t indices_cb_id_index_addr = output_cb_1.get_write_ptr();
    volatile tt_l1_ptr int* indices_cb_id_index_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr int*>(indices_cb_id_index_addr);

#if NUM_BYTES == 4
    volatile tt_l1_ptr int* input_addr_ptr = reinterpret_cast<volatile tt_l1_ptr int*>(input_l1_addr);
#elif NUM_BYTES == 2
    volatile tt_l1_ptr uint16_t* input_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(input_l1_addr);
#elif NUM_BYTES == 1
    volatile tt_l1_ptr char* input_addr_ptr = reinterpret_cast<volatile tt_l1_ptr char*>(input_l1_addr);
#endif

    uint32_t num_non_zero_indices = 0;
    uint32_t local_index = 0;
    for (uint32_t i = 0; i < actual_elements; i++) {
        if (input_addr_ptr[i] != 0) {
            indices_cb_id_index_addr_ptr[num_non_zero_indices] = i;
            num_non_zero_indices++;
        }
        local_index++;
    }

    output_cb_0.reserve_back(1);
    uint32_t output_l1_addr = output_cb_0.get_write_ptr();
    volatile tt_l1_ptr uint32_t* output_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(output_l1_addr);
    output_addr_ptr[0] = num_non_zero_indices;
    // Use WRITE_PTR since data was written via get_write_ptr()
    noc.async_write(
        experimental::use<experimental::CircularBuffer::AddrSelector::WRITE_PTR>(output_cb_0),
        out0,
        32,
        {.offset_bytes = 0},
        {.page_id = 0});
    noc.async_write_barrier();
    output_cb_0.push_back(1);

    // Use WRITE_PTR since data was written via get_write_ptr()
    noc.async_write(
        experimental::use<experimental::CircularBuffer::AddrSelector::WRITE_PTR>(output_cb_1),
        out1,
        aligned_elements * 4,
        {.offset_bytes = 0},
        {.page_id = 0});
    noc.async_write_barrier();
    output_cb_1.push_back(1);
}
