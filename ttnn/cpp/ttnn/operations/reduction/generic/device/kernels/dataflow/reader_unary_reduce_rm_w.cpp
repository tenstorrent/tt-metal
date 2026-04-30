// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

// Row-major input pages (one page = one full tensor row along W). RM circular buffer index must match host factory.
void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_pages = get_arg_val<uint32_t>(1);
    uint32_t start_page = get_arg_val<uint32_t>(2);
    constexpr uint32_t scaler_bits = get_compile_time_arg_val(0);
    constexpr auto tensor_args = TensorAccessorArgs<1>();

    constexpr uint32_t cb_id_scaler = tt::CBIndex::c_2;
    float scaler_f = __builtin_bit_cast(float, scaler_bits);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_id_scaler, REDUCE_OP, REDUCE_DIM>(scaler_f);

    constexpr uint32_t cb_id_rm = tt::CBIndex::c_24;
    constexpr uint32_t onepage = 1;

    const uint32_t page_bytes = get_local_cb_interface(cb_id_rm).fifo_page_size;

    auto tensor_accessor = TensorAccessor(tensor_args, src_addr);

    experimental::Noc noc;
    experimental::CircularBuffer cb_rm(cb_id_rm);

    uint32_t end_page = start_page + num_pages;
    for (uint32_t page_id = start_page; page_id < end_page; page_id++) {
        cb_rm.reserve_back(onepage);
        noc.async_read(tensor_accessor, cb_rm, page_bytes, {.page_id = page_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_rm.push_back(onepage);
    }
}
