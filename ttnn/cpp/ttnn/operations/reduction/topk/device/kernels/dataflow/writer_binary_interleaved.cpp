// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    // Runtime args
    const uint32_t dst_addr0 = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr1 = get_arg_val<uint32_t>(1);
    const uint32_t id = get_arg_val<uint32_t>(2);
    const uint32_t work_per_core = get_arg_val<uint32_t>(3);

    // Compile time args
    constexpr uint32_t values_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t output_ind_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t Ht = get_compile_time_arg_val(2);
    constexpr uint32_t Kt = get_compile_time_arg_val(3);
    constexpr uint32_t total_number_of_cores = get_compile_time_arg_val(4);
    constexpr auto values_tensor_args = TensorAccessorArgs<5>();
    constexpr auto indices_tensor_args = TensorAccessorArgs<values_tensor_args.next_compile_time_args_offset()>();

    // Constants
    constexpr uint32_t onetile = 1;

    // Tensor config
    const auto values_tensor_accessor = TensorAccessor(values_tensor_args, dst_addr0);
    const auto indices_tensor_accessor = TensorAccessor(indices_tensor_args, dst_addr1);

    Noc noc;
    DataflowBuffer values_cb(values_cb_index);
    DataflowBuffer indices_cb(output_ind_cb_index);
    const uint32_t tile_bytes_val = values_cb.get_entry_size();
    const uint32_t tile_bytes_idx = indices_cb.get_entry_size();

    // Get Kt rows of values and then Kt rows of indices from compute kernel
    for (uint32_t core_loop = 0; core_loop < work_per_core; core_loop++) {
        const uint32_t row = id + core_loop * total_number_of_cores;

        // TopK values
        for (uint32_t k = 0; k < Kt; ++k) {
            values_cb.wait_front(onetile);
            noc.async_write(
                values_cb, values_tensor_accessor, tile_bytes_val, {.offset_bytes = 0}, {.page_id = row * Kt + k});
            noc.async_write_barrier();
            values_cb.pop_front(onetile);
        }  // k loop

        // TopK indices
        for (uint32_t k = 0; k < Kt; ++k) {
            indices_cb.wait_front(onetile);
            noc.async_write(
                indices_cb, indices_tensor_accessor, tile_bytes_idx, {.offset_bytes = 0}, {.page_id = row * Kt + k});
            noc.async_write_barrier();
            indices_cb.pop_front(onetile);
        }  // k loop
    }  // core_loop loop
}
