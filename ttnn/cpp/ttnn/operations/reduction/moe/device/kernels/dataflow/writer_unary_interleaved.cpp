// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    uint32_t dst_addr0 = get_arg_val<uint32_t>(0);

    constexpr uint32_t out_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t Ht = get_compile_time_arg_val(1);
    constexpr uint32_t K = get_compile_time_arg_val(2);
    constexpr uint32_t packed_identity_scalar = get_compile_time_arg_val(3);
    constexpr uint32_t Kt = K % 32 == 0 ? K / 32 : K / 32 + 1;

    constexpr auto out_args = TensorAccessorArgs<4>();

    // can amortize the noc reads by doing them side by side for the two tensors
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(out_cb_index);
    const DataFormat data_format = get_dataformat(out_cb_index);

    // Reduce ops need to multiply by a scalar. We always want to multiply by 1.0f
    constexpr uint32_t scale_cb_index = tt::CBIndex::c_3;
    dataflow_kernel_lib::generate_reduce_scaler(scale_cb_index, packed_identity_scalar);

    const auto interleaved_accessor0 = TensorAccessor(out_args, dst_addr0, tile_bytes);

    experimental::Noc noc;
    experimental::CircularBuffer cb_out(out_cb_index);

    uint32_t tile_id = 0;
    cb_out.wait_front(Ht * Kt);
    for (uint32_t j = 0; j < Ht; ++j) {
        for (uint32_t i = 0; i < Kt; ++i) {
            noc.async_write(
                cb_out,
                interleaved_accessor0,
                tile_bytes,
                {.offset_bytes = tile_id * tile_bytes},
                {.page_id = tile_id});
            tile_id++;
        }
    }
    noc.async_write_barrier();
    cb_out.pop_front(Ht * Kt);
}
