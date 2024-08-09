// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"

void kernel_main() {
    uint32_t dst_addr0  = get_arg_val<uint32_t>(0);

    constexpr uint32_t out_cb_index = get_compile_time_arg_val(0);
    constexpr bool out_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t Ht = get_compile_time_arg_val(2);
    constexpr uint32_t K = get_compile_time_arg_val(3);
    constexpr uint32_t packed_identity_scalar = get_compile_time_arg_val(4);
    constexpr uint32_t Kt =  K % 32 == 0 ? K/32 : K/32 + 1;

    // can amortize the noc reads by doing them side by side for the two tensors
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(out_cb_index);
    const DataFormat data_format = get_dataformat(out_cb_index);

    // Reduce ops need to multiply by a scalar. We always want to multiply by 1.0f
    constexpr uint32_t scale_cb_index = tt::CB::c_in3;
    generate_reduce_scaler(scale_cb_index, packed_identity_scalar);

    const InterleavedAddrGenFast<out_is_dram> interleaved_accessor0 = {
        .bank_base_address = dst_addr0,
        .page_size = tile_bytes,
        .data_format = data_format
    };

    uint32_t tile_id = 0;
    cb_wait_front(out_cb_index, Ht*Kt);
    uint32_t l1_read_addr = get_read_ptr(out_cb_index);
    for (uint32_t j = 0; j < Ht; ++j) {
        for (uint32_t i = 0; i < Kt; ++i) {
            noc_async_write_tile(tile_id, interleaved_accessor0, l1_read_addr);
            l1_read_addr += tile_bytes;
            tile_id++;
        }
    }
    noc_async_write_barrier();
    cb_pop_front(out_cb_index, Ht*Kt);
}
