// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(1);
    constexpr uint32_t aligned_stick_nbytes_dram = get_compile_time_arg_val(2);
    constexpr uint32_t stride_h = get_compile_time_arg_val(3);
    constexpr uint32_t stride_w = get_compile_time_arg_val(4);
    constexpr uint32_t input_width = get_compile_time_arg_val(5);
    // work_per_core is now runtime (per-core) so unused cores skip iteration and the cliff core can carry a partial
    // tail.
    constexpr auto src_args = TensorAccessorArgs<8>();

    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t work_per_core = get_arg_val<uint32_t>(1);
    // noc_async_read_sharded compile-time collapses to noc.async_read for interleaved and
    // single-page-per-row (H-sharded) buffers, and splits per-shard for W/B-sharded inputs.
    // Matches the pattern transpose/slice/pad/permute/repeat use.
    const auto s_in = TensorAccessor(src_args, src_addr);

    Noc noc;
    experimental::CB cb_in0(cb_id_in0);

    uint32_t src_index = get_arg_val<uint32_t>(2);
    uint32_t curr_src_row_index = get_arg_val<uint32_t>(3);
    for (uint32_t input_idx = 0; input_idx < work_per_core; input_idx++) {
        uint32_t curr_src_offset = src_index;
        cb_in0.reserve_back(1);
        uint32_t l1_write_addr = cb_in0.get_write_ptr();
        for (uint32_t i = 0; i < stride_h; i++) {
            for (uint32_t j = 0; j < stride_w; j++) {
                tt::data_movement::common::noc_async_read_sharded(
                    noc, l1_write_addr, s_in, curr_src_offset, /*offset=*/0, /*size=*/stick_nbytes);
                curr_src_offset++;
                l1_write_addr += aligned_stick_nbytes_dram;
            }
            curr_src_offset += input_width - stride_w;
        }
        noc.async_read_barrier();
        cb_in0.push_back(1);

        curr_src_row_index += stride_w;
        if (curr_src_row_index >= (input_width)) {
            src_index += input_width * (stride_h - 1);
            curr_src_row_index = 0;
        }
        src_index += stride_w;
    }
}
