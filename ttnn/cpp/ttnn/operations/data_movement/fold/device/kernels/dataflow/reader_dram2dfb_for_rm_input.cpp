// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t stick_nbytes = get_arg(args::stick_nbytes);
    constexpr uint32_t aligned_stick_nbytes_dram = get_arg(args::aligned_stick_nbytes_dram);
    constexpr uint32_t stride_h = get_arg(args::stride_h);
    constexpr uint32_t stride_w = get_arg(args::stride_w);
    constexpr uint32_t input_width = get_arg(args::input_width);

    // Generic accessor + noc_async_read_sharded helper: compile-time dispatch to noc.async_read for interleaved /
    // H-sharded inputs and per-shard reads for W/B-sharded inputs.
    const auto s_in = TensorAccessor(tensor::src);

    Noc noc;
    DataflowBuffer dfb_in0(dfb::in0);

    // work_per_core is runtime (per-core) so unused cores skip iteration and the cliff core can carry a partial tail.
    uint32_t work_per_core = get_arg(args::work_per_core);
    uint32_t src_index = get_arg(args::src_index);
    uint32_t curr_src_row_index = get_arg(args::curr_src_row_index);
    for (uint32_t input_idx = 0; input_idx < work_per_core; input_idx++) {
        uint32_t curr_src_offset = src_index;
        dfb_in0.reserve_back(1);
        uint32_t l1_write_addr = dfb_in0.get_write_ptr();
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
        dfb_in0.push_back(1);

        curr_src_row_index += stride_w;
        if (curr_src_row_index >= (input_width)) {
            src_index += input_width * (stride_h - 1);
            curr_src_row_index = 0;
        }
        src_index += stride_w;
    }
}
