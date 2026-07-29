// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t stick_nbytes = get_arg(args::stick_nbytes);
    constexpr uint32_t aligned_stick_nbytes_dram = get_arg(args::aligned_stick_nbytes);
    constexpr uint32_t stride_h = get_arg(args::stride_h);
    constexpr uint32_t stride_w = get_arg(args::stride_w);
    constexpr uint32_t input_width = get_arg(args::input_width);
    constexpr uint32_t work_per_core = get_arg(args::work_per_core);

    const auto s_in = TensorAccessor(tensor::src);

    Noc noc;
    DataflowBuffer cb_in0(dfb::src0);

    uint32_t src_index = get_arg(args::src_index);
    uint32_t curr_src_row_index = get_arg(args::curr_src_row_index);
    for (uint32_t input_idx = 0; input_idx < work_per_core; input_idx++) {
        uint32_t curr_src_offset = src_index;
        cb_in0.reserve_back(1);
        uint32_t l1_offset = 0;
        for (uint32_t i = 0; i < stride_h; i++) {
            for (uint32_t j = 0; j < stride_w; j++) {
                noc.async_read(s_in, cb_in0, stick_nbytes, {.page_id = curr_src_offset}, {.offset_bytes = l1_offset});
                curr_src_offset++;
                l1_offset += aligned_stick_nbytes_dram;
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
