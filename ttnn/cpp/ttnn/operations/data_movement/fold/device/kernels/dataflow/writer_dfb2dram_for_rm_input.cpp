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

template <
    uint32_t stick_nbytes,
    uint32_t aligned_stick_nbytes_dram,
    uint32_t stride_h,
    uint32_t stride_w,
    uint32_t needs_intermediate>
TT_KERNEL void writer(uint32_t work_per_core, uint32_t dst_index) {
    constexpr uint32_t patch_size = stride_h * stride_w;
    constexpr uint32_t output_stick_nbytes = stick_nbytes * patch_size;
    // Generic accessor + noc_async_write_sharded helper: compile-time dispatch to noc.async_write for interleaved /
    // H-sharded outputs and per-shard writes for W/B-sharded outputs.
    const auto s_out = TensorAccessor(tensor::dst);

    Noc noc;
    DataflowBuffer dfb_in0(dfb::in0);
    uint32_t intermed_l1_scratch = 0;
    // In aligned builds, dfb::in1 is an unbound compile-only sentinel that must stay in this discarded branch.
    if constexpr (needs_intermediate) {
        DataflowBuffer dfb_in1(dfb::in1);
        intermed_l1_scratch = dfb_in1.get_write_ptr();
    }

    // work_per_core is runtime (per-core) so unused cores skip iteration and the cliff core can carry a partial tail.
    for (uint32_t input_idx = 0; input_idx < work_per_core; input_idx++) {
        dfb_in0.wait_front(1);
        if constexpr (needs_intermediate) {
            // Datatypes are multiples of 2 bytes, so a uint16_t pointer is sufficient.
            volatile tt_l1_ptr uint16_t* patch_data = (volatile uint16_t*)intermed_l1_scratch;
            uint32_t idx = 0;
            uint32_t l1_addr = dfb_in0.get_read_ptr();
            for (uint32_t i = 0; i < patch_size; i++) {
                for (uint32_t j = 0; j < (stick_nbytes / 2); j++) {
                    patch_data[idx++] = *(volatile uint16_t*)(l1_addr + j * 2);
                }
                l1_addr += aligned_stick_nbytes_dram;
            }
            tt::data_movement::common::noc_async_write_sharded(
                noc, intermed_l1_scratch, s_out, dst_index, /*offset=*/0, output_stick_nbytes);
        } else {
            tt::data_movement::common::noc_async_write_sharded(
                noc, dfb_in0.get_read_ptr(), s_out, dst_index, /*offset=*/0, output_stick_nbytes);
        }
        noc.async_write_barrier();
        dfb_in0.pop_front(1);
        dst_index++;
    }
}
