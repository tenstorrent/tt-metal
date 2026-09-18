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

using namespace tt::data_movement::common;
void kernel_main() {
    constexpr uint32_t stick_nbytes = get_arg(args::stick_nbytes);
    constexpr uint32_t aligned_stick_nbytes_dram = get_arg(args::aligned_stick_nbytes_dram);
    constexpr uint32_t stride_h = get_arg(args::stride_h);
    constexpr uint32_t stride_w = get_arg(args::stride_w);
    constexpr uint32_t input_width = get_arg(args::input_width);

    constexpr uint32_t patch_size = stride_h * stride_w;
    constexpr uint32_t output_stick_nbytes = stick_nbytes * patch_size;
    // Generic accessor + noc_async_write_sharded helper: compile-time dispatch to noc.async_write for interleaved /
    // H-sharded outputs and per-shard writes for W/B-sharded outputs.
    const auto s_out = TensorAccessor(tensor::dst);

    Noc noc;
    DataflowBuffer dfb_in0(dfb::in0);

    // work_per_core is runtime (per-core) so unused cores skip iteration and the cliff core can carry a partial tail.
    uint32_t work_per_core = get_arg(args::work_per_core);
    uint32_t dst_index = get_arg(args::dst_index);

    // The src1 scratch DFB is bound (and used) only when the stick is not L1-aligned.
    // Its binding, and every reference to it, is #ifdef-gated on the same condition the host
    // uses to bind it (FOLD_RM_NOT_L1_ALIGNED). In the aligned build the DFB is neither bound
    // nor touched (matching the host, which does not allocate it).
#ifdef FOLD_RM_NOT_L1_ALIGNED
    DataflowBuffer dfb_in1(dfb::in1);
    uint32_t intermed_l1_scratch = dfb_in1.get_write_ptr();
    // Datatypes will be multiple of 2 bytes only so it is safe to use uint16_t pointer
    volatile tt_l1_ptr uint16_t* patch_data = (volatile uint16_t*)intermed_l1_scratch;
#endif
    for (uint32_t input_idx = 0; input_idx < work_per_core; input_idx++) {
        dfb_in0.wait_front(1);
#ifdef FOLD_RM_NOT_L1_ALIGNED
        {
            uint32_t idx = 0;
            uint32_t l1_addr = dfb_in0.get_read_ptr();
            for (uint32_t i = 0; i < patch_size; i++) {
                for (uint32_t j = 0; j < (stick_nbytes / 2); j++) {
                    patch_data[idx++] = *(volatile uint16_t*)(l1_addr + j * 2);
                }
                l1_addr += aligned_stick_nbytes_dram;
            }
            noc_async_write_sharded(noc, intermed_l1_scratch, s_out, dst_index, /*offset=*/0, output_stick_nbytes);
        }
#else
        {
            noc_async_write_sharded(
                noc, dfb_in0.get_read_ptr(), s_out, dst_index, /*offset=*/0, output_stick_nbytes);
        }
#endif
        noc.async_write_barrier();
        dfb_in0.pop_front(1);
        dst_index++;
    }
}
