// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "experimental/kernel_args.h"

using namespace tt::data_movement::common;
void kernel_main() {
    constexpr uint32_t stick_nbytes = get_arg(args::stick_nbytes);
    constexpr uint32_t aligned_stick_nbytes_dram = get_arg(args::aligned_stick_nbytes);
    constexpr uint32_t stride_h = get_arg(args::stride_h);
    constexpr uint32_t stride_w = get_arg(args::stride_w);
    constexpr uint32_t input_width = get_arg(args::input_width);
    constexpr uint32_t work_per_core = get_arg(args::work_per_core);
    constexpr bool is_l1_aligned = get_arg(args::is_l1_aligned);

    const auto s_out = TensorAccessor(tensor::dst);
    uint32_t dst_index = get_arg(args::dst_index);

    constexpr uint32_t patch_size = stride_h * stride_w;

    Noc noc;
    DataflowBuffer cb_in0(dfb::src0);
    // src1 is an intermediate L1 scratch buffer, used only on the !is_l1_aligned path.
    // Its DFB binding (and therefore its dfb::src1 token) only exists in that build, so
    // the constexpr alias and every use are gated behind the matching preprocessor define.
#ifdef USE_SCRATCH_SRC1
    DataflowBuffer cb_in1(dfb::src1);

    uint32_t intermed_l1_scratch = cb_in1.get_write_ptr();
    // Datatypes will be multiple of 2 bytes only so it is safe to use uint16_t pointer
    volatile tt_l1_ptr uint16_t* patch_data = (volatile uint16_t*)intermed_l1_scratch;
#endif
    for (uint32_t input_idx = 0; input_idx < work_per_core; input_idx++) {
        cb_in0.wait_front(1);
        uint32_t l1_addr = cb_in0.get_read_ptr();
        if constexpr (!is_l1_aligned) {
#ifdef USE_SCRATCH_SRC1
            uint32_t idx = 0;
            for (uint32_t i = 0; i < patch_size; i++) {
                for (uint32_t j = 0; j < (stick_nbytes / 2); j++) {
                    patch_data[idx++] = *(volatile uint16_t*)(l1_addr + j * 2);
                }
                l1_addr += aligned_stick_nbytes_dram;
            }
            // Scratch buffer (cb_in1) is populated at its WRITE_PTR; no push_back has advanced it yet.
            // The legacy `use<CB::AddrSelector::WRITE_PTR>(cb_in1)` sources the NoC write from the
            // write pointer; a bare DataflowBuffer source would use its read pointer, so use a
            // CoreLocalMem view of the write pointer to preserve the original semantics.
            noc.async_write(
                CoreLocalMem<uint32_t>(cb_in1.get_write_ptr()),
                s_out,
                stick_nbytes * patch_size,
                {},
                {.page_id = dst_index});
#endif
        } else {
            // If L1 aligned, write directly from the circular buffer
            noc.async_write(cb_in0, s_out, stick_nbytes * patch_size, {}, {.page_id = dst_index});
        }
        noc.async_write_barrier();
        cb_in0.pop_front(1);
        dst_index++;
    }
}
