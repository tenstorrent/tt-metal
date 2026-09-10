// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <limits.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    Noc noc;

    constexpr uint32_t stick_size = get_arg(args::stick_size);
    constexpr uint32_t W = get_arg(args::W);
    constexpr uint32_t H = get_arg(args::H);
    constexpr bool skip_negative_entries = get_arg(args::skip_negative_entries);

    // The input tensor is bound (and DMA'd through the accessor) only on the DRAM path;
    // SRC0_IS_DRAM is defined by the host when the input is DRAM-interleaved. On the
    // sharded / L1-interleaved paths the DFB is the working buffer directly (borrowed
    // input shard, or plain scratch) and no accessor is bound.
#ifdef SRC0_IS_DRAM
    const auto s0 = TensorAccessor(tensor::input);
#endif

    DataflowBuffer dfb_in0(dfb::in0);

    // Use dfb as L1 scratch memory
    uint32_t cb_addr = dfb_in0.get_write_ptr();
    volatile tt_l1_ptr uint32_t* stick = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_addr);

    for (uint32_t h = 0; h < H; h++) {
#ifdef SRC0_IS_DRAM
        noc.async_read(s0, CoreLocalMem<uint32_t>(cb_addr), stick_size, {.page_id = h}, {});
        noc.async_read_barrier();
#endif
        for (uint32_t i = 0; i < W; i++) {
            int32_t val = stick[i];
            if constexpr (skip_negative_entries) {
                // NOTE: If you increment beyond INT32_MAX you will wrap around and get a negative result
                //  values greater than INT32_MAX will overflow and become negative
                if (val < INT32_MAX && val >= 0) {
                    stick[i] = val + 1;
                }
            } else {
                stick[i] = val + 1;
            }
        }
#ifdef SRC0_IS_DRAM
        noc.async_write(CoreLocalMem<uint32_t>(cb_addr), s0, stick_size, {}, {.page_id = h});
        noc.async_write_barrier();
#endif
    }
}
