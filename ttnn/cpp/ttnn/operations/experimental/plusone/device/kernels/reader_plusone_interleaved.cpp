// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <limits.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/scratchpad.h"
#include "api/tensor/local_tensor_accessor.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    Noc noc;

    constexpr uint32_t stick_size = get_arg(args::stick_size);
    constexpr uint32_t W = get_arg(args::W);
    constexpr uint32_t H = get_arg(args::H);
    constexpr bool skip_negative_entries = get_arg(args::skip_negative_entries);

    // The working region is reached through the binding that matches the input's
    // placement (chosen by the host):
    //  - IN0_IS_LOCAL: sharded input — the region is the input shard's node-local L1,
    //    viewed through the tensor binding.
    //  - otherwise: a private scratchpad; the DRAM path (SRC0_IS_DRAM) DMAs through it.
    // The element type is volatile, as the raw pointer this replaces was: the NoC read
    // barrier is not a compiler barrier, so the loads must not be hoisted or elided.
#ifdef IN0_IS_LOCAL
    const LocalTensorAccessor<volatile uint32_t> in0(tensor::input);
#else
    const Scratchpad<volatile uint32_t> in0(scratch::in0);
#endif

#ifdef SRC0_IS_DRAM
    const auto s0 = TensorAccessor(tensor::input);
#endif

    for (uint32_t h = 0; h < H; h++) {
#ifdef SRC0_IS_DRAM
        noc.async_read(s0, in0, stick_size, {.page_id = h}, {});
        noc.async_read_barrier();
#endif
        for (uint32_t i = 0; i < W; i++) {
            int32_t val = in0[i];
            if constexpr (skip_negative_entries) {
                // NOTE: If you increment beyond INT32_MAX you will wrap around and get a negative result
                //  values greater than INT32_MAX will overflow and become negative
                if (val < INT32_MAX && val >= 0) {
                    in0[i] = val + 1;
                }
            } else {
                in0[i] = val + 1;
            }
        }
#ifdef SRC0_IS_DRAM
        noc.async_write(in0, s0, stick_size, {}, {.page_id = h});
        noc.async_write_barrier();
#endif
    }
}
