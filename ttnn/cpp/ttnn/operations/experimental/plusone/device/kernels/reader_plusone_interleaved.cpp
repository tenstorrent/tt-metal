// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <limits.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

// MetalV2 plusone reader. The device-side dataflow — the Noc object, the CoreLocalMem reads/writes, and the
// CircularBuffer scratch — is preserved verbatim from the descriptor-era kernel; only the binding mechanisms
// change: the scratch CB id comes from the DFB binding token (dfb::), the input accessor from the tensor
// binding (ta::input), and W / H / stick_size from the named-arg namespace (args::, declared
// enqueue-invariant in the spec). Src-is-DRAM and skip-negatives select code paths via host-set defines.
void kernel_main() {
    Noc noc;

    constexpr uint32_t cb_id_in0 = dfb::plusone_scratch;

    const uint32_t W = get_arg(args::W);
    const uint32_t H = get_arg(args::H);
    const uint32_t stick_size = get_arg(args::stick_size);

    const auto s0 = TensorAccessor(ta::input);

    CircularBuffer cb_in0(cb_id_in0);

    // Use cb as L1 scratch memory
    uint32_t cb_addr = cb_in0.get_write_ptr();
    volatile tt_l1_ptr uint32_t* stick = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_addr);

    for (uint32_t h = 0; h < H; h++) {
#ifdef SRC_IS_DRAM
        noc.async_read(s0, CoreLocalMem<uint32_t>(cb_addr), stick_size, {.page_id = h}, {});
        noc.async_read_barrier();
#endif
        for (uint32_t i = 0; i < W; i++) {
            int32_t val = stick[i];
#ifdef SKIP_NEGATIVE_ENTRIES
            // NOTE: If you increment beyond INT32_MAX you will wrap around and get a negative result
            //  values greater than INT32_MAX will overflow and become negative
            if (val < INT32_MAX && val >= 0) {
                stick[i] = val + 1;
            }
#else
            stick[i] = val + 1;
#endif
        }
#ifdef SRC_IS_DRAM
        noc.async_write(CoreLocalMem<uint32_t>(cb_addr), s0, stick_size, {}, {.page_id = h});
        noc.async_write_barrier();
#endif
    }
}
