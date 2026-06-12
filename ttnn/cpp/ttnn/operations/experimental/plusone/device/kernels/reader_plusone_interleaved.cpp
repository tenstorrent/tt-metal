// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <limits.h>
#include "api/dataflow/dataflow_api.h"

// MetalV2 plusone reader — the descriptor-era kernel, ported to the spec binding namespaces and otherwise
// unchanged. Reads one row-major stick at a time into an L1 scratch buffer, increments each element, and
// writes it back in place. The logic is preserved; only the binding mechanisms change:
//   - the scratch CB id comes from the DFB binding token (dfb::), not a compile-time arg slot;
//   - the input address comes from the TensorAccessor binding (ta::input), not a positional runtime arg;
//   - W / H come from the named-arg namespace (args::), declared enqueue-invariant in the spec;
//   - src-is-DRAM and skip-negative-entries select code paths via host-set defines.
// For a DRAM input the stick is NOC-copied to scratch, modified, and copied back; for a sharded L1 input
// the scratch CB is borrowed from the input buffer (see the factory), so the modify is already in place.
void kernel_main() {
    constexpr uint32_t cb_id_in0 = dfb::plusone_scratch;

    const uint32_t W = get_arg(args::W);
    const uint32_t H = get_arg(args::H);

    const auto s0 = TensorAccessor(ta::input);

    uint32_t cb_addr = get_write_ptr(cb_id_in0);
    volatile tt_l1_ptr uint32_t* stick = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_addr);

    for (uint32_t h = 0; h < H; h++) {
#ifdef SRC_IS_DRAM
        noc_async_read_page(h, s0, cb_addr);
        noc_async_read_barrier();
#endif
        for (uint32_t i = 0; i < W; i++) {
            int32_t val = stick[i];
#ifdef SKIP_NEGATIVE_ENTRIES
            // NOTE: incrementing beyond INT32_MAX wraps to a negative result, so values >= INT32_MAX are skipped.
            if (val < INT32_MAX && val >= 0) {
                stick[i] = val + 1;
            }
#else
            stick[i] = val + 1;
#endif
        }
#ifdef SRC_IS_DRAM
        noc_async_write_page(h, s0, cb_addr);
        noc_async_write_barrier();
#endif
    }
}
