// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <limits.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/scratchpad.h"
#include "api/tensor/noc_traits.h"
#include "internal/local_mem/address_access.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    Noc noc;

    constexpr uint32_t stick_size = get_arg(args::stick_size);
    constexpr uint32_t W = get_arg(args::W);
    constexpr uint32_t H = get_arg(args::H);
    constexpr bool skip_negative_entries = get_arg(args::skip_negative_entries);

    // Interleaved input (NEEDS_NOC_COPY): copy each stick into a scratchpad, update it, and write it back.
    // Sharded input: the DFB is the shard itself, so update it in place.
#ifdef NEEDS_NOC_COPY
    const auto s0 = TensorAccessor(tensor::input);
    Scratchpad<uint32_t> in0(scratch::in0);
    volatile tt_l1_ptr uint32_t* stick = &in0[0];
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
    // The L2 cache calls below take a raw address; AddressAccess is the last-resort path for them (#61229).
    const uintptr_t stick_addr = local_mem::detail::AddressAccess::address(in0.local_mem());
#endif
#else
    DataflowBuffer dfb_in0(dfb::in0);
    const uint32_t cb_addr = dfb_in0.get_write_ptr();
    volatile tt_l1_ptr uint32_t* stick = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_addr);
#endif

    for (uint32_t h = 0; h < H; h++) {
#ifdef NEEDS_NOC_COPY
        noc.async_read(s0, in0, stick_size, {.page_id = h}, {.offset_bytes = 0});
        noc.async_read_barrier();
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
        // Quasar DM: the NoC wrote the stick to L1 behind the CPU's caches. Drop any L2 line left by the
        // previous stick (or an earlier program) so the loads below see the new data (as paged_fill_cache does).
        invalidate_l2_cache_range(stick_addr, stick_size);
#endif
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
#ifdef NEEDS_NOC_COPY
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
        // Quasar DM: the stores above are in the CPU's caches, and the NoC reads L1 directly. Flush them so
        // the write below sends the incremented stick (as fill_rm does).
        flush_l2_cache_range(stick_addr, stick_size);
#endif
        noc.async_write(in0, s0, stick_size, {.offset_bytes = 0}, {.page_id = h});
        noc.async_write_barrier();
#endif
    }
}
