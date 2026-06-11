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

// Metal 2.0: the input address comes from the TensorAccessor binding (ta::s0_args), the CB id from the
// DFB binding token (dfb::), and the structural scalars (stick_size, W, H, src0_is_dram,
// skip_negative_entries) from named compile-time args (args::, constexpr) instead of positional
// compile-time / runtime slots. The op is in-place: the reader NoC-reads the input into the L1 scratch
// DFB, increments, and NoC-writes back to the SAME buffer (the DFB is bound as both producer and
// consumer). When sharded the DFB borrows the input's L1 storage, so the increment happens in place with
// no NoC traffic (src0_is_dram == false).
void kernel_main() {
    Noc noc;

    constexpr uint32_t cb_id_in0 = dfb::cb_id_in0;
    constexpr bool src0_is_dram = (bool)get_arg(args::src0_is_dram);
    constexpr uint32_t stick_size = get_arg(args::stick_size);
    constexpr uint32_t W = get_arg(args::W);
    constexpr uint32_t H = get_arg(args::H);
    constexpr bool skip_negative_entries = (bool)get_arg(args::skip_negative_entries);

    const auto s0 = TensorAccessor(ta::s0_args);

    CircularBuffer cb_in0(cb_id_in0);

    // Use cb as L1 scratch memory
    uint32_t cb_addr = cb_in0.get_write_ptr();
    volatile tt_l1_ptr uint32_t* stick = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_addr);

    for (uint32_t h = 0; h < H; h++) {
        if (src0_is_dram) {
            noc.async_read(s0, CoreLocalMem<uint32_t>(cb_addr), stick_size, {.page_id = h}, {});
            noc.async_read_barrier();
        }
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
        if (src0_is_dram) {
            noc.async_write(CoreLocalMem<uint32_t>(cb_addr), s0, stick_size, {}, {.page_id = h});
            noc.async_write_barrier();
        }
    }
}
