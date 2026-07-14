// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Feeds the OuterStream toy chain (test_outer_stream.py):
//   cb_a (c_0): the full Ht*Wt walk, one tile per (ht, wt), pushed in order.
//   cb_b (c_1): ONE tile per row (b[ht]), pushed at the top of each row so it is present when
//               the compute kernel starts that row. cb_b stays shallow (2-deep) — the producer
//               never stages the whole column, which is the point of OuterStream.
// Runtime args: a_addr, b_addr, Ht, Wt. Page bytes come from each CB's configured page size.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t src0_addr = get_arg_val<uint32_t>(0);  // a: Ht*Wt tiles
    const uint32_t src1_addr = get_arg_val<uint32_t>(1);  // b: Ht tiles (one per row)
    const uint32_t Ht = get_arg_val<uint32_t>(2);
    const uint32_t Wt = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb0 = 0;
    constexpr uint32_t cb1 = 1;

    constexpr auto src0_args = TensorAccessorArgs<0>();
    constexpr auto src1_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();

    const uint32_t bytes0 = get_local_cb_interface(cb0).fifo_page_size;
    const uint32_t bytes1 = get_local_cb_interface(cb1).fifo_page_size;

    const auto s0 = TensorAccessor(src0_args, src0_addr);
    const auto s1 = TensorAccessor(src1_args, src1_addr);

    Noc noc;
    CircularBuffer c0(cb0), c1(cb1);

    constexpr uint32_t onetile = 1;
    uint32_t a_id = 0;
    for (uint32_t ht = 0; ht < Ht; ++ht) {
        // One b tile for this row.
        c1.reserve_back(onetile);
        noc.async_read(s1, c1, bytes1, {.page_id = ht}, {.offset_bytes = 0});
        noc.async_read_barrier();
        c1.push_back(onetile);
        // Wt a tiles for this row.
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            c0.reserve_back(onetile);
            noc.async_read(s0, c0, bytes0, {.page_id = a_id}, {.offset_bytes = 0});
            noc.async_read_barrier();
            c0.push_back(onetile);
            ++a_id;
        }
    }
}
