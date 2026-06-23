// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 conversion (in place; this kernel is transpose-owned). The device-side NoC + local-copy
// logic is unchanged; only the resource bindings move to the Metal 2.0 namespaces (dfb::/args::).
// cb_src (dfb::cb_src) is the borrowed input shard — read by L1 address (get_read_ptr); cb_dst
// (dfb::cb_dst) is the row-gathered tile-staging buffer produced for the compute kernel.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// get_arg lives in `namespace experimental` and is normally found via ADL on the args:: accessor
// type. The CoreLocalMem/UnicastEndpoint NoC-with-state includes this kernel pulls in change the
// lookup context enough that ADL doesn't resolve it here, so bring the overload set in explicitly.
using experimental::get_arg;

void kernel_main() {
    constexpr uint32_t num_hw_blocks_per_core = get_arg(args::num_hw_blocks_per_core);
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t H_per_tile = get_arg(args::H_per_tile);
    constexpr uint32_t H_per_tile_last = get_arg(args::H_per_tile_last);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t W_size_bytes = get_arg(args::W_size_bytes);
    constexpr uint32_t l1_write_offset_bytes = get_arg(args::l1_write_offset_bytes);

    const uint32_t stick_size_bytes = W_size_bytes;

    Noc noc;
    DataflowBuffer cb_src(dfb::cb_src);
    DataflowBuffer cb_dst(dfb::cb_dst);

    uint32_t src_addr = cb_src.get_read_ptr();

    noc.set_async_read_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
        UnicastEndpoint{},
        stick_size_bytes,
        {.noc_x = (uint32_t)my_x[noc.get_noc_id()], .noc_y = (uint32_t)my_y[noc.get_noc_id()], .addr = src_addr});

    for (uint32_t n = 0; n < num_hw_blocks_per_core; n++) {
        for (uint32_t h = 0; h < Ht; ++h) {
            cb_dst.reserve_back(Wt);
            uint32_t l1_write_addr = cb_dst.get_write_ptr();
            uint32_t H_curr = h == Ht - 1 ? H_per_tile_last : H_per_tile;
            for (uint32_t h_datum = 0; h_datum < H_curr; ++h_datum) {
                CoreLocalMem<uint32_t> dst(l1_write_addr);
                noc.async_read_with_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                    UnicastEndpoint{},
                    dst,
                    stick_size_bytes,
                    {.noc_x = (uint32_t)my_x[noc.get_noc_id()],
                     .noc_y = (uint32_t)my_y[noc.get_noc_id()],
                     .addr = src_addr},
                    {.offset_bytes = 0});
                l1_write_addr += l1_write_offset_bytes;
                src_addr += stick_size_bytes;
            }
            noc.async_read_barrier();
            cb_dst.push_back(Wt);
        }
    }
}
