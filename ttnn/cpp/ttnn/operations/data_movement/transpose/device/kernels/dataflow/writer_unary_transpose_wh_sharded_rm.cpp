// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_hw_blocks_per_core = get_arg(args::num_hw_blocks_per_core);
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t W_per_tile = get_arg(args::W_per_tile);
    constexpr uint32_t W_per_tile_last = get_arg(args::W_per_tile_last);
    constexpr uint32_t H_size_bytes = get_arg(args::H_size_bytes);
    constexpr uint32_t l1_read_offset_bytes = get_arg(args::l1_read_offset_bytes);

    const uint32_t stick_size_bytes = H_size_bytes;

    Noc noc;
#ifdef OUT_STAGING
    DataflowBuffer dfb_src(dfb::out_staging);  // transposed columns packed by the compute kernel
#endif
    DataflowBuffer dfb_dst(dfb::out);  // the resident output shard

    uint32_t dst_addr = dfb_dst.get_write_ptr();

    // temporary fix until pack_untilze is fully fixed: the host binds the staging buffer (and
    // defines OUT_STAGING) only when Ht > 8; otherwise the compute kernel packs straight into the
    // output shard and there is nothing for this kernel to move.
#ifdef OUT_STAGING
    noc.set_async_write_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
        UnicastEndpoint{},
        stick_size_bytes,
        {.noc_x = (uint32_t)my_x[noc.get_noc_id()], .noc_y = (uint32_t)my_y[noc.get_noc_id()], .addr = dst_addr});

    for (uint32_t n = 0; n < num_hw_blocks_per_core; n++) {
        for (uint32_t w = 0; w < Wt; ++w) {
            dfb_src.wait_front(Ht);
            uint32_t l1_read_addr = dfb_src.get_read_ptr();
            uint32_t W_curr = w == Wt - 1 ? W_per_tile_last : W_per_tile;
            for (uint32_t w_datum = 0; w_datum < W_curr; ++w_datum) {
                CoreLocalMem<uint32_t> src(l1_read_addr);
                noc.async_write_with_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                    src,
                    UnicastEndpoint{},
                    stick_size_bytes,
                    {.offset_bytes = 0},
                    {.noc_x = (uint32_t)my_x[noc.get_noc_id()],
                     .noc_y = (uint32_t)my_y[noc.get_noc_id()],
                     .addr = dst_addr});
                l1_read_addr += l1_read_offset_bytes;
                dst_addr += stick_size_bytes;
            }
            noc.async_writes_flushed();
            dfb_src.pop_front(Ht);
        }
    }
    noc.async_write_barrier();
#endif
}
