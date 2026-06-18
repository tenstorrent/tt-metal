// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 port (in place — used only by the TransposeWHShardedRM factory, and only launched when
// Ht > 8). Logic UNCHANGED; only the access mechanism moves to named bindings:
//   - compute-output staging CB c_27 -> dfb::out_stage (scratch; read by base pointer; consumer)
//   - output shard CB c_16           -> dfb::out       (borrowed output shard; written by base pointer)
//   - positional CTAs                -> get_arg(args::...)
// The writer is bound (and this source compiled) only on the Ht > 8 path, where both DFBs exist; the
// `if constexpr (Ht > 8)` guard is preserved verbatim.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
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

    constexpr auto cb_out = dfb::out_stage;
    constexpr auto cb_out0 = dfb::out;

    const uint32_t stick_size_bytes = H_size_bytes;

    Noc noc;
    DataflowBuffer cb_src(cb_out);
    DataflowBuffer cb_dst(cb_out0);

    uint32_t dst_addr = cb_dst.get_write_ptr();

    // temporary fix until pack_untilze is fully fixed
    if constexpr (Ht > 8) {
        noc.set_async_write_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
            UnicastEndpoint{},
            stick_size_bytes,
            {.noc_x = (uint32_t)my_x[noc.get_noc_id()], .noc_y = (uint32_t)my_y[noc.get_noc_id()], .addr = dst_addr});

        for (uint32_t n = 0; n < num_hw_blocks_per_core; n++) {
            for (uint32_t w = 0; w < Wt; ++w) {
                cb_src.wait_front(Ht);
                uint32_t l1_read_addr = cb_src.get_read_ptr();
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
                cb_src.pop_front(Ht);
            }
        }
        noc.async_write_barrier();
    }
}
