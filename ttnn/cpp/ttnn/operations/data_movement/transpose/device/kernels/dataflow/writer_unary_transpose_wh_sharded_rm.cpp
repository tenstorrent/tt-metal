// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"
#include "experimental/idma_copy_rows.h"

// The NoC-with-state includes hide get_arg from ADL on the args:: accessors.
using experimental::get_arg;

// Built only when compute stages its output (Ht > 8, or always on Quasar): copies the valid H datums of
// each padded staged row into the dense output shard.
void kernel_main() {
    constexpr uint32_t num_hw_blocks_per_core = get_arg(args::num_hw_blocks_per_core);
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t W_per_tile = get_arg(args::W_per_tile);
    constexpr uint32_t W_per_tile_last = get_arg(args::W_per_tile_last);
    constexpr uint32_t H_size_bytes = get_arg(args::H_size_bytes);
    constexpr uint32_t l1_read_offset_bytes = get_arg(args::l1_read_offset_bytes);

    const uint32_t stick_size_bytes = H_size_bytes;

    DataflowBuffer dfb_src(dfb::cb_src);

    // The output shard is resident in this core's L1.
    const auto s = TensorAccessor(tensor::output);
    uint32_t dst_addr = noc_address_backend::extract_local_address(s.get_noc_addr(0));

#ifdef ARCH_QUASAR
    for (uint32_t n = 0; n < num_hw_blocks_per_core; n++) {
        for (uint32_t w = 0; w < Wt; ++w) {
            dfb_src.wait_front(Ht);
            uint32_t W_curr = w == Wt - 1 ? W_per_tile_last : W_per_tile;
            experimental::idma_copy_rows(
                dfb_src.get_read_ptr(), l1_read_offset_bytes, dst_addr, stick_size_bytes, W_curr);
            dst_addr += W_curr * stick_size_bytes;
            dfb_src.pop_front(Ht);
        }
    }
#else
    Noc noc;
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
