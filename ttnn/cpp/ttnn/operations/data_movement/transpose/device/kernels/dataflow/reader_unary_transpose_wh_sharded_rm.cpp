// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

// The NoC-with-state includes hide get_arg from ADL on the args:: accessors.
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

    DataflowBuffer dfb_dst(dfb::cb_dst);

    // The input shard is resident in this core's L1.
    const auto s = TensorAccessor(tensor::input);
    uint32_t src_addr = noc_address_backend::extract_local_address(s.get_noc_addr(0));

#ifdef ARCH_QUASAR
    // Quasar: a NoC read addressed to this core's own coordinates does not complete, so copy locally.
    for (uint32_t n = 0; n < num_hw_blocks_per_core; n++) {
        for (uint32_t h = 0; h < Ht; ++h) {
            dfb_dst.reserve_back(Wt);
            uint32_t H_curr = h == Ht - 1 ? H_per_tile_last : H_per_tile;
            uint32_t l1_write_addr = dfb_dst.get_write_ptr();
            for (uint32_t h_datum = 0; h_datum < H_curr; ++h_datum) {
                auto src = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(src_addr);
                auto dst = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(l1_write_addr);
                for (uint32_t i = 0; i < stick_size_bytes; ++i) {
                    dst[i] = src[i];
                }
                l1_write_addr += l1_write_offset_bytes;
                src_addr += stick_size_bytes;
            }
            dfb_dst.push_back(Wt);
        }
    }
#else
    Noc noc;

    noc.set_async_read_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
        UnicastEndpoint{},
        stick_size_bytes,
        {.noc_x = (uint32_t)my_x[noc.get_noc_id()], .noc_y = (uint32_t)my_y[noc.get_noc_id()], .addr = src_addr});

    for (uint32_t n = 0; n < num_hw_blocks_per_core; n++) {
        for (uint32_t h = 0; h < Ht; ++h) {
            dfb_dst.reserve_back(Wt);
            uint32_t l1_write_addr = dfb_dst.get_write_ptr();
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
            dfb_dst.push_back(Wt);
        }
    }
#endif
}
