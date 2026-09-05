// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#if !defined(ARCH_QUASAR)
// ckernel::load_blocking (the store drain below) is WH/BH only: Quasar's ckernel.h has no load_blocking,
// and from a data-movement build it #errors unless COMPILE_FOR_TRISC is defined. The Quasar branch uses a
// plain volatile load instead. Same guard as data_movement/common/kernels/common.hpp.
#include "ckernel.h"
#endif

void kernel_main() {
    const auto num_unpadded_W = get_arg(args::num_unpadded_W);
    const auto num_padded_Wt = get_arg(args::num_padded_Wt);
    const auto num_unpadded_Z = get_arg(args::num_unpadded_Z);
    const auto num_padded_Zt = get_arg(args::num_padded_Zt);
    const auto num_unpadded_Yt = get_arg(args::num_unpadded_Yt);
    const auto num_padded_Yt = get_arg(args::num_padded_Yt);
    const auto num_unpadded_Xt = get_arg(args::num_unpadded_Xt);
    const auto num_padded_Xt = get_arg(args::num_padded_Xt);
    const auto pad_value = get_arg(args::pad_value);

    const auto s1 = TensorAccessor(tensor::dst);
    Noc noc;
    DataflowBuffer dfb_out0(dfb::out0);
    DataflowBuffer dfb_pad(dfb::pad);

    const uint32_t tile_size = dfb_out0.get_tile_size();

    dfb_pad.reserve_back(1);  // in this kernel we are not pushing anything into DFBs, just using the space

    uint32_t pad_buffer_l1_addr = dfb_pad.get_write_ptr();

    // Fill pad tile with pad value
    volatile tt_l1_ptr uint32_t* pad_buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pad_buffer_l1_addr);
    const uint32_t num_elems = tile_size / sizeof(uint32_t);
    for (uint32_t z = 0; z < num_elems; z++) {
        pad_buffer[z] = pad_value;
    }
    // The fill above is baby-RISCV stores and pad_tiles() below hands this buffer to the NoC as the
    // source of every pad-tile write. A store can retire before its write-request lands in L1, and the
    // RISCV core and the NoC are different L1 clients with no program-order guarantee between them
    // (WormholeB0/TensixTile/BabyRISCV/MemoryOrdering.md). Read back the last filled word so the fill
    // is processed before the first NoC write is issued. Same construction as #50374.
#if defined(ARCH_QUASAR)
    // Quasar has no ckernel::load_blocking; a volatile load is the local ordering barrier there.
    (void)*(pad_buffer + num_elems - 1);
#else
    (void)ckernel::load_blocking(pad_buffer + num_elems - 1);
#endif

    uint32_t src_tile_id = 0;
    uint32_t dst_tile_id = 0;

    CoreLocalMem<uint32_t> pad_src(pad_buffer_l1_addr);
    auto pad_tiles = [&](uint32_t num_tiles) {
        for (uint32_t pad_tile = 0; pad_tile < num_tiles; pad_tile++) {
            noc.async_write(pad_src, s1, tile_size, {.offset_bytes = 0}, {.page_id = dst_tile_id, .offset_bytes = 0});
            dst_tile_id++;
        }
        noc.async_write_barrier();
    };

    for (uint32_t w = 0; w < num_unpadded_W; w++) {
        for (uint32_t z = 0; z < num_unpadded_Z; z++) {
            for (uint32_t yt = 0; yt < num_unpadded_Yt; yt++) {
                for (uint32_t xt = 0; xt < num_unpadded_Xt; xt++) {
                    dfb_out0.wait_front(1);
                    noc.async_write(
                        dfb_out0, s1, tile_size, {.offset_bytes = 0}, {.page_id = dst_tile_id, .offset_bytes = 0});
                    noc.async_write_barrier();
                    dfb_out0.pop_front(1);
                    dst_tile_id++;
                }
                pad_tiles(num_padded_Xt);
            }
            pad_tiles(num_padded_Yt);
        }
        pad_tiles(num_padded_Zt);
    }
    pad_tiles(num_padded_Wt);
}
