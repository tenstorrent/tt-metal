// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 conversion (in place; this kernel is transpose-owned). Resource access uses the Metal 2.0
// named handles (dfb::/args::). The input shard is resident in THIS core's L1, read by base L1 address
// from tensor::input (a local TensorAccessor); cb_dst (dfb::cb_dst) is the row-gathered tile-staging
// buffer feeding the compute kernel.
//
// QSR: the input shard is read with a plain CPU LOCAL COPY, not a self/loopback NOC read. On Quasar a
// NOC read addressed to the core's own (my_x,my_y) never drains its async_read_barrier — the reader
// pins at RBW and fold deadlocks (and the bytes never land, so the transpose output reads all-zero).
// The byte-wise local copy mirrors the width-only pad reader (reader_pad_dims_rm_sharded_stickwise.cpp),
// the proven Quasar local-copy idiom.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

#define u8_l1_ptr volatile tt_l1_ptr uint8_t*

// get_arg lives in `namespace experimental` and is normally found via ADL on the args:: accessor
// type. The includes this kernel pulls in change the lookup context enough that ADL doesn't resolve it
// here, so bring the overload set in explicitly.
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

    DataflowBuffer cb_dst(dfb::cb_dst);

    // Local input-shard base L1 address from the resident input TensorAccessor (no borrowed self-loop CB).
    const auto s = TensorAccessor(tensor::input);
    uint32_t src_addr = (uint32_t)NOC_LOCAL_ADDR_OFFSET(s.get_noc_addr(0));

    for (uint32_t n = 0; n < num_hw_blocks_per_core; n++) {
        for (uint32_t h = 0; h < Ht; ++h) {
            cb_dst.reserve_back(Wt);
            uint32_t l1_write_addr = cb_dst.get_write_ptr();
            uint32_t H_curr = h == Ht - 1 ? H_per_tile_last : H_per_tile;
            for (uint32_t h_datum = 0; h_datum < H_curr; ++h_datum) {
                // Local L1->L1 copy of one input stick into the staging buffer (no NOC; see file header).
                auto src = reinterpret_cast<u8_l1_ptr>(src_addr);
                auto dst = reinterpret_cast<u8_l1_ptr>(l1_write_addr);
                for (uint32_t i = 0; i < stick_size_bytes; ++i) {
                    dst[i] = src[i];
                }
                l1_write_addr += l1_write_offset_bytes;
                src_addr += stick_size_bytes;
            }
            cb_dst.push_back(Wt);
        }
    }
}
