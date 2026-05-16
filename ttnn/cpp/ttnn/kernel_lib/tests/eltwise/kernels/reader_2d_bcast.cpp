// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Reader for the 2D `eltwise_chain` bcast validation kernel.
// Pushes `n_a_tiles` tiles from src0_addr into cb_a (c_0) and
// `n_b_tiles` tiles from src1_addr into cb_b (c_1) — independently sized so the
// compute kernel can wait on the per-mode broadcast window (Ht*Wt for A,
// Ht / Wt / 1 for B depending on RowBcast/ColBcast/Scalar).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t n_a_tiles = get_arg_val<uint32_t>(2);
    uint32_t n_b_tiles = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;
    constexpr auto src0_args = TensorAccessorArgs<0>();
    constexpr auto src1_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();

    experimental::Noc noc;
    experimental::CircularBuffer cb0(cb_id_in0);
    experimental::CircularBuffer cb1(cb_id_in1);

    uint32_t src0_tile_bytes = get_tile_size(cb_id_in0);
    uint32_t src1_tile_bytes = get_tile_size(cb_id_in1);
    const auto s0 = TensorAccessor(src0_args, src0_addr);
    const auto s1 = TensorAccessor(src1_args, src1_addr);

    constexpr uint32_t onetile = 1;

    for (uint32_t tile_id = 0; tile_id < n_a_tiles; ++tile_id) {
        cb0.reserve_back(onetile);
        noc.async_read(s0, cb0, src0_tile_bytes, {.page_id = tile_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb0.push_back(onetile);
    }
    for (uint32_t tile_id = 0; tile_id < n_b_tiles; ++tile_id) {
        cb1.reserve_back(onetile);
        noc.async_read(s1, cb1, src1_tile_bytes, {.page_id = tile_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb1.push_back(onetile);
    }
}
