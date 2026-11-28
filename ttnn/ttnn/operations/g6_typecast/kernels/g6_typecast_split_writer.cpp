// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_typecast TILE sub-tile split writer (Perf 1, perf_experiments/subtile_split; BRISC / NoC1): writes the NF faces of
// the (tiny-tile) output CB page to their true place in the 32x32 output tile `tile`, faces [first_face, first_face +
// NF).
//  - plain formats: ONE write of NF * face bytes to byte offset first_face * face.
//  - bfloat8_b: the tiny tile is [NF * 16 exponents, padded to exp_sec][NF * 256 mantissa]; the full tile is
//    [64 exponents (16 per face)][4 x 256 mantissa] -> TWO writes: exponents to 16 * first_face, mantissas to
//    64 + 256 * first_face.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "g6_typecast_zones.hpp"

void kernel_main() {
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t out_page_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t hdr = get_compile_time_arg_val(2);
    constexpr uint32_t face = get_compile_time_arg_val(3);
    constexpr uint32_t NF = get_compile_time_arg_val(4);
    constexpr uint32_t exp_sec = get_compile_time_arg_val(5);
    constexpr auto output_args = TensorAccessorArgs<6>();

    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t tile = get_arg_val<uint32_t>(0);
    const uint32_t first_face = get_arg_val<uint32_t>(1);

    const auto acc = TensorAccessor(output_args, output_addr, out_page_bytes);
    {
        G6_ZONE("WRITE_WAIT_DATA");
        cb_wait_front(cb_output_tiles, 1);
    }
    G6_ZONE("WRITE_UNIT");
    const uint32_t l1 = get_read_ptr(cb_output_tiles);
    if constexpr (hdr != 0) {
        noc_async_write_one_packet(l1, acc.get_noc_addr(tile, 16 * first_face), 16 * NF);
        noc_async_write_one_packet(l1 + exp_sec, acc.get_noc_addr(tile, hdr + face * first_face), face * NF);
    } else {
        noc_async_write_one_packet(l1, acc.get_noc_addr(tile, face * first_face), face * NF);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_output_tiles, 1);
}
