// SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

// Specialized q/k reader for nlp_create_qkv_heads_rope (prefill RoPE convention, Ht == 1, num_rows == 1).
//
// Functionally identical to the generic rotary_embedding interleaved reader, but batches every NoC read
// behind a SINGLE barrier instead of one barrier per tile. The generic reader issues
// read(rotated) -> barrier -> read(sin) -> barrier -> read(in) -> barrier -> read(cos) -> barrier per
// tile, i.e. ~4*Wt fully serialized DRAM round-trips. Here all 4*Wt reads are issued async and overlap,
// gated by one barrier -- the per-tile barrier serialization is this op's dominant cost at small Wt.
//
// CB push order matches the compute kernel's per-tile FIFO consumption: rotated_in / in / cos / sin each
// receive Wt tiles in natural slot order (rotated in the half-swapped order the rotate-half expects).
void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t cos_addr = get_arg_val<uint32_t>(1);
    const uint32_t sin_addr = get_arg_val<uint32_t>(2);
    // arg 3 (num_rows) is always 1 for this op (Ht == 1); arg 5 (start_row_id) unused here.
    const uint32_t start_id = get_arg_val<uint32_t>(4);
    const uint32_t cos_sin_start_id = get_arg_val<uint32_t>(6);

    constexpr uint32_t input_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t rotated_input_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t cos_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t sin_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t scalar_cb_id = get_compile_time_arg_val(4);
    constexpr uint16_t scalar_value = get_compile_time_arg_val(5);
    // arg 6 (Ht) == 1, arg 8 (HtWt) == Wt: unused (num_rows == 1).
    constexpr uint32_t Wt = get_compile_time_arg_val(7);
    constexpr uint32_t half_Wt = get_compile_time_arg_val(9);
    constexpr auto src_args = TensorAccessorArgs<10>();
    constexpr auto cos_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();

    const auto s0 = TensorAccessor(src_args, src_addr);
    const auto s1 = TensorAccessor(cos_args, cos_addr);
    const auto s2 = TensorAccessor(sin_args, sin_addr);

    const uint32_t in_tile_bytes = get_tile_size(input_cb_id);
    const uint32_t rot_tile_bytes = get_tile_size(rotated_input_cb_id);
    const uint32_t cos_tile_bytes = get_tile_size(cos_cb_id);
    const uint32_t sin_tile_bytes = get_tile_size(sin_cb_id);

    // Scalar tile (the -1 used by rotate-half) -- a single packed value, no NoC read.
    constexpr uint32_t onetile = 1;
    cb_reserve_back(scalar_cb_id, onetile);
    volatile tt_l1_ptr uint16_t* scalar_buffer =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(scalar_cb_id));
    scalar_buffer[0] = scalar_value;
    cb_push_back(scalar_cb_id, onetile);

    // Reserve all four CBs up front, then issue every tile read async and barrier ONCE.
    cb_reserve_back(input_cb_id, Wt);
    cb_reserve_back(rotated_input_cb_id, Wt);
    cb_reserve_back(cos_cb_id, Wt);
    cb_reserve_back(sin_cb_id, Wt);
    uint32_t in_l1 = get_write_ptr(input_cb_id);
    uint32_t rot_l1 = get_write_ptr(rotated_input_cb_id);
    uint32_t cos_l1 = get_write_ptr(cos_cb_id);
    uint32_t sin_l1 = get_write_ptr(sin_cb_id);

    for (uint32_t j = 0; j < Wt; ++j) {
        // input slot j = src page (start_id + j); rotated slot j = src page with the two head-dim halves
        // swapped (start_id + (j + half_Wt) mod Wt) -- the rotate-half tile order (half_Wt is tile-aligned).
        const uint32_t rot_j = (j < half_Wt) ? (j + half_Wt) : (j - half_Wt);
        noc_async_read_tile(start_id + j, s0, in_l1 + j * in_tile_bytes);
        noc_async_read_tile(start_id + rot_j, s0, rot_l1 + j * rot_tile_bytes);
        noc_async_read_tile(cos_sin_start_id + j, s1, cos_l1 + j * cos_tile_bytes);
        noc_async_read_tile(cos_sin_start_id + j, s2, sin_l1 + j * sin_tile_bytes);
    }
    noc_async_read_barrier();

    cb_push_back(input_cb_id, Wt);
    cb_push_back(rotated_input_cb_id, Wt);
    cb_push_back(cos_cb_id, Wt);
    cb_push_back(sin_cb_id, Wt);
}
