// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// packed_dft_bf16_reader.cpp — BRISC0 / reader for the TRUE-bf16 packed
// direct-DFT kernel. Same dataflow as the fp32 reader (see

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "packed_dft_bf16_common.h"

FORCE_INLINE void push_pair(
    uint32_t a_tile_idx,
    uint32_t b_tile_idx,
    const InterleavedAddrGenFast<true>& a_gen,
    const InterleavedAddrGenFast<true>& b_gen) {
    cb_reserve_back(CB_A, 1);
    cb_reserve_back(CB_B, 1);
    noc_async_read_tile(a_tile_idx, a_gen, get_write_ptr(CB_A));
    noc_async_read_tile(b_tile_idx, b_gen, get_write_ptr(CB_B));
    noc_async_read_barrier();
    cb_push_back(CB_A, 1);
    cb_push_back(CB_B, 1);
}

void kernel_main() {
    const uint32_t in_r_addr       = get_arg_val<uint32_t>(0);
    const uint32_t in_i_addr       = get_arg_val<uint32_t>(1);
    const uint32_t tw_r_addr       = get_arg_val<uint32_t>(2);
    const uint32_t tw_i_addr       = get_arg_val<uint32_t>(3);
    const uint32_t tw_i_neg_addr   = get_arg_val<uint32_t>(4);
    const uint32_t base_tile_idx   = get_arg_val<uint32_t>(5);
    const uint32_t tiles_per_core  = get_arg_val<uint32_t>(6);

    const DataFormat df = get_dataformat(CB_A);
    const uint32_t   ts = get_tile_size(CB_A);

    InterleavedAddrGenFast<true> in_r_gen = {
        .bank_base_address = in_r_addr,     .page_size = ts, .data_format = df};
    InterleavedAddrGenFast<true> in_i_gen = {
        .bank_base_address = in_i_addr,     .page_size = ts, .data_format = df};
    InterleavedAddrGenFast<true> tw_r_gen = {
        .bank_base_address = tw_r_addr,     .page_size = ts, .data_format = df};
    InterleavedAddrGenFast<true> tw_i_gen = {
        .bank_base_address = tw_i_addr,     .page_size = ts, .data_format = df};
    InterleavedAddrGenFast<true> tw_i_neg_gen = {
        .bank_base_address = tw_i_neg_addr, .page_size = ts, .data_format = df};

    constexpr uint32_t kTwiddleTile = 0;

    for (uint32_t k = 0; k < tiles_per_core; ++k) {
        const uint32_t t = base_tile_idx + k;

        // out_R = in_R · T_R + in_I · (-T_I)
        push_pair(t, kTwiddleTile, in_r_gen, tw_r_gen);
        push_pair(t, kTwiddleTile, in_i_gen, tw_i_neg_gen);

        // out_I = in_R · T_I + in_I · T_R
        push_pair(t, kTwiddleTile, in_r_gen, tw_i_gen);
        push_pair(t, kTwiddleTile, in_i_gen, tw_r_gen);
    }
}
