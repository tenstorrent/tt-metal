// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dataflow_api.h"

// TODO: Should get this from somewhere
constexpr uint32_t tile_height = 32;

#if defined BFP16
using input_token_t = uint16_t;
#else
using input_token_t = uint32_t;
#endif

// TODO: Can probably make this not global
uint32_t pad_token;
uint64_t pad_noc_addr;
uint64_t zero_noc_addr;
uint64_t one_noc_addr;

template <typename T>
FORCE_INLINE constexpr void prepare_local_cache(
    uint32_t local_cache_cb, const T& weights, uint32_t weight_stick_size, uint32_t pad_token_arg_idx = 0) {
#if defined PADDED
    pad_token = get_arg_val<uint32_t>(pad_token_arg_idx);
    cb_reserve_back(local_cache_cb, 1);
    uint32_t local_pad_addr = get_write_ptr(local_cache_cb);
    uint64_t src_noc_addr = get_noc_addr(pad_token, weights);
    noc_async_read(src_noc_addr, local_pad_addr, weight_stick_size);
    noc_async_read_barrier();
    pad_noc_addr = get_noc_addr(local_pad_addr);
#elif defined BINARY
    cb_reserve_back(local_cache_cb, 2);
    uint32_t local_write_addr = get_write_ptr(local_cache_cb);
    uint64_t src_noc_addr = get_noc_addr(0, weights);
    noc_async_read(src_noc_addr, local_write_addr, weight_stick_size);
    zero_noc_addr = get_noc_addr(local_write_addr);

    local_write_addr += weight_stick_size;
    src_noc_addr = get_noc_addr(1, weights);
    noc_async_read(src_noc_addr, local_write_addr, weight_stick_size);
    one_noc_addr = get_noc_addr(local_write_addr);

    noc_async_read_barrier();
#endif
}

template <typename T>
FORCE_INLINE uint64_t get_token_noc_addr(input_token_t token, const T& weights) {
#if defined PADDED
    if (token == pad_token) {
        return pad_noc_addr;
    } else {
        return get_noc_addr(token, weights);
    }
#elif defined BINARY
    if (token == 0) {
        return zero_noc_addr;
    } else {
        return one_noc_addr;
    }
#elif defined BFP16
    union {
        float f;
        uint32_t u;
    } u;
    u.u = (uint32_t)token << 16;
    uint32_t token_casted = static_cast<uint32_t>(u.f);
    return get_noc_addr(token_casted, weights);
#else
    return get_noc_addr(token, weights);
#endif
}
