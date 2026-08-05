// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

// TODO: Should get this from somewhere
constexpr uint32_t tile_height = 32;

#if defined BFP16
using input_token_t = uint16_t;
#else
using input_token_t = uint32_t;
#endif

// TODO: Can probably make this not global
uint32_t pad_token;
uint32_t pad_local_addr;
uint32_t zero_local_addr;
uint32_t one_local_addr;

template <typename T>
FORCE_INLINE constexpr void prepare_local_cache(
    const Noc& noc,
    uint32_t local_cache_cb,
    const T& weights,
    uint32_t weight_stick_size,
    uint32_t pad_token_arg_idx = 0) {
#if defined PADDED
    pad_token = get_arg_val<uint32_t>(pad_token_arg_idx);
    CircularBuffer cb(local_cache_cb);
    cb.reserve_back(1);
    pad_local_addr = cb.get_write_ptr();
    noc.async_read(weights, CoreLocalMem<uint32_t>(pad_local_addr), weight_stick_size, {.page_id = pad_token}, {});
    noc.async_read_barrier();
#elif defined BINARY
    CircularBuffer cb(local_cache_cb);
    cb.reserve_back(2);
    zero_local_addr = cb.get_write_ptr();
    noc.async_read(weights, CoreLocalMem<uint32_t>(zero_local_addr), weight_stick_size, {.page_id = 0}, {});

    one_local_addr = zero_local_addr + weight_stick_size;
    noc.async_read(weights, CoreLocalMem<uint32_t>(one_local_addr), weight_stick_size, {.page_id = 1}, {});

    noc.async_read_barrier();
#endif
}

// Issues an async read of one token's weight stick (or a chunk of it) into the destination L1
// address. Caller must barrier before use.
template <typename T>
FORCE_INLINE void read_token_async(
    const Noc& noc,
    input_token_t token,
    const T& weights,
    uint32_t dst_l1_addr,
    uint32_t size_bytes,
    uint32_t weight_offset_bytes = 0) {
#if defined PADDED
    if (token == pad_token) {
        const uint8_t noc_id = noc.get_noc_id();
        UnicastEndpoint src;
        noc.async_read(
            src,
            CoreLocalMem<uint32_t>(dst_l1_addr),
            size_bytes,
            {.noc_x = my_x[noc_id], .noc_y = my_y[noc_id], .addr = pad_local_addr + weight_offset_bytes},
            {});
        return;
    }
    noc.async_read(
        weights,
        CoreLocalMem<uint32_t>(dst_l1_addr),
        size_bytes,
        {.page_id = static_cast<uint32_t>(token), .offset_bytes = weight_offset_bytes},
        {});
#elif defined BINARY
    const uint8_t noc_id = noc.get_noc_id();
    UnicastEndpoint src;
    const uint32_t local_addr = (token == 0) ? zero_local_addr : one_local_addr;
    noc.async_read(
        src,
        CoreLocalMem<uint32_t>(dst_l1_addr),
        size_bytes,
        {.noc_x = my_x[noc_id], .noc_y = my_y[noc_id], .addr = local_addr + weight_offset_bytes},
        {});
#elif defined BFP16
    union {
        float f;
        uint32_t u;
    } u;
    u.u = static_cast<uint32_t>(token) << 16;
    uint32_t token_casted = static_cast<uint32_t>(u.f);
    noc.async_read(
        weights,
        CoreLocalMem<uint32_t>(dst_l1_addr),
        size_bytes,
        {.page_id = token_casted, .offset_bytes = weight_offset_bytes},
        {});
#else
    noc.async_read(
        weights,
        CoreLocalMem<uint32_t>(dst_l1_addr),
        size_bytes,
        {.page_id = static_cast<uint32_t>(token), .offset_bytes = weight_offset_bytes},
        {});
#endif
}
