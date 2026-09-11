// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// NOTE: This is the Metal 2.0 fork of embeddings_common.hpp, which lives beside it. Embedding readers
// on Metal 2.0 include this fork; the original serves the readers still on the legacy API. Until the
// last of them migrates and the original is retired, changes here likely belong there too.
//
// prepare_local_cache below takes the local weight cache as a binding token and the pad token by
// value, both of which follow from its callers being on named bindings and named arguments: there is
// no buffer index to hand it, and no positional runtime argument for it to index into. It has two
// overloads for the cache: a DataflowBuffer token (embeddings.cpp, embedding_ind_tilized.cpp, whose
// factories still declare the cache as a reader self-loop DFB) and a Scratchpad token
// (embeddings_tilize.cpp, whose factory declares it as the reader-private scratchpad it really is --
// a DM self-loop DFB is rejected on Gen2/Quasar). Both leave the same three addresses behind for
// read_token_async.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/scratchpad.h"
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

// Fills the caller's local weight cache with the weight sticks that read_token_async serves out of
// local SRAM instead of fetching per token: the pad row under PADDED, rows 0 and 1 under BINARY. The cache is
// reserved and written but never committed, because nothing downstream drains it — the reads below
// address it by the saved write pointer.
//
// `pad_token_value` is the token that PADDED treats as padding. The caller supplies it because the
// two readers take it from different arguments.
//
// `weight_col_offset_bytes` is the core's column slice of the weight row (non-zero only for
// width-/block-sharded output); it is applied per read so that `weights` can be built on the clean
// buffer base.
template <typename T>
FORCE_INLINE constexpr void prepare_local_cache(
    const Noc& noc,
    DFBBindingToken local_cache,
    const T& weights,
    uint32_t weight_stick_size,
    uint32_t pad_token_value = 0,
    uint32_t weight_col_offset_bytes = 0) {
#if defined PADDED
    pad_token = pad_token_value;
    DataflowBuffer dfb(local_cache);
    dfb.reserve_back(1);
    pad_local_addr = dfb.get_write_ptr();
    noc.async_read(
        weights,
        CoreLocalMem<uint32_t>(pad_local_addr),
        weight_stick_size,
        {.page_id = pad_token, .offset_bytes = weight_col_offset_bytes},
        {});
    noc.async_read_barrier();
#elif defined BINARY
    DataflowBuffer dfb(local_cache);
    dfb.reserve_back(2);
    zero_local_addr = dfb.get_write_ptr();
    noc.async_read(
        weights,
        CoreLocalMem<uint32_t>(zero_local_addr),
        weight_stick_size,
        {.page_id = 0, .offset_bytes = weight_col_offset_bytes},
        {});

    one_local_addr = zero_local_addr + weight_stick_size;
    noc.async_read(
        weights,
        CoreLocalMem<uint32_t>(one_local_addr),
        weight_stick_size,
        {.page_id = 1, .offset_bytes = weight_col_offset_bytes},
        {});

    noc.async_read_barrier();
#endif
}

// Scratchpad overload: same fill, with the cache a reader-private scratchpad. The saved addresses are
// the scratchpad's base (PADDED: the pad row; BINARY: row 0) and, for BINARY, base + weight_stick_size
// (row 1), exactly where the DFB overload placed them relative to its write pointer.
template <typename T>
FORCE_INLINE constexpr void prepare_local_cache(
    const Noc& noc,
    const ScratchpadBindingToken& local_cache,
    const T& weights,
    uint32_t weight_stick_size,
    uint32_t pad_token_value = 0,
    uint32_t weight_col_offset_bytes = 0) {
#if defined PADDED
    pad_token = pad_token_value;
    Scratchpad<uint32_t> cache(local_cache);
    pad_local_addr = cache.get_base_address();
    noc.async_read(
        weights,
        cache,
        weight_stick_size,
        {.page_id = pad_token, .offset_bytes = weight_col_offset_bytes},
        {.offset_bytes = 0});
    noc.async_read_barrier();
#elif defined BINARY
    Scratchpad<uint32_t> cache(local_cache);
    zero_local_addr = cache.get_base_address();
    noc.async_read(
        weights,
        cache,
        weight_stick_size,
        {.page_id = 0, .offset_bytes = weight_col_offset_bytes},
        {.offset_bytes = 0});

    one_local_addr = zero_local_addr + weight_stick_size;
    noc.async_read(
        weights,
        cache,
        weight_stick_size,
        {.page_id = 1, .offset_bytes = weight_col_offset_bytes},
        {.offset_bytes = weight_stick_size});

    noc.async_read_barrier();
#endif
}

// Issues an async read of one token's weight stick (or a chunk of it) into the destination L1
// address. Caller must barrier before use.
//
// `chunk_offset_bytes` selects the chunk within the (already column-sliced) stick and applies to
// both the DRAM read and the local-cache replay. `weight_col_offset_bytes` is the core's column
// slice of the weight row; it applies to the DRAM read only, because prepare_local_cache already
// filled the local cache from that offset. Passing it here (rather than folding it into the
// accessor's base address) keeps `weights` on the clean buffer base.
template <typename T>
FORCE_INLINE void read_token_async(
    const Noc& noc,
    input_token_t token,
    const T& weights,
    uint32_t dst_l1_addr,
    uint32_t size_bytes,
    uint32_t chunk_offset_bytes = 0,
    uint32_t weight_col_offset_bytes = 0) {
    const uint32_t weight_offset_bytes = weight_col_offset_bytes + chunk_offset_bytes;
#if defined PADDED
    if (token == pad_token) {
        const uint8_t noc_id = noc.get_noc_id();
        UnicastEndpoint src;
        noc.async_read(
            src,
            CoreLocalMem<uint32_t>(dst_l1_addr),
            size_bytes,
            {.noc_x = my_x[noc_id], .noc_y = my_y[noc_id], .addr = pad_local_addr + chunk_offset_bytes},
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
        {.noc_x = my_x[noc_id], .noc_y = my_y[noc_id], .addr = local_addr + chunk_offset_bytes},
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
