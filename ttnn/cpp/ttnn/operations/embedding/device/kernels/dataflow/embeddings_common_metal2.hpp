// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// NOTE: This is the Metal 2.0 fork of embeddings_common.hpp, which lives beside it. Embedding readers
// on Metal 2.0 include this fork; the original serves the readers still on the legacy API. Until the
// last of them migrates and the original is retired, changes here likely belong there too.
//
// prepare_local_cache below takes the local weight cache as a Scratchpad binding token and the pad
// token by value, both of which follow from its callers being on named bindings and named arguments:
// there is no buffer index to hand it, and no positional runtime argument for it to index into.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/scratchpad.h"
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
// Element pointers into the local weight cache, set by prepare_local_cache.
uint32_t* pad_local_ptr;
uint32_t* zero_local_ptr;
uint32_t* one_local_ptr;

// Fills the caller's local weight cache with the weight sticks that read_token_async serves out of
// local SRAM instead of fetching per token: the pad row under PADDED, rows 0 and 1 under BINARY. The cache is
// the reader's private scratchpad; read_token_async replays it through the saved element pointers.
//
// `pad_token_value` is the token that PADDED treats as padding. The caller supplies it because the
// two readers take it from different arguments.
//
// `weight_col_offset_bytes` is the core's column slice of the weight row (non-zero only for
// width-/block-sharded output); it is applied per read so that `weights` can be built on the clean
// buffer base.
// `local_cache` is a Scratchpad binding token. The cache is only ever NoC-accessed — filled here and
// replayed via a NoC loopback read in read_token_async — so the scratchpad's plain (cached) L1 view
// is correct and the RISC never reads it, so no scoped_lock is needed.
template <typename CacheToken, typename T>
FORCE_INLINE void prepare_local_cache(
    const Noc& noc,
    CacheToken local_cache,
    const T& weights,
    uint32_t weight_stick_size,
    uint32_t pad_token_value = 0,
    uint32_t weight_col_offset_bytes = 0) {
#if defined PADDED
    pad_token = pad_token_value;
    Scratchpad<uint32_t> cache(local_cache);
    pad_local_ptr = &cache[0];
    noc.async_read(
        weights, cache, weight_stick_size, {.page_id = pad_token, .offset_bytes = weight_col_offset_bytes}, {});
    noc.async_read_barrier();
#elif defined BINARY
    Scratchpad<uint32_t> cache(local_cache);
    zero_local_ptr = &cache[0];
    noc.async_read(weights, cache, weight_stick_size, {.page_id = 0, .offset_bytes = weight_col_offset_bytes}, {});

    // Row 1 is a NoC read destination, so weight_stick_size is NoC-aligned and divides exactly.
    one_local_ptr = &cache[weight_stick_size / sizeof(uint32_t)];
    noc.async_read(
        weights,
        cache,
        weight_stick_size,
        {.page_id = 1, .offset_bytes = weight_col_offset_bytes},
        {.offset_bytes = weight_stick_size});

    noc.async_read_barrier();
#endif
}

// Issues an async read of one token's weight stick (or a chunk of it) into the local destination
// `dst`. Caller must barrier before use.
//
// `chunk_offset_bytes` selects the chunk within the (already column-sliced) stick and applies to
// both the DRAM read and the local-cache replay. `weight_col_offset_bytes` is the core's column
// slice of the weight row; it applies to the DRAM read only, because prepare_local_cache already
// filled the local cache from that offset. Passing it here (rather than folding it into the
// accessor's base address) keeps `weights` on the clean buffer base.

// Turn a bfloat16 index into a row id.
FORCE_INLINE uint32_t decoded_token_id(input_token_t token) {
#if defined BFP16
    union {
        float f;
        uint32_t u;
    } converted;
    converted.u = static_cast<uint32_t>(token) << 16;
    return static_cast<uint32_t>(converted.f);
#else
    return static_cast<uint32_t>(token);
#endif
}

template <typename T, typename D>
FORCE_INLINE void read_token_async(
    const Noc& noc,
    input_token_t token,
    const T& weights,
    CoreLocalMem<D> dst,
    uint32_t size_bytes,
    uint32_t chunk_offset_bytes = 0,
    uint32_t weight_col_offset_bytes = 0) {
    const uint32_t weight_offset_bytes = weight_col_offset_bytes + chunk_offset_bytes;
    const uint32_t token_id = decoded_token_id(token);
#if defined PADDED
    if (token_id == pad_token) {
        // NoC loopback read out of this core's local cache.
        noc.async_read(
            CoreLocalMem<uint32_t>(pad_local_ptr), dst, size_bytes, {.offset_bytes = chunk_offset_bytes}, {});
        return;
    }
    noc.async_read(weights, dst, size_bytes, {.page_id = token_id, .offset_bytes = weight_offset_bytes}, {});
#elif defined BINARY
    // NoC loopback read out of this core's local cache.
    const CoreLocalMem<uint32_t> local_src((token_id == 0) ? zero_local_ptr : one_local_ptr);
    noc.async_read(local_src, dst, size_bytes, {.offset_bytes = chunk_offset_bytes}, {});
#else
    noc.async_read(weights, dst, size_bytes, {.page_id = token_id, .offset_bytes = weight_offset_bytes}, {});
#endif
}

// Raw L1 address form of the above, for destinations that are a DFB write pointer.
template <typename T>
FORCE_INLINE void read_token_async(
    const Noc& noc,
    input_token_t token,
    const T& weights,
    uint32_t dst_l1_addr,
    uint32_t size_bytes,
    uint32_t chunk_offset_bytes = 0,
    uint32_t weight_col_offset_bytes = 0) {
    read_token_async(
        noc,
        token,
        weights,
        CoreLocalMem<uint32_t>(dst_l1_addr),
        size_bytes,
        chunk_offset_bytes,
        weight_col_offset_bytes);
}
