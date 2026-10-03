// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/wavelet/common/signal_extension.hpp"
#define ALWI inline __attribute__((always_inline))

#if defined(ARCH_WORMHOLE)
#define LWT_CACHE_REFILL_CALLABLE __attribute__((noinline))
#define LWT_BOUNDARY_CALLABLE __attribute__((noinline))
#elif defined(ARCH_BLACKHOLE)
#define LWT_CACHE_REFILL_CALLABLE ALWI
#if defined(PROFILE_KERNEL) && PROFILE_KERNEL
#define LWT_BOUNDARY_CALLABLE __attribute__((noinline))
#else
#define LWT_BOUNDARY_CALLABLE ALWI
#endif
#else
#error "TTNN wavelet stick cache supports only Wormhole and Blackhole"
#endif

namespace ttnn::operations::wavelet::kernels::primitives {

constexpr uint32_t kInvalidStick = 0xFFFFFFFFU;

struct StickReadCache {
    uint32_t cb_id;
    uint32_t stick_nbytes;
    uint32_t stick_width;
    uint32_t stick_capacity;
    uint32_t cached_stick_id;
    uint32_t cached_stick_count;
    uint32_t reserved_stick_count;
    uint32_t source_page;
    bool valid;
    uint32_t page_size;
};

ALWI uint32_t min_u32(const uint32_t lhs, const uint32_t rhs) { return lhs < rhs ? lhs : rhs; }

ALWI bool cache_contains_stick(const StickReadCache& cache, const uint32_t source_stick) {
    return cache.valid && source_stick >= cache.cached_stick_id &&
           source_stick < cache.cached_stick_id + cache.cached_stick_count;
}

template <typename SrcAccessor>
LWT_CACHE_REFILL_CALLABLE void cache_source_sticks(
    const SrcAccessor& src,
    StickReadCache& cache,
    const uint32_t source_stick,
    const uint32_t source_stick_count,
    const uint32_t source_length) {
    CircularBuffer cache_buffer(cache.cb_id);
    Noc noc;

    if (cache.valid) {
        cache_buffer.pop_front(cache.reserved_stick_count);
    }

    const uint32_t available_sticks = source_stick_count - source_stick;
    const uint32_t cached_sticks = min_u32(cache.stick_capacity, available_sticks);
    const uint32_t reserve_sticks = cache.stick_capacity;

    cache_buffer.reserve_back(reserve_sticks);
    const uint32_t cache_l1_addr = cache_buffer.get_write_ptr();
    const bool page_per_stick = cache.page_size == cache.stick_nbytes;
#pragma GCC unroll 8
    for (uint32_t i = 0; i < reserve_sticks; ++i) {
        const uint32_t stick_index = source_stick + i;
        const bool is_cached_stick = i < cached_sticks;
        const bool is_partial_tail = is_cached_stick && !page_per_stick && stick_index + 1 == source_stick_count &&
                                     source_length % cache.stick_width != 0;
        const uint32_t read_nbytes =
            is_partial_tail ? (source_length - stick_index * cache.stick_width) * sizeof(float) : cache.stick_nbytes;
        if (!is_cached_stick || is_partial_tail) {
            auto* destination = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cache_l1_addr + i * cache.stick_nbytes);
            for (uint32_t word = 0; word < cache.stick_nbytes / sizeof(uint32_t); ++word) {
                destination[word] = 0U;
            }
        }
        if (!is_cached_stick) {
            continue;
        }
        const uint32_t page_id = page_per_stick ? cache.source_page + stick_index : cache.source_page;
        const uint32_t page_offset = page_per_stick ? 0 : stick_index * cache.stick_nbytes;
        noc.async_read<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
            src,
            CoreLocalMem<uint32_t>(cache_l1_addr + i * cache.stick_nbytes),
            read_nbytes,
            {.page_id = page_id, .offset_bytes = page_offset},
            {});
    }
    noc.async_read_barrier();
    cache_buffer.push_back(reserve_sticks);
    cache_buffer.wait_front(reserve_sticks);

    cache.cached_stick_id = source_stick;
    cache.cached_stick_count = cached_sticks;
    cache.reserved_stick_count = reserve_sticks;
    cache.valid = true;
}

template <typename SrcAccessor>
ALWI float read_source_value(
    const SrcAccessor& src, StickReadCache& cache, const uint32_t source_index, const uint32_t source_length) {
    const uint32_t source_stick = source_index / cache.stick_width;
    const uint32_t source_lane = source_index % cache.stick_width;
    const uint32_t source_stick_count = (source_length + cache.stick_width - 1) / cache.stick_width;

    if (!cache_contains_stick(cache, source_stick)) {
        cache_source_sticks(src, cache, source_stick, source_stick_count, source_length);
    }

    const auto* cached_values = reinterpret_cast<const float*>(CircularBuffer(cache.cb_id).get_read_ptr());
    const uint32_t cached_offset = source_stick - cache.cached_stick_id;
    return cached_values[cached_offset * cache.stick_width + source_lane];
}

template <typename SrcAccessor>
LWT_BOUNDARY_CALLABLE float read_extended_source_value(
    const SrcAccessor& src, StickReadCache& cache, const uint32_t source_index, const uint32_t source_length) {
    return read_source_value(src, cache, source_index, source_length);
}

template <typename SrcAccessor>
struct CachedSourceReader {
    const SrcAccessor& src;
    StickReadCache& cache;
    uint32_t source_length;

    ALWI float operator()(const uint32_t source_index) const {
        return read_extended_source_value(src, cache, source_index, source_length);
    }
};

template <ttnn::operations::wavelet::BoundaryMode Mode, typename SrcAccessor>
LWT_BOUNDARY_CALLABLE float read_extended_value(
    const SrcAccessor& src,
    StickReadCache& cache,
    const uint32_t input_length,
    const uint32_t left_pad,
    const uint32_t out_idx) {
    static_assert(
        ttnn::operations::wavelet::is_supported_lwt_boundary_mode(Mode), "Unsupported compile-time boundary mode");
    const int32_t logical = static_cast<int32_t>(out_idx) - static_cast<int32_t>(left_pad);
    if constexpr (Mode == ttnn::operations::wavelet::BoundaryMode::kAntireflect) {
        const auto extended = ttnn::operations::wavelet::make_antireflect_index_i32(logical, input_length);
        return ttnn::operations::wavelet::evaluate_antireflect_index_i32(
            extended,
            input_length,
            CachedSourceReader<SrcAccessor>{
                .src = src,
                .cache = cache,
                .source_length = input_length,
            });
    } else {
        const ttnn::operations::wavelet::ExtendedIndexI32 extended =
            ttnn::operations::wavelet::make_extended_index_i32<Mode>(logical, input_length);
        return ttnn::operations::wavelet::evaluate_extended_index_i32<Mode>(
            extended,
            input_length,
            CachedSourceReader<SrcAccessor>{
                .src = src,
                .cache = cache,
                .source_length = input_length,
            });
    }
}

template <ttnn::operations::wavelet::BoundaryMode Mode, typename SrcAccessor>
ALWI float read_padded_value(
    const SrcAccessor& src,
    StickReadCache& cache,
    const uint32_t input_length,
    const uint32_t left_pad,
    const uint32_t out_idx) {
    static_assert(
        ttnn::operations::wavelet::is_supported_lwt_boundary_mode(Mode), "Unsupported compile-time boundary mode");

    if (out_idx >= left_pad) {
        const uint32_t source_index = out_idx - left_pad;
        if (source_index < input_length) {
            return read_source_value(src, cache, source_index, input_length);
        }
    }
    return read_extended_value<Mode>(src, cache, input_length, left_pad, out_idx);
}

ALWI void release_cache(StickReadCache& cache) {
    if (!cache.valid) {
        return;
    }

    CircularBuffer(cache.cb_id).pop_front(cache.reserved_stick_count);
    cache.cached_stick_id = kInvalidStick;
    cache.cached_stick_count = 0;
    cache.reserved_stick_count = 0;
    cache.valid = false;
}

}  // namespace ttnn::operations::wavelet::kernels::primitives

#undef LWT_BOUNDARY_CALLABLE
#undef LWT_CACHE_REFILL_CALLABLE
