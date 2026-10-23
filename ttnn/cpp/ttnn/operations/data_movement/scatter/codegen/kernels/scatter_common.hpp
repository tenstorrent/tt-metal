// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Typed L1 read/write helpers for element-level scatter operations.
// Reused from gather_common.hpp — provides get_value_from_tile, write_value_to_tile.
//
// Optimized: no volatile (L1 data stable during scatter), no runtime switch.
#pragma once

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

// Runtime page-map ABI (kept fixed-width across every reader strategy):
//   [rank, input_dims[8], index_dims[8]]
// Readers enumerate pages in the full input tensor.  A valid scatter update is
// present only when every page coordinate lies inside the compact index/source
// prefix.  In that case this maps the full-input ordinal to the compact prefix
// ordinal; otherwise the caller copies input through without reading index/src.
constexpr uint32_t SCATTER_MAX_PAGE_RANK = 8;

template <uint32_t runtime_offset>
FORCE_INLINE bool map_scatter_input_page(const uint32_t input_page, uint32_t& index_page) {
    const uint32_t rank = get_arg_val<uint32_t>(runtime_offset);
    uint32_t remaining = input_page;
    uint32_t mapped = 0;
    uint32_t index_stride = 1;

    for (int32_t axis = static_cast<int32_t>(rank) - 1; axis >= 0; --axis) {
        const uint32_t input_extent = get_arg_val<uint32_t>(runtime_offset + 1 + static_cast<uint32_t>(axis));
        const uint32_t index_extent =
            get_arg_val<uint32_t>(runtime_offset + 1 + SCATTER_MAX_PAGE_RANK + static_cast<uint32_t>(axis));
        const uint32_t coordinate = remaining % input_extent;
        remaining /= input_extent;
        if (coordinate >= index_extent) {
            return false;
        }
        mapped += coordinate * index_stride;
        index_stride *= index_extent;
    }
    index_page = mapped;
    return true;
}

template <typename T>
FORCE_INLINE uint32_t read_data_from_type(const uint32_t l1_addr, const uint32_t count) {
    // tt_l1_ptr marks the L1 address space (standard on silicon) and lets the
    // emule JIT source-patcher rebase this deref of a passed-in offset onto the
    // emulated L1 bridge; without it emule derefs the raw offset (SIGSEGV).
    tt_l1_ptr T* ptr = reinterpret_cast<tt_l1_ptr T*>(l1_addr);
    return ptr[count];
}

FORCE_INLINE uint32_t
get_value_from_tile(const uint32_t l1_read_addr, const uint32_t count, const uint32_t data_format_size) {
    if constexpr (true) {
        // Most common: bf16 (2 bytes)
        if (data_format_size == 2) {
            return read_data_from_type<uint16_t>(l1_read_addr, count);
        }
        if (data_format_size == 4) {
            return read_data_from_type<uint32_t>(l1_read_addr, count);
        }
        return read_data_from_type<uint8_t>(l1_read_addr, count);
    }
}

template <typename T>
FORCE_INLINE void write_data_from_type(const uint32_t l1_addr, const uint32_t count, const uint32_t value) {
    // See read_data_from_type: tt_l1_ptr lets the emule patcher rebase the
    // passed-in L1 offset onto the emulated L1 bridge.
    tt_l1_ptr T* ptr = reinterpret_cast<tt_l1_ptr T*>(l1_addr);
    ptr[count] = value;
}

FORCE_INLINE void write_value_to_tile(
    const uint32_t l1_read_addr, const uint32_t count, const uint32_t data_format_size, const uint32_t value) {
    if (data_format_size == 2) {
        write_data_from_type<uint16_t>(l1_read_addr, count, value);
        return;
    }
    if (data_format_size == 4) {
        write_data_from_type<uint32_t>(l1_read_addr, count, value);
        return;
    }
    write_data_from_type<uint8_t>(l1_read_addr, count, value);
}

// Public scatter reduction ABI.  ``reduction_mode`` is 0=replace, 1=add,
// 2=multiply, 3=max/amax, 4=min/amin; ``value_kind`` is 1=float32,
// 2=int32, 3=uint32, 4=uint16.  Kind 5 (bfloat16) exists on the host only:
// a bf16 reduction is served by the dedicated ROW_MAJOR accumulator kernel
// (scatter_reader_bf16_reduce_rm.cpp), which promotes to FP32 so duplicate
// updates round only once. No case below handles kind 5 -- the fallthrough
// would keep ``current_value`` and silently drop every update -- so the
// generic readers static_assert that kind 5 never arrives with a reduction.
FORCE_INLINE uint32_t scatter_reduce_value(
    const uint32_t current_value,
    const uint32_t source_value,
    const uint32_t reduction_mode,
    const uint32_t value_kind) {
    if (reduction_mode == 0) {
        return source_value;
    }
    if (value_kind == 1) {
        union FloatBits {
            uint32_t bits;
            float value;
        } current, source, result;
        current.bits = current_value;
        source.bits = source_value;
        if (reduction_mode == 1) {
            result.value = current.value + source.value;
        } else if (reduction_mode == 2) {
            result.value = current.value * source.value;
        } else if (reduction_mode == 3) {
            // Match std::max(current, source): comparison-false (including
            // either unordered NaN case) preserves the current accumulator.
            result.value = current.value < source.value ? source.value : current.value;
        } else {
            // Match std::min(current, source), including its NaN policy.
            result.value = source.value < current.value ? source.value : current.value;
        }
        return result.bits;
    }
    if (value_kind == 4) {
        const uint32_t current = current_value & 0xffffu;
        const uint32_t source = source_value & 0xffffu;
        if (reduction_mode == 1) {
            return static_cast<uint16_t>(current + source);
        }
        if (reduction_mode == 2) {
            return static_cast<uint16_t>(current * source);
        }
        if (reduction_mode == 3) {
            return current < source ? source : current;
        }
        return source < current ? source : current;
    }
    // INT32 and UINT32 have identical two's-complement bit results for
    // wrapping add/multiply.  Unsigned arithmetic also avoids signed-overflow
    // undefined behavior in the dataflow compiler.
    if (value_kind == 2 || value_kind == 3) {
        if (reduction_mode == 1) {
            return current_value + source_value;
        }
        if (reduction_mode == 2) {
            return current_value * source_value;
        }
        if (value_kind == 2) {
            const int32_t current = static_cast<int32_t>(current_value);
            const int32_t source = static_cast<int32_t>(source_value);
            return reduction_mode == 3 ? static_cast<uint32_t>(current < source ? source : current)
                                       : static_cast<uint32_t>(source < current ? source : current);
        }
        return reduction_mode == 3 ? (current_value < source_value ? source_value : current_value)
                                   : (source_value < current_value ? source_value : current_value);
    }
    // A malformed/unsupported compile-time kind must never turn a reduction
    // request into an accidental overwrite.  Host validation prevents this
    // branch; preserving the current value is the safest device-side guard.
    return current_value;
}
