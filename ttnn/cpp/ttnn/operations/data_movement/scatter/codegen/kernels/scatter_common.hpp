// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Page-map and typed L1 element helpers shared by the scatter codegen readers. The element helpers
// are a copy of gather's codegen_gather_common.hpp: plain (non-volatile) L1 loads and stores, which
// is sound because a reader is the only agent touching its resident output row between the copy-in
// barrier and the push.
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
    if (data_format_size == 2) {
        return read_data_from_type<uint16_t>(l1_read_addr, count);
    }
    if (data_format_size == 4) {
        return read_data_from_type<uint32_t>(l1_read_addr, count);
    }
    return read_data_from_type<uint8_t>(l1_read_addr, count);
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
