// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <type_traits>

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/debug/assert.h"

namespace dataflow_kernel_lib {

// Face size in uint32 (128 u32 = 256 bf16 = 16x16 face)
constexpr uint32_t FACE_SIZE_U32 = 128;

// Face size in uint32 for float32 (256 u32 = 256 f32 = 16x16 face)
constexpr uint32_t FACE_SIZE_U32_FP32 = 256;

/**
 * @brief Convert an L1 address to a volatile L1 pointer
 *
 * @param addr L1 memory address
 * @return Volatile pointer to uint32_t in L1 memory
 */
FORCE_INLINE volatile tt_l1_ptr uint32_t* addr_to_l1_ptr(uint32_t addr) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
}

/**
 * @brief Create NOC source args for a local L1 address on this core
 *
 * @param addr L1 memory address
 * @param noc_id NOC index (defaults to the current core's noc_index)
 * @return UnicastEndpoint src_args_type with this core's NOC coordinates and the given address
 */
FORCE_INLINE auto local_noc_addr(uint32_t addr, uint8_t noc_id = noc_index) {
    return noc_traits_t<UnicastEndpoint>::src_args_type{
        .noc_x = my_x[noc_id], .noc_y = my_y[noc_id], .addr = addr};
}

/**
 * @brief Zero out the exact tile size for a DFB's current write entry using the device zero API.
 *
 * @param dfb DataflowBuffer whose current write entry should be zeroed
 */
FORCE_INLINE void zero_tile(::DataflowBuffer dfb) {
    Noc noc;
    noc.async_write_zeros(dfb, dfb.get_tile_size());
    noc.write_zeros_l1_barrier();
}

/**
 * @brief Reserve, zero-fill, and push one tile into a DataflowBuffer
 *
 * @tparam dfb_id DataflowBuffer ID whose tile byte size should be used
 */
template <uint32_t dfb_id>
FORCE_INLINE void prepare_zero_tile() {
    ::DataflowBuffer dfb(dfb_id);
    dfb.reserve_back(1);
    zero_tile(dfb);
    dfb.push_back(1);
}

/**
 * @brief Alignment-aware byte-range fill for L1 memory.
 *
 * Writes 4 bytes at a time over the 4-byte-aligned interior, then uses
 * element-sized stores for any unaligned head/tail bytes to avoid RV32IM
 * alignment faults (e.g. when padding a row whose byte offset is not 4-byte
 * aligned).
 *
 * @tparam val_size  Element size in bytes: 1 (FP8), 2 (BF16/FP16), or 4 (FP32).
 * @param start_addr  Start L1 byte address of the region to fill.
 * @param n_bytes     Number of bytes to fill (not element count).
 * @param val         Fill value; pass the element encoding in the low bits
 *                    (e.g. a bfloat16 bit pattern in bits [15:0]).
 *                    For val_size < 4 the value is replicated into a uint32_t
 *                    for the bulk writes.
 *
 * @note For val_size == 4 the head/tail element stores compile out entirely.
 *       Both start_addr and n_bytes must be multiples of val_size; an
 *       unaligned range silently under-fills the head/tail bytes.
 */
template <uint32_t val_size>
FORCE_INLINE void fill_l1_range(uint32_t start_addr, uint32_t n_bytes, uint32_t val) {
    static_assert(
        val_size == sizeof(uint8_t) || val_size == sizeof(uint16_t) || val_size == sizeof(uint32_t),
        "Unsupported val_size: must be 1, 2, or 4");
    using IntType = std::conditional_t<
        (val_size == sizeof(uint8_t)),
        uint8_t,
        std::conditional_t<(val_size == sizeof(uint16_t)), uint16_t, uint32_t>>;

    const uint32_t end_addr = start_addr + n_bytes;
    // The 4-byte-aligned interior [start_addr_4B, end_addr_4B) must stay inside
    // [start_addr, end_addr), so the start rounds up and the end rounds down.
    // Masking with ~0x3 clears the low 2 bits, which always rounds down; that is
    // correct for the end but would pull the start before start_addr. Adding 0x3
    // first turns the mask into a round-up (e.g. start_addr 5 -> 5+3=8 -> 8, the
    // first aligned address >= 5), while masking the end alone rounds it down.
    const uint32_t aligned_start = (start_addr + 0x3) & 0xFFFFFFFC;
    // Clamp so short, unaligned ranges (no complete aligned word) yield an empty
    // interior instead of start_addr_4B > end_addr_4B, which would corrupt
    // neighbouring bytes via the head/tail loops below.
    const uint32_t start_addr_4B = aligned_start < end_addr ? aligned_start : end_addr;
    const uint32_t aligned_end = end_addr & 0xFFFFFFFC;
    const uint32_t end_addr_4B = aligned_end > start_addr_4B ? aligned_end : start_addr_4B;

    uint32_t val_4B = val;
    if constexpr (val_size == sizeof(uint8_t)) {
        const uint8_t byte_val = static_cast<uint8_t>(val);
        val_4B =
            (uint32_t(byte_val) << 24) | (uint32_t(byte_val) << 16) | (uint32_t(byte_val) << 8) | uint32_t(byte_val);
    } else if constexpr (val_size == sizeof(uint16_t)) {
        const uint16_t short_val = static_cast<uint16_t>(val);
        val_4B = (uint32_t(short_val) << 16) | uint32_t(short_val);
    }

    // Bulk 4-byte-aligned writes.
    {
        auto* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(start_addr_4B);
        auto* e = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(end_addr_4B);
        for (; p < e; ++p) {
            *p = val_4B;
        }
    }

    // Callers must pass a val_size-aligned range; for val_size == 4 the head/tail
    // loops below compile out entirely, so this is the only enforcement of that contract.
    ASSERT(start_addr % val_size == 0 && n_bytes % val_size == 0);

    // Element-sized stores for unaligned head and tail.
    // For val_size == 4 these loops compile out since start_addr_4B == start_addr
    // and end_addr_4B == end_addr when the range is already 4-byte aligned.
    if constexpr (val_size < sizeof(uint32_t)) {
        const IntType val_ = static_cast<IntType>(val);
        auto* head = reinterpret_cast<volatile tt_l1_ptr IntType*>(start_addr);
        auto* head_end = reinterpret_cast<volatile tt_l1_ptr IntType*>(start_addr_4B);
        for (; head < head_end; ++head) {
            *head = val_;
        }
        auto* tail = reinterpret_cast<volatile tt_l1_ptr IntType*>(end_addr_4B);
        auto* tail_end = reinterpret_cast<volatile tt_l1_ptr IntType*>(end_addr);
        for (; tail < tail_end; ++tail) {
            *tail = val_;
        }
    }
}

}  // namespace dataflow_kernel_lib
