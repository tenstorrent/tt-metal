// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// This file contains common kernel functions used in data movement device kernels
// It's best to copy and paste the functions in rather than include the header as code size will likely explode
// Best to separate in to cpp/hpp at some point to avoid the code size explosion but need to figure out the linking
// issues
#include <stdio.h>
#include <cstring>
#include <type_traits>

#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"

constexpr uint64_t ALIGN_REQ_64 = 64;
constexpr uint64_t MASK_64 = 0xFFFFFFFFFFFFFFC0;
constexpr uint64_t OFFSET_64 = 0x000000000000003F;
constexpr uint64_t ALIGN_REQ_16 = 16;
constexpr uint64_t MASK_16 = 0xFFFFFFFFFFFFFFF0;
constexpr uint64_t OFFSET_16 = 0x000000000000000F;

namespace tt::data_movement::common {

template <uint32_t max_transfer_size, bool only_reads>
FORCE_INLINE void enhanced_noc_async_read(
    Noc noc, const uint64_t src_noc_addr, const uint32_t dst_l1_addr, const uint32_t bytes) {
    constexpr uint32_t page_size = (only_reads && max_transfer_size <= NOC_MAX_BURST_SIZE)
                                       ? NOC_MAX_BURST_SIZE
                                       : (max_transfer_size == 0 ? NOC_MAX_BURST_SIZE + 1 : max_transfer_size);
    noc.async_read<NocOptions::DEFAULT, page_size>(
        UnicastEndpoint{},
        CoreLocalMem<uint32_t>(dst_l1_addr),
        bytes,
        {.noc_x = (uint32_t)NOC_UNICAST_ADDR_X(src_noc_addr),
         .noc_y = (uint32_t)NOC_UNICAST_ADDR_Y(src_noc_addr),
         .addr = (uint32_t)NOC_LOCAL_ADDR_OFFSET(src_noc_addr)},
        {.offset_bytes = 0});
}

template <uint32_t max_transfer_size, bool only_reads>
[[deprecated("Use the overload with leading Noc parameter instead.")]]
FORCE_INLINE void enhanced_noc_async_read(
    const uint64_t src_noc_addr, const uint32_t dst_l1_addr, const uint32_t bytes) {
    Noc noc;
    return enhanced_noc_async_read<max_transfer_size, only_reads>(noc, src_noc_addr, dst_l1_addr, bytes);
}

template <uint32_t max_transfer_size, bool only_writes>
FORCE_INLINE void enhanced_noc_async_write(
    Noc noc, const uint32_t src_l1_addr, const uint64_t dst_noc_addr, const uint32_t bytes) {
    constexpr uint32_t page_size = (only_writes && max_transfer_size <= NOC_MAX_BURST_SIZE)
                                       ? NOC_MAX_BURST_SIZE
                                       : (max_transfer_size == 0 ? NOC_MAX_BURST_SIZE + 1 : max_transfer_size);
    noc.async_write<NocOptions::DEFAULT, page_size>(
        CoreLocalMem<uint32_t>(src_l1_addr),
        UnicastEndpoint{},
        bytes,
        {.offset_bytes = 0},
        {.noc_x = (uint32_t)NOC_UNICAST_ADDR_X(dst_noc_addr),
         .noc_y = (uint32_t)NOC_UNICAST_ADDR_Y(dst_noc_addr),
         .addr = (uint32_t)NOC_LOCAL_ADDR_OFFSET(dst_noc_addr)});
}

template <uint32_t max_transfer_size, bool only_writes>
[[deprecated("Use the overload with leading Noc parameter instead.")]]
FORCE_INLINE void enhanced_noc_async_write(
    const uint32_t src_l1_addr, const uint64_t dst_noc_addr, const uint32_t bytes) {
    Noc noc;
    return enhanced_noc_async_write<max_transfer_size, only_writes>(noc, src_l1_addr, dst_noc_addr, bytes);
}

// Self-NOC src/dst args for a local L1 address on the executing core, expressed in the
// caller's noc coordinate space (NOC 0 and NOC 1 have different coordinate spaces, so the
// noc id must match what the read/write will use).
FORCE_INLINE noc_traits_t<UnicastEndpoint>::src_args_type self_l1_src_args(Noc noc, uint32_t addr) {
    const uint8_t id = noc.get_noc_id();
    return {.noc_x = my_x[id], .noc_y = my_y[id], .addr = addr};
}
FORCE_INLINE noc_traits_t<UnicastEndpoint>::dst_args_type self_l1_dst_args(Noc noc, uint32_t addr) {
    const uint8_t id = noc.get_noc_id();
    return {.noc_x = my_x[id], .noc_y = my_y[id], .addr = addr};
}

template <bool guaranteed_16B_aligned, bool copy_async, bool use_read_datamover, uint32_t max_transfer_size>
FORCE_INLINE void tt_memmove(Noc noc, const uint32_t dst_l1_addr, const uint32_t src_l1_addr, const uint32_t bytes) {
    constexpr uint32_t page_size = max_transfer_size == 0 ? NOC_MAX_BURST_SIZE + 1 : max_transfer_size;
    if constexpr (use_read_datamover) {
        if constexpr (guaranteed_16B_aligned) {
            noc.async_read<NocOptions::DEFAULT, page_size>(
                UnicastEndpoint{},
                CoreLocalMem<uint32_t>(dst_l1_addr),
                bytes,
                self_l1_src_args(noc, src_l1_addr),
                {.offset_bytes = 0});
            if constexpr (!copy_async) {
                noc.async_read_barrier();
            }
        } else {
            if ((dst_l1_addr & OFFSET_16) == (src_l1_addr & OFFSET_16)) {
                noc.async_read<NocOptions::DEFAULT, page_size>(
                    UnicastEndpoint{},
                    CoreLocalMem<uint32_t>(dst_l1_addr),
                    bytes,
                    self_l1_src_args(noc, src_l1_addr),
                    {.offset_bytes = 0});
                if constexpr (!copy_async) {
                    noc.async_read_barrier();
                }
            } else {
                invalidate_l1_cache();
                memmove((void*)(dst_l1_addr), (void*)(src_l1_addr), (size_t)(bytes));
            }
        }
    } else {
        if constexpr (guaranteed_16B_aligned) {
            noc.async_write<NocOptions::DEFAULT, page_size>(
                CoreLocalMem<uint32_t>(src_l1_addr),
                UnicastEndpoint{},
                bytes,
                {.offset_bytes = 0},
                self_l1_dst_args(noc, dst_l1_addr));
            if constexpr (!copy_async) {
                noc.async_write_barrier();
            }
        } else {
            if ((dst_l1_addr & OFFSET_16) == (src_l1_addr & OFFSET_16)) {
                noc.async_write<NocOptions::DEFAULT, page_size>(
                    CoreLocalMem<uint32_t>(src_l1_addr),
                    UnicastEndpoint{},
                    bytes,
                    {.offset_bytes = 0},
                    self_l1_dst_args(noc, dst_l1_addr));
                if constexpr (!copy_async) {
                    noc.async_write_barrier();
                }
            } else {
                invalidate_l1_cache();
                memmove((void*)(dst_l1_addr), (void*)(src_l1_addr), (size_t)(bytes));
            }
        }
    }
}

template <bool guaranteed_16B_aligned, bool copy_async, bool use_read_datamover, uint32_t max_transfer_size>
[[deprecated("Use the overload with leading Noc parameter instead.")]]
FORCE_INLINE void tt_memmove(const uint32_t dst_l1_addr, const uint32_t src_l1_addr, const uint32_t bytes) {
    Noc noc;
    return tt_memmove<guaranteed_16B_aligned, copy_async, use_read_datamover, max_transfer_size>(
        noc, dst_l1_addr, src_l1_addr, bytes);
}

template <typename T = uint32_t>
FORCE_INLINE void fill_with_val(uint32_t begin_addr, uint32_t n, T val) {
    auto* ptr = reinterpret_cast<volatile tt_l1_ptr T*>(begin_addr);
    for (uint32_t i = 0; i < n; ++i) {
        ptr[i] = val;
    }
}

// Utility functions
template <uint32_t a, uint32_t b>
FORCE_INLINE constexpr uint32_t div_up() {
    static_assert(b > 0, "divisor must be greater than 0");
    return static_cast<uint32_t>((a + b - 1) / b);
}

template <uint32_t a, uint32_t b>
FORCE_INLINE constexpr uint32_t round_up() {
    return b * div_up<a, b>();
}

// Utility functions
FORCE_INLINE uint32_t div_up(const uint32_t a, const uint32_t b) { return static_cast<uint32_t>((a + b - 1) / b); }

FORCE_INLINE uint32_t round_up(const uint32_t a, const uint32_t b) { return b * div_up(a, b); }

// Function template to swap two elements in a uint32_t array
template <size_t N>
FORCE_INLINE void swap_elements(uint32_t (&array)[N], size_t i, size_t j) {
    // Perform the swap
    uint32_t temp = array[i];
    array[i] = array[j];
    array[j] = temp;
}

// 2D Transpose function for debug use in reader/writer kernels
FORCE_INLINE void transpose_2d(
    uint32_t input_l1_addr,
    uint32_t output_l1_addr,
    uint32_t X,
    uint32_t W,
    uint32_t element_size,
    uint32_t input_page_size,
    uint32_t output_page_size) {
    volatile tt_l1_ptr uint8_t* input_ptr = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(input_l1_addr);
    volatile tt_l1_ptr uint8_t* output_ptr = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(output_l1_addr);
    // transpose from XW, where X is outer and W inner, to WX, where W is outer and X is inner
    // each element is element_size bytes
    // each row is W elements, and each row is separated by input_page_size bytes
    // each output row is X elements, and each row is separated by output_page_size bytes

    for (uint32_t x = 0; x < X; ++x) {
        for (uint32_t w = 0; w < W; ++w) {
            // Compute the input and output addresses
            uint32_t input_addr = x * input_page_size + w * element_size;
            uint32_t output_addr = w * output_page_size + x * element_size;
            // Copy the element - do we have memcpy? use this for now
            for (uint32_t i = 0; i < element_size; ++i) {
                output_ptr[output_addr + i] = input_ptr[input_addr + i];
            }
        }
    }
}

template <uint32_t AlignReq>
FORCE_INLINE uint32_t align_address(const uint32_t address, const uint64_t mask) {
    return (address & mask) + AlignReq;
}

// Wait for a specified number of cycles
// This is a blocking wait, so it should only be used for debugging purposes
// It is not recommended to use this in production code
inline void spin(uint32_t cycles) {
#if defined(RISCV_DEBUG_REG_WALL_CLOCK_L) && defined(RISCV_DEBUG_REG_WALL_CLOCK_H)
    volatile uint tt_reg_ptr* clock_lo = reinterpret_cast<volatile uint tt_reg_ptr*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    volatile uint tt_reg_ptr* clock_hi = reinterpret_cast<volatile uint tt_reg_ptr*>(RISCV_DEBUG_REG_WALL_CLOCK_H);
    uint64_t wall_clock_timestamp = clock_lo[0] | ((uint64_t)clock_hi[0] << 32);
    uint64_t wall_clock = 0;
    do {
        wall_clock = clock_lo[0] | ((uint64_t)clock_hi[0] << 32);
    } while (wall_clock < (wall_clock_timestamp + cycles));
#else
    (void)cycles;  // wall-clock debug register not on Quasar yet. Once present, need update. Issue #48543
#endif
}

template <uint32_t Size, class Enable = void>
struct ByteSizeAddressType {
    typedef uint8_t type;
};

template <uint32_t Size>
struct ByteSizeAddressType<Size, typename std::enable_if<Size == 2>::type> {
    typedef uint16_t type;
};

template <uint32_t Size>
struct ByteSizeAddressType<Size, typename std::enable_if<Size == 4>::type> {
    typedef uint32_t type;
};

// Split a logical-row transfer across shards when a BLOCK/WIDTH-sharded buffer's row
// spans multiple cores. Falls through to a single noc_async_{read,write} for interleaved
// buffers, HEIGHT-sharded RM (whole row stays on one core), and shards whose width covers
// the full row.
//
// dest_id is a logical row index; pages_per_row converts it to the starting page index in
// the destination buffer. pages_per_row is the last (W) axis of dspec.tensor_shape() after
// squeeze — equivalent to (and now generalized from) the prior tensor_shape()[1] indexing,
// which assumed a fully-squeezed 2D dspec and so misread the C dim for 4D NCHW inputs.
template <typename AddrGenType>
FORCE_INLINE void noc_async_write_sharded(
    Noc noc, uint32_t l1_addr, AddrGenType tensor, uint32_t dest_id, uint32_t offset, uint32_t size) {
    if constexpr (AddrGenType::DSpec::is_interleaved) {
        noc.async_write(
            CoreLocalMem<uint32_t>(l1_addr), tensor, size, {}, {.page_id = dest_id, .offset_bytes = offset});
    } else {
        const auto& dspec = tensor.dspec();
        const uint32_t r = dspec.rank();
        const uint32_t pages_per_row = (r > 1) ? dspec.tensor_shape()[r - 1] : 1u;
        if (pages_per_row <= 1) {
            noc.async_write(
                CoreLocalMem<uint32_t>(l1_addr), tensor, size, {}, {.page_id = dest_id, .offset_bytes = offset});
            return;
        }
        const uint32_t page_size = tensor.get_aligned_page_size();
        uint32_t sharded_dest_id = dest_id * pages_per_row + offset / page_size;
        uint32_t sharded_offset = offset % page_size;
        uint32_t num_pages = div_up(size + sharded_offset, page_size);
        for (uint32_t i = 0; i < num_pages; i++) {
            uint32_t write_size = std::min(size - i * page_size, page_size - sharded_offset);
            noc.async_write(
                CoreLocalMem<uint32_t>(l1_addr),
                tensor,
                write_size,
                {},
                {.page_id = sharded_dest_id, .offset_bytes = sharded_offset});
            sharded_dest_id++;
            sharded_offset = 0;
            l1_addr += write_size;
        }
    }
}

template <typename AddrGenType>
[[deprecated("Use the overload with leading Noc parameter instead.")]]
FORCE_INLINE void noc_async_write_sharded(
    const uint32_t l1_addr,
    const AddrGenType tensor,
    const uint32_t dest_id,
    const uint32_t offset,
    const uint32_t size) {
    Noc noc;
    return noc_async_write_sharded(noc, l1_addr, tensor, dest_id, offset, size);
}

template <typename AddrGenType>
FORCE_INLINE void noc_async_read_sharded(
    Noc noc, uint32_t l1_addr, AddrGenType tensor, uint32_t src_id, uint32_t offset, uint32_t size) {
    if constexpr (AddrGenType::DSpec::is_interleaved) {
        noc.async_read(tensor, CoreLocalMem<uint32_t>(l1_addr), size, {.page_id = src_id, .offset_bytes = offset}, {});
    } else {
        const auto& dspec = tensor.dspec();
        const uint32_t r = dspec.rank();
        const uint32_t pages_per_row = (r > 1) ? dspec.tensor_shape()[r - 1] : 1u;
        if (pages_per_row <= 1) {
            noc.async_read(
                tensor, CoreLocalMem<uint32_t>(l1_addr), size, {.page_id = src_id, .offset_bytes = offset}, {});
            return;
        }
        const uint32_t page_size = tensor.get_aligned_page_size();
        uint32_t sharded_src_id = src_id * pages_per_row + offset / page_size;
        uint32_t sharded_offset = offset % page_size;
        uint32_t num_pages = div_up(size + sharded_offset, page_size);
        for (uint32_t i = 0; i < num_pages; i++) {
            uint32_t read_size = std::min(size - i * page_size, page_size - sharded_offset);
            noc.async_read(
                tensor,
                CoreLocalMem<uint32_t>(l1_addr),
                read_size,
                {.page_id = sharded_src_id, .offset_bytes = sharded_offset},
                {});
            sharded_src_id++;
            sharded_offset = 0;
            l1_addr += read_size;
        }
    }
}

template <typename AddrGenType>
[[deprecated("Use the overload with leading Noc parameter instead.")]]
FORCE_INLINE void noc_async_read_sharded(
    const uint32_t l1_addr,
    const AddrGenType tensor,
    const uint32_t src_id,
    const uint32_t offset,
    const uint32_t size) {
    Noc noc;
    return noc_async_read_sharded(noc, l1_addr, tensor, src_id, offset, size);
}

}  // namespace tt::data_movement::common
