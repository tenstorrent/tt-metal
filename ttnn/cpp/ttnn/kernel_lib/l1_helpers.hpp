// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "experimental/noc.h"
#include "experimental/endpoints.h"

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
 * @brief Zero out the exact tile size for a CB using NOC reads from the hardware zeros region
 *
 * @tparam cb_id Circular buffer ID whose tile byte size should be used
 * @param write_addr L1 address where the zeroed tile should be written
 */
template <uint32_t cb_id>
FORCE_INLINE void zero_tile(uint32_t write_addr) {
    constexpr uint32_t bytes_to_zero = get_tile_size(cb_id);
    static_assert(bytes_to_zero % MEM_ZEROS_SIZE == 0, "CB tile size must be a multiple of MEM_ZEROS_SIZE");
    constexpr uint32_t num_zeros_reads = bytes_to_zero / MEM_ZEROS_SIZE;

    experimental::Noc noc;
    experimental::UnicastEndpoint zeros_src;
    experimental::UnicastEndpoint local_dst;
    uint32_t noc_x = my_x[noc_index];
    uint32_t noc_y = my_y[noc_index];

    noc.set_async_read_state<experimental::Noc::VcSelection::DEFAULT, MEM_ZEROS_SIZE>(
        zeros_src, MEM_ZEROS_SIZE, {.noc_x = noc_x, .noc_y = noc_y, .addr = MEM_ZEROS_BASE});

    for (uint32_t i = 0; i < num_zeros_reads; ++i) {
        noc.async_read_with_state<experimental::Noc::VcSelection::DEFAULT, MEM_ZEROS_SIZE>(
            zeros_src,
            local_dst,
            MEM_ZEROS_SIZE,
            {.noc_x = noc_x, .noc_y = noc_y, .addr = MEM_ZEROS_BASE},
            {.noc_x = noc_x, .noc_y = noc_y, .addr = write_addr});
        write_addr += MEM_ZEROS_SIZE;
    }
    noc.async_read_barrier();
}

}  // namespace dataflow_kernel_lib
