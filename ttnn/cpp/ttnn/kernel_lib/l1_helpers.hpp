// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
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
 * @brief Format-aware zero out faces in a tile using NOC reads from the hardware zeros region
 *
 * @tparam data_format Data format (Float16_b or Float32) to determine face size
 * @tparam half_tile If true, zero faces 0-1 only. If false, zero all 4 faces.
 * @param write_addr L1 address where the zeroed data should be written
 */
template <DataFormat data_format, bool half_tile>
FORCE_INLINE void zero_faces(uint32_t write_addr) {
    constexpr uint32_t num_faces = half_tile ? 2 : 4;
    constexpr uint32_t face_size_u32 = (data_format == DataFormat::Float32) ? FACE_SIZE_U32_FP32 : FACE_SIZE_U32;
    constexpr uint32_t bytes_to_zero = num_faces * face_size_u32 * sizeof(uint32_t);
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
