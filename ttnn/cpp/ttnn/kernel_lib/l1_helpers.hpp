// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

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

    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    noc_async_read_one_packet_set_state(zeros_noc_addr, MEM_ZEROS_SIZE);

    for (uint32_t i = 0; i < num_zeros_reads; ++i) {
        noc_async_read_one_packet_with_state(MEM_ZEROS_BASE, write_addr);
        write_addr += MEM_ZEROS_SIZE;
    }
    noc_async_read_barrier();
}

/**
 * @brief Reserve, zero-fill, and push one tile into a circular buffer
 *
 * @tparam cb_id Circular buffer ID whose tile byte size should be used
 */
template <uint32_t cb_id>
FORCE_INLINE void prepare_zero_tile() {
    cb_reserve_back(cb_id, 1);
    zero_tile<cb_id>(get_write_ptr(cb_id));
    cb_push_back(cb_id, 1);
}

}  // namespace dataflow_kernel_lib
