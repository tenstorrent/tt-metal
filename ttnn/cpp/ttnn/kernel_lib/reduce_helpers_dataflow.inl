// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Implementation file for reduce_helpers_dataflow.hpp
// Do not include directly - include reduce_helpers_dataflow.hpp instead

namespace dataflow_kernel_lib {

template <bool half_tile>
FORCE_INLINE void fill_row0_with_noc_copy(uint32_t write_addr_base, uint32_t scaler) {
    // Write row 0 of face 0 only (8 u32 = 32 bytes)
    volatile tt_l1_ptr uint32_t* ptr = addr_to_l1_ptr(write_addr_base);
    for (uint32_t j = 0; j < ROW_SIZE_U32; ++j) {
        ptr[j] = scaler;
    }

    // Use NOC self-reads to copy row 0 to other faces
    // This is much faster than CPU writes
    // Face offsets: face 0 = 0, face 1 = 512 (1<<9), face 2 = 1024 (2<<9), face 3 = 1536 (3<<9)
    constexpr uint32_t row_size_bytes = ROW_SIZE_U32 * sizeof(uint32_t);  // 32 bytes
    uint64_t src_noc_addr = get_noc_addr(write_addr_base);
    noc_async_read_one_packet_set_state(src_noc_addr, row_size_bytes);

    // Copy to face 1
    noc_async_read_one_packet_with_state(src_noc_addr, write_addr_base + (1 << 9));

    if constexpr (!half_tile) {
        // Copy to faces 2 and 3
        noc_async_read_one_packet_with_state(src_noc_addr, write_addr_base + (2 << 9));
        noc_async_read_one_packet_with_state(src_noc_addr, write_addr_base + (3 << 9));
    }

    noc_async_read_barrier();
}

template <bool half_tile>
FORCE_INLINE void generate_reduce_scaler(const uint32_t cb_id, const uint32_t scaler) {
    ASSERT(cb_id < NUM_CIRCULAR_BUFFERS);
    ASSERT(get_cb_interface(cb_id).fifo_num_pages == 1);
    // Verify scaler is properly packed: high 16 bits must equal low 16 bits
    ASSERT((scaler >> 16) == (scaler & 0xFFFF));

    cb_reserve_back(cb_id, 1);
    uint32_t write_addr = get_write_ptr(cb_id);

    zero_faces<half_tile>(write_addr);

    if (scaler != 0) {
        fill_row0_with_noc_copy<half_tile>(write_addr, scaler);
    }

    cb_push_back(cb_id, 1);
}

}  // namespace dataflow_kernel_lib
