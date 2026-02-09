// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Implementation file for reduce_helpers_dataflow.hpp
// Do not include directly - include reduce_helpers_dataflow.hpp instead

namespace dataflow_kernel_lib {

template <bool half_tile>
FORCE_INLINE void fill_row0(volatile tt_l1_ptr uint32_t* ptr, uint32_t scaler) {
    constexpr uint32_t num_faces = half_tile ? 2 : 4;

    for (uint32_t face = 0; face < num_faces; ++face) {
        uint32_t face_offset = face * FACE_SIZE_U32;
        for (uint32_t column = 0; column < ROW_SIZE_U32; ++column) {
            ptr[face_offset + column] = scaler;
        }
    }
}

template <bool half_tile>
FORCE_INLINE void generate_reduce_scaler(const uint32_t cb_id, const uint32_t scaler) {
    ASSERT(cb_id < NUM_CIRCULAR_BUFFERS);
    // Verify scaler is properly packed: high 16 bits must equal low 16 bits
    ASSERT((scaler >> 16) == (scaler & 0xFFFF));

    cb_reserve_back(cb_id, 1);
    uint32_t write_addr = get_write_ptr(cb_id);

    zero_faces<half_tile>(write_addr);

    if (scaler != 0) {
        fill_row0<half_tile>(addr_to_l1_ptr(write_addr), scaler);
    }

    cb_push_back(cb_id, 1);
}

}  // namespace dataflow_kernel_lib
