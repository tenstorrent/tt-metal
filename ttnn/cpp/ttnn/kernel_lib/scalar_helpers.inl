// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Implementation file for scalar_helpers.hpp
// Do not include directly - include scalar_helpers.hpp instead

namespace dataflow_kernel_lib {

// ============================================================================
// 16-bit element implementations (bfloat16)
// ============================================================================

FORCE_INLINE void generate_bcast_col_scalar_bfloat16(const uint32_t cb_id, const uint32_t scaler) {
    ASSERT(cb_id < NUM_CIRCULAR_BUFFERS);
    ASSERT((scaler >> 16) == (scaler & 0xFFFF));

    const uint16_t scalar_val = scaler >> 16;

    cb_reserve_back(cb_id, 1);
    uint32_t write_addr = get_write_ptr(cb_id);

    zero_faces<false>(write_addr);

    if (scaler != 0) {
        volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(write_addr);
        // Fill column 0 of faces 0 and 2 (left column of faces)
        for (uint32_t face : {0u, 2u}) {
            uint32_t face_offset = face * FACE_ELEMENTS;
            for (uint32_t row = 0; row < FACE_ROWS; ++row) {
                ptr[face_offset + row * FACE_COLS] = scalar_val;
            }
        }
    }

    cb_push_back(cb_id, 1);
}

FORCE_INLINE void generate_bcast_row_scalar_bfloat16(const uint32_t cb_id, const uint32_t scaler) {
    ASSERT(cb_id < NUM_CIRCULAR_BUFFERS);
    ASSERT((scaler >> 16) == (scaler & 0xFFFF));

    cb_reserve_back(cb_id, 1);
    uint32_t write_addr = get_write_ptr(cb_id);

    zero_faces<false>(write_addr);

    if (scaler != 0) {
        // Use u32 writes for efficiency (writes 2 bf16 elements at a time)
        volatile tt_l1_ptr uint32_t* ptr = addr_to_l1_ptr(write_addr);
        constexpr uint32_t face_size_u32 = FACE_ELEMENTS / 2;  // 128 u32 per face
        constexpr uint32_t row_size_u32 = FACE_COLS / 2;       // 8 u32 per row

        // Fill row 0 of faces 0 and 1 (top row of faces)
        for (uint32_t face : {0u, 1u}) {
            uint32_t face_offset = face * face_size_u32;
            for (uint32_t col = 0; col < row_size_u32; ++col) {
                ptr[face_offset + col] = scaler;
            }
        }
    }

    cb_push_back(cb_id, 1);
}

FORCE_INLINE void generate_bcast_scalar_bfloat16(const uint32_t cb_id, const uint32_t scaler) {
    ASSERT(cb_id < NUM_CIRCULAR_BUFFERS);
    ASSERT((scaler >> 16) == (scaler & 0xFFFF));

    cb_reserve_back(cb_id, 1);
    uint32_t write_addr = get_write_ptr(cb_id);

    zero_faces<false>(write_addr);

    if (scaler != 0) {
        volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(write_addr);
        ptr[0] = scaler >> 16;
    }

    cb_push_back(cb_id, 1);
}

// ============================================================================
// 32-bit element implementations (float32, int32)
// ============================================================================

FORCE_INLINE void generate_bcast_col_scalar(const uint32_t cb_id, const uint32_t scaler) {
    ASSERT(cb_id < NUM_CIRCULAR_BUFFERS);

    cb_reserve_back(cb_id, 1);
    uint32_t write_addr = get_write_ptr(cb_id);

    zero_faces<false>(write_addr);

    if (scaler != 0) {
        volatile tt_l1_ptr uint32_t* ptr = addr_to_l1_ptr(write_addr);
        // Fill column 0 of faces 0 and 2 (left column of faces)
        for (uint32_t face : {0u, 2u}) {
            uint32_t face_offset = face * FACE_ELEMENTS;
            for (uint32_t row = 0; row < FACE_ROWS; ++row) {
                ptr[face_offset + row * FACE_COLS] = scaler;
            }
        }
    }

    cb_push_back(cb_id, 1);
}

FORCE_INLINE void generate_bcast_row_scalar(const uint32_t cb_id, const uint32_t scaler) {
    ASSERT(cb_id < NUM_CIRCULAR_BUFFERS);

    cb_reserve_back(cb_id, 1);
    uint32_t write_addr = get_write_ptr(cb_id);

    zero_faces<false>(write_addr);

    if (scaler != 0) {
        volatile tt_l1_ptr uint32_t* ptr = addr_to_l1_ptr(write_addr);
        // Fill row 0 of faces 0 and 1 (top row of faces)
        for (uint32_t face : {0u, 1u}) {
            uint32_t face_offset = face * FACE_ELEMENTS;
            for (uint32_t col = 0; col < FACE_COLS; ++col) {
                ptr[face_offset + col] = scaler;
            }
        }
    }

    cb_push_back(cb_id, 1);
}

FORCE_INLINE void generate_bcast_scalar(const uint32_t cb_id, const uint32_t scaler) {
    ASSERT(cb_id < NUM_CIRCULAR_BUFFERS);

    cb_reserve_back(cb_id, 1);
    uint32_t write_addr = get_write_ptr(cb_id);

    zero_faces<false>(write_addr);

    if (scaler != 0) {
        volatile tt_l1_ptr uint32_t* ptr = addr_to_l1_ptr(write_addr);
        ptr[0] = scaler;
    }

    cb_push_back(cb_id, 1);
}

}  // namespace dataflow_kernel_lib
