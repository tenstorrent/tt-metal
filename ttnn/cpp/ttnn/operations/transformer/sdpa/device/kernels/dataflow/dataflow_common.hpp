// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <algorithm>
#include "api/dataflow/dataflow_api.h"
#include <tt-metalium/constants.hpp>
#include "api/debug/assert.h"

template <uint32_t tile_bytes, uint32_t num_readers>
constexpr uint32_t get_barrier_read_threshold() {
    return ((512 / num_readers) * (1024 + 128)) / tile_bytes;
}

void fill_zeros_async(uint32_t write_addr, uint32_t tile_bytes) {
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_addr);
    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    // Fill tile with zeros
    uint32_t bytes_left = tile_bytes;
    for (;;) {
        uint32_t read_size = bytes_left > MEM_ZEROS_SIZE ? MEM_ZEROS_SIZE : bytes_left;
        noc_async_read(zeros_noc_addr, write_addr, read_size);
        write_addr += read_size;
        bytes_left -= read_size;
        if (bytes_left == 0) {
            break;
        }
    }
}

template <uint32_t tile_bytes>
void fill_tile_zeros(uint32_t cb_id, uint32_t tile_id) {
    static_assert(tile_bytes % 4 == 0, "tile_bytes must be a multiple of 4");

    uint32_t write_addr = get_write_ptr(cb_id) + tile_id * tile_bytes;
    fill_zeros_async(write_addr, tile_bytes);
    noc_async_read_barrier();
}

template <uint32_t num_heads, uint32_t block_size_t, uint32_t Wt>
uint32_t virtual_seq_tile_id_to_physical_tile_id(
    uint32_t seq_tile_idx, uint32_t cur_head, const volatile tt_l1_ptr uint32_t* const page_table_ptr) {
    // Given some index in the sequence tiles in range [0, max_seq_len_t]
    // Return the physical tile id for that tile row
    constexpr uint32_t block_stride = num_heads * block_size_t * Wt;
    const uint32_t head_offset = cur_head * block_size_t * Wt;

    const uint32_t virtual_block = seq_tile_idx / block_size_t;
    const uint32_t physical_block = page_table_ptr[virtual_block];
    const uint32_t block_row_offset = seq_tile_idx % block_size_t;
    const uint32_t block_offset = block_row_offset * Wt;
    return physical_block * block_stride + head_offset + block_offset;
}

class TensorTileShape {
public:
    uint32_t shape[4];
    uint32_t strides[4];
    // Constructor to initialize with 4D shape
    TensorTileShape(uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3) {
        shape[0] = d0;
        shape[1] = d1;
        shape[2] = d2;
        shape[3] = d3;

        // Calculate strides (row-major order)
        strides[3] = 1;
        strides[2] = strides[3] * shape[3];
        strides[1] = strides[2] * shape[2];
        strides[0] = strides[1] * shape[1];
    }

    // Get flattened index from 4D coordinates
    uint32_t id_of(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3) const {
        return i0 * strides[0] + i1 * strides[1] + i2 * strides[2] + i3 * strides[3];
    }
};

template <uint32_t tile_bytes, typename ReaderType>
void read_chunk_with_padding(
    const ReaderType& reader,
    const uint32_t cb_id,
    uint32_t start_tile_id,
    const uint32_t src_rows,
    const uint32_t src_cols,
    const uint32_t dst_rows,
    const uint32_t dst_cols,
    const uint32_t barrier_threshold,
    const bool transpose = false,
    const uint32_t skip_src_cols = 0) {
    /*
    Method always reads tiles from memory in row-major order.
    It assumes that the block of rows x cols in stored in contiguous tile order.
    That means, it won't work if the chunk to read is a slice of the last dimension.

    This handles the case where the dst CB is larger than the src CB, with some padding on the
    rows or cols of the DST CB.
    */
    // Read Q chunk
    const uint32_t num_tiles = dst_rows * dst_cols;
    cb_reserve_back(cb_id, num_tiles);
    const uint32_t base_write_ptr = get_write_ptr(cb_id);
    uint32_t outer_ptr_stride = transpose ? tile_bytes : dst_cols * tile_bytes;
    uint32_t inner_ptr_stride = transpose ? tile_bytes * dst_rows : tile_bytes;

    uint32_t barrier_count = 0;
    for (uint32_t row = 0; row < src_rows; ++row) {
        uint32_t write_ptr = base_write_ptr + row * outer_ptr_stride;
        for (uint32_t col = 0; col < src_cols; ++col) {
            noc_async_read_tile(start_tile_id, reader, write_ptr);
            start_tile_id += 1;
            write_ptr += inner_ptr_stride;

            if (++barrier_count == barrier_threshold) {
                noc_async_read_barrier();
                barrier_count = 0;
            }
        }
        start_tile_id += skip_src_cols;  // Skip src cols if needed
    }

    // Zero out the padding
    for (uint32_t row = 0; row < dst_rows; ++row) {
        for (uint32_t col = 0; col < dst_cols; ++col) {
            if (row < src_rows && col < src_cols) {
                continue;
            }
            uint32_t tile_id = transpose ? col * dst_rows + row : row * dst_cols + col;
            fill_tile_zeros<tile_bytes>(cb_id, tile_id);
        }
    }
    noc_async_read_barrier();

    cb_push_back(cb_id, num_tiles);
}

template <uint32_t num_heads, uint32_t block_size_t, uint32_t Wt, typename ReaderType>
void read_paged_chunk_with_padding(
    const ReaderType& reader,
    const uint32_t cb_id,
    const uint32_t cur_head,
    const uint32_t chunk_start_row,
    const uint32_t src_rows,
    const uint32_t src_cols,
    const uint32_t dst_rows,
    const uint32_t dst_cols,
    const uint32_t tile_bytes,
    const uint32_t barrier_threshold,
    const volatile tt_l1_ptr uint32_t* const page_table_ptr,
    const bool transpose = false,
    const uint32_t skip_src_cols = 0) {
    const uint32_t num_tiles = dst_rows * dst_cols;
    cb_reserve_back(cb_id, num_tiles);
    const uint32_t base_write_ptr = get_write_ptr(cb_id);

    // Stride calculation based on transpose flag
    uint32_t outer_ptr_stride = transpose ? tile_bytes : dst_cols * tile_bytes;
    uint32_t inner_ptr_stride = transpose ? tile_bytes * dst_rows : tile_bytes;

    uint32_t barrier_count = 0;
    for (uint32_t row = 0; row < src_rows; ++row) {
        uint32_t write_ptr = base_write_ptr + row * outer_ptr_stride;
        uint32_t virtual_row_num = chunk_start_row + row;
        uint32_t physical_tile_id = virtual_seq_tile_id_to_physical_tile_id<num_heads, block_size_t, Wt>(
            virtual_row_num, cur_head, page_table_ptr);

        for (uint32_t col = 0; col < src_cols; ++col) {
            noc_async_read_tile(physical_tile_id, reader, write_ptr);
            physical_tile_id += 1;
            write_ptr += inner_ptr_stride;

            if (++barrier_count == barrier_threshold) {
                noc_async_read_barrier();
                barrier_count = 0;
            }
        }
        physical_tile_id += skip_src_cols;  // Skip src cols if needed
    }
    noc_async_read_barrier();
    cb_push_back(cb_id, num_tiles);
}

template <uint32_t tile_bytes>
void copy_tile(uint64_t noc_read_addr_base, uint32_t q_write_ptr_base, uint32_t src_tile_id, uint32_t dst_tile_id) {
    noc_async_read(
        noc_read_addr_base + src_tile_id * tile_bytes, q_write_ptr_base + dst_tile_id * tile_bytes, tile_bytes);
}

template <uint32_t tile_bytes>
void fill_neginf_tile_bfp4(uint32_t cb_id, uint32_t tile_id) {
    constexpr uint32_t num_exponents = tt::constants::FACE_HEIGHT * (tt::constants::TILE_HW / tt::constants::FACE_HW);
    constexpr uint32_t num_mantissas = tt::constants::TILE_HW / 2;
    static_assert(
        tile_bytes == num_exponents + num_mantissas, "tile_bytes must be equal to bfp4 num_exponents + num_mantissas");

    uint32_t write_addr = get_write_ptr(cb_id) + tile_id * tile_bytes;
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_addr);

    // Fill the first 64 bytes (16 uint32_t values) with 0xFFFFFFFF for exponents
    constexpr uint32_t NEG_INF_EXP = 0xFFFFFFFF;
    constexpr uint32_t exp_words = num_exponents / sizeof(uint32_t);  // 16 words

    for (uint32_t i = 0; i < exp_words; i++) {
        ptr[i] = NEG_INF_EXP;
    }

    // Fill the next 512 bytes (128 uint32_t values) with 0xCCCCCCCC for mantissas
    constexpr uint32_t NEG_INF_MANT = 0xCCCCCCCC;
    constexpr uint32_t mant_words = num_mantissas / sizeof(uint32_t);  // 128 words

    for (uint32_t i = exp_words; i < exp_words + mant_words; i++) {
        ptr[i] = NEG_INF_MANT;
    }
}

template <uint32_t tile_bytes>
inline void fill_custom_diagonal_tile_bfp4(
    uint32_t cb_id, uint32_t tile_id, int32_t leading_diagonal_offset, int32_t trailing_diagonal_offset) {
    // Assert that we're not in a case where the entire tile should be fully masked or fully allowed
    // Those cases should be handled before calling this function
    ASSERT(leading_diagonal_offset >= -32 && trailing_diagonal_offset >= -32);

    // Clear the tile first
    fill_tile_zeros<tile_bytes>(cb_id, tile_id);

    /**
     * In bfp4_b, -inf is represented as 0xF exp and 0xC mantissa.
     * bfp4_b tile is laid out in memory as:
     * [face0 exp][face1 exp][face2 exp][face3 exp]
     * [face0 mant][face1 mant][face2 mant][face3 mant]
     * where each face's exp is 16 bytes and each face's mant is 16x16x.5B = 128B.
     *
     * Diagonal offsets:
     * - leading_diagonal_offset: positive shifts upper triangle mask upward (allows more values)
     * - trailing_diagonal_offset: positive enables lower triangle masking
     * - A position (row, col) is masked if:
     *   - col > row + leading_diagonal_offset (upper triangle), OR
     *   - col < row - trailing_diagonal_offset (lower triangle)
     *
     * Tile face layout (32x32 tile divided into 4 16x16 faces):
     *   Face 0 (rows 0-15, cols 0-15)  | Face 1 (rows 0-15, cols 16-31)
     *   Face 2 (rows 16-31, cols 0-15) | Face 3 (rows 16-31, cols 16-31)
     */

    constexpr uint32_t NEG_INF_EXP = 0xFFFFFFFF;
    constexpr uint32_t NEG_INF_MANT = 0xCCCCCCCC;  // All mantissas set to 0xC
    constexpr uint32_t bf4_mant_per_uint32 = 8;    // 8 mantissas per uint32
    constexpr uint32_t bf4_exp_per_uint32 = 4;     // 4 exponents per uint32

    constexpr uint32_t uint32_datums_per_face_row = tt::constants::FACE_WIDTH / bf4_mant_per_uint32;
    constexpr uint32_t uint32_datums_per_face = (tt::constants::FACE_HW) / bf4_mant_per_uint32;
    constexpr uint32_t uint32_exp_per_face = tt::constants::FACE_HEIGHT / bf4_exp_per_uint32;

    volatile tt_l1_ptr uint32_t* uint32_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id) + tile_id * tile_bytes);

    // Fill all exponents with NEG_INF_EXP
    for (uint32_t i = 0; i < uint32_exp_per_face * 4; i++) {
        uint32_ptr[i] = NEG_INF_EXP;
    }

    // Calculate face offsets in uint32 words
    constexpr uint32_t face0_offset = uint32_exp_per_face * 4;
    constexpr uint32_t face1_offset = face0_offset + uint32_datums_per_face;
    constexpr uint32_t face2_offset = face1_offset + uint32_datums_per_face;
    constexpr uint32_t face3_offset = face2_offset + uint32_datums_per_face;

    // Define face boundaries
    struct FaceInfo {
        uint32_t row_start;
        uint32_t row_end;
        uint32_t col_start;
        uint32_t col_end;
        uint32_t offset;
    };

    FaceInfo faces[4] = {
        {0, 15, 0, 15, face0_offset},   // Face 0: top-left
        {0, 15, 16, 31, face1_offset},  // Face 1: top-right
        {16, 31, 0, 15, face2_offset},  // Face 2: bottom-left
        {16, 31, 16, 31, face3_offset}  // Face 3: bottom-right
    };

    // Process each face
    for (uint32_t face_idx = 0; face_idx < 4; face_idx++) {
        FaceInfo& face = faces[face_idx];

        // Check if face is fully masked by upper triangle: col_min > row_max + leading_diagonal_offset
        bool fully_masked_upper = (leading_diagonal_offset < 32) &&
                                  ((int32_t)face.col_start > (int32_t)face.row_end + leading_diagonal_offset);

        // Check if face is fully masked by lower triangle: col_max < row_min - trailing_diagonal_offset
        bool fully_masked_lower = (trailing_diagonal_offset < 32) &&
                                  ((int32_t)face.col_end < (int32_t)face.row_start - trailing_diagonal_offset);

        if (fully_masked_upper || fully_masked_lower) {
            // Fully masked: fill entire face with -inf
            for (uint32_t i = 0; i < uint32_datums_per_face; i++) {
                uint32_ptr[face.offset + i] = NEG_INF_MANT;
            }
            continue;
        }

        // Check if face is fully allowed (not masked)
        // All positions must satisfy: col <= row + leading_diagonal_offset AND col >= row - trailing_diagonal_offset
        bool fully_allowed_upper = (leading_diagonal_offset >= 32) ||
                                   ((int32_t)face.col_end <= (int32_t)face.row_start + leading_diagonal_offset);
        bool fully_allowed_lower = (trailing_diagonal_offset >= 32) ||
                                   ((int32_t)face.col_start >= (int32_t)face.row_end - trailing_diagonal_offset);

        if (fully_allowed_upper && fully_allowed_lower) {
            // Fully allowed: leave as zeros (already cleared)
            continue;
        }

        // Partially masked: process row by row
        for (uint32_t row = 0; row < tt::constants::FACE_HEIGHT; row++) {
            uint32_t global_row = face.row_start + row;
            uint32_t row_offset = row * uint32_datums_per_face_row;

            for (uint32_t col_group = 0; col_group < uint32_datums_per_face_row; col_group++) {
                uint32_t mask_value = 0;

                // Process each mantissa in this uint32
                for (uint32_t i = 0; i < bf4_mant_per_uint32; i++) {
                    uint32_t local_col = col_group * bf4_mant_per_uint32 + i;
                    uint32_t global_col = face.col_start + local_col;

                    // Check if this position should be masked
                    bool masked_upper = (leading_diagonal_offset < 32) &&
                                        ((int32_t)global_col > (int32_t)global_row + leading_diagonal_offset);
                    bool masked_lower = (trailing_diagonal_offset < 32) &&
                                        ((int32_t)global_col < (int32_t)global_row - trailing_diagonal_offset);

                    if (masked_upper || masked_lower) {
                        mask_value |= 0xC << (i * 4);  // Set mantissa to 0xC (-inf)
                    }
                }

                uint32_ptr[face.offset + row_offset + col_group] = mask_value;
            }
        }
    }
}

template <uint32_t tile_bytes>
void fill_vertical_tile_bfp4(uint32_t cb_id, uint32_t tile_id, uint32_t unpad_col_in_tile) {
    /*
    This tile should be set such that tile[:, unpad_col_in_tile:] = -inf
    For block float 4 format where 8 mantissas are packed per uint32
    */

    // Prefill with zeros (fast)
    fill_tile_zeros<tile_bytes>(cb_id, tile_id);

    constexpr uint32_t NEG_INF_EXP = 0xFFFFFFFF;
    constexpr uint32_t NEG_INF_MANT = 0xCCCCCCCC;  // All mantissas set to 0xC
    constexpr uint32_t bf4_mant_per_uint32 = 8;    // 8 mantissas per uint32
    constexpr uint32_t bf4_exp_per_uint32 = 4;     // 4 exponents per uint32

    constexpr uint32_t uint32_datums_per_face_row = tt::constants::FACE_WIDTH / bf4_mant_per_uint32;
    constexpr uint32_t uint32_datums_per_face = (tt::constants::FACE_HW) / bf4_mant_per_uint32;
    constexpr uint32_t uint32_exp_per_face = tt::constants::FACE_HEIGHT / bf4_exp_per_uint32;

    volatile tt_l1_ptr uint32_t* uint32_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id) + tile_id * tile_bytes);

    // Calculate face offsets in uint32 words
    constexpr uint32_t exp_section_size = uint32_exp_per_face * 4;
    constexpr uint32_t face0_offset = exp_section_size;
    constexpr uint32_t face1_offset = face0_offset + uint32_datums_per_face;
    constexpr uint32_t face2_offset = face1_offset + uint32_datums_per_face;
    constexpr uint32_t face3_offset = face2_offset + uint32_datums_per_face;

    // First, handle the exponents for rows that need -inf
    // We need to set the exponents to NEG_INF_EXP for all rows
    for (uint32_t i = 0; i < exp_section_size; i++) {
        uint32_ptr[i] = NEG_INF_EXP;
    }

    // Process left faces (0 and 2)
    if (unpad_col_in_tile < tt::constants::FACE_WIDTH) {
        const uint32_t unpad_col_group = unpad_col_in_tile / bf4_mant_per_uint32;
        const uint32_t unpad_col_pos = unpad_col_in_tile % bf4_mant_per_uint32;

        for (uint32_t k = 0; k < 4; k += 2) {  // Face 0 and 2
            uint32_t face_offset = (k == 0) ? face0_offset : face2_offset;

            for (uint32_t r = 0; r < tt::constants::FACE_HEIGHT; ++r) {
                uint32_t row_offset = r * uint32_datums_per_face_row;

                // Handle the uint32 containing the boundary column
                if (unpad_col_pos > 0) {
                    uint32_t mask = 0;
                    for (uint32_t i = unpad_col_pos; i < bf4_mant_per_uint32; i++) {
                        mask |= 0xC << (i * 4);
                    }
                    uint32_ptr[face_offset + row_offset + unpad_col_group] = mask;
                }

                // Fill all uint32s to the right of the boundary
                for (uint32_t c = unpad_col_group + (unpad_col_pos > 0 ? 1 : 0); c < uint32_datums_per_face_row; ++c) {
                    uint32_ptr[face_offset + row_offset + c] = NEG_INF_MANT;
                }
            }
        }
    }

    // Process right faces (1 and 3)
    const uint32_t unpad_col_in_right_face =
        (unpad_col_in_tile < tt::constants::FACE_WIDTH) ? 0 : unpad_col_in_tile - tt::constants::FACE_WIDTH;
    const uint32_t unpad_col_in_right_group = unpad_col_in_right_face / bf4_mant_per_uint32;
    const uint32_t unpad_col_in_right_pos = unpad_col_in_right_face % bf4_mant_per_uint32;

    for (uint32_t k = 1; k < 4; k += 2) {  // Face 1 and 3
        uint32_t face_offset = (k == 1) ? face1_offset : face3_offset;

        for (uint32_t r = 0; r < tt::constants::FACE_HEIGHT; ++r) {
            uint32_t row_offset = r * uint32_datums_per_face_row;

            // Handle the uint32 containing the boundary column
            if (unpad_col_in_right_pos > 0) {
                uint32_t mask = 0;
                for (uint32_t i = unpad_col_in_right_pos; i < bf4_mant_per_uint32; i++) {
                    mask |= 0xC << (i * 4);
                }
                uint32_ptr[face_offset + row_offset + unpad_col_in_right_group] = mask;
            }

            // Fill all uint32s to the right of the boundary
            for (uint32_t c = unpad_col_in_right_group + (unpad_col_in_right_pos > 0 ? 1 : 0);
                 c < uint32_datums_per_face_row;
                 ++c) {
                uint32_ptr[face_offset + row_offset + c] = NEG_INF_MANT;
            }
        }
    }
}

enum class MaskType { FULLY_ALLOWED, FULLY_MASKED, PARTIAL_MASK };

template <uint32_t cb_mask_in>
void generate_mask(
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t q_chunk,
    uint32_t k_chunk,
    bool is_causal = true,
    uint32_t sliding_window_size = 0) {
    uint32_t mask_size_tiles = Sq_chunk_t * Sk_chunk_t;
    cb_reserve_back(cb_mask_in, mask_size_tiles);

    uint32_t write_ptr_base = get_write_ptr(cb_mask_in);
    uint64_t noc_write_addr_base = get_noc_addr(write_ptr_base);
    constexpr uint32_t tile_bytes = get_tile_size(cb_mask_in);

    int zero_tile_idx = -1;
    int inf_tile_idx = -1;
    int triu_diag_tile_idx = -1;
    int tril_diag_tile_idx = -1;

    int32_t min_window_start, max_window_start, min_window_end, max_window_end;
    for (uint32_t q_tile = 0; q_tile < Sq_chunk_t; ++q_tile) {
        uint32_t global_q_tile = Sq_chunk_t * q_chunk + q_tile;
        uint32_t q_tile_start = global_q_tile * tt::constants::TILE_HEIGHT;
        uint32_t q_tile_end = q_tile_start + tt::constants::TILE_HEIGHT - 1;  // inclusive
        if (sliding_window_size > 0) {
            // Calculate the sliding window bounds for this Q tile
            if (is_causal) {
                // Causal sliding window: window spans [q_pos - sliding_window_size + 1, q_pos]
                // For the entire Q tile, we need the union of all windows
                min_window_start = (int32_t)q_tile_start - (int32_t)sliding_window_size + 1;
                if (min_window_start < 0) {
                    min_window_start = 0;
                }
                max_window_start = (int32_t)q_tile_end - (int32_t)sliding_window_size + 1;
                if (max_window_start < 0) {
                    max_window_start = 0;
                }
                max_window_end = (int32_t)q_tile_end + 1;    // exclusive
                min_window_end = (int32_t)q_tile_start + 1;  // exclusive
            } else {
                // Non-causal sliding window: window spans [q_pos - sliding_window_size/2, q_pos +
                // sliding_window_size/2]
                int32_t half_window = (int32_t)sliding_window_size / 2;
                min_window_start = (int32_t)q_tile_start - half_window;
                if (min_window_start < 0) {
                    min_window_start = 0;
                }
                max_window_start = (int32_t)q_tile_end - half_window;
                if (max_window_start < 0) {
                    max_window_start = 0;
                }
                max_window_end = (int32_t)q_tile_end + half_window + 1;    // exclusive
                min_window_end = (int32_t)q_tile_start + half_window + 1;  // exclusive
            }
        }
        for (uint32_t k_tile = 0; k_tile < Sk_chunk_t; ++k_tile) {
            uint32_t in_mask_tile_id = q_tile * Sk_chunk_t + k_tile;
            uint32_t global_k_tile = Sk_chunk_t * k_chunk + k_tile;

            // Determine the masking pattern for this tile
            MaskType mask_type = MaskType::FULLY_ALLOWED;
            int32_t leading_diagonal_offset = 0;
            int32_t trailing_diagonal_offset = 0;
            if (sliding_window_size > 0) {
                // Calculate tile boundaries in sequence positions
                uint32_t k_tile_start = global_k_tile * tt::constants::TILE_HEIGHT;
                uint32_t k_tile_end = k_tile_start + tt::constants::TILE_HEIGHT - 1;  // inclusive
                // Check if K tile overlaps with the union of all sliding windows for this Q tile
                bool k_tile_outside_window =
                    ((int32_t)k_tile_end < min_window_start) ||
                    ((int32_t)k_tile_start >=
                     max_window_end);  // outside the start (bottom left) or end of the window (top right)
                if (k_tile_outside_window) {
                    // K tile is completely outside all sliding windows
                    mask_type = MaskType::FULLY_MASKED;
                } else {
                    // K tile overlaps with sliding windows, but we need to check if it's fully contained
                    bool k_tile_fully_contained =
                        ((int32_t)k_tile_start >= min_window_start) &&
                        ((int32_t)k_tile_end < min_window_end);  // fully contained within the window
                    if (k_tile_fully_contained) {
                        mask_type = MaskType::FULLY_ALLOWED;
                    } else {
                        leading_diagonal_offset = (min_window_end - 1) - k_tile_start;
                        trailing_diagonal_offset = k_tile_end - max_window_start;
                        mask_type = MaskType::PARTIAL_MASK;
                    }
                }
            } else {
                // No sliding window
                if (is_causal) {
                    if (global_k_tile < global_q_tile) {
                        mask_type = MaskType::FULLY_ALLOWED;
                    } else if (global_k_tile == global_q_tile) {
                        leading_diagonal_offset = 0;
                        trailing_diagonal_offset = 32;  // we won't apply any masking along the trailing diagonal when
                                                        // no sliding window is applied
                        mask_type = MaskType::PARTIAL_MASK;
                    } else {
                        mask_type = MaskType::FULLY_MASKED;
                    }
                } else {
                    mask_type = MaskType::FULLY_ALLOWED;
                }
            }

            // Apply the appropriate masking
            switch (mask_type) {
                case MaskType::FULLY_ALLOWED:
                    if (zero_tile_idx == -1) {
                        fill_tile_zeros<tile_bytes>(cb_mask_in, in_mask_tile_id);
                        zero_tile_idx = in_mask_tile_id;
                    } else {
                        copy_tile<tile_bytes>(noc_write_addr_base, write_ptr_base, zero_tile_idx, in_mask_tile_id);
                    }
                    break;
                case MaskType::FULLY_MASKED:
                    if (inf_tile_idx == -1) {
                        fill_neginf_tile_bfp4<tile_bytes>(cb_mask_in, in_mask_tile_id);
                        inf_tile_idx = in_mask_tile_id;
                    } else {
                        copy_tile<tile_bytes>(noc_write_addr_base, write_ptr_base, inf_tile_idx, in_mask_tile_id);
                    }
                    break;
                case MaskType::PARTIAL_MASK:
                    fill_custom_diagonal_tile_bfp4<tile_bytes>(
                        cb_mask_in, in_mask_tile_id, leading_diagonal_offset, trailing_diagonal_offset);
            }
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_mask_in, mask_size_tiles);
}

template <uint32_t cb_mask_in>
void generate_noncausal_padded_mask(uint32_t Sq_chunk_t, uint32_t Sk_chunk_t, uint32_t unpadded_Sk) {
    uint32_t mask_size_tiles = Sq_chunk_t * Sk_chunk_t;
    cb_reserve_back(cb_mask_in, mask_size_tiles);

    uint32_t write_ptr_base = get_write_ptr(cb_mask_in);
    uint64_t noc_write_addr_base = get_noc_addr(write_ptr_base);
    constexpr uint32_t tile_bytes = get_tile_size(cb_mask_in);

    int zero_tile_idx = -1;
    int inf_tile_idx = -1;
    int vertical_tile_idx = -1;
    const uint32_t unpadded_Sk_in_chunk =
        unpadded_Sk % (Sk_chunk_t * tt::constants::TILE_WIDTH);  // TODO: constant for tile width
    uint32_t unpad_tile_col_in_chunk = unpadded_Sk_in_chunk / tt::constants::TILE_WIDTH;
    uint32_t unpad_col_in_tile = unpadded_Sk_in_chunk % tt::constants::TILE_WIDTH;

    for (uint32_t q_tile = 0; q_tile < Sq_chunk_t; ++q_tile) {
        for (uint32_t k_tile = 0; k_tile < Sk_chunk_t; ++k_tile) {
            uint32_t in_mask_tile_id = q_tile * Sk_chunk_t + k_tile;
            const bool do_zero = k_tile < unpad_tile_col_in_chunk;
            const bool do_inf =
                k_tile > unpad_tile_col_in_chunk || k_tile == unpad_tile_col_in_chunk && unpad_col_in_tile == 0;

            if (do_zero) {
                if (zero_tile_idx == -1) {
                    fill_tile_zeros<tile_bytes>(cb_mask_in, in_mask_tile_id);
                    zero_tile_idx = in_mask_tile_id;
                } else {
                    copy_tile<tile_bytes>(noc_write_addr_base, write_ptr_base, zero_tile_idx, in_mask_tile_id);
                }
            } else if (do_inf) {
                if (inf_tile_idx == -1) {
                    fill_neginf_tile_bfp4<tile_bytes>(cb_mask_in, in_mask_tile_id);
                    inf_tile_idx = in_mask_tile_id;
                } else {
                    copy_tile<tile_bytes>(noc_write_addr_base, write_ptr_base, inf_tile_idx, in_mask_tile_id);
                }
            } else {
                if (vertical_tile_idx == -1) {
                    fill_vertical_tile_bfp4<tile_bytes>(cb_mask_in, in_mask_tile_id, unpad_col_in_tile);
                    vertical_tile_idx = in_mask_tile_id;
                } else {
                    copy_tile<tile_bytes>(noc_write_addr_base, write_ptr_base, vertical_tile_idx, in_mask_tile_id);
                }
            }
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_mask_in, mask_size_tiles);
}

template <typename FirstReaderType, typename SecondReaderType>
struct CatAddrGenerator {
    FirstReaderType first_reader;
    SecondReaderType second_reader;
    TensorTileShape first_shape;
    TensorTileShape second_shape;
    uint32_t first_seq_padded;
    uint32_t second_seq_padded;

    CatAddrGenerator(
        const FirstReaderType& first_reader,
        TensorTileShape first_logical_shape,
        uint32_t first_seq_padded,
        const SecondReaderType& second_reader,
        TensorTileShape second_logical_shape,
        uint32_t second_seq_padded) :
        first_reader(first_reader),
        second_reader(second_reader),
        first_shape(first_logical_shape),
        second_shape(second_logical_shape),
        first_seq_padded(first_seq_padded),
        second_seq_padded(second_seq_padded) {}

    uint32_t maybe_read_tile(
        uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3, uint32_t end_seq_tile, uint32_t dst_addr) const {
        if (d2 < first_shape.shape[2]) {
            uint32_t tile_id = first_shape.id_of(d0, d1, d2, d3);
            noc_async_read_tile(tile_id, first_reader, dst_addr);
            return 1;
        } else if (d2 >= first_seq_padded && (d2 - first_seq_padded) < second_shape.shape[2]) {
            uint32_t adjusted_seq = d2 - first_seq_padded;
            uint32_t tile_id = second_shape.id_of(d0, d1, adjusted_seq, d3);
            noc_async_read_tile(tile_id, second_reader, dst_addr);
            return 1;
        } else {
            // fill with zeros
            fill_zeros_async(dst_addr, first_reader.page_size);
            return 1;
        }
    }

    uint32_t maybe_write_tile(
        uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3, uint32_t end_seq_tile, uint32_t src_addr) const {
        if (d2 < first_shape.shape[2]) {
            uint32_t tile_id = first_shape.id_of(d0, d1, d2, d3);
            noc_async_write_tile(tile_id, first_reader, src_addr);
            return 1;
        } else if (d2 >= first_seq_padded && (d2 - first_seq_padded) < second_shape.shape[2]) {
            uint32_t adjusted_seq = d2 - first_seq_padded;
            uint32_t tile_id = second_shape.id_of(d0, d1, adjusted_seq, d3);
            noc_async_write_tile(tile_id, second_reader, src_addr);
            return 1;
        }
        return 0;
    }
};

template <typename ReaderType>
struct PaddedAddrGenerator {
    ReaderType reader;
    TensorTileShape tensor_shape;

    PaddedAddrGenerator(const ReaderType& reader, TensorTileShape tensor_shape) :
        reader(reader), tensor_shape(tensor_shape) {}

    uint32_t maybe_read_tile(
        uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3, uint32_t end_seq_tile, uint32_t dst_addr) const {
        if (d2 < tensor_shape.shape[2] && d2 < end_seq_tile) {
            uint32_t tile_id = tensor_shape.id_of(d0, d1, d2, d3);
            noc_async_read_tile(tile_id, reader, dst_addr);
            return 1;
        } else {
            // fill with zeros
            fill_zeros_async(dst_addr, reader.page_size);
            return 1;
        }
    }

    uint32_t maybe_write_tile(
        uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3, uint32_t end_seq_tile, uint32_t src_addr) const {
        if (d2 < tensor_shape.shape[2] && d2 < end_seq_tile) {
            uint32_t tile_id = tensor_shape.id_of(d0, d1, d2, d3);
            noc_async_write_tile(tile_id, reader, src_addr);
            return 1;
        }
        return 0;
    }
};

struct Slice {
    uint32_t d0;        // batch dimension
    uint32_t d1;        // head dimension

    uint32_t d2_start;  // sequence start
    uint32_t d2_end;    // sequence end
    uint32_t d3_start;  // feature start
    uint32_t d3_end;    // feature end

    Slice() = default;

    Slice(uint32_t d0, uint32_t d1, uint32_t d2_start, uint32_t d2_end, uint32_t d3_start, uint32_t d3_end) :
        d0(d0), d1(d1), d2_start(d2_start), d2_end(d2_end), d3_start(d3_start), d3_end(d3_end) {}

    uint32_t get_d2_size() const { return d2_end - d2_start; }
    uint32_t get_d3_size() const { return d3_end - d3_start; }
};

template <typename CatAddrGeneratorType>
void read_block(
    const CatAddrGeneratorType& cat_addr_generator,
    const Slice& src_slice,
    const uint32_t end_seq_tile,
    const uint32_t cb_id,
    const uint32_t tile_bytes,
    const bool transpose) {
    const uint32_t src_rows = src_slice.get_d2_size();
    const uint32_t src_cols = src_slice.get_d3_size();
    const uint32_t num_tiles = src_rows * src_cols;
    cb_reserve_back(cb_id, num_tiles);
    const uint32_t base_write_ptr = get_write_ptr(cb_id);
    uint32_t outer_ptr_stride = transpose ? tile_bytes : src_cols * tile_bytes;
    uint32_t inner_ptr_stride = transpose ? tile_bytes * src_rows : tile_bytes;

    uint32_t barrier_count = 0;
    for (uint32_t row = 0; row < src_rows; ++row) {
        uint32_t write_ptr = base_write_ptr + row * outer_ptr_stride;
        for (uint32_t col = 0; col < src_cols; ++col) {
            uint32_t did_read = cat_addr_generator.maybe_read_tile(
                src_slice.d0,
                src_slice.d1,
                src_slice.d2_start + row,
                src_slice.d3_start + col,
                end_seq_tile,
                write_ptr);

            write_ptr += inner_ptr_stride;
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_id, num_tiles);
}

template <typename CatAddrGeneratorType>
void write_block(
    const CatAddrGeneratorType& cat_addr_generator,
    const Slice& dst_slice,
    const uint32_t end_seq_tile,
    const uint32_t cb_id,
    const uint32_t tile_bytes) {
    const uint32_t dst_rows = dst_slice.get_d2_size();
    const uint32_t dst_cols = dst_slice.get_d3_size();
    const uint32_t num_tiles = dst_rows * dst_cols;
    const uint32_t base_read_ptr = get_read_ptr(cb_id);
    uint32_t outer_ptr_stride = dst_cols * tile_bytes;
    uint32_t inner_ptr_stride = tile_bytes;

    uint32_t barrier_count = 0;

    cb_wait_front(cb_id, num_tiles);
    for (uint32_t row = 0; row < dst_rows; ++row) {
        uint32_t read_ptr = base_read_ptr + row * outer_ptr_stride;
        for (uint32_t col = 0; col < dst_cols; ++col) {
            uint32_t did_write = cat_addr_generator.maybe_write_tile(
                dst_slice.d0, dst_slice.d1, dst_slice.d2_start + row, dst_slice.d3_start + col, end_seq_tile, read_ptr);
            read_ptr += inner_ptr_stride;
        }
    }
    noc_async_write_barrier();
    cb_pop_front(cb_id, num_tiles);
}

template <uint32_t tile_bytes>
void fill_attention_sink_tiles(uint32_t cb_id, uint32_t num_tiles, uint32_t source_tile_addr) {
    /*
    Fill num_tiles tiles in the CB by copying the first element from the source tile
    to the first element of every row in each destination tile.

    The source_tile_addr should contain a tile with the attention sink value at position [0,0].
    Each output tile will have this value replicated in the first column (first element of every row).

    IMPORTANT: All non-first-column positions are initialized to -infinity (0xFF80 in bfloat16)
    to ensure they don't affect the reduce_c<MAX> operation. Without this, stale L1 values
    could corrupt the max computation.
    */

    // Get the first element from the source tile (position [0,0])
    volatile tt_l1_ptr uint16_t* source_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(source_tile_addr);
    uint16_t sink_value = source_ptr[0];  // First element is at offset 0 in bfloat16 tile

    // -infinity in bfloat16 format (sign=1, exp=0xFF, mantissa=0)
    // This ensures stale L1 values don't affect the max computation
    constexpr uint16_t neg_inf_bf16 = 0xFF80;

    uint32_t write_ptr = get_write_ptr(cb_id);

    // Tile is 32x32 in row-major order within faces
    // For bfloat16, each element is 2 bytes (uint16_t)
    // Tile layout: Face0, Face1, Face2, Face3
    // Face0: rows 0-15, cols 0-15 (top-left)
    // Face1: rows 0-15, cols 16-31 (top-right)
    // Face2: rows 16-31, cols 0-15 (bottom-left)
    // Face3: rows 16-31, cols 16-31 (bottom-right)

    constexpr uint32_t face_height = 16;
    constexpr uint32_t face_width = 16;
    constexpr uint32_t elements_per_face = face_height * face_width;
    constexpr uint32_t elements_per_tile = 4 * elements_per_face;  // 1024 elements

    // Fill each tile in the CB
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        volatile tt_l1_ptr uint16_t* tile_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(write_ptr);

        // First, initialize the ENTIRE tile to -infinity
        // This prevents stale L1 values from corrupting the max computation
        for (uint32_t i = 0; i < elements_per_tile; ++i) {
            tile_ptr[i] = neg_inf_bf16;
        }

        // Then fill Face 0 and Face 2 with sink values (first column of tile is in these faces)
        // Face 0: rows 0-15, Face 2: rows 16-31
        for (uint32_t face = 0; face < 4; face += 2) {  // Process Face 0 and Face 2
            uint32_t face_offset = face * elements_per_face;

            // Set first element of each row in this face
            for (uint32_t row = 0; row < face_height; ++row) {
                uint32_t row_offset = row * face_width;
                tile_ptr[face_offset + row_offset] = sink_value;
            }
        }

        write_ptr += tile_bytes;
    }
}
