// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <algorithm>
#include "dataflow_api.h"
#include <tt-metalium/constants.hpp>

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

template <bool is_dram = true, uint32_t tile_bytes>
void read_chunk_with_padding(
    const InterleavedAddrGenFast<is_dram>& reader,
    const uint32_t cb_id,
    uint32_t start_tile_id,
    const uint32_t src_rows,
    const uint32_t src_cols,
    const uint32_t dst_rows,
    const uint32_t dst_cols,
    const uint32_t barrier_threshold,
    const bool transpose = false) {
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

template <uint32_t num_heads, uint32_t block_size_t, uint32_t Wt, bool is_dram = true>
void read_paged_chunk_with_padding(
    const InterleavedAddrGenFast<is_dram>& reader,
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
    const bool transpose = false) {
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
inline void fill_diagonal_tile_bfp4(uint32_t cb_id, uint32_t tile_id) {
    // Clear the tile first
    fill_tile_zeros<tile_bytes>(cb_id, tile_id);

    /**
     * In bfp4_b, -inf is represented as 0xF exp and 0xC mantissa.
     * bfp4_b tile is laid out in memory as:
     * [face0 exp][face1 exp][face2 exp][face3 exp]
     * [face0 mant][face1 mant][face2 mant][face3 mant]
     * where each face's exp is 16 bytes and each face's mant is 16x16x.5B = 128B.
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

    // Process face 0 and face 3 (diagonal faces)
    for (uint32_t face_idx = 0; face_idx < 4; face_idx += 3) {
        uint32_t face_offset = (face_idx == 0) ? face0_offset : face3_offset;

        for (uint32_t row = 0; row < tt::constants::FACE_HEIGHT; row++) {
            uint32_t diag_col = row;                                   // Diagonal position is at (row, row)
            uint32_t diag_group_idx = diag_col / bf4_mant_per_uint32;  // Which uint32 group contains diagonal
            uint32_t pos_in_group = diag_col % bf4_mant_per_uint32;    // Position within the group
            uint32_t row_offset = row * uint32_datums_per_face_row;

            // Process the uint32 containing the diagonal element
            // Create mask where positions after diagonal are -inf mantissa (upper triangle)
            uint32_t diag_mask = 0;
            for (uint32_t i = pos_in_group + 1; i < bf4_mant_per_uint32; i++) {
                diag_mask |= 0xC << (i * 4);
            }
            uint32_ptr[face_offset + row_offset + diag_group_idx] = diag_mask;

            // Fill all uint32s to the right of diagonal with -inf mantissa
            for (uint32_t col_group = diag_group_idx + 1; col_group < uint32_datums_per_face_row; col_group++) {
                uint32_ptr[face_offset + row_offset + col_group] = NEG_INF_MANT;
            }
        }
    }

    // Fill face 1 completely with -inf
    for (uint32_t i = 0; i < uint32_datums_per_face; i++) {
        uint32_ptr[face1_offset + i] = NEG_INF_MANT;
    }

    // Face 2 is all zeros (already set by fill_tile at the beginning)
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

template <uint32_t cb_mask_in>
void generate_causal_mask(uint32_t Sq_chunk_t, uint32_t Sk_chunk_t, uint32_t q_chunk, uint32_t k_chunk) {
    uint32_t mask_size_tiles = Sq_chunk_t * Sk_chunk_t;
    cb_reserve_back(cb_mask_in, mask_size_tiles);

    uint32_t write_ptr_base = get_write_ptr(cb_mask_in);
    uint64_t noc_write_addr_base = get_noc_addr(write_ptr_base);
    constexpr uint32_t tile_bytes = get_tile_size(cb_mask_in);

    int zero_tile_idx = -1;
    int inf_tile_idx = -1;
    int diag_tile_idx = -1;

    for (uint32_t q_tile = 0; q_tile < Sq_chunk_t; ++q_tile) {
        for (uint32_t k_tile = 0; k_tile < Sk_chunk_t; ++k_tile) {
            uint32_t in_mask_tile_id = q_tile * Sk_chunk_t + k_tile;
            uint32_t global_q_tile = Sq_chunk_t * q_chunk + q_tile;
            uint32_t global_k_tile = Sk_chunk_t * k_chunk + k_tile;

            if (global_k_tile < global_q_tile) {
                if (zero_tile_idx == -1) {
                    fill_tile_zeros<tile_bytes>(cb_mask_in, in_mask_tile_id);
                    zero_tile_idx = in_mask_tile_id;
                } else {
                    copy_tile<tile_bytes>(noc_write_addr_base, write_ptr_base, zero_tile_idx, in_mask_tile_id);
                }
            } else if (global_k_tile == global_q_tile) {
                if (diag_tile_idx == -1) {
                    fill_diagonal_tile_bfp4<tile_bytes>(cb_mask_in, in_mask_tile_id);
                    diag_tile_idx = in_mask_tile_id;
                } else {
                    copy_tile<tile_bytes>(noc_write_addr_base, write_ptr_base, diag_tile_idx, in_mask_tile_id);
                }
            } else {
                if (inf_tile_idx == -1) {
                    fill_neginf_tile_bfp4<tile_bytes>(cb_mask_in, in_mask_tile_id);
                    inf_tile_idx = in_mask_tile_id;
                } else {
                    copy_tile<tile_bytes>(noc_write_addr_base, write_ptr_base, inf_tile_idx, in_mask_tile_id);
                }
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

struct CatAddrGenerator {
    InterleavedAddrGenFast<true> first_reader;
    InterleavedAddrGenFast<true> second_reader;
    TensorTileShape first_shape;
    TensorTileShape second_shape;
    uint32_t first_seq_padded;
    uint32_t second_seq_padded;

    CatAddrGenerator(
        const InterleavedAddrGenFast<true>& first_reader,
        TensorTileShape first_logical_shape,
        uint32_t first_seq_padded,
        const InterleavedAddrGenFast<true>& second_reader,
        TensorTileShape second_logical_shape,
        uint32_t second_seq_padded) :
        first_reader(first_reader),
        second_reader(second_reader),
        first_shape(first_logical_shape),
        second_shape(second_logical_shape),
        first_seq_padded(first_seq_padded),
        second_seq_padded(second_seq_padded) {}

    uint32_t maybe_read_tile(uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3, uint32_t dst_addr) const {
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

    uint32_t maybe_write_tile(uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3, uint32_t src_addr) const {
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

struct Slice {
    uint32_t d0;        // batch dimension
    uint32_t d1;        // head dimension
    uint32_t d2_start;  // sequence start
    uint32_t d2_end;    // sequence end
    uint32_t d3_start;  // feature start
    uint32_t d3_end;    // feature end

    Slice(uint32_t d0, uint32_t d1, uint32_t d2_start, uint32_t d2_end, uint32_t d3_start, uint32_t d3_end) :
        d0(d0), d1(d1), d2_start(d2_start), d2_end(d2_end), d3_start(d3_start), d3_end(d3_end) {}

    uint32_t get_d2_size() const { return d2_end - d2_start; }
    uint32_t get_d3_size() const { return d3_end - d3_start; }
};

void read_block(
    const CatAddrGenerator& cat_addr_generator,
    const Slice& src_slice,
    const uint32_t cb_id,
    const uint32_t tile_bytes,
    const uint32_t barrier_threshold,
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
                src_slice.d0, src_slice.d1, src_slice.d2_start + row, src_slice.d3_start + col, write_ptr);

            write_ptr += inner_ptr_stride;
            barrier_count += did_read;
            if (barrier_count == barrier_threshold) {
                noc_async_read_barrier();
                barrier_count = 0;
            }
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_id, num_tiles);
}

void write_block(
    const CatAddrGenerator& cat_addr_generator,
    const Slice& dst_slice,
    const uint32_t cb_id,
    const uint32_t tile_bytes,
    const uint32_t barrier_threshold) {
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
                dst_slice.d0, dst_slice.d1, dst_slice.d2_start + row, dst_slice.d3_start + col, read_ptr);
            read_ptr += inner_ptr_stride;

            barrier_count += did_write;
            if (barrier_count == barrier_threshold) {
                noc_async_writes_flushed();
                barrier_count = 0;
            }
        }
    }
    noc_async_write_barrier();
    cb_pop_front(cb_id, num_tiles);
}
