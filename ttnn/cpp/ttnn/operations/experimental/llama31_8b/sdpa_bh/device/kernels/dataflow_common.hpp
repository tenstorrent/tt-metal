// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <algorithm>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "api/tile_constants.h"
#include "api/debug/assert.h"
#include "q_chunk_remapping.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/sliding_window_geometry.hpp"

template <uint32_t tile_bytes, uint32_t num_readers>
constexpr uint32_t get_barrier_read_threshold() {
    return ((512 / num_readers) * (1024 + 128)) / tile_bytes;
}

inline void fill_zeros_async(const Noc& noc, uint32_t cb_id, uint32_t tile_bytes, uint32_t offset_bytes = 0) {
    CircularBuffer cb(cb_id);
    noc.async_write_zeros(cb, tile_bytes, {.offset_bytes = offset_bytes});
}

template <uint32_t tile_bytes, bool wait_for_barrier = true>
void fill_tile_zeros(const Noc& noc, uint32_t cb_id, uint32_t tile_id) {
    static_assert(tile_bytes % 4 == 0, "tile_bytes must be a multiple of 4");

    fill_zeros_async(noc, cb_id, tile_bytes, tile_id * tile_bytes);
    if (wait_for_barrier) {
        noc.write_zeros_l1_barrier();
    }
}

template <uint32_t tile_bytes, bool wait_for_barrier = true>
void fill_tile_zeros(uint32_t cb_id, uint32_t tile_id) {
    Noc noc;
    fill_tile_zeros<tile_bytes, wait_for_barrier>(noc, cb_id, tile_id);
}

template <typename PageT, uint32_t num_heads, uint32_t block_size_t, uint32_t Wt, uint32_t capacity_t = 0>
uint32_t virtual_seq_tile_id_to_physical_tile_id(
    uint32_t seq_tile_idx, uint32_t cur_head, const volatile tt_l1_ptr PageT* const page_table_ptr) {
    constexpr uint32_t block_stride = num_heads * block_size_t * Wt;
    const uint32_t head_offset = cur_head * block_size_t * Wt;

    if constexpr (capacity_t > 0) {
        seq_tile_idx %= capacity_t;
    }
    const uint32_t virtual_block = seq_tile_idx / block_size_t;
    const uint32_t physical_block = static_cast<uint32_t>(page_table_ptr[virtual_block]);
    const uint32_t block_row_offset = seq_tile_idx % block_size_t;
    const uint32_t block_offset = block_row_offset * Wt;
    return physical_block * block_stride + head_offset + block_offset;
}

template <typename PageTableArgs>
volatile tt_l1_ptr uint32_t* read_page_table_for_batch(
    Noc noc,
    uint32_t cb_id,
    uint32_t batch_idx,
    const PageTableArgs& page_table_args,
    uint32_t page_table_addr,
    uint32_t page_table_stick_size) {
    CircularBuffer cb(cb_id);
    uint32_t page_table_cb_wr_ptr = cb.get_write_ptr();
    const auto page_table_reader = TensorAccessor(page_table_args, page_table_addr, page_table_stick_size);
    noc.async_read(
        page_table_reader,
        CoreLocalMem<uint32_t>(page_table_cb_wr_ptr),
        page_table_stick_size,
        {.page_id = batch_idx},
        {});
    noc.async_read_barrier();
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page_table_cb_wr_ptr);
}

class TensorTileShape {
public:
    uint32_t shape[4];
    uint32_t strides[4];
    TensorTileShape(uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3) {
        shape[0] = d0;
        shape[1] = d1;
        shape[2] = d2;
        shape[3] = d3;

        strides[3] = 1;
        strides[2] = strides[3] * shape[3];
        strides[1] = strides[2] * shape[2];
        strides[0] = strides[1] * shape[1];
    }

    uint32_t id_of(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3) const {
        return i0 * strides[0] + i1 * strides[1] + i2 * strides[2] + i3 * strides[3];
    }

    uint32_t d2() const { return shape[2]; }
    uint32_t stride2() const { return strides[2]; }
};

template <uint32_t D0, uint32_t D1, uint32_t D2, uint32_t D3>
class StaticTensorTileShape {
public:
    constexpr StaticTensorTileShape() = default;
    constexpr StaticTensorTileShape(uint32_t, uint32_t, uint32_t, uint32_t) {}

    uint32_t id_of(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3) const {
        return i0 * D1 * D2 * D3 + i1 * D2 * D3 + i2 * D3 + i3;
    }

    static constexpr uint32_t d2() { return D2; }
    static constexpr uint32_t stride2() { return D3; }
};

template <uint32_t tile_bytes, typename ReaderType, bool push_num_tiles = true>
uint32_t read_chunk_with_padding(
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
    Noc noc;
    const uint32_t num_tiles = dst_rows * dst_cols;
    CircularBuffer cb(cb_id);
    cb.reserve_back(num_tiles);
    const uint32_t base_write_ptr = cb.get_write_ptr();
    uint32_t outer_ptr_stride = transpose ? tile_bytes : dst_cols * tile_bytes;
    uint32_t inner_ptr_stride = transpose ? tile_bytes * dst_rows : tile_bytes;

    uint32_t barrier_count = 0;
    for (uint32_t row = 0; row < src_rows; ++row) {
        uint32_t write_ptr = base_write_ptr + row * outer_ptr_stride;
        for (uint32_t col = 0; col < src_cols; ++col) {
            noc.async_read(reader, CoreLocalMem<uint32_t>(write_ptr), tile_bytes, {.page_id = start_tile_id}, {});
            start_tile_id += 1;
            write_ptr += inner_ptr_stride;

            if (++barrier_count == barrier_threshold) {
                noc.async_read_barrier();
                barrier_count = 0;
            }
        }
        start_tile_id += skip_src_cols;
    }

    for (uint32_t row = 0; row < dst_rows; ++row) {
        for (uint32_t col = 0; col < dst_cols; ++col) {
            if (row < src_rows && col < src_cols) {
                continue;
            }
            uint32_t tile_id = transpose ? col * dst_rows + row : row * dst_cols + col;
            fill_tile_zeros<tile_bytes, false>(noc, cb_id, tile_id);
        }
    }
    noc.async_read_barrier();
    noc.write_zeros_l1_barrier();

    if constexpr (push_num_tiles) {
        cb.push_back(num_tiles);
        return 0;
    } else {
        return num_tiles;
    }
}

template <uint32_t tile_bytes, typename ReaderType>
FORCE_INLINE void read_q_subblock(
    const ReaderType& reader,
    const uint32_t cb_id,
    uint32_t& start_tile_id,
    const uint32_t sb_start_row,
    const uint32_t subblock_h,
    const uint32_t src_rows,
    const uint32_t src_cols,
    const uint32_t dst_cols,
    const uint32_t barrier_threshold) {
    Noc noc;
    const uint32_t sb_tiles = subblock_h * dst_cols;
    CircularBuffer cb(cb_id);
    cb.reserve_back(sb_tiles);
    const uint32_t base_write_ptr = cb.get_write_ptr();

    uint32_t barrier_count = 0;
    for (uint32_t row = sb_start_row; row < sb_start_row + subblock_h; ++row) {
        const uint32_t local_row = row - sb_start_row;
        uint32_t write_ptr = base_write_ptr + local_row * dst_cols * tile_bytes;

        if (row < src_rows) {
            for (uint32_t col = 0; col < src_cols; ++col) {
                noc.async_read(reader, CoreLocalMem<uint32_t>(write_ptr), tile_bytes, {.page_id = start_tile_id++}, {});
                write_ptr += tile_bytes;
                if (++barrier_count == barrier_threshold) {
                    noc.async_read_barrier();
                    barrier_count = 0;
                }
            }
            for (uint32_t col = src_cols; col < dst_cols; ++col) {
                fill_tile_zeros<tile_bytes, false>(noc, cb_id, local_row * dst_cols + col);
            }
        } else {
            for (uint32_t col = 0; col < dst_cols; ++col) {
                fill_tile_zeros<tile_bytes, false>(noc, cb_id, local_row * dst_cols + col);
            }
        }
    }

    noc.async_read_barrier();
    noc.write_zeros_l1_barrier();
    cb.push_back(sb_tiles);
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
    Noc noc;
    const uint32_t num_tiles = dst_rows * dst_cols;
    CircularBuffer cb(cb_id);
    cb.reserve_back(num_tiles);
    const uint32_t base_write_ptr = cb.get_write_ptr();

    uint32_t outer_ptr_stride = transpose ? tile_bytes : dst_cols * tile_bytes;
    uint32_t inner_ptr_stride = transpose ? tile_bytes * dst_rows : tile_bytes;

    uint32_t barrier_count = 0;
    for (uint32_t row = 0; row < src_rows; ++row) {
        uint32_t write_ptr = base_write_ptr + row * outer_ptr_stride;
        uint32_t virtual_row_num = chunk_start_row + row;
        uint32_t physical_tile_id = virtual_seq_tile_id_to_physical_tile_id<uint32_t, num_heads, block_size_t, Wt>(
            virtual_row_num, cur_head, page_table_ptr);

        for (uint32_t col = 0; col < src_cols; ++col) {
            noc.async_read(reader, CoreLocalMem<uint32_t>(write_ptr), tile_bytes, {.page_id = physical_tile_id}, {});
            physical_tile_id += 1;
            write_ptr += inner_ptr_stride;

            if (++barrier_count == barrier_threshold) {
                noc.async_read_barrier();
                barrier_count = 0;
            }
        }
        physical_tile_id += skip_src_cols;
    }

    for (uint32_t row = 0; row < dst_rows; ++row) {
        for (uint32_t col = 0; col < dst_cols; ++col) {
            if (row < src_rows && col < src_cols) {
                continue;
            }
            uint32_t tile_id = transpose ? col * dst_rows + row : row * dst_cols + col;
            fill_zeros_async(noc, cb_id, tile_bytes, tile_id * tile_bytes);
        }
    }
    noc.async_read_barrier();
    noc.write_zeros_l1_barrier();
    cb.push_back(num_tiles);
}

template <uint32_t tile_bytes>
void copy_tile(
    Noc noc, uint32_t src_l1_addr_base, uint32_t dst_l1_addr_base, uint32_t src_tile_id, uint32_t dst_tile_id) {
    const uint8_t noc_id = noc.get_noc_id();
    UnicastEndpoint src;
    noc.async_read(
        src,
        CoreLocalMem<uint32_t>(dst_l1_addr_base + dst_tile_id * tile_bytes),
        tile_bytes,
        {.noc_x = my_x[noc_id], .noc_y = my_y[noc_id], .addr = src_l1_addr_base + src_tile_id * tile_bytes},
        {});
}

template <uint32_t tile_bytes>
void fill_neginf_tile(uint32_t cb_id, uint32_t tile_id) {
    constexpr uint32_t num_exponents = tt::constants::FACE_HEIGHT * (tt::constants::TILE_HW / tt::constants::FACE_HW);
    constexpr uint32_t bfp4_size = num_exponents + tt::constants::TILE_HW / 2;
    constexpr uint32_t bf16_size = tt::constants::TILE_HW * 2;

    CircularBuffer cb(cb_id);
    uint32_t write_addr = cb.get_write_ptr() + tile_id * tile_bytes;
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_addr);
    constexpr uint32_t total_words = tile_bytes / sizeof(uint32_t);

    if constexpr (tile_bytes == bf16_size) {
        for (uint32_t i = 0; i < total_words; i++) {
            ptr[i] = 0xFF80FF80;
        }
    } else {
        constexpr uint32_t exp_words = num_exponents / sizeof(uint32_t);
        for (uint32_t i = 0; i < exp_words; i++) {
            ptr[i] = 0xFFFFFFFF;
        }
        constexpr uint32_t mant_pattern = (tile_bytes == bfp4_size) ? 0xCCCCCCCC : 0x80808080;
        for (uint32_t i = exp_words; i < total_words; i++) {
            ptr[i] = mant_pattern;
        }
    }
}

template <uint32_t tile_bytes>
void fill_vertical_tile_bf16(Noc noc, uint32_t cb_id, uint32_t tile_id, uint32_t unpad_col_in_tile) {
    fill_tile_zeros<tile_bytes>(noc, cb_id, tile_id);

    constexpr uint32_t NEGINF_PAIR = 0xFF80FF80;
    constexpr uint32_t bf16_per_uint32 = 2;
    constexpr uint32_t uint32_per_face_row = tt::constants::FACE_WIDTH / bf16_per_uint32;
    constexpr uint32_t uint32_per_face = tt::constants::FACE_HW / bf16_per_uint32;

    CircularBuffer cb(cb_id);
    volatile tt_l1_ptr uint32_t* ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr() + tile_id * tile_bytes);

    constexpr uint32_t face_offsets[4] = {0, uint32_per_face, 2 * uint32_per_face, 3 * uint32_per_face};
    constexpr uint32_t face_col_starts[4] = {0, 16, 0, 16};

    for (uint32_t f = 0; f < 4; f++) {
        uint32_t face_col_start = face_col_starts[f];

        if (unpad_col_in_tile <= face_col_start) {
            for (uint32_t i = 0; i < uint32_per_face; i++) {
                ptr[face_offsets[f] + i] = NEGINF_PAIR;
            }
        } else if (unpad_col_in_tile >= face_col_start + tt::constants::FACE_WIDTH) {
        } else {
            uint32_t local_col = unpad_col_in_tile - face_col_start;
            uint32_t boundary_word = local_col / bf16_per_uint32;
            uint32_t boundary_pos = local_col % bf16_per_uint32;

            for (uint32_t row = 0; row < tt::constants::FACE_HEIGHT; row++) {
                uint32_t row_base = face_offsets[f] + row * uint32_per_face_row;

                if (boundary_pos != 0) {
                    ptr[row_base + boundary_word] = 0xFF800000;
                }

                uint32_t start_word = boundary_word + (boundary_pos != 0 ? 1 : 0);
                for (uint32_t w = start_word; w < uint32_per_face_row; w++) {
                    ptr[row_base + w] = NEGINF_PAIR;
                }
            }
        }
    }
}

template <uint32_t tile_bytes>
void fill_causal_diagonal_tile_bf16(Noc noc, uint32_t cb_id, uint32_t tile_id) {
    fill_tile_zeros<tile_bytes>(noc, cb_id, tile_id);

    constexpr uint32_t NEGINF_PAIR = 0xFF80FF80;
    constexpr uint32_t bf16_per_uint32 = 2;
    constexpr uint32_t uint32_per_face_row = tt::constants::FACE_WIDTH / bf16_per_uint32;
    constexpr uint32_t uint32_per_face = tt::constants::FACE_HW / bf16_per_uint32;

    CircularBuffer cb(cb_id);
    volatile tt_l1_ptr uint32_t* ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr() + tile_id * tile_bytes);

    constexpr uint32_t face_offsets[4] = {0, uint32_per_face, 2 * uint32_per_face, 3 * uint32_per_face};

    for (uint32_t i = 0; i < uint32_per_face; i++) {
        ptr[face_offsets[1] + i] = NEGINF_PAIR;
    }

    constexpr uint32_t diag_faces[2] = {0, 3};
    for (uint32_t f = 0; f < 2; f++) {
        uint32_t face_base = face_offsets[diag_faces[f]];
        for (uint32_t row = 0; row < tt::constants::FACE_HEIGHT; row++) {
            uint32_t first_masked = row + 1;
            if (first_masked >= tt::constants::FACE_WIDTH) {
                continue;
            }

            uint32_t row_base = face_base + row * uint32_per_face_row;
            uint32_t boundary_word = first_masked / bf16_per_uint32;
            uint32_t boundary_pos = first_masked % bf16_per_uint32;

            if (boundary_pos != 0) {
                ptr[row_base + boundary_word] = 0xFF800000;
                boundary_word++;
            }

            for (uint32_t w = boundary_word; w < uint32_per_face_row; w++) {
                ptr[row_base + w] = NEGINF_PAIR;
            }
        }
    }
}

template <uint32_t tile_bytes, int32_t diagonal_offset = 0, bool leading_edge = true>
void fill_diagonal_edge_tile_bf16(Noc noc, uint32_t cb_id, uint32_t tile_id) {
    fill_tile_zeros<tile_bytes>(noc, cb_id, tile_id);

    constexpr uint32_t neginf_bf16 = 0xFF80;
    constexpr uint32_t bf16_per_uint32 = 2;
    constexpr uint32_t uint32_per_face_row = tt::constants::FACE_WIDTH / bf16_per_uint32;
    constexpr uint32_t uint32_per_face = tt::constants::FACE_HW / bf16_per_uint32;

    CircularBuffer cb(cb_id);
    volatile tt_l1_ptr uint32_t* ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr() + tile_id * tile_bytes);

    constexpr uint32_t face_offsets[4] = {
        0,
        uint32_per_face,
        2 * uint32_per_face,
        3 * uint32_per_face,
    };
    constexpr uint32_t face_row_starts[4] = {0, 0, 16, 16};
    constexpr uint32_t face_col_starts[4] = {0, 16, 0, 16};

    for (uint32_t f = 0; f < 4; f++) {
        const uint32_t face_base = face_offsets[f];
        const uint32_t row_start = face_row_starts[f];
        const uint32_t col_start = face_col_starts[f];
        for (uint32_t row = 0; row < tt::constants::FACE_HEIGHT; row++) {
            const uint32_t global_row = row_start + row;
            const int32_t boundary_col = static_cast<int32_t>(global_row) + diagonal_offset;
            const uint32_t row_base = face_base + row * uint32_per_face_row;
            for (uint32_t word = 0; word < uint32_per_face_row; word++) {
                const int32_t col0 = static_cast<int32_t>(col_start + word * bf16_per_uint32);
                const int32_t col1 = col0 + 1;
                uint32_t mask_word = 0;
                if constexpr (leading_edge) {
                    if (col0 < boundary_col) {
                        mask_word |= neginf_bf16;
                    }
                    if (col1 < boundary_col) {
                        mask_word |= neginf_bf16 << 16;
                    }
                } else {
                    if (col0 > boundary_col) {
                        mask_word |= neginf_bf16;
                    }
                    if (col1 > boundary_col) {
                        mask_word |= neginf_bf16 << 16;
                    }
                }
                if (mask_word != 0) {
                    ptr[row_base + word] = mask_word;
                }
            }
        }
    }
}

template <uint32_t sliding_window_size, bool is_causal_lw, uint32_t cb_mask_in, uint32_t tile_bytes>
void fill_sliding_window_edge_tiles(Noc noc, uint32_t start_tile_idx) {
    constexpr uint32_t half_window = sliding_window_size / 2;
    constexpr uint32_t leading_remainder =
        is_causal_lw ? (sliding_window_size % tt::constants::TILE_HEIGHT) : (half_window % tt::constants::TILE_HEIGHT);
    constexpr uint32_t trailing_remainder = is_causal_lw ? 0 : (half_window % tt::constants::TILE_HEIGHT);
    constexpr int32_t leading_prev_offset =
        is_causal_lw ? (leading_remainder == 0 ? 1 : static_cast<int32_t>(33 - leading_remainder))
                     : (leading_remainder == 0 ? 0 : static_cast<int32_t>(32 - leading_remainder));
    constexpr int32_t leading_current_offset =
        is_causal_lw ? (leading_remainder == 0 ? 1 : 1 - static_cast<int32_t>(leading_remainder))
                     : (leading_remainder == 0 ? 0 : -static_cast<int32_t>(leading_remainder));
    constexpr int32_t trailing_primary_offset = static_cast<int32_t>(trailing_remainder);
    constexpr int32_t trailing_next_offset = static_cast<int32_t>(trailing_remainder) - 32;

    fill_diagonal_edge_tile_bf16<tile_bytes, trailing_primary_offset, false>(noc, cb_mask_in, start_tile_idx);
    fill_diagonal_edge_tile_bf16<tile_bytes, leading_prev_offset, true>(noc, cb_mask_in, start_tile_idx + 1);
    fill_diagonal_edge_tile_bf16<tile_bytes, leading_current_offset, true>(noc, cb_mask_in, start_tile_idx + 2);
    fill_diagonal_edge_tile_bf16<tile_bytes, trailing_next_offset, false>(noc, cb_mask_in, start_tile_idx + 3);
}

template <
    uint32_t global_n_partial_col,
    uint32_t joint_l_partial_col,
    uint32_t cb_mask_in,
    bool is_causal_lw = false,
    uint32_t sliding_window_size = 0>
void generate_lightweight_mask_tiles(
    Noc noc,
    uint32_t global_n_partial_col_rt = global_n_partial_col,
    uint32_t joint_l_partial_col_rt = joint_l_partial_col) {
    constexpr uint32_t partial_mask_tiles = (global_n_partial_col > 0 ? 1 : 0) + (joint_l_partial_col > 0 ? 1 : 0);
    constexpr bool has_sliding_window = sliding_window_size > 0;
    constexpr uint32_t sliding_diag_tiles = has_sliding_window ? kSlidingWindowEdgeTiles : 0;
    constexpr uint32_t causal_diag_tiles = (!has_sliding_window && is_causal_lw) ? 1 : 0;
    constexpr uint32_t total_mask_tiles = 1 + sliding_diag_tiles + causal_diag_tiles + partial_mask_tiles;
    constexpr uint32_t mask_tile_size_bytes = get_tile_size(cb_mask_in);

    CircularBuffer cb(cb_mask_in);
    cb.reserve_back(total_mask_tiles);

    fill_neginf_tile<mask_tile_size_bytes>(cb_mask_in, 0);

    uint32_t tile_idx = 1;

    if constexpr (has_sliding_window) {
        fill_sliding_window_edge_tiles<sliding_window_size, is_causal_lw, cb_mask_in, mask_tile_size_bytes>(
            noc, tile_idx);
        tile_idx += kSlidingWindowEdgeTiles;
    } else if constexpr (is_causal_lw) {
        fill_causal_diagonal_tile_bf16<mask_tile_size_bytes>(noc, cb_mask_in, tile_idx++);
    }

    if constexpr (partial_mask_tiles > 0) {
        if constexpr (global_n_partial_col > 0) {
            fill_vertical_tile_bf16<mask_tile_size_bytes>(noc, cb_mask_in, tile_idx++, global_n_partial_col_rt);
        }
        if constexpr (joint_l_partial_col > 0) {
            fill_vertical_tile_bf16<mask_tile_size_bytes>(noc, cb_mask_in, tile_idx++, joint_l_partial_col_rt);
        }
    }

    cb.push_back(total_mask_tiles);
}

template <uint32_t tile_bytes>
inline void fill_custom_diagonal_tile_bfp4(
    Noc noc, uint32_t cb_id, uint32_t tile_id, int32_t leading_diagonal_offset, int32_t trailing_diagonal_offset) {
    ASSERT(leading_diagonal_offset >= -32 && trailing_diagonal_offset >= -32);

    fill_tile_zeros<tile_bytes>(noc, cb_id, tile_id);

    constexpr uint32_t NEG_INF_EXP = 0xFFFFFFFF;
    constexpr uint32_t NEG_INF_MANT = 0xCCCCCCCC;
    constexpr uint32_t bf4_mant_per_uint32 = 8;
    constexpr uint32_t bf4_exp_per_uint32 = 4;

    constexpr uint32_t uint32_datums_per_face_row = tt::constants::FACE_WIDTH / bf4_mant_per_uint32;
    constexpr uint32_t uint32_datums_per_face = (tt::constants::FACE_HW) / bf4_mant_per_uint32;
    constexpr uint32_t uint32_exp_per_face = tt::constants::FACE_HEIGHT / bf4_exp_per_uint32;

    CircularBuffer cb(cb_id);
    volatile tt_l1_ptr uint32_t* uint32_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr() + tile_id * tile_bytes);

    for (uint32_t i = 0; i < uint32_exp_per_face * 4; i++) {
        uint32_ptr[i] = NEG_INF_EXP;
    }

    constexpr uint32_t face0_offset = uint32_exp_per_face * 4;
    constexpr uint32_t face1_offset = face0_offset + uint32_datums_per_face;
    constexpr uint32_t face2_offset = face1_offset + uint32_datums_per_face;
    constexpr uint32_t face3_offset = face2_offset + uint32_datums_per_face;

    struct FaceInfo {
        uint32_t row_start;
        uint32_t row_end;
        uint32_t col_start;
        uint32_t col_end;
        uint32_t offset;
    };

    FaceInfo faces[4] = {
        {0, 15, 0, 15, face0_offset},
        {0, 15, 16, 31, face1_offset},
        {16, 31, 0, 15, face2_offset},
        {16, 31, 16, 31, face3_offset}};

    for (uint32_t face_idx = 0; face_idx < 4; face_idx++) {
        FaceInfo& face = faces[face_idx];

        bool fully_masked_upper = (leading_diagonal_offset < 32) &&
                                  ((int32_t)face.col_start > (int32_t)face.row_end + leading_diagonal_offset);

        bool fully_masked_lower = (trailing_diagonal_offset < 32) &&
                                  ((int32_t)face.col_end < (int32_t)face.row_start - trailing_diagonal_offset);

        if (fully_masked_upper || fully_masked_lower) {
            for (uint32_t i = 0; i < uint32_datums_per_face; i++) {
                uint32_ptr[face.offset + i] = NEG_INF_MANT;
            }
            continue;
        }

        bool fully_allowed_upper = (leading_diagonal_offset >= 32) ||
                                   ((int32_t)face.col_end <= (int32_t)face.row_start + leading_diagonal_offset);
        bool fully_allowed_lower = (trailing_diagonal_offset >= 32) ||
                                   ((int32_t)face.col_start >= (int32_t)face.row_end - trailing_diagonal_offset);

        if (fully_allowed_upper && fully_allowed_lower) {
            continue;
        }

        for (uint32_t row = 0; row < tt::constants::FACE_HEIGHT; row++) {
            uint32_t global_row = face.row_start + row;
            uint32_t row_offset = row * uint32_datums_per_face_row;

            for (uint32_t col_group = 0; col_group < uint32_datums_per_face_row; col_group++) {
                uint32_t mask_value = 0;

                for (uint32_t i = 0; i < bf4_mant_per_uint32; i++) {
                    uint32_t local_col = col_group * bf4_mant_per_uint32 + i;
                    uint32_t global_col = face.col_start + local_col;

                    bool masked_upper = (leading_diagonal_offset < 32) &&
                                        ((int32_t)global_col > (int32_t)global_row + leading_diagonal_offset);
                    bool masked_lower = (trailing_diagonal_offset < 32) &&
                                        ((int32_t)global_col < (int32_t)global_row - trailing_diagonal_offset);

                    if (masked_upper || masked_lower) {
                        mask_value |= 0xC << (i * 4);
                    }
                }

                uint32_ptr[face.offset + row_offset + col_group] = mask_value;
            }
        }
    }
}

template <uint32_t tile_bytes>
void fill_vertical_tile_bfp4(Noc noc, uint32_t cb_id, uint32_t tile_id, uint32_t unpad_col_in_tile) {
    fill_tile_zeros<tile_bytes>(noc, cb_id, tile_id);

    constexpr uint32_t NEG_INF_EXP = 0xFFFFFFFF;
    constexpr uint32_t NEG_INF_MANT = 0xCCCCCCCC;
    constexpr uint32_t bf4_mant_per_uint32 = 8;
    constexpr uint32_t bf4_exp_per_uint32 = 4;

    constexpr uint32_t uint32_datums_per_face_row = tt::constants::FACE_WIDTH / bf4_mant_per_uint32;
    constexpr uint32_t uint32_datums_per_face = (tt::constants::FACE_HW) / bf4_mant_per_uint32;
    constexpr uint32_t uint32_exp_per_face = tt::constants::FACE_HEIGHT / bf4_exp_per_uint32;

    CircularBuffer cb(cb_id);
    volatile tt_l1_ptr uint32_t* uint32_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr() + tile_id * tile_bytes);

    constexpr uint32_t exp_section_size = uint32_exp_per_face * 4;
    constexpr uint32_t face0_offset = exp_section_size;
    constexpr uint32_t face1_offset = face0_offset + uint32_datums_per_face;
    constexpr uint32_t face2_offset = face1_offset + uint32_datums_per_face;
    constexpr uint32_t face3_offset = face2_offset + uint32_datums_per_face;

    for (uint32_t i = 0; i < exp_section_size; i++) {
        uint32_ptr[i] = NEG_INF_EXP;
    }

    if (unpad_col_in_tile < tt::constants::FACE_WIDTH) {
        const uint32_t unpad_col_group = unpad_col_in_tile / bf4_mant_per_uint32;
        const uint32_t unpad_col_pos = unpad_col_in_tile % bf4_mant_per_uint32;

        for (uint32_t k = 0; k < 4; k += 2) {
            uint32_t face_offset = (k == 0) ? face0_offset : face2_offset;

            for (uint32_t r = 0; r < tt::constants::FACE_HEIGHT; ++r) {
                uint32_t row_offset = r * uint32_datums_per_face_row;

                if (unpad_col_pos > 0) {
                    uint32_t mask = 0;
                    for (uint32_t i = unpad_col_pos; i < bf4_mant_per_uint32; i++) {
                        mask |= 0xC << (i * 4);
                    }
                    uint32_ptr[face_offset + row_offset + unpad_col_group] = mask;
                }

                for (uint32_t c = unpad_col_group + (unpad_col_pos > 0 ? 1 : 0); c < uint32_datums_per_face_row; ++c) {
                    uint32_ptr[face_offset + row_offset + c] = NEG_INF_MANT;
                }
            }
        }
    }

    const uint32_t unpad_col_in_right_face =
        (unpad_col_in_tile < tt::constants::FACE_WIDTH) ? 0 : unpad_col_in_tile - tt::constants::FACE_WIDTH;
    const uint32_t unpad_col_in_right_group = unpad_col_in_right_face / bf4_mant_per_uint32;
    const uint32_t unpad_col_in_right_pos = unpad_col_in_right_face % bf4_mant_per_uint32;

    for (uint32_t k = 1; k < 4; k += 2) {
        uint32_t face_offset = (k == 1) ? face1_offset : face3_offset;

        for (uint32_t r = 0; r < tt::constants::FACE_HEIGHT; ++r) {
            uint32_t row_offset = r * uint32_datums_per_face_row;

            if (unpad_col_in_right_pos > 0) {
                uint32_t mask = 0;
                for (uint32_t i = unpad_col_in_right_pos; i < bf4_mant_per_uint32; i++) {
                    mask |= 0xC << (i * 4);
                }
                uint32_ptr[face_offset + row_offset + unpad_col_in_right_group] = mask;
            }

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
void generate_causal_sliding_window_mask(
    Noc noc,
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t q_chunk,
    uint32_t k_chunk,
    bool is_causal = true,
    uint32_t sliding_window_size = 0) {
    uint32_t mask_size_tiles = Sq_chunk_t * Sk_chunk_t;
    CircularBuffer cb(cb_mask_in);
    cb.reserve_back(mask_size_tiles);

    uint32_t write_ptr_base = cb.get_write_ptr();
    constexpr uint32_t tile_bytes = get_tile_size(cb_mask_in);

    int zero_tile_idx = -1;
    int inf_tile_idx = -1;

    int32_t min_window_start, max_window_start, min_window_end, max_window_end;
    for (uint32_t q_tile = 0; q_tile < Sq_chunk_t; ++q_tile) {
        uint32_t global_q_tile = Sq_chunk_t * q_chunk + q_tile;
        uint32_t q_tile_start = global_q_tile * tt::constants::TILE_HEIGHT;
        uint32_t q_tile_end = q_tile_start + tt::constants::TILE_HEIGHT - 1;
        if (sliding_window_size > 0) {
            if (is_causal) {
                min_window_start = (int32_t)q_tile_start - (int32_t)sliding_window_size + 1;
                if (min_window_start < 0) {
                    min_window_start = 0;
                }
                max_window_start = (int32_t)q_tile_end - (int32_t)sliding_window_size + 1;
                if (max_window_start < 0) {
                    max_window_start = 0;
                }
                max_window_end = (int32_t)q_tile_end + 1;
                min_window_end = (int32_t)q_tile_start + 1;
            } else {
                int32_t half_window = (int32_t)sliding_window_size / 2;
                min_window_start = (int32_t)q_tile_start - half_window;
                if (min_window_start < 0) {
                    min_window_start = 0;
                }
                max_window_start = (int32_t)q_tile_end - half_window;
                if (max_window_start < 0) {
                    max_window_start = 0;
                }
                max_window_end = (int32_t)q_tile_end + half_window + 1;
                min_window_end = (int32_t)q_tile_start + half_window + 1;
            }
        }
        for (uint32_t k_tile = 0; k_tile < Sk_chunk_t; ++k_tile) {
            uint32_t in_mask_tile_id = q_tile * Sk_chunk_t + k_tile;
            uint32_t global_k_tile = Sk_chunk_t * k_chunk + k_tile;

            MaskType mask_type = MaskType::FULLY_ALLOWED;
            int32_t leading_diagonal_offset = 0;
            int32_t trailing_diagonal_offset = 0;
            if (sliding_window_size > 0) {
                uint32_t k_tile_start = global_k_tile * tt::constants::TILE_HEIGHT;
                uint32_t k_tile_end = k_tile_start + tt::constants::TILE_HEIGHT - 1;
                bool k_tile_outside_window =
                    ((int32_t)k_tile_end < min_window_start) || ((int32_t)k_tile_start >= max_window_end);
                if (k_tile_outside_window) {
                    mask_type = MaskType::FULLY_MASKED;
                } else {
                    bool k_tile_fully_contained =
                        ((int32_t)k_tile_start >= max_window_start) && ((int32_t)k_tile_end < min_window_end);
                    if (k_tile_fully_contained) {
                        mask_type = MaskType::FULLY_ALLOWED;
                    } else {
                        leading_diagonal_offset = (min_window_end - 1) - k_tile_start;
                        trailing_diagonal_offset = k_tile_end - max_window_start;
                        mask_type = MaskType::PARTIAL_MASK;
                    }
                }
            } else {
                if (is_causal) {
                    if (global_k_tile < global_q_tile) {
                        mask_type = MaskType::FULLY_ALLOWED;
                    } else if (global_k_tile == global_q_tile) {
                        leading_diagonal_offset = 0;
                        trailing_diagonal_offset = 32;
                        mask_type = MaskType::PARTIAL_MASK;
                    } else {
                        mask_type = MaskType::FULLY_MASKED;
                    }
                } else {
                    mask_type = MaskType::FULLY_ALLOWED;
                }
            }

            switch (mask_type) {
                case MaskType::FULLY_ALLOWED:
                    if (zero_tile_idx == -1) {
                        fill_tile_zeros<tile_bytes>(noc, cb_mask_in, in_mask_tile_id);
                        zero_tile_idx = in_mask_tile_id;
                    } else {
                        copy_tile<tile_bytes>(noc, write_ptr_base, write_ptr_base, zero_tile_idx, in_mask_tile_id);
                    }
                    break;
                case MaskType::FULLY_MASKED:
                    if (inf_tile_idx == -1) {
                        fill_neginf_tile<tile_bytes>(cb_mask_in, in_mask_tile_id);
                        inf_tile_idx = in_mask_tile_id;
                    } else {
                        copy_tile<tile_bytes>(noc, write_ptr_base, write_ptr_base, inf_tile_idx, in_mask_tile_id);
                    }
                    break;
                case MaskType::PARTIAL_MASK:
                    fill_custom_diagonal_tile_bfp4<tile_bytes>(
                        noc, cb_mask_in, in_mask_tile_id, leading_diagonal_offset, trailing_diagonal_offset);
            }
        }
    }
    noc.async_read_barrier();
    cb.push_back(mask_size_tiles);
}

template <uint32_t cb_mask_in>
void generate_noncausal_padded_mask(Noc noc, uint32_t Sq_chunk_t, uint32_t Sk_chunk_t, uint32_t unpadded_Sk) {
    uint32_t mask_size_tiles = Sq_chunk_t * Sk_chunk_t;
    CircularBuffer cb(cb_mask_in);
    cb.reserve_back(mask_size_tiles);

    uint32_t write_ptr_base = cb.get_write_ptr();
    constexpr uint32_t tile_bytes = get_tile_size(cb_mask_in);

    int zero_tile_idx = -1;
    int inf_tile_idx = -1;
    int vertical_tile_idx = -1;
    const uint32_t raw_mod = unpadded_Sk % (Sk_chunk_t * tt::constants::TILE_WIDTH);
    const uint32_t unpadded_Sk_in_chunk = (raw_mod == 0) ? (Sk_chunk_t * tt::constants::TILE_WIDTH) : raw_mod;
    uint32_t unpad_tile_col_in_chunk = unpadded_Sk_in_chunk / tt::constants::TILE_WIDTH;
    uint32_t unpad_col_in_tile = unpadded_Sk_in_chunk % tt::constants::TILE_WIDTH;

    for (uint32_t q_tile = 0; q_tile < Sq_chunk_t; ++q_tile) {
        for (uint32_t k_tile = 0; k_tile < Sk_chunk_t; ++k_tile) {
            uint32_t in_mask_tile_id = q_tile * Sk_chunk_t + k_tile;
            const bool do_zero = k_tile < unpad_tile_col_in_chunk;
            const bool do_inf =
                (k_tile > unpad_tile_col_in_chunk) || (k_tile == unpad_tile_col_in_chunk && unpad_col_in_tile == 0);

            if (do_zero) {
                if (zero_tile_idx == -1) {
                    fill_tile_zeros<tile_bytes>(noc, cb_mask_in, in_mask_tile_id);
                    zero_tile_idx = in_mask_tile_id;
                } else {
                    copy_tile<tile_bytes>(noc, write_ptr_base, write_ptr_base, zero_tile_idx, in_mask_tile_id);
                }
            } else if (do_inf) {
                if (inf_tile_idx == -1) {
                    fill_neginf_tile<tile_bytes>(cb_mask_in, in_mask_tile_id);
                    inf_tile_idx = in_mask_tile_id;
                } else {
                    copy_tile<tile_bytes>(noc, write_ptr_base, write_ptr_base, inf_tile_idx, in_mask_tile_id);
                }
            } else {
                if (vertical_tile_idx == -1) {
                    fill_vertical_tile_bfp4<tile_bytes>(noc, cb_mask_in, in_mask_tile_id, unpad_col_in_tile);
                    vertical_tile_idx = in_mask_tile_id;
                } else {
                    copy_tile<tile_bytes>(noc, write_ptr_base, write_ptr_base, vertical_tile_idx, in_mask_tile_id);
                }
            }
        }
    }
    noc.async_read_barrier();
    cb.push_back(mask_size_tiles);
}

template <typename ReaderType>
inline void issue_block_reads(
    const ReaderType& reader,
    uint32_t base_tile_id,
    uint32_t row_stride,
    uint32_t num_rows,
    uint32_t cols,
    uint32_t dst_row_origin,
    uint32_t dst_addr,
    uint32_t outer_stride,
    uint32_t inner_stride,
    uint32_t barrier_threshold,
    uint32_t& barrier_count) {
    uint32_t tile_id = base_tile_id;
    uint32_t tile_bytes;
    if constexpr (has_get_aligned_page_size_v<ReaderType>) {
        tile_bytes = reader.get_aligned_page_size();
    } else {
        tile_bytes = reader.page_size;
    }
    Noc noc;
    for (uint32_t r = 0; r < num_rows; ++r) {
        uint32_t dst = dst_addr + (dst_row_origin + r) * outer_stride;
        for (uint32_t col = 0; col < cols; ++col) {
            noc.async_read(reader, CoreLocalMem<uint32_t>(dst), tile_bytes, {.page_id = tile_id}, {});
            ++tile_id;
            dst += inner_stride;
            if (barrier_threshold > 0 && ++barrier_count == barrier_threshold) {
                noc.async_read_barrier();
                barrier_count = 0;
            }
        }
        tile_id += row_stride - cols;
    }
}

template <typename ReaderType>
inline void zero_fill_block(
    const ReaderType& reader,
    uint32_t num_rows,
    uint32_t cols,
    uint32_t dst_row_origin,
    uint32_t dst_cb_id,
    uint32_t dst_addr,
    uint32_t outer_stride,
    uint32_t inner_stride) {
    Noc noc;
    uint32_t page_size;
    if constexpr (has_get_aligned_page_size_v<ReaderType>) {
        page_size = reader.get_aligned_page_size();
    } else {
        page_size = reader.page_size;
    }
    for (uint32_t r = 0; r < num_rows; ++r) {
        uint32_t dst = dst_addr + (dst_row_origin + r) * outer_stride;
        for (uint32_t col = 0; col < cols; ++col) {
            fill_zeros_async(noc, dst_cb_id, page_size, dst - dst_addr);
            dst += inner_stride;
        }
    }
}

template <typename WriterType>
inline void issue_block_writes(
    Noc noc,
    const WriterType& writer,
    uint32_t base_tile_id,
    uint32_t row_stride,
    uint32_t num_rows,
    uint32_t cols,
    uint32_t src_row_origin,
    uint32_t src_addr,
    uint32_t outer_stride,
    uint32_t inner_stride,
    uint32_t trid = 0) {
    uint32_t tile_id = base_tile_id;
    uint32_t tile_bytes;
    if constexpr (has_get_aligned_page_size_v<WriterType>) {
        tile_bytes = writer.get_aligned_page_size();
    } else {
        tile_bytes = writer.page_size;
    }
    for (uint32_t r = 0; r < num_rows; ++r) {
        uint32_t src = src_addr + (src_row_origin + r) * outer_stride;
        for (uint32_t col = 0; col < cols; ++col) {
            noc.async_write<NocOptions::TXN_ID>(
                CoreLocalMem<uint32_t>(src), writer, tile_bytes, {}, {.page_id = tile_id}, {.trid = trid});
            ++tile_id;
            src += inner_stride;
        }
        tile_id += row_stride - cols;
    }
}

struct Slice {
    uint32_t d0;
    uint32_t d1;

    uint32_t d2_start;
    uint32_t d2_end;
    uint32_t d3_start;
    uint32_t d3_end;

    Slice() = default;

    Slice(uint32_t d0, uint32_t d1, uint32_t d2_start, uint32_t d2_end, uint32_t d3_start, uint32_t d3_end) :
        d0(d0), d1(d1), d2_start(d2_start), d2_end(d2_end), d3_start(d3_start), d3_end(d3_end) {}

    uint32_t get_d2_size() const { return d2_end - d2_start; }
    uint32_t get_d3_size() const { return d3_end - d3_start; }
};

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

    void issue_reads(
        const Slice& slice,
        uint32_t,
        uint32_t dst_cb_id,
        uint32_t dst_addr,
        uint32_t outer_stride,
        uint32_t inner_stride,
        uint32_t barrier_threshold) const {
        const uint32_t d2_start = slice.d2_start;
        const uint32_t d2_end = slice.d2_end;
        const uint32_t cols = slice.get_d3_size();
        const uint32_t first_end = first_shape.shape[2];
        const uint32_t gap_end = first_seq_padded;
        const uint32_t second_end = first_seq_padded + second_shape.shape[2];
        uint32_t barrier_count = 0;

        const uint32_t s0_end = std::min(d2_end, first_end);
        if (d2_start < s0_end) {
            issue_block_reads(
                first_reader,
                first_shape.id_of(slice.d0, slice.d1, d2_start, slice.d3_start),
                first_shape.strides[2],
                s0_end - d2_start,
                cols,
                0,
                dst_addr,
                outer_stride,
                inner_stride,
                barrier_threshold,
                barrier_count);
        }
        const uint32_t s1_start = std::max(d2_start, first_end);
        const uint32_t s1_end = std::min(d2_end, gap_end);
        if (s1_start < s1_end) {
            zero_fill_block(
                first_reader,
                s1_end - s1_start,
                cols,
                s1_start - d2_start,
                dst_cb_id,
                dst_addr,
                outer_stride,
                inner_stride);
        }
        const uint32_t s2_start = std::max(d2_start, gap_end);
        const uint32_t s2_end = std::min(d2_end, second_end);
        if (s2_start < s2_end) {
            issue_block_reads(
                second_reader,
                second_shape.id_of(slice.d0, slice.d1, s2_start - first_seq_padded, slice.d3_start),
                second_shape.strides[2],
                s2_end - s2_start,
                cols,
                s2_start - d2_start,
                dst_addr,
                outer_stride,
                inner_stride,
                barrier_threshold,
                barrier_count);
        }
        const uint32_t s3_start = std::max(d2_start, second_end);
        if (s3_start < d2_end) {
            zero_fill_block(
                first_reader,
                d2_end - s3_start,
                cols,
                s3_start - d2_start,
                dst_cb_id,
                dst_addr,
                outer_stride,
                inner_stride);
        }
    }

    void issue_writes(
        Noc noc,
        const Slice& slice,
        uint32_t,
        uint32_t src_addr,
        uint32_t outer_stride,
        uint32_t inner_stride,
        uint32_t trid = 0) const {
        const uint32_t d2_start = slice.d2_start;
        const uint32_t d2_end = slice.d2_end;
        const uint32_t cols = slice.get_d3_size();
        const uint32_t first_end = first_shape.shape[2];
        const uint32_t gap_end = first_seq_padded;
        const uint32_t second_end = first_seq_padded + second_shape.shape[2];

        const uint32_t s0_end = std::min(d2_end, first_end);
        if (d2_start < s0_end) {
            issue_block_writes(
                noc,
                first_reader,
                first_shape.id_of(slice.d0, slice.d1, d2_start, slice.d3_start),
                first_shape.strides[2],
                s0_end - d2_start,
                cols,
                0,
                src_addr,
                outer_stride,
                inner_stride,
                trid);
        }
        const uint32_t s2_start = std::max(d2_start, gap_end);
        const uint32_t s2_end = std::min(d2_end, second_end);
        if (s2_start < s2_end) {
            issue_block_writes(
                noc,
                second_reader,
                second_shape.id_of(slice.d0, slice.d1, s2_start - first_seq_padded, slice.d3_start),
                second_shape.strides[2],
                s2_end - s2_start,
                cols,
                s2_start - d2_start,
                src_addr,
                outer_stride,
                inner_stride,
                trid);
        }
    }
};

template <typename ReaderType, typename TensorShapeType = TensorTileShape>
struct PaddedAddrGenerator {
    ReaderType reader;
    TensorShapeType tensor_shape;

    PaddedAddrGenerator(const ReaderType& reader, TensorShapeType tensor_shape) :
        reader(reader), tensor_shape(tensor_shape) {}

    void issue_reads(
        const Slice& slice,
        uint32_t end_seq_tile,
        uint32_t dst_cb_id,
        uint32_t dst_addr,
        uint32_t outer_stride,
        uint32_t inner_stride,
        uint32_t barrier_threshold) const {
        const uint32_t d2_start = slice.d2_start;
        const uint32_t rows = slice.get_d2_size();
        const uint32_t cols = slice.get_d3_size();
        const uint32_t shape_d2 = tensor_shape.d2();
        const uint32_t bound = shape_d2 < end_seq_tile ? shape_d2 : end_seq_tile;
        const uint32_t valid_rows = (d2_start >= bound) ? 0 : std::min(rows, bound - d2_start);
        uint32_t barrier_count = 0;

        issue_block_reads(
            reader,
            tensor_shape.id_of(slice.d0, slice.d1, d2_start, slice.d3_start),
            tensor_shape.stride2(),
            valid_rows,
            cols,
            0,
            dst_addr,
            outer_stride,
            inner_stride,
            barrier_threshold,
            barrier_count);
        zero_fill_block(reader, rows - valid_rows, cols, valid_rows, dst_cb_id, dst_addr, outer_stride, inner_stride);
    }

    void issue_writes(
        Noc noc,
        const Slice& slice,
        uint32_t end_seq_tile,
        uint32_t src_addr,
        uint32_t outer_stride,
        uint32_t inner_stride,
        uint32_t trid = 0) const {
        const uint32_t d2_start = slice.d2_start;
        const uint32_t rows = slice.get_d2_size();
        const uint32_t cols = slice.get_d3_size();
        const uint32_t shape_d2 = tensor_shape.d2();
        const uint32_t bound = shape_d2 < end_seq_tile ? shape_d2 : end_seq_tile;
        const uint32_t valid_rows = (d2_start >= bound) ? 0 : std::min(rows, bound - d2_start);

        issue_block_writes(
            noc,
            reader,
            tensor_shape.id_of(slice.d0, slice.d1, d2_start, slice.d3_start),
            tensor_shape.stride2(),
            valid_rows,
            cols,
            0,
            src_addr,
            outer_stride,
            inner_stride,
            trid);
    }

    void issue_writes_no_padding(
        Noc noc, const Slice& slice, uint32_t src_addr, uint32_t outer_stride, uint32_t inner_stride, uint32_t trid = 0)
        const {
        issue_block_writes(
            noc,
            reader,
            tensor_shape.id_of(slice.d0, slice.d1, slice.d2_start, slice.d3_start),
            tensor_shape.stride2(),
            slice.get_d2_size(),
            slice.get_d3_size(),
            0,
            src_addr,
            outer_stride,
            inner_stride,
            trid);
    }
};

template <typename ReaderType, typename TensorShapeType>
PaddedAddrGenerator(const ReaderType&, TensorShapeType) -> PaddedAddrGenerator<ReaderType, TensorShapeType>;

template <typename CatAddrGeneratorType>
__attribute__((noinline)) void fetch_block(
    const CatAddrGeneratorType& cat_addr_generator,
    const Slice& src_slice,
    const uint32_t end_seq_tile,
    const uint32_t dst_cb_id,
    const uint32_t dst_addr,
    const uint32_t tile_bytes,
    const bool transpose,
    const uint32_t barrier_threshold = 0) {
    Noc noc;
    const uint32_t src_rows = src_slice.get_d2_size();
    const uint32_t src_cols = src_slice.get_d3_size();
    const uint32_t outer_ptr_stride = transpose ? tile_bytes : src_cols * tile_bytes;
    const uint32_t inner_ptr_stride = transpose ? tile_bytes * src_rows : tile_bytes;

    cat_addr_generator.issue_reads(
        src_slice, end_seq_tile, dst_cb_id, dst_addr, outer_ptr_stride, inner_ptr_stride, barrier_threshold);
    noc.async_read_barrier();
    noc.write_zeros_l1_barrier();
}

template <typename CatAddrGeneratorType>
void read_block(
    const CatAddrGeneratorType& cat_addr_generator,
    const Slice& src_slice,
    const uint32_t end_seq_tile,
    const uint32_t cb_id,
    const uint32_t tile_bytes,
    const bool transpose,
    const uint32_t barrier_threshold = 0) {
    const uint32_t num_tiles = src_slice.get_d2_size() * src_slice.get_d3_size();
    CircularBuffer cb(cb_id);
    cb.reserve_back(num_tiles);
    fetch_block(
        cat_addr_generator,
        src_slice,
        end_seq_tile,
        cb_id,
        cb.get_write_ptr(),
        tile_bytes,
        transpose,
        barrier_threshold);
    cb.push_back(num_tiles);
}

template <typename CatAddrGeneratorType>
void write_block(
    Noc noc,
    const CatAddrGeneratorType& cat_addr_generator,
    const Slice& dst_slice,
    const uint32_t end_seq_tile,
    const uint32_t cb_id,
    const uint32_t tile_bytes) {
    const uint32_t dst_rows = dst_slice.get_d2_size();
    const uint32_t dst_cols = dst_slice.get_d3_size();
    const uint32_t num_tiles = dst_rows * dst_cols;
    CircularBuffer cb(cb_id);
    const uint32_t outer_ptr_stride = dst_cols * tile_bytes;
    const uint32_t inner_ptr_stride = tile_bytes;

    cb.wait_front(num_tiles);
    cat_addr_generator.issue_writes(
        noc, dst_slice, end_seq_tile, cb.get_read_ptr(), outer_ptr_stride, inner_ptr_stride);
    noc.async_write_barrier();
    cb.pop_front(num_tiles);
}

FORCE_INLINE uint32_t head_wrap_tile_skip(uint32_t cols, uint32_t row_stride, uint32_t head_rows) {
    return cols - head_rows * row_stride;
}

template <uint32_t head_rows = 0, typename TensorAccessorType>
void write_block(
    Noc noc,
    const TensorAccessorType& out_writer,
    const uint32_t cb_out,
    const uint32_t out_chunk_tiles,
    const uint32_t rows,
    const uint32_t cols,
    const uint32_t out_tile_id,
    const uint32_t tile_bytes,
    const uint32_t barrier_threshold,
    const uint32_t row_stride = 0,
    const uint32_t first_row_in_head = 0) {
    uint32_t barrier_count = 0;
    uint32_t tile_id = out_tile_id;
    const uint32_t row_skip = (row_stride ? row_stride : cols) - cols;
    [[maybe_unused]] const uint32_t head_skip = head_wrap_tile_skip(cols, row_stride ? row_stride : cols, head_rows);
    [[maybe_unused]] uint32_t row_in_head = first_row_in_head;

    CircularBuffer cb(cb_out);
    cb.wait_front(out_chunk_tiles);

    uint32_t tile_offset = 0;
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t col = 0; col < cols; ++col) {
            noc.async_write(cb, out_writer, tile_bytes, {.offset_bytes = tile_offset}, {.page_id = tile_id});
            ++tile_id;
            tile_offset += tile_bytes;

            if (++barrier_count == barrier_threshold) {
                noc.async_writes_flushed();
                barrier_count = 0;
            }
        }
        tile_id += row_skip;
        if constexpr (head_rows != 0) {
            if (++row_in_head == head_rows) {
                row_in_head = 0;
                tile_id += head_skip;
            }
        }
    }
    noc.async_write_barrier();
    cb.pop_front(out_chunk_tiles);
}

template <uint32_t head_rows = 0, typename TensorAccessorType>
void write_block_row_grouped(
    Noc noc,
    const TensorAccessorType& out_writer,
    const uint32_t cb_out,
    const uint32_t total_rows,
    const uint32_t write_rows,
    const uint32_t cols,
    const uint32_t out_tile_id,
    const uint32_t tile_bytes,
    const uint32_t sbh,
    const uint32_t barrier_threshold,
    const uint32_t row_stride = 0,
    const uint32_t first_row_in_head = 0) {
    constexpr uint32_t default_trid = 0;
    uint32_t tile_id = out_tile_id;
    uint32_t barrier_count = 0;
    const uint32_t row_skip = (row_stride ? row_stride : cols) - cols;
    [[maybe_unused]] const uint32_t head_skip = head_wrap_tile_skip(cols, row_stride ? row_stride : cols, head_rows);
    [[maybe_unused]] uint32_t row_in_head = first_row_in_head;

    const uint32_t num_full_groups = total_rows / sbh;
    const uint32_t remainder_rows = total_rows - num_full_groups * sbh;
    const uint32_t num_groups = num_full_groups + (remainder_rows ? 1 : 0);

    CircularBuffer cb(cb_out);
    for (uint32_t rg = 0; rg < num_groups; ++rg) {
        const uint32_t rows_this_group = (rg < num_full_groups) ? sbh : remainder_rows;
        const uint32_t tiles_this_group = rows_this_group * cols;
        cb.wait_front(tiles_this_group);
        for (uint32_t r = 0; r < rows_this_group; ++r) {
            const uint32_t row = rg * sbh + r;
            if (row < write_rows) {
                for (uint32_t col = 0; col < cols; ++col) {
                    uint32_t tile_offset = (r * cols + col) * tile_bytes;
                    noc.async_write(cb, out_writer, tile_bytes, {.offset_bytes = tile_offset}, {.page_id = tile_id});
                    ++tile_id;
                    if (++barrier_count == barrier_threshold) {
                        noc.async_writes_flushed<NocOptions::TXN_ID>({.trid = default_trid});
                        barrier_count = 0;
                    }
                }
                tile_id += row_skip;
                if constexpr (head_rows != 0) {
                    if (++row_in_head == head_rows) {
                        row_in_head = 0;
                        tile_id += head_skip;
                    }
                }
            }
        }
        noc.async_writes_flushed<NocOptions::TXN_ID>({.trid = default_trid});
        cb.pop_front(tiles_this_group);
    }
    noc.async_write_barrier();
}

template <bool all_rows_valid = false, typename CatAddrGeneratorType>
void write_block_row_grouped_trid(
    Noc noc,
    const CatAddrGeneratorType& cat_addr_generator,
    const Slice& dst_slice,
    const uint32_t end_seq_tile,
    const uint32_t cb_out,
    const uint32_t tile_bytes,
    const uint32_t sbh,
    const uint32_t flush_trid) {
    const uint32_t total_rows = dst_slice.get_d2_size();
    const uint32_t cols = dst_slice.get_d3_size();
    const uint32_t outer_stride = cols * tile_bytes;

    const uint32_t num_full_groups = total_rows / sbh;
    const uint32_t remainder_rows = total_rows - num_full_groups * sbh;
    const uint32_t num_groups = num_full_groups + (remainder_rows ? 1 : 0);

    CircularBuffer cb(cb_out);
    for (uint32_t rg = 0; rg < num_groups; ++rg) {
        const uint32_t rows_this_group = (rg < num_full_groups) ? sbh : remainder_rows;
        const uint32_t tiles_this_group = rows_this_group * cols;
        cb.wait_front(tiles_this_group);
        const Slice group_slice(
            dst_slice.d0,
            dst_slice.d1,
            dst_slice.d2_start + rg * sbh,
            dst_slice.d2_start + rg * sbh + rows_this_group,
            dst_slice.d3_start,
            dst_slice.d3_end);
        if constexpr (all_rows_valid) {
            cat_addr_generator.issue_writes_no_padding(
                noc, group_slice, cb.get_read_ptr(), outer_stride, tile_bytes, flush_trid);
        } else {
            cat_addr_generator.issue_writes(
                noc, group_slice, end_seq_tile, cb.get_read_ptr(), outer_stride, tile_bytes, flush_trid);
        }
        noc.async_writes_flushed<NocOptions::TXN_ID>({.trid = flush_trid});
        cb.pop_front(tiles_this_group);
    }
}

template <uint32_t tile_bytes>
void fill_attention_sink_tiles(uint32_t cb_id, uint32_t num_tiles, uint32_t source_tile_addr) {
    volatile tt_l1_ptr uint16_t* source_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(source_tile_addr);
    uint16_t sink_value = source_ptr[0];

    constexpr uint16_t neg_inf_bf16 = 0xFF80;

    CircularBuffer cb(cb_id);
    uint32_t write_ptr = cb.get_write_ptr();

    constexpr uint32_t face_height = 16;
    constexpr uint32_t face_width = 16;
    constexpr uint32_t elements_per_face = face_height * face_width;
    constexpr uint32_t elements_per_tile = 4 * elements_per_face;

    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        volatile tt_l1_ptr uint16_t* tile_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(write_ptr);

        for (uint32_t i = 0; i < elements_per_tile; ++i) {
            tile_ptr[i] = neg_inf_bf16;
        }

        for (uint32_t face = 0; face < 4; face += 2) {
            uint32_t face_offset = face * elements_per_face;

            for (uint32_t row = 0; row < face_height; ++row) {
                uint32_t row_offset = row * face_width;
                tile_ptr[face_offset + row_offset] = sink_value;
            }
        }

        write_ptr += tile_bytes;
    }
}

template <bool is_chunked, uint32_t sliding_window_size, bool padded_or_joint_masks, uint32_t cb_mask_in>
void generate_mask(
    Noc noc,
    const uint32_t Sq_chunk_t,
    const uint32_t Sk_chunk_t,
    const uint32_t q_chunk,
    const uint32_t chunk_start_t_in_q_chunks,
    const bool generate_mask_0,
    const bool generate_mask_1,
    const uint32_t unpadded_Sk_mask_0,
    const uint32_t unpadded_Sk_mask_1,
    const bool is_causal) {
    if (is_causal || sliding_window_size > 0) {
        uint32_t offset_q_chunk = q_chunk;
        if constexpr (is_chunked) {
            offset_q_chunk += chunk_start_t_in_q_chunks;
        }
        uint32_t q_low_idx = offset_q_chunk * Sq_chunk_t;
        uint32_t q_high_idx = q_low_idx + Sq_chunk_t;

        const uint32_t mask_k_end =
            is_causal ? q_high_idx : (unpadded_Sk_mask_0 + tt::constants::TILE_HEIGHT - 1) / tt::constants::TILE_HEIGHT;
        for (uint32_t k_chunk = 0; (k_chunk * Sk_chunk_t) < mask_k_end; ++k_chunk) {
            const uint32_t k_low_idx = k_chunk * Sk_chunk_t;
            const uint32_t k_high_idx = k_low_idx + Sk_chunk_t;
            if (!(q_low_idx >= k_high_idx) || sliding_window_size > 0) {
                generate_causal_sliding_window_mask<cb_mask_in>(
                    noc, Sq_chunk_t, Sk_chunk_t, offset_q_chunk, k_chunk, is_causal, sliding_window_size);
            }
        }
    } else if constexpr (padded_or_joint_masks) {
        if (generate_mask_0) {
            generate_noncausal_padded_mask<cb_mask_in>(noc, Sq_chunk_t, Sk_chunk_t, unpadded_Sk_mask_0);
        }
        if (generate_mask_1) {
            generate_noncausal_padded_mask<cb_mask_in>(noc, Sq_chunk_t, Sk_chunk_t, unpadded_Sk_mask_1);
        }
    }
}
