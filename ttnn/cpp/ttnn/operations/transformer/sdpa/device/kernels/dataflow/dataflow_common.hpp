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
#include <tt-metalium/constants.hpp>
#include "api/debug/assert.h"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/q_chunk_remapping.hpp"
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

// Convenience overload for callers that don't already have a Noc instance in scope.
template <uint32_t tile_bytes, bool wait_for_barrier = true>
void fill_tile_zeros(uint32_t cb_id, uint32_t tile_id) {
    Noc noc;
    fill_tile_zeros<tile_bytes, wait_for_barrier>(noc, cb_id, tile_id);
}

// capacity_t = 0 means "no wrap" (legacy / unbounded cache).  Nonzero means the cache
// holds capacity_t tile-rows in a circular buffer; the kernel wraps seq_tile_idx
// modulo capacity_t before resolving the page_table entry, so callers can pass
// absolute (un-wrapped) positions even with bounded sliding-window allocations.
// capacity_t must be a multiple of block_size_t (validated on the caller side); that
// guarantees the wrap preserves intra-block offsets.
template <typename PageT, uint32_t num_heads, uint32_t block_size_t, uint32_t Wt, uint32_t capacity_t = 0>
uint32_t virtual_seq_tile_id_to_physical_tile_id(
    uint32_t seq_tile_idx, uint32_t cur_head, const volatile tt_l1_ptr PageT* const page_table_ptr) {
    // Given some index in the sequence tiles in range [0, max_seq_len_t]
    // Return the physical tile id for that tile row
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
    // Third argument page_size from runtime args overrides TensorAccessorArgs::AlignedPageSize, which may be stale on
    // program cache hits.
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
    /*
      Method always reads tiles from memory in row-major order.
      It assumes that the block of rows x cols in stored in contiguous tile order.
      That means, it won't work if the chunk to read is a slice of the last dimension.

      This handles the case where the dst CB is larger than the src CB, with some padding on the
      rows or cols of the DST CB.
    */
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

    // Zero out the padding
    for (uint32_t row = 0; row < dst_rows; ++row) {
        for (uint32_t col = 0; col < dst_cols; ++col) {
            if (row < src_rows && col < src_cols) {
                continue;
            }
            uint32_t tile_id = transpose ? col * dst_rows + row : row * dst_cols + col;
            fill_tile_zeros<tile_bytes, false>(noc, cb_id, tile_id);
        }
    }
    // NOC reads and async_write_zeros use the same completion path on WH/BH but different
    // paths on Quasar (NOC channels vs iDMA). Issue both — second is a no-op on WH/BH.
    noc.async_read_barrier();
    noc.write_zeros_l1_barrier();

    if constexpr (push_num_tiles) {
        cb.push_back(num_tiles);
        return 0;
    } else {
        return num_tiles;
    }
}

// Read subblock_h rows of Q tiles and push to CB.
// start_tile_id is passed by reference and advances across successive calls.
// Used for interleaved Q subblock push to overlap Q reads with compute.
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
            // Zero-pad extra columns (src_cols < dst_cols case)
            for (uint32_t col = src_cols; col < dst_cols; ++col) {
                fill_tile_zeros<tile_bytes, false>(noc, cb_id, local_row * dst_cols + col);
            }
        } else {
            // Entire row is padding
            for (uint32_t col = 0; col < dst_cols; ++col) {
                fill_tile_zeros<tile_bytes, false>(noc, cb_id, local_row * dst_cols + col);
            }
        }
    }

    // NOC reads and async_write_zeros use the same completion path on WH/BH but different
    // paths on Quasar (NOC channels vs iDMA). Issue both — second is a no-op on WH/BH.
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

    // Stride calculation based on transpose flag
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
        physical_tile_id += skip_src_cols;  // Skip src cols if needed
    }

    // Zero out the padding
    for (uint32_t row = 0; row < dst_rows; ++row) {
        for (uint32_t col = 0; col < dst_cols; ++col) {
            if (row < src_rows && col < src_cols) {
                continue;
            }
            uint32_t tile_id = transpose ? col * dst_rows + row : row * dst_cols + col;
            fill_zeros_async(noc, cb_id, tile_bytes, tile_id * tile_bytes);
        }
    }
    // NOC reads and async_write_zeros use the same completion path on WH/BH but different
    // paths on Quasar (NOC channels vs iDMA). Issue both — second is a no-op on WH/BH.
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

// Generic fill with -inf that works for all supported mask formats (bfp4, bfp8, bfloat16)
template <uint32_t tile_bytes>
void fill_neginf_tile(uint32_t cb_id, uint32_t tile_id) {
    constexpr uint32_t num_exponents = tt::constants::FACE_HEIGHT * (tt::constants::TILE_HW / tt::constants::FACE_HW);
    constexpr uint32_t bfp4_size = num_exponents + tt::constants::TILE_HW / 2;
    constexpr uint32_t bfp8_size = num_exponents + tt::constants::TILE_HW;
    constexpr uint32_t bf16_size = tt::constants::TILE_HW * 2;

    CircularBuffer cb(cb_id);
    uint32_t write_addr = cb.get_write_ptr() + tile_id * tile_bytes;
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_addr);
    constexpr uint32_t total_words = tile_bytes / sizeof(uint32_t);

    if constexpr (tile_bytes == bf16_size) {
        // BFLOAT16: fill with 0xFF80FF80 (-inf in bf16, two values per word)
        for (uint32_t i = 0; i < total_words; i++) {
            ptr[i] = 0xFF80FF80;
        }
    } else {
        // BFP formats (bfp4, bfp8): first 64 bytes are exponents, rest are mantissas
        constexpr uint32_t exp_words = num_exponents / sizeof(uint32_t);
        for (uint32_t i = 0; i < exp_words; i++) {
            ptr[i] = 0xFFFFFFFF;
        }
        // bfp4: 0xCC per nibble (sign + magnitude), bfp8: 0x80 per byte (sign bit set)
        constexpr uint32_t mant_pattern = (tile_bytes == bfp4_size) ? 0xCCCCCCCC : 0x80808080;
        for (uint32_t i = exp_words; i < total_words; i++) {
            ptr[i] = mant_pattern;
        }
    }
}

/**
 * Fill a bf16 tile with a vertical partial mask: columns [0, unpad_col) = 0, columns [unpad_col, 32) = -inf.
 * Used for lightweight mask when padding boundary falls inside a tile.
 *
 * bf16 tile layout: 4 faces of 16x16, each face row-major with 8 uint32 words per row (2 bf16 per uint32).
 * Face 0: rows[0:16], cols[0:16]   Face 1: rows[0:16], cols[16:32]
 * Face 2: rows[16:32], cols[0:16]  Face 3: rows[16:32], cols[16:32]
 *
 * Within each uint32 word: low 16 bits = even column, high 16 bits = odd column.
 */
template <uint32_t tile_bytes>
void fill_vertical_tile_bf16(Noc noc, uint32_t cb_id, uint32_t tile_id, uint32_t unpad_col_in_tile) {
    // Start with all zeros (valid)
    fill_tile_zeros<tile_bytes>(noc, cb_id, tile_id);

    constexpr uint32_t NEGINF_PAIR = 0xFF80FF80;
    constexpr uint32_t bf16_per_uint32 = 2;
    constexpr uint32_t uint32_per_face_row = tt::constants::FACE_WIDTH / bf16_per_uint32;  // 8
    constexpr uint32_t uint32_per_face = tt::constants::FACE_HW / bf16_per_uint32;         // 128

    CircularBuffer cb(cb_id);
    volatile tt_l1_ptr uint32_t* ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr() + tile_id * tile_bytes);

    // Face offsets in uint32 words
    constexpr uint32_t face_offsets[4] = {
        0,                    // Face 0: rows[0:16], cols[0:16]
        uint32_per_face,      // Face 1: rows[0:16], cols[16:32]
        2 * uint32_per_face,  // Face 2: rows[16:32], cols[0:16]
        3 * uint32_per_face   // Face 3: rows[16:32], cols[16:32]
    };
    constexpr uint32_t face_col_starts[4] = {0, 16, 0, 16};

    for (uint32_t f = 0; f < 4; f++) {
        uint32_t face_col_start = face_col_starts[f];

        if (unpad_col_in_tile <= face_col_start) {
            // Entire face is padded -> fill with -inf
            for (uint32_t i = 0; i < uint32_per_face; i++) {
                ptr[face_offsets[f] + i] = NEGINF_PAIR;
            }
        } else if (unpad_col_in_tile >= face_col_start + tt::constants::FACE_WIDTH) {
            // Entire face is valid -> already zeros, skip
        } else {
            // Partial face: boundary falls within this face
            uint32_t local_col = unpad_col_in_tile - face_col_start;  // 0-15
            uint32_t boundary_word = local_col / bf16_per_uint32;     // which uint32
            uint32_t boundary_pos = local_col % bf16_per_uint32;      // position within uint32

            for (uint32_t row = 0; row < tt::constants::FACE_HEIGHT; row++) {
                uint32_t row_base = face_offsets[f] + row * uint32_per_face_row;

                // Handle boundary word (may have one valid + one -inf bf16)
                if (boundary_pos != 0) {
                    // Low 16 bits = even col (valid, 0), high 16 bits = odd col (-inf, 0xFF80)
                    ptr[row_base + boundary_word] = 0xFF800000;
                }

                // Fill remaining words with -inf
                uint32_t start_word = boundary_word + (boundary_pos != 0 ? 1 : 0);
                for (uint32_t w = start_word; w < uint32_per_face_row; w++) {
                    ptr[row_base + w] = NEGINF_PAIR;
                }
            }
        }
    }
}

/**
 * Fill a bf16 tile with the standard causal diagonal mask pattern:
 * row r, col c: 0 if c <= r, -inf if c > r.
 *
 * bf16 tile layout: 4 faces of 16x16.
 * Face 0 (rows 0-15, cols 0-15): diagonal — row r has cols 0..r valid, r+1..15 masked
 * Face 1 (rows 0-15, cols 16-31): entirely -inf (cols > rows for these rows)
 * Face 2 (rows 16-31, cols 0-15): entirely zero (cols < rows for these rows)
 * Face 3 (rows 16-31, cols 16-31): same diagonal pattern as face 0 (shifted by 16 in both dims)
 */
template <uint32_t tile_bytes>
void fill_causal_diagonal_tile_bf16(Noc noc, uint32_t cb_id, uint32_t tile_id) {
    // Start with all zeros
    fill_tile_zeros<tile_bytes>(noc, cb_id, tile_id);

    constexpr uint32_t NEGINF_PAIR = 0xFF80FF80;
    constexpr uint32_t bf16_per_uint32 = 2;
    constexpr uint32_t uint32_per_face_row = tt::constants::FACE_WIDTH / bf16_per_uint32;  // 8
    constexpr uint32_t uint32_per_face = tt::constants::FACE_HW / bf16_per_uint32;         // 128

    CircularBuffer cb(cb_id);
    volatile tt_l1_ptr uint32_t* ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr() + tile_id * tile_bytes);

    // Face offsets in uint32 words
    constexpr uint32_t face_offsets[4] = {
        0,                    // Face 0: rows[0:16], cols[0:16]
        uint32_per_face,      // Face 1: rows[0:16], cols[16:32]
        2 * uint32_per_face,  // Face 2: rows[16:32], cols[0:16]
        3 * uint32_per_face   // Face 3: rows[16:32], cols[16:32]
    };

    // Face 1: entirely -inf (cols 16-31 > rows 0-15)
    for (uint32_t i = 0; i < uint32_per_face; i++) {
        ptr[face_offsets[1] + i] = NEGINF_PAIR;
    }

    // Face 2: entirely zero — already done by fill_tile_zeros

    // Face 0 and Face 3: diagonal pattern (identical — both have local row r with cols 0..r valid, r+1..15 masked)
    constexpr uint32_t diag_faces[2] = {0, 3};
    for (uint32_t f = 0; f < 2; f++) {
        uint32_t face_base = face_offsets[diag_faces[f]];
        for (uint32_t row = 0; row < tt::constants::FACE_HEIGHT; row++) {
            // First masked column in this row: row + 1
            uint32_t first_masked = row + 1;
            if (first_masked >= tt::constants::FACE_WIDTH) {
                continue;  // Row 15: all cols 0-15 are valid
            }

            uint32_t row_base = face_base + row * uint32_per_face_row;
            uint32_t boundary_word = first_masked / bf16_per_uint32;
            uint32_t boundary_pos = first_masked % bf16_per_uint32;

            // Boundary word: low bf16 may be valid (col = first_masked - 1 = row)
            if (boundary_pos != 0) {
                // Low 16 bits = col row (valid, 0), high 16 bits = col row+1 (masked, -inf)
                ptr[row_base + boundary_word] = 0xFF800000;
                boundary_word++;
            }

            // Remaining words: all -inf
            for (uint32_t w = boundary_word; w < uint32_per_face_row; w++) {
                ptr[row_base + w] = NEGINF_PAIR;
            }
        }
    }
}

/**
 * Fill a bf16 tile with a diagonal-edge mask pattern relative to `boundary_col = row + diagonal_offset`:
 *   - leading edge (leading_edge=true):  col c is -inf if c < boundary_col
 *   - trailing edge (leading_edge=false): col c is -inf if c > boundary_col
 */
template <uint32_t tile_bytes, int32_t diagonal_offset = 0, bool leading_edge = true>
void fill_diagonal_edge_tile_bf16(Noc noc, uint32_t cb_id, uint32_t tile_id) {
    fill_tile_zeros<tile_bytes>(noc, cb_id, tile_id);

    constexpr uint32_t neginf_bf16 = 0xFF80;
    constexpr uint32_t bf16_per_uint32 = 2;
    constexpr uint32_t uint32_per_face_row = tt::constants::FACE_WIDTH / bf16_per_uint32;  // 8
    constexpr uint32_t uint32_per_face = tt::constants::FACE_HW / bf16_per_uint32;         // 128

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

/**
 * Emit the four sliding-window edge tiles (trailing_primary, leading_prev, leading_current,
 * trailing_next) starting at `start_tile_idx` in the mask CB.
 *
 * The per-tile diagonal offsets place each edge's transition at the correct sub-tile column.
 * `leading_remainder`/`trailing_remainder` are the window edges' positions within a 32-row tile;
 * when they are 0 the window is tile-aligned and the prev/next straddle tiles go unused. The
 * causal window has only a leading (left) edge — its trailing tile reuses the plain causal diagonal.
 *
 * NOTE: this is per-row *stamp* geometry (floor + remainder), intentionally distinct from the
 * ceil-based loop bounds in SlidingWindowLoopGeometry. See sliding_window_geometry.hpp.
 */
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

    // Tile 0: trailing/right edge. Causal uses the normal causal diagonal.
    fill_diagonal_edge_tile_bf16<tile_bytes, trailing_primary_offset, /*leading_edge=*/false>(
        noc, cb_mask_in, start_tile_idx);
    // Tile 1/2: left edge when it straddles two K tiles; tile 1 is unused for aligned windows.
    fill_diagonal_edge_tile_bf16<tile_bytes, leading_prev_offset, /*leading_edge=*/true>(
        noc, cb_mask_in, start_tile_idx + 1);
    fill_diagonal_edge_tile_bf16<tile_bytes, leading_current_offset, /*leading_edge=*/true>(
        noc, cb_mask_in, start_tile_idx + 2);
    // Tile 3: non-causal right edge when it straddles into the next K tile; unused for causal/aligned windows.
    fill_diagonal_edge_tile_bf16<tile_bytes, trailing_next_offset, /*leading_edge=*/false>(
        noc, cb_mask_in, start_tile_idx + 3);
}

/**
 * Generate lightweight mask tiles into a single CB for ring joint SDPA.
 * Layout without sliding: [neginf_tile(0)] [causal_diag_tile?] [global_n_partial_tile?] [joint_l_partial_tile?]
 * Layout with sliding:    [neginf_tile(0)] [trailing_primary(1)] [leading_prev(2)]
 *                         [leading_current(3)] [trailing_next(4)] [partial tiles...]
 * Tiles are pushed once and stay permanently fronted for the entire kernel lifetime.
 *
 * @tparam global_n_partial_col  Column within tile where global_n padding starts (0 = tile-aligned, no partial)
 * @tparam joint_l_partial_col   Column within tile where joint_l padding starts (0 = tile-aligned, no partial)
 * @tparam cb_mask_in            CB to generate mask tiles into (must be constexpr for get_tile_size)
 * @tparam is_causal_lw          Whether to include the causal diagonal tile
 */
template <
    uint32_t global_n_partial_col,
    uint32_t joint_l_partial_col,
    uint32_t cb_mask_in,
    bool is_causal_lw = false,
    uint32_t sliding_window_size = 0>
void generate_lightweight_mask_tiles(Noc noc) {
    constexpr uint32_t partial_mask_tiles = (global_n_partial_col > 0 ? 1 : 0) + (joint_l_partial_col > 0 ? 1 : 0);
    constexpr bool has_sliding_window = sliding_window_size > 0;
    constexpr uint32_t sliding_diag_tiles = has_sliding_window ? kSlidingWindowEdgeTiles : 0;
    constexpr uint32_t causal_diag_tiles = (!has_sliding_window && is_causal_lw) ? 1 : 0;
    constexpr uint32_t total_mask_tiles = 1 + sliding_diag_tiles + causal_diag_tiles + partial_mask_tiles;
    constexpr uint32_t mask_tile_size_bytes = get_tile_size(cb_mask_in);

    CircularBuffer cb(cb_mask_in);
    cb.reserve_back(total_mask_tiles);

    // Tile 0: neginf tile
    fill_neginf_tile<mask_tile_size_bytes>(cb_mask_in, 0);

    uint32_t tile_idx = 1;

    if constexpr (has_sliding_window) {
        fill_sliding_window_edge_tiles<sliding_window_size, is_causal_lw, cb_mask_in, mask_tile_size_bytes>(
            noc, tile_idx);
        tile_idx += kSlidingWindowEdgeTiles;
    } else if constexpr (is_causal_lw) {
        fill_causal_diagonal_tile_bf16<mask_tile_size_bytes>(noc, cb_mask_in, tile_idx++);
    }

    // Subsequent tiles: partial mask tiles for boundary conditions
    if constexpr (partial_mask_tiles > 0) {
        if constexpr (global_n_partial_col > 0) {
            fill_vertical_tile_bf16<mask_tile_size_bytes>(noc, cb_mask_in, tile_idx++, global_n_partial_col);
        }
        if constexpr (joint_l_partial_col > 0) {
            fill_vertical_tile_bf16<mask_tile_size_bytes>(noc, cb_mask_in, tile_idx++, joint_l_partial_col);
        }
    }

    cb.push_back(total_mask_tiles);
}

template <uint32_t tile_bytes>
inline void fill_custom_diagonal_tile_bfp4(
    Noc noc,
    uint32_t cb_id,
    uint32_t tile_id,
    int32_t leading_diagonal_offset,
    int32_t trailing_diagonal_offset) {
    // Assert that we're not in a case where the entire tile should be fully masked or fully allowed
    // Those cases should be handled before calling this function
    ASSERT(leading_diagonal_offset >= -32 && trailing_diagonal_offset >= -32);

    // Clear the tile first
    fill_tile_zeros<tile_bytes>(noc, cb_id, tile_id);

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

    CircularBuffer cb(cb_id);
    volatile tt_l1_ptr uint32_t* uint32_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr() + tile_id * tile_bytes);

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
void fill_vertical_tile_bfp4(Noc noc, uint32_t cb_id, uint32_t tile_id, uint32_t unpad_col_in_tile) {
    /*
    This tile should be set such that tile[:, unpad_col_in_tile:] = -inf
    For block float 4 format where 8 mantissas are packed per uint32
    */

    // Prefill with zeros (fast)
    fill_tile_zeros<tile_bytes>(noc, cb_id, tile_id);

    constexpr uint32_t NEG_INF_EXP = 0xFFFFFFFF;
    constexpr uint32_t NEG_INF_MANT = 0xCCCCCCCC;  // All mantissas set to 0xC
    constexpr uint32_t bf4_mant_per_uint32 = 8;    // 8 mantissas per uint32
    constexpr uint32_t bf4_exp_per_uint32 = 4;     // 4 exponents per uint32

    constexpr uint32_t uint32_datums_per_face_row = tt::constants::FACE_WIDTH / bf4_mant_per_uint32;
    constexpr uint32_t uint32_datums_per_face = (tt::constants::FACE_HW) / bf4_mant_per_uint32;
    constexpr uint32_t uint32_exp_per_face = tt::constants::FACE_HEIGHT / bf4_exp_per_uint32;

    CircularBuffer cb(cb_id);
    volatile tt_l1_ptr uint32_t* uint32_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr() + tile_id * tile_bytes);

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
    // When raw_mod == 0, the last K chunk is fully valid (no padding) — treat as full chunk size
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

// Issue noc.async_read for a (num_rows x cols) tile block. tile_id starts at base_tile_id,
// advances by ++ per col and by row_stride per row (i.e., tile_id += row_stride - cols after each
// inner col loop). dst starts at dst_addr + dst_row_origin * outer_stride, advances by
// inner_stride per col and outer_stride per row. No barrier - caller must noc.async_read_barrier().
// barrier_threshold > 0 fires a partial barrier every barrier_threshold tiles.
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

// Zero-fill a (num_rows × cols) tile block in L1. Same dst arithmetic as issue_block_reads.
// reader is used only to derive the per-tile page size. No periodic barrier: fills source
// from local L1 (MEM_ZEROS_BASE) and completes fast, so they don't push the NIU outstanding
// counter the way DRAM reads do. Caller's trailing read barrier handles visibility.
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

// Issue noc.async_write for a (num_rows x cols) tile block. Same tile_id/src arithmetic
// as issue_block_reads (with src instead of dst). No barrier - caller must noc.async_write_barrier().
// trid=0 is the default tag for untagged writes; nonzero tags writes for per-TRID barriers/flushes.
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
    uint32_t d0;  // batch dimension
    uint32_t d1;  // head dimension

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

    // Issue async NoC reads for a slice to L1. No barrier — caller must issue a
    // read barrier. Splits [slice.d2_start, slice.d2_end) into up to four segments
    // (first tensor / gap / second tensor / tail); each valid segment hoists id_of once.
    // end_seq_tile is unused: bounds come from first_shape/second_shape; signature kept for
    // API symmetry with PaddedAddrGenerator (fetch_block dispatches generically).
    void issue_reads(
        const Slice& slice,
        uint32_t /*end_seq_tile*/,
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

        // Segment 0: first tensor.
        const uint32_t s0_end = std::min(d2_end, first_end);
        if (d2_start < s0_end) {
            issue_block_reads(
                first_reader,
                first_shape.id_of(slice.d0, slice.d1, d2_start, slice.d3_start),
                first_shape.strides[2],
                s0_end - d2_start,
                cols,
                /*dst_row_origin=*/0,
                dst_addr,
                outer_stride,
                inner_stride,
                barrier_threshold,
                barrier_count);
        }
        // Segment 1: gap (zero-fill).
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
        // Segment 2: second tensor (d2 shifted by first_seq_padded).
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
        // Segment 3: tail (zero-fill).
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

    // Issue async NoC writes for a slice from L1. No barrier — caller must issue a
    // write barrier. Same segment split as issue_reads; gap and tail produce no
    // writes (those rows aren't mapped to either tensor). end_seq_tile unused (see issue_reads).
    void issue_writes(
        Noc noc,
        const Slice& slice,
        uint32_t /*end_seq_tile*/,
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

        // Segment 0: first tensor.
        const uint32_t s0_end = std::min(d2_end, first_end);
        if (d2_start < s0_end) {
            issue_block_writes(
                noc,
                first_reader,
                first_shape.id_of(slice.d0, slice.d1, d2_start, slice.d3_start),
                first_shape.strides[2],
                s0_end - d2_start,
                cols,
                /*src_row_origin=*/0,
                src_addr,
                outer_stride,
                inner_stride,
                trid);
        }
        // Gap rows: no writes.
        // Segment 2: second tensor.
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
        // Tail rows: no writes.
    }
};

template <typename ReaderType, typename TensorShapeType = TensorTileShape>
struct PaddedAddrGenerator {
    ReaderType reader;
    TensorShapeType tensor_shape;

    PaddedAddrGenerator(const ReaderType& reader, TensorShapeType tensor_shape) :
        reader(reader), tensor_shape(tensor_shape) {}

    // Issue async NoC reads for a slice to L1. No barrier — caller must issue a
    // read barrier. Splits valid rows from the padded tail at loop level (no
    // in_bounds branch in the hot path); valid rows advance tile_id by arithmetic only.
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

        // Valid segment: real reads.
        issue_block_reads(
            reader,
            tensor_shape.id_of(slice.d0, slice.d1, d2_start, slice.d3_start),
            tensor_shape.stride2(),
            valid_rows,
            cols,
            /*dst_row_origin=*/0,
            dst_addr,
            outer_stride,
            inner_stride,
            barrier_threshold,
            barrier_count);
        // Padded tail: zero-fill.
        zero_fill_block(
            reader,
            rows - valid_rows,
            cols,
            /*dst_row_origin=*/valid_rows,
            dst_cb_id,
            dst_addr,
            outer_stride,
            inner_stride);
    }

    // Issue async NoC writes for a slice from L1. No barrier — caller must
    // noc.async_write_barrier(). Out-of-bound rows produce no writes.
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

        // Valid segment only; out-of-bound rows produce no writes.
        issue_block_writes(
            noc,
            reader,
            tensor_shape.id_of(slice.d0, slice.d1, d2_start, slice.d3_start),
            tensor_shape.stride2(),
            valid_rows,
            cols,
            /*src_row_origin=*/0,
            src_addr,
            outer_stride,
            inner_stride,
            trid);
    }

    void issue_writes_no_padding(
        Noc noc,
        const Slice& slice,
        uint32_t src_addr,
        uint32_t outer_stride,
        uint32_t inner_stride,
        uint32_t trid = 0) const {
        issue_block_writes(
            noc,
            reader,
            tensor_shape.id_of(slice.d0, slice.d1, slice.d2_start, slice.d3_start),
            tensor_shape.stride2(),
            slice.get_d2_size(),
            slice.get_d3_size(),
            /*src_row_origin=*/0,
            src_addr,
            outer_stride,
            inner_stride,
            trid);
    }
};

template <typename ReaderType, typename TensorShapeType>
PaddedAddrGenerator(const ReaderType&, TensorShapeType) -> PaddedAddrGenerator<ReaderType, TensorShapeType>;

// Fetch tiles via NOC reads into a given L1 address. No CB lifecycle — caller manages
// the reserve/push sequence on the destination CB. Used by forwarding paths that mcast before pushing.
//
// Dispatches to the generator's issue_reads. PaddedAddrGenerator's overload hoists id_of
// (4 muls + 3 adds) and the row-only validity check out of the inner col loop;
// CatAddrGenerator keeps per-tile dispatch (off the ring SDPA hot path).
//
// noinline: reader (NCRISC) has 3+ call sites and the issue_reads body is large enough that
// inlining at every site overflows the TENSIX kernel-config ringbuffer. Function-call
// overhead is negligible next to the per-block NoC reads.
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
    // issue_reads internally emits noc.async_read (NOC) AND zero_fill_block → async_write_zeros
    // (iDMA on Quasar). NOC reads and async_write_zeros use the same completion path on WH/BH
    // but different paths on Quasar. Issue both — second is a no-op on WH/BH.
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

// Pop a (rows × cols) tile block out of a CB and write it via NoC. Symmetric to read_block:
// wait-front + dispatch + barrier + pop-front. Dispatches to the generator's issue_writes
// (PaddedAddrGenerator hoists id_of out of the inner col loop; CatAddrGenerator keeps per-tile
// dispatch). Out-of-bound rows are skipped (no zero-fill on writes).
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

template <typename TensorAccessorType>
void write_block(
    Noc noc,
    const TensorAccessorType& out_writer,
    const uint32_t cb_out,
    const uint32_t out_chunk_tiles,
    const uint32_t rows,
    const uint32_t cols,
    const uint32_t out_tile_id,
    const uint32_t tile_bytes,
    const uint32_t barrier_threshold) {
    uint32_t barrier_count = 0;
    uint32_t tile_id = out_tile_id;

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
    }
    noc.async_write_barrier();
    cb.pop_front(out_chunk_tiles);
}

// Single-chip linear-tile-id drain. Iterates total_rows in groups of sbh rows (last group is
// a smaller remainder if not divisible); per-group wait-front + flush-before-pop lets
// cb_out be sized to a few groups instead of the full chunk. Rows in [write_rows, total_rows)
// are padding — popped but not written. Periodic barrier_threshold flushes guard the NoC ack
// queue; the final write barrier ensures DRAM arrival before return. Single-chip never
// sets a non-zero trid → drain flushes trid 0 (the default trid all writes here carry).
template <typename TensorAccessorType>
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
    const uint32_t barrier_threshold) {
    constexpr uint32_t default_trid = 0;
    uint32_t tile_id = out_tile_id;
    uint32_t barrier_count = 0;

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
            }
        }
        // Flush THIS drain's writes (default trid) before pop so compute can safely reuse the L1 slot.
        noc.async_writes_flushed<NocOptions::TXN_ID>({.trid = default_trid});
        cb.pop_front(tiles_this_group);
    }
    noc.async_write_barrier();
}

// Multi-chip row-grouped drain of cb_out to DRAM;
// writes overlap with compute's next row-group push. Padding past end_seq_tile is silently skipped
// (out-of-bound rows produce no writes). flush_trid is the TRID stamped on each individual
// write via noc.async_write<NocOptions::TXN_ID>(..., {.trid=flush_trid});
// per-group flush uses noc.async_writes_flushed<TXN_ID>(flush_trid) so it waits exactly for
// THIS drain's writes to be source-L1-acked. Caller handles any final DRAM-arrival NoC barrier.
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

    CircularBuffer cb(cb_id);
    uint32_t write_ptr = cb.get_write_ptr();

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
            // Bump it up to the chunk start
            offset_q_chunk += chunk_start_t_in_q_chunks;
        }
        uint32_t q_low_idx = offset_q_chunk * Sq_chunk_t;  // This is the sequence index of the first tile of this chunk
        uint32_t q_high_idx = q_low_idx + Sq_chunk_t;

        for (uint32_t k_chunk = 0; (k_chunk * Sk_chunk_t) < q_high_idx; ++k_chunk) {
            const uint32_t k_low_idx = k_chunk * Sk_chunk_t;
            const uint32_t k_high_idx = k_low_idx + Sk_chunk_t;
            // Finding the diagonal is harder now that q_chunk_size and k_chunk_size can differ
            // Q-range = [q_low, q_high)
            // K-range = [k_low, k_high)
            // does_overlap = not (q_low >= k_high or k_low >= q_high)
            // Due to loop bounds, we should never have k_low >= q_high. Can simplify this conditional check
            // Read mask chunk
            if (!(q_low_idx >= k_high_idx) || sliding_window_size > 0) {
                // If no sliding window, only generate mask along diagonal
                // Otherwise, generate mask for all chunks
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
