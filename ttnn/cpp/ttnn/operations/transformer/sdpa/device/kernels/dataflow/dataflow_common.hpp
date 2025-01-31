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

template <bool is_dram = true>
void read_chunk_with_padding(
    const InterleavedAddrGenFast<is_dram>& reader,
    const uint32_t cb_id,
    uint32_t start_tile_id,
    const uint32_t src_rows,
    const uint32_t src_cols,
    const uint32_t dst_rows,
    const uint32_t dst_cols,
    const uint32_t tile_bytes,
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
void fill_tile(uint32_t cb_id, uint32_t tile_id, uint32_t val) {
    if (val == 0) {
        constexpr uint32_t num_zeros_reads = 2048 / MEM_ZEROS_SIZE;
        uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
        uint32_t write_addr = get_write_ptr(cb_id) + tile_id * tile_bytes;
        volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_addr);

        // Fill tile with zeros
        for (uint32_t i = 0; i < num_zeros_reads; ++i) {
            noc_async_read(zeros_noc_addr, write_addr, MEM_ZEROS_SIZE);
            write_addr += MEM_ZEROS_SIZE;
        }
        noc_async_read_barrier();
    } else {
        // Fill 2 uint16 datums in each writes to optimize for performance
        volatile tt_l1_ptr uint32_t* ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id) + tile_id * tile_bytes);
        constexpr int num_uint32_datums_tile = (tt::constants::TILE_HW) / 2;
        for (int k = 0; k < num_uint32_datums_tile; k++) {
            ptr[k] = val;
        }
    }
}

template <uint32_t tile_bytes>
void fill_diagonal_tile(uint32_t cb_id, uint32_t tile_id, uint32_t partial_val) {
    /*
    We want to fill cur_pos_in_tile + 1 to the end
    */

    fill_tile<tile_bytes>(cb_id, tile_id, 0);

    const uint16_t datum_val = partial_val >> 16;
    volatile tt_l1_ptr uint16_t* uint16_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_id) + tile_id * tile_bytes);
    volatile tt_l1_ptr uint32_t* uint32_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id) + tile_id * tile_bytes);

    constexpr uint32_t uint16_datums_per_face_row = tt::constants::FACE_WIDTH;
    constexpr uint32_t uint32_datums_per_face_row = tt::constants::FACE_WIDTH / 2;
    constexpr uint32_t uint32_datums_per_face = (tt::constants::FACE_HW) / 2;
    // Fill diagonal faces with diagonal -inf
    for (uint32_t k = 0; k < 4; k += 3) {
        uint32_t uint16_face_idx = k << 8;
        uint32_t uint32_face_idx = k << 7;
        for (uint32_t r = 0; r < uint16_datums_per_face_row; ++r) {
            const uint32_t col_start = r + 1;
            const uint32_t col_start_uint32 = (col_start + 1) >> 1;
            if ((col_start) % 2 == 1) {
                uint16_ptr[uint16_face_idx + r * uint16_datums_per_face_row + col_start] = datum_val;
            }
            for (uint32_t c = col_start_uint32; c < uint32_datums_per_face_row; ++c) {
                uint32_ptr[uint32_face_idx + r * uint32_datums_per_face_row + c] = partial_val;
            }
        }
    }

    // Fill face 1 with full -inf
    uint32_t uint32_face_idx = 1 << 7;
    for (uint32_t j = 0; j < uint32_datums_per_face; j++) {
        uint32_ptr[uint32_datums_per_face + j] = partial_val;
    }
}

template <uint32_t tile_bytes>
void fill_vertical_tile(uint32_t cb_id, uint32_t tile_id, uint32_t unpad_col_in_tile, uint32_t partial_val) {
    /*
    This tile should be set such that tile[:, unpad_col_in_tile:] = partial_val
    */

    // Prefill with zeros (fast)
    fill_tile<tile_bytes>(cb_id, tile_id, 0);

    const uint16_t datum_val = partial_val >> 16;
    volatile tt_l1_ptr uint16_t* uint16_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_id) + tile_id * tile_bytes);
    volatile tt_l1_ptr uint32_t* uint32_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id) + tile_id * tile_bytes);

    constexpr uint32_t uint16_datums_per_face_row = tt::constants::FACE_WIDTH;
    constexpr uint32_t uint32_datums_per_face_row = tt::constants::FACE_WIDTH / 2;
    constexpr uint32_t uint32_datums_per_face = (tt::constants::FACE_HW) / 2;
    // Fill cols unpad_col_in_tile: with partial_val
    // Start with left faces
    if (unpad_col_in_tile < uint16_datums_per_face_row) {
        const uint32_t unpad_col_in_tile_uint32 = (unpad_col_in_tile + 1) >> 1;
        for (uint32_t k = 0; k < 4; k += 2) {
            uint32_t uint16_face_idx = k << 8;
            uint32_t uint32_face_idx = k << 7;
            for (uint32_t r = 0; r < uint16_datums_per_face_row; ++r) {
                if ((unpad_col_in_tile) % 2 == 1) {
                    uint16_ptr[uint16_face_idx + r * uint16_datums_per_face_row + unpad_col_in_tile] = datum_val;
                }
                for (uint32_t c = unpad_col_in_tile_uint32; c < uint32_datums_per_face_row; ++c) {
                    uint32_ptr[uint32_face_idx + r * uint32_datums_per_face_row + c] = partial_val;
                }
            }
        }
    }

    const uint32_t unpad_col_in_right_face =
        (unpad_col_in_tile < uint16_datums_per_face_row) ? 0 : unpad_col_in_tile - uint16_datums_per_face_row;
    const uint32_t unpad_col_in_right_face_uint32 = (unpad_col_in_right_face + 1) >> 1;
    for (uint32_t k = 1; k < 4; k += 2) {
        uint32_t uint16_face_idx = k << 8;
        uint32_t uint32_face_idx = k << 7;
        for (uint32_t r = 0; r < uint16_datums_per_face_row; ++r) {
            if ((unpad_col_in_right_face) % 2 == 1) {
                uint16_ptr[uint16_face_idx + r * uint16_datums_per_face_row + unpad_col_in_right_face] = datum_val;
            }
            for (uint32_t c = unpad_col_in_right_face_uint32; c < uint32_datums_per_face_row; ++c) {
                uint32_ptr[uint32_face_idx + r * uint32_datums_per_face_row + c] = partial_val;
            }
        }
    }
}

template <uint32_t cb_mask_in>
void generate_causal_mask(uint32_t Sq_chunk_t, uint32_t Sk_chunk_t, uint32_t q_chunk, uint32_t k_chunk) {
    uint32_t mask_size_tiles = Sq_chunk_t * Sk_chunk_t;
    constexpr uint32_t NEG_INF = 0xFF80FF80;
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
                    fill_tile<tile_bytes>(cb_mask_in, in_mask_tile_id, 0);
                    zero_tile_idx = in_mask_tile_id;
                } else {
                    copy_tile<tile_bytes>(noc_write_addr_base, write_ptr_base, zero_tile_idx, in_mask_tile_id);
                }
            } else if (global_k_tile == global_q_tile) {
                if (diag_tile_idx == -1) {
                    fill_diagonal_tile<tile_bytes>(cb_mask_in, in_mask_tile_id, NEG_INF);
                    diag_tile_idx = in_mask_tile_id;
                } else {
                    copy_tile<tile_bytes>(noc_write_addr_base, write_ptr_base, diag_tile_idx, in_mask_tile_id);
                }
            } else {
                if (inf_tile_idx == -1) {
                    fill_tile<tile_bytes>(cb_mask_in, in_mask_tile_id, NEG_INF);
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
    constexpr uint32_t NEG_INF = 0xFF80FF80;
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
                    fill_tile<tile_bytes>(cb_mask_in, in_mask_tile_id, 0);
                    zero_tile_idx = in_mask_tile_id;
                } else {
                    copy_tile<tile_bytes>(noc_write_addr_base, write_ptr_base, zero_tile_idx, in_mask_tile_id);
                }
            } else if (do_inf) {
                if (inf_tile_idx == -1) {
                    fill_tile<tile_bytes>(cb_mask_in, in_mask_tile_id, NEG_INF);
                    inf_tile_idx = in_mask_tile_id;
                } else {
                    copy_tile<tile_bytes>(noc_write_addr_base, write_ptr_base, inf_tile_idx, in_mask_tile_id);
                }
            } else {
                if (vertical_tile_idx == -1) {
                    fill_vertical_tile<tile_bytes>(cb_mask_in, in_mask_tile_id, unpad_col_in_tile, NEG_INF);
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
        }
        return 0;
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
