// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include <vector>

/******************************************************************************
 *                                                                             *
 *                   Common Functions for Dataflow Kernels                     *
 *                                                                             *
 ******************************************************************************/

/******************************************************************************
 *                   Generic Utility Functions                                 *
 ******************************************************************************/
template <uint32_t tile_bytes, uint32_t num_readers>
constexpr uint32_t get_barrier_read_threshold() {
    return ((512 / num_readers) * (1024 + 128)) / tile_bytes;
}

/******************************************************************************
 *                   Page Cache Functions            *
 ******************************************************************************/
template <uint32_t num_heads, uint32_t block_size_t, uint32_t Wt>
uint32_t virtual_seq_tile_id_to_physical_tile_id(
    uint32_t seq_tile_idx, uint32_t cur_head, volatile tt_l1_ptr const uint32_t* const page_table_ptr) {
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

/******************************************************************************
 *                   Generic Tile Manipulation Functions                       *
 ******************************************************************************/
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
        constexpr int num_uint32_datums_tile = (32 * 32) / 2;
        for (int k = 0; k < num_uint32_datums_tile; k++) {
            ptr[k] = val;
        }
    }
}

template <uint32_t tile_bytes>
void fill_tile_partial(uint32_t cb_id, uint32_t tile_id, uint32_t cur_pos_in_tile, uint32_t partial_val) {
    /*
    We want to fill cur_pos_in_tile + 1 to the end
    */

    fill_tile<tile_bytes>(cb_id, tile_id, 0);
    if (cur_pos_in_tile == 31 || partial_val == 0) {
        return;
    }
    const uint16_t datum_val = partial_val >> 16;
    volatile tt_l1_ptr uint16_t* uint16_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_id) + tile_id * tile_bytes);
    volatile tt_l1_ptr uint32_t* uint32_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id) + tile_id * tile_bytes);
    int face_start = (cur_pos_in_tile < 15) ? 0 : 1;
    uint32_t fill_pos_in_face = (cur_pos_in_tile + 1) % 16;
    if (face_start == 0) {
        // Fill 2 datums in each writes to optimize for performance
        constexpr int num_uint32_datums_tile_face = (16 * 16) / 2;
        for (int k = 1; k < 4; k += 2) {
            uint32_t uint32_face_idx = k << 7;
            for (int j = 0; j < num_uint32_datums_tile_face; j++) {
                uint32_ptr[uint32_face_idx + j] = partial_val;
            }
        }
    }

    // Again, optimizing performance by filling 2 uint16 datums in each write.
    // If the fill_pos_in_face is odd then we fill that pos with single datum,
    // otherwise we fill 2 datums in each write
    bool is_odd_pos_filled = fill_pos_in_face % 2 == 1;
    uint32_t fill_pos_in_uint32_face = (fill_pos_in_face + 1) >> 1;
    constexpr uint32_t num_cols_in_face = 16;
    constexpr uint32_t num_rows_in_face = 16;
    constexpr uint32_t num_cols_in_uint32_face = num_cols_in_face >> 1;
    for (int k = face_start; k < 4; k += 2) {
        uint32_t uint16_face_idx = k << 8;
        uint32_t uint32_face_idx = k << 7;

        for (uint32_t face_row_idx = 0; face_row_idx < num_rows_in_face; face_row_idx++) {
            // Here, if the fill_pos_in_face is odd then we fill that pos with single uint16 value
            if (is_odd_pos_filled) {
                uint16_ptr[uint16_face_idx + (fill_pos_in_face + num_cols_in_face * face_row_idx)] = datum_val;
            }

            for (uint32_t uint32_face_col_idx = fill_pos_in_uint32_face; uint32_face_col_idx < num_cols_in_uint32_face;
                 uint32_face_col_idx++) {
                uint32_ptr[uint32_face_idx + (uint32_face_col_idx + num_cols_in_uint32_face * face_row_idx)] =
                    partial_val;
            }
        }
    }
}

/******************************************************************************
 *                   Attention Mask Functions                                 *
 ******************************************************************************/
template <
    uint32_t cb_mask_in,
    uint32_t mask_chunk_tiles,
    uint32_t mask_tile_bytes,
    uint32_t barrier_threshold,
    uint32_t PNHt,
    uint32_t Sk_chunk_t>
uint32_t read_mask_chunk(uint32_t PSt, uint32_t mask_start_tile_id, const InterleavedAddrGenFast<true> mask_reader) {
    // Read mask chunk
    cb_reserve_back(cb_mask_in, mask_chunk_tiles);
    uint32_t mask_write_ptr = get_write_ptr(cb_mask_in);
    uint32_t barrier_count = 0;
    for (uint32_t row = 0; row < PNHt; ++row) {
        uint32_t mask_tile_id = mask_start_tile_id + row * PSt;
        for (uint32_t col = 0; col < Sk_chunk_t; ++col) {
            noc_async_read_tile(mask_tile_id, mask_reader, mask_write_ptr);
            mask_tile_id++;
            mask_write_ptr += mask_tile_bytes;

            if (++barrier_count == barrier_threshold) {
                noc_async_read_barrier();
                barrier_count = 0;
            }
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_mask_in, mask_chunk_tiles);
    mask_start_tile_id += mask_chunk_tiles;
    return mask_start_tile_id;
}

template <uint32_t cb_mask_in, uint32_t PNHt>
void generate_mask(uint32_t k_num_chunks, uint32_t PSt, uint32_t cur_pos) {
    /*
    example 1: 64 seqlen at cur_pos 40, 2 cores, 32 chunk size
    PSt = 2
    k_num_chunks = 2
    Sk_chunk_t = 1
    cur_pos = 40
    cur_pos_in_chunk = 8
    cur_pos_in_chunk_t = 0
    cur_pos_in_tile = 8

    example 2: 1024 seqlen at cur_pos 990, 2 cores, 128 chunk size
    PSt = 32
    k_num_chunks = 8
    Sk_chunk_t = 4
    cur_pos = 990
    cur_pos_in_chunk = 94
    cur_pos_in_chunk_t = 2
    cur_pos_in_tile = 30

    example 3: 64 seqlen at cur_pos 63, 2 cores, 32 chunk size
    PSt = 2
    k_num_chunks = 2
    Sk_chunk_t = 1
    cur_pos = 63
    cur_pos_in_chunk = 31
    cur_pos_in_chunk_t = 0
    cur_pos_in_tile = 31

    example 3: 64 seqlen at cur_pos 0, 2 cores, 32 chunk size
    PSt = 2
    k_num_chunks = 2
    Sk_chunk_t = 1
    cur_pos = 0
    cur_pos_in_chunk = 0
    cur_pos_in_chunk_t = 0
    cur_pos_in_tile = 0
    */

    uint32_t Sk_chunk_t = PSt / k_num_chunks;
    // the cb_mask in is of size PNHt * Sk_chunk_t
    uint32_t total_read_tiles = PNHt * Sk_chunk_t;
    uint32_t cur_pos_in_chunk = cur_pos % (Sk_chunk_t * 32);
    uint32_t cur_pos_in_chunk_t = cur_pos_in_chunk / 32;
    uint32_t cur_pos_in_tile = cur_pos_in_chunk % 32;
    constexpr uint32_t NEG_INF = 0xFF80FF80;  // TODO: Make sure this is -inf

    cb_reserve_back(cb_mask_in, total_read_tiles);

    uint64_t noc_read_addr_base = get_noc_addr(get_read_ptr(cb_mask_in));
    uint32_t q_write_ptr_base = get_read_ptr(cb_mask_in);
    constexpr uint32_t tile_bytes = get_tile_size(cb_mask_in);

    for (uint32_t i = 0; i < Sk_chunk_t; ++i) {
        if (i < cur_pos_in_chunk_t) {
            // fill with zero
            if (i == 0) {
                fill_tile<tile_bytes>(cb_mask_in, i, 0);
            } else {
                copy_tile<tile_bytes>(
                    noc_read_addr_base, q_write_ptr_base, 0, i);  // copy from cb_mask_in[0] to cb_mask_in[i]
                if (i == cur_pos_in_chunk_t - 1) {
                    noc_async_read_barrier();
                }
            }
        } else if (i == cur_pos_in_chunk_t) {
            // fill with partial zero/-inf
            fill_tile_partial<tile_bytes>(cb_mask_in, i, cur_pos_in_tile, NEG_INF);
        } else {
            // fill with -inf
            if (i == cur_pos_in_chunk_t + 1) {
                fill_tile<tile_bytes>(cb_mask_in, i, NEG_INF);
            } else {
                copy_tile<tile_bytes>(
                    noc_read_addr_base,
                    q_write_ptr_base,
                    cur_pos_in_chunk_t + 1,
                    i);  // copy from cb_mask_in[cur_pos_in_chunk_t+1] to cb_mask_in[i]
                if (i == Sk_chunk_t - 1) {
                    noc_async_read_barrier();
                }
            }
        }
        for (uint32_t j = 1; j < PNHt; ++j) {
            // copy from cb_mask_in[i] to cb_mask_in[j*Sk_chunk_t + i]
            copy_tile<tile_bytes>(noc_read_addr_base, q_write_ptr_base, i, j * Sk_chunk_t + i);
            if (j == PNHt - 1) {
                noc_async_read_barrier();
            }
        }
    }

    cb_push_back(cb_mask_in, total_read_tiles);
}

/******************************************************************************
 *                   Writer Kernel Specific Functions                         *
 ******************************************************************************/

template <
    uint32_t out_chunk_tiles,
    uint32_t cb_out,
    uint32_t cb_out_m,
    uint32_t cb_out_l,
    uint32_t cb_intermed_out,
    uint32_t PNHt>
void worker_compute(
    uint64_t in0_sender_semaphore_noc_addr,
    uint32_t worker_id,
    uint32_t reduce_core_noc_x,
    uint32_t reduce_core_noc_y) {
    uint32_t out_tile_id = 0;

    // Wait for compute to deliver output chunk
    cb_wait_front(cb_out, out_chunk_tiles);
    cb_wait_front(cb_out_m, PNHt);
    cb_wait_front(cb_out_l, PNHt);

    // Write output chunk to reducer
    constexpr uint32_t tile_bytes = get_tile_size(cb_out);
    uint32_t worker_offset = worker_id * (out_chunk_tiles + 2 * PNHt) * tile_bytes;
    constexpr uint32_t o_write_size = out_chunk_tiles * tile_bytes;
    constexpr uint32_t ml_write_size = PNHt * tile_bytes;
    uint64_t output_write_addr =
        get_noc_addr(reduce_core_noc_x, reduce_core_noc_y, get_write_ptr(cb_intermed_out)) + worker_offset;
    noc_async_write(get_read_ptr(cb_out), output_write_addr, o_write_size);
    output_write_addr += o_write_size;
    noc_async_write(get_read_ptr(cb_out_m), output_write_addr, ml_write_size);
    output_write_addr += ml_write_size;
    noc_async_write(get_read_ptr(cb_out_l), output_write_addr, ml_write_size);

    // increment semaphore
    noc_async_write_barrier();
    noc_semaphore_inc(in0_sender_semaphore_noc_addr, 1);

    // pop front
    cb_pop_front(cb_out, out_chunk_tiles);
    cb_pop_front(cb_out_m, PNHt);
    cb_pop_front(cb_out_l, PNHt);
}
