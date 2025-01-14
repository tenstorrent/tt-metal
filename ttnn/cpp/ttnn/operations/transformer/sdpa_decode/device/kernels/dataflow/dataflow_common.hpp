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
void fill_tile_partial(
    uint32_t cb_id, uint32_t tile_id, uint32_t cur_pos_in_tile, uint32_t partial_val, uint32_t base_val = 0) {
    /*
    We want to fill cur_pos_in_tile + 1 to the end
    */

    fill_tile<tile_bytes>(cb_id, tile_id, base_val);
    if (cur_pos_in_tile == 31 || partial_val == base_val) {
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

template <uint32_t tile_bytes>
void fill_tile_partial_transposed(
    uint32_t cb_id, uint32_t tile_id, uint32_t cur_pos_in_tile, uint32_t partial_val, uint32_t base_val = 0) {
    /*
    We want to fill cur_pos_in_tile + 1 to the end in the row dimension
    Tile layout (32x32):
    [Phase 0: 16x16] [Phase 1: 16x16]
    [Phase 2: 16x16] [Phase 3: 16x16]

    Memory layout: Phases are stored contiguously
    [Phase 0][Phase 1][Phase 2][Phase 3]
    */

    fill_tile<tile_bytes>(cb_id, tile_id, base_val);
    if (cur_pos_in_tile == 31 || partial_val == base_val) {
        return;
    }

    const uint16_t datum_val = partial_val >> 16;
    volatile tt_l1_ptr uint16_t* uint16_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_id) + tile_id * tile_bytes);
    volatile tt_l1_ptr uint32_t* uint32_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id) + tile_id * tile_bytes);

    // Calculate which phase and position within phase we're at
    uint32_t phase_start = (cur_pos_in_tile < 15) ? 0 : 2;  // Top half (0,1) or bottom half (2,3)

    // Fill all remaining columns in the current row
    if (phase_start == 0) {
        uint32_t row_in_phase = cur_pos_in_tile + 1;
        uint32_t uint32_face_idx_1 = 1 << 7;

        // For top half (phases 0 and 1)
        for (uint32_t row = row_in_phase; row < 16; row++) {
            // Fill remaining positions in current row for both phases
            for (uint32_t col = 0; col < 8; col++) {  // 8 uint32 values per row (16 uint16 values)
                uint32_ptr[row * 8 + col] = partial_val;
                uint32_ptr[uint32_face_idx_1 + row * 8 + col] = partial_val;
            }
        }
    }

    uint32_t row_in_phase = (phase_start == 0) ? 0 : (cur_pos_in_tile + 1) % 16;
    uint32_t uint32_face_idx_2 = 2 << 7;
    uint32_t uint32_face_idx_3 = 3 << 7;

    // For bottom half (phases 2 and 3)
    for (uint32_t row = row_in_phase; row < 16; row++) {
        // Fill remaining positions in current row for both phases
        for (uint32_t col = 0; col < 8; col++) {  // 8 uint32 values per row (16 uint16 values)
            uint32_ptr[uint32_face_idx_2 + row * 8 + col] = partial_val;
            uint32_ptr[uint32_face_idx_3 + row * 8 + col] = partial_val;
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

template <uint32_t cb_mask_in, uint32_t PNHt, uint32_t Sk_chunk_t>
void generate_mask(uint32_t k_num_chunks, uint32_t cur_pos) {
    /*
    example 1: 64 seqlen at cur_pos 40, 2 cores, 32 chunk size
    k_num_chunks = 2
    Sk_chunk_t = 1
    cur_pos = 40
    cur_pos_in_chunk = 8
    cur_pos_in_chunk_t = 0
    cur_pos_in_tile = 8

    example 2: 1024 seqlen at cur_pos 990, 2 cores, 128 chunk size
    k_num_chunks = 8
    Sk_chunk_t = 4
    cur_pos = 990
    cur_pos_in_chunk = 94
    cur_pos_in_chunk_t = 2
    cur_pos_in_tile = 30

    example 3: 64 seqlen at cur_pos 63, 2 cores, 32 chunk size
    k_num_chunks = 2
    Sk_chunk_t = 1
    cur_pos = 63
    cur_pos_in_chunk = 31
    cur_pos_in_chunk_t = 0
    cur_pos_in_tile = 31

    example 3: 64 seqlen at cur_pos 0, 2 cores, 32 chunk size
    k_num_chunks = 2
    Sk_chunk_t = 1
    cur_pos = 0
    cur_pos_in_chunk = 0
    cur_pos_in_chunk_t = 0
    cur_pos_in_tile = 0
    */

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

template <uint32_t cb_out, uint32_t out_chunk_tiles, uint32_t barrier_threshold>
uint32_t write_tiles_to_memory(
    uint32_t& out_tile_id, const InterleavedAddrGenFast<true>& out_writer, uint32_t& barrier_count) {
    constexpr uint32_t tile_bytes = get_tile_size(cb_out);
    uint32_t l1_read_addr = get_read_ptr(cb_out);
    for (uint32_t tile = 0; tile < out_chunk_tiles; ++tile) {
        noc_async_write_tile(out_tile_id, out_writer, l1_read_addr);
        ++out_tile_id;
        l1_read_addr += tile_bytes;
        if (++barrier_count == barrier_threshold) {
            noc_async_writes_flushed();
            barrier_count = 0;
        }
    }
    return barrier_count;
}

template <uint32_t cb_out, uint32_t ELEMENT_SIZE, uint32_t barrier_threshold>
uint32_t write_partial_tiles_to_memory(
    uint32_t& out_tile_id,
    const InterleavedAddrGenFast<true>& out_writer,
    uint32_t& barrier_count,
    uint32_t cur_head,
    uint32_t num_heads_to_write,
    uint32_t out_chunk_tiles) {
    constexpr uint32_t FACE_HW = 16;
    constexpr uint32_t FACE_ELEMENT_CNT = FACE_HW * FACE_HW;  // 256
    constexpr uint32_t tile_bytes = get_tile_size(cb_out);
    constexpr uint32_t FACE_LINE_BYTES = FACE_HW * ELEMENT_SIZE;

    for (uint32_t tile = 0; tile < out_chunk_tiles; ++tile) {
        uint64_t out_writer_noc_addr = get_noc_addr(out_tile_id, out_writer);
        uint32_t l1_read_addr = get_read_ptr(cb_out) + tile * tile_bytes;

        // write partial output for each head
        for (uint32_t head = 0; head < num_heads_to_write; ++head) {
            uint32_t starting_row = cur_head * num_heads_to_write + head;
            uint32_t in_tile_offset_by_starting_head =
                starting_row < FACE_HW
                    ? starting_row * FACE_LINE_BYTES
                    : (starting_row + FACE_HW) * FACE_LINE_BYTES;  // Skip the second face which has FACE_HW rows
            uint64_t out_writer_noc_addr_head = out_writer_noc_addr + in_tile_offset_by_starting_head;
            uint32_t l1_read_addr_head = l1_read_addr + in_tile_offset_by_starting_head;

            // Write first phase
            noc_async_write(l1_read_addr_head, out_writer_noc_addr_head, FACE_LINE_BYTES);

            // Write second phase
            noc_async_write(
                l1_read_addr_head + FACE_ELEMENT_CNT * ELEMENT_SIZE,
                out_writer_noc_addr_head + FACE_ELEMENT_CNT * ELEMENT_SIZE,
                FACE_LINE_BYTES);

            if (++barrier_count == barrier_threshold) {
                noc_async_writes_flushed();
                barrier_count = 0;
            }
        }

        ++out_tile_id;
    }
    return barrier_count;
}

/******************************************************************************
 *                   Reader Kernel Specific Functions                         *
 ******************************************************************************/

template <
    uint32_t DHt,
    uint32_t Sk_chunk_t,
    uint32_t barrier_threshold,
    uint32_t k_chunk_tiles,
    uint32_t mask_chunk_tiles,
    uint32_t mask_tile_bytes,
    uint32_t PNHt,
    bool use_attention_mask,
    uint32_t cb_k_in,
    uint32_t cb_v_in,
    uint32_t cb_mask_in>
void read_kv_mask_chunks(
    uint32_t k_chunk_start,
    uint32_t k_chunk_end,
    uint32_t k_start_tile_id,
    uint32_t v_start_tile_id,
    uint32_t mask_start_tile_id,
    const InterleavedAddrGenFast<true>& k_reader,
    const InterleavedAddrGenFast<true>& v_reader,
    const InterleavedAddrGenFast<true>& mask_reader,
    uint32_t k_tile_bytes,
    uint32_t v_tile_bytes,
    uint32_t PSt) {
    uint32_t barrier_count = 0;
    for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {
        // Read K chunk transposed
        cb_reserve_back(cb_k_in, k_chunk_tiles);
        uint32_t k_write_ptr = get_write_ptr(cb_k_in);
        barrier_count = 0;
        for (uint32_t col = 0; col < DHt; ++col) {
            uint32_t k_tile_id = k_start_tile_id + col;
            for (uint32_t row = 0; row < Sk_chunk_t; ++row) {
                noc_async_read_tile(k_tile_id, k_reader, k_write_ptr);
                if (++barrier_count == barrier_threshold) {
                    noc_async_read_barrier();
                    barrier_count = 0;
                }
                k_tile_id += DHt;
                k_write_ptr += k_tile_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_k_in, k_chunk_tiles);
        k_start_tile_id += k_chunk_tiles;

        if constexpr (use_attention_mask) {
            mask_start_tile_id =
                read_mask_chunk<cb_mask_in, mask_chunk_tiles, mask_tile_bytes, barrier_threshold, PNHt, Sk_chunk_t>(
                    PSt, mask_start_tile_id, mask_reader);
        }

        // Read V chunk
        cb_reserve_back(cb_v_in, k_chunk_tiles);
        uint32_t v_write_ptr = get_write_ptr(cb_v_in);
        barrier_count = 0;
        uint32_t v_tile_id = v_start_tile_id;
        for (uint32_t row = 0; row < Sk_chunk_t; ++row) {
            for (uint32_t col = 0; col < DHt; ++col) {
                noc_async_read_tile(v_tile_id, v_reader, v_write_ptr);
                if (++barrier_count == barrier_threshold) {
                    noc_async_read_barrier();
                    barrier_count = 0;
                }
                v_tile_id++;
                v_write_ptr += v_tile_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_v_in, k_chunk_tiles);
        v_start_tile_id += k_chunk_tiles;
    }
}
