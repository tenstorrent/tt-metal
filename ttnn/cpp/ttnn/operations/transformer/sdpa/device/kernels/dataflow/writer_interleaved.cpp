// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"

template<uint32_t tile_bytes, uint32_t num_readers>
constexpr uint32_t get_barrier_read_threshold() {
    return ((512 / num_readers) * (1024 + 128)) / tile_bytes;
}

template <uint32_t tile_bytes>
void copy_tile(uint64_t noc_read_addr_base, uint32_t q_write_ptr_base, uint32_t src_tile_id, uint32_t dst_tile_id) {
    noc_async_read(noc_read_addr_base + src_tile_id*tile_bytes, q_write_ptr_base + dst_tile_id*tile_bytes, tile_bytes);
}

template <uint32_t tile_bytes>
void fill_tile(uint32_t cb_id, uint32_t tile_id, uint32_t val) {
    if (val == 0){
        constexpr uint32_t num_zeros_reads = 2048 / MEM_ZEROS_SIZE;
        uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
        uint32_t write_addr = get_write_ptr(cb_id) + tile_id*tile_bytes;
        volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_addr);

        // Fill tile with zeros
        for (uint32_t i = 0; i < num_zeros_reads; ++i) {
            noc_async_read(zeros_noc_addr, write_addr, MEM_ZEROS_SIZE);
            write_addr += MEM_ZEROS_SIZE;
        }
        noc_async_read_barrier();
    }
    else {
        // Fill 2 uint16 datums in each writes to optimize for performance
        volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id) + tile_id*tile_bytes);
        constexpr int num_uint32_datums_tile = (32 * 32) / 2;
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

    // DPRINT << "Fill partial tile" << ENDL();
    const uint16_t datum_val = partial_val>>16;
    volatile tt_l1_ptr uint16_t* uint16_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_id) + tile_id*tile_bytes);
    volatile tt_l1_ptr uint32_t* uint32_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id) + tile_id*tile_bytes);

    constexpr uint32_t uint16_datums_per_face = 16;
    // Fill diagonal faces with diagonal -inf
    for (uint32_t k = 0; k < 4; k+=3) {
        uint32_t uint16_face_idx = k << 8;
        for (uint32_t r = 0; r < uint16_datums_per_face; ++r) {
            for (uint32_t c = r+1; c < uint16_datums_per_face; ++c) {
                uint16_ptr[uint16_face_idx + r * uint16_datums_per_face + c] = partial_val;
            }
        }
    }

    // Fill face 1 with full -inf
    uint32_t uint16_face_idx = 1 << 8;
    for (uint32_t j = 0; j < uint16_datums_per_face*uint16_datums_per_face; j++) {
        uint16_ptr[uint16_face_idx + j] = partial_val;
    }
}

template <uint32_t cb_mask_in>
void generate_mask(uint32_t Sq_chunk_t, uint32_t Sk_chunk_t, uint32_t q_chunk, uint32_t k_chunk) {
    uint32_t mask_size_tiles = Sq_chunk_t * Sk_chunk_t;
    constexpr uint32_t NEG_INF = 0xFF80FF80; // TODO: Make sure this is -inf
    cb_reserve_back(cb_mask_in, mask_size_tiles);

    uint32_t write_ptr_base = get_write_ptr(cb_mask_in);
    uint64_t noc_write_addr_base = get_noc_addr(write_ptr_base);
    constexpr uint32_t tile_bytes = get_tile_size(cb_mask_in);

    int zero_tile_idx = -1;
    int inf_tile_idx = -1;
    int diag_tile_idx = -1;

    // TODO: cache indices of prepared tiles
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
            }
            else if (global_k_tile == global_q_tile) {
                if (diag_tile_idx == -1) {
                    fill_diagonal_tile<tile_bytes>(cb_mask_in, in_mask_tile_id, NEG_INF);
                    diag_tile_idx = in_mask_tile_id;
                } else {
                    copy_tile<tile_bytes>(noc_write_addr_base, write_ptr_base, diag_tile_idx, in_mask_tile_id);
                }
            }
            else {
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

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NQH = get_compile_time_arg_val(1);
    constexpr uint32_t NKH = get_compile_time_arg_val(2);
    constexpr uint32_t St = get_compile_time_arg_val(3);
    constexpr uint32_t DHt = get_compile_time_arg_val(4);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(5);
    constexpr uint32_t q_num_chunks = get_compile_time_arg_val(6);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(7);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(8);
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(9);
    constexpr uint32_t scale_val = get_compile_time_arg_val(10);
    constexpr uint32_t num_cores = get_compile_time_arg_val(11);

    const uint32_t out_addr  = get_arg_val<uint32_t>(0);
    const uint32_t core_id    = get_arg_val<uint32_t>(1);
    const uint32_t local_batch_start = get_arg_val<uint32_t>(2);
    const uint32_t local_batch_end = get_arg_val<uint32_t>(3);
    const uint32_t local_nh_start = get_arg_val<uint32_t>(4);
    const uint32_t local_nh_end = get_arg_val<uint32_t>(5);
    const uint32_t local_q_start = get_arg_val<uint32_t>(6);
    const uint32_t local_q_end = get_arg_val<uint32_t>(7);

    const uint32_t q_chunks_per_core = local_q_end - local_q_start;

    constexpr uint32_t mask_chunk_tiles = Sq_chunk_t * Sk_chunk_t;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * DHt;

    constexpr bool is_dram = true;
    constexpr uint32_t cb_out = tt::CB::c_out0;
    constexpr uint32_t cb_mask_in = tt::CB::c_in3;

    constexpr uint32_t tile_bytes = get_tile_size(cb_out);
    constexpr DataFormat data_format = get_dataformat(cb_out);

    const InterleavedAddrGenFast<is_dram> out_writer = {
        .bank_base_address = out_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };

    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<tile_bytes, num_cores>();
    uint32_t barrier_count = 0;

    constexpr uint32_t cb_scale_in = tt::CB::c_in4;
    constexpr uint32_t cb_identity_scale_in = tt::CB::c_in5;

    generate_bcast_unary_scalar(cb_scale_in, scale_val);
    generate_reduce_scaler(cb_identity_scale_in, identity_scalar_packed);

    uint32_t out_tile_id = 0;

    for (uint32_t nb = local_batch_start; nb < local_batch_end; ++nb) {
        const uint32_t q_batch_offset = nb * NQH * St * DHt;
        for (uint32_t nq = local_nh_start; nq < local_nh_end; ++nq) {
            for (uint32_t q_iter = 0; q_iter < q_chunks_per_core; ++q_iter) {
                uint32_t q_chunk;
                #if defined BALANCED_Q_PARALLEL
                uint32_t q_chunk_div_2 = q_chunks_per_core / 2;
                if (q_iter < q_chunk_div_2) { // bottom half
                    q_chunk = local_q_start + q_iter;
                } else {
                    uint32_t back_q_iter = q_iter - q_chunk_div_2; // Back half should start at 0
                    q_chunk = q_num_chunks - 1 - (local_q_start + back_q_iter);
                }
                #else
                q_chunk = local_q_start + q_iter;
                #endif

                uint32_t q_head_offset = nq * St * DHt;
                uint32_t q_chunk_offset = q_chunk * Sq_chunk_t * DHt;
                out_tile_id = q_batch_offset + q_head_offset + q_chunk_offset;

                const uint32_t q_low_idx = q_chunk * Sq_chunk_t; // This is the sequence index of the first tile of this chunk
                const uint32_t q_high_idx = q_low_idx + Sq_chunk_t;

                for (uint32_t k_chunk = 0; (k_chunk * Sk_chunk_t) < q_high_idx; ++k_chunk) {
                    const uint32_t k_low_idx = k_chunk * Sk_chunk_t;
                    const uint32_t k_high_idx = k_low_idx + Sk_chunk_t;
                    // Finding the diagonal is harder now that q_chunk_size and k_chunk_size can differ
                    // Q-range = [q_low, q_high)
                    // K-range = [k_low, k_high)
                    // does_overlap = not (q_low >= k_high or k_low >= q_high)
                    // Due to loop bounds, we should never have k_low >= q_high. Can simplify this conditional check
                    // Read mask chunk
                    if (!(q_low_idx >= k_high_idx)) {
                        generate_mask<cb_mask_in>(Sq_chunk_t, Sk_chunk_t, q_chunk, k_chunk);
                    }
                }

                // Wait for compute to deliver output chunk
                cb_wait_front(cb_out, out_chunk_tiles);
                barrier_count = 0;
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
                noc_async_write_barrier();
                cb_pop_front(cb_out, out_chunk_tiles);
            }
        }
    }
}
