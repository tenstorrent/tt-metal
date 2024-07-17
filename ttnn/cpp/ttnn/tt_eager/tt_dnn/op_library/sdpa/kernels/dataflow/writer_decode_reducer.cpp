// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tt_eager/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"
#include "tt_eager/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"

#include "debug/dprint.h"

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
        const uint16_t scalar_val = val>>16;
        volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_id) + tile_id*tile_bytes);
        for (int k = 0; k < 1024; k++) {
            ptr[k] = scalar_val;
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
        // DPRINT << "Fill entire tile to 0 and exit" << ENDL();
        return;
    }
    // DPRINT << "Fill partial tile" << ENDL();
    const uint16_t scalar_val = partial_val>>16;
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_id) + tile_id*tile_bytes);
    int phase_start = (cur_pos_in_tile < 15) ? 0:1;
    uint32_t fill_pos_in_phase = (cur_pos_in_tile+1) % 16;
    // DPRINT << "phase_start: " << phase_start << ENDL();
    // DPRINT << "fill_pos_in_phase: " << fill_pos_in_phase << ENDL();
    if (phase_start == 0) {
        // DPRINT << "Fill second and fourth phase" << ENDL();
        for (int k = 1; k < 4; k+=2) {
            uint32_t idx = k << 8;
            // DPRINT << "k: " << k << ENDL();
            // DPRINT << "idx: " << idx << ENDL();
            for (int j = 0; j < 256; j++) {
                ptr[idx + j] = scalar_val;
            }
        }
    }
    // DPRINT << "Fill phase" << ENDL();
    for (int k = phase_start; k < 4; k+=2) {
        uint32_t idx = k << 8;
        // DPRINT << "k: " << k << ENDL();
        // DPRINT << "idx: " << idx << ENDL();
        for (int j_start_pos = fill_pos_in_phase; j_start_pos < 16; j_start_pos++) {
            for (int j = j_start_pos; j < 256; j+=16) {
                ptr[idx + j] = scalar_val;
            }
        }
    }
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
    constexpr uint32_t NEG_INF = 0xFF80FF80; // TODO: Make sure this is -inf

    cb_reserve_back(cb_mask_in, total_read_tiles);

    uint64_t noc_read_addr_base = get_noc_addr(get_read_ptr(cb_mask_in));
    uint32_t q_write_ptr_base = get_read_ptr(cb_mask_in);
    constexpr uint32_t tile_bytes = get_tile_size(cb_mask_in);

    // DPRINT << "[Writer Reducer] Generate Attention Mask" << ENDL();
    // DPRINT << "k_num_chunks: " << k_num_chunks << ENDL();
    // DPRINT << "cur_pos: " << cur_pos << ENDL();
    // DPRINT << "Sk_chunk_t: " << Sk_chunk_t << ENDL();
    // DPRINT << "cur_pos_in_chunk: " << cur_pos_in_chunk << ENDL();
    // DPRINT << "cur_pos_in_chunk_t: " << cur_pos_in_chunk_t << ENDL();
    // DPRINT << "cur_pos_in_tile: " << cur_pos_in_tile << ENDL();

    for (uint32_t i = 0; i < Sk_chunk_t; ++i) {
        // DPRINT << "iteration " << i << ENDL();
        if (i < cur_pos_in_chunk_t) {
            // DPRINT << "fill with zero" << ENDL();
            // fill with zero
            if (i == 0) {
                fill_tile<tile_bytes>(cb_mask_in, i, 0);
            }
            else {
                copy_tile<tile_bytes>(noc_read_addr_base, q_write_ptr_base, 0, i); // copy from cb_mask_in[0] to cb_mask_in[i]
                if (i == cur_pos_in_chunk_t-1){
                    noc_async_read_barrier();
                }
            }
        }
        else if (i == cur_pos_in_chunk_t) {
            // DPRINT << "fill with partial zero/-inf" << ENDL();
            // fill with partial zero/-inf
            fill_tile_partial<tile_bytes>(cb_mask_in, i, cur_pos_in_tile, NEG_INF);
        }
        else {
            // DPRINT << "fill with -inf" << ENDL();
            // fill with -inf
            if (i == cur_pos_in_chunk_t+1){
                fill_tile<tile_bytes>(cb_mask_in, i, NEG_INF);
            }
            else {
                copy_tile<tile_bytes>(noc_read_addr_base, q_write_ptr_base, cur_pos_in_chunk_t+1, i); // copy from cb_mask_in[cur_pos_in_chunk_t+1] to cb_mask_in[i]
                if (i == Sk_chunk_t-1){
                    noc_async_read_barrier();
                }
            }
        }
        for (uint32_t j = 1; j < PNHt; ++j) {
            // DPRINT << "Should not reach" << ENDL();
            // copy from cb_mask_in[i] to cb_mask_in[j*Sk_chunk_t + i]
            copy_tile<tile_bytes>(noc_read_addr_base, q_write_ptr_base, i, j*Sk_chunk_t + i);
            if (j == PNHt-1){
                noc_async_read_barrier();
            }
        }
    }

    cb_push_back(cb_mask_in, total_read_tiles);
}

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);  // batch size
    constexpr uint32_t PNHt = get_compile_time_arg_val(1);  // padded number of heads in tiles
    constexpr uint32_t St = get_compile_time_arg_val(2);  // full sequence length of kv cache in tiles
    constexpr uint32_t DHt = get_compile_time_arg_val(3);  // head dim
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(4);
    constexpr uint32_t scale_val = get_compile_time_arg_val(5);
    constexpr uint32_t num_cores_per_batch = get_compile_time_arg_val(6);  // num cores per batch
    constexpr uint32_t num_cores = get_compile_time_arg_val(7);  // num running cores in total
    constexpr uint32_t in0_receiver_semaphore_addr   = get_compile_time_arg_val(8);  // semaphore for reciever
    constexpr uint32_t cur_batch = get_compile_time_arg_val(9);
    constexpr bool is_out_sharded = get_compile_time_arg_val(10);

    const uint32_t out_addr  = get_arg_val<uint32_t>(0);
    const uint32_t cur_pos = get_arg_val<uint32_t>(1);
    const uint32_t PSt = get_arg_val<uint32_t>(2);  // padded layer length in tiles
    const uint32_t k_num_chunks = get_arg_val<uint32_t>(3);  // number of chunks in K, where k_num_chunks*Sk_chunk_t = PSt
    const uint32_t k_chunk_start = get_arg_val<uint32_t>(4);
    const uint32_t k_chunk_end = get_arg_val<uint32_t>(5);

    constexpr uint32_t out_chunk_tiles = PNHt * DHt;
    constexpr uint32_t num_cores_to_wait = num_cores_per_batch-1;
    constexpr uint32_t num_tiles_to_wait = (out_chunk_tiles+2*PNHt)*num_cores_to_wait;

    constexpr bool is_dram = true;
    constexpr uint32_t cb_out = tt::CB::c_out4;
    constexpr uint32_t cb_intermed_out = tt::CB::c_out3;  // this cb holds the output intermediates from other worker cores
    constexpr uint32_t cb_out_o = tt::CB::c_out0;
    constexpr uint32_t cb_m_in = tt::CB::c_in6;
    constexpr uint32_t cb_l_in = tt::CB::c_in7;

    constexpr uint32_t tile_bytes = get_tile_size(cb_out);
    constexpr uint32_t tile_bytes_intermed = get_tile_size(cb_intermed_out);
    constexpr DataFormat data_format = get_dataformat(cb_out);

    const InterleavedAddrGenFast<is_dram> out_writer = {
        .bank_base_address = out_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };

    uint64_t intermed_l1_read_addr = get_noc_addr(get_read_ptr(cb_intermed_out));

    volatile tt_l1_ptr uint32_t* in0_receiver_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_receiver_semaphore_addr);

    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<tile_bytes, num_cores>();
    uint32_t barrier_count = 0;

    constexpr uint32_t cb_mask_in = tt::CB::c_in3;
    constexpr uint32_t cb_scale_in = tt::CB::c_in4;
    constexpr uint32_t cb_identity_scale_in = tt::CB::c_in5;

    // generate and send scalar to compute
    generate_bcast_unary_scalar(cb_scale_in, scale_val);
    generate_reduce_scaler(cb_identity_scale_in, identity_scalar_packed);
    generate_mask<cb_mask_in, PNHt>(k_num_chunks, PSt, cur_pos);

    // DPRINT << "[Writer Reducer] Pushed statistics to copmute" << ENDL();

    if (k_chunk_end - k_chunk_start < k_num_chunks){
        // This indicates that there are computes done by other workers. Needs to wait for them and send to reducer's compute
        // Wait for compute to deliver output chunk, and write to compute again for reduction
        // data in cb_intermed_out is arranged as [o,m,l,o,m,l,...] with size (out_chunk_tiles + 2*PNHt)*num_cores_to_wait
        // wait on in0 semaphore value to become VALID (set by sender)
        // DPRINT << "[Writer Reducer] Waiting for semaphore to be set from "<< num_cores_to_wait << " cores" << ENDL();
        noc_semaphore_wait(in0_receiver_semaphore_addr_ptr, num_cores_to_wait);
        // DPRINT << "[Writer Reducer] Received signal that semaphore has set" << ENDL();
        // noc_semaphore_set(in0_receiver_semaphore_addr_ptr, 0);

        // cb_wait_front(cb_intermed_out, num_tiles_to_wait);
        constexpr uint32_t q_read_size = out_chunk_tiles*tile_bytes_intermed;
        constexpr uint32_t ml_read_size = PNHt*tile_bytes_intermed;
        // DPRINT << "[Writer Reducer] Received intermediate chunks from worker cores" << ENDL();
        // DPRINT << "[Writer Reducer] Sending intermediate chunks to compute" << ENDL();
        for(uint32_t block = 0; block < num_cores_per_batch; ++block) {

            // DPRINT << "[Writer Reducer] Iteration " << block << ENDL();
            cb_reserve_back(cb_out_o, out_chunk_tiles);
            cb_reserve_back(cb_m_in, PNHt);
            cb_reserve_back(cb_l_in, PNHt);
            // DPRINT << "[Writer Reducer] Reserved space in cb for Q, M, L" << ENDL();

            uint32_t q_write_ptr = get_read_ptr(cb_out_o);
            noc_async_read(intermed_l1_read_addr, q_write_ptr, q_read_size);
            intermed_l1_read_addr+=q_read_size;
            noc_async_read_barrier();
            cb_push_back(cb_out_o, out_chunk_tiles);
            // DPRINT << "[Writer Reducer] pushed Q" << ENDL();

            uint32_t m_write_ptr = get_read_ptr(cb_m_in);
            noc_async_read(intermed_l1_read_addr, m_write_ptr, ml_read_size);
            intermed_l1_read_addr+=ml_read_size;
            noc_async_read_barrier();
            cb_push_back(cb_m_in, PNHt);
            // DPRINT << "[Writer Reducer] pushed M" << ENDL();

            uint32_t l_write_ptr = get_read_ptr(cb_l_in);
            noc_async_read(intermed_l1_read_addr, l_write_ptr, ml_read_size);
            intermed_l1_read_addr+=ml_read_size;
            noc_async_read_barrier();
            cb_push_back(cb_l_in, PNHt);
            // DPRINT << "[Writer Reducer] pushed L" << ENDL();

            // DPRINT << "[Writer Reducer] Done iteration " << block << ENDL();
        }
        // cb_pop_front(cb_intermed_out, num_tiles_to_wait);

        // DPRINT << "[Writer Reducer] Done sending intermediate chunks to compute" << ENDL();
    }

    // Offset for current batch
    const uint32_t out_batch_offset = cur_batch * out_chunk_tiles;

    // Write entire out into its corresponding batch
    uint32_t out_tile_id = out_batch_offset;
    cb_wait_front(cb_out, out_chunk_tiles);

    // DPRINT << "[Writer Reducer] recieved output chunk from reduce compute" << ENDL();
    if (! is_out_sharded){
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
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, out_chunk_tiles);

    // DPRINT << "[Writer Reducer] Wrote output chunk to memory. Done Reduce Writer" << ENDL();

}
