// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "debug/assert.h"

#include "../../rt_args_common.hpp"

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
void fill_tile_partial(uint32_t cb_id, uint32_t tile_id, uint32_t cur_pos_in_tile, uint32_t partial_val) {
    /*
    We want to fill cur_pos_in_tile + 1 to the end
    */

    fill_tile<tile_bytes>(cb_id, tile_id, 0);
    if (cur_pos_in_tile == 31 || partial_val == 0) {
        return;
    }
    const uint16_t datum_val = partial_val>>16;
    volatile tt_l1_ptr uint16_t* uint16_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_id) + tile_id*tile_bytes);
    volatile tt_l1_ptr uint32_t* uint32_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id) + tile_id*tile_bytes);
    int face_start = (cur_pos_in_tile < 15) ? 0:1;
    uint32_t fill_pos_in_face = (cur_pos_in_tile+1) % 16;
    if (face_start == 0) {
        // Fill 2 datums in each writes to optimize for performance
        constexpr int num_uint32_datums_tile_face = (16 * 16) / 2;
        for (int k = 1; k < 4; k+=2) {
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
    constexpr uint32_t num_cols_in_uint32_face = num_cols_in_face>>1;
    for (int k = face_start; k < 4; k+=2) {
        uint32_t uint16_face_idx = k << 8;
        uint32_t uint32_face_idx = k << 7;

        for (uint32_t face_row_idx = 0; face_row_idx < num_rows_in_face; face_row_idx++) {
            // Here, if the fill_pos_in_face is odd then we fill that pos with single uint16 value
            if(is_odd_pos_filled){
                uint16_ptr[uint16_face_idx + (fill_pos_in_face + num_cols_in_face * face_row_idx)] = datum_val;
            }

            for (uint32_t uint32_face_col_idx = fill_pos_in_uint32_face; uint32_face_col_idx < num_cols_in_uint32_face; uint32_face_col_idx++) {

                uint32_ptr[uint32_face_idx + (uint32_face_col_idx + num_cols_in_uint32_face * face_row_idx)] = partial_val;
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

    for (uint32_t i = 0; i < Sk_chunk_t; ++i) {
        if (i < cur_pos_in_chunk_t) {
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
            // fill with partial zero/-inf
            fill_tile_partial<tile_bytes>(cb_mask_in, i, cur_pos_in_tile, NEG_INF);
        }
        else {
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
            // copy from cb_mask_in[i] to cb_mask_in[j*Sk_chunk_t + i]
            copy_tile<tile_bytes>(noc_read_addr_base, q_write_ptr_base, i, j*Sk_chunk_t + i);
            if (j == PNHt-1){
                noc_async_read_barrier();
            }
        }
    }

    cb_push_back(cb_mask_in, total_read_tiles);
}

template <uint32_t out_chunk_tiles, uint32_t cb_out, uint32_t cb_out_m, uint32_t cb_out_l, uint32_t cb_intermed_out, uint32_t PNHt>
void worker_compute(uint64_t in0_sender_semaphore_noc_addr, uint32_t worker_id, uint32_t reduce_core_noc_x, uint32_t reduce_core_noc_y) {

    uint32_t out_tile_id = 0;

    // Wait for compute to deliver output chunk
    cb_wait_front(cb_out, out_chunk_tiles);
    cb_wait_front(cb_out_m, PNHt);
    cb_wait_front(cb_out_l, PNHt);

    // Write output chunk to reducer
    constexpr uint32_t tile_bytes = get_tile_size(cb_out);
    uint32_t worker_offset = worker_id * (out_chunk_tiles+2*PNHt)*tile_bytes;
    constexpr uint32_t o_write_size = out_chunk_tiles*tile_bytes;
    constexpr uint32_t ml_write_size = PNHt*tile_bytes;
    uint64_t output_write_addr = get_noc_addr(reduce_core_noc_x, reduce_core_noc_y, get_write_ptr(cb_intermed_out)) + worker_offset;
    noc_async_write(get_read_ptr(cb_out), output_write_addr, o_write_size);
    output_write_addr+=o_write_size;
    noc_async_write(get_read_ptr(cb_out_m), output_write_addr, ml_write_size);
    output_write_addr+=ml_write_size;
    noc_async_write(get_read_ptr(cb_out_l), output_write_addr, ml_write_size);

    // increment semaphore
    noc_async_write_barrier();
    noc_semaphore_inc(in0_sender_semaphore_noc_addr, 1);

    // pop front
    cb_pop_front(cb_out, out_chunk_tiles);
    cb_pop_front(cb_out_m, PNHt);
    cb_pop_front(cb_out_l, PNHt);
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
    uint32_t reducer_semaphore_addr   = get_semaphore(get_compile_time_arg_val(8));  // semaphore for reducer
    uint32_t output_semaphore_addr   = get_semaphore(get_compile_time_arg_val(9));  // semaphore for sender
    constexpr bool is_out_sharded = get_compile_time_arg_val(10);
    constexpr uint32_t k_chunk_size = get_compile_time_arg_val(11);
    constexpr uint32_t num_q_heads = get_compile_time_arg_val(12);
    constexpr uint32_t num_kv_heads = get_compile_time_arg_val(13);
    constexpr uint32_t num_cores_per_head = get_compile_time_arg_val(14);
    constexpr uint32_t num_heads_per_core = get_compile_time_arg_val(15);
    constexpr uint32_t num_reducer_cores = get_compile_time_arg_val(16);
    constexpr uint32_t num_output_cores = get_compile_time_arg_val(17);
    constexpr uint32_t ELEMENT_SIZE = get_compile_time_arg_val(18);

    uint32_t arg_idx = 0;
    const uint32_t out_addr  = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t worker_id_for_reduce = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t worker_id_for_output = get_arg_val<uint32_t>(arg_idx++);
    const bool is_worker = get_arg_val<uint32_t>(arg_idx++) == 0;
    const bool do_output = get_arg_val<uint32_t>(arg_idx++) == 1;
    const uint32_t cur_head_group = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t cur_batch = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_num_in_reduce = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_num_in_output = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t cur_pos_arg = get_arg_val<uint32_t>(arg_idx++);

    // idle core
    if (out_addr == 0){
        return;
    }
    // Get cur_pos
    uint32_t cur_pos = 0;
    // using UINT32_MAX as a flag to indicate that cur_pos is not provided as a list
    if (cur_pos_arg != UINT32_MAX){
        cur_pos = cur_pos_arg;
    }
    else {
        constexpr uint32_t cb_index_id = tt::CB::dataflow0;
        cb_wait_front(cb_index_id, 1);
        uint32_t index_cb_ptr = get_read_ptr(cb_index_id);
        volatile tt_l1_ptr uint32_t* index_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_cb_ptr);
        cur_pos = index_ptr[cur_batch];
    }

    if (cur_pos == UINT32_MAX) {
        // cur_pos of -1 indicates that the user should be skipped
        return;
    }
    // Sequence length assignment
    auto [PSt, k_num_chunks, k_chunk_start, k_chunk_end] = get_runtime_args(cur_pos, cur_batch, core_num_in_reduce, num_cores_per_head, k_chunk_size);

    tt_l1_ptr uint32_t * all_reducer_noc_x          = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_reducer_cores;
    tt_l1_ptr uint32_t * all_reducer_noc_y          = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_reducer_cores;
    tt_l1_ptr uint32_t * all_output_noc_x          = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_output_cores;
    tt_l1_ptr uint32_t * all_output_noc_y          = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx++));

    uint32_t reduce_core_index = (cur_batch * num_cores_per_batch) / num_cores_per_head + cur_head_group;
    uint32_t reduce_core_noc_x = all_reducer_noc_x[reduce_core_index];
    uint32_t reduce_core_noc_y = all_reducer_noc_y[reduce_core_index];

    const uint64_t in0_sender_semaphore_noc_addr = get_noc_addr(reduce_core_noc_x, reduce_core_noc_y, reducer_semaphore_addr);

    if (k_chunk_start == k_chunk_end) {
        return; // early exit because no computes needs to be done
    }

    constexpr uint32_t out_chunk_tiles = PNHt * DHt;
    uint32_t num_cores_to_wait = num_cores_per_head-1;
    if (num_cores_per_head>k_num_chunks) num_cores_to_wait = k_num_chunks-1;
    uint32_t num_tiles_to_wait = (out_chunk_tiles+2*PNHt)*num_cores_to_wait;

    constexpr bool is_dram = true;
    constexpr uint32_t cb_out = tt::CB::c_out4;
    constexpr uint32_t cb_intermed_out = tt::CB::c_out3;  // this cb holds the output intermediates from other worker cores
    constexpr uint32_t cb_out_o = tt::CB::c_out0;
    constexpr uint32_t cb_m_in = tt::CB::c_in6;
    constexpr uint32_t cb_l_in = tt::CB::c_in7;

    constexpr uint32_t cb_mask_in = tt::CB::c_in3;
    constexpr uint32_t cb_scale_in = tt::CB::c_in4;
    constexpr uint32_t cb_identity_scale_in = tt::CB::c_in5;

    constexpr uint32_t cb_out_worker = tt::CB::c_out0;
    constexpr uint32_t cb_out_m = tt::CB::c_out1;
    constexpr uint32_t cb_out_l = tt::CB::c_out2;

    // generate and send scaler to compute
    generate_bcast_unary_scalar(cb_scale_in, scale_val);
    generate_reduce_scaler(cb_identity_scale_in, identity_scalar_packed);
    if (is_worker) {
        ASSERT(num_heads_per_core == 1);  // if there are workers, then head must be split across workers so there should not be more than one head per core
        worker_compute<out_chunk_tiles, cb_out_worker, cb_out_m, cb_out_l, cb_intermed_out, PNHt>(in0_sender_semaphore_noc_addr, worker_id_for_reduce, reduce_core_noc_x, reduce_core_noc_y);
        return;
    }

    // *** Reducer Compute Below ***
    constexpr uint32_t tile_bytes = get_tile_size(cb_out);
    constexpr uint32_t tile_bytes_intermed = get_tile_size(cb_intermed_out);
    constexpr DataFormat data_format = get_dataformat(cb_out);

    const InterleavedAddrGenFast<is_dram> out_writer = {
        .bank_base_address = out_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };

    uint64_t intermed_l1_read_addr = get_noc_addr(get_read_ptr(cb_intermed_out));

    volatile tt_l1_ptr uint32_t* in0_receiver_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reducer_semaphore_addr);

    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<tile_bytes, num_cores>();
    uint32_t barrier_count = 0;

    // generate and send mask to compute
    generate_mask<cb_mask_in, PNHt>(k_num_chunks, PSt, cur_pos);

    for (uint32_t cur_head = cur_head_group*num_heads_per_core; cur_head < cur_head_group*num_heads_per_core + num_heads_per_core; ++cur_head) {
        if (k_chunk_end - k_chunk_start < k_num_chunks){
            ASSERT(num_heads_per_core == 1);  // if there are workers, then head must be split across workers so there should not be more than one head per core
            // This indicates that there are computes done by other workers. Needs to wait for them and send to reducer's compute
            // Wait for compute to deliver output chunk, and write to compute again for reduction
            // data in cb_intermed_out is arranged as [o,m,l,o,m,l,...] with size (out_chunk_tiles + 2*PNHt)*num_cores_to_wait
            // wait on in0 semaphore value to become VALID (set by sender)
            noc_semaphore_wait(in0_receiver_semaphore_addr_ptr, num_cores_to_wait);
            // noc_semaphore_set(in0_receiver_semaphore_addr_ptr, 0);

            // cb_wait_front(cb_intermed_out, num_tiles_to_wait);
            constexpr uint32_t q_read_size = out_chunk_tiles*tile_bytes_intermed;
            constexpr uint32_t ml_read_size = PNHt*tile_bytes_intermed;
            for(uint32_t block = 0; block < num_cores_to_wait+1; ++block) {

                cb_reserve_back(cb_out_o, out_chunk_tiles);
                cb_reserve_back(cb_m_in, PNHt);
                cb_reserve_back(cb_l_in, PNHt);

                uint32_t q_write_ptr = get_read_ptr(cb_out_o);
                noc_async_read(intermed_l1_read_addr, q_write_ptr, q_read_size);
                intermed_l1_read_addr+=q_read_size;
                noc_async_read_barrier();
                cb_push_back(cb_out_o, out_chunk_tiles);

                uint32_t m_write_ptr = get_read_ptr(cb_m_in);
                noc_async_read(intermed_l1_read_addr, m_write_ptr, ml_read_size);
                intermed_l1_read_addr+=ml_read_size;
                noc_async_read_barrier();
                cb_push_back(cb_m_in, PNHt);

                uint32_t l_write_ptr = get_read_ptr(cb_l_in);
                noc_async_read(intermed_l1_read_addr, l_write_ptr, ml_read_size);
                intermed_l1_read_addr+=ml_read_size;
                noc_async_read_barrier();
                cb_push_back(cb_l_in, PNHt);
            }
        }
        // Offset for current batch
        const uint32_t out_batch_offset = cur_batch * out_chunk_tiles;

        // Write entire out into its corresponding batch
        uint32_t out_tile_id = out_batch_offset;
        cb_wait_front(cb_out, out_chunk_tiles);

        if constexpr(num_kv_heads > 1){
            // if gqa, we will need to write partial outputs for each head
            constexpr uint32_t TILE_WIDTH = 32;
            // we are assuming here that num_heads_to_write = nh/nkv is a power of 2 here, so that we don't write partial across phase
            uint32_t num_heads_to_write = num_q_heads/num_kv_heads; // each head is one row in a tile
            uint32_t SUBTILE_LINE_BYTES = 16*ELEMENT_SIZE; //size of 16 elements (in a row)
            uint32_t starting_row = cur_head * num_heads_to_write;
            uint32_t in_tile_offset_by_starting_head = starting_row < 16 ? starting_row * SUBTILE_LINE_BYTES : (starting_row - 16) * SUBTILE_LINE_BYTES + 512*ELEMENT_SIZE;

            if (! is_out_sharded){
                for (uint32_t tile = 0; tile < out_chunk_tiles; ++tile) {

                    uint64_t out_writer_noc_addr = get_noc_addr(out_tile_id, out_writer) + in_tile_offset_by_starting_head;
                    uint32_t l1_read_addr = get_read_ptr(cb_out) + tile*tile_bytes + in_tile_offset_by_starting_head;

                    // write partial output for each head
                    for (uint32_t head = 0; head < num_heads_to_write; ++head) {

                        // Write first phase
                        noc_async_write(l1_read_addr, out_writer_noc_addr, SUBTILE_LINE_BYTES);

                        // Write second phase
                        noc_async_write(l1_read_addr+256*ELEMENT_SIZE, out_writer_noc_addr+256*ELEMENT_SIZE, SUBTILE_LINE_BYTES);

                        l1_read_addr += SUBTILE_LINE_BYTES;
                        out_writer_noc_addr += SUBTILE_LINE_BYTES;

                        if (++barrier_count == barrier_threshold) {
                            noc_async_writes_flushed();
                            barrier_count = 0;
                        }
                    }

                    ++out_tile_id;
                }
            }
            // sharded out case
            else if (do_output){
                // read from reducer cores
                constexpr uint32_t num_reducers_per_output = num_reducer_cores / num_output_cores;
                constexpr uint32_t num_reducers_to_wait = num_reducers_per_output-1;
                volatile tt_l1_ptr uint32_t* output_self_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(output_semaphore_addr);
                noc_semaphore_wait(output_self_semaphore_addr_ptr, num_reducers_to_wait);

                uint32_t reduce_core_read_index_start = (cur_batch * num_cores_per_batch) / num_cores_per_head;

                for (uint32_t reduce_core_read_index = reduce_core_read_index_start + 1; reduce_core_read_index < reduce_core_read_index_start+num_reducers_per_output; reduce_core_read_index++){
                    uint32_t reduce_core_read_noc_x = all_reducer_noc_x[reduce_core_read_index];
                    uint32_t reduce_core_read_noc_y = all_reducer_noc_y[reduce_core_read_index];

                    uint64_t out_reader_base_noc_addr = get_noc_addr(reduce_core_read_noc_x, reduce_core_read_noc_y, get_read_ptr(cb_out)) + in_tile_offset_by_starting_head;

                    for (uint32_t tile = 0; tile < out_chunk_tiles; ++tile) {
                        uint32_t l1_write_addr = get_write_ptr(cb_out) + tile*tile_bytes + in_tile_offset_by_starting_head;
                        uint32_t out_reader_noc_addr = out_reader_base_noc_addr;

                        // write partial output for each head
                        for (uint32_t head = 0; head < num_heads_to_write; ++head) {

                            // Write first phase
                            noc_async_read(out_reader_noc_addr, l1_write_addr, SUBTILE_LINE_BYTES);

                            // Write second phase
                            noc_async_read(out_reader_noc_addr+256*ELEMENT_SIZE, l1_write_addr+256*ELEMENT_SIZE, SUBTILE_LINE_BYTES);

                            l1_write_addr += SUBTILE_LINE_BYTES;
                            out_reader_noc_addr += SUBTILE_LINE_BYTES;

                            if (++barrier_count == barrier_threshold) {
                                noc_async_read_barrier();
                                barrier_count = 0;
                            }
                        }
                        out_reader_noc_addr += tile_bytes;
                    }
                }
                noc_async_read_barrier();
            } else {
                // tell output core that its output is ready
                uint32_t output_core_noc_x = all_output_noc_x[cur_batch];
                uint32_t output_core_noc_y = all_output_noc_y[cur_batch];
                const uint64_t output_core_semaphore_noc_addr = get_noc_addr(output_core_noc_x, output_core_noc_y, output_semaphore_addr);
                noc_semaphore_inc(output_core_semaphore_noc_addr, 1);
            }
        } else {
            // if mqa, we don't need to gather outputs for other heads so we can just write entire tiles to memory
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
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, out_chunk_tiles);
    }
}
