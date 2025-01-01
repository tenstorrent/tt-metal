// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "debug/assert.h"

#include "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/transformer/speculative_sdpa_decode/device/kernels/speculative_common.hpp"

#include "debug/dprint.h"  // required in all kernels using DPRINT

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);     // batch size
    constexpr uint32_t PNHt = get_compile_time_arg_val(1);  // padded number of heads in tiles
    constexpr uint32_t St = get_compile_time_arg_val(2);    // full sequence length of kv cache in tiles
    constexpr uint32_t DHt = get_compile_time_arg_val(3);   // head dim
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(4);  // number of tiles in seqlen of a k/v/mask chunk
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(5);
    constexpr uint32_t scale_val = get_compile_time_arg_val(6);
    constexpr uint32_t num_cores_per_batch = get_compile_time_arg_val(7);          // num cores per batch
    constexpr uint32_t num_cores = get_compile_time_arg_val(8);                    // num running cores in total
    uint32_t reducer_semaphore_addr = get_semaphore(get_compile_time_arg_val(9));  // semaphore for reducer
    uint32_t output_semaphore_addr = get_semaphore(get_compile_time_arg_val(10));  // semaphore for output ready
    constexpr bool is_out_sharded = get_compile_time_arg_val(11);
    constexpr uint32_t k_chunk_size = get_compile_time_arg_val(12);
    constexpr uint32_t num_q_heads = get_compile_time_arg_val(13);
    constexpr uint32_t num_kv_heads = get_compile_time_arg_val(14);
    constexpr uint32_t num_cores_per_head = get_compile_time_arg_val(15);
    constexpr uint32_t num_heads_per_core = get_compile_time_arg_val(16);
    constexpr uint32_t num_reducer_cores = get_compile_time_arg_val(17);
    constexpr uint32_t num_output_cores = get_compile_time_arg_val(18);
    constexpr uint32_t ELEMENT_SIZE = get_compile_time_arg_val(19);
    constexpr bool is_causal = get_compile_time_arg_val(20) == 1;

    constexpr uint32_t Spec_chunk_t =
        get_compile_time_arg_val(21);  // speculative chunk size (in tiles), for the first and last chunk
    constexpr uint32_t speculative_chunk_size = Spec_chunk_t * tt::constants::TILE_HEIGHT;
    constexpr bool use_priority_tensor = get_compile_time_arg_val(22) == 1;
    constexpr uint32_t lambda_val = get_compile_time_arg_val(23);

    uint32_t arg_idx = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t out_spec_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t l2_dist_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t l2_norm_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t priority_addr = get_arg_val<uint32_t>(arg_idx++);
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
    if (out_addr == 0) {
        return;
    }
    // Get cur_pos
    constexpr uint32_t cur_pos_base = St * 32 - 1;
    uint32_t cur_pos = cur_pos_base;  // default to non-causal, which we do attention on the entire kv cache. In this
                                      // case we set cur_pos to the last position
    // using UINT32_MAX as a flag to indicate that cur_pos is not provided as a list
    if constexpr (is_causal) {
        if (cur_pos_arg != UINT32_MAX) {
            cur_pos = cur_pos_arg;
        } else {
            constexpr uint32_t cb_index_id = tt::CBIndex::c_8;
            cb_wait_front(cb_index_id, 1);
            uint32_t index_cb_ptr = get_read_ptr(cb_index_id);
            volatile tt_l1_ptr uint32_t* index_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_cb_ptr);
            cur_pos = index_ptr[cur_batch];
        }

        if (cur_pos == UINT32_MAX) {
            // cur_pos of -1 indicates that the user should be skipped
            return;
        }
    }
    // Sequence length assignment
    auto
        [k_num_chunks,
         k_chunk_start,
         k_chunk_end,
         speculative_height_dim_start_tile_offset1,
         speculative_height_dim_start_tile_offset2,
         non_spec_height_dim_start_tile_offset,
         adjusted_cur_pos_for_non_spec,
         adjusted_cur_pos_for_spec,
         do_speculative_compute] =
            get_speculative_runtime_args(
                cur_pos,
                cur_batch,
                core_num_in_reduce,
                num_cores_per_head,
                k_chunk_size,
                speculative_chunk_size,
                Spec_chunk_t);

    // Get semaphore noc addresses for reducer and output cores
    tt_l1_ptr uint32_t* all_reducer_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_reducer_cores;
    tt_l1_ptr uint32_t* all_reducer_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_reducer_cores;
    tt_l1_ptr uint32_t* all_output_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_output_cores;
    tt_l1_ptr uint32_t* all_output_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx++));

    uint32_t reduce_core_index = (cur_batch * num_cores_per_batch) / num_cores_per_head + cur_head_group;
    uint32_t reduce_core_noc_x = all_reducer_noc_x[reduce_core_index];
    uint32_t reduce_core_noc_y = all_reducer_noc_y[reduce_core_index];

    const uint64_t reducer_semaphore_noc_addr =
        get_noc_addr(reduce_core_noc_x, reduce_core_noc_y, reducer_semaphore_addr);

    uint32_t output_core_noc_x = all_output_noc_x[cur_batch];
    uint32_t output_core_noc_y = all_output_noc_y[cur_batch];

    const uint64_t output_semaphore_noc_addr =
        get_noc_addr(output_core_noc_x, output_core_noc_y, output_semaphore_addr);

    if (k_chunk_start == k_chunk_end) {
        return;  // early exit because no computes needs to be done
    }

    constexpr uint32_t out_chunk_tiles = PNHt * DHt;
    uint32_t num_cores_to_wait = num_cores_per_head - 1;
    if (num_cores_per_head > k_num_chunks) {
        num_cores_to_wait = k_num_chunks - 1;
    }
    num_cores_to_wait += 1;  // add 1 for speculative compute (specific to this kernel)
    uint32_t num_tiles_to_wait = (out_chunk_tiles + 2 * PNHt) * num_cores_to_wait;

    constexpr bool is_dram = true;
    constexpr uint32_t cb_out = tt::CBIndex::c_20;
    constexpr uint32_t cb_intermed_out =
        tt::CBIndex::c_19;  // this cb holds the output intermediates from other worker cores
    constexpr uint32_t cb_out_o = tt::CBIndex::c_16;
    constexpr uint32_t cb_m_in = tt::CBIndex::c_6;
    constexpr uint32_t cb_l_in = tt::CBIndex::c_7;

    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_scale_in = tt::CBIndex::c_4;
    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;

    constexpr uint32_t cb_out_worker = tt::CBIndex::c_16;
    constexpr uint32_t cb_out_m = tt::CBIndex::c_17;
    constexpr uint32_t cb_out_l = tt::CBIndex::c_18;

    // generate and send scaler to compute
    generate_bcast_unary_scalar(cb_scale_in, scale_val);
    generate_reduce_scaler(cb_identity_scale_in, identity_scalar_packed);

    constexpr uint32_t tile_bytes = get_tile_size(cb_out);
    constexpr uint32_t tile_bytes_intermed = get_tile_size(cb_intermed_out);
    constexpr DataFormat data_format = get_dataformat(cb_out);

    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<tile_bytes, num_cores>();
    uint32_t barrier_count = 0;

    if (do_speculative_compute) {
        DPRINT << "do_spec_compute" << ENDL();
        if constexpr (is_causal) {
            generate_mask<cb_mask_in, PNHt, Spec_chunk_t>(
                1 /*assume 1 num chunks and use adjusted_cur_pos_for_spec*/, adjusted_cur_pos_for_spec);
        }
        worker_compute<out_chunk_tiles, cb_out_worker, cb_out_m, cb_out_l, cb_intermed_out, PNHt>(
            reducer_semaphore_noc_addr,
            0 /*speculative output should always have index=0 in intermed cb*/,
            reduce_core_noc_x,
            reduce_core_noc_y);

        // write speculative output to memory
        cb_wait_front(cb_out, out_chunk_tiles);
        const InterleavedAddrGenFast<is_dram> out_spec_writer = {
            .bank_base_address = out_spec_addr, .page_size = tile_bytes, .data_format = data_format};

        const uint32_t out_batch_offset = cur_batch * out_chunk_tiles;
        uint32_t out_tile_id = out_batch_offset;

        if constexpr (num_kv_heads > 1) {
            // if gqa, we will need to write partial outputs for each head
            // we are assuming here that num_heads_to_write = nh/nkv is a power of 2 here, so that we don't write
            // partial across phase
            constexpr uint32_t num_heads_to_write = num_q_heads / num_kv_heads;  // each head is one row in a tile

            if (!is_out_sharded) {
                barrier_count = write_partial_tiles_to_memory<cb_out, ELEMENT_SIZE, barrier_threshold>(
                    out_tile_id,
                    out_spec_writer,
                    barrier_count,
                    cur_head_group * num_heads_per_core /*cur_head*/,
                    num_heads_to_write,
                    out_chunk_tiles);
            }
            // sharded out case
            else {
                ASSERT(false);  // TODO: implement sharded out case. Not supported yet
            }
        } else {
            // if mqa, we don't need to gather outputs for other heads so we can just write entire tiles to memory
            if (!is_out_sharded) {
                barrier_count = write_tiles_to_memory<cb_out, out_chunk_tiles, barrier_threshold>(
                    out_tile_id, out_spec_writer, barrier_count);
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, out_chunk_tiles);
        noc_semaphore_inc(output_semaphore_noc_addr, 1);  // signal speculative output is ready
        DPRINT << "done spec_compute" << ENDL();
    }

    if (is_worker) {
        DPRINT << "do_worker_compute" << ENDL();
        ASSERT(num_heads_per_core == 1);  // if there are workers, then head must be split across workers so there
                                          // should not be more than one head per core
        worker_compute<out_chunk_tiles, cb_out_worker, cb_out_m, cb_out_l, cb_intermed_out, PNHt>(
            reducer_semaphore_noc_addr,
            worker_id_for_reduce + 1 /*index=0 is reserved for speculative output*/,
            reduce_core_noc_x,
            reduce_core_noc_y);
        DPRINT << "done with worker_compute" << ENDL();
        return;
    }

    // *** Reducer Compute Below ***
    DPRINT << "do_reducer_compute" << ENDL();
    const InterleavedAddrGenFast<is_dram> out_writer = {
        .bank_base_address = out_addr, .page_size = tile_bytes, .data_format = data_format};

    uint64_t intermed_l1_read_addr = get_noc_addr(get_read_ptr(cb_intermed_out));

    volatile tt_l1_ptr uint32_t* in0_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reducer_semaphore_addr);

    // generate and send mask to compute if causal
    if constexpr (is_causal) {
        generate_mask<cb_mask_in, PNHt, Sk_chunk_t>(k_num_chunks, adjusted_cur_pos_for_non_spec);
    }

    for (uint32_t cur_head = cur_head_group * num_heads_per_core;
         cur_head < cur_head_group * num_heads_per_core + num_heads_per_core;
         ++cur_head) {
        ASSERT(num_heads_per_core == 1);  // if there are workers, then head must be split across workers so there
                                          // should not be more than one head per core
        // This indicates that there are computes done by other workers. Needs to wait for them and send to
        // reducer's compute Wait for compute to deliver output chunk, and write to compute again for reduction data
        // in cb_intermed_out is arranged as [o,m,l,o,m,l,...] with size (out_chunk_tiles +
        // 2*PNHt)*num_cores_to_wait wait on in0 semaphore value to become VALID (set by sender)
        DPRINT << "waiting for in0 semaphore " << num_cores_to_wait << ENDL();
        noc_semaphore_wait(in0_receiver_semaphore_addr_ptr, num_cores_to_wait);
        DPRINT << "got sem inc" << ENDL();
        // noc_semaphore_set(in0_receiver_semaphore_addr_ptr, 0);

        // cb_wait_front(cb_intermed_out, num_tiles_to_wait);
        constexpr uint32_t q_read_size = out_chunk_tiles * tile_bytes_intermed;
        constexpr uint32_t ml_read_size = PNHt * tile_bytes_intermed;
        for (uint32_t block = 0; block < num_cores_to_wait; ++block) {
            DPRINT << "block " << block << ENDL();
            cb_reserve_back(cb_out_o, out_chunk_tiles);
            DPRINT << "b1" << ENDL();
            cb_reserve_back(cb_m_in, PNHt);
            DPRINT << "b2" << ENDL();
            cb_reserve_back(cb_l_in, PNHt);

            DPRINT << "ckpt1" << ENDL();
            uint32_t q_write_ptr = get_read_ptr(cb_out_o);
            noc_async_read(intermed_l1_read_addr, q_write_ptr, q_read_size);
            intermed_l1_read_addr += q_read_size;
            noc_async_read_barrier();
            cb_push_back(cb_out_o, out_chunk_tiles);

            DPRINT << "ckpt2" << ENDL();
            uint32_t m_write_ptr = get_read_ptr(cb_m_in);
            noc_async_read(intermed_l1_read_addr, m_write_ptr, ml_read_size);
            intermed_l1_read_addr += ml_read_size;
            noc_async_read_barrier();
            cb_push_back(cb_m_in, PNHt);

            DPRINT << "ckpt3" << ENDL();
            uint32_t l_write_ptr = get_read_ptr(cb_l_in);
            noc_async_read(intermed_l1_read_addr, l_write_ptr, ml_read_size);
            intermed_l1_read_addr += ml_read_size;
            noc_async_read_barrier();
            cb_push_back(cb_l_in, PNHt);
        }

        // Offset for current batch
        const uint32_t out_batch_offset = cur_batch * out_chunk_tiles;

        DPRINT << "done send intermed" << ENDL();

        // Write entire out into its corresponding batch
        uint32_t out_tile_id = out_batch_offset;
        cb_wait_front(cb_out, out_chunk_tiles);

        DPRINT << "start write out" << ENDL();

        if constexpr (num_kv_heads > 1) {
            // if gqa, we will need to write partial outputs for each head
            // we are assuming here that num_heads_to_write = nh/nkv is a power of 2 here, so that we don't write
            // partial across phase
            constexpr uint32_t num_heads_to_write = num_q_heads / num_kv_heads;  // each head is one row in a tile

            if (!is_out_sharded) {
                barrier_count = write_partial_tiles_to_memory<cb_out, ELEMENT_SIZE, barrier_threshold>(
                    out_tile_id, out_writer, barrier_count, cur_head, num_heads_to_write, out_chunk_tiles);
            }
            // sharded out case
            else {
                ASSERT(false);  // TODO: implement sharded out case. Not supported yet
            }
        } else {
            // if mqa, we don't need to gather outputs for other heads so we can just write entire tiles to memory
            if (!is_out_sharded) {
                barrier_count = write_tiles_to_memory<cb_out, out_chunk_tiles, barrier_threshold>(
                    out_tile_id, out_writer, barrier_count);
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, out_chunk_tiles);
    }
    noc_semaphore_inc(output_semaphore_noc_addr, 1);  // signal ground truth output is ready

    // assume output core also does verification
    if (do_output) {
        constexpr uint32_t l2_norm_scalar_bytes = 2 /*2 bytes for bfloat16*/;

        if constexpr (num_q_heads < 32) {
            // push in tile mask to mask out rows > num_q_heads
            constexpr uint32_t tile_bytes = get_tile_size(cb_scale_in);
            constexpr uint32_t ONE = 0x3F803F80;  // 1.0 in bfloat16 double packed
            constexpr uint32_t ZERO = 0x00000000;
            cb_reserve_back(cb_scale_in, 1);
            fill_tile_partial_transposed<tile_bytes>(cb_scale_in, 0, num_q_heads - 1, ZERO, ONE);
            cb_push_back(cb_scale_in, 1);
        }

        // wait for ground truth norm to be ready
        const InterleavedAddrGen<is_dram> s_out_norm = {
            .bank_base_address = l2_norm_addr, .page_size = l2_norm_scalar_bytes};
        uint64_t l2_norm_noc_addr = get_noc_addr(cur_batch, s_out_norm);
        DPRINT << "l2 norm addr: " << get_read_ptr(cb_out_m) << ENDL();
        cb_wait_front(cb_out_m, 1);
        // read in norm value
        noc_async_write(get_read_ptr(cb_out_m), l2_norm_noc_addr, l2_norm_scalar_bytes);
        noc_async_write_barrier();

        // wait for dist norm to be ready
        const InterleavedAddrGen<is_dram> s_out_dist_norm = {
            .bank_base_address = l2_dist_addr, .page_size = l2_norm_scalar_bytes};
        uint64_t l2_dist_norm_noc_addr = get_noc_addr(cur_batch, s_out_dist_norm);
        DPRINT << "l2 dist norm addr: " << get_read_ptr(cb_out_l) << ENDL();
        cb_wait_front(cb_out_l, 1);
        // read in dist norm value
        noc_async_write(get_read_ptr(cb_out_l), l2_dist_norm_noc_addr, l2_norm_scalar_bytes);
        noc_async_write_barrier();

        {
            // Use union for safe type punning
            union {
                uint32_t i;
                float f;
            } converter;
            // read in norm value
            volatile tt_l1_ptr uint16_t* norm_val_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(cb_out_m));
            converter.i = static_cast<uint32_t>(norm_val_ptr[0]) << 16;  // shift to high bits
            float converted_norm_val = converter.f;
            DPRINT << "norm val: " << converted_norm_val << ENDL();
            // read in dist value
            volatile tt_l1_ptr uint16_t* dist_val_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(cb_out_l));
            converter.i = static_cast<uint32_t>(dist_val_ptr[0]) << 16;  // shift to high bits
            float converted_dist_val = converter.f;
            DPRINT << "dist norm val: " << converted_dist_val << ENDL();
            // get scale value in float
            converter.i = lambda_val;
            float lambda_val_float = converter.f;
            DPRINT << "lambda val: " << lambda_val_float << ENDL();

            // verify speculation result
            // to save computation, dist_val is squared and norm_val is squared
            // so we want to check wether sqrt(dist_val) <= lambda * sqrt(norm_val)
            // we can square both sides to get dist_val <= lambda^2 * norm_val
            bool is_speculation_correct =
                converted_dist_val <= (lambda_val_float * lambda_val_float) * converted_norm_val;
            uint32_t priority_val = is_speculation_correct ? 2 : 0;
            DPRINT << "priority val: " << priority_val << ENDL();
            // write priority to a local cb, in this case, cb_out_l
            volatile tt_l1_ptr uint32_t* priority_val_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_out_l));
            priority_val_ptr[0] = priority_val;

            // write priority to memory
            constexpr uint32_t priority_scalar_bytes = 4;  // 4 bytes for uint32_t
            const InterleavedAddrGen<is_dram> s_out_priority = {
                .bank_base_address = priority_addr, .page_size = priority_scalar_bytes};
            uint64_t priority_noc_addr = get_noc_addr(cur_batch, s_out_priority);
            noc_async_write(get_read_ptr(cb_out_l), priority_noc_addr, priority_scalar_bytes);
            noc_async_write_barrier();
        }
    }
    DPRINT << "done reducer_compute" << ENDL();
}
