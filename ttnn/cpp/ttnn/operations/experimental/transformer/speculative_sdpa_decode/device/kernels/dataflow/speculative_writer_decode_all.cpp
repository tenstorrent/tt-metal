// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "debug/assert.h"

#include "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp"

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);     // batch size
    constexpr uint32_t PNHt = get_compile_time_arg_val(1);  // padded number of heads in tiles
    constexpr uint32_t St = get_compile_time_arg_val(2);    // full sequence length of kv cache in tiles
    constexpr uint32_t DHt = get_compile_time_arg_val(3);   // head dim
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(4);
    constexpr uint32_t scale_val = get_compile_time_arg_val(5);
    constexpr uint32_t num_cores_per_batch = get_compile_time_arg_val(6);          // num cores per batch
    constexpr uint32_t num_cores = get_compile_time_arg_val(7);                    // num running cores in total
    uint32_t reducer_semaphore_addr = get_semaphore(get_compile_time_arg_val(8));  // semaphore for reducer
    uint32_t output_semaphore_addr = get_semaphore(get_compile_time_arg_val(9));   // semaphore for sender
    constexpr bool is_out_sharded = get_compile_time_arg_val(10);
    constexpr uint32_t k_chunk_size = get_compile_time_arg_val(11);
    constexpr uint32_t num_q_heads = get_compile_time_arg_val(12);
    constexpr uint32_t num_kv_heads = get_compile_time_arg_val(13);
    constexpr uint32_t num_cores_per_head = get_compile_time_arg_val(14);
    constexpr uint32_t num_heads_per_core = get_compile_time_arg_val(15);
    constexpr uint32_t num_reducer_cores = get_compile_time_arg_val(16);
    constexpr uint32_t num_output_cores = get_compile_time_arg_val(17);
    constexpr uint32_t ELEMENT_SIZE = get_compile_time_arg_val(18);
    constexpr bool is_causal = get_compile_time_arg_val(19) == 1;

    uint32_t arg_idx = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(arg_idx++);
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
    auto [PSt, k_num_chunks, k_chunk_start, k_chunk_end] =
        get_runtime_args(cur_pos, cur_batch, core_num_in_reduce, num_cores_per_head, k_chunk_size);

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

    const uint64_t in0_sender_semaphore_noc_addr =
        get_noc_addr(reduce_core_noc_x, reduce_core_noc_y, reducer_semaphore_addr);

    if (k_chunk_start == k_chunk_end) {
        return;  // early exit because no computes needs to be done
    }

    constexpr uint32_t out_chunk_tiles = PNHt * DHt;
    uint32_t num_cores_to_wait = num_cores_per_head - 1;
    if (num_cores_per_head > k_num_chunks) {
        num_cores_to_wait = k_num_chunks - 1;
    }
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
    if (is_worker) {
        ASSERT(num_heads_per_core == 1);  // if there are workers, then head must be split across workers so there
                                          // should not be more than one head per core
        worker_compute<out_chunk_tiles, cb_out_worker, cb_out_m, cb_out_l, cb_intermed_out, PNHt>(
            in0_sender_semaphore_noc_addr, worker_id_for_reduce, reduce_core_noc_x, reduce_core_noc_y);
        return;
    }

    // *** Reducer Compute Below ***
    constexpr uint32_t tile_bytes = get_tile_size(cb_out);
    constexpr uint32_t tile_bytes_intermed = get_tile_size(cb_intermed_out);
    constexpr DataFormat data_format = get_dataformat(cb_out);

    const InterleavedAddrGenFast<is_dram> out_writer = {
        .bank_base_address = out_addr, .page_size = tile_bytes, .data_format = data_format};

    uint64_t intermed_l1_read_addr = get_noc_addr(get_read_ptr(cb_intermed_out));

    volatile tt_l1_ptr uint32_t* in0_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reducer_semaphore_addr);

    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<tile_bytes, num_cores>();
    uint32_t barrier_count = 0;

    // generate and send mask to compute if causal
    if constexpr (is_causal) {
        generate_mask<cb_mask_in, PNHt>(k_num_chunks, PSt, cur_pos);
    }

    for (uint32_t cur_head = cur_head_group * num_heads_per_core;
         cur_head < cur_head_group * num_heads_per_core + num_heads_per_core;
         ++cur_head) {
        if (k_chunk_end - k_chunk_start < k_num_chunks) {
            ASSERT(num_heads_per_core == 1);  // if there are workers, then head must be split across workers so there
                                              // should not be more than one head per core
            // This indicates that there are computes done by other workers. Needs to wait for them and send to
            // reducer's compute Wait for compute to deliver output chunk, and write to compute again for reduction data
            // in cb_intermed_out is arranged as [o,m,l,o,m,l,...] with size (out_chunk_tiles +
            // 2*PNHt)*num_cores_to_wait wait on in0 semaphore value to become VALID (set by sender)
            noc_semaphore_wait(in0_receiver_semaphore_addr_ptr, num_cores_to_wait);
            // noc_semaphore_set(in0_receiver_semaphore_addr_ptr, 0);

            // cb_wait_front(cb_intermed_out, num_tiles_to_wait);
            constexpr uint32_t q_read_size = out_chunk_tiles * tile_bytes_intermed;
            constexpr uint32_t ml_read_size = PNHt * tile_bytes_intermed;
            for (uint32_t block = 0; block < num_cores_to_wait + 1; ++block) {
                cb_reserve_back(cb_out_o, out_chunk_tiles);
                cb_reserve_back(cb_m_in, PNHt);
                cb_reserve_back(cb_l_in, PNHt);

                uint32_t q_write_ptr = get_read_ptr(cb_out_o);
                noc_async_read(intermed_l1_read_addr, q_write_ptr, q_read_size);
                intermed_l1_read_addr += q_read_size;
                noc_async_read_barrier();
                cb_push_back(cb_out_o, out_chunk_tiles);

                uint32_t m_write_ptr = get_read_ptr(cb_m_in);
                noc_async_read(intermed_l1_read_addr, m_write_ptr, ml_read_size);
                intermed_l1_read_addr += ml_read_size;
                noc_async_read_barrier();
                cb_push_back(cb_m_in, PNHt);

                uint32_t l_write_ptr = get_read_ptr(cb_l_in);
                noc_async_read(intermed_l1_read_addr, l_write_ptr, ml_read_size);
                intermed_l1_read_addr += ml_read_size;
                noc_async_read_barrier();
                cb_push_back(cb_l_in, PNHt);
            }
        }
        // Offset for current batch
        const uint32_t out_batch_offset = cur_batch * out_chunk_tiles;

        // Write entire out into its corresponding batch
        uint32_t out_tile_id = out_batch_offset;
        cb_wait_front(cb_out, out_chunk_tiles);

        if constexpr (num_kv_heads > 1) {
            // if gqa, we will need to write partial outputs for each head
            constexpr uint32_t TILE_WIDTH = 32;
            // we are assuming here that num_heads_to_write = nh/nkv is a power of 2 here, so that we don't write
            // partial across phase
            uint32_t num_heads_to_write = num_q_heads / num_kv_heads;  // each head is one row in a tile
            uint32_t SUBTILE_LINE_BYTES = 16 * ELEMENT_SIZE;           // size of 16 elements (in a row)

            if (!is_out_sharded) {
                for (uint32_t tile = 0; tile < out_chunk_tiles; ++tile) {
                    uint64_t out_writer_noc_addr = get_noc_addr(out_tile_id, out_writer);
                    uint32_t l1_read_addr = get_read_ptr(cb_out) + tile * tile_bytes;

                    // write partial output for each head
                    for (uint32_t head = 0; head < num_heads_to_write; ++head) {
                        uint32_t starting_row = cur_head * num_heads_to_write + head;
                        uint32_t in_tile_offset_by_starting_head =
                            starting_row < 16 ? starting_row * SUBTILE_LINE_BYTES
                                              : (starting_row - 16) * SUBTILE_LINE_BYTES + 512 * ELEMENT_SIZE;
                        uint64_t out_writer_noc_addr_head = out_writer_noc_addr + in_tile_offset_by_starting_head;
                        uint32_t l1_read_addr_head = l1_read_addr + in_tile_offset_by_starting_head;

                        // Write first phase
                        noc_async_write(l1_read_addr_head, out_writer_noc_addr_head, SUBTILE_LINE_BYTES);

                        // Write second phase
                        noc_async_write(
                            l1_read_addr_head + 256 * ELEMENT_SIZE,
                            out_writer_noc_addr_head + 256 * ELEMENT_SIZE,
                            SUBTILE_LINE_BYTES);

                        if (++barrier_count == barrier_threshold) {
                            noc_async_writes_flushed();
                            barrier_count = 0;
                        }
                    }

                    ++out_tile_id;
                }
            }
            // sharded out case
            else if (do_output) {
                // read from reducer cores
                constexpr uint32_t num_reducers_per_output = num_reducer_cores / num_output_cores;
                constexpr uint32_t num_reducers_to_wait = num_reducers_per_output - 1;
                volatile tt_l1_ptr uint32_t* output_self_semaphore_addr_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(output_semaphore_addr);
                noc_semaphore_wait(output_self_semaphore_addr_ptr, num_reducers_to_wait);

                uint32_t reduce_core_read_index_start = (cur_batch * num_cores_per_batch) / num_cores_per_head;

                for (uint32_t reduce_core_read_index = reduce_core_read_index_start + 1;
                     reduce_core_read_index < reduce_core_read_index_start + num_reducers_per_output;
                     reduce_core_read_index++) {
                    uint32_t reduce_core_read_noc_x = all_reducer_noc_x[reduce_core_read_index];
                    uint32_t reduce_core_read_noc_y = all_reducer_noc_y[reduce_core_read_index];

                    uint64_t out_reader_base_noc_addr =
                        get_noc_addr(reduce_core_read_noc_x, reduce_core_read_noc_y, get_read_ptr(cb_out));

                    for (uint32_t tile = 0; tile < out_chunk_tiles; ++tile) {
                        uint32_t l1_write_addr = get_write_ptr(cb_out) + tile * tile_bytes;
                        uint32_t out_reader_noc_addr = out_reader_base_noc_addr;
                        // write partial output for each head
                        for (uint32_t head = 0; head < num_heads_to_write; ++head) {
                            uint32_t starting_row = cur_head * num_heads_to_write + head;
                            uint32_t in_tile_offset_by_starting_head =
                                starting_row < 16 ? starting_row * SUBTILE_LINE_BYTES
                                                  : (starting_row - 16) * SUBTILE_LINE_BYTES + 512 * ELEMENT_SIZE;
                            uint32_t out_reader_noc_addr_head = out_reader_noc_addr + in_tile_offset_by_starting_head;
                            uint32_t l1_write_addr_head = l1_write_addr + in_tile_offset_by_starting_head;

                            // Write first phase
                            noc_async_read(out_reader_noc_addr_head, l1_write_addr_head, SUBTILE_LINE_BYTES);

                            // Write second phase
                            noc_async_read(
                                out_reader_noc_addr_head + 256 * ELEMENT_SIZE,
                                l1_write_addr_head + 256 * ELEMENT_SIZE,
                                SUBTILE_LINE_BYTES);

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
                const uint64_t output_core_semaphore_noc_addr =
                    get_noc_addr(output_core_noc_x, output_core_noc_y, output_semaphore_addr);
                noc_semaphore_inc(output_core_semaphore_noc_addr, 1);
            }
        } else {
            // if mqa, we don't need to gather outputs for other heads so we can just write entire tiles to memory
            if (!is_out_sharded) {
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
