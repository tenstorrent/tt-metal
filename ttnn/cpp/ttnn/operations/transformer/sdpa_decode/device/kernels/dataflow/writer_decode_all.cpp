// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "api/debug/assert.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp"
#include "dataflow_common.hpp"

#define MAX_TREE_REDUCTION_ROUNDS 6

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);     // batch size
    constexpr uint32_t PNHt = get_compile_time_arg_val(1);  // padded number of heads in tiles
    constexpr uint32_t St = get_compile_time_arg_val(2);    // full sequence length of kv cache in tiles
    constexpr uint32_t DHt = get_compile_time_arg_val(3);   // head dim
    constexpr uint32_t vDHt = get_compile_time_arg_val(4);  // head dim for V
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(5);  // number of tiles in seqlen of a k/v/mask chunk
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(6);
    constexpr uint32_t zero_scalar_packed = get_compile_time_arg_val(7);
    constexpr uint32_t scale_val = get_compile_time_arg_val(8);
    constexpr uint32_t num_cores_per_batch = get_compile_time_arg_val(9);           // num cores per batch
    constexpr uint32_t num_cores = get_compile_time_arg_val(10);                    // num running cores in total
    uint32_t reducer_semaphore_addr = get_semaphore(get_compile_time_arg_val(11));  // semaphore for reducer
    uint32_t output_semaphore_addr = get_semaphore(get_compile_time_arg_val(12));   // semaphore for sender
    constexpr bool is_out_sharded = get_compile_time_arg_val(13);
    constexpr uint32_t k_chunk_size = get_compile_time_arg_val(14);
    constexpr uint32_t num_q_heads = get_compile_time_arg_val(15);
    constexpr uint32_t num_kv_heads = get_compile_time_arg_val(16);
    constexpr uint32_t num_cores_per_head = get_compile_time_arg_val(17);
    constexpr uint32_t num_heads_per_core = get_compile_time_arg_val(18);
    constexpr uint32_t num_reducer_cores = get_compile_time_arg_val(19);
    constexpr uint32_t num_output_cores = get_compile_time_arg_val(20);
    constexpr uint32_t ELEMENT_SIZE = get_compile_time_arg_val(21);
    constexpr bool is_causal = get_compile_time_arg_val(22) == 1;
    constexpr uint32_t max_dynamic_chunk_size = get_compile_time_arg_val(23);
    constexpr uint32_t q_heads_parallel_factor = get_compile_time_arg_val(24);
    constexpr uint32_t sliding_window_size = get_compile_time_arg_val(25);
    constexpr bool use_mla = get_compile_time_arg_val(26) == 1;
    constexpr uint32_t num_tree_reduction_rounds = get_compile_time_arg_val(27);
    constexpr auto out_args = TensorAccessorArgs<27>();

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

    // Tree reduction parameters
    const bool is_tree_root = get_arg_val<uint32_t>(arg_idx++) == 1;
    const uint32_t parent_core_in_group = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t send_at_round = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_children = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t my_active_rounds = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t reduction_group_base_idx = get_arg_val<uint32_t>(arg_idx++);

    // Semaphore encoding: each round uses a 4-bit field (nibble) in the semaphore value
    // Round 0: bits 0-3, Round 1: bits 4-7, Round 2: bits 8-11, etc.
    // step_semaphore_inc[r] = 1 << (r * 4) is the value to add to increment round r's counter
    constexpr uint32_t step_semaphore_inc[6] = {1, 16, 256, 4096, 65536, 1048576};
    // step_semaphore_shift[r] = r * 4 is the bit position to read round r's counter
    constexpr uint32_t step_semaphore_shift[6] = {0, 4, 8, 12, 16, 20};

    // Read children_per_round array

    uint32_t children_per_round[MAX_TREE_REDUCTION_ROUNDS];
    for (uint32_t r = 0; r < MAX_TREE_REDUCTION_ROUNDS; ++r) {
        children_per_round[r] = get_arg_val<uint32_t>(arg_idx++);
    }

    // Read reduction group physical core coordinates
    tt_l1_ptr uint32_t* reduction_group_core_xs = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores_per_head;
    tt_l1_ptr uint32_t* reduction_group_core_ys = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores_per_head;

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
            cur_pos = index_ptr[(uint32_t)(cur_batch / q_heads_parallel_factor)];
        }

        if (cur_pos == UINT32_MAX) {
            // cur_pos of -1 indicates that the user should be skipped
            return;
        }
    }

    auto Sk_chunk_t_dynamic = get_dynamic_Sk_chunk_t<Sk_chunk_t, max_dynamic_chunk_size>(cur_pos);
    auto k_chunk_size_dynamic = Sk_chunk_t_dynamic * tt::constants::TILE_HEIGHT;

    // Sequence length assignment
    auto [PSt, k_num_chunks, k_chunk_start, k_chunk_end, window_start_unaligned, window_start_chunk] =
        get_workload_for_core(
            cur_pos,
            cur_batch,
            core_num_in_reduce,
            num_cores_per_head,
            k_chunk_size_dynamic,
            sliding_window_size > 0 ? std::optional<uint32_t>(sliding_window_size) : std::nullopt);

    // Check if this core has local data to process
    bool has_local_data = (k_chunk_start != k_chunk_end);

    // Cores without data don't participate in tree reduction at all
    if (!has_local_data) {
        return;
    }

    // Determine which children actually participate in reduction (based on chunk allocation)
    // A child at core_num has data if core_num < k_num_chunks
    uint32_t num_active_children = 0;
    uint32_t active_children_per_round[MAX_TREE_REDUCTION_ROUNDS];
    uint32_t num_active_rounds = 0;

    for (uint32_t r = 0; r < MAX_TREE_REDUCTION_ROUNDS; ++r) {
        uint32_t child_id = children_per_round[r];
        if (child_id != UINT32_MAX && child_id < k_num_chunks) {
            // This child has data
            active_children_per_round[r] = child_id;
            num_active_children++;
            num_active_rounds = r + 1;
        } else {
            active_children_per_round[r] = UINT32_MAX;
        }
    }

    // We send if we have a parent (and we have data, which we do if we reach here)
    const bool should_send_to_parent = (parent_core_in_group != UINT32_MAX);

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

    constexpr uint32_t out_chunk_tiles = PNHt * vDHt;
    uint32_t num_cores_to_wait = num_cores_per_head - 1;
    if (num_cores_per_head > k_num_chunks) {
        num_cores_to_wait = k_num_chunks - 1;
    }
    uint32_t num_tiles_to_wait = (out_chunk_tiles + 2 * PNHt) * num_cores_to_wait;

    constexpr uint32_t cb_out = tt::CBIndex::c_20;
    constexpr uint32_t cb_intermed_out =
        tt::CBIndex::c_19;  // this cb holds the output intermediates from other worker cores
    constexpr uint32_t cb_out_o = tt::CBIndex::c_16;
    constexpr uint32_t cb_m_in = tt::CBIndex::c_6;
    constexpr uint32_t cb_l_in = tt::CBIndex::c_7;

    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_sliding_window_mask_in = tt::CBIndex::c_13;  // Separate buffer for sliding window mask
    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_11;
    constexpr uint32_t cb_zero_in = tt::CBIndex::c_12;

    constexpr uint32_t cb_out_worker = tt::CBIndex::c_16;
    constexpr uint32_t cb_out_m = tt::CBIndex::c_17;
    constexpr uint32_t cb_out_l = tt::CBIndex::c_18;

    // generate and send scaler to compute
    // These helper functions respect tile size of CBs (ie. no need for special handling of tiny tiles)
    generate_reduce_scaler(cb_identity_scale_in, identity_scalar_packed);
    generate_reduce_scaler(cb_zero_in, zero_scalar_packed);
    generate_bcast_col_scalar(cb_col_identity, identity_scalar_packed);

    // Generate sliding window mask only if we have local data and need it
    if (has_local_data && k_chunk_start == window_start_chunk && window_start_unaligned > 0) {
        // If this core processes the first chunk and we need to apply sliding window mask, generate it here
        generate_sliding_window_mask<cb_sliding_window_mask_in, PNHt>(
            k_num_chunks, Sk_chunk_t_dynamic, window_start_unaligned);
    }

    // generate and send mask to compute if causal (only if we have local data to process)
    if constexpr (is_causal) {
        // These helper functions respect tile size of CBs (ie. no need for special handling of tiny tiles)
        generate_mask<cb_mask_in, PNHt>(k_num_chunks, Sk_chunk_t_dynamic, cur_pos);
    }

    // *** Tree Reduction Logic ***
    // Each core participates in tree reduction by:
    // 1. Computing its local attention (done in compute kernel)
    // 2. For each round it receives: wait for child, receive data, combine with local (done in compute kernel)
    // 3. If not root: send combined result to parent
    // 4. If root: write final output

    constexpr uint32_t tile_bytes = get_tile_size(cb_out);
    constexpr uint32_t tile_bytes_intermed = get_tile_size(cb_intermed_out);

    const auto out_writer = TensorAccessor(out_args, out_addr, tile_bytes);

    volatile tt_l1_ptr uint32_t* in0_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reducer_semaphore_addr);

    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<tile_bytes, num_cores>();
    uint32_t barrier_count = 0;

    noc_async_write_barrier();  // #19201 BH hang workaround

    for (uint32_t cur_head = cur_head_group * num_heads_per_core;
         cur_head < cur_head_group * num_heads_per_core + num_heads_per_core;
         ++cur_head) {
        // Tree reduction: receive from children at each round
        // Each round, we wait for one child (if any), read remote_sum, remote_max, remote_output, and push to CBs
        // The compute kernel processes each child's data before we move to the next round
        // Only receive from children that actually have data
        if (num_active_children > 0) {
            // If there are workers, then head must be split across workers
            ASSERT(num_heads_per_core == 1);

            for (uint32_t round = 0; round < num_active_rounds; ++round) {
                uint32_t child_id = active_children_per_round[round];

                if (child_id != UINT32_MAX) {
                    // Wait for this specific child to send its results
                    // Poll until round-specific nibble is >= 1
                    // Each round uses a 4-bit field: round 0 = bits 0-3, round 1 = bits 4-7, etc.
                    while (true) {
                        invalidate_l1_cache();
                        uint32_t sem_val = *in0_receiver_semaphore_addr_ptr;
                        uint8_t step_sem = (sem_val >> step_semaphore_shift[round]) & 0x0F;
                        if (step_sem >= 1) {
                            break;
                        }
                    }

                    // Now read the data from the intermediate buffer
                    constexpr uint32_t tile_bytes_intermed = get_tile_size(cb_intermed_out);
                    constexpr uint32_t o_read_size = out_chunk_tiles * tile_bytes_intermed;
                    constexpr uint32_t ml_read_size = PNHt * tile_bytes_intermed;

                    // Calculate offset based on round (child writes at round offset)
                    uint32_t block_offset = round * (out_chunk_tiles + 2 * PNHt) * tile_bytes_intermed;
                    uint64_t intermed_l1_read_addr = get_noc_addr(get_read_ptr(cb_intermed_out)) + block_offset;

                    // Reserve space in CBs and read data
                    // Order: l, m, o (same as sender writes)
                    cb_reserve_back(cb_l_in, PNHt);
                    uint32_t l_write_ptr = get_write_ptr(cb_l_in);
                    noc_async_read(intermed_l1_read_addr, l_write_ptr, ml_read_size);
                    intermed_l1_read_addr += ml_read_size;
                    noc_async_read_barrier();
                    cb_push_back(cb_l_in, PNHt);

                    cb_reserve_back(cb_m_in, PNHt);
                    uint32_t m_write_ptr = get_write_ptr(cb_m_in);
                    noc_async_read(intermed_l1_read_addr, m_write_ptr, ml_read_size);
                    intermed_l1_read_addr += ml_read_size;
                    noc_async_read_barrier();
                    cb_push_back(cb_m_in, PNHt);

                    cb_reserve_back(cb_out_o, out_chunk_tiles);
                    uint32_t o_write_ptr = get_write_ptr(cb_out_o);
                    noc_async_read(intermed_l1_read_addr, o_write_ptr, o_read_size);
                    noc_async_read_barrier();
                    cb_push_back(cb_out_o, out_chunk_tiles);
                }
            }
        }

        // SENDER: send intermediates to parent (only need to do this ONCE, once you send you are done)
        // We have data (checked at function start), so send it
        if (!is_tree_root && should_send_to_parent) {
            // Wait for compute to finish writing to cb_out_worker, cb_out_m, cb_out_l
            cb_wait_front(cb_out_worker, out_chunk_tiles);
            cb_wait_front(cb_out_m, PNHt);
            cb_wait_front(cb_out_l, PNHt);

            constexpr uint32_t tile_bytes = get_tile_size(cb_out_worker);
            uint32_t block_offset = send_at_round * (out_chunk_tiles + 2 * PNHt) * tile_bytes;
            constexpr uint32_t o_write_size = out_chunk_tiles * tile_bytes;
            constexpr uint32_t ml_write_size = PNHt * tile_bytes;

            // Get parent's NOC address
            uint32_t parent_noc_x = reduction_group_core_xs[parent_core_in_group];
            uint32_t parent_noc_y = reduction_group_core_ys[parent_core_in_group];
            uint64_t output_write_addr =
                get_noc_addr(parent_noc_x, parent_noc_y, get_write_ptr(cb_intermed_out)) + block_offset;

            // Send l, m, o to parent (same order as original worker_compute)
            noc_async_write(get_read_ptr(cb_out_l), output_write_addr, ml_write_size);
            output_write_addr += ml_write_size;
            noc_async_write(get_read_ptr(cb_out_m), output_write_addr, ml_write_size);
            output_write_addr += ml_write_size;
            noc_async_write(get_read_ptr(cb_out_worker), output_write_addr, o_write_size);
            noc_async_write_barrier();
            uint64_t parent_semaphore_noc_addr = get_noc_addr(parent_noc_x, parent_noc_y, reducer_semaphore_addr);
            noc_semaphore_inc(parent_semaphore_noc_addr, step_semaphore_inc[send_at_round]);

            // pop front
            cb_pop_front(cb_out_worker, out_chunk_tiles);
            cb_pop_front(cb_out_m, PNHt);
            cb_pop_front(cb_out_l, PNHt);
            noc_async_atomic_barrier();
            // Senders can return, dont need to participate
            return;
        }

        if (!is_tree_root) {
            return;
        }

        // ROOT CORE REMAINING WRITER WORK
        // Offset for current batch
        uint32_t out_tile_id = cur_batch * out_chunk_tiles;
        if constexpr (num_kv_heads > 1 || !is_out_sharded) {
            cb_wait_front(cb_out, out_chunk_tiles);
        }
        noc_async_writes_flushed();

        if constexpr (num_kv_heads > 1) {
            // if gqa, we will need to write partial outputs for each head
            // we are assuming here that num_heads_to_write = nh/nkv is a power of 2 here, so that we don't write
            // partial across phase
            constexpr uint32_t num_heads_to_write = num_q_heads / num_kv_heads;  // each head is one row in a tile
            if (!is_out_sharded) {
                barrier_count = write_partial_tiles_to_memory<cb_out, ELEMENT_SIZE, barrier_threshold, PNHt>(
                    out_tile_id, out_writer, barrier_count, cur_head, num_heads_to_write, out_chunk_tiles);
            }
            // sharded out case
            else if (do_output) {
                constexpr uint32_t SUBTILE_LINE_BYTES = 16 * ELEMENT_SIZE;
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
            // MQA (Multi Query Attention):  we don't need to gather outputs for other heads so we can just write entire
            // tiles to memory
            if (!is_out_sharded) {
                barrier_count = write_tiles_to_memory<cb_out, out_chunk_tiles, barrier_threshold>(
                    out_tile_id, out_writer, barrier_count);
            }
        }
        if constexpr (num_kv_heads > 1 || !is_out_sharded) {
            noc_async_write_barrier();
            cb_pop_front(cb_out, out_chunk_tiles);
        }
    }
}
