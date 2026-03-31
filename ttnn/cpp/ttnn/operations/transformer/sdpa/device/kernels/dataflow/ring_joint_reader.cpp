// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "dataflow_common.hpp"
#include "fused_op_receiver.hpp"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NH = get_compile_time_arg_val(1);
    constexpr uint32_t NHK = get_compile_time_arg_val(2);
    constexpr uint32_t DHt = get_compile_time_arg_val(3);
    constexpr uint32_t vDHt = get_compile_time_arg_val(4);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(5);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(6);
    constexpr uint32_t local_padded_N = get_compile_time_arg_val(7);
    constexpr uint32_t local_padded_Nt = get_compile_time_arg_val(8);
    constexpr uint32_t padded_Nt = get_compile_time_arg_val(9);
    constexpr uint32_t logical_n = get_compile_time_arg_val(10);
    constexpr uint32_t logical_nt = get_compile_time_arg_val(11);
    constexpr uint32_t Lt = get_compile_time_arg_val(12);
    constexpr uint32_t L = get_compile_time_arg_val(13);
    constexpr uint32_t num_local_q_chunks = get_compile_time_arg_val(14);
    constexpr uint32_t num_joint_q_chunks = get_compile_time_arg_val(15);
    constexpr uint32_t num_local_k_chunks = get_compile_time_arg_val(16);
    constexpr uint32_t num_joint_k_chunks = get_compile_time_arg_val(17);
    constexpr uint32_t num_q_chunks = get_compile_time_arg_val(18);
    constexpr uint32_t ring_size = get_compile_time_arg_val(19);
    constexpr uint32_t qk_subblock_h = get_compile_time_arg_val(20);
    constexpr uint32_t is_causal = get_compile_time_arg_val(21);
    constexpr uint32_t is_balanced = get_compile_time_arg_val(22);

    constexpr auto q_args = TensorAccessorArgs<23>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto gathered_k_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto gathered_v_args = TensorAccessorArgs<gathered_k_args.next_compile_time_args_offset()>();
    constexpr auto joint_q_args = TensorAccessorArgs<gathered_v_args.next_compile_time_args_offset()>();
    constexpr auto joint_k_args = TensorAccessorArgs<joint_q_args.next_compile_time_args_offset()>();
    constexpr auto joint_v_args = TensorAccessorArgs<joint_k_args.next_compile_time_args_offset()>();

    uint32_t argidx = 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t k_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t v_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t gathered_k_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t gathered_v_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t joint_q_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t joint_k_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t joint_v_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t global_q_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t global_q_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t q_per_core = global_q_end - global_q_start;

    const uint32_t is_chain_participant = get_arg_val<uint32_t>(argidx++);
    const uint32_t is_injector = get_arg_val<uint32_t>(argidx++);
    const uint32_t is_sink = get_arg_val<uint32_t>(argidx++);
    const uint32_t chain_batch = get_arg_val<uint32_t>(argidx++);
    const uint32_t chain_head = get_arg_val<uint32_t>(argidx++);
    const uint32_t chain_q_chunk_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t chain_q_chunk_count = get_arg_val<uint32_t>(argidx++);
    const uint32_t prev_physical_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t prev_physical_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t next_physical_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t next_physical_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t next_core_q_chunks = get_arg_val<uint32_t>(argidx++);
    const uint32_t mcast_num_dests = get_arg_val<uint32_t>(argidx++);
    const uint32_t mcast_sender_wait = get_arg_val<uint32_t>(argidx++);

    RingSDPAOpReceiver fused_op_receiver = RingSDPAOpReceiver(
        true, /* wait_for_op_signal */
        argidx);

    // After fused-op receiver consumed its runtime args, remaining RT args are S&F chain metadata

    // Compile-time semaphore ids and mcast flag are appended after all TensorAccessorArgs()
    uint32_t sender_semaphore_addr =
        get_semaphore(get_compile_time_arg_val(joint_v_args.next_compile_time_args_offset()));
    uint32_t receiver_semaphore_addr =
        get_semaphore(get_compile_time_arg_val(joint_v_args.next_compile_time_args_offset() + 1));
    uint32_t valid_semaphore_addr =
        get_semaphore(get_compile_time_arg_val(joint_v_args.next_compile_time_args_offset() + 2));
    constexpr bool mcast_enabled = get_compile_time_arg_val(joint_v_args.next_compile_time_args_offset() + 3) == 1;

    // VALID sem used to write L1-L1 valid semaphore
    volatile tt_l1_ptr uint32_t* valid_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(valid_semaphore_addr);
    *(valid_semaphore_addr_ptr) = VALID;

    volatile tt_l1_ptr uint32_t* receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_semaphore_addr);

    volatile tt_l1_ptr uint32_t* sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_semaphore_addr);

    uint64_t receiver_semaphore_noc_addr = 0;
    uint64_t mcast_base_noc_addr = 0;
    uint64_t mcast_sem_noc_addr = 0;
    uint32_t sender_wait_count = 1;

    const uint64_t sender_semaphore_noc_addr = get_noc_addr(prev_physical_x, prev_physical_y, sender_semaphore_addr);

    if constexpr (mcast_enabled) {
        if (is_injector) {
            // prev_physical = mcast_start (first receiver), next_physical = mcast_end (last receiver)
            mcast_base_noc_addr = get_noc_multicast_addr(
                prev_physical_x,
                prev_physical_y,
                next_physical_x,
                next_physical_y,
                0);  // addr=0; will OR in actual L1 addr at use site
            mcast_sem_noc_addr = mcast_base_noc_addr | receiver_semaphore_addr;
            sender_wait_count = mcast_sender_wait;
        }
    } else {
        receiver_semaphore_noc_addr = get_noc_addr(next_physical_x, next_physical_y, receiver_semaphore_addr);
    }

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;

    constexpr uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);
    constexpr uint32_t v_tile_bytes = get_tile_size(cb_v_in);

    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint32_t v_chunk_tiles = Sk_chunk_t * vDHt;
    constexpr uint32_t q_num_subblocks = Sq_chunk_t / qk_subblock_h;
    constexpr bool use_q_subblock_push = (q_num_subblocks > 1);
    constexpr uint32_t q_heads_per_k = NH / NHK;

    const auto q_reader = TensorAccessor(q_args, q_addr, q_tile_bytes);
    const auto local_k_reader = TensorAccessor(k_args, k_addr, k_tile_bytes);
    const auto local_v_reader = TensorAccessor(v_args, v_addr, v_tile_bytes);
    const auto gathered_k_reader = TensorAccessor(gathered_k_args, gathered_k_addr, k_tile_bytes);
    const auto gathered_v_reader = TensorAccessor(gathered_v_args, gathered_v_addr, v_tile_bytes);
    const auto joint_q_reader = TensorAccessor(joint_q_args, joint_q_addr, q_tile_bytes);
    const auto joint_k_reader = TensorAccessor(joint_k_args, joint_k_addr, k_tile_bytes);
    const auto joint_v_reader = TensorAccessor(joint_v_args, joint_v_addr, v_tile_bytes);

    const auto input_q_tile_logical = TensorTileShape(B, NH, local_padded_Nt, DHt);
    const auto input_k_tile_logical = TensorTileShape(B, NHK, local_padded_Nt, DHt);
    const auto input_v_tile_logical = TensorTileShape(B, NH, local_padded_Nt, vDHt);
    const auto gathered_k_input_tile_logical = TensorTileShape(B, NHK, padded_Nt, DHt);
    const auto gathered_v_input_tile_logical = TensorTileShape(B, NH, padded_Nt, vDHt);
    const auto joint_input_tile_logical = TensorTileShape(B, NH, Lt, DHt);

    const auto q_generator = PaddedAddrGenerator(q_reader, input_q_tile_logical);
    const auto local_k_generator = PaddedAddrGenerator(local_k_reader, input_k_tile_logical);
    const auto local_v_generator = PaddedAddrGenerator(local_v_reader, input_v_tile_logical);
    const auto gathered_k_generator = PaddedAddrGenerator(gathered_k_reader, gathered_k_input_tile_logical);
    const auto gathered_v_generator = PaddedAddrGenerator(gathered_v_reader, gathered_v_input_tile_logical);
    const auto joint_q_generator = PaddedAddrGenerator(joint_q_reader, joint_input_tile_logical);
    const auto joint_k_generator = PaddedAddrGenerator(joint_k_reader, joint_input_tile_logical);
    const auto joint_v_generator = PaddedAddrGenerator(joint_v_reader, joint_input_tile_logical);

    // Tracks whether Q has been pushed for q_per_core == 1 optimization.
    // When q_per_core == 1, Q is identical across ring iterations so we only push it once.
    bool q_pushed = false;

    /**
     * Iterate over ring indices.
     * On the first iteration, read from local K, V.
     * On subsequent iterations, read from gathered K, V. Sync with AllGather fused signaler.
     */
    uint32_t ring_index = fused_op_receiver.seq.ring_index;
    uint32_t half_sequence = num_q_chunks / 2;
    for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
        uint32_t ring_id;
        {
            DeviceZoneScopedN("READER-wait-ag-sync");
            // find out which is the latest ring_id that synchronized
            ring_id = fused_op_receiver.get_next_ring_id_and_sync();
        }
        // Iterate over KV blocks gathered on ring.
        // Only the last ring ID will append joint_K, joint_V to K, V.
        const bool do_joint_kv = ring_id == ring_size - 1;
        const uint32_t num_kv_chunks = do_joint_kv ? num_local_k_chunks + num_joint_k_chunks : num_local_k_chunks;

        const uint32_t global_n_tile_id = logical_n / tt::constants::TILE_HEIGHT;  // Floor division to get tile ID
        const uint32_t ring_iter_kv_start_tile = ring_id * local_padded_Nt;
        const bool ring_iter_processes_KV_chunks = ring_iter_kv_start_tile <= global_n_tile_id;

        // In causal non balanced case when processing KV received from other devices:
        // - skip over KV received from subsequent devices
        // - do non-causal attention on the KV from preceding devices
        const bool ring_iter_does_work = (ring_iter_processes_KV_chunks || (do_joint_kv && L != 0)) &&
                                         !(is_causal && ring_index < ring_id && !is_balanced);

        uint32_t KV_chunks_processed_in_iter = 0;
        if (!ring_iter_does_work) {
            continue;
        }

        uint32_t iter_num_kv_chunks = num_kv_chunks;

        // In causal balanced case processing KV received from other devices:
        //
        // We will have two logical chunks of the input sequence, logical indexes are:
        // ring_index and (seq_len / 2 * num_devices) - ring_index
        //
        // With this in mind we have two distinct cases when receiving from other device:
        // - 1st part of the sequence precedes both chunks on the sender device, 2nd part attends to both
        // - both chunks preced 2nd part of the sequence in received KV
        // Indexes are updated accordingly; compute is skipped
        if (is_causal && is_balanced && ring_index > ring_id) {
            iter_num_kv_chunks /= 2;
        }

        for (uint32_t global_q_chunk = global_q_start; global_q_chunk < global_q_end; ++global_q_chunk) {
            // global_q_chunk is index into `B * NH * num_q_chunks`. Need to get nb, nq, q_chunk from this.
            const uint32_t nb = global_q_chunk / (NH * num_q_chunks);
            const uint32_t nq = (global_q_chunk % (NH * num_q_chunks)) / num_q_chunks;
            const uint32_t q_chunk = global_q_chunk % num_q_chunks;
            const auto q_row_start_tile = q_chunk * Sq_chunk_t;
            const bool is_joint_q = q_chunk >= num_local_q_chunks;

            if (q_chunk < half_sequence && is_balanced && ring_index < ring_id) {
                continue;
            }

            Slice q_slice;
            uint32_t q_end_seq_tile;
            if (is_joint_q) {
                // Get row index into the joint Q tensor
                const uint32_t joint_q_row_start_tile = (q_chunk - num_local_q_chunks) * Sq_chunk_t;
                q_slice = Slice(nb, nq, joint_q_row_start_tile, joint_q_row_start_tile + Sq_chunk_t, 0, DHt);
                q_end_seq_tile = Lt;
            } else {
                // Index into the Q input tensor
                q_slice = Slice(nb, nq, q_row_start_tile, q_row_start_tile + Sq_chunk_t, 0, DHt);
                q_end_seq_tile = local_padded_Nt;
            }

            // Chain forwarding conditions are k_chunk-invariant — compute once before the KV loop
            const uint32_t q_iter_local = global_q_chunk - global_q_start;
            const bool should_forward = is_chain_participant && !is_sink && (nb == chain_batch && nq == chain_head) &&
                                        (q_iter_local < next_core_q_chunks);
            const bool should_receive = is_chain_participant && !is_injector && (nb == chain_batch && nq == chain_head);

            // When q_per_core == 1, Q is identical across ring iterations: compute keeps it
            // fronted in the CB, so we only need to read it once on the first active ring iteration.
            const bool need_q_read = (q_per_core > 1) || !q_pushed;

            for (uint32_t k_chunk = 0; k_chunk < iter_num_kv_chunks; ++k_chunk) {
                /**
                 * Iterate over all KV chunks for this Q chunk.
                 * If this is the last ring ID, we will also read from joint KV.
                 * If this k chunk is in the spatial input and beyond the logical N, we will skip it.
                 */
                const bool kv_chunk_is_joint = k_chunk >= num_local_k_chunks;
                // Global index into the padded KV tensor
                const uint32_t kv_global_start_tile = local_padded_Nt * ring_id + k_chunk * Sk_chunk_t;
                const bool kv_chunk_is_beyond_logical_n = !kv_chunk_is_joint && (kv_global_start_tile >= logical_nt);

                if (kv_chunk_is_beyond_logical_n) {
                    // This is a KV chunk on spatial input beyond the logical N, and not joint KV. Skip it.
                    continue;
                }
                KV_chunks_processed_in_iter++;

                Slice k_slice;
                Slice v_slice;
                uint32_t
                    end_seq_tile;  // further information to `read_block` to determine whether it should pad with zeros.

                const uint32_t nk = nq / q_heads_per_k;
                if (kv_chunk_is_joint) {
                    const uint32_t joint_k_chunk = k_chunk - num_local_k_chunks;
                    const uint32_t joint_k_row_start_tile = joint_k_chunk * Sk_chunk_t;

                    k_slice = Slice(nb, nk, joint_k_row_start_tile, joint_k_row_start_tile + Sk_chunk_t, 0, DHt);
                    v_slice = Slice(nb, nq, joint_k_row_start_tile, joint_k_row_start_tile + Sk_chunk_t, 0, vDHt);
                    end_seq_tile = Lt;
                } else {
                    if (ring_iter == 0) {
                        // Local KV
                        const uint32_t local_k_row_start_tile = k_chunk * Sk_chunk_t;

                        k_slice = Slice(nb, nk, local_k_row_start_tile, local_k_row_start_tile + Sk_chunk_t, 0, DHt);
                        v_slice = Slice(nb, nq, local_k_row_start_tile, local_k_row_start_tile + Sk_chunk_t, 0, vDHt);
                        end_seq_tile = std::min(logical_nt, local_padded_Nt);
                    } else {
                        // Gathered KV
                        const uint32_t gathered_kv_start_tile = ring_iter_kv_start_tile + k_chunk * Sk_chunk_t;
                        k_slice = Slice(nb, nk, gathered_kv_start_tile, gathered_kv_start_tile + Sk_chunk_t, 0, DHt);
                        v_slice = Slice(nb, nq, gathered_kv_start_tile, gathered_kv_start_tile + Sk_chunk_t, 0, vDHt);
                        end_seq_tile = std::min(logical_nt, local_padded_Nt * (ring_id + 1));
                    }
                }

                // K: either read locally (injector or not participant) or receive from previous core
                cb_reserve_back(cb_k_in, k_chunk_tiles);
                uint32_t cb_k_start_address = get_write_ptr(cb_k_in);
                if (should_receive) {
                    // Receive forwarded K chunk from previous core
                    noc_semaphore_set(receiver_semaphore_addr_ptr, INVALID);
                    noc_semaphore_inc(sender_semaphore_noc_addr, 1);
                    noc_semaphore_wait(receiver_semaphore_addr_ptr, VALID);
                    cb_push_back(cb_k_in, k_chunk_tiles);
                } else {
                    read_block(
                        kv_chunk_is_joint ? joint_k_generator
                                          : (ring_iter == 0 ? local_k_generator : gathered_k_generator),
                        k_slice,
                        end_seq_tile,
                        cb_k_in,
                        k_tile_bytes,
                        true /*transpose*/
                    );
                }

                // Forward K chunk to next core(s): initiate async write (NOC write channel)
                // For mcast: send linked data + companion semaphore back-to-back.
                if (should_forward) {
                    noc_semaphore_wait(sender_semaphore_addr_ptr, sender_wait_count);
                    noc_semaphore_set(sender_semaphore_addr_ptr, 0);
                    if constexpr (mcast_enabled) {
                        uint64_t k_mcast_addr = mcast_base_noc_addr | cb_k_start_address;
                        noc_async_write_multicast(
                            cb_k_start_address,
                            k_mcast_addr,
                            k_chunk_tiles * k_tile_bytes,
                            mcast_num_dests,
                            true /* linked: semaphore mcast follows */);
                        noc_semaphore_set_multicast(valid_semaphore_addr, mcast_sem_noc_addr, mcast_num_dests);
                    } else {
                        uint64_t k_unicast_data_addr =
                            get_noc_addr(next_physical_x, next_physical_y, cb_k_start_address);
                        noc_async_write(cb_k_start_address, k_unicast_data_addr, k_chunk_tiles * k_tile_bytes);
                        noc_async_writes_flushed();
                        noc_semaphore_set_remote(valid_semaphore_addr, receiver_semaphore_noc_addr);
                    }
                }

                // Download Q on the first K iteration — after K is downloaded and forwarded.
                // Push Q one subblock at a time so compute can start QK matmul incrementally.
                // Placed after K forward so no outstanding NOC writes remain
                // (noc_async_read_barrier inside subblock read would deadlock with in-flight writes).
                if (k_chunk == 0 && need_q_read) {
                    if constexpr (use_q_subblock_push) {
                        for (uint32_t q_sub = 0; q_sub < q_num_subblocks; ++q_sub) {
                            const uint32_t sb_row_start = q_slice.d2_start + q_sub * qk_subblock_h;
                            const uint32_t sb_row_end = sb_row_start + qk_subblock_h;
                            Slice q_sub_slice(q_slice.d0, q_slice.d1, sb_row_start, sb_row_end, 0, DHt);
                            read_block(
                                is_joint_q ? joint_q_generator : q_generator,
                                q_sub_slice,
                                q_end_seq_tile,
                                cb_q_in,
                                q_tile_bytes,
                                false /*transpose*/
                            );
                        }
                    } else {
                        read_block(
                            is_joint_q ? joint_q_generator : q_generator,
                            q_slice,
                            q_end_seq_tile,
                            cb_q_in,
                            q_tile_bytes,
                            false /*transpose*/
                        );
                    }
                    q_pushed = true;
                }

                // V: either read locally (injector or not participant) or receive from previous core
                cb_reserve_back(cb_v_in, v_chunk_tiles);
                uint32_t cb_v_start_address = get_write_ptr(cb_v_in);
                if (should_receive) {
                    // Receive forwarded V chunk from previous core
                    noc_semaphore_set(receiver_semaphore_addr_ptr, INVALID);
                    noc_semaphore_inc(sender_semaphore_noc_addr, 1);
                    noc_semaphore_wait(receiver_semaphore_addr_ptr, VALID);
                    cb_push_back(cb_v_in, v_chunk_tiles);
                } else {
                    read_block(
                        kv_chunk_is_joint ? joint_v_generator
                                          : (ring_iter == 0 ? local_v_generator : gathered_v_generator),
                        v_slice,
                        end_seq_tile,
                        cb_v_in,
                        v_tile_bytes,
                        false /*transpose*/
                    );
                }

                // Forward V chunk to next core(s) if applicable
                if (should_forward) {
                    noc_semaphore_wait(sender_semaphore_addr_ptr, sender_wait_count);
                    noc_semaphore_set(sender_semaphore_addr_ptr, 0);
                    if constexpr (mcast_enabled) {
                        uint64_t v_mcast_addr = mcast_base_noc_addr | cb_v_start_address;
                        noc_async_write_multicast(
                            cb_v_start_address,
                            v_mcast_addr,
                            v_chunk_tiles * v_tile_bytes,
                            mcast_num_dests,
                            true /* linked: semaphore mcast follows */);
                        noc_semaphore_set_multicast(valid_semaphore_addr, mcast_sem_noc_addr, mcast_num_dests);
                    } else {
                        uint64_t v_unicast_data_addr =
                            get_noc_addr(next_physical_x, next_physical_y, cb_v_start_address);
                        noc_async_write(cb_v_start_address, v_unicast_data_addr, v_chunk_tiles * v_tile_bytes);
                        noc_async_writes_flushed();
                        noc_semaphore_set_remote(valid_semaphore_addr, receiver_semaphore_noc_addr);
                    }
                }
            }
        }
        if (KV_chunks_processed_in_iter % 2 == 0) {
            cb_reserve_back(cb_k_in, k_chunk_tiles);
            cb_reserve_back(cb_v_in, v_chunk_tiles);
            cb_push_back(cb_k_in, k_chunk_tiles);
            cb_push_back(cb_v_in, v_chunk_tiles);
        }
    }
}
