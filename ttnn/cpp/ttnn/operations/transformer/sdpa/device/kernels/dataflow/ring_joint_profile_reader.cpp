// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Profile reader for ring_joint_sdpa.
 *
 * Key simplifications from ring_joint_reader.cpp:
 * - No fused_op_receiver synchronization (KV is pre-staged)
 * - ring_index is a compile-time arg
 * - Always reads from gathered_k/v buffer (pre-staged in arrival order)
 * - No store-and-forward chain between cores
 *
 * The gathered KV buffer contains all KV from all devices in arrival order:
 * [local_kv | arrival_1_kv | arrival_2_kv | ...]
 * So ring_iter maps directly to buffer offset: ring_iter * local_padded_Nt
 */

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "dataflow_common.hpp"
#include "tools/profiler/kernel_profiler.hpp"

// Skip actual DRAM reads to isolate compute timing
// #define SKIP_DRAM_ACCESS 1

/**
 * Profile indexer for ring_joint_sdpa.
 * Computes ring_id arrival order without synchronization.
 * Matches the logic in fused_op_indexer.hpp but without semaphore waits.
 */
struct ProfileRingIndexer {
    uint32_t ring_size = 0;
    uint32_t ring_index = 0;

    uint32_t received_inputs[2] = {0, 0};
    uint32_t expected_inputs[2] = {0, 0};
    uint32_t curr_dir = 0;
    uint32_t curr_transfer_idx = 0;

    ProfileRingIndexer(uint32_t ring_size_, uint32_t ring_index_) : ring_size(ring_size_), ring_index(ring_index_) {
        // For Linear topology:
        // from_forward = ring_size - 1 - ring_index
        // from_backward = ring_index
        expected_inputs[0] = ring_index_;                   // backward
        expected_inputs[1] = ring_size_ - 1 - ring_index_;  // forward
        curr_dir = 0;
    }

    uint32_t get_next_ring_id() {
        uint32_t sender_ring_id;
        if (curr_transfer_idx == 0) {
            // First iteration: local slice
            sender_ring_id = ring_index;
        } else {
            received_inputs[curr_dir] += 1;
            if (curr_dir == 1) {
                // Receiving from forward direction, go backwards
                sender_ring_id = (ring_index - received_inputs[curr_dir] + ring_size) % ring_size;
            } else {
                // Receiving from backward direction, go forward
                sender_ring_id = (ring_index + received_inputs[curr_dir]) % ring_size;
            }
        }

        // Determine next direction
        if (curr_transfer_idx == 0) {
            if (expected_inputs[curr_dir] == 0) {
                curr_dir = 1 - curr_dir;
            }
        } else {
            uint32_t next_dir = 1 - curr_dir;
            if (received_inputs[next_dir] < expected_inputs[next_dir]) {
                curr_dir = next_dir;
            }
        }

        curr_transfer_idx++;
        return sender_ring_id;
    }
};

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
    constexpr uint32_t ring_index = get_compile_time_arg_val(20);
    constexpr uint32_t is_causal = get_compile_time_arg_val(21);
    constexpr uint32_t is_balanced = get_compile_time_arg_val(22);

    // Compile-time flag for whether joint tensors are provided
    constexpr bool use_joint_tensors = L > 0;

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

    // Profile indexer: computes ring_id without synchronization
    ProfileRingIndexer indexer(ring_size, ring_index);

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;

    constexpr uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);
    constexpr uint32_t v_tile_bytes = get_tile_size(cb_v_in);

    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint32_t v_chunk_tiles = Sk_chunk_t * vDHt;
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

    /**
     * Iterate over ring indices.
     * Profile version: no synchronization, KV pre-staged in arrival order.
     */
    uint32_t half_sequence = num_q_chunks / 2;
    for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
        // Get ring_id using the indexer (matches fused_op_indexer logic)
        uint32_t ring_id = indexer.get_next_ring_id();

        // Only the last ring ID will append joint_K, joint_V to K, V.
        const bool do_joint_kv = ring_id == ring_size - 1;
        const uint32_t num_kv_chunks = do_joint_kv ? num_local_k_chunks + num_joint_k_chunks : num_local_k_chunks;

        const uint32_t global_n_tile_id = logical_n / tt::constants::TILE_HEIGHT;
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

        // In causal balanced case processing KV received from other devices
        if (is_causal && is_balanced && ring_index > ring_id) {
            iter_num_kv_chunks /= 2;
        }

        for (uint32_t q_iter = 0; q_iter < (global_q_end - global_q_start); ++q_iter) {
            // Linear flat index for this iteration
            uint32_t linear_flat = global_q_start + q_iter;

#if defined BALANCED_Q_PARALLEL
            // Apply per-head zigzag for load balancing
            uint32_t global_q_chunk = linear_to_zigzag(linear_flat, num_q_chunks);
#else
            uint32_t global_q_chunk = linear_flat;
#endif

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
            uint32_t end_seq_tile;
            if (is_joint_q) {
                const uint32_t joint_q_row_start_tile = (q_chunk - num_local_q_chunks) * Sq_chunk_t;
                q_slice = Slice(nb, nq, joint_q_row_start_tile, joint_q_row_start_tile + Sq_chunk_t, 0, DHt);
                end_seq_tile = Lt;
            } else {
                q_slice = Slice(nb, nq, q_row_start_tile, q_row_start_tile + Sq_chunk_t, 0, DHt);
                end_seq_tile = local_padded_Nt;
            }

            // Read Q chunk
#if SKIP_DRAM_ACCESS
            {
                constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
                cb_reserve_back(cb_q_in, q_chunk_tiles);
                cb_push_back(cb_q_in, q_chunk_tiles);
            }
#else
            if constexpr (use_joint_tensors) {
                if (is_joint_q) {
                    read_block(joint_q_generator, q_slice, end_seq_tile, cb_q_in, q_tile_bytes, false /*transpose*/);
                } else {
                    read_block(q_generator, q_slice, end_seq_tile, cb_q_in, q_tile_bytes, false /*transpose*/);
                }
            } else {
                read_block(q_generator, q_slice, end_seq_tile, cb_q_in, q_tile_bytes, false /*transpose*/);
            }
#endif

            for (uint32_t k_chunk = 0; k_chunk < iter_num_kv_chunks; ++k_chunk) {
                const bool kv_chunk_is_joint = k_chunk >= num_local_k_chunks;
                const uint32_t kv_global_start_tile = local_padded_Nt * ring_id + k_chunk * Sk_chunk_t;
                const bool kv_chunk_is_beyond_logical_n = !kv_chunk_is_joint && (kv_global_start_tile >= logical_nt);

                if (kv_chunk_is_beyond_logical_n) {
                    continue;
                }
                KV_chunks_processed_in_iter++;

                Slice k_slice;
                Slice v_slice;
                uint32_t kv_end_seq_tile;

                const uint32_t nk = nq / q_heads_per_k;
                if (kv_chunk_is_joint) {
                    const uint32_t joint_k_chunk = k_chunk - num_local_k_chunks;
                    const uint32_t joint_k_row_start_tile = joint_k_chunk * Sk_chunk_t;

                    k_slice = Slice(nb, nk, joint_k_row_start_tile, joint_k_row_start_tile + Sk_chunk_t, 0, DHt);
                    v_slice = Slice(nb, nq, joint_k_row_start_tile, joint_k_row_start_tile + Sk_chunk_t, 0, vDHt);
                    kv_end_seq_tile = Lt;
                } else {
                    // Profile: Always read from gathered buffer (pre-staged in arrival order)
                    // The gathered buffer is arranged as: [ring_iter_0_kv | ring_iter_1_kv | ...]
                    // So we use ring_iter * local_padded_Nt as the base offset
                    const uint32_t ring_iter_kv_start_tile = ring_iter * local_padded_Nt;
                    const uint32_t gathered_kv_start_tile = ring_iter_kv_start_tile + k_chunk * Sk_chunk_t;
                    k_slice = Slice(nb, nk, gathered_kv_start_tile, gathered_kv_start_tile + Sk_chunk_t, 0, DHt);
                    v_slice = Slice(nb, nq, gathered_kv_start_tile, gathered_kv_start_tile + Sk_chunk_t, 0, vDHt);
                    // For end_seq_tile calculation, we need to consider logical_n within the arrival order
                    // This is a simplification for the profile kernel
                    kv_end_seq_tile = padded_Nt;  // Use full gathered buffer, compute kernel handles masking
                }

                // Read K (always from gathered buffer or joint buffer)
#if SKIP_DRAM_ACCESS
                cb_reserve_back(cb_k_in, k_chunk_tiles);
                cb_push_back(cb_k_in, k_chunk_tiles);
#else
                {
                    DeviceZoneScopedN("K-RESERVE");
                    cb_reserve_back(cb_k_in, k_chunk_tiles);
                }
                {
                    DeviceZoneScopedN("K-READ");
                    if constexpr (use_joint_tensors) {
                        if (kv_chunk_is_joint) {
                            read_block(
                                joint_k_generator, k_slice, kv_end_seq_tile, cb_k_in, k_tile_bytes, true /*transpose*/);
                        } else {
                            read_block(
                                gathered_k_generator,
                                k_slice,
                                kv_end_seq_tile,
                                cb_k_in,
                                k_tile_bytes,
                                true /*transpose*/);
                        }
                    } else {
                        read_block(
                            gathered_k_generator, k_slice, kv_end_seq_tile, cb_k_in, k_tile_bytes, true /*transpose*/);
                    }
                }
#endif

                // Read V (always from gathered buffer or joint buffer)
#if SKIP_DRAM_ACCESS
                cb_reserve_back(cb_v_in, v_chunk_tiles);
                cb_push_back(cb_v_in, v_chunk_tiles);
#else
                {
                    DeviceZoneScopedN("V-RESERVE");
                    cb_reserve_back(cb_v_in, v_chunk_tiles);
                }
                {
                    DeviceZoneScopedN("V-READ");
                    if constexpr (use_joint_tensors) {
                        if (kv_chunk_is_joint) {
                            read_block(
                                joint_v_generator,
                                v_slice,
                                kv_end_seq_tile,
                                cb_v_in,
                                v_tile_bytes,
                                false /*transpose*/);
                        } else {
                            read_block(
                                gathered_v_generator,
                                v_slice,
                                kv_end_seq_tile,
                                cb_v_in,
                                v_tile_bytes,
                                false /*transpose*/);
                        }
                    } else {
                        read_block(
                            gathered_v_generator, v_slice, kv_end_seq_tile, cb_v_in, v_tile_bytes, false /*transpose*/);
                    }
                }
#endif
            }
        }
        // Pad to ensure even number of chunks for double buffering
        if (KV_chunks_processed_in_iter % 2 == 0) {
            cb_reserve_back(cb_k_in, k_chunk_tiles);
            cb_reserve_back(cb_v_in, v_chunk_tiles);
            cb_push_back(cb_k_in, k_chunk_tiles);
            cb_push_back(cb_v_in, v_chunk_tiles);
        }
    }
}
