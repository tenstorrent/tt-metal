// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "dataflow_common.hpp"
#include "chain_link.hpp"
#include "fused_op_receiver.hpp"

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NH = get_compile_time_arg_val(1);
    constexpr uint32_t NHK = get_compile_time_arg_val(2);
    // K chain selection: batch chain when NHK == 1 (MLA mode), else head chain
    // Derived from NHK to enable conditional arg layout (saves resources when unused)
    constexpr bool k_uses_batch_chain = (NHK == 1);
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
    constexpr bool use_zigzag_balancing = get_compile_time_arg_val(23) == 1;
    constexpr uint32_t num_q_readers = get_compile_time_arg_val(25);

    constexpr auto q_args = TensorAccessorArgs<26>();
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

    // Head chain runtime args (always present)
    const ChainConfig head_cfg = ChainConfig::read_from_args(argidx);

    // Batch chain runtime args (only present when k_uses_batch_chain / NHK == 1)
    ChainConfig batch_cfg;  // default zero-initialized
    uint32_t max_q_per_core = 0;
    if constexpr (k_uses_batch_chain) {
        batch_cfg = ChainConfig::read_from_args(argidx);
        max_q_per_core = get_arg_val<uint32_t>(argidx++);
    }

    RingSDPAOpReceiver fused_op_receiver = RingSDPAOpReceiver(
        true, /* wait_for_op_signal */
        argidx);

    // After fused-op receiver consumed its runtime args, remaining RT args are S&F chain metadata

    // Compile-time semaphore ids and chain flags are appended after all TensorAccessorArgs()
    // Head chain semaphores (head-level chain, always built)
    uint32_t head_sender_semaphore_addr =
        get_semaphore(get_compile_time_arg_val(joint_v_args.next_compile_time_args_offset()));
    uint32_t head_receiver_semaphore_addr =
        get_semaphore(get_compile_time_arg_val(joint_v_args.next_compile_time_args_offset() + 1));
    uint32_t head_valid_semaphore_addr =
        get_semaphore(get_compile_time_arg_val(joint_v_args.next_compile_time_args_offset() + 2));
    constexpr bool head_mcast_enabled = get_compile_time_arg_val(joint_v_args.next_compile_time_args_offset() + 3) == 1;

    // Batch chain semaphores (only present when k_uses_batch_chain / NHK == 1)
    // Initialize to 0; will be overwritten if k_uses_batch_chain
    uint32_t batch_sender_semaphore_addr = 0;
    uint32_t batch_receiver_semaphore_addr = 0;
    uint32_t batch_valid_semaphore_addr = 0;

    // batch_mcast_enabled: read from compile-time args if present, else false (for template instantiation)
    constexpr bool batch_mcast_enabled = []() {
        if constexpr (k_uses_batch_chain) {
            return get_compile_time_arg_val(joint_v_args.next_compile_time_args_offset() + 7) == 1;
        }
        return false;
    }();

    if constexpr (k_uses_batch_chain) {
        batch_sender_semaphore_addr =
            get_semaphore(get_compile_time_arg_val(joint_v_args.next_compile_time_args_offset() + 4));
        batch_receiver_semaphore_addr =
            get_semaphore(get_compile_time_arg_val(joint_v_args.next_compile_time_args_offset() + 5));
        batch_valid_semaphore_addr =
            get_semaphore(get_compile_time_arg_val(joint_v_args.next_compile_time_args_offset() + 6));
    }

    // TODO: CB indices below are hardcoded and duplicated from the program factory.
    // They should be passed as compile-time args so the factory is the single source of truth.
    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;

    constexpr uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);
    constexpr uint32_t v_tile_bytes = get_tile_size(cb_v_in);

    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint32_t v_chunk_tiles = Sk_chunk_t * vDHt;

    // Head chain (head-level): matches (batch, head), used by V and optionally K
    ChainLink<head_mcast_enabled, true> head_chain(
        head_cfg.participates,
        head_cfg.is_injector,
        head_cfg.is_sink,
        head_sender_semaphore_addr,
        head_receiver_semaphore_addr,
        head_valid_semaphore_addr,
        head_cfg.signal_target_x<head_mcast_enabled>(),
        head_cfg.signal_target_y<head_mcast_enabled>(),
        head_cfg.next_physical_x,
        head_cfg.next_physical_y,
        head_cfg.mcast_start_x,
        head_cfg.mcast_start_y,
        head_cfg.mcast_end_x,
        head_cfg.mcast_end_y,
        head_cfg.mcast_num_dests,
        head_cfg.mcast_sender_wait,
        v_chunk_tiles,
        v_tile_bytes,
        head_cfg.batch,
        head_cfg.head,
        head_cfg.next_core_q_chunks);

    // Batch chain (batch-level): matches batch only, used by K when NHK == 1 (MLA mode)
    ChainLink<batch_mcast_enabled, false> batch_chain(
        batch_cfg.participates,
        batch_cfg.is_injector,
        batch_cfg.is_sink,
        batch_sender_semaphore_addr,
        batch_receiver_semaphore_addr,
        batch_valid_semaphore_addr,
        batch_cfg.signal_target_x<batch_mcast_enabled>(),
        batch_cfg.signal_target_y<batch_mcast_enabled>(),
        batch_cfg.next_physical_x,
        batch_cfg.next_physical_y,
        batch_cfg.mcast_start_x,
        batch_cfg.mcast_start_y,
        batch_cfg.mcast_end_x,
        batch_cfg.mcast_end_y,
        batch_cfg.mcast_num_dests,
        batch_cfg.mcast_sender_wait,
        k_chunk_tiles,
        k_tile_bytes,
        batch_cfg.batch,
        0,  // chain_head unused for batch-level chain
        batch_cfg.next_core_q_chunks);

    // V always uses head chain
    auto& v_chain = head_chain;

    // K uses batch chain when NHK == 1 (MLA), else head chain (compile-time IIFE selection)
    auto& k_chain = [&]() -> auto& {
        if constexpr (k_uses_batch_chain) {
            return batch_chain;
        } else {
            return head_chain;
        }
    }();

    constexpr uint32_t q_num_subblocks = Sq_chunk_t / qk_subblock_h;
    constexpr bool use_q_subblock_push = (q_num_subblocks > 1);
    constexpr uint32_t q_heads_per_k = NH / NHK;

    // Throttle Q DRAM reads so many readers don't saturate the NoC outstanding-read budget.
    constexpr uint32_t q_barrier_threshold = get_barrier_read_threshold<q_tile_bytes, num_q_readers>();

    const auto q_reader = TensorAccessor(q_args, q_addr);
    const auto local_k_reader = TensorAccessor(k_args, k_addr);
    const auto local_v_reader = TensorAccessor(v_args, v_addr);
    const auto gathered_k_reader = TensorAccessor(gathered_k_args, gathered_k_addr);
    const auto gathered_v_reader = TensorAccessor(gathered_v_args, gathered_v_addr);
    const auto joint_q_reader = TensorAccessor(joint_q_args, joint_q_addr);
    const auto joint_k_reader = TensorAccessor(joint_k_args, joint_k_addr);
    const auto joint_v_reader = TensorAccessor(joint_v_args, joint_v_addr);

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
        // find out which is the latest ring_id that synchronized
        uint32_t ring_id = fused_op_receiver.get_next_ring_id_and_sync();
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
            // Mirror compute's K-loop extension: include the straddle chunk so K/V tiles
            // for it get loaded. Compute -inf-masks its late-half columns via lw_mask.
            using Straddle = KCausalStraddleInfo<local_padded_Nt, Sk_chunk_t>;
            if constexpr (Straddle::has_straddle) {
                iter_num_kv_chunks = Straddle::straddle_chunk_id + 1;
            }
        }

        // When K mcast is enabled, loop max_q_per_core times to stay synchronized with injector
        // Cores with less work do padded iterations (K mcast sync only, no real work)
        const uint32_t loop_q_count = (k_uses_batch_chain && batch_mcast_enabled) ? max_q_per_core : q_per_core;

        for (uint32_t q_iter = 0; q_iter < loop_q_count; ++q_iter) {
            // Check if this is a real iteration (has actual work) or padded (K mcast sync only)
            const bool is_padded_iter = (q_iter >= q_per_core);

            // Calculate global_q_chunk for all iterations (including padded).
            // For padded iterations, global index may be out of bounds, but q_chunk = global_q_chunk % num_q_chunks
            // gives a valid position that correctly determines whether to skip this iteration.
            uint32_t global_q_chunk = remap_q_index(global_q_start + q_iter, num_q_chunks, use_zigzag_balancing);

            // global_q_chunk is index into `B * NH * num_q_chunks`. Need to get nb, nq, q_chunk from this.
            const uint32_t nb = global_q_chunk / (NH * num_q_chunks);
            const uint32_t nq = (global_q_chunk % (NH * num_q_chunks)) / num_q_chunks;
            const uint32_t q_chunk = global_q_chunk % num_q_chunks;
            const auto q_row_start_tile = q_chunk * Sq_chunk_t;
            const bool is_joint_q = q_chunk >= num_local_q_chunks;

            const bool balanced_skip_q = q_chunk < half_sequence && is_balanced && ring_index < ring_id;

            // Balanced causal skip: this Q chunk is handled by the paired device. Reader sends
            // nothing (no Q, no K/V) — compute's normalize-only path on the last ring iter does
            // not read Q (normalize uses only restored sum/out).
            // Skip logic applies to all iterations (including padded) so injector and receivers
            // make the same skip decisions, keeping K mcast sync aligned.
            if (balanced_skip_q) {
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

            // Iteration counter for chain forwarding decisions
            const uint32_t q_iter_local = q_iter;

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

                // K: either read locally (injector or not participant) or receive from chain
                if constexpr (k_uses_batch_chain && batch_mcast_enabled) {
                    // Ensures that compute has completed with the previous K chunk before we overwrite the buffer with
                    // the next K chunk for mcast.
                    const uint32_t reserve_tiles = is_padded_iter ? 2 * k_chunk_tiles : k_chunk_tiles;
                    cb_reserve_back(cb_k_in, reserve_tiles);
                } else {
                    cb_reserve_back(cb_k_in, k_chunk_tiles);
                }
                uint32_t cb_k_start_address = get_write_ptr(cb_k_in);
                if (k_chain.should_receive(nb, nq)) {
                    k_chain.receive();
                } else {
                    // Injector or non-participant: read K from DRAM
                    // For padded iterations, injector still reads K to broadcast to receivers
                    fetch_block(
                        kv_chunk_is_joint ? joint_k_generator
                                          : (ring_iter == 0 ? local_k_generator : gathered_k_generator),
                        k_slice,
                        end_seq_tile,
                        cb_k_start_address,
                        k_tile_bytes,
                        true /*transpose*/
                    );
                }

                // Forward K chunk via chain (uses K's data size explicitly)
                if (k_chain.should_forward(nb, nq, q_iter_local)) {
                    k_chain.forward(cb_k_start_address, k_chunk_tiles, k_tile_bytes);
                }

                // Skip Q, V reads and V forward for padded iterations (K mcast sync only).
                // Note: cb_push_back is intentionally skipped — without it, the write pointer
                // doesn't advance, so cb_reserve_back returns the same address each iteration.
                // This lets the buffer act as a reusable staging area for the mcast.
                if (is_padded_iter) {
                    continue;
                }

                // Make K available to compute
                cb_push_back(cb_k_in, k_chunk_tiles);
                KV_chunks_processed_in_iter++;

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
                                false /*transpose*/,
                                q_barrier_threshold);
                        }
                    } else {
                        read_block(
                            is_joint_q ? joint_q_generator : q_generator,
                            q_slice,
                            q_end_seq_tile,
                            cb_q_in,
                            q_tile_bytes,
                            false /*transpose*/,
                            q_barrier_threshold);
                    }
                    q_pushed = true;
                }

                // V: either read locally (injector or not participant) or receive from chain
                cb_reserve_back(cb_v_in, v_chunk_tiles);
                uint32_t cb_v_start_address = get_write_ptr(cb_v_in);
                if (v_chain.should_receive(nb, nq)) {
                    v_chain.receive();
                } else {
                    fetch_block(
                        kv_chunk_is_joint ? joint_v_generator
                                          : (ring_iter == 0 ? local_v_generator : gathered_v_generator),
                        v_slice,
                        end_seq_tile,
                        cb_v_start_address,
                        v_tile_bytes,
                        false /*transpose*/
                    );
                }

                // Forward V to next core(s) before push_back — prevents compute from
                // popping the buffer while the mcast is still reading from it.
                if (v_chain.should_forward(nb, nq, q_iter_local)) {
                    v_chain.forward(cb_v_start_address);
                }

                // Make V available to compute
                cb_push_back(cb_v_in, v_chunk_tiles);
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
