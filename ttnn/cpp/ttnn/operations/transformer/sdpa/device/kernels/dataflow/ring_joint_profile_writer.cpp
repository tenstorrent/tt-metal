// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Profile writer for ring_joint_sdpa.
 *
 * Key simplifications from ring_joint_writer.cpp:
 * - No fused_op_receiver synchronization
 * - ring_index is a compile-time arg
 * - Simpler iteration without inter-device coordination
 */

#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"
#include "dataflow_common.hpp"
#include "tools/profiler/kernel_profiler.hpp"

template <typename ReaderType, typename TensorAccessorType>
void read_prev_output_and_lse(
    const PaddedAddrGenerator<ReaderType>& cat_out_generator,
    const TensorAccessorType& lse_writer,
    const TensorTileShape& lse_tile_logical,
    const uint32_t nb,
    const uint32_t nq,
    const uint32_t Sq_chunk_t,
    const Slice out_slice,
    const uint32_t end_seq_tile,
    const uint32_t lse_seq_start_tile,
    const uint32_t lse_seq_end_tile,
    const uint32_t cb_prev_out,
    const uint32_t cb_lse_in,
    const uint32_t tile_bytes,
    const uint32_t lse_tile_bytes) {
    DeviceZoneScopedN("WRITER-READ-PREV");
    // Read previous output for this Q chunk
    read_block(cat_out_generator, out_slice, end_seq_tile, cb_prev_out, tile_bytes, false);

    // Read previous LSE for this Q chunk
    cb_reserve_back(cb_lse_in, Sq_chunk_t);
    uint32_t lse_addr = get_write_ptr(cb_lse_in);
    for (uint32_t i = lse_seq_start_tile; i < lse_seq_end_tile; i++) {
        noc_async_read_tile(lse_tile_logical.id_of(nb, nq, i, 0), lse_writer, lse_addr);
        lse_addr += lse_tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_lse_in, Sq_chunk_t);
}

template <typename ReaderType, typename TensorAccessorType>
void write_output_and_lse(
    const PaddedAddrGenerator<ReaderType>& cat_out_generator,
    const TensorAccessorType& lse_writer,
    const TensorTileShape& lse_tile_logical,
    const uint32_t nb,
    const uint32_t nq,
    const uint32_t Sq_chunk_t,
    const Slice out_slice,
    const uint32_t end_seq_tile,
    const uint32_t lse_seq_start_tile,
    const uint32_t lse_seq_end_tile,
    const uint32_t cb_out,
    const uint32_t cb_lse_out,
    const uint32_t tile_bytes,
    const uint32_t lse_tile_bytes) {
    DeviceZoneScopedN("WRITER-OUT");
    write_block(cat_out_generator, out_slice, end_seq_tile, cb_out, tile_bytes);

    cb_wait_front(cb_lse_out, Sq_chunk_t);
    uint32_t lse_addr = get_read_ptr(cb_lse_out);
    for (uint32_t i = lse_seq_start_tile; i < lse_seq_end_tile; i++) {
        noc_async_write_tile(lse_tile_logical.id_of(nb, nq, i, 0), lse_writer, lse_addr);
        lse_addr += lse_tile_bytes;
    }
    noc_async_writes_flushed();
    cb_pop_front(cb_lse_out, Sq_chunk_t);
}

/**
 * Profile indexer for ring_joint_sdpa.
 * Computes ring_id arrival order without synchronization.
 */
struct ProfileRingIndexer {
    uint32_t ring_size = 0;
    uint32_t ring_index = 0;

    uint32_t received_inputs[2] = {0, 0};
    uint32_t expected_inputs[2] = {0, 0};
    uint32_t curr_dir = 0;
    uint32_t curr_transfer_idx = 0;

    ProfileRingIndexer(uint32_t ring_size_, uint32_t ring_index_) : ring_size(ring_size_), ring_index(ring_index_) {
        expected_inputs[0] = ring_index_;                   // backward
        expected_inputs[1] = ring_size_ - 1 - ring_index_;  // forward
        curr_dir = 0;
    }

    uint32_t get_next_ring_id() {
        uint32_t sender_ring_id;
        if (curr_transfer_idx == 0) {
            sender_ring_id = ring_index;
        } else {
            received_inputs[curr_dir] += 1;
            if (curr_dir == 1) {
                sender_ring_id = (ring_index - received_inputs[curr_dir] + ring_size) % ring_size;
            } else {
                sender_ring_id = (ring_index + received_inputs[curr_dir]) % ring_size;
            }
        }

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

    // Compile-time flag for whether joint tensors are provided
    constexpr bool use_joint_tensors = L > 0;
    constexpr uint32_t num_joint_q_chunks = get_compile_time_arg_val(15);
    constexpr uint32_t num_local_k_chunks = get_compile_time_arg_val(16);
    constexpr uint32_t num_joint_k_chunks = get_compile_time_arg_val(17);
    constexpr uint32_t num_q_chunks = get_compile_time_arg_val(18);
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(19);
    constexpr uint32_t scale_val = get_compile_time_arg_val(20);
    constexpr uint32_t ring_size = get_compile_time_arg_val(21);
    constexpr uint32_t ring_index = get_compile_time_arg_val(22);
    constexpr uint32_t global_n_partial_col = get_compile_time_arg_val(23);
    constexpr uint32_t joint_l_partial_col = get_compile_time_arg_val(24);
    constexpr uint32_t is_causal = get_compile_time_arg_val(25) == 1;
    constexpr uint32_t is_balanced = get_compile_time_arg_val(26) == 1;

    constexpr auto out_args = TensorAccessorArgs<27>();
    constexpr auto joint_out_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto lse_args = TensorAccessorArgs<joint_out_args.next_compile_time_args_offset()>();

    uint32_t argidx = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t joint_out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t lse_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t global_q_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t global_q_end = get_arg_val<uint32_t>(argidx++);

    // Profile indexer: computes ring_id without synchronization
    ProfileRingIndexer indexer(ring_size, ring_index);

    constexpr uint32_t cb_lse_in = tt::CBIndex::c_6;
    constexpr uint32_t cb_prev_out = tt::CBIndex::c_7;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_lse_out = tt::CBIndex::c_17;
    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
    constexpr uint32_t tile_bytes = get_tile_size(cb_out);
    constexpr uint32_t lse_tile_bytes = get_tile_size(cb_lse_in);

    const auto out_writer = TensorAccessor(out_args, out_addr, tile_bytes);
    const auto joint_out_writer = TensorAccessor(joint_out_args, joint_out_addr, tile_bytes);
    const auto lse_writer = TensorAccessor(lse_args, lse_addr, lse_tile_bytes);

    const auto output_tile_logical = TensorTileShape(B, NH, local_padded_Nt, vDHt);
    const auto joint_tile_logical = TensorTileShape(B, NH, Lt, vDHt);
    const auto lse_tile_logical = TensorTileShape(B, NH, local_padded_Nt + Lt, 1);

    const auto out_generator = PaddedAddrGenerator(out_writer, output_tile_logical);
    const auto joint_out_generator = PaddedAddrGenerator(joint_out_writer, joint_tile_logical);
    const auto lse_generator = PaddedAddrGenerator(lse_writer, lse_tile_logical);

    constexpr uint32_t cb_scale_in = tt::CBIndex::c_4;
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_8;
    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;

    generate_bcast_unary_scalar(cb_scale_in, scale_val);
    generate_bcast_col_scalar(cb_col_identity, identity_scalar_packed);
    generate_reduce_scaler(cb_identity_scale_in, identity_scalar_packed);

    // Lightweight mask: generate all mask tiles once into single CB before the ring loop.
    constexpr bool local_n_has_padding = local_padded_Nt % Sk_chunk_t != 0;
    constexpr bool global_n_has_padding = logical_n % (Sk_chunk_t * tt::constants::TILE_HEIGHT) != 0;
    constexpr bool joint_has_padding = L > 0 && L % (Sk_chunk_t * tt::constants::TILE_HEIGHT) != 0;
    constexpr bool needs_lightweight_mask =
        (local_n_has_padding || global_n_has_padding || joint_has_padding) && !is_causal;
    if constexpr (needs_lightweight_mask) {
        generate_lightweight_mask_tiles<global_n_partial_col, joint_l_partial_col, cb_mask_in>();
    }

    uint32_t half_sequence = num_q_chunks / 2;
    for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
        uint32_t ring_id = indexer.get_next_ring_id();
        const bool do_joint_kv = ring_id == ring_size - 1;
        const uint32_t num_kv_chunks = do_joint_kv ? num_local_k_chunks + num_joint_k_chunks : num_local_k_chunks;

        const uint32_t ring_iter_kv_start_tile = ring_id * local_padded_Nt;
        const uint32_t ring_iter_kv_end_tile = ring_iter_kv_start_tile + num_local_k_chunks * Sk_chunk_t;
        const uint32_t global_n_tile_id = logical_n / tt::constants::TILE_HEIGHT;
        const bool ring_iter_processes_KV_chunks = ring_iter_kv_start_tile <= global_n_tile_id;
        const bool ring_iter_does_work = (ring_iter_processes_KV_chunks || (do_joint_kv && L != 0)) &&
                                         !(is_causal && ring_index < ring_id && !is_balanced);
        if (!ring_iter_does_work) {
            continue;
        }

        // GLOBAL N MASK
        const int32_t global_n_within_ring_iter = logical_n - ring_id * local_padded_N;
        const bool global_n_is_within_ring_iter =
            global_n_within_ring_iter > 0 && global_n_within_ring_iter <= (int32_t)local_padded_N;
        const bool global_n_needs_masking = global_n_within_ring_iter % (Sk_chunk_t * tt::constants::TILE_HEIGHT) != 0;
        const bool ring_iter_needs_global_n_mask = global_n_is_within_ring_iter && global_n_needs_masking;

        // LOCAL N MASK
        const bool local_n_needs_masking = local_padded_Nt % Sk_chunk_t != 0;
        const bool ring_iter_needs_local_n_mask = local_n_needs_masking && !global_n_is_within_ring_iter;

        // JOINT L MASK
        const bool joint_n_needs_masking = L % (Sk_chunk_t * tt::constants::TILE_HEIGHT) != 0;
        const bool ring_iter_needs_joint_n_mask = joint_n_needs_masking && do_joint_kv;

        for (uint32_t q_iter = 0; q_iter < (global_q_end - global_q_start); ++q_iter) {
            // Linear flat index for this iteration
            uint32_t linear_flat = global_q_start + q_iter;

#if defined BALANCED_Q_PARALLEL
            // Apply per-head zigzag for load balancing
            uint32_t global_q_chunk = linear_to_zigzag(linear_flat, num_q_chunks);
#else
            uint32_t global_q_chunk = linear_flat;
#endif

            const uint32_t nb = global_q_chunk / (NH * num_q_chunks);
            const uint32_t nq = (global_q_chunk % (NH * num_q_chunks)) / num_q_chunks;
            const uint32_t q_chunk = global_q_chunk % num_q_chunks;

            // Only truly causal case appear in the iteration with local KV
            bool causality = (ring_iter == 0 ? is_causal : false);

            if (q_chunk < half_sequence && is_balanced && ring_index < ring_id) {
                continue;
            }

            if (is_causal) {
                generate_mask<false, 0, true, cb_mask_in>(
                    Sq_chunk_t,
                    Sk_chunk_t,
                    q_chunk,
                    0,
                    ring_iter_needs_global_n_mask || ring_iter_needs_local_n_mask,
                    ring_iter_needs_joint_n_mask,
                    ring_iter_needs_global_n_mask ? global_n_within_ring_iter : local_padded_N,
                    L,
                    causality);
            }

            const bool is_joint_q = q_chunk >= num_local_q_chunks;
            Slice out_slice;
            uint32_t end_seq_tile;
            if (is_joint_q) {
                const uint32_t joint_out_row_start_tile = (q_chunk - num_local_q_chunks) * Sq_chunk_t;
                out_slice = Slice(nb, nq, joint_out_row_start_tile, joint_out_row_start_tile + Sq_chunk_t, 0, vDHt);
                end_seq_tile = Lt;
            } else {
                const uint32_t out_row_start_tile = q_chunk * Sq_chunk_t;
                out_slice = Slice(nb, nq, out_row_start_tile, out_row_start_tile + Sq_chunk_t, 0, vDHt);
                end_seq_tile = local_padded_Nt * (ring_id + 1);
            }

            uint32_t lse_seq_start_tile;
            uint32_t lse_seq_end_tile;
            if (is_joint_q) {
                lse_seq_start_tile = local_padded_Nt + (q_chunk - num_local_q_chunks) * Sq_chunk_t;
                lse_seq_end_tile = lse_seq_start_tile + Sq_chunk_t;
                lse_seq_start_tile = std::min(lse_seq_start_tile, local_padded_Nt + Lt);
                lse_seq_end_tile = std::min(lse_seq_end_tile, local_padded_Nt + Lt);
            } else {
                lse_seq_start_tile = q_chunk * Sq_chunk_t;
                lse_seq_end_tile = lse_seq_start_tile + Sq_chunk_t;
                lse_seq_start_tile = std::min(lse_seq_start_tile, local_padded_Nt);
                lse_seq_end_tile = std::min(lse_seq_end_tile, local_padded_Nt);
            }

            if (ring_iter > 0) {
                if constexpr (use_joint_tensors) {
                    if (is_joint_q) {
                        read_prev_output_and_lse(
                            joint_out_generator,
                            lse_writer,
                            lse_tile_logical,
                            nb,
                            nq,
                            Sq_chunk_t,
                            out_slice,
                            end_seq_tile,
                            lse_seq_start_tile,
                            lse_seq_end_tile,
                            cb_prev_out,
                            cb_lse_in,
                            tile_bytes,
                            lse_tile_bytes);
                    } else {
                        read_prev_output_and_lse(
                            out_generator,
                            lse_writer,
                            lse_tile_logical,
                            nb,
                            nq,
                            Sq_chunk_t,
                            out_slice,
                            end_seq_tile,
                            lse_seq_start_tile,
                            lse_seq_end_tile,
                            cb_prev_out,
                            cb_lse_in,
                            tile_bytes,
                            lse_tile_bytes);
                    }
                } else {
                    read_prev_output_and_lse(
                        out_generator,
                        lse_writer,
                        lse_tile_logical,
                        nb,
                        nq,
                        Sq_chunk_t,
                        out_slice,
                        end_seq_tile,
                        lse_seq_start_tile,
                        lse_seq_end_tile,
                        cb_prev_out,
                        cb_lse_in,
                        tile_bytes,
                        lse_tile_bytes);
                }
            }

            if constexpr (use_joint_tensors) {
                if (is_joint_q) {
                    write_output_and_lse(
                        joint_out_generator,
                        lse_writer,
                        lse_tile_logical,
                        nb,
                        nq,
                        Sq_chunk_t,
                        out_slice,
                        end_seq_tile,
                        lse_seq_start_tile,
                        lse_seq_end_tile,
                        cb_out,
                        cb_lse_out,
                        tile_bytes,
                        lse_tile_bytes);
                } else {
                    write_output_and_lse(
                        out_generator,
                        lse_writer,
                        lse_tile_logical,
                        nb,
                        nq,
                        Sq_chunk_t,
                        out_slice,
                        end_seq_tile,
                        lse_seq_start_tile,
                        lse_seq_end_tile,
                        cb_out,
                        cb_lse_out,
                        tile_bytes,
                        lse_tile_bytes);
                }
            } else {
                write_output_and_lse(
                    out_generator,
                    lse_writer,
                    lse_tile_logical,
                    nb,
                    nq,
                    Sq_chunk_t,
                    out_slice,
                    end_seq_tile,
                    lse_seq_start_tile,
                    lse_seq_end_tile,
                    cb_out,
                    cb_lse_out,
                    tile_bytes,
                    lse_tile_bytes);
            }
        }
        noc_async_write_barrier();  // Ensure writes of output and LSE complete before next iteration
    }
}
