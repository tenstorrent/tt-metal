// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "dataflow_common.hpp"
#include "fused_op_receiver.hpp"
#include "debug/dprint.h"

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NH = get_compile_time_arg_val(1);
    constexpr uint32_t DHt = get_compile_time_arg_val(2);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(3);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(4);
    constexpr uint32_t local_padded_N = get_compile_time_arg_val(5);
    constexpr uint32_t local_padded_Nt = get_compile_time_arg_val(6);
    constexpr uint32_t padded_Nt = get_compile_time_arg_val(7);
    constexpr uint32_t logical_n = get_compile_time_arg_val(8);
    constexpr uint32_t logical_nt = get_compile_time_arg_val(9);
    constexpr uint32_t Lt = get_compile_time_arg_val(10);
    constexpr uint32_t L = get_compile_time_arg_val(11);
    constexpr uint32_t num_local_q_chunks = get_compile_time_arg_val(12);
    constexpr uint32_t num_joint_q_chunks = get_compile_time_arg_val(13);
    constexpr uint32_t num_local_k_chunks = get_compile_time_arg_val(14);
    constexpr uint32_t num_joint_k_chunks = get_compile_time_arg_val(15);
    constexpr uint32_t num_q_chunks = get_compile_time_arg_val(16);
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(17);
    constexpr uint32_t scale_val = get_compile_time_arg_val(18);
    constexpr uint32_t ring_size = get_compile_time_arg_val(19);

    constexpr auto out_args = TensorAccessorArgs<20>();
    constexpr auto joint_out_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto lse_args = TensorAccessorArgs<joint_out_args.next_compile_time_args_offset()>();

    // // Only one iteration of the ring will contain the masked portion of the spatial input.
    // constexpr uint32_t N_mask_ring_id = mask_chunk_0 / N_k_num_chunks_local;
    // // The last iteration will concatenate L, which contains the masked portion of the joint tensor.
    // constexpr uint32_t L_mask_ring_id = ring_size - 1;

    uint32_t argidx = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t joint_out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t lse_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t global_q_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t global_q_end = get_arg_val<uint32_t>(argidx++);

    RingSDPAOpReceiver fused_op_receiver = RingSDPAOpReceiver(
        false, /* wait_for_op_signal */
        argidx);

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

    const auto output_tile_logical = TensorTileShape(B, NH, local_padded_Nt, DHt);
    const auto joint_tile_logical = TensorTileShape(B, NH, Lt, DHt);
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

    for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
        uint32_t ring_id = fused_op_receiver.get_next_ring_id_and_sync();
        const bool do_joint_kv = ring_id == ring_size - 1;
        const uint32_t num_kv_chunks = do_joint_kv ? num_local_k_chunks + num_joint_k_chunks : num_local_k_chunks;

        // If one of these KV chunks contains N as a global index, we need to generate that mask.
        const uint32_t ring_iter_kv_start_tile = ring_id * local_padded_Nt;
        const uint32_t ring_iter_kv_end_tile = ring_iter_kv_start_tile + num_local_k_chunks * Sk_chunk_t;
        const bool global_n_needs_masking = logical_n % (Sk_chunk_t * tt::constants::TILE_HEIGHT) != 0;
        const uint32_t global_n_tile_id = logical_n / tt::constants::TILE_HEIGHT;  // Floor division to get tile ID
        const bool ring_iter_needs_global_n_mask = global_n_needs_masking &&
                                                   (ring_iter_kv_start_tile <= global_n_tile_id) &&
                                                   (ring_iter_kv_end_tile > global_n_tile_id);

        // otherwise if the local_padded_Nt does not divide by q_chunk_size, we need to generate a mask for that last
        // chunk - if it is not beyond global N.
        const bool local_n_needs_masking = local_padded_Nt % Sk_chunk_t != 0;
        // Check if this ring iter has any real work to do on local N. If the iter's start KV chunk is beyond global N,
        // all work will be skipped, so no mask should be generated.
        const bool ring_iter_processes_KV_chunks = ring_iter_kv_start_tile <= global_n_tile_id;
        const bool ring_iter_needs_local_n_mask = local_n_needs_masking && ring_iter_processes_KV_chunks;

        // If this iter processes the joint KV and the joint length L does not divide by q_chunk_size, we need to
        // generate a mask for that last chunk.
        const bool joint_n_needs_masking = L % (Sk_chunk_t * tt::constants::TILE_HEIGHT) != 0;
        const bool ring_iter_needs_joint_n_mask = joint_n_needs_masking && do_joint_kv;

        // Find out if there's any work to do on this ring iter.
        const bool ring_iter_does_work = ring_iter_processes_KV_chunks || (do_joint_kv && L != 0);
        if (!ring_iter_does_work) {
            continue;
        }

        for (uint32_t global_q_chunk = global_q_start; global_q_chunk < global_q_end; ++global_q_chunk) {
            // global_q_chunk is index into `B * NH * num_q_chunks`. Need to get nb, nq, q_chunk from this.
            const uint32_t nb = global_q_chunk / (NH * num_q_chunks);
            const uint32_t nq = (global_q_chunk % (NH * num_q_chunks)) / num_q_chunks;
            const uint32_t q_chunk = global_q_chunk % num_q_chunks;

            if (ring_iter_needs_global_n_mask) {
                DPRINT << "ring id: " << ring_id << " Generating global N mask" << ENDL();
                generate_noncausal_padded_mask<cb_mask_in>(Sq_chunk_t, Sk_chunk_t, logical_n);
            } else if (ring_iter_needs_local_n_mask) {
                DPRINT << "ring id: " << ring_id << " Generating local N mask" << ENDL();
                generate_noncausal_padded_mask<cb_mask_in>(Sq_chunk_t, Sk_chunk_t, local_padded_N);
            }
            if (ring_iter_needs_joint_n_mask) {
                DPRINT << "ring id: " << ring_id << " Generating joint N mask" << ENDL();
                generate_noncausal_padded_mask<cb_mask_in>(Sq_chunk_t, Sk_chunk_t, L);
            }
            // /*
            // If `use_joint_mask`, then one or both of input tensors are padded.
            // We already know that input tensors are padded up to Sk_chunk_t.
            // Therefore, for the last K chunk of the first tensor and the last K chunk of the joint tensor,
            // we should generate the vertical mask.
            // */
            // if (ring_id == N_mask_ring_id) {
            //     if (mask_chunk_0 != (uint32_t)(-1)) {
            //         generate_noncausal_padded_mask<cb_mask_in>(Sq_chunk_t, Sk_chunk_t, logical_N);
            //     }
            // }
            // if (ring_id == L_mask_ring_id) {
            //     if (mask_chunk_1 != (uint32_t)(-1)) {
            //         generate_noncausal_padded_mask<cb_mask_in>(Sq_chunk_t, Sk_chunk_t, logical_L);
            //     }
            // }

            const bool is_joint_q = q_chunk >= num_local_q_chunks;
            Slice out_slice;
            if (is_joint_q) {
                const uint32_t joint_out_row_start_tile = (q_chunk - num_local_q_chunks) * Sq_chunk_t;
                out_slice = Slice(nb, nq, joint_out_row_start_tile, joint_out_row_start_tile + Sq_chunk_t, 0, DHt);
            } else {
                const uint32_t out_row_start_tile = q_chunk * Sq_chunk_t;
                out_slice = Slice(nb, nq, out_row_start_tile, out_row_start_tile + Sq_chunk_t, 0, DHt);
            }

            // If not on the first iteration, read LSE input and previous output chunk.
            // No race condition because writer kernel writes previous output before reading it again

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
                // Read previous output for this Q chunk
                read_block(is_joint_q ? joint_out_generator : out_generator, out_slice, cb_prev_out, tile_bytes, false);

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

            write_block(is_joint_q ? joint_out_generator : out_generator, out_slice, cb_out, tile_bytes);

            cb_wait_front(cb_lse_out, Sq_chunk_t);
            uint32_t lse_addr = get_read_ptr(cb_lse_out);

            for (uint32_t i = lse_seq_start_tile; i < lse_seq_end_tile; i++) {
                noc_async_write_tile(lse_tile_logical.id_of(nb, nq, i, 0), lse_writer, lse_addr);
                lse_addr += lse_tile_bytes;
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_lse_out, Sq_chunk_t);
        }
        noc_async_write_barrier();  // Ensure writes of output and LSE complete before next iteration
    }
}
