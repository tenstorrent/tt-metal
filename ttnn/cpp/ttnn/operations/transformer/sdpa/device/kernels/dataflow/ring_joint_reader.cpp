// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "dataflow_common.hpp"
#include "fused_op_receiver.hpp"

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NH = get_compile_time_arg_val(1);
    constexpr uint32_t DHt = get_compile_time_arg_val(2);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(3);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(4);
    constexpr uint32_t local_Nt = get_compile_time_arg_val(5);
    constexpr uint32_t global_Nt = get_compile_time_arg_val(6);
    constexpr uint32_t logical_Lt = get_compile_time_arg_val(7);
    constexpr uint32_t padded_Lqt = get_compile_time_arg_val(8);
    constexpr uint32_t padded_Lkt = get_compile_time_arg_val(9);
    constexpr uint32_t num_cores = get_compile_time_arg_val(10);
    constexpr uint32_t ring_size = get_compile_time_arg_val(11);
    constexpr uint32_t N_k_num_chunks_local = get_compile_time_arg_val(12);
    constexpr uint32_t L_k_num_chunks = get_compile_time_arg_val(13);
    constexpr uint32_t global_logical_NK_chunks = get_compile_time_arg_val(14);
    constexpr uint32_t global_padded_NK_chunks = get_compile_time_arg_val(15);
    constexpr uint32_t q_num_chunks = get_compile_time_arg_val(16);

    constexpr auto q_args = TensorAccessorArgs<17>();
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

    RingSDPAOpReceiver fused_op_receiver = RingSDPAOpReceiver(
        true, /* wait_for_op_signal */
        argidx);

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;

    constexpr uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);
    constexpr uint32_t v_tile_bytes = get_tile_size(cb_v_in);

    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<q_tile_bytes, num_cores>();

    const auto q_reader = TensorAccessor(q_args, q_addr, q_tile_bytes);
    const auto local_k_reader = TensorAccessor(k_args, k_addr, k_tile_bytes);
    const auto local_v_reader = TensorAccessor(v_args, v_addr, v_tile_bytes);
    const auto gathered_k_reader = TensorAccessor(gathered_k_args, gathered_k_addr, k_tile_bytes);
    const auto gathered_v_reader = TensorAccessor(gathered_v_args, gathered_v_addr, v_tile_bytes);
    const auto joint_q_reader = TensorAccessor(joint_q_args, joint_q_addr, q_tile_bytes);
    const auto joint_k_reader = TensorAccessor(joint_k_args, joint_k_addr, k_tile_bytes);
    const auto joint_v_reader = TensorAccessor(joint_v_args, joint_v_addr, v_tile_bytes);

    const auto q_input_tile_logical = TensorTileShape(B, NH, local_Nt, DHt);
    const auto local_kv_input_tile_logical = TensorTileShape(B, NH, local_Nt, DHt);
    const auto gathered_kv_input_tile_logical = TensorTileShape(B, NH, global_Nt, DHt);
    const auto joint_tile_logical = TensorTileShape(B, NH, logical_Lt, DHt);

    const auto cat_q_generator =
        CatAddrGenerator(q_reader, q_input_tile_logical, local_Nt, joint_q_reader, joint_tile_logical, padded_Lqt);

    const auto cat_local_k_generator = CatAddrGenerator(
        local_k_reader, local_kv_input_tile_logical, local_Nt, joint_k_reader, joint_tile_logical, padded_Lkt);
    const auto cat_local_v_generator = CatAddrGenerator(
        local_v_reader, local_kv_input_tile_logical, local_Nt, joint_v_reader, joint_tile_logical, padded_Lkt);

    const auto cat_gathered_k_generator = CatAddrGenerator(
        gathered_k_reader, gathered_kv_input_tile_logical, global_Nt, joint_k_reader, joint_tile_logical, padded_Lkt);
    const auto cat_gathered_v_generator = CatAddrGenerator(
        gathered_v_reader, gathered_kv_input_tile_logical, global_Nt, joint_v_reader, joint_tile_logical, padded_Lkt);

    /**
     * Iterate over ring indices.
     * On the first iteration, read from local K, V.
     * On subsequent iterations, read from gathered K, V. Sync with AllGather fused signaler.
     */
    for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
        // find out which is the latest ring_id that synchronized
        uint32_t ring_id = fused_op_receiver.get_next_ring_id_and_sync();
        // Iterate over KV blocks gathered on ring.
        // Only the last iteration will append joint_K, joint_V to K, V.
        const uint32_t iter_k_num_chunks =
            ring_id == ring_size - 1 ? (N_k_num_chunks_local + L_k_num_chunks) : N_k_num_chunks_local;

        const uint32_t iter_k_chunk_start = ring_id * N_k_num_chunks_local;

        const uint32_t iter_k_chunk_end = iter_k_chunk_start + iter_k_num_chunks;

        // On the first ring iteration, read from local K, V. On subsequent iterations, read from gathered K, V.
        const auto k_generator = ring_iter == 0 ? cat_local_k_generator : cat_gathered_k_generator;
        const auto v_generator = ring_iter == 0 ? cat_local_v_generator : cat_gathered_v_generator;

        for (uint32_t global_q_chunk = global_q_start; global_q_chunk < global_q_end; ++global_q_chunk) {
            // global_q_chunk is index into `B * NH * num_q_chunks`. Need to get nb, nq, q_chunk from this.
            const uint32_t nb = global_q_chunk / (NH * q_num_chunks);
            const uint32_t nq = (global_q_chunk % (NH * q_num_chunks)) / q_num_chunks;
            const uint32_t q_chunk = global_q_chunk % q_num_chunks;
            const auto q_row_start_tile = q_chunk * Sq_chunk_t;
            const auto q_slice = Slice(nb, nq, q_row_start_tile, q_row_start_tile + Sq_chunk_t, 0, DHt);

            read_block(
                cat_q_generator, q_slice, cb_q_in, q_tile_bytes, barrier_threshold, false /*transpose*/
            );

            for (uint32_t k_chunk = iter_k_chunk_start; k_chunk < iter_k_chunk_end; ++k_chunk) {
                if (k_chunk >= global_logical_NK_chunks && k_chunk < global_padded_NK_chunks) {
                    // This is a KV chunk on spatial input beyond the chunk-padded length of the spatial input.
                    // If k_chunk >= global_padded_NK_chunks, then this is a joint KV chunk.
                    continue;
                }

                uint32_t k_chunk_adjusted = k_chunk;
                if (ring_iter == 0) {
                    /**
                     * If reading from local K, V, adjust the k chunk index to be relative to the local K, V.
                     */
                    k_chunk_adjusted = k_chunk - iter_k_chunk_start;
                }

                const auto kv_row_start_tile = k_chunk_adjusted * Sk_chunk_t;
                const auto kv_slice = Slice(nb, nq, kv_row_start_tile, kv_row_start_tile + Sk_chunk_t, 0, DHt);

                read_block(
                    k_generator, kv_slice, cb_k_in, k_tile_bytes, barrier_threshold, true /*transpose*/
                );

                read_block(
                    v_generator, kv_slice, cb_v_in, v_tile_bytes, barrier_threshold, false /*transpose*/
                );
            }
        }
    }
}
