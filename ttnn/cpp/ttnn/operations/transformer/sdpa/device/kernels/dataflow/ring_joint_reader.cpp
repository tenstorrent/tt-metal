// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
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

    uint32_t argidx = 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t k_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t v_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t joint_q_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t joint_k_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t joint_v_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t global_q_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t global_q_end = get_arg_val<uint32_t>(argidx++);

    RingSDPAOpReceiver fused_op_receiver = RingSDPAOpReceiver(
        true, /* wait_for_op_signal */
        argidx);

    constexpr bool is_dram = true;

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;

    constexpr uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    constexpr DataFormat q_data_format = get_dataformat(cb_q_in);
    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);
    constexpr DataFormat k_data_format = get_dataformat(cb_k_in);
    constexpr uint32_t v_tile_bytes = get_tile_size(cb_v_in);
    constexpr DataFormat v_data_format = get_dataformat(cb_v_in);

    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<q_tile_bytes, num_cores>();

    const InterleavedAddrGenFast<is_dram> q_reader = {
        .bank_base_address = q_addr, .page_size = q_tile_bytes, .data_format = q_data_format};
    const InterleavedAddrGenFast<is_dram> k_reader = {
        .bank_base_address = k_addr, .page_size = k_tile_bytes, .data_format = k_data_format};
    const InterleavedAddrGenFast<is_dram> v_reader = {
        .bank_base_address = v_addr, .page_size = v_tile_bytes, .data_format = v_data_format};
    const InterleavedAddrGenFast<is_dram> joint_q_reader = {
        .bank_base_address = joint_q_addr, .page_size = q_tile_bytes, .data_format = q_data_format};
    const InterleavedAddrGenFast<is_dram> joint_k_reader = {
        .bank_base_address = joint_k_addr, .page_size = k_tile_bytes, .data_format = k_data_format};
    const InterleavedAddrGenFast<is_dram> joint_v_reader = {
        .bank_base_address = joint_v_addr, .page_size = v_tile_bytes, .data_format = v_data_format};

    const auto q_input_tile_logical = TensorTileShape(B, NH, local_Nt, DHt);
    const auto kv_input_tile_logical = TensorTileShape(B, NH, global_Nt, DHt);
    const auto joint_tile_logical = TensorTileShape(B, NH, logical_Lt, DHt);
    const auto cat_q_generator =
        CatAddrGenerator(q_reader, q_input_tile_logical, local_Nt, joint_q_reader, joint_tile_logical, padded_Lqt);
    const auto cat_k_generator =
        CatAddrGenerator(k_reader, kv_input_tile_logical, global_Nt, joint_k_reader, joint_tile_logical, padded_Lkt);
    const auto cat_v_generator =
        CatAddrGenerator(v_reader, kv_input_tile_logical, global_Nt, joint_v_reader, joint_tile_logical, padded_Lkt);

    for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
        // find out which is the latest ring_id that synchronized
        uint32_t ring_id = fused_op_receiver.get_next_ring_id_and_sync();
        // Iterate over KV blocks gathered on ring.
        // Only the last iteration will append joint_K, joint_V to K, V.
        const uint32_t iter_k_num_chunks =
            ring_id == ring_size - 1 ? (N_k_num_chunks_local + L_k_num_chunks) : N_k_num_chunks_local;
        const uint32_t iter_k_chunk_start = ring_id * N_k_num_chunks_local;
        const uint32_t iter_k_chunk_end = iter_k_chunk_start + iter_k_num_chunks;

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

                const auto kv_row_start_tile = k_chunk * Sk_chunk_t;
                const auto kv_slice = Slice(nb, nq, kv_row_start_tile, kv_row_start_tile + Sk_chunk_t, 0, DHt);

                read_block(
                    cat_k_generator, kv_slice, cb_k_in, k_tile_bytes, barrier_threshold, true /*transpose*/
                );

                read_block(
                    cat_v_generator, kv_slice, cb_v_in, v_tile_bytes, barrier_threshold, false /*transpose*/
                );
            }
        }
    }
}
