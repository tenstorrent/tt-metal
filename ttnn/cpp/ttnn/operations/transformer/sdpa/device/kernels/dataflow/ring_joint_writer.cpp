// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "dataflow_common.hpp"
#include "debug/dprint.h"
#include <tt-metalium/constants.hpp>

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
    constexpr uint32_t logical_N = get_compile_time_arg_val(10);
    constexpr uint32_t logical_L = get_compile_time_arg_val(11);
    constexpr uint32_t num_cores = get_compile_time_arg_val(12);
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(13);
    constexpr uint32_t scale_val = get_compile_time_arg_val(14);
    constexpr bool use_joint_mask = get_compile_time_arg_val(15) == 1;
    constexpr uint32_t mask_chunk_0 = get_compile_time_arg_val(16);
    constexpr uint32_t mask_chunk_1 = get_compile_time_arg_val(17);
    constexpr uint32_t ring_size = get_compile_time_arg_val(18);
    constexpr uint32_t N_k_num_chunks_local = get_compile_time_arg_val(19);
    constexpr uint32_t L_k_num_chunks = get_compile_time_arg_val(20);
    constexpr uint32_t global_logical_NK_chunks = get_compile_time_arg_val(21);
    constexpr uint32_t global_padded_NK_chunks = get_compile_time_arg_val(22);

    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * DHt;

    // Only one iteration of the ring will contain the masked portion of the spatial input.
    constexpr uint32_t N_mask_ring_id = mask_chunk_0 / N_k_num_chunks_local;
    // The last iteration will concatenate L, which contains the masked portion of the joint tensor.
    constexpr uint32_t L_mask_ring_id = ring_size - 1;

    DPRINT << "WRITER: N_mask_ring_id: " << N_mask_ring_id << ENDL();
    DPRINT << "WRITER: L_mask_ring_id: " << L_mask_ring_id << ENDL();

    uint32_t argidx = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t joint_out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t lse_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_batch_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_batch_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_nh_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_nh_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_q_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_q_end = get_arg_val<uint32_t>(argidx++);

    constexpr bool is_dram = true;
    constexpr uint32_t cb_lse_in = tt::CBIndex::c_6;
    constexpr uint32_t cb_prev_out = tt::CBIndex::c_7;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_lse_out = tt::CBIndex::c_17;
    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
    constexpr uint32_t tile_bytes = get_tile_size(cb_out);
    constexpr DataFormat data_format = get_dataformat(cb_out);

    const InterleavedAddrGenFast<is_dram> out_writer = {
        .bank_base_address = out_addr, .page_size = tile_bytes, .data_format = data_format};
    const InterleavedAddrGenFast<is_dram> joint_out_writer = {
        .bank_base_address = joint_out_addr, .page_size = tile_bytes, .data_format = data_format};
    const InterleavedAddrGenFast<is_dram> lse_writer = {
        .bank_base_address = lse_addr, .page_size = tile_bytes, .data_format = data_format};

    const auto output_tile_logical = TensorTileShape(B, NH, local_Nt, DHt);
    const auto joint_tile_logical = TensorTileShape(B, NH, logical_Lt, DHt);
    const auto cat_out_generator =
        CatAddrGenerator(out_writer, output_tile_logical, local_Nt, joint_out_writer, joint_tile_logical, padded_Lqt);

    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<tile_bytes, num_cores>();
    uint32_t barrier_count = 0;

    constexpr uint32_t cb_scale_in = tt::CBIndex::c_4;
    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;

    generate_bcast_unary_scalar(cb_scale_in, scale_val);
    generate_reduce_scaler(cb_identity_scale_in, identity_scalar_packed);

    for (uint32_t ring_id = 0; ring_id < ring_size; ++ring_id) {
        DPRINT << "WRITER: ring_id: " << ring_id << ENDL();
        for (uint32_t nb = local_batch_start; nb < local_batch_end; ++nb) {
            for (uint32_t nq = local_nh_start; nq < local_nh_end; ++nq) {
                for (uint32_t q_chunk = local_q_start; q_chunk < local_q_end; ++q_chunk) {
                    if constexpr (use_joint_mask) {
                        /*
                        If `use_joint_mask`, then one or both of input tensors are padded.
                        We already know that input tensors are padded up to Sk_chunk_t.
                        Therefore, for the last K chunk of the first tensor and the last K chunk of the joint tensor,
                        we should generate the vertical mask.
                        */
                        if (ring_id == N_mask_ring_id) {
                            if (mask_chunk_0 != (uint32_t)(-1)) {
                                DPRINT << "WRITER: N_mask_ring_id: " << ring_id << ENDL();
                                generate_noncausal_padded_mask<cb_mask_in>(Sq_chunk_t, Sk_chunk_t, logical_N);
                            }
                        }
                        if (ring_id == L_mask_ring_id) {
                            if (mask_chunk_1 != (uint32_t)(-1)) {
                                DPRINT << "WRITER: L_mask_ring_id: " << ring_id << ENDL();
                                generate_noncausal_padded_mask<cb_mask_in>(Sq_chunk_t, Sk_chunk_t, logical_L);
                            }
                        }
                    }

                    const uint32_t out_row_start_tile = q_chunk * Sq_chunk_t;
                    const auto dst_slice = Slice(nb, nq, out_row_start_tile, out_row_start_tile + Sq_chunk_t, 0, DHt);

                    // If ring_id > 0, read LSE input and previous output chunk.
                    // No race condition because writer kernel writes previous output before reading it again

                    if (ring_id > 0) {
                        // Read previous output for this Q chunk
                        read_block(cat_out_generator, dst_slice, cb_prev_out, tile_bytes, barrier_threshold, false);

                        // Read previous LSE for this Q chunk
                        cb_reserve_back(cb_lse_in, Sq_chunk_t);
                        // DEBUG: logging each LSE to different element in batch. Remove
                        uint32_t lse_tile_id = (ring_id - 1) * B * NH * (local_Nt + logical_Lt) +
                                               nb * NH * (local_Nt + logical_Lt) + nq * (local_Nt + logical_Lt) +
                                               q_chunk * Sq_chunk_t;
                        uint32_t lse_addr = get_write_ptr(cb_lse_in);
                        // Don't write beyond the end of the LSE in sequence length
                        uint32_t lse_seq_start = q_chunk * Sq_chunk_t;
                        uint32_t lse_seq_end = lse_seq_start + Sq_chunk_t;
                        lse_seq_start = std::min(lse_seq_start, (local_Nt + logical_Lt));
                        lse_seq_end = std::min(lse_seq_end, (local_Nt + logical_Lt));

                        for (uint32_t i = lse_seq_start; i < lse_seq_end; i++) {
                            noc_async_read_tile(lse_tile_id, lse_writer, lse_addr);
                            lse_tile_id++;
                            lse_addr += tile_bytes;
                        }
                        noc_async_read_barrier();
                        cb_push_back(cb_lse_in, Sq_chunk_t);
                    }

                    write_block(cat_out_generator, dst_slice, cb_out, tile_bytes, barrier_threshold);

                    cb_wait_front(cb_lse_out, Sq_chunk_t);
                    // DEBUG: logging each LSE to different element in batch. Remove
                    uint32_t lse_tile_id = ring_id * B * NH * (local_Nt + logical_Lt) +
                                           nb * NH * (local_Nt + logical_Lt) + nq * (local_Nt + logical_Lt) +
                                           q_chunk * Sq_chunk_t;
                    uint32_t lse_addr = get_read_ptr(cb_lse_out);
                    uint32_t lse_seq_start = q_chunk * Sq_chunk_t;
                    uint32_t lse_seq_end = lse_seq_start + Sq_chunk_t;
                    lse_seq_start = std::min(lse_seq_start, (local_Nt + logical_Lt));
                    lse_seq_end = std::min(lse_seq_end, (local_Nt + logical_Lt));

                    for (uint32_t i = lse_seq_start; i < lse_seq_end; i++) {
                        noc_async_write_tile(lse_tile_id, lse_writer, lse_addr);
                        lse_tile_id++;
                        lse_addr += tile_bytes;
                    }
                    noc_async_writes_flushed();
                    cb_pop_front(cb_lse_out, Sq_chunk_t);
                }
            }
        }
        noc_async_write_barrier();
    }
}
