// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
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
    constexpr uint32_t q_num_chunks = get_compile_time_arg_val(23);

    constexpr auto out_args = TensorAccessorArgs<24>();
    constexpr auto joint_out_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto lse_args = TensorAccessorArgs<joint_out_args.next_compile_time_args_offset()>();

    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * DHt;

    // Only one iteration of the ring will contain the masked portion of the spatial input.
    constexpr uint32_t N_mask_ring_id = mask_chunk_0 / N_k_num_chunks_local;
    // The last iteration will concatenate L, which contains the masked portion of the joint tensor.
    constexpr uint32_t L_mask_ring_id = ring_size - 1;

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

    const auto output_tile_logical = TensorTileShape(B, NH, local_Nt, DHt);
    const auto joint_tile_logical = TensorTileShape(B, NH, logical_Lt, DHt);
    const auto cat_out_generator =
        CatAddrGenerator(out_writer, output_tile_logical, local_Nt, joint_out_writer, joint_tile_logical, padded_Lqt);

    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<tile_bytes, num_cores>();
    uint32_t barrier_count = 0;

    constexpr uint32_t cb_scale_in = tt::CBIndex::c_4;
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_8;
    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;

    generate_bcast_unary_scalar(cb_scale_in, scale_val);
    generate_bcast_col_scalar(cb_col_identity, identity_scalar_packed);
    generate_reduce_scaler(cb_identity_scale_in, identity_scalar_packed);

    for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
        uint32_t ring_id = fused_op_receiver.get_next_ring_id_and_sync();
        for (uint32_t global_q_chunk = global_q_start; global_q_chunk < global_q_end; ++global_q_chunk) {
            // global_q_chunk is index into `B * NH * num_q_chunks`. Need to get nb, nq, q_chunk from this.
            const uint32_t nb = global_q_chunk / (NH * q_num_chunks);
            const uint32_t nq = (global_q_chunk % (NH * q_num_chunks)) / q_num_chunks;
            const uint32_t q_chunk = global_q_chunk % q_num_chunks;

            generate_mask<false, false, use_joint_mask, false, 0, cb_mask_in>(
                Sq_chunk_t,
                Sk_chunk_t,
                0,
                0,
                ring_id == N_mask_ring_id && mask_chunk_0 != (uint32_t)(-1),
                ring_id == L_mask_ring_id && mask_chunk_1 != (uint32_t)(-1),
                0,
                logical_N,
                logical_L,
                0);

            if (ring_iter > 0) {
                read_prev_output_and_lse(
                    cat_out_generator,
                    lse_writer,
                    nb,
                    nq,
                    NH,
                    q_chunk,
                    Sq_chunk_t,
                    DHt,
                    local_Nt,
                    logical_Lt,
                    cb_prev_out,
                    cb_lse_in,
                    tile_bytes,
                    lse_tile_bytes,
                    barrier_threshold);
            }

            const uint32_t out_row_start_tile = q_chunk * Sq_chunk_t;
            const auto dst_slice = Slice(nb, nq, out_row_start_tile, out_row_start_tile + Sq_chunk_t, 0, DHt);
            write_block(cat_out_generator, dst_slice, cb_out, tile_bytes, barrier_threshold);

            write_lse_output(
                lse_writer, nb, nq, NH, q_chunk, Sq_chunk_t, local_Nt, logical_Lt, cb_lse_out, lse_tile_bytes);
        }
        noc_async_write_barrier();  // Ensure writes of output and LSE complete before next iteration
    }
}
