// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "dataflow_common.hpp"

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NH = get_compile_time_arg_val(1);
    constexpr uint32_t DHt = get_compile_time_arg_val(2);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(3);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(4);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(5);
    constexpr uint32_t valid_Nt = get_compile_time_arg_val(6);
    constexpr uint32_t valid_Lt = get_compile_time_arg_val(7);
    constexpr uint32_t padded_Nqt = get_compile_time_arg_val(8);
    constexpr uint32_t padded_Nkt = get_compile_time_arg_val(9);
    constexpr uint32_t padded_Lqt = get_compile_time_arg_val(10);
    constexpr uint32_t padded_Lkt = get_compile_time_arg_val(11);
    constexpr uint32_t num_cores = get_compile_time_arg_val(12);

    constexpr auto q_args = TensorAccessorArgs<13>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto joint_q_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto joint_k_args = TensorAccessorArgs<joint_q_args.next_compile_time_args_offset()>();
    constexpr auto joint_v_args = TensorAccessorArgs<joint_k_args.next_compile_time_args_offset()>();

    uint32_t argidx = 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t k_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t v_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t joint_q_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t joint_k_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t joint_v_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_batch_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_batch_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_nh_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_nh_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_q_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_q_end = get_arg_val<uint32_t>(argidx++);

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;

    constexpr uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);
    constexpr uint32_t v_tile_bytes = get_tile_size(cb_v_in);

    const auto q_reader = TensorAccessor(q_args, q_addr, q_tile_bytes);
    const auto k_reader = TensorAccessor(k_args, k_addr, k_tile_bytes);
    const auto v_reader = TensorAccessor(v_args, v_addr, v_tile_bytes);
    const auto joint_q_reader = TensorAccessor(joint_q_args, joint_q_addr, q_tile_bytes);
    const auto joint_k_reader = TensorAccessor(joint_k_args, joint_k_addr, k_tile_bytes);
    const auto joint_v_reader = TensorAccessor(joint_v_args, joint_v_addr, v_tile_bytes);

    const auto input_tile_logical = TensorTileShape(B, NH, valid_Nt, DHt);
    const auto joint_tile_logical = TensorTileShape(B, NH, valid_Lt, DHt);
    const auto cat_q_generator =
        CatAddrGenerator(q_reader, input_tile_logical, padded_Nqt, joint_q_reader, joint_tile_logical, padded_Lqt);
    const auto cat_k_generator =
        CatAddrGenerator(k_reader, input_tile_logical, padded_Nkt, joint_k_reader, joint_tile_logical, padded_Lkt);
    const auto cat_v_generator =
        CatAddrGenerator(v_reader, input_tile_logical, padded_Nkt, joint_v_reader, joint_tile_logical, padded_Lkt);

    for (uint32_t nb = local_batch_start; nb < local_batch_end; ++nb) {
        for (uint32_t nq = local_nh_start; nq < local_nh_end; ++nq) {
            for (uint32_t q_chunk = local_q_start; q_chunk < local_q_end; ++q_chunk) {
                const auto q_row_start_tile = q_chunk * Sq_chunk_t;
                const auto q_row_end_tile = q_row_start_tile + Sq_chunk_t;
                const auto q_slice = Slice(nb, nq, q_row_start_tile, q_row_end_tile, 0, DHt);

                read_block(
                    cat_q_generator, q_slice, q_row_end_tile, cb_q_in, q_tile_bytes, false /*transpose*/
                );

                for (uint32_t k_chunk = 0; k_chunk < k_num_chunks; ++k_chunk) {
                    const auto kv_row_start_tile = k_chunk * Sk_chunk_t;
                    const auto kv_row_end_tile = kv_row_start_tile + Sk_chunk_t;
                    const auto kv_slice = Slice(nb, nq, kv_row_start_tile, kv_row_end_tile, 0, DHt);

                    read_block(
                        cat_k_generator, kv_slice, kv_row_end_tile, cb_k_in, k_tile_bytes, true /*transpose*/
                    );

                    read_block(
                        cat_v_generator, kv_slice, kv_row_end_tile, cb_v_in, v_tile_bytes, false /*transpose*/
                    );
                }
            }
        }
    }
}
