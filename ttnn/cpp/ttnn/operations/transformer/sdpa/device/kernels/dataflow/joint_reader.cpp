// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "experimental/kernel_args.h"
#include "dataflow_common.hpp"

void kernel_main() {
    Noc noc;

    constexpr auto B = get_arg(args::B);
    constexpr auto NH = get_arg(args::NH);
    constexpr auto DHt = get_arg(args::DHt);
    constexpr auto Sq_chunk_t = get_arg(args::Sq_chunk_t);
    constexpr auto Sk_chunk_t = get_arg(args::Sk_chunk_t);
    constexpr auto k_num_chunks = get_arg(args::k_num_chunks);
    constexpr auto valid_Nt = get_arg(args::valid_Nt);
    constexpr auto valid_Lt = get_arg(args::valid_Lt);
    constexpr auto padded_Nqt = get_arg(args::padded_Nqt);
    constexpr auto padded_Nkt = get_arg(args::padded_Nkt);
    constexpr auto padded_Lqt = get_arg(args::padded_Lqt);
    constexpr auto padded_Lkt = get_arg(args::padded_Lkt);

    const auto local_batch_start = get_arg(args::local_batch_start);
    const auto local_batch_end = get_arg(args::local_batch_end);
    const auto local_nh_start = get_arg(args::local_nh_start);
    const auto local_nh_end = get_arg(args::local_nh_end);
    const auto local_q_start = get_arg(args::local_q_start);
    const auto local_q_end = get_arg(args::local_q_end);

    // Q/K/V CBs are bound by name; the magic-number CB index is gone.
    constexpr uint32_t q_tile_bytes = get_tile_size(dfb::q_in);
    constexpr uint32_t k_tile_bytes = get_tile_size(dfb::k_in);
    constexpr uint32_t v_tile_bytes = get_tile_size(dfb::v_in);

    const auto q_reader = TensorAccessor(tensor::input_q);
    const auto k_reader = TensorAccessor(tensor::input_k);
    const auto v_reader = TensorAccessor(tensor::input_v);
    const auto joint_q_reader = TensorAccessor(tensor::joint_q);
    const auto joint_k_reader = TensorAccessor(tensor::joint_k);
    const auto joint_v_reader = TensorAccessor(tensor::joint_v);

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
                    cat_q_generator, q_slice, q_row_end_tile, dfb::q_in, q_tile_bytes, false /*transpose*/
                );

                for (uint32_t k_chunk = 0; k_chunk < k_num_chunks; ++k_chunk) {
                    const auto kv_row_start_tile = k_chunk * Sk_chunk_t;
                    const auto kv_row_end_tile = kv_row_start_tile + Sk_chunk_t;
                    const auto kv_slice = Slice(nb, nq, kv_row_start_tile, kv_row_end_tile, 0, DHt);

                    read_block(
                        cat_k_generator, kv_slice, kv_row_end_tile, dfb::k_in, k_tile_bytes, true /*transpose*/
                    );

                    read_block(
                        cat_v_generator, kv_slice, kv_row_end_tile, dfb::v_in, v_tile_bytes, false /*transpose*/
                    );
                }
            }
        }
    }
}
