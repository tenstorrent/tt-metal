// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 joint SDPA reader. Owned exclusively by JointSDPAProgramFactory, so ported in place:
// named CTAs/RTAs (args::), DFB handles (dfb::), and typed tensor bindings (TensorAccessor(tensor::
// name)) replace the legacy positional CTAs, base-address RTAs, and TensorAccessorArgs plumbing.
// On Gen1 dfb::name yields the framework-allocated physical CB id, so the shared dataflow_common.hpp
// helpers (which build CircularBuffer from the id) work unchanged. Behavior is identical.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "dataflow_common.hpp"

void kernel_main() {
    Noc noc;

    constexpr uint32_t B = get_arg(args::B);
    constexpr uint32_t NH = get_arg(args::NH);
    constexpr uint32_t DHt = get_arg(args::DHt);
    constexpr uint32_t Sq_chunk_t = get_arg(args::Sq_chunk_t);
    constexpr uint32_t Sk_chunk_t = get_arg(args::Sk_chunk_t);
    constexpr uint32_t k_num_chunks = get_arg(args::k_num_chunks);
    constexpr uint32_t valid_Nt = get_arg(args::valid_Nt);
    constexpr uint32_t valid_Lt = get_arg(args::valid_Lt);
    constexpr uint32_t padded_Nqt = get_arg(args::padded_Nqt);
    constexpr uint32_t padded_Nkt = get_arg(args::padded_Nkt);
    constexpr uint32_t padded_Lqt = get_arg(args::padded_Lqt);
    constexpr uint32_t padded_Lkt = get_arg(args::padded_Lkt);
    [[maybe_unused]] constexpr uint32_t num_cores = get_arg(args::num_cores);

    const uint32_t local_batch_start = get_arg(args::local_batch_start);
    const uint32_t local_batch_end = get_arg(args::local_batch_end);
    const uint32_t local_nh_start = get_arg(args::local_nh_start);
    const uint32_t local_nh_end = get_arg(args::local_nh_end);
    const uint32_t local_q_start = get_arg(args::local_q_start);
    const uint32_t local_q_end = get_arg(args::local_q_end);

    constexpr uint32_t cb_q_in = dfb::q_in;
    constexpr uint32_t cb_k_in = dfb::k_in;
    constexpr uint32_t cb_v_in = dfb::v_in;

    constexpr uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);
    constexpr uint32_t v_tile_bytes = get_tile_size(cb_v_in);

    const auto q_reader = TensorAccessor(tensor::q);
    const auto k_reader = TensorAccessor(tensor::k);
    const auto v_reader = TensorAccessor(tensor::v);
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
