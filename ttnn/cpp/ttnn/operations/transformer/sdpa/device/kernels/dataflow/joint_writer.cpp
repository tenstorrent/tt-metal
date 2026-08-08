// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 joint SDPA writer. Owned exclusively by JointSDPAProgramFactory, so ported in place:
// named CTAs/RTAs (args::), DFB handles (dfb::), typed tensor bindings (TensorAccessor(tensor::
// name)). The writer generates the mask/scale/col-identity DFBs consumed by compute and drains the
// output DFB to DRAM. On Gen1 dfb::name is the physical CB id, so the shared helpers + generate_*
// utilities work unchanged. Behavior is identical.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
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
    constexpr uint32_t unpadded_N = get_arg(args::unpadded_N);
    constexpr uint32_t unpadded_L = get_arg(args::unpadded_L);
    [[maybe_unused]] constexpr uint32_t num_cores = get_arg(args::num_cores);
    constexpr uint32_t identity_scalar_packed = get_arg(args::identity_scalar_packed);
    [[maybe_unused]] constexpr uint32_t scale_val = get_arg(args::scale_val);
    constexpr bool use_joint_mask = get_arg(args::use_joint_mask) == 1;
    constexpr uint32_t mask_chunk_0 = get_arg(args::mask_chunk_0);
    constexpr uint32_t mask_chunk_1 = get_arg(args::mask_chunk_1);

    const uint32_t local_batch_start = get_arg(args::local_batch_start);
    const uint32_t local_batch_end = get_arg(args::local_batch_end);
    const uint32_t local_nh_start = get_arg(args::local_nh_start);
    const uint32_t local_nh_end = get_arg(args::local_nh_end);
    const uint32_t local_q_start = get_arg(args::local_q_start);
    const uint32_t local_q_end = get_arg(args::local_q_end);

    constexpr uint32_t cb_out = dfb::out;
    constexpr uint32_t cb_mask_in = dfb::mask;
    constexpr uint32_t tile_bytes = get_tile_size(cb_out);

    const auto out_writer = TensorAccessor(tensor::out);
    const auto joint_out_writer = TensorAccessor(tensor::joint_out);

    const auto output_tile_logical = TensorTileShape(B, NH, valid_Nt, DHt);
    const auto joint_tile_logical = TensorTileShape(B, NH, valid_Lt, DHt);
    const auto cat_out_generator =
        CatAddrGenerator(out_writer, output_tile_logical, padded_Nqt, joint_out_writer, joint_tile_logical, padded_Lqt);

    constexpr uint32_t cb_identity_scale_in = dfb::scale;
    constexpr uint32_t cb_col_identity = dfb::col_identity;

    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_identity_scale_in,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();
    generate_bcast_col_scalar(CircularBuffer(cb_col_identity), identity_scalar_packed);

    for (uint32_t nb = local_batch_start; nb < local_batch_end; ++nb) {
        for (uint32_t nq = local_nh_start; nq < local_nh_end; ++nq) {
            for (uint32_t q_chunk = local_q_start; q_chunk < local_q_end; ++q_chunk) {
                generate_mask<false, 0, use_joint_mask, cb_mask_in>(
                    noc,
                    Sq_chunk_t,
                    Sk_chunk_t,
                    q_chunk,
                    0,
                    mask_chunk_0 != (uint32_t)(-1),
                    mask_chunk_1 != (uint32_t)(-1),
                    unpadded_N,
                    unpadded_L,
                    false);

                const uint32_t out_row_start_tile = q_chunk * Sq_chunk_t;
                const auto dst_slice = Slice(nb, nq, out_row_start_tile, out_row_start_tile + Sq_chunk_t, 0, DHt);
                const auto out_row_end_tile = out_row_start_tile + Sq_chunk_t;
                write_block(noc, cat_out_generator, dst_slice, out_row_end_tile, cb_out, tile_bytes);
            }
        }
    }
}
