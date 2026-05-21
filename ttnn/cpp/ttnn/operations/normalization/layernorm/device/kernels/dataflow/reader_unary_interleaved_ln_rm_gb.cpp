// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/dataflow/endpoints.h"

namespace generic = norm::kernel_util::generic;

void kernel_main() {
    auto NCHt = get_arg(args::NCHt);
    auto Wt = get_arg(args::Wt);
    auto tile_offset = get_arg(args::start_tile_row);

    Noc noc;
    DataflowBuffer cb_in0(dfb::cb_in);
#ifdef FUSE_PRE_ADD
    DataflowBuffer cb_in1(dfb::cb_inb);
#endif
#ifdef FUSE_GAMMA
    DataflowBuffer cb_gamma(dfb::cb_gamma);
#endif
#ifdef FUSE_BETA
    DataflowBuffer cb_beta(dfb::cb_beta);
#endif

    const uint32_t src0_tile_bytes = get_tile_size(dfb::cb_in);
    const DataFormat src0_data_format = get_dataformat(dfb::cb_in);

    constexpr auto blk = get_arg(args::block_size);
    constexpr bool use_welford = get_arg(args::use_welford) == 1;
    [[maybe_unused]] constexpr auto W = get_arg(args::W);

    const auto src_a = TensorAccessor(ta::src_a);

#ifdef FUSE_GAMMA
    const uint32_t gamma_tile_bytes = get_tile_size(dfb::cb_gamma);
    const auto addrg = TensorAccessor(ta::gamma);
#endif
#ifdef FUSE_BETA
    const uint32_t beta_tile_bytes = get_tile_size(dfb::cb_beta);
    const auto addrb = TensorAccessor(ta::beta);
#endif
#ifdef FUSE_PRE_ADD
    const uint32_t src1_tile_bytes = get_tile_size(dfb::cb_inb);
    const auto src_b = TensorAccessor(ta::src_b);
#endif

    if constexpr (!use_welford) {
        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
            dfb::cb_scaler,
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW,
            dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR,
            /*compute_uses_reduce_tile=*/true>();
    }
    const uint32_t eps = get_arg(args::eps);
    generate_bcast_col_scalar(dfb::cb_eps, eps);

    uint32_t offs = 0;

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        for (auto block : generic::blocks(Wt, blk)) {
            cb_in0.reserve_back(block.full_block_size());
            uint32_t idx = 0;
            for (auto r : block.local()) {
                noc.async_read(
                    src_a,
                    cb_in0,
                    src0_tile_bytes,
                    {.page_id = offs + block.start() + r + tile_offset},
                    {.offset_bytes = idx * src0_tile_bytes});
                idx++;
            }
            noc.async_read_barrier();
            cb_in0.push_back(block.full_block_size());

#ifdef FUSE_PRE_ADD
            cb_in1.reserve_back(block.full_block_size());
            idx = 0;
            for (auto r : block.local()) {
                noc.async_read(
                    src_b,
                    cb_in1,
                    src1_tile_bytes,
                    {.page_id = offs + block.start() + r + tile_offset},
                    {.offset_bytes = idx * src1_tile_bytes});
                idx++;
            }
            noc.async_read_barrier();
            cb_in1.push_back(block.full_block_size());
#endif
        }

#if defined FUSE_GAMMA || defined FUSE_BETA
        if (ncht == 0) {
            for (auto block : generic::blocks(Wt, blk)) {
#ifdef FUSE_GAMMA
                {
                    cb_gamma.reserve_back(block.full_block_size());
                    UnicastEndpoint local_ep;
                    uint32_t idx = 0;
                    for (auto r : block.local()) {
                        noc.async_read(
                            addrg,
                            cb_gamma,
                            64,
                            {.page_id = block.start() + r},
                            {.offset_bytes = idx * gamma_tile_bytes});
                        noc.async_read_barrier();
                        noc.async_read(
                            local_ep,
                            cb_gamma,
                            32,
                            {.noc_x = my_x[noc.get_noc_id()],
                             .noc_y = my_y[noc.get_noc_id()],
                             .addr = cb_gamma.get_write_ptr() + idx * gamma_tile_bytes + 32},
                            {.offset_bytes = idx * gamma_tile_bytes + 512});
                        idx++;
                    }
                    noc.async_read_barrier();
                    cb_gamma.push_back(block.full_block_size());
                }
#endif

#ifdef FUSE_BETA
                {
                    cb_beta.reserve_back(block.full_block_size());
                    UnicastEndpoint local_ep;
                    uint32_t idx = 0;
                    for (auto r : block.local()) {
                        noc.async_read(
                            addrb,
                            cb_beta,
                            64,
                            {.page_id = block.start() + r},
                            {.offset_bytes = idx * beta_tile_bytes});
                        noc.async_read_barrier();
                        noc.async_read(
                            local_ep,
                            cb_beta,
                            32,
                            {.noc_x = my_x[noc.get_noc_id()],
                             .noc_y = my_y[noc.get_noc_id()],
                             .addr = cb_beta.get_write_ptr() + idx * beta_tile_bytes + 32},
                            {.offset_bytes = idx * beta_tile_bytes + 512});
                        idx++;
                    }
                    noc.async_read_barrier();
                    cb_beta.push_back(block.full_block_size());
                }
#endif
            }
        }
#endif
        offs += Wt;
    }
}
