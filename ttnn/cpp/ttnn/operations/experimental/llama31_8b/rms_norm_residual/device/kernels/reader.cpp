// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar_metal2.hpp"
#include "ttnn/operations/kernel_helper_functions/local_l1_copy.hpp"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "ttnn/operations/normalization/layernorm/device/kernels/layernorm_scaler_tiles.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/dataflow/endpoints.h"

namespace generic = norm::kernel_util::generic;

void kernel_main() {
    const uint32_t NCHt = get_arg(args::NCHt);
    const uint32_t Wt = get_arg(args::Wt);
    const uint32_t tile_offset = get_arg(args::reader_start);

    const Noc noc;
    DataflowBuffer dfb_in0(dfb::in);
    DataflowBuffer dfb_in1(dfb::inb);
    DataflowBuffer dfb_gamma(dfb::gamma);

    const uint32_t src0_tile_bytes = dfb_in0.get_tile_size();

    constexpr auto blk = get_arg(args::block_size);
    constexpr auto W = get_arg(args::W);

    const auto src_a = TensorAccessor(tensor::src);

    const uint32_t gamma_tile_bytes = dfb_gamma.get_tile_size();
    const uint32_t gamma_datum_bytes = gamma_tile_bytes / tt::constants::TILE_HW;
    const uint32_t gamma_row_bytes = tt::constants::TILE_WIDTH * gamma_datum_bytes;
    const uint32_t gamma_face_bytes = tt::constants::FACE_HW * gamma_datum_bytes;
    const uint32_t gamma_half_row_bytes = tt::constants::FACE_WIDTH * gamma_datum_bytes;
    const auto addrg = TensorAccessor(tensor::gamma);
    const uint32_t src1_tile_bytes = dfb_in1.get_tile_size();
    const auto src_b = TensorAccessor(tensor::src_b);

    DataflowBuffer dfb_scaler(dfb::scaler);
    {
        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
            dfb::scaler,
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW,
            dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>(dfb_scaler);
        constexpr uint32_t partial_last_tile_cols = W % tt::constants::TILE_WIDTH;
        constexpr uint32_t num_scaler_tiles = norm::layernorm::reduce_scaler_tile_count(W, tt::constants::TILE_WIDTH);
        if constexpr (num_scaler_tiles == 2) {
            dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
                dfb::scaler,
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_ROW,
                dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>(dfb_scaler, partial_last_tile_cols);
        }
    }
    const uint32_t eps = get_arg(args::eps);
    DataflowBuffer dfb_eps(dfb::eps);
    generate_bcast_col_scalar(dfb_eps, eps);

    const uint32_t gamma_blocks = (Wt + blk - 1) / blk;
    auto read_gamma_blocks = [&](uint32_t b0, uint32_t b1) {
        [[maybe_unused]] const UnicastEndpoint local_ep;
        dfb_gamma.reserve_back(static_cast<uint16_t>((b1 - b0) * blk));
        for (uint32_t b = b0; b < b1; b++) {
            const uint32_t start = b * blk;
            const uint32_t n = (start + blk <= Wt) ? blk : (Wt - start);
            for (uint32_t r = 0; r < n; r++) {
                noc.async_read(
                    addrg,
                    dfb_gamma,
                    gamma_row_bytes,
                    {.page_id = start + r},
                    {.offset_bytes = ((b - b0) * blk + r) * gamma_tile_bytes});
            }
        }
        noc.async_read_barrier();
        for (uint32_t b = b0; b < b1; b++) {
            const uint32_t start = b * blk;
            const uint32_t n = (start + blk <= Wt) ? blk : (Wt - start);
            for (uint32_t r = 0; r < n; r++) {
                [[maybe_unused]] const uint32_t o = ((b - b0) * blk + r) * gamma_tile_bytes;
                noc.async_read(
                    local_ep,
                    dfb_gamma,
                    gamma_half_row_bytes,
                    {.noc_x = my_x[noc.get_noc_id()],
                     .noc_y = my_y[noc.get_noc_id()],
                     .addr = dfb_gamma.get_write_ptr() + o + gamma_half_row_bytes},
                    {.offset_bytes = o + gamma_face_bytes});
            }
        }
        noc.async_read_barrier();
        for (uint32_t b = b0; b < b1; b++) {
            dfb_gamma.push_back(static_cast<uint16_t>(blk));
        }
    };
    auto read_gamma = [&]() {
        for (uint32_t b = 0; b < gamma_blocks; b++) {
            read_gamma_blocks(b, b + 1);
        }
    };

    uint32_t offs = 0;

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        {
            constexpr uint32_t group_tiles = blk * 8;
            for (uint32_t b0 = 0; b0 < Wt; b0 += group_tiles) {
                dfb_in0.reserve_back(static_cast<uint16_t>(group_tiles));
                dfb_in1.reserve_back(static_cast<uint16_t>(group_tiles));
                for (uint32_t r = 0; r < group_tiles; r++) {
                    noc.async_read(
                        src_a,
                        dfb_in0,
                        src0_tile_bytes,
                        {.page_id = offs + b0 + r + tile_offset},
                        {.offset_bytes = r * src0_tile_bytes});
                    noc.async_read(
                        src_b,
                        dfb_in1,
                        src1_tile_bytes,
                        {.page_id = offs + b0 + r + tile_offset},
                        {.offset_bytes = r * src1_tile_bytes});
                }
                noc.async_read_barrier();
                dfb_in0.push_back(static_cast<uint16_t>(group_tiles));
                dfb_in1.push_back(static_cast<uint16_t>(group_tiles));
            }
        }

        if (ncht == 0) {
            read_gamma();
        }
        offs += Wt;
    }
}
