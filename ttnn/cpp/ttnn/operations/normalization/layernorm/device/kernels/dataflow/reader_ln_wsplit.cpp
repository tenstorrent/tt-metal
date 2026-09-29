// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Width-split interleaved RMSNorm reader (LayerNormDefaultProgramConfig.width_split = nsplit > 1; one tile row per
// core, this core owns Wt tiles of it starting at tile reader_start):
//   1. the reduce scaler and eps tiles (as reader_unary_interleaved_ln_rm_gb.cpp), then a, block by block
//   2. the partial E[x^2] exchange: compute's dfb::part tile goes into slot my_slot of dfb::recv on every core of the
//      row (itself included; the buffer is at the same L1 address on every core), each one's partial_ready semaphore
//      is bumped, and once nsplit partials have arrived dfb::recv is handed to compute
//   3. the first half of the output blocks, from dfb::out2 (the writer reads b and gamma, drains h = a + b, then the
//      second half of the output blocks, so the writes use both NoCs)
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar_metal2.hpp"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "ttnn/operations/normalization/layernorm/device/kernels/layernorm_scaler_tiles.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "wsplit_dataflow.h"
#include <tt-metalium/constants.hpp>

namespace generic = norm::kernel_util::generic;

void kernel_main() {
    const uint32_t Wt = get_arg(args::Wt);
    const uint32_t tile_offset = get_arg(args::reader_start);
    const uint32_t my_slot = get_arg(args::my_slot);
    const uint32_t gamma_start = get_arg(args::gamma_start);
    constexpr auto nsplit = get_arg(args::nsplit);
    constexpr auto blk = get_arg(args::block_size);
    constexpr auto W = get_arg(args::W);
    const uint32_t peer_x[3] = {get_arg(args::peer_x0), get_arg(args::peer_x1), get_arg(args::peer_x2)};
    const uint32_t peer_y[3] = {get_arg(args::peer_y0), get_arg(args::peer_y1), get_arg(args::peer_y2)};

    const Noc noc;
    DataflowBuffer dfb_in0(dfb::in);
    DataflowBuffer dfb_part(dfb::part);
    DataflowBuffer dfb_recv(dfb::recv);
    Semaphore partial_ready(sem::partial_ready);
    dfb_recv.reserve_back(nsplit);
    const uint32_t recv_base = dfb_recv.get_write_ptr();

    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        dfb::scaler,
        ckernel::PoolType::SUM,
        ckernel::ReduceDim::REDUCE_ROW,
        dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();
    static_assert(W % tt::constants::TILE_WIDTH == 0, "width split: tile-aligned width");
    DataflowBuffer dfb_eps(dfb::eps);
    generate_bcast_col_scalar(dfb_eps, get_arg(args::eps));

    // 1. a
    const uint32_t src0_tile_bytes = dfb_in0.get_tile_size();
    const auto src_a = TensorAccessor(tensor::src);
    for (auto block : generic::blocks(Wt, blk)) {
        dfb_in0.reserve_back(static_cast<uint16_t>(block.full_block_size()));
        uint32_t idx = 0;
        for (auto r : block.local()) {
            noc.async_read(
                src_a,
                dfb_in0,
                src0_tile_bytes,
                {.page_id = tile_offset + block.start() + r},
                {.offset_bytes = idx * src0_tile_bytes});
            idx++;
        }
        noc.async_read_barrier();
        dfb_in0.push_back(static_cast<uint16_t>(block.full_block_size()));
    }

    // 1b. gamma (while compute squares and reduces; needed for x * gamma right after the partial): one row-major stick
    // per tile into row 0 of the tile, then its second half-row to row 0 of face 1; compute never pops it (waits on
    // block.start() + block.full_block_size())
    {
        DataflowBuffer dfb_gamma(dfb::gamma);
        const uint32_t gamma_tile_bytes = dfb_gamma.get_tile_size();
        const uint32_t gamma_datum_bytes = gamma_tile_bytes / tt::constants::TILE_HW;
        const uint32_t gamma_row_bytes = tt::constants::TILE_WIDTH * gamma_datum_bytes;
        const uint32_t gamma_face_bytes = tt::constants::FACE_HW * gamma_datum_bytes;
        const uint32_t gamma_half_row_bytes = tt::constants::FACE_WIDTH * gamma_datum_bytes;
        const auto addrg = TensorAccessor(tensor::gamma);
        uint32_t gb_entries = 0;
        for (auto block : generic::blocks(Wt, blk)) {
            gb_entries += block.full_block_size();
        }
        dfb_gamma.reserve_back(static_cast<uint16_t>(gb_entries));
        for (uint32_t t = 0; t < Wt; ++t) {
            noc.async_read(
                addrg,
                dfb_gamma,
                gamma_row_bytes,
                {.page_id = gamma_start + t},
                {.offset_bytes = t * gamma_tile_bytes});
        }
        noc.async_read_barrier();
        const UnicastEndpoint local_ep;
        for (uint32_t t = 0; t < Wt; ++t) {
            noc.async_read(
                local_ep,
                dfb_gamma,
                gamma_half_row_bytes,
                {.noc_x = my_x[noc.get_noc_id()],
                 .noc_y = my_y[noc.get_noc_id()],
                 .addr = dfb_gamma.get_write_ptr() + (t * gamma_tile_bytes) + gamma_half_row_bytes},
                {.offset_bytes = (t * gamma_tile_bytes) + gamma_face_bytes});
        }
        noc.async_read_barrier();
        for (auto block : generic::blocks(Wt, blk)) {
            dfb_gamma.push_back(static_cast<uint16_t>(block.full_block_size()));
        }
    }

    // 2. partial E[x^2] exchange with the row's cores
    dfb_part.wait_front(1);
    const uint32_t part_bytes = dfb_part.get_tile_size();
    const uint32_t part_addr = dfb_part.get_read_ptr();
    for (uint32_t q = 0; q < nsplit; ++q) {
        noc.async_write(
            CoreLocalMem<uint32_t>(part_addr),
            UnicastEndpoint{},
            part_bytes,
            {},
            {.noc_x = peer_x[q], .noc_y = peer_y[q], .addr = recv_base + my_slot * part_bytes});
    }
    noc.async_write_barrier();
    for (uint32_t q = 0; q < nsplit; ++q) {
        partial_ready.up(noc, peer_x[q], peer_y[q], 1);
    }
    dfb_part.pop_front(1);
    partial_ready.wait(nsplit);
    partial_ready.set(0);
    dfb_recv.push_back(nsplit);

    // 3. the first half of the output blocks (the writer drains h = a + b, then the rest)
    {
        DataflowBuffer dfb_out2(dfb::out2);
        norm::layernorm::wsplit::drain_blocks(
            noc,
            dfb_out2,
            TensorAccessor(tensor::dst),
            tile_offset,
            Wt,
            blk,
            0,
            norm::layernorm::wsplit::reader_out_blocks(Wt, blk));
    }
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
