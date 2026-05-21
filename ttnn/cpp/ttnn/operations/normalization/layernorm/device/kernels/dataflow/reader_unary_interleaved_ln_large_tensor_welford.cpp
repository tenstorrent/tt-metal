// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "layernorm_dataflow_utils.h"

namespace generic = norm::kernel_util::generic;
namespace layernorm_dataflow_utils = norm::layernorm::device::kernels::dataflow;

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

    constexpr auto blk = get_arg(args::block_size);
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

    const uint32_t eps = get_arg(args::eps);
    generate_bcast_col_scalar(dfb::cb_eps, eps);

    uint32_t offs = 0;
    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        // First pass
        for (auto block : generic::blocks(Wt, blk)) {
            layernorm_dataflow_utils::read_block_to_cb(
                noc, cb_in0, src_a, src0_tile_bytes, offs + block.start() + tile_offset, block);
#ifdef FUSE_PRE_ADD
            layernorm_dataflow_utils::read_block_to_cb(
                noc, cb_in1, src_b, src1_tile_bytes, offs + block.start() + tile_offset, block);
#endif
        }

        // Second pass
        for (auto block : generic::blocks(Wt, blk)) {
            layernorm_dataflow_utils::read_block_to_cb(
                noc, cb_in0, src_a, src0_tile_bytes, offs + block.start() + tile_offset, block);
#ifdef FUSE_PRE_ADD
            layernorm_dataflow_utils::read_block_to_cb(
                noc, cb_in1, src_b, src1_tile_bytes, offs + block.start() + tile_offset, block);
#endif
#ifdef FUSE_GAMMA
            layernorm_dataflow_utils::read_block_to_cb(noc, cb_gamma, addrg, gamma_tile_bytes, block.start(), block);
#endif
#ifdef FUSE_BETA
            layernorm_dataflow_utils::read_block_to_cb(noc, cb_beta, addrb, beta_tile_bytes, block.start(), block);
#endif
        }
        offs += Wt;
    }
}
