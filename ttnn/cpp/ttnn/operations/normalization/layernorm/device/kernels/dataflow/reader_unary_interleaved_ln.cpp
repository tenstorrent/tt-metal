// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "ttnn/cpp/ttnn/operations/normalization/kernel_util/dataflow/custom_tiles.h"
#include "layernorm_dataflow_utils.h"

namespace generic = norm::kernel_util::generic;
namespace layernorm_dataflow_utils = norm::layernorm::device::kernels::dataflow;

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t NCHt = get_arg_val<uint32_t>(1);
    uint32_t Wt = get_arg_val<uint32_t>(2);
    uint32_t tile_offset = get_arg_val<uint32_t>(3);
    uint32_t gamma_addr = get_arg_val<uint32_t>(6);
    uint32_t beta_addr = get_arg_val<uint32_t>(7);
    uint32_t b_addr = get_arg_val<uint32_t>(8);
    uint32_t W = get_arg_val<uint32_t>(9);

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0, cb_id_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_id_gamma = tt::CBIndex::c_5;
    constexpr uint32_t cb_id_beta = tt::CBIndex::c_6;

    // ublocks size defined in tiles
    const uint32_t src0_tile_bytes = get_tile_size(cb_id_in0);

    constexpr uint32_t blk = get_compile_time_arg_val(0);  // needed for correctness of softmax/LN kernels
    constexpr bool use_welford = get_compile_time_arg_val(1) == 1;
    constexpr auto src0_args = TensorAccessorArgs<2>();
    [[maybe_unused]] constexpr auto src1_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<src1_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

    const auto src_a = TensorAccessor(src0_args, src_addr, src0_tile_bytes);
#ifdef FUSE_GAMMA
    const uint32_t gamma_tile_bytes = get_tile_size(cb_id_gamma);
    const auto addrg = TensorAccessor(gamma_args, gamma_addr, gamma_tile_bytes);
#endif
#ifdef FUSE_BETA
    const uint32_t beta_tile_bytes = get_tile_size(cb_id_beta);
    const auto addrb = TensorAccessor(beta_args, beta_addr, beta_tile_bytes);
#endif
#ifdef FUSE_PRE_ADD
    const uint32_t src1_tile_bytes = get_tile_size(cb_id_in1);
    const auto src_b = TensorAccessor(src1_args, b_addr, src1_tile_bytes);
#endif

    // Generate constant tiles for layernorm compute
    if constexpr (!use_welford) {
        // Scaler(s) for reduce
        constexpr uint32_t cb_in_2 = tt::CBIndex::c_2;
        uint32_t scaler = get_arg_val<uint32_t>(4);
        generate_reduce_scaler(cb_in_2, scaler);
        const auto partial_last_tile_cols = W % tt::constants::TILE_WIDTH;
        if (partial_last_tile_cols > 0 && !use_welford) {
            norm::kernel_util::dataflow::generate_partial_reduce_scaler(cb_in_2, scaler, partial_last_tile_cols);
        }
    }

    constexpr uint32_t eps_cb_id = 3;
    const uint32_t eps = get_arg_val<uint32_t>(5);
    generate_bcast_col_scalar(eps_cb_id, eps);

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    uint32_t offs = 0;
    auto read_in0_and_in1 = [&]() {
        for (auto block : generic::blocks(Wt, blk)) {
            const auto total_offset = offs + block.start() + tile_offset;
            layernorm_dataflow_utils::read_block_to_cb(cb_id_in0, src_a, src0_tile_bytes, total_offset, block);
#ifdef FUSE_PRE_ADD
            layernorm_dataflow_utils::read_block_to_cb(cb_id_in1, src_b, src1_tile_bytes, total_offset, block);
#endif
        }  // wt loop
    };
    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        read_in0_and_in1();
#if defined FUSE_GAMMA || defined FUSE_BETA
        if (ncht == 0) {
            for (auto block : generic::blocks(Wt, blk)) {
#ifdef FUSE_GAMMA
                {
                    layernorm_dataflow_utils::read_block_to_cb(
                        cb_id_gamma, addrg, gamma_tile_bytes, block.start(), block);
                }
#endif

#ifdef FUSE_BETA
                {
                    layernorm_dataflow_utils::read_block_to_cb(
                        cb_id_beta, addrb, beta_tile_bytes, block.start(), block);
                }
#endif
            }  // wt loop
        }
#endif
        offs += Wt;
    }  // ncht loop
}
