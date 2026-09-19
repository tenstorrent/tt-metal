// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"

namespace ckl = compute_kernel_lib;

ALWI void ACQ() {
    tile_regs_acquire();
    tile_regs_wait();
}
ALWI void REL() {
    tile_regs_commit();
    tile_regs_release();
}

void kernel_main() {
    auto batch_start = get_arg(args::batch_start);
    auto batch_end = get_arg(args::batch_end);
    auto seq_t_start = get_arg(args::seq_t_start);
    auto seq_t_end = get_arg(args::seq_t_end);

    constexpr uint32_t onetile = 1;
    // Magic CB indices are gone: each buffer is a named DFB binding. The local
    // aliases keep the LLK/FIFO call sites readable; each is the dfb:: handle,
    // which converts implicitly to a CB id at the LLK call sites.
    constexpr auto in_dfb = dfb::input;
    constexpr auto cos_dfb = dfb::cos;
    constexpr auto sin_dfb = dfb::sin;
    constexpr auto trans_mat_dfb = dfb::trans_mat;

    constexpr auto rotated_in_interm_dfb = dfb::rotated_interm;
    constexpr auto cos_interm_dfb = dfb::cos_interm;
    constexpr auto sin_interm_dfb = dfb::sin_interm;
    constexpr auto out_dfb = dfb::out;
    constexpr auto Wt = get_arg(args::Wt);
    // Indexed RoPE supplies a per-core runtime head count; other callers keep a compile-time count.
    const auto n_heads = get_arg(args::n_heads);
    constexpr auto rotary_Ht = get_arg(args::rotary_Ht);
    constexpr auto bulk_block_input = [](auto dfb_id) {
        return ckl::input(
            dfb_id,
            ckl::WaitPolicy::Upfront,
            ckl::PopPolicy::AtEnd,
            ckl::InputTileMapping::Block,
            ckl::DataFormatReconfig::Disabled);
    };
    constexpr auto bulk_output = [](auto dfb_id) {
        return ckl::output(dfb_id, ckl::ReservePolicy::None, ckl::PushPolicy::AtEnd, ckl::DataFormatReconfig::Disabled);
    };
    constexpr auto sin_cos_input = [](auto dfb_id) {
        return ckl::input(
            dfb_id,
            RELOAD_IMPL == 0 ? ckl::WaitPolicy::None : ckl::WaitPolicy::Upfront,
            RELOAD_IMPL == 0 ? ckl::PopPolicy::None : ckl::PopPolicy::AtEnd,
            ckl::InputTileMapping::Block,
            ckl::DataFormatReconfig::Disabled,
            RELOAD_IMPL == 0 ? ckl::TileAddressing::Offset : ckl::TileAddressing::Direct);
    };
#ifdef FUSE_RMS
    // x <- x * rsqrt(mean_W(x^2) + eps) per row before the rotation; the rope math then reads xn instead of the input.
    constexpr auto xx_dfb = dfb::xx;
    constexpr auto ex2pe_dfb = dfb::ex2pe;
    constexpr auto xn_dfb = dfb::xn;
    constexpr auto scaler_dfb = dfb::scaler;
    constexpr uint32_t rms_eps_bits = get_arg(args::rms_eps_bits);
    constexpr auto x_dfb = xn_dfb;
    DataflowBuffer xx_dfb_obj(xx_dfb);
    DataflowBuffer ex2pe_dfb_obj(ex2pe_dfb);
    DataflowBuffer xn_dfb_obj(xn_dfb);
    DataflowBuffer scaler_dfb_obj(scaler_dfb);
#else
    constexpr auto x_dfb = in_dfb;
#endif

    DataflowBuffer in_dfb_obj(in_dfb);
    DataflowBuffer cos_dfb_obj(cos_dfb);
    DataflowBuffer sin_dfb_obj(sin_dfb);
    DataflowBuffer trans_mat_dfb_obj(trans_mat_dfb);
    DataflowBuffer rotated_in_interm_dfb_obj(rotated_in_interm_dfb);
    DataflowBuffer cos_interm_dfb_obj(cos_interm_dfb);
    DataflowBuffer sin_interm_dfb_obj(sin_interm_dfb);
    DataflowBuffer out_dfb_obj(out_dfb);

    const uint32_t rotary_seq_t_end = seq_t_end < rotary_Ht ? seq_t_end : rotary_Ht;
    const uint32_t my_rotary_seq_tiles = seq_t_start < rotary_seq_t_end ? rotary_seq_t_end - seq_t_start : 0;
    const uint32_t my_cos_sin_tiles = my_rotary_seq_tiles * Wt;

    compute_kernel_hw_startup<SrcOrder::Reverse>(in_dfb, trans_mat_dfb, out_dfb);
    // Start from the state at the end of each iteration so same-format reconfigurations compile out.
    // TODO(#52395): compute_kernel_hw_startup is a call-once API and should be the kernel's first Tensix-engine call,
    // but here it follows another engine op (init_sfpu / a prior startup); see the issue.
    compute_kernel_hw_startup(cos_interm_dfb, sin_interm_dfb, out_dfb);

    // Get the trans_mat
    trans_mat_dfb_obj.wait_front(onetile);
#ifdef FUSE_RMS
    scaler_dfb_obj.wait_front(onetile);
#endif

    uint32_t in0_index = 0;
    uint32_t in1_index = 0;
    uint32_t interm_index = 0;

    for (uint32_t batch_id = batch_start; batch_id < batch_end; ++batch_id) {
#if RELOAD_IMPL == 0
        if (my_cos_sin_tiles > 0) {
            sin_dfb_obj.wait_front(my_cos_sin_tiles);
            cos_dfb_obj.wait_front(my_cos_sin_tiles);
        }
#endif
        for (uint32_t head_num = 0; head_num < n_heads; ++head_num) {
            uint32_t sin_cos_row_cnt = 0;
            for (uint32_t seq_tile = seq_t_start; seq_tile < rotary_seq_t_end; ++seq_tile) {
                // input cb wait and reserve
                in_dfb_obj.wait_front(Wt);
#if RELOAD_IMPL == 1
                sin_dfb_obj.wait_front(Wt);
                cos_dfb_obj.wait_front(Wt);
#endif

                rotated_in_interm_dfb_obj.reserve_back(Wt);
                sin_interm_dfb_obj.reserve_back(Wt);
                cos_interm_dfb_obj.reserve_back(Wt);
                out_dfb_obj.reserve_back(Wt);

#ifdef FUSE_RMS
                // (1) xx = x * x (fp32 tiles)
                reconfig_data_format(in_dfb, in_dfb);
                pack_reconfig_data_format(xx_dfb);
                xx_dfb_obj.reserve_back(Wt);
                mul_init(in_dfb, in_dfb);
                ACQ();
                for (uint32_t j = 0; j < Wt; ++j) {
                    mul_tiles(in_dfb, in_dfb, j, j, j);
                    pack_tile(j, xx_dfb, j);
                }
                REL();
                xx_dfb_obj.push_back(Wt);
                xx_dfb_obj.wait_front(Wt);
                // (2) mean over W (reduce with the 1/W scaler; REDUCE_ROW SUM wants scaler as SrcA), + eps, rsqrt
                reconfig_data_format(scaler_dfb, xx_dfb);
                pack_reconfig_data_format(ex2pe_dfb);
                ex2pe_dfb_obj.reserve_back(onetile);
                reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(xx_dfb, scaler_dfb, ex2pe_dfb);
                ACQ();
                for (uint32_t j = 0; j < Wt; ++j) {
                    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(xx_dfb, scaler_dfb, j, 0, 0);
                }
                reduce_uninit();
                binop_with_scalar_tile_init();
                add_unary_tile(0, rms_eps_bits);
                rsqrt_tile_init();
                rsqrt_tile(0);
                pack_tile(0, ex2pe_dfb, 0);
                REL();
                xx_dfb_obj.pop_front(Wt);
                ex2pe_dfb_obj.push_back(onetile);
                ex2pe_dfb_obj.wait_front(onetile);
                // (3) xn = x * scale (the scale sits in column 0 of every row: broadcast over columns)
                reconfig_data_format(in_dfb, ex2pe_dfb);
                pack_reconfig_data_format(xn_dfb);
                xn_dfb_obj.reserve_back(Wt);
                mul_bcast_cols_init(in_dfb, ex2pe_dfb);
                ACQ();
                for (uint32_t j = 0; j < Wt; ++j) {
                    mul_tiles_bcast_cols(in_dfb, ex2pe_dfb, j, 0, j);
                    pack_tile(j, xn_dfb, j);
                }
                REL();
                xn_dfb_obj.push_back(Wt);
                xn_dfb_obj.wait_front(Wt);
                ex2pe_dfb_obj.pop_front(onetile);
                in_dfb_obj.pop_front(Wt);  // the rope math below reads xn
                // Leave the unpacker in the state the rope math's first reconfig assumes (trans_mat SrcA, x SrcB).
                reconfig_data_format(trans_mat_dfb, x_dfb);
#endif

                // // rotated = x @ trans_mat
                // Matmul uses SrcOrder::Reverse: trans_mat is SrcA and input is SrcB.
                reconfig_data_format(cos_interm_dfb, trans_mat_dfb, sin_interm_dfb, x_dfb);
                pack_reconfig_data_format(out_dfb, rotated_in_interm_dfb);
                matmul_init(x_dfb, trans_mat_dfb);
                ACQ();
                for (uint32_t j = 0; j < Wt; ++j) {
                    matmul_tiles(x_dfb, trans_mat_dfb, j, in1_index, j);
                    pack_tile(j, rotated_in_interm_dfb, j);
                }
                REL();
                rotated_in_interm_dfb_obj.push_back(Wt);
                reconfig_data_format(trans_mat_dfb, rotated_in_interm_dfb, x_dfb, sin_dfb);
                pack_reconfig_data_format(rotated_in_interm_dfb, sin_interm_dfb);
                mul_init(rotated_in_interm_dfb, sin_dfb);
                // sin_interim = rotated * sin
                ckl::eltwise_chain<ckl::InitReconfigOwner::Caller>(
                    ckl::IterationShape::tiles(Wt).block_size(/*block_size=*/Wt),
                    ckl::BinaryFpu<
                        ckl::BinaryFpuOp::Mul,
                        bulk_block_input(rotated_in_interm_dfb),
                        sin_cos_input(sin_dfb)>{0u, sin_cos_row_cnt * Wt},
                    ckl::PackTile<bulk_output(sin_interm_dfb)>{});

                reconfig_data_format(rotated_in_interm_dfb, x_dfb, sin_dfb, cos_dfb);
                pack_reconfig_data_format(sin_interm_dfb, cos_interm_dfb);
                // cos_interim = x * cos
                ckl::eltwise_chain<ckl::InitReconfigOwner::Caller>(
                    ckl::IterationShape::tiles(Wt).block_size(/*block_size=*/Wt),
                    ckl::BinaryFpu<
                        ckl::BinaryFpuOp::Mul,
                        ckl::input(
                            x_dfb,
                            ckl::WaitPolicy::None,
                            ckl::PopPolicy::AtEnd,
                            ckl::InputTileMapping::Block,
                            ckl::DataFormatReconfig::Disabled),
                        sin_cos_input(cos_dfb)>{0u, sin_cos_row_cnt * Wt},
                    ckl::PackTile<bulk_output(cos_interm_dfb)>{});

                reconfig_data_format(in_dfb, cos_interm_dfb, cos_dfb, sin_interm_dfb);
                pack_reconfig_data_format(cos_interm_dfb, out_dfb);
                // out = cos_interim + sin_interim
                ckl::add<bulk_block_input(cos_interm_dfb), bulk_block_input(sin_interm_dfb), bulk_output(out_dfb)>(
                    ckl::IterationShape::tiles(Wt).block_size(/*block_size=*/Wt));

#if RELOAD_IMPL == 0
                // no-reload needs to increment this counter
                // Used a sin/cos row
                sin_cos_row_cnt++;
#endif
            }
        }

#if RELOAD_IMPL == 0
        if (my_cos_sin_tiles > 0) {
            sin_dfb_obj.pop_front(my_cos_sin_tiles);
            cos_dfb_obj.pop_front(my_cos_sin_tiles);
        }
#endif
    }

    // Done with the transformation matrix, so remove from CB
    trans_mat_dfb_obj.pop_front(onetile);
#ifdef FUSE_RMS
    scaler_dfb_obj.pop_front(onetile);
#endif
}
