// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ttnn.experimental.moe_block.moe_tail: the MoE block's tail as one program a chip, out [S/SP, H/TP] (a mesh row's
// tokens, this chip's width slice):
//  * a latent model (dims.latent_dim, e.g. Kimi-K3): TailEngine - the latent all-gather over TP into the engine's L1,
//    the combine's own rows tilized there, rmsnorm (folded: its weight in W_up, 1 / rms on the fp32 sums), the up
//    projection split by columns, + the shared expert's output;
//  * otherwise: the all-to-all - the combine's own rows tilized, + the shared expert's owner rows, the TP all-to-all
//  straight
//    into [S/SP, H/TP].

#pragma once

#include <optional>

#include <tt-metalium/base_types.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::moe_block {
class MoEBlockResources;
}

namespace ttnn::operations::experimental::moe_block::moe_tail {

// y_own: the combine's owner rows [1, 1, S/N, Hr] (bf16 ROW_MAJOR, DRAM). shared_expert: the shared expert's output,
// [1, 1, S/SP, H/TP] with a latent, [1, 1, S/N, H] (or none) without (bf16 TILE, DRAM). w_up: a latent model's
// prepare_moe_tail_weights. Returns [1, 1, S/SP, H/TP] bf16 TILE (in `output` when given).
// row_blocks: the head's row blocks (MoEHeadOutputs.row_blocks - required, the caller passes the head's: 1 for a
// tail without a head): y_own, shared_expert and the output hold that many blocks' rows, block after block - the owner
// rows in the head's block layout; one program for the mesh whose kernels loop over the blocks. split: the call's piece
// of the chunk when MoEBlock runs it as MoEBlockConfig.chunk_splits pieces (a padded forward's real rows, as
// moe_head's).
Tensor moe_tail(
    const Tensor& y_own,
    const std::optional<Tensor>& shared_expert,
    MoEBlockResources& resources,
    uint32_t row_blocks,
    const std::optional<Tensor>& w_up = std::nullopt,
    tt::tt_metal::MathFidelity math_fidelity = tt::tt_metal::MathFidelity::HiFi2,
    const std::optional<Tensor>& output = std::nullopt,
    const std::optional<Tensor>& actual_start = std::nullopt,
    const std::optional<Tensor>& actual_end = std::nullopt,
    const std::optional<Tensor>& metadata = std::nullopt,
    uint32_t split = 0);

// A latent model's up projection in the engine's layout: latent_up [H, Hr] (host, float32 or bfloat16) column-split
// over TP (chip (r, c) takes output columns [c H/TP, (c + 1) H/TP)), the rmsnorm weight latent_norm [Hr] folded in
// (W' = diag(g) W_up^T), as K-range regions of each engine column spread over the DRAM banks. dtype: bfloat8_b (the
// reference's) or bfloat16.
// Host only: does the tail of a call (a split's rows as the head's row_blocks blocks) fit its L1? Raises (TT_FATAL)
// with why not. Plans uncached (the call's plan is made by its first call / weight preparation): a latent model's
// engine for W_up tiles of `dtype`, else the all-to-all.
void plan_moe_tail(MoEBlockResources& resources, uint32_t row_blocks, DataType dtype = DataType::BFLOAT8_B);

Tensor prepare_moe_tail_weights(
    const Tensor& latent_up,
    const Tensor& latent_norm,
    MoEBlockResources& resources,
    uint32_t row_blocks,
    DataType dtype = DataType::BFLOAT8_B);

}  // namespace ttnn::operations::experimental::moe_block::moe_tail
