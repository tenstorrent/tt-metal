// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_tail_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/moe_block/common/moe_block_resources.hpp"

#include "moe_tail.hpp"

namespace ttnn::operations::experimental::moe_block::moe_tail {

void bind_moe_tail(nb::module_& python_module) {
    ttnn::bind_function<"moe_tail", "ttnn.experimental.moe_block.">(
        python_module,
        R"doc(
        The MoE block's tail as one program a chip: the combine's owner rows (+ the shared expert's output) to the
        residual's layout [1, 1, S/SP, H/TP] (a mesh row's tokens, this chip's width slice). The TP group is the mesh
        row; a ring when the fabric wraps it.

        A latent model (resources.dims.latent_dim set): the latent all-gather over TP into the engine's L1, the own
        rows tilized there, out = rmsnorm(z) @ W_up[:, chip's columns] + shared_expert (the norm folded: its weight in
        w_up, 1 / rms applied to the fp32 sums). Otherwise: out = the TP all-to-all of tilize(y_own) + shared_expert.

        Args:
            y_own: the combine's owner rows [1, 1, S/N, Hr], bfloat16 ROW_MAJOR, DRAM interleaved.
            shared_expert: bfloat16 TILE, DRAM interleaved: [1, 1, S/SP, H/TP] with a latent (required),
                [1, 1, S/N, H] without (or None).

        Keyword args:
            resources: the block's MoEBlockResources (persistent words, plans; created once a mesh).
            row_blocks: the head's row blocks (MoEHeadOutputs.row_blocks; required - 1 for a tail without a head):
                y_own, shared_expert and the output hold that many blocks' rows, block after block (the owner rows
                in the head's block layout); one program a chip whose kernels loop over the blocks.
            w_up: a latent model's prepare_moe_tail_weights (required with a latent, None without).
            math_fidelity: the up projection's fidelity (HiFi2; never LoFi). The squares' sums run at HiFi4.
            output: a preallocated output of the result's spec.
            actual_start, actual_end, metadata: a padded forward's span, as moe_head's (one chunk's scalars, or the
                forward's metadata words). Without a latent the padding row tiles are skipped: their output rows are
                zero. The latent engine computes every row.
            split: the call's piece of the chunk (MoEBlockConfig.chunk_splits), as moe_head's; with a span only.

        Returns:
            [1, 1, S/SP, H/TP] bfloat16 TILE, DRAM interleaved.

        The program is cached by structure (resources.plan_key, shapes, fidelity): a repeated call patches only the
        tensors' addresses. Persistent counters only count up or are re-armed by their waiters, so traced replays
        need no reset.
        )doc",
        &moe_tail,
        nb::arg("y_own"),
        nb::arg("shared_expert"),
        nb::kw_only(),
        nb::arg("resources"),
        nb::arg("row_blocks"),
        nb::arg("w_up") = nb::none(),
        nb::arg("math_fidelity") = tt::tt_metal::MathFidelity::HiFi2,
        nb::arg("output") = nb::none(),
        nb::arg("actual_start") = nb::none(),
        nb::arg("actual_end") = nb::none(),
        nb::arg("metadata") = nb::none(),
        nb::arg("split") = 0);

    python_module.def(
        "plan_moe_tail",
        &plan_moe_tail,
        nb::arg("resources"),
        nb::kw_only(),
        nb::arg("row_blocks"),
        nb::arg("dtype") = DataType::BFLOAT8_B,
        "Host only: raises when the tail of a call (a split's rows as row_blocks blocks: the head's) does not fit "
        "its L1 (a latent model's engine for W_up tiles of dtype, else the all-to-all); plans uncached.");
    ttnn::bind_function<"prepare_moe_tail_weights", "ttnn.experimental.moe_block.">(
        python_module,
        R"doc(
        A latent model's up projection for moe_tail, on device: latent_up [H, Hr] (host, float32 or bfloat16)
        split by output columns over TP (chip (r, c): columns [c H/TP, (c + 1) H/TP)), the rmsnorm weight latent_norm
        [Hr] folded in (W' = diag(g) W_up^T), laid out as the engine reads it (K-range regions of each engine column,
        one DRAM burst each). Upload once a layer and keep it.

        Args:
            latent_up: [H, Hr] host tensor.
            latent_norm: [Hr] host tensor.

        Keyword args:
            resources: the block's MoEBlockResources.
            row_blocks: the head's row blocks the tail will run with (HeadFit.row_blocks / MoEHeadOutputs.row_blocks;
                1 for a tail without a head): the engine's plan the layout follows.
            dtype: bfloat8_b (default) or bfloat16.
        )doc",
        &prepare_moe_tail_weights,
        nb::arg("latent_up"),
        nb::arg("latent_norm"),
        nb::kw_only(),
        nb::arg("resources"),
        nb::arg("row_blocks"),
        nb::arg("dtype") = tt::tt_metal::DataType::BFLOAT8_B);
}

}  // namespace ttnn::operations::experimental::moe_block::moe_tail
