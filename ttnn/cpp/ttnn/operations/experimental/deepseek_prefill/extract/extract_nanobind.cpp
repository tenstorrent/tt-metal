// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "extract_nanobind.hpp"

#include <cstdint>

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "extract.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::extract::detail {

void bind_extract(nb::module_& mod) {
    ttnn::bind_function<"extract", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            Extract operation for DeepSeek prefill MoE.

            Each device independently looks up its global expert id via
            global_expert_id = global_expert_idx_table[local_expert_id]
            (all device-resident; no host round-trip), reads start[global_expert_id] and
            counts[global_expert_id] from its own DRAM, and copies
            global_tensor[start : start + ceil_tile(counts), :] into the first rows/tokens
            of the output tensor.

            The output has shape [max_dispatched_tokens_per_expert, hidden_dim]; rows/tokens beyond
            ceil_tile(counts[global_expert_id]) are left uninitialized.

            Args:
                global_tensor (ttnn.Tensor): 2D BFLOAT8_B TILE-layout DRAM interleaved tensor.
                start (ttnn.Tensor): 1D tensor (or 2D with first dim == 1) of UINT32,
                    DRAM interleaved, giving the per-expert starting row/token offsets.
                counts (ttnn.Tensor): 1D tensor (or 2D with first dim == 1) of UINT32,
                    DRAM interleaved, giving the per-expert row/token counts.
                global_expert_idx_table (ttnn.Tensor): 1D tensor (or 2D with first dim == 1) of
                    UINT32, DRAM interleaved, mapping local_expert_id -> global_expert_id.
                local_expert_id (int): UINT32 scalar index into global_expert_idx_table.
                max_dispatched_tokens_per_expert (int): Host-known upper bound on the number of
                    rows/tokens that can be extracted per expert. Must be a multiple of TILE_HEIGHT (32).
                    Defines the output tensor's row/token dimension.

            Returns:
                ttnn.Tensor: [max_dispatched_tokens_per_expert, hidden_dim] BFLOAT8_B TILE-layout
                    DRAM tensor. Valid rows/tokens are in [0, ceil_tile(counts[global_expert_id])).
        )doc",
        &ttnn::operations::experimental::deepseek_prefill::extract::extract,
        nb::arg("global_tensor").noconvert(),
        nb::arg("start").noconvert(),
        nb::arg("counts").noconvert(),
        nb::arg("global_expert_idx_table").noconvert(),
        nb::kw_only(),
        nb::arg("local_expert_id"),
        nb::arg("max_dispatched_tokens_per_expert"));
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::extract::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {

void bind_extract(::nanobind::module_& mod) { extract::detail::bind_extract(mod); }

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
