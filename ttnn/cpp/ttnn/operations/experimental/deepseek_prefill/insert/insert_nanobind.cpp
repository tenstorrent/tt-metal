// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "insert_nanobind.hpp"

#include <cstdint>

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "insert.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::insert::detail {

void bind_insert(nb::module_& mod) {
    ttnn::bind_function<"insert", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            Insert operation for DeepSeek prefill MoE — inverse of extract.

            Each device independently looks up its global expert id via
            global_expert_id = global_expert_idx_table[local_expert_id]
            (all device-resident; no host round-trip), reads start[global_expert_id] and
            counts[global_expert_id] from its own DRAM, and copies the first
            ceil_tile(counts[global_expert_id]) rows/tokens of local_tensor into
            global_tensor[start : start + ceil_tile(counts), :].

            The op writes in place into global_tensor and returns a handle to
            that same tensor (no new allocation).

            Args:
                global_tensor (ttnn.Tensor): 2D BFLOAT8_B TILE-layout DRAM interleaved
                    tensor that the slice is written into.
                local_tensor (ttnn.Tensor): 2D BFLOAT8_B TILE-layout DRAM interleaved
                    tensor providing the source rows/tokens. Must have the same hidden_dim
                    as global_tensor.
                start (ttnn.Tensor): 1D tensor (or 2D with first dim == 1) of UINT32,
                    DRAM interleaved, giving the per-expert starting row/token offsets in
                    global_tensor.
                counts (ttnn.Tensor): 1D tensor (or 2D with first dim == 1) of UINT32,
                    DRAM interleaved, giving the per-expert row/token counts.
                global_expert_idx_table (ttnn.Tensor): 1D tensor (or 2D with first dim == 1)
                    of UINT32, DRAM interleaved, mapping local_expert_id -> global_expert_id.
                local_expert_id (int): UINT32 scalar index into global_expert_idx_table.

            Returns:
                ttnn.Tensor: The same global_tensor handle, now with the slice
                    written into [start, start + ceil_tile(counts)).
        )doc",
        &ttnn::operations::experimental::deepseek_prefill::insert::insert,
        nb::arg("global_tensor").noconvert(),
        nb::arg("local_tensor").noconvert(),
        nb::arg("start").noconvert(),
        nb::arg("counts").noconvert(),
        nb::arg("global_expert_idx_table").noconvert(),
        nb::kw_only(),
        nb::arg("local_expert_id"));
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::insert::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {

void bind_insert(::nanobind::module_& mod) { insert::detail::bind_insert(mod); }

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
