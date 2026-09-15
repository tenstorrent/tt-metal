// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ltx_rope_materialize_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>

#include "ltx_rope_materialize.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::transformer {

void bind_ltx_rope_materialize(nanobind::module_& mod) {
    ttnn::bind_function<"ltx_rope_materialize", "ttnn.experimental.">(
        mod,
        R"doc(
            Materialize LTX video self-RoPE and temporal cross-PE into caller-owned mesh tensors.

            The compact self tables have shape ``[1, 3, positions, rates]`` and the compact
            cross tables have shape ``[1, 1, positions, rates]``. All compact tables are FP32
            TILE tensors replicated across the mesh. ``metadata`` is a fixed-address UINT32
            ROW_MAJOR DRAM tensor containing ``[video_N_real, frames, height, width]``.

            Outputs are caller-supplied BF16 TILE tensors distributed over sequence on
            ``sp_axis`` and heads on ``tp_axis``. The operation updates all four tensors in
            place and returns their handles as ``(self_cos, self_sin, cross_cos, cross_sin)``.
        )doc",
        &ttnn::experimental::ltx_rope_materialize,
        nanobind::arg("compact_self_cos").noconvert(),
        nanobind::arg("compact_self_sin").noconvert(),
        nanobind::arg("compact_cross_cos").noconvert(),
        nanobind::arg("compact_cross_sin").noconvert(),
        nanobind::arg("metadata").noconvert(),
        nanobind::arg("self_cos_output").noconvert(),
        nanobind::arg("self_sin_output").noconvert(),
        nanobind::arg("cross_cos_output").noconvert(),
        nanobind::arg("cross_sin_output").noconvert(),
        nanobind::kw_only(),
        nanobind::arg("sp_axis"),
        nanobind::arg("tp_axis"));
}

}  // namespace ttnn::operations::experimental::transformer
