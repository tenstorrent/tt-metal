// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_concat_heads_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "nlp_concat_heads.hpp"

namespace ttnn::operations::bringup::nlp_concat_heads::detail {

void bind_nlp_concat_heads(nb::module_& mod) {
    ttnn::bind_function<"nlp_concat_heads", "ttnn.bringup.">(
        mod,
        R"doc(
            Shuffles [B, num_heads, S, head_dim] tensor into tensor with shape [B, 1, S, num_heads * head_dim].
            Fork of ttnn.experimental.nlp_concat_heads (ttnn/ttnn/bringup/nlp_concat_heads).

            max_grid (ttnn.CoreCoord, optional): bring-up option, default None (the device's whole compute grid,
                the source op). An interleaved input's work split runs on the rectangle (0, 0) .. (max_grid - 1)
                of the grid (deepseek_v41_flash_d_p FIX.15: a Galaxy's 110-core cap).
        )doc",
        &ttnn::experimental::bringup::nlp_concat_heads,
        nb::arg("input_tensor").noconvert(),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("max_grid") = nb::none());
}

}  // namespace ttnn::operations::bringup::nlp_concat_heads::detail
