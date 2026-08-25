// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "decode_gated_delta_rule_nanobind.hpp"
#include "decode_gated_delta_rule.hpp"

#include "ttnn-nanobind/bind_function.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

namespace ttnn::operations::transformer {

void bind_decode_gated_delta_rule(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Fused T=1 (decode step) Gated Delta Rule forward: one reader/compute/writer
        program, one core per head, replacing the ~12-kernel python decode graph.

        Args:
            q (ttnn.Tensor):    [B, 1, H, K] TILE
            k (ttnn.Tensor):    [B, 1, H, K] TILE
            v (ttnn.Tensor):    [B, 1, H, V] TILE
            beta (ttnn.Tensor): [B, 1, H] TILE
            g (ttnn.Tensor):    [B, 1, H] TILE, log-space decay

        Keyword Args:
            scale (float, optional): defaults to K**-0.5.
            initial_state (ttnn.Tensor, optional): [B, H, K, V] TILE, same dtype; zeros if absent.
            inplace_state (bool): default False. When True (requires initial_state),
                new_state is written into initial_state's buffer and initial_state is
                returned as new_state (trace-safe, no allocation).
            memory_config (ttnn.MemoryConfig, optional).

        Returns:
            tuple[ttnn.Tensor, ttnn.Tensor]: o [B,1,H,V] ROW_MAJOR (pass through
                ttnn.to_layout for TILE; each head's [V] stick is one whole DRAM
                page so the writer only issues full-page writes), new_state
                [B,H,K,V] TILE.
        )doc";

    ttnn::bind_function<"decode_gated_delta_rule", "ttnn.transformer.">(
        mod,
        doc,
        &ttnn::transformer::decode_gated_delta_rule,
        nb::arg("q").noconvert(),
        nb::arg("k").noconvert(),
        nb::arg("v").noconvert(),
        nb::arg("beta").noconvert(),
        nb::arg("g").noconvert(),
        nb::kw_only(),
        nb::arg("scale") = nb::none(),
        nb::arg("initial_state") = nb::none(),
        nb::arg("inplace_state") = false,
        nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::operations::transformer
