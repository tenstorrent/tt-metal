// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gated_delta_prefill_query_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/types.hpp"

#include "gated_delta_prefill_query.hpp"

namespace ttnn::operations::experimental::transformer {

void bind_gated_delta_prefill_query(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Gated DeltaNet: prefill the recurrent state over a K/V sequence, then query.

        Runs the gated delta-rule recurrence over the ``seq_len`` K/V tokens starting from
        ``state``, using a per-head per-token decay ``decay`` (g, log-space) and write-strength
        ``gate`` (beta), then applies the single query ``q`` to the final state to emit the
        first decode output token.

        EXPERIMENTAL / SCAFFOLDING: the current kernels wire the data path end-to-end but do
        not yet implement the recurrence (``state'`` is a passthrough copy of ``state``; ``O``
        is a placeholder). The interface and shapes are final.

        Args:
            q (ttnn.Tensor):     [1, 1,  Nk, d]  ROW_MAJOR bf16 — single query token
            k (ttnn.Tensor):     [1, Nk, S,  d]  TILE bf16
            v (ttnn.Tensor):     [1, Nv, S,  d]  TILE bf16
            gate (ttnn.Tensor):  [1, Nv, S,  1]  TILE fp32 — beta (write strength), per V-head per token
            decay (ttnn.Tensor): [1, Nv, S,  1]  TILE fp32 — g (log-space decay), per V-head per token
            state (ttnn.Tensor): [1, Nv, d,  d]  TILE fp32 — recurrent state

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): output memory config.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional).

        Returns:
            tuple[ttnn.Tensor, ttnn.Tensor]:
                O [1, 1, Nv, d] bf16, state' [1, Nv, d, d] fp32
        )doc";

    ttnn::bind_function<"gated_delta_prefill_query", "ttnn.experimental.">(
        mod,
        doc,
        &ttnn::experimental::gated_delta_prefill_query,
        nb::arg("q"),
        nb::arg("k"),
        nb::arg("v"),
        nb::arg("gate"),
        nb::arg("decay"),
        nb::arg("state"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::transformer
