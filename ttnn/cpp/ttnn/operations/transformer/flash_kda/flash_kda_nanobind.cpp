// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/flash_kda/flash_kda_nanobind.hpp"
#include "ttnn/operations/transformer/flash_kda/flash_kda.hpp"
#include "ttnn-nanobind/bind_function.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

namespace ttnn::operations::transformer {

void bind_flash_kda(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Flash KDA (Kimi Delta Attention) recurrent state update — single step, one core per item.

        All inputs must be float32. Computes, per item:
            S_tilde = S_prev * g            (g varies per key-dim row)
            pred    = k @ S_tilde
            err     = v - pred
            delta   = beta * err
            S_new   = S_tilde + (k outer delta)
            out     = q @ S_new

        Args:
            S_prev (ttnn.Tensor):  [N, Dk, Dv]  previous recurrent state
            g (ttnn.Tensor):       [N, Dk, 1]   per-key-dim-row decay (column layout)
            k (ttnn.Tensor):       [N, 1, Dk]   key vector (row layout)
            v (ttnn.Tensor):       [N, 1, Dv]   value vector (row layout)
            beta (ttnn.Tensor):    [N, 1, 1]    per-item scalar gate
            q (ttnn.Tensor):       [N, 1, Dk]   query vector (row layout)

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): output memory config.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional).

        Returns:
            tuple[ttnn.Tensor, ttnn.Tensor]:
                S_new [N, Dk, Dv], out [N, 1, Dv]
        )doc";

    ttnn::bind_function<"flash_kda", "ttnn.transformer.">(
        mod,
        doc,
        &ttnn::transformer::flash_kda,
        nb::arg("S_prev").noconvert(),
        nb::arg("g").noconvert(),
        nb::arg("k").noconvert(),
        nb::arg("v").noconvert(),
        nb::arg("beta").noconvert(),
        nb::arg("q").noconvert(),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}

}  // namespace ttnn::operations::transformer
