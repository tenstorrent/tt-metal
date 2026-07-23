// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/kda_recurrent/kda_recurrent_nanobind.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/transformer/kda_recurrent/kda_recurrent.hpp"

namespace ttnn::operations::transformer {

void bind_kda_recurrent_step(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Execute one fused recurrent Kimi Delta Attention state step.

        Inputs are pre-normalized FP32 tiled tensors. ``decay`` is the
        exponentiated vector gate in column layout, so each K element owns one
        recurrent-state row.

        Args:
            q_scaled (ttnn.Tensor): [BH, 1, K] normalized and scaled query.
            k_unit (ttnn.Tensor): [BH, 1, K] normalized key.
            v (ttnn.Tensor): [BH, 1, V] value.
            decay (ttnn.Tensor): [BH, K, 1] exponentiated vector decay.
            beta (ttnn.Tensor): [BH, 1, 1] write strength.
            state (ttnn.Tensor): [BH, K, V] recurrent state.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): output memory config.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional).

        Returns:
            tuple[ttnn.Tensor, ttnn.Tensor]: output [BH, 1, V] and final state [BH, K, V].
        )doc";

    ttnn::bind_function<"kda_recurrent_step", "ttnn.transformer.">(
        mod,
        doc,
        &ttnn::transformer::kda_recurrent_step,
        nb::arg("q_scaled").noconvert(),
        nb::arg("k_unit").noconvert(),
        nb::arg("v").noconvert(),
        nb::arg("decay").noconvert(),
        nb::arg("beta").noconvert(),
        nb::arg("state").noconvert(),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}

}  // namespace ttnn::operations::transformer
