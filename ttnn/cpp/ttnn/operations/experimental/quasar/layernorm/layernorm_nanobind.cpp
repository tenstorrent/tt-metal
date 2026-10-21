// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "layernorm.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::quasar::detail {

void bind_layernorm(nb::module_& mod) {
    // LayerNormDefaultProgramConfig / LayerNormShardedMultiCoreProgramConfig are NOT re-registered
    // here: they are bound once on the main ttnn module (normalization/layernorm_nanobind.cpp) and
    // resolve at call time within the same extension module.
    const auto* doc = R"doc(
        Layer norm over :attr:`input_tensor` (Quasar / Metal 2.0, standard Welford path only).
        Same contract as :func:`ttnn.layer_norm`; see that op for the full description.

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword args:
            epsilon (float): Defaults to 1e-12.
            weight (ttnn.Tensor, optional): gamma. Defaults to `None`.
            bias (ttnn.Tensor, optional): beta. Defaults to `None`.
            residual_input_tensor (ttnn.Tensor, optional): Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): Defaults to `None`.
            program_config (ttnn.LayerNormProgramConfig, optional): Defaults to
                `LayerNormDefaultProgramConfig(use_welford=True)`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Defaults to `None`.
            recip_tensor (ttnn.Tensor, optional): reciprocal LUT for the Welford algorithm. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.
        )doc";

    ttnn::bind_function<"layer_norm", "ttnn.experimental.quasar.">(
        mod,
        doc,
        nb::overload_cast<
            const ttnn::Tensor&,
            float,
            const std::optional<const ttnn::Tensor>&,
            const std::optional<const ttnn::Tensor>&,
            const std::optional<const ttnn::Tensor>&,
            const std::optional<ttnn::MemoryConfig>&,
            const std::optional<const ttnn::prim::LayerNormProgramConfig>&,
            std::optional<const ttnn::DeviceComputeKernelConfig>,
            const std::optional<const ttnn::Tensor>&>(&ttnn::operations::experimental::quasar::layer_norm),
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("epsilon") = 1e-12,
        nb::arg("weight") = nb::none(),
        nb::arg("bias") = nb::none(),
        nb::arg("residual_input_tensor") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("program_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("recip_tensor") = nb::none());
}

}  // namespace ttnn::operations::experimental::quasar::detail
