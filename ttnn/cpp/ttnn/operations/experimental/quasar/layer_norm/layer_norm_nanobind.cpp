// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "layer_norm_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "layernorm.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::quasar::detail {

void bind_layer_norm(nb::module_& mod) {
    const auto* doc = R"doc(
        Layer normalization over the last dimension of :attr:`input_tensor` (Quasar / Metal 2.0 clone of ttnn.layer_norm).

        Args:
            input_tensor (ttnn.Tensor): the input tensor (TILE layout, interleaved or block/width sharded).

        Keyword args:
            epsilon (float): Defaults to 1e-12.
            weight (ttnn.Tensor, optional): gamma. Defaults to `None`.
            bias (ttnn.Tensor, optional): beta. Defaults to `None`.
            residual_input_tensor (ttnn.Tensor, optional): fused pre-add. Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): Defaults to the input's memory config.
            program_config (ttnn.LayerNormDefaultProgramConfig | ttnn.LayerNormShardedMultiCoreProgramConfig, optional)
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional)
            recip_tensor (ttnn.Tensor, optional): reciprocal LUT for the Welford path (unsupported on Quasar).

        Returns:
            ttnn.Tensor: the normalized tensor.
        )doc";

    ttnn::bind_function<"layer_norm", "ttnn.experimental.quasar.">(
        mod,
        doc,
        ttnn::overload_t(
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
            nb::arg("recip_tensor") = nb::none()));
}

}  // namespace ttnn::operations::experimental::quasar::detail
