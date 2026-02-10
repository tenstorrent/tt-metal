// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "distributed_mla_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "distributed_mla.hpp"

namespace ttnn::operations::transformer::sdpa_prefill {

void bind_distributed_mla(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Distributed multi-latent attention operation for sequence distributed SDPA.
        This is a boilerplate implementation that determines device order numbers
        along the specified cluster axis for multi-device SDPA computation.

        Args:
            input_tensor (ttnn.Tensor): Input tensor to process.
            cluster_axis (int, optional): The axis on the mesh device to use for distribution. Defaults to `None`.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Output memory configuration.

        Returns:
            ttnn.Tensor: Currently returns a copy of the input tensor. In the full implementation,
                        this would return the distributed SDPA result.

        Example:
            >>> input_tensor = ttnn.from_torch(torch.randn([1, 8, 128, 64]), dtype=ttnn.bfloat16, device=mesh_device)
            >>> output = ttnn.transformer.sdpa_prefill.distributed_mla(input_tensor, cluster_axis=0)
        )doc";

    using OperationType = decltype(ttnn::transformer::sdpa_prefill::distributed_mla);
    ttnn::bind_registered_operation(
        mod,
        ttnn::transformer::sdpa_prefill::distributed_mla,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const std::optional<uint32_t> cluster_axis,
               const std::optional<ttnn::MemoryConfig>& memory_config) {
                return self(input_tensor, cluster_axis, memory_config);
            },
            nb::arg("input_tensor").noconvert(),
            nb::kw_only(),
            nb::arg("cluster_axis") = nb::none(),
            nb::arg("memory_config") = nb::none()});
}

}  // namespace ttnn::operations::transformer::sdpa_prefill
