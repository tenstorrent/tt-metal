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

        Q is split along sequence across one axis of devices, K and V remain full sequence length.
        The operation automatically generates causal mask and skips compute where possible.

        Args:
            q_tensor (ttnn.Tensor): Query tensor [B, NH_Q, S_Q, DH] (sharded per device).
            k_tensor (ttnn.Tensor): Key tensor [B, NH_KV, S_KV, DH] (full sequence).
            v_tensor (ttnn.Tensor): Value tensor [B, NH_KV, S_KV, DH] (full sequence).

        Keyword Args:
            cluster_axis (int, optional): The axis on the mesh device to use for distribution. Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): Output memory configuration.
            scale (float, optional): Scaling factor for attention scores. Defaults to 1/sqrt(head_dim).

        Returns:
            ttnn.Tensor: Distributed attention output [B, NH_Q, S_Q, DH] (sharded per device).

        Example:
            >>> q = ttnn.from_torch(torch.randn([1, 8, 128, 64]), dtype=ttnn.bfloat16, device=mesh_device)
            >>> k = ttnn.from_torch(torch.randn([1, 8, 256, 64]), dtype=ttnn.bfloat16, device=mesh_device)
            >>> v = ttnn.from_torch(torch.randn([1, 8, 256, 64]), dtype=ttnn.bfloat16, device=mesh_device)
            >>> output = ttnn.transformer.sdpa_prefill.distributed_mla(q, k, v, cluster_axis=0)
        )doc";

    using OperationType = decltype(ttnn::transformer::sdpa_prefill::distributed_mla);
    ttnn::bind_registered_operation(
        mod,
        ttnn::transformer::sdpa_prefill::distributed_mla,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& q_tensor,
               const ttnn::Tensor& k_tensor,
               const ttnn::Tensor& v_tensor,
               const std::optional<uint32_t> cluster_axis,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               std::optional<float> scale) {
                return self(q_tensor, k_tensor, v_tensor, cluster_axis, memory_config, scale);
            },
            nb::arg("q_tensor").noconvert(),
            nb::arg("k_tensor").noconvert(),
            nb::arg("v_tensor").noconvert(),
            nb::kw_only(),
            nb::arg("cluster_axis") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("scale") = nb::none()});
}

}  // namespace ttnn::operations::transformer::sdpa_prefill
