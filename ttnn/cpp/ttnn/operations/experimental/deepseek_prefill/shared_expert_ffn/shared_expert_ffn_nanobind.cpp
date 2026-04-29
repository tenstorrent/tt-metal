// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "shared_expert_ffn_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

#include "ttnn-nanobind/bind_function.hpp"
#include "shared_expert_ffn.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::shared_expert_ffn::detail {

void bind_shared_expert_ffn(nb::module_& mod) {
    ttnn::bind_function<"shared_expert_ffn", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
        Composite shared expert FFN for DeepSeek MoE prefill.

        Computes the gated FFN with multi-chip tensor parallelism:
            gate_out  = x @ gate_proj          (with fused SiLU activation)
            up_out    = x @ up_proj
            activated = gate_out * up_out
            full_out  = activated @ down_proj
            output    = reduce_scatter(full_out, dim=-1, cluster_axis=cluster_axis)

        When ``tp_axis_size == 1`` the reduce_scatter step is skipped and
        ``full_out`` is returned directly.

        Args:
            x (ttnn.Tensor): Input tensor (replicated across the reduce-scatter axis).
            gate_proj (ttnn.Tensor): Gate projection weight, sharded on output dim.
            up_proj (ttnn.Tensor): Up projection weight, sharded on output dim.
            down_proj (ttnn.Tensor): Down projection weight, sharded on input dim.
            cluster_axis (int): Mesh axis to reduce-scatter along.
            tp_axis_size (int): Size of the mesh along ``cluster_axis``.
                When 1, reduce_scatter is skipped.

        Keyword Args:
            num_links (int, optional): Number of ethernet links for CCL. Defaults to 1.
            topology (ttnn.Topology, optional): CCL topology. Defaults to Linear.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Compute kernel config. Defaults to None.
            subdevice_id (ttnn.SubDeviceId, optional): Sub-device whose Tensix cores will run the matmuls and reduce_scatter. Defaults to None (uses the first sub-device).

        Returns:
            ttnn.Tensor: Output tensor.

        Example:
            >>> output = ttnn.experimental.deepseek_prefill.shared_expert_ffn(
                    x, gate_proj, up_proj, down_proj,
                    cluster_axis=1, tp_axis_size=mesh_device.shape[1],
                    num_links=1, topology=ttnn.Topology.Linear,
                    compute_kernel_config=compute_kernel_config)
        )doc",
        &shared_expert_ffn,
        nb::arg("x").noconvert(),
        nb::arg("gate_proj").noconvert(),
        nb::arg("up_proj").noconvert(),
        nb::arg("down_proj").noconvert(),
        nb::arg("cluster_axis"),
        nb::arg("tp_axis_size"),
        nb::kw_only(),
        nb::arg("num_links") = 1,
        nb::arg("topology") = tt::tt_fabric::Topology::Linear,
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("subdevice_id") = nb::none());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::shared_expert_ffn::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {

void bind_shared_expert_ffn(::nanobind::module_& mod) { shared_expert_ffn::detail::bind_shared_expert_ffn(mod); }

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
