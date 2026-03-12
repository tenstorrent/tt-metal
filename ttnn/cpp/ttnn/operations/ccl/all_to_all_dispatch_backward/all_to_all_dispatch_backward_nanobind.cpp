// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_dispatch_backward_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "all_to_all_dispatch_backward.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

namespace ttnn::operations::ccl {

void bind_all_to_all_dispatch_backward(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Backward pass for all_to_all_dispatch. Computes the gradient of the input
        token tensor (forward input) given the gradient of the dispatched output tensor
        (forward output).

        This is the transpose of all_to_all_dispatch: each device receives gradient
        slices from all dispatch destination devices and sums them to produce the
        gradient of the original input tokens.

        B = local batch size per device
        S = local sequence length per device
        H = hidden size
        K = selected experts per token
        D = total number of devices
        A = cluster axis to dispatch along
        D[A] = number of devices along the cluster axis
        E = number of experts

        Args:
            grad_output (ttnn.Tensor): gradient of the all_to_all_dispatch output tensor.
                Expected shape [1, B*D[A], S, H] per device (or [1, B, S*D[A], H] if
                output_shard_dim=2 was used in the forward). Row Major, Interleaved format.
            expert_metadata_tensor (ttnn.Tensor): same expert metadata tensor output by
                the forward pass. Shape [1, B*D[A], S, K] per device (replicated). Row Major, uint16.
            expert_mapping_tensor (ttnn.Tensor): same one-hot expert-to-device mapping used
                in the forward pass. Shape [1, 1, E, D] per device (replicated). Row Major, uint16.

        Keyword Args:
            num_links (number, optional): number of cross-device links. Defaults to `None`.
            topology (ttnn.Topology, optional): fabric topology. Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): output memory configuration. Defaults to `None`.
            cluster_axis (int, optional): cluster axis used in the forward pass. Defaults to `None`.
            output_shard_dim (int, optional): shard dimension used in the forward pass. Defaults to `1`.
            subdevice_id (ttnn.SubDeviceId, optional): subdevice id. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): pre-allocated output tensor. Defaults to `None`.

        Returns:
            ttnn.Tensor: gradient of the forward input tensor, shape [D[A], B, S, H] per device.
                The caller should sum over dimension 0 to obtain the final [1, B, S, H] gradient.
                Row Major, Interleaved format.

        Example:
            >>> grad_input = ttnn.all_to_all_dispatch_backward(
                                grad_output,
                                expert_metadata_tensor,
                                expert_mapping_tensor,
                                cluster_axis=cluster_axis,
                                num_links=num_links,
                                topology=topology,
                                memory_config=memory_config)
        )doc";

    using OperationType = decltype(ttnn::all_to_all_dispatch_backward);
    ttnn::bind_registered_operation(
        mod,
        ttnn::all_to_all_dispatch_backward,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& grad_output,
               const ttnn::Tensor& expert_metadata_tensor,
               const ttnn::Tensor& expert_mapping_tensor,
               const std::optional<uint32_t> output_shard_dim,
               const std::optional<uint32_t> cluster_axis,
               const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const std::optional<uint32_t> num_links,
               const std::optional<tt::tt_fabric::Topology> topology) {
                return self(
                    grad_output,
                    expert_mapping_tensor,
                    expert_metadata_tensor,
                    num_links,
                    topology,
                    memory_config,
                    cluster_axis,
                    output_shard_dim,
                    subdevice_id,
                    output_tensor);
            },
            nb::arg("grad_output").noconvert(),
            nb::arg("expert_metadata_tensor").noconvert(),
            nb::arg("expert_mapping_tensor").noconvert(),
            nb::kw_only(),
            nb::arg("output_shard_dim") = 1,
            nb::arg("cluster_axis") = nb::none(),
            nb::arg("subdevice_id") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("num_links") = nb::none(),
            nb::arg("topology").noconvert() = nb::none()});
}

}  // namespace ttnn::operations::ccl
