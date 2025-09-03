// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_combine_pybind.hpp"

#include <cstdint>
#include <optional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "all_to_all_combine.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/fabric_edm_types.hpp>

namespace ttnn::operations::ccl {

void py_bind_all_to_all_combine(py::module& module) {
    auto doc =
        R"doc(all_to_all_combine(input_tensor: ttnn.Tensor, expert_indices_tensor: ttnn.Tensor, expert_mapping_tensor: ttnn.Tensor, local_reduce: bool = false, num_links: Optional[int] = 1, topology: Optional[ttnn.Topology] = std::nullopt, memory_config: Optional[ttnn.MemoryConfig] = std::nullopt, axis: Optional[int] = std::nullopt, subdevice_id: Optional[ttnn.SubDeviceId] = std::nullopt, optional_output_tensor: Optional[ttnn.Tensor] = std::nullopt)) -> ttnn.Tensor

            All to all combine operation for combining the output tokens from the experts, based on the expert indices and expert mapping tensors. If cluster axis is specified then we combine the tokens only on that axis.
            B = batch size
            S = sequence length
            H = hidden size
            K = selected experts per token
            D = total number of devices
            A = cluster axis to combine along
            D[A] = number of devices along the cluster axis, just D if cluster axis is not specified.
            E = number of experts
            T = total number of tokens = B * S

            Args:
                input_tensor (ttnn.Tensor): the input tensor containing the tokens to combine. The tensor is expected to be [B, S, 1, H] ([B/D[A], S, 1, H] per device) where each row is a token. The tensor is expected to be in Row Major, Interleaved format. It is duplicated on the non-cluster axis.
                expert_indices_tensor (ttnn.Tensor): the expert indices tensor containing the ranking of the experts for each token. The tensor is expected to be [B, S, 1, K] ([B/D[A], S, 1, K] per device) where each value in the row is the expert index inside the mapping table. The tensor is expected to be in Row Major, Interleaved format. It is duplicated on the non-cluster axis.
                expert_mapping_tensor (ttnn.Tensor): the one-hot encoded expert to device mapping tensor containing the location of the experts among each device and each mesh. The tensor is expected to be [D, 1, E, D] ([1, 1, E, D] per device) where each value in the row is 1 if the expert is on the device, 0 otherwise. The tensor is expected to be in Row Major, Interleaved format. This tensor is expected to be the same across all devices.

            Keyword Args:
                local_reduce (bool, optional): whether or not the tokens are locally reduce prior to combining. The expectation is that the expert output tokens corresponding to the dispatched token are already reduced. Defaults to `False`.
                num_links (number, optional): the number of cross-device links to use for combining the tokens. Defaults to `1`.
                topology (ttnn.Topology, optional): the topology to use when combining the tokens. Defaults to what the mesh topology is initialized with. CAREFUL: no guarantees that the topology is valid for the given Fabric Init unless it matches the topology of the mesh.
                memory_config (ttnn.MemoryConfig, optional): Output memory configuration for the output tensors. Defaults to `None`.
                axis (int, optional): the cluster axis to combine along. Defaults to `None` though we assert out when it is not specified.
                subdevice_id (ttnn.SubDeviceId, optional): the subdevice id for the subdevice on which we allocate the worker cores. Defaults to `None`.
                optional_output_tensor (ttnn.Tensor, optional): the optional output tensor to use for the combined tokens. Defaults to `None`.

            Returns:
                ttnn.Tensor: The combined tokens tensor. The tensor is expected to be [K, B, S, H] ([K, B/D[A], S, H] per device) where each row is either a token if that token was dispatched to that device, or a placeholder row if that token was not dispatched to that device. The tensor is expected to be in Row Major, Interleaved format.

            Example:

                >>> output_tensor = ttnn.all_to_all_combine(
                                        input_tensor,
                                        expert_mapping_tensor,
                                        expert_metadata_tensor,
                                        num_links=num_links,
                                        topology=topology,
                                        memory_config=output_memory_config,
                                        local_reduce=local_reduce,
                                        axis=axis)
            )doc";

    using OperationType = decltype(ttnn::all_to_all_combine);
    ttnn::bind_registered_operation(
        module,
        ttnn::all_to_all_combine,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& expert_mapping_tensor,
               const ttnn::Tensor& expert_metadata_tensor,
               const bool local_reduce,
               const std::optional<uint32_t> num_links,
               const std::optional<tt::tt_fabric::Topology> topology,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<uint32_t>& axis,
               const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
               const std::optional<ttnn::Tensor>& optional_output_tensor) {
                return self(
                    input_tensor,
                    expert_mapping_tensor,
                    expert_metadata_tensor,
                    local_reduce,
                    num_links,
                    topology,
                    memory_config,
                    axis,
                    subdevice_id,
                    optional_output_tensor);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("expert_indices_tensor").noconvert(),
            py::arg("expert_mapping_tensor").noconvert(),
            py::kw_only(),
            py::arg("local_reduce") = false,
            py::arg("num_links") = 1,
            py::arg("topology") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("axis") = std::nullopt,
            py::arg("subdevice_id") = std::nullopt,
            py::arg("optional_output_tensor") = std::nullopt});
}

}  // namespace ttnn::operations::ccl
