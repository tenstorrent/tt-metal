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
        R"doc(all_to_all_combine(input_tensor: ttnn.Tensor, expert_metadata_tensor: ttnn.Tensor, expert_mapping_tensor: ttnn.Tensor, *, local_reduce: bool = false, num_links: Optional[int] = None, topology: Optional[ttnn.Topology] = None, memory_config: Optional[ttnn.MemoryConfig] = None, cluster_axis: Optional[int] = None, subdevice_id: Optional[ttnn.SubDeviceId] = None, output_tensor: Optional[ttnn.Tensor] = None) -> ttnn.Tensor

            All to all combine operation for combining the output tokens from the experts, based on the expert metadata and expert mapping tensors. If cluster axis is specified then we combine the tokens only on that axis. This operation is the inverse of the all-to-all dispatch operation, used for returning the results of the experts back to the input tokens' originating devices.
            B = local batch size/batch size per device
            S = local sequence length/sequence length per device
            H = hidden size
            K = selected experts per token
            D = total number of devices
            A = cluster axis to combine along
            D[A] = number of devices along the cluster axis, just D if cluster axis is not specified.
            E = local experts/experts per device
            T = total number of tokens per device = B * S

            Args:
                input_tensor (ttnn.Tensor): The sparse input tensor containing the tokens to combine when an expert on the device was selected and garbage values when tokens were not operated on by any expert on the device. The tensor is expected to be [E, B, S, H] per device (expert parallel across all devices such that we have [E*D, B, S, H] total when gathered along the expert dimension) where each row is a token. The tensor is expected to be in Row Major, Interleaved format. It is duplicated on the non-cluster axis dimension of the mesh.
                expert_metadata_tensor (ttnn.Tensor): The dense expert metadata tensor containing the ranking of the experts for all tokens on the mesh. This is an all-gather of each token's expert scores across all devices. The tensor is expected to be [1, B, S, K] per device (fully tensor sharded across all devices such that we have [D, B, S, K] total when gathered along dimension 0) where each value in the row is the expert index inside the mapping table. The tensor is expected to be in Row Major, Interleaved format. It is duplicated on the non-cluster axis dimension of the mesh.
                expert_mapping_tensor (ttnn.Tensor): The one-hot encoded expert to device mapping tensor that maps each expert to the device that it lives on in the mesh. The tensor is expected to be [1, 1, E, D] per device, replicated across all devices, where each value in the row is 1 if the expert is on the device, 0 otherwise. The tensor is expected to be in Row Major, Interleaved format. This tensor is expected to be the same across all devices.

            Keyword Args:
                local_reduce (bool, optional): Indicates whether or not the tokens are locally reduced prior to combining. The expectation is that the expert output tokens corresponding to the dispatched token are already reduced. Defaults to `False`.
                num_links (number, optional): The number of cross-device links to use for combining the tokens. Defaults to `None`, for which the number of links is determined automatically.
                topology (ttnn.Topology, optional): The topology to use when combining the tokens. Defaults to `None`, for which the topology is determined automatically.
                memory_config (ttnn.MemoryConfig, optional): Output memory configuration for the output tensors. Defaults to `None`.
                cluster_axis (int, optional): The cluster axis to combine along. Defaults to `None`, though we assert out when it is not specified.
                subdevice_id (ttnn.SubDeviceId, optional): The subdevice id for the subdevice on which we allocate the worker cores. Defaults to `None`.
                output_tensor (ttnn.Tensor, optional): The optional output tensor to use for the combined tokens. Defaults to `None`.
                output_shard_dim (int, optional): The dimension to shard the output tokens along. Defaults to `1`, which is the batch dimension.

            Returns:
                ttnn.Tensor: The combined tokens tensor. The tensor is expected to be [K, B/D, S, H] per device if output_shard_dim is 1, or [K, B, S/D, H] per device if output_shard_dim is 2. The tensor is unique per-device, such that the global shape is [K, B, S, H] when gathered along the output_shard_dim dimension. The tensor is expected to be in Row Major, Interleaved format. This tensor is sparse, such that if a token was not dispatched to any experts along the device's cluster axis, the row is populated with zeros to enable subsequentreduce-scatter and all-reduce operations without any additional logic.

            Example:

                >>> output_tensor = ttnn.all_to_all_combine(
                                        input_tensor,
                                        expert_metadata_tensor,
                                        expert_mapping_tensor,
                                        num_links=num_links,
                                        topology=topology,
                                        memory_config=output_memory_config,
                                        local_reduce=local_reduce,
                                        cluster_axis=cluster_axis,
                                        output_shard_dim=output_shard_dim)
            )doc";

    using OperationType = decltype(ttnn::all_to_all_combine);
    ttnn::bind_registered_operation(
        module,
        ttnn::all_to_all_combine,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& expert_metadata_tensor,
               const ttnn::Tensor& expert_mapping_tensor,
               const bool local_reduce,
               const std::optional<uint32_t> output_shard_dim,
               const std::optional<uint32_t> cluster_axis,
               const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const std::optional<uint32_t> num_links,
               const std::optional<tt::tt_fabric::Topology> topology) {
                return self(
                    input_tensor,
                    expert_mapping_tensor,
                    expert_metadata_tensor,
                    local_reduce,
                    num_links,
                    topology,
                    memory_config,
                    cluster_axis,
                    output_shard_dim,
                    subdevice_id,
                    output_tensor);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("expert_metadata_tensor").noconvert(),
            py::arg("expert_mapping_tensor").noconvert(),
            py::kw_only(),
            py::arg("local_reduce") = false,
            py::arg("output_shard_dim") = 1,
            py::arg("cluster_axis") = std::nullopt,
            py::arg("subdevice_id") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("num_links") = std::nullopt,
            py::arg("topology").noconvert() = std::nullopt});
}

}  // namespace ttnn::operations::ccl
