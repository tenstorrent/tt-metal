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
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

namespace ttnn::operations::ccl {

void py_bind_all_to_all_combine(py::module& module) {
    const auto* doc =
        R"doc(
        All to all combine operation for combining the output tokens from the experts, based on the expert metadata and expert mapping tensors. If cluster axis is specified then we combine the tokens only on that axis. This operation is the inverse of the all-to-all dispatch operation, used for returning the results of the experts back to the input tokens' originating devices.
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
            expert_metadata_tensor (ttnn.Tensor): the expert metadata tensor containing the ranking of the experts for each token. The tensor is expected to be [B, S, 1, K] ([B/D[A], S, 1, K] per device) where each value in the row is the expert index inside the mapping table. The tensor is expected to be in Row Major, Interleaved format. It is duplicated on the non-cluster axis.
            expert_mapping_tensor (ttnn.Tensor): the one-hot encoded expert to device mapping tensor containing the location of the experts among each device and each mesh. The tensor is expected to be [D, 1, E, D] ([1, 1, E, D] per device) where each value in the row is 1 if the expert is on the device, 0 otherwise. The tensor is expected to be in Row Major, Interleaved format. This tensor is expected to be the same across all devices.

        Keyword Args:
            local_reduce (bool, optional): whether or not the tokens are locally reduce prior to combining. The expectation is that the expert output tokens corresponding to the dispatched token are already reduced. Defaults to `False`.
            num_links (number, optional): the number of cross-device links to use for combining the tokens. Defaults to `None`, for which the number of links is determined automatically.
            topology (ttnn.Topology, optional): the topology to use when combining the tokens. Defaults to `None`, for which the topology is determined automatically.
            memory_config (ttnn.MemoryConfig, optional): Output memory configuration for the output tensors. Defaults to `None`.
            cluster_axis (int, optional): the cluster axis to combine along. Defaults to `None`, though we assert out when it is not specified.
            subdevice_id (ttnn.SubDeviceId, optional): the subdevice id for the subdevice on which we allocate the worker cores. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): the optional output tensor to use for the combined tokens. Defaults to `None`.
            output_shard_dim (int, optional): the dimension to shard the output tokens along. Defaults to `1`, which is the batch dimension.

        Returns:
            ttnn.Tensor: The combined tokens tensor. The tensor is expected to be [K, B, S, H] sharded along the output_shard_dim dimension across the number of devices along the cluster axis if it was set, or all devices if it was not set, (e.g. [K, B/D[A], S, H] per device if output_shard_dim is 1 or [K, B, S/D[A], H] per device if output_shard_dim is 2). The tensor is expected to be in Row Major, Interleaved format. The rows are sparsely populated such that each row is either a token if that token was dispatched to that device, or a placeholder row if that token was not dispatched to that device.

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
