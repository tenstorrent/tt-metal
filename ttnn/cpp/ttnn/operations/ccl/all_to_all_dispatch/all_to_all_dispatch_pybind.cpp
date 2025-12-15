// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_dispatch_pybind.hpp"

#include <cstdint>
#include <optional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "all_to_all_dispatch.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

namespace ttnn::operations::ccl {

void py_bind_all_to_all_dispatch(py::module& module) {
    const auto* doc =
        R"doc(
        All to all dispatch operation for dispatching the input tokens to devices with the selected experts, based on the expert indices and expert mapping tensors. If cluster axis is specified then we dispatch the tokens to the experts only on that axis. This operation sends tokens to their selected experts, with empty rows for tokens that did not select any experts on that device.
        B = local batch size/batch size per device
        S = local sequence length/sequence length per device
        H = hidden size
        K = selected experts per token
        D = total number of devices
        A = cluster axis to dispatch along
        D[A] = number of devices along the cluster axis, just D if cluster axis is not specified.
        E = local experts/experts per device
        T = total number of tokens per device = B * S

        Args:
            input_tensor (ttnn.Tensor): The input tensor containing the tokens to dispatch. The tensor is expected to be [B, S, 1, H] per device, sharded along either the batch dimension or the sequence dimension, such that the global shape is either [B*D[A], S, 1, H] or [B, S*D[A], 1, H]. Each row is a token. The tensor is expected to be in Row Major, Interleaved format. It is duplicated on the non-cluster axis.
            expert_indices_tensor (ttnn.Tensor): The expert indices tensor containing the ranking of the experts for each token. The tensor is expected to be [B, S, 1, K] per device, sharded identically to the input_tensor. Each value in the row is an expert index, which corresponds to a row index in the expert mapping tensor. This tensor only contains the expert ranking for the tokens local to that device. The tensor is expected to be in Row Major, Interleaved format. It is duplicated on the non-cluster axis.
            expert_mapping_tensor (ttnn.Tensor): The one-hot encoded expert to device mapping tensor containing the location of the experts among each device and each mesh. The tensor is expected to be [1, 1, E, D] per device, fully replicated across all devices. Each row corresponds to an expert, and the value in each corresponding column is 1 if the expert is on the device, 0 otherwise. The tensor is expected to be in Row Major, Interleaved format. This tensor is expected to be the same across all devices.

        Keyword Args:
            cluster_axis (int, optional): the cluster axis to dispatch along. Defaults to `None` though we assert out when it is not specified.
            num_links (number, optional): the number of cross-device links to use for dispatching the tokens. Defaults to `None`, for which the number of links is determined automatically.
            topology (ttnn.Topology, optional): the topology to use when dispatching the tokens. Defaults to what the mesh topology is initialized with. CAREFUL: no guarantees that the topology is valid for the given Fabric Init unless it matches the topology of the mesh.
            memory_config (ttnn.MemoryConfig, optional): Output memory configuration for the output tensors. Defaults to `None`.
            subdevice_id (ttnn.SubDeviceId, optional): the subdevice id for the subdevice on which we allocate the worker cores. Defaults to `None`.
            output_concat_dim (int, optional): the dimension to concat the output tokens along. Defaults to `1`, which is the batch dimension.
            output_tensors (Tuple[ttnn.Tensor, ttnn.Tensor], optional): the optional output tensors to use for the dispatched tokens and the metadata. Defaults to `None`.

        Returns:
            Tuple[ttnn.Tensor, ttnn.Tensor]: The sparse output tokens tensor and the metadata tensor. The output tensor on each device is sparsely populated with all the tokens that are dispatched to that device. The non-dispatched tokens have placeholder rows populated with garbage. The metadata tensor is used to track the expert indices.

            output_tensor: The output tensor is expected to be [1, B*D[A], S, H] per device if output_concat_dim is 1 or [1, B, S*D[A], H] per device if output_concat_dim is 2, sharded fully such that we have [D, B*D[A], S, H] or [D, B, S*D[A], H] total when gathered along dimension 0. Each row is either a token if that token was dispatched to that device, or a placeholder row if that token was not dispatched to that device. The tensor is expected to be in Row Major, Interleaved format.
            expert_metadata_tensor: The metadata tensor is expected to be [1, B*D[A], S, K] per device if output_concat_dim is 1 or [1, B, S*D[A], K] per device if output_concat_dim is 2, replicated across all devices. Each row contains the all the expert indices selected for each token on the mesh. This is equivalent to an all-gather of the expert indices. The tensor is expected to be in Row Major, Interleaved format.

        Example:
            >>> output_tensor, metadata_tensor = ttnn.all_to_all_dispatch(
                            input_tensor,
                            expert_indices_tensor,
                            expert_mapping_tensor,
                            cluster_axis=cluster_axis,
                            num_links=num_links,
                            topology=topology,
                            memory_config=memory_config,
                            subdevice_id=subdevice_id,
                            output_concat_dim=output_concat_dim)
        )doc";

    using OperationType = decltype(ttnn::all_to_all_dispatch);
    ttnn::bind_registered_operation(
        module,
        ttnn::all_to_all_dispatch,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& expert_indices_tensor,
               const ttnn::Tensor& expert_mapping_tensor,
               const std::optional<uint32_t> output_concat_dim,
               const std::optional<uint32_t> cluster_axis,
               const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<std::array<ttnn::Tensor, 2>>& output_tensors,
               const std::optional<uint32_t> num_links,
               const std::optional<tt::tt_fabric::Topology> topology) {
                return self(
                    input_tensor,
                    expert_indices_tensor,
                    expert_mapping_tensor,
                    cluster_axis,
                    output_tensors,
                    num_links,
                    topology,
                    memory_config,
                    subdevice_id,
                    output_concat_dim);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("expert_indices_tensor").noconvert(),
            py::arg("expert_mapping_tensor").noconvert(),
            py::kw_only(),
            py::arg("output_concat_dim") = 1,
            py::arg("cluster_axis") = std::nullopt,
            py::arg("subdevice_id") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensors") = std::nullopt,
            py::arg("num_links") = std::nullopt,
            py::arg("topology").noconvert() = std::nullopt});
}

}  // namespace ttnn::operations::ccl
