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
#include <tt-metalium/fabric_edm_types.hpp>

namespace ttnn::operations::ccl {

void py_bind_all_to_all_dispatch(py::module& module) {
    auto doc =
        R"doc(all_to_all_dispatch(input_tensor: ttnn.Tensor, expert_indices_tensor: ttnn.Tensor, expert_mapping_tensor: ttnn.Tensor, num_links: int = 1, topology: ttnn.Topology = ttnn.Topology.Linear, memory_config: Optional[ttnn.MemoryConfig] = std::nullopt, subdevice_id: Optional[ttnn.SubDeviceId] = std::nullopt, global_semaphore: Optional[ttnn.GlobalSemaphore] = std::nullopt, queue_id: int = 0) -> Tuple[ttnn.Tensor, ttnn.Tensor]

            All to all dispatch operation for dispatching the input tokens to devices with the selected experts, based on the expert indices and expert mapping tensors. If cluster axis is specified then we dispatch the tokens to the experts only on that axis. The global semaphore is the cross-device semaphore for synchronizing the dispatching of the tokens.
            B = batch size
            S = sequence length
            H = hidden size
            K = selected experts per token
            D = total number of devices
            A = cluster axis to dispatch along
            D[A] = number of devices along the cluster axis, just D if cluster axis is not specified.
            E = number of experts
            T = total number of tokens = B * S

            Args:
                input_tensor (ttnn.Tensor): the input tensor containing the tokens to dispatch. The tensor is expected to be [B, S, 1, H] ([B/D[A], S, 1, H] per device) where each row is a token. The tensor is expected to be in Row Major, Interleaved format. It is duplicated on the non-cluster axis.
                expert_indices_tensor (ttnn.Tensor): the expert indices tensor containing the ranking of the experts for each token. The tensor is expected to be [B, S, 1, K] ([B/D[A], S, 1, K] per device) where each value in the row is the expert index inside the mapping table. The tensor is expected to be in Row Major, Interleaved format. It is duplicated on the non-cluster axis.
                expert_mapping_tensor (ttnn.Tensor): the one-hot encoded expert to device mapping tensor containing the location of the experts among each device and each mesh. The tensor is expected to be [D, 1, E, D] ([1, 1, E, D] per device) where each value in the row is 1 if the expert is on the device, 0 otherwise. The tensor is expected to be in Row Major, Interleaved format. This tensor is expected to be the same across all devices.


            Keyword Args:
                cluster_axis (int, optional): the cluster axis to dispatch along. Defaults to `None` though we assert out when it is not specified.
                num_links (number, optional): the number of cross-device links to use for dispatching the tokens. Defaults to `1`.
                topology (ttnn.Topology, optional): the topology to use when dispatching the tokens. Defaults to `ttnn.Topology.Linear`.
                memory_config (ttnn.MemoryConfig, optional): Output memory configuration for the output tensors. Defaults to `None`.
                subdevice_id (ttnn.SubDeviceId, optional): the subdevice id for the subdevice on which we allocate the worker cores. Defaults to `None`.
                global_semaphore (ttnn.GlobalSemaphore, optional): the global semaphore for synchronizing the dispatching of the tokens. Defaults to `None`.
                queue_id (int, optional): command queue id. Defaults to `0`.

           Returns:
               Tuple[ttnn.Tensor, ttnn.Tensor]: The sparse output tokens tensor and the metadata tensor. The output tensor on each device is sparsely populated with all the tokens that are dispatched to that device. The non-dispatched tokens have placeholder rows populated with garbage. The metadata tensor is used to track the expert indices.

               output_tensor: The output tensor is expected to be [D, B, S, H] ([1, B, S, H] per device) where each row is either a token if that token was dispatched to that device, or a placeholder row if that token was not dispatched to that device. The tensor is expected to be in Row Major, Interleaved format.
               metadata_tensor: The metadata tensor is expected to be [D, B, S, K] ([1, B, S, K] per device) where each row contains the all the expert indices selected for each token. This is the all-to-all of the expert indices. The tensor is expected to be in Row Major, Interleaved format.

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
                                global_semaphore=global_semaphore,
                                queue_id=queue_id))doc";

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
               const std::optional<std::array<ttnn::Tensor, 2>>& optional_output_tensors,
               const std::optional<uint32_t> axis,
               const uint32_t num_links,
               const tt::tt_fabric::Topology topology,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
               const std::optional<GlobalSemaphore>& global_semaphore,
               QueueId queue_id) {
                return self(
                    queue_id,
                    input_tensor,
                    expert_indices_tensor,
                    expert_mapping_tensor,
                    axis,
                    optional_output_tensors,
                    num_links,
                    topology,
                    memory_config,
                    subdevice_id,
                    global_semaphore);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("expert_indices_tensor").noconvert(),
            py::arg("expert_mapping_tensor").noconvert(),
            py::kw_only(),
            py::arg("output_tensors") = std::nullopt,
            py::arg("cluster_axis") = std::nullopt,
            py::arg("num_links") = 1,
            py::arg("topology") = tt::tt_fabric::Topology::Linear,
            py::arg("memory_config") = std::nullopt,
            py::arg("subdevice_id") = std::nullopt,
            py::arg("global_semaphore") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId,
        });
}

}  // namespace ttnn::operations::ccl
