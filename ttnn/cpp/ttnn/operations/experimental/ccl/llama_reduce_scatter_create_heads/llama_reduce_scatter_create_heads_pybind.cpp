// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "llama_reduce_scatter_create_heads_pybind.hpp"
#include "llama_reduce_scatter_create_heads.hpp"
#include <tt-metalium/sub_device_types.hpp>
namespace ttnn::operations::experimental::ccl {
namespace py = pybind11;

void py_bind_llama_rs_create_heads(py::module& module) {
    auto doc =
        R"doc(llama_rs_create_heads(input_tensor: ttnn.Tensor, dims: List[int], memory_config: Optional[MemoryConfig] = std::nullopt, queue_id: int = 0) -> ttnn.Tensor

            Reduce_scatter after FF1/3 for Llama70B.

            Args:
                input_tensor (ttnn.Tensor): the input tensor.
                intermediate_packet_buffer (ttnn.Tensor): the intermediate packet buffer tensor.
                dim (number): the reduce dimension
                cross_device_semaphore (ttnn.GlobalSemaphore): the cross device semaphore.
                subdevice_id (ttnn.SubDeviceId): the subdevice id.
                cluster_axis (number): the cluster axis.
                mesh_device (ttnn.MeshDevice): the mesh device.
                num_links (number, optional): the number of links. Defaults to `3`.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
                queue_id (int, optional): command queue id. Defaults to `0`.

           Returns:
               ttnn.Tensor: the output tensor.

            Example:

                >>> tensor = ttnn.experimental.llama_rs_create_heads(
                                tt_input_tensors_list[i],
                                tt_intermediate_tensors_list[i],
                                dim,
                                ccl_semaphore_handles[i],
                                worker_sub_device_id,
                                cluster_axis=1,
                                mesh_device=mesh_device,
                                num_links=num_links,
                                memory_config=output_mem_config))doc";

    using OperationType = decltype(ttnn::experimental::llama_rs_create_heads);
    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::llama_rs_create_heads,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               ttnn::Tensor& intermediate_packet_buffer,
               uint32_t dim,
               const GlobalSemaphore& cross_device_semaphore,
               const tt::tt_metal::SubDeviceId& subdevice_id,
               const uint32_t cluster_axis,
               const MeshDevice& mesh_device,
               ttnn::ccl::Topology topology,
               const uint32_t num_links,
               const uint32_t num_heads,
               const uint32_t num_kv_heads,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::MemoryConfig>& qkv_memory_config,
               const bool use_noc1_only,
               const bool use_optimal_ccl_for_llama,
               QueueId queue_id) {
                return self(
                    queue_id,
                    input_tensor,
                    intermediate_packet_buffer,
                    dim,
                    cross_device_semaphore,
                    subdevice_id,
                    cluster_axis,
                    mesh_device,
                    topology,
                    num_links,
                    num_heads,
                    num_kv_heads,
                    memory_config,
                    qkv_memory_config,
                    use_noc1_only,
                    use_optimal_ccl_for_llama);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("intermediate_packet_buffer").noconvert(),
            py::arg("dim"),
            py::arg("cross_device_semaphore"),
            py::arg("subdevice_id"),
            py::arg("cluster_axis"),
            py::arg("mesh_device"),
            py::arg("topology"),
            py::kw_only(),
            py::arg("num_links") = 1,
            py::arg("num_heads"),
            py::arg("num_kv_heads"),
            py::arg("memory_config") = std::nullopt,
            py::arg("qkv_memory_config") = std::nullopt,
            py::arg("use_noc1_only") = false,
            py::arg("use_optimal_ccl_for_llama") = false,
            py::arg("queue_id") = DefaultQueueId,
        });
}

}  // namespace ttnn::operations::experimental::ccl
