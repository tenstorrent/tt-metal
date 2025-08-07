// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_attention_all_gather_async_nanobind.hpp"

#include <cstdint>
#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/experimental/ccl/ring_attention_all_gather_async/ring_attention_all_gather_async.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

namespace {
template <typename ccl_operation_t>
void bind_ring_attention_all_gather_async(nb::module_& mod, const ccl_operation_t& operation, const char* doc) {
    // namespace py = nanobind;
    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const ccl_operation_t& self,
               const std::vector<ttnn::Tensor>& input_tensor,
               std::vector<ttnn::Tensor>& persistent_output_buffer,
               const int32_t dim,
               const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
               const uint32_t num_links,
               const uint32_t cluster_axis,
               const MeshDevice& mesh_device,
               const std::optional<MemoryConfig>& memory_config,
               const ttnn::ccl::Topology topology,
               std::optional<tt::tt_metal::SubDeviceId> subdevice_id) -> std::vector<ttnn::Tensor> {
                return self(
                    input_tensor,
                    persistent_output_buffer,
                    dim,
                    multi_device_global_semaphore,
                    cluster_axis,
                    mesh_device,
                    topology,
                    num_links,
                    memory_config,
                    subdevice_id);
            },
            nb::arg("input_tensor"),
            nb::arg("persistent_output_buffer"),
            nb::arg("dim"),
            nb::kw_only(),
            nb::arg("multi_device_global_semaphore"),
            nb::arg("num_links") = nb::none(),
            nb::arg("cluster_axis"),
            nb::arg("mesh_device"),
            nb::arg("memory_config") = nb::none(),
            nb::arg("topology"),
            nb::arg("subdevice_id") = nb::none()});
}

}  // namespace

void bind_ring_attention_all_gather_async(nb::module_& mod) {
    bind_ring_attention_all_gather_async(
        mod,
        ttnn::experimental::ring_attention_all_gather_async,
        R"doc(

        Performs an all-gather operation on multi-device :attr:`input_tensor` across all devices.

        Args:
            input_tensor (ttnn.Tensor): multi-device tensor.
            dim (int): Dimension to perform operation.
            cluster_axis (int): Provided a MeshTensor, the axis corresponding to MeshDevice to perform the line-all-gather operation on.
            mesh_device (MeshDevice): Device mesh to perform the line-all-gather operation on.
        * cluster_axis and mesh_device parameters are applicable only for Linear Topology.

        Mesh Tensor Programming Guide : https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming_Mesh_of_Devices/Programming_Mesh_of_Devices_with_TT-NN.md

        Keyword Args:
            num_links (int, optional): Number of links to use for the all-gather operation. Defaults to `1`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `input tensor memory config`.
            topology (ttnn.Topology, optional): The topology configuration to run the operation in. Valid options are Ring and Linear. Defaults to `ttnn.Topology.Ring`.

        Returns:
            ttnn.Tensor: the output tensor.

        Example:
            >>> full_tensor = torch.randn([1, 1, 32, 256], dtype=torch.bfloat16)
            >>> physical_device_ids = ttnn.get_t3k_physical_device_ids_ring()
            >>> mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), physical_device_ids=physical_device_ids[:8])
            >>> ttnn_tensor = ttnn.from_torch(
                            full_tensor,
                            dtype=input_dtype,
                            device=mesh_device,
                            layout=layout,
                            memory_config=mem_config,
                            mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=(1, 8), dims=(-1, -2)))
            >>> ttnn_tensor = ttnn.to_device(ttnn_tensor, mesh_device)
            >>> output = ttnn.all_gather(ttnn_tensor, dim=0, topology=ttnn.Topology.Ring)

        )doc");
}

}  // namespace ttnn::operations::experimental::ccl
