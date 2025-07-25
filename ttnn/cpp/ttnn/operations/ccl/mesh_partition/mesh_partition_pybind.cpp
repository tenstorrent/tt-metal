// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_partition_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "mesh_partition.hpp"

namespace ttnn::operations::ccl {

void py_bind_mesh_partition(py::module& module) {
    auto doc =
        R"doc(mesh_partition(input_tensor: ttnn.Tensor, dims: List[int], memory_config: Optional[MemoryConfig] = std::nullopt, queue_id: int = 0) -> ttnn.Tensor

            Partitions the input tensor across the mesh such that each device has the i/num_devices-th partition of the input tensor along the specified dimension. This is the inverse of all_gather

            Args:
                input_tensor (ttnn.Tensor): the input tensor.
                dim (number): the dimension to partition along
                cluster_axis (number): the cluster axis on the mesh.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
                queue_id (int, optional): command queue id. Defaults to `0`.

           Returns:
               ttnn.Tensor: the output tensor.

            Example:

                >>> tensor = ttnn.experimental.mesh_partition(
                                tt_input_tensors_list[i],
                                dim,
                                cluster_axis=1,
                                memory_config=output_mem_config))doc";

    using OperationType = decltype(ttnn::mesh_partition);
    ttnn::bind_registered_operation(
        module,
        ttnn::mesh_partition,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               int32_t dim,
               std::optional<uint32_t> cluster_axis,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               QueueId queue_id) { return self(queue_id, input_tensor, dim, cluster_axis, memory_config); },
            py::arg("input_tensor").noconvert(),
            py::arg("dim"),
            py::arg("cluster_axis") = std::nullopt,
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId,

        });
}

}  // namespace ttnn::operations::ccl
