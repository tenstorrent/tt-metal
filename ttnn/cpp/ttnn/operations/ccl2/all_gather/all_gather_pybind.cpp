// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_pybind.hpp"

namespace ttnn::operations::ccl2 {

void py_bind_all_gather(py::module& module) {
    auto doc =
        R"doc(mesh_partition(input_tensor: ttnn.Tensor, dims: List[int], memory_config: Optional[MemoryConfig] = std::nullopt)) -> ttnn.Tensor

            Partitions the input tensor across the mesh such that each device has the i/num_devices-th partition of the input tensor along the specified dimension. This is the inverse of all_gather

            Args:
                input_tensor (ttnn.Tensor): the input tensor.
                dim (number): the dimension to partition along
                cluster_axis (number): the cluster axis on the mesh.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

           Returns:
               ttnn.Tensor: the output tensor.

            Example:

                >>> tensor = ttnn.experimental.mesh_partition(
                                tt_input_tensors_list[i],
                                dim,
                                cluster_axis=1,
                                memory_config=output_mem_config))doc";

    using OperationType = decltype(ttnn::ccl2::all_gather);
    ttnn::bind_registered_operation(
        module,
        ttnn::ccl2::all_gather,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const int32_t dim,
               const ttnn::ccl2::Topology topology,
               const std::optional<ttnn::MemoryConfig>& output_memory_config,
               const std::optional<tt::tt_metal::SubDeviceId> subdevice_id) -> ttnn::Tensor {
                return self(input_tensor, dim, topology, output_memory_config, subdevice_id);
            },
            py::arg("input_tensor"),
            py::arg("dim"),
            py::arg("topology"),
            py::kw_only(),
            py::arg("output_memory_config") = std::nullopt,
            py::arg("subdevice_id") = std::nullopt});
}

}  // namespace ttnn::operations::ccl2
