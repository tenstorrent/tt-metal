// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_partition_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "mesh_partition.hpp"

namespace ttnn::operations::ccl {

void bind_mesh_partition(nb::module_& mod) {
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

    using OperationType = decltype(ttnn::mesh_partition);
    ttnn::bind_registered_operation(
        mod,
        ttnn::mesh_partition,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               int32_t dim,
               std::optional<uint32_t> cluster_axis,
               const std::optional<ttnn::MemoryConfig>& memory_config) {
                return self(input_tensor, dim, cluster_axis, memory_config);
            },
            nb::arg("input_tensor").noconvert(),
            nb::arg("dim"),
            nb::arg("cluster_axis") = nb::none(),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()});
}

}  // namespace ttnn::operations::ccl
