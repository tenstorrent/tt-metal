// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_pybind.hpp"

#include <cstdint>
#include <optional>

#include <pybind11/pybind11.h>

#include "ttnn-pybind/decorators.hpp"
#include "all_gather.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/fabric_edm_types.hpp>

namespace ttnn::operations::ccl {

void py_bind_all_gather(py::module& module) {
    auto doc =
        R"doc(all_gather(input_tensor: ttnn.Tensor, dim: int, cluster_axis: Optional[int] = None, topology: ttnn.Topology = ttnn.Topology.Linear, output_tensor: Optional[ttnn.Tensor] = None, memory_config: Optional[ttnn.MemoryConfig] = None, subdevice_id: Optional[ttnn.SubDeviceId] = None, queue_id: int = 0) -> ttnn.Tensor

            All-gather operation across devices along a selected dimension and optional cluster axis.

            Args:
                input_tensor (ttnn.Tensor): Input tensor to be gathered.
                dim (int): Dimension along which to gather.

            Keyword Args:
                cluster_axis (int, optional): The cluster axis to gather across. Defaults to `None`.
                topology (ttnn.Topology, optional): Fabric topology. Defaults to `ttnn.Topology.Linear`.
                output_tensor (ttnn.Tensor, optional): Preallocated output tensor.
                memory_config (ttnn.MemoryConfig, optional): Output memory configuration.
                subdevice_id (ttnn.SubDeviceId, optional): Subdevice id for worker cores.

           Returns:
               ttnn.Tensor: The gathered tensor.)doc";

    using OperationType = decltype(ttnn::all_gather);
    ttnn::bind_registered_operation(
        module,
        ttnn::all_gather,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               int32_t dim,
               const std::optional<uint32_t> cluster_axis,
               const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               std::optional<ttnn::Tensor>& optional_output_tensor,
               const std::optional<uint32_t> num_links,
               const std::optional<tt::tt_fabric::Topology> topology) {
                return self(
                    input_tensor,
                    dim,
                    cluster_axis,
                    subdevice_id,
                    memory_config,
                    optional_output_tensor,
                    num_links,
                    topology);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("dim"),
            py::kw_only(),
            py::arg("cluster_axis") = std::nullopt,
            py::arg("subdevice_id") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("num_links") = std::nullopt,
            py::arg("topology") = std::nullopt,
        });
}

}  // namespace ttnn::operations::ccl
