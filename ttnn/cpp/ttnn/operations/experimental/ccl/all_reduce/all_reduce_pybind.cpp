// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_reduce_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/experimental/ccl/all_reduce/all_reduce.hpp"
#include "ttnn/types.hpp"

#include "ttnn/operations/reduction/generic/generic_reductions.hpp"

namespace ttnn::operations::experimental::ccl {

namespace {

template <typename ccl_operation_t>
void bind_all_reduce(pybind11::module& module, const ccl_operation_t& operation, const char* doc) {
    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               ttnn::operations::reduction::ReduceType math_op,
               const uint32_t num_links,
               const ttnn::MemoryConfig& memory_config,
               ttnn::ccl::Topology topology,
               const std::optional<size_t> num_workers,
               const std::optional<size_t> num_buffers_per_channel) -> ttnn::Tensor {
                return self(
                    input_tensor, math_op, num_links, memory_config, topology, num_workers, num_buffers_per_channel);
            },
            py::arg("input_tensor"),
            py::arg("math_op"),
            py::kw_only(),
            py::arg("num_links") = 1,
            py::arg("memory_config") = std::nullopt,
            py::arg("topology") = ttnn::ccl::Topology::Ring,
            py::arg("num_workers") = std::nullopt,
            py::arg("num_buffers_per_channel") = std::nullopt});
}

}  // namespace

void py_bind_all_reduce(pybind11::module& module) {
    bind_all_reduce(
        module,
        ttnn::experimental::all_reduce,
        R"doc(

        Performs an all_reduce operation on multi-device :attr:`input_tensor` across all devices.

        Args:
            input_tensor (ttnn.Tensor): multi-device tensor

        Keyword Args:
            num_links (int, optional): Number of links to use for the all-gather operation. Defaults to `1`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `input tensor memory config`.
            num_workers (int, optional): Number of workers to use for the operation. Defaults to `None`.
            num_buffers_per_channel (int, optional): Number of buffers per channel to use for the operation. Defaults to `None`.
            topology (ttnn.Topology, optional): The topology configuration to run the operation in. Valid options are Ring and Linear. Defaults to `ttnn.Topology.Ring`.

        Returns:
            ttnn.Tensor: the output tensor.

        Example:

            >>> full_tensor = torch.randn([1, 1, 256, 256], dtype=torch.bfloat16)
            >>> num_devices = 8
            >>> input_tensors = torch.chunk(full_tensor, num_devices, dim)
            >>> physical_device_ids = ttnn.get_t3k_physical_device_ids_ring()
            >>> mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), physical_device_ids=physical_device_ids[:8])
            >>> tt_input_tensors = []
            >>> for i, t in enumerate(input_tensors):
                    tt_input_tensors.append(ttnn.Tensor(t, input_dtype).to(layout).to(mesh_device.get_devices()[i], mem_config))
            >>> input_tensor_mesh = ttnn.aggregate_as_tensor(tt_input_tensors)

            >>> output = ttnn.experimental.all_reduce(input_tensor_mesh, topology=ttnn.Topology.Linear)

        )doc");
}

}  // namespace ttnn::operations::experimental::ccl
