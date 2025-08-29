// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "neighbor_pad_async_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/experimental/ccl/neighbor_pad_async/neighbor_pad_async.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

namespace {

template <typename ccl_operation_t>
void bind_neighbor_pad_async(pybind11::module& module, const ccl_operation_t& operation, const char* doc) {
    // namespace py = pybind11;

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const int32_t dim,
               const uint32_t padding,
               const std::string& padding_mode,
               const bool direction,
               const uint32_t cluster_axis,
               const GlobalSemaphore& final_semaphore,
               const GlobalSemaphore& barrier_semaphore,
               const MeshDevice& mesh_device,
               const uint32_t num_links,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const ttnn::ccl::Topology topology) -> ttnn::Tensor {
                return self(
                    input_tensor,
                    dim,
                    padding,
                    padding_mode,
                    direction,
                    cluster_axis,
                    final_semaphore,
                    barrier_semaphore,
                    mesh_device,
                    num_links,
                    memory_config,
                    topology);
            },
            py::arg("input_tensor"),
            py::arg("dim"),
            py::arg("padding"),
            py::arg("padding_mode"),
            py::arg("direction"),
            py::arg("cluster_axis"),
            py::arg("final_semaphore"),
            py::arg("barrier_semaphore"),
            py::arg("mesh_device"),
            py::kw_only(),
            py::arg("num_links") = 1,
            py::arg("memory_config") = std::nullopt,
            py::arg("topology") = ttnn::ccl::Topology::Linear});
}

}  // namespace

void py_bind_neighbor_pad_async(pybind11::module& module) {
    bind_neighbor_pad_async(
        module,
        ttnn::experimental::neighbor_pad_async,
        R"doc(

        Performs a padding operation on multi-device :attr:`input_tensor` across all devices, where the padding values come from the neighbor device's tensor.

        Args:
            input_tensor (ttnn.Tensor): multi-device tensor.
            dim (int): Dimension to pad on.
            padding (uint): How much to pad.
            padding_mode (string): replicate, constant, reflect.
            direction (bool): Direction to get the padding values from. 0 = left, 1 = right
            cluster_axis (int): Provided a MeshTensor, the axis corresponding to MeshDevice to perform the neighbor_pad operation on.

        Keyword Args:
            num_links (int, optional): Number of links to use for the neighbor_pad operation. Defaults to `1`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `input tensor memory config`.
            topology (ttnn.Topology, optional): The topology configuration to run the operation in. Valid options are Ring and Linear. Defaults to `ttnn.Topology.Ring`.

        Returns:
            ttnn.Tensor: the padded output tensor.
        )doc");
}

}  // namespace ttnn::operations::experimental::ccl
