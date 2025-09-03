// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "slice_reshard_async_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/experimental/ccl/slice_reshard_async/slice_reshard_async.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

namespace {

template <typename ccl_operation_t>
void bind_slice_reshard_async(pybind11::module& module, const ccl_operation_t& operation, const char* doc) {
    // namespace py = pybind11;

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const int32_t dim,
               const uint32_t output_dim_offset,
               const uint32_t output_dim_shape,
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
                    output_dim_offset,
                    output_dim_shape,
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
            py::arg("output_dim_offset"),
            py::arg("output_dim_shape"),
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

void py_bind_slice_reshard_async(pybind11::module& module) {
    bind_slice_reshard_async(
        module,
        ttnn::experimental::slice_reshard_async,
        R"doc(

        Performs a slice_resharding operation on multi-device :attr:`input_tensor` across all devices.

        Args:
            input_tensor (ttnn.Tensor): multi-device tensor.
            dim (int): Dimension to shard on.
            output_dim_offset (int): Start of the output tensor in the shard dimension, in the context of the input tensor..
            output_dim_shape (int): Shape of the output tensor in the shard dimension, before sharding.
            cluster_axis (int): Provided a MeshTensor, the axis corresponding to MeshDevice to perform the slice_reshard operation on.

        Keyword Args:
            num_links (int, optional): Number of links to use for the slice_reshard operation. Defaults to `1`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `input tensor memory config`.
            topology (ttnn.Topology, optional): The topology configuration to run the operation in. Valid options are Ring and Linear. Defaults to `ttnn.Topology.Ring`.

        Returns:
            ttnn.Tensor: the slice_resharded output tensor.
        )doc");
}

}  // namespace ttnn::operations::experimental::ccl
