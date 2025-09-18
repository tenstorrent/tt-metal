// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "barrier_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/barrier/barrier.hpp"
#include "ttnn/types.hpp"

#include "ttnn/operations/reduction/generic/generic_reductions.hpp"

namespace ttnn::operations::ccl {

namespace detail {

template <typename ccl_operation_t>
void bind_barrier(pybind11::module& module, const ccl_operation_t& operation, const char* doc) {
    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::MemoryConfig& memory_config,
               ttnn::ccl::Topology topology) -> ttnn::Tensor { return self(input_tensor, memory_config, topology); },
            py::arg("input_tensor"),
            py::kw_only(),  // The following are optional by key word only
            py::arg("memory_config") = std::nullopt,
            py::arg("topology") = ttnn::ccl::Topology::Ring});
}
}  // namespace detail

void py_bind_barrier(pybind11::module& module) {
    detail::bind_barrier(
        module,
        ttnn::barrier,
        R"doc(

        Performs a barrier operation among all the devices covered in the input multi-device tensor to reduce skew between chips upon completion of the operation

        Args:
            input_tensor (ttnn.Tensor): multi-device tensor

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `input tensor memory config`.
            topology (ttnn.Topology, optional): The topology configuration to run the operation in. Valid options is currently only Ring, defaults to Ring`.


        Returns:
            ttnn.Tensor: the output tensor which is a copy of the input tensor
        Example:
            >>> full_tensor = torch.randn([1, 1, 256, 256], dtype=torch.bfloat16)
            >>> mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 8))
            >>> input_tensor = ttnn.from_torch(
                    full_tensor,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
                )
            >>> output = ttnn.barrier(input_tensor, topology=ttnn.Topology.Ring)

        )doc");
}

}  // namespace ttnn::operations::ccl
