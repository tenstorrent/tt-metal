// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "barrier_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/barrier/barrier.hpp"
#include "ttnn/types.hpp"

#include "ttnn/operations/reduction/generic/generic_reductions.hpp"

namespace nb = nanobind;

namespace ttnn::operations::ccl {

namespace {

template <typename ccl_operation_t>
void bind_operation_barrier(nb::module_& mod, const ccl_operation_t& operation, const char* doc) {
    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::MemoryConfig& memory_config,
               ttnn::ccl::Topology topology) -> ttnn::Tensor { return self(input_tensor, memory_config, topology); },
            nb::arg("input_tensor"),
            nb::kw_only(),  // The following are optional by key word only
            nb::arg("memory_config") = nb::none(),
            nb::arg("topology") = ttnn::ccl::Topology::Ring});
}
}  // namespace

void bind_barrier(nb::module_& mod) {
    bind_operation_barrier(
        mod,
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
