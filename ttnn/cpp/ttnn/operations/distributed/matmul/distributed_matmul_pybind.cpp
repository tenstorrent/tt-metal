// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "distributed_matmul_pybind.hpp"

#include <pybind11/pybind11.h>

#include "ttnn-pybind/decorators.hpp"
#include "distributed_matmul.hpp"

namespace ttnn::operations::distributed {

void py_bind_distributed_matmul(py::module& module) {
    auto doc =
        R"doc(distributed_matmul(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, memory_config: Optional[ttnn.MemoryConfig] = None, dtype: Optional[ttnn.DataType] = None) -> ttnn.Tensor

            Distributed matrix multiplication operation that supports multi-device tensors with arbitrary sharding/replication layouts.

            This is a prototype implementation for the unified programming model that virtualizes device boundaries.
            Currently, this delegates to the standard matmul operation but will eventually support:
            - Automatic handling of different input tensor topologies (sharded/replicated)
            - Optional output tensor topology specification
            - Cross-device communication as needed
            - Execution across any number of devices without manual CCL insertion

            Args:
                input_tensor_a (ttnn.Tensor): First input tensor for matrix multiplication. Must be on a mesh device.
                input_tensor_b (ttnn.Tensor): Second input tensor for matrix multiplication. Must be on the same mesh device as input_tensor_a.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Output memory configuration. Defaults to input_tensor_a's memory config.
                dtype (ttnn.DataType, optional): Output data type. Defaults to input_tensor_a's dtype.

            Returns:
                ttnn.Tensor: Result of the distributed matrix multiplication.

            Example:
                >>> import torch
                >>> import ttnn
                >>> mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))
                >>> # Create replicated tensors
                >>> a = torch.randn([4, 1, 64, 64], dtype=torch.bfloat16)
                >>> b = torch.randn([1, 1, 64, 128], dtype=torch.bfloat16)
                >>> tt_a = ttnn.from_torch(a, device=mesh_device, layout=ttnn.TILE_LAYOUT,
                ...                         mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
                >>> tt_b = ttnn.from_torch(b, device=mesh_device, layout=ttnn.TILE_LAYOUT,
                ...                         mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
                >>> output = ttnn.distributed.matmul(tt_a, tt_b)
                >>> print(output.shape)
                [4, 1, 64, 128]
        )doc";

    using OperationType = decltype(ttnn::distributed::matmul);
    ttnn::bind_registered_operation(
        module,
        ttnn::distributed::matmul,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::DataType>& dtype) {
                return self(input_tensor_a, input_tensor_b, memory_config, dtype);
            },
            py::arg("input_tensor_a").noconvert(),
            py::arg("input_tensor_b").noconvert(),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("dtype") = std::nullopt,
        });
}

}  // namespace ttnn::operations::distributed
