// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "barrier_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
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
               ttnn::ccl::Topology topology)-> ttnn::Tensor {
                return self(input_tensor, memory_config, topology);
            },
            py::arg("input_tensor"),
            py::kw_only(),//The following are optional by key word only
            py::arg("memory_config") = std::nullopt,
            py::arg("topology") = ttnn::ccl::Topology::Ring});
}
}  // namespace detail

void py_bind_barrier(pybind11::module& module) {

    detail::bind_barrier(
        module,
        ttnn::barrier,
        R"doc(

        Performs a hop latency test on multi-device :attr:`input_tensor` across all devices.

        Args:
            input_tensor (ttnn.Tensor): multi-device tensor
            num_samples(uint): the number of samples
            max_concurrent_samples(uint): the maximum number of concurrent samples
            sample_page_size(uint): the page size of the test

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `input tensor memory config`.
            topology (ttnn.Topology, optional): The topology configuration to run the operation in. Valid options is currently only Ring`.


        Returns:
            ttnn.Tensor: the output tensor which is a copy of the input tensor
        Example:
            >>> full_tensor = torch.randn([1, 1, 256, 256], dtype=torch.bfloat16)
            >>> input_tensors = torch.chunk(full_tensor, num_devices, dim)
            >>> physical_device_ids = ttnn.get_t3k_physical_device_ids_ring()
            >>> mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), physical_device_ids=physical_device_ids[:8])
            >>> tt_input_tensors = []
            >>> for i, t in enumerate(input_tensors):
                    tt_input_tensors.append(ttnn.Tensor(t, input_dtype).to(layout).to(mesh_device.get_devices()[i], mem_config))
            >>> input_tensor_mesh = ttnn.aggregate_as_tensor(tt_input_tensors)

            >>> output = ttnn.reduce_scatter(input_tensor_mesh, topology=ttnn.Topology.Ring)


        )doc");
}

}
