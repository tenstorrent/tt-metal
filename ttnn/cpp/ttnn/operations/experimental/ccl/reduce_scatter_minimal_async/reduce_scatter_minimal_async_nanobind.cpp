// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_scatter_minimal_async_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/reduce_scatter_minimal_async.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

namespace {

template <typename ccl_operation_t>
void bind_reduce_scatter_minimal_async(nb::module_& mod, const ccl_operation_t& operation, const char* doc) {
    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               ttnn::Tensor& persistent_intermediate_buffer,
               ttnn::Tensor& persistent_output_buffer,
               const int32_t dim,
               const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
               const uint32_t num_links,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const ttnn::ccl::Topology topology,
               std::optional<tt::tt_metal::SubDeviceId> subdevice_id) -> ttnn::Tensor {
                return self(
                    input_tensor,
                    persistent_intermediate_buffer,
                    persistent_output_buffer,
                    dim,
                    multi_device_global_semaphore,
                    num_links,
                    memory_config,
                    topology,
                    subdevice_id);
            },
            nb::arg("input_tensor"),
            nb::arg("persistent_intermediate_buffer"),
            nb::arg("persistent_output_buffer"),
            nb::arg("dim"),
            nb::arg("multi_device_global_semaphore"),
            nb::kw_only(),
            nb::arg("num_links") = 1,
            nb::arg("memory_config") = std::nullopt,
            nb::arg("topology") = ttnn::ccl::Topology::Ring,
            nb::arg("subdevice_id") = std::nullopt});
}

}  // namespace

void bind_reduce_scatter_minimal_async(nb::module_& mod) {
    bind_reduce_scatter_minimal_async(
        mod,
        ttnn::experimental::reduce_scatter_minimal_async,
        R"doc(

        Performs an reduce-scatter operation on multi-device :attr:`input_tensor` across all devices.

        Args:
            input_tensor (ttnn.Tensor): multi-device tensor.
            dim (int): Dimension to scatter.
            mesh_device (MeshDevice): Device mesh to perform the line-all-gather operation on.

        Mesh Tensor Programming Guide : https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming%20Mesh%20of%20Devices/Programming%20Mesh%20of%20Devices%20with%20TT-NN.md

        Keyword Args:
            num_links (int, optional): Number of links to use for the all-gather operation. Defaults to `1`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `input tensor memory config`.
            topology (ttnn.Topology, optional): The topology configuration to run the operation in. Valid options are Ring and Linear. Defaults to `ttnn.Topology.Ring`.

        Returns:
            ttnn.Tensor: the output tensor.

        Example:

        )doc");
}

}  // namespace ttnn::operations::experimental::ccl
