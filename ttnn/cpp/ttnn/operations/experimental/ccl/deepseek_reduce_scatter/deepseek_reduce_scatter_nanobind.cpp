// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "deepseek_reduce_scatter_nanobind.hpp"

#include <cstdint>
#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/experimental/ccl/deepseek_reduce_scatter/deepseek_reduce_scatter.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

namespace {

template <typename ccl_operation_t>
void bind_deepseek_reduce_scatter(nb::module_& mod, const ccl_operation_t& operation, const char* doc) {
    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::MemoryConfig& output_memory_config,
               uint32_t num_links,  // TODO: (GR) leave for iterative testing on T3K, remove after switching to GLX
               std::optional<uint32_t> cluster_axis,
               std::optional<tt::tt_metal::SubDeviceId> sub_device_id) -> ttnn::Tensor {
                return self(input_tensor, output_memory_config, num_links, cluster_axis, sub_device_id);
            },
            nb::arg("input_tensor"),
            nb::arg("output_memory_config"),
            nb::kw_only(),
            nb::arg("num_links") = 1,
            nb::arg("cluster_axis") = nb::none(),
            nb::arg("sub_device_id") = nb::none()});
}

}  // namespace

// TODO: (GR) update doc string
void bind_deepseek_reduce_scatter(nb::module_& mod) {
    bind_deepseek_reduce_scatter(
        mod,
        ttnn::experimental::deepseek_reduce_scatter,
        R"doc(
        Performs a reduce-scatter operation on multi-device :attr:`input_tensor` across all devices.

        Args:
            input_tensor (ttnn.Tensor): multi-device tensor.
            dim (int): Dimension to scatter.
            mesh_device (MeshDevice): Device mesh to perform the line-all-gather operation on.

        Mesh Tensor Programming Guide : https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming_Mesh_of_Devices/Programming_Mesh_of_Devices_with_TT-NN.md

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
