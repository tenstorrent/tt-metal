// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_reduce_scatter_async_nanobind.hpp"

#include <cstdint>
#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/experimental/ccl/matmul_reduce_scatter_async/matmul_reduce_scatter_async.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::ccl {

namespace {

template <typename ccl_operation_t>
void bind_matmul_reduce_scatter_async(nb::module_& mod, const ccl_operation_t& operation, const char* doc) {
    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"),
            nb::arg("weight_tensor"),
            nb::arg("persistent_intermediate_buffer"),
            nb::arg("persistent_output_buffer"),
            nb::arg("dim"),
            nb::arg("multi_device_global_semaphore"),
            nb::arg("reduce_scatter_core_grid_offset"),
            nb::kw_only(),
            nb::arg("barrier_semaphore") = nb::none(),
            nb::arg("bias") = nb::none(),
            nb::arg("num_links") = 1,
            nb::arg("memory_config_rs") = nb::none(),
            nb::arg("intermediate_memory_config_rs") = nb::none(),
            nb::arg("topology") = ttnn::ccl::Topology::Ring,
            nb::arg("subdevice_id") = nb::none(),
            nb::arg("memory_config_mm") = nb::none(),
            nb::arg("transpose_a") = false,
            nb::arg("transpose_b") = false,
            nb::arg("dtype") = nb::none(),
            nb::arg("program_config") = nb::none(),
            nb::arg("activation") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none(),
            nb::arg("core_grid") = nb::none()});
}

}  // namespace

void bind_matmul_reduce_scatter_async(nb::module_& mod) {
    bind_matmul_reduce_scatter_async(
        mod,
        ttnn::experimental::matmul_reduce_scatter_async,
        R"doc(
        Performs an reduce-scatter operation on multi-device :attr:`input_tensor` across all devices.

        Args:
            * :attr:`input_tensor` (ttnn.Tensor): multi-device tensor
            * :attr:`weight_tensor` (ttnn.Tensor): multi-device tensor
            * :attr:`dim` (int)
            * :attr:`reduce_scatter_core_grid_offset` (ttnn.CoreCoord): Core grid offset for the reduce-scatter operation.

        Keyword Args:
            * :attr:`bias` (ttnn.Tensor): the bias tensor to be added. If specified, needs to be on the device. Defaults to `None`.
            * :attr:`num_links` (int): Number of links to use for the all-gather operation.
            * :attr:`topology` (ttnn.Topology): Communication topology for the reduce-scatter phase. Defaults to `ttnn.Topology.Ring`.
            * :attr:`memory_config_rs` (Optional[ttnn.MemoryConfig]): Memory configuration for the Reduce Scatter operation.
            * :attr:`memory_config_mm` (Optional[ttnn.MemoryConfig]): Memory configuration for the Matmul operation.
            * :attr:`transpose_a` (bool)
            * :attr:`transpose_b` (bool)
            * :attr:`dtype` (Optional[DataType])
            * :attr:`program_config` (Optional[ttnn.MatmulProgramConfig])
            * :attr:`activation` (Optional[str])
            * :attr:`compute_kernel_config` (Optional[DeviceComputeKernelConfig])
            * :attr:`core_grid` (Optional[ttnn.CoreGrid])
        )doc");
}

}  // namespace ttnn::operations::experimental::ccl
