// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_async_generic_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/experimental/ccl/all_to_all_async_generic/all_to_all_async_generic.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

namespace {

template <typename ccl_operation_t>
void bind_all_to_all_async_generic_op(nb::module_& mod, const ccl_operation_t& operation, const char* doc) {
    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"),
            nb::arg("in_dim"),
            nb::arg("out_dim"),
            nb::kw_only(),
            nb::arg("persistent_output_buffer") = nb::none(),
            nb::arg("num_links") = 1,
            nb::arg("memory_config") = nb::none(),
            nb::arg("topology") = ttnn::ccl::Topology::Linear,  // TODO_NANOBIND: nb_cast?
            nb::arg("subdevice_id") = nb::none(),
            nb::arg("cluster_axis") = nb::none()});
}

}  // namespace

void bind_all_to_all_async_generic(nb::module_& mod) {
    bind_all_to_all_async_generic_op(
        mod,
        ttnn::experimental::all_to_all_async_generic,
        R"doc(
        Performs an asynchronous all-to-all collective communication operation across multiple devices.

        This operation redistributes data between devices by having each device split its input tensor
        into N chunks (where N is the number of devices) and sending the i-th chunk to device i.
        The operation uses asynchronous kernels for improved performance.

        Args:
            input_tensor (ttnn.Tensor): Input tensor to redistribute.
            in_dim (int): The dimension number to split.
            out_dim (int): The dimension number to concatenate.

        Keyword Args:
            persistent_output_buffer (ttnn.Tensor, optional): Buffer where final output will be written.
            num_links (int, optional): Number of fabric links to use for communication.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for buffers.
            topology (ttnn.Topology, optional): Network topology to use. Defaults to ttnn.Topology.Ring.
            subdevice_id (SubDeviceId, optional): Target specific subdevice for the operation.
            cluster_axis (int, optional): The axis along which to cluster the communication.

        Returns:
            ttnn.Tensor: Output tensor containing redistributed data.

        Example:
            >>> # Redistribute data from dim 2 to dim 3 across 4 devices
            >>> output_buf = ttnn.zeros_like(input_tensor)       # Output buffer
            >>> result = ttnn.experimental.all_to_all_async_generic(
            ...     input_tensor,
            ...     persistent_output_buffer=output_buf,  # Optional buffer
            ...     in_dim=2,
            ...     out_dim=3,
            ...     cluster_axis=1
            ... )
        )doc");
}

}  // namespace ttnn::operations::experimental::ccl
