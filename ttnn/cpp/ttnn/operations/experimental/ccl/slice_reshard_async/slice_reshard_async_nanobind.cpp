// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "slice_reshard_async_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/experimental/ccl/slice_reshard_async/slice_reshard_async.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

namespace {

template <typename ccl_operation_t>
void bind_slice_reshard_async_op(nb::module_& mod, const ccl_operation_t& operation, const char* doc) {
    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"),
            nb::arg("dim"),
            nb::arg("output_dim_offset"),
            nb::arg("output_dim_shape"),
            nb::arg("cluster_axis"),
            nb::arg("final_semaphore"),
            nb::arg("barrier_semaphore"),
            nb::kw_only(),
            nb::arg("num_links") = 1,
            nb::arg("memory_config") = std::nullopt,
            nb::arg("topology") = ttnn::ccl::Topology::Linear});  // TODO_NANOBIND: cast?
}

}  // namespace

void bind_slice_reshard_async(nb::module_& mod) {
    bind_slice_reshard_async_op(
        mod,
        ttnn::experimental::slice_reshard_async,
        R"doc(

        Slice a multi-device tensor, and then reshard the tensor across devices.  The slice is computed in the space of the aggregate multi-device tensor, not in each device tensor individually.  The use case for this is when the input tensor is padded, and then after various operations changes in size in the dimension it is sharded on.  This could lead to unneccessary amounts of padding, so the padding is trimmed, and the resulting slice of the aggregate input tensor is resharded across devices.

        Args:
            input_tensor (ttnn.Tensor): multi-device tensor.
            dim (int): Dimension to shard on.
            output_dim_offset (int): Start of the output tensor in the shard dimension, in the context of the input tensor.
            output_dim_shape (int): Shape of the output tensor in the shard dimension, before sharding.
            cluster_axis (int): Provided a MeshTensor, the axis corresponding to MeshDevice to perform the slice_reshard operation on.

        Keyword Args:
            num_links (int, optional): Number of links to use for the slice_reshard operation. Defaults to `1`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `input tensor memory config`.
            topology (ttnn.Topology, optional): The topology configuration to run the operation in. Valid options are Ring and Linear. Defaults to `ttnn.Topology.Linear`.

        Returns:
            ttnn.Tensor: the slice_resharded output tensor.
        )doc");
}

}  // namespace ttnn::operations::experimental::ccl
