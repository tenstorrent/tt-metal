// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "neighbor_pad_async_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/ccl/neighbor_pad_async/neighbor_pad_async.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

void bind_neighbor_pad_async(nb::module_& mod) {
    ttnn::bind_function<"neighbor_pad_async", "ttnn.experimental.">(
        mod,
        R"doc(

        Performs a halo-padding operation on multi-device input tensor, where the padding values come from the neighbor device's tensor when available, or as specified by padding mode when no neighbor device is present.

        Args:
            input_tensor (ttnn.Tensor): multi-device tensor.
            dim (int): Dimension to pad on.
            padding_left (uint): How much to pad to the left (top).
            padding_right (uint): How much to pad to the right (bottom).
            padding_mode (string): replicate, constant, reflect.
            cluster_axis (int): Provided a MeshTensor, the axis corresponding to MeshDevice to perform the neighbor_pad operation on.

        Keyword Args:
            num_links (int, optional): Number of links to use for the neighbor_pad operation. Defaults to `1`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `input tensor memory config`.
            topology (ttnn.Topology, optional): The topology configuration to run the operation in. Valid options are Ring and Linear. Defaults to `ttnn.Topology.Linear`.

        Returns:
            ttnn.Tensor: the padded output tensor.
        )doc",
        &ttnn::experimental::neighbor_pad_async,
        nb::arg("input_tensor"),
        nb::arg("dim"),
        nb::arg("padding_left"),
        nb::arg("padding_right"),
        nb::arg("padding_mode"),
        nb::arg("cluster_axis"),
        nb::arg("final_semaphore"),
        nb::arg("barrier_semaphore"),
        nb::kw_only(),
        nb::arg("num_links") = 1,
        nb::arg("memory_config") = nb::none(),
        nb::arg("topology") = nb::cast(ttnn::ccl::Topology::Linear),
        nb::arg("secondary_cluster_axis") = nb::none(),
        nb::arg("secondary_mesh_shape") = nb::none());
}

}  // namespace ttnn::operations::experimental::ccl
